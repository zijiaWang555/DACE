import prefetcher as pf
import dataUtils as ut
import numpy as np
import torch
import os
import sys
import time
import datetime
import math
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torch.utils.data import Dataset
from nflows import transforms
from nflows import distributions
from nflows import utils
from nflows import flows
import nflows.nn as nn_
from torchquadMy.torchquad import set_up_backend
from torchquadMy.torchquad import VEGAS
from torchquadMy.torchquad import BatchMulVEGAS
from ot_util import OT



class Config:
    gpu_id = 0
    seed = 42
    flow_id = 0
    flow_id_source = 0

    use_data_buffer = True
    data_buffer_size = 512
    data_buffer_sample_size = 100
    train_batch_size = 512
    val_batch_size = 262144
    learning_rate = 1e-5
    num_training_steps = 400000
    grad_norm_clip_value = 5.

    num_flow_steps = 1
    hidden_features = 32

    data_id_target = 2
    data_id_source = 1

    num_transform_blocks = 2
    dropout_probability = 0
    tail_bound = 3
    use_batch_norm = False
    num_bins = 8
    linear_transform_type = 'lu'

    anneal_learning_rate = True
    REUSE_FROM_FILE = True

    transfer = True
    pretrain_step = 50000

os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(Config.gpu_id)
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.manual_seed(Config.seed)
np.random.seed(Config.seed)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

PROJECT_PATH = 'xxx/'
data_PATH = PROJECT_PATH + 'data/'
ckpts_PATH = PROJECT_PATH + 'models/'

class HiddenPrints:
    def __init__(self, activated=True):
        self.activated = activated
        self.original_stdout = None

    def open(self):
        """ no output """
        sys.stdout.close()
        sys.stdout = self.original_stdout

    def close(self):
        """ output """
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __enter__(self):
        if self.activated:
            self.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.activated:
            self.open()



class myDataset(Dataset):
    def __init__(self, dataset_name, data_id, split='train', frac=None):
        path = os.path.join(data_PATH, '{}-{}.npy'.format(dataset_name, data_id))
        self.data = np.load(path).astype(np.float32)
        self.n, self.dim = self.data.shape

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.n

def create_linear_transform(features):
    if Config.linear_transform_type == 'permutation':
        return transforms.RandomPermutation(features=features)
    elif Config.linear_transform_type == 'lu':
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=features),
            transforms.LULinear(features, identity_init=True)
        ])
    elif Config.linear_transform_type == 'svd':
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=features),
            transforms.SVDLinear(features, num_householder=10, identity_init=True)
        ])
    else:
        raise ValueError

def create_base_transform(i, hf, features):
    # tmp_mask = utils.create_alternating_binary_mask(features, even=(i % 2 == 0))
    return transforms.coupling.PiecewiseRationalQuadraticCouplingTransform(
        mask=utils.create_alternating_binary_mask(features, even=(i % 2 == 0)),
        transform_net_create_fn=lambda in_features, out_features: nn_.nets.ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hf,
            context_features=None,
            num_blocks=Config.num_transform_blocks,
            activation=F.relu,
            dropout_probability=Config.dropout_probability,
            use_batch_norm=Config.use_batch_norm
        ),
        num_bins=Config.num_bins,
        tails='linear',
        tail_bound=Config.tail_bound,
        apply_unconditional_transform=True
    )

# torch.masked_select()
def create_transform(num, hf, features):
    transform = transforms.CompositeTransform([
        transforms.CompositeTransform([
            create_linear_transform(features),
            create_base_transform(i, hf, features)
        ]) for i in range(num)
    ] + [
        create_linear_transform(features)
    ])
    return transform

def ErrorMetric(est_card, card):
    if isinstance(est_card, torch.FloatTensor) or isinstance(est_card, torch.IntTensor):
        est_card = est_card.cpu().detach().numpy()
    if isinstance(est_card, torch.Tensor):
        est_card = est_card.cpu().detach().numpy()
    est_card = np.float64(est_card)
    card = np.float64(card)
    if card == 0 and est_card != 0:
        return est_card
    if card != 0 and est_card == 0:
        return card
    if card == 0 and est_card == 0:
        return 1.0
    return max(est_card / card, card / est_card)

def BatchErrorMetrix(est_list, oracle_list):
    ret = np.zeros(len(est_list))
    ID = 0
    for est, real in zip(est_list, oracle_list):
        ret[ID] = ErrorMetric(est, real)
        ID = ID + 1
    return ret



def train(args):
    # Load Data
    train_dataset = myDataset(args.dataset, Config.data_id_target)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=False,
        drop_last=False
    )
    val_dataset = myDataset(args.dataset, Config.data_id_target)
    val_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=Config.val_batch_size,
        shuffle=False,
        drop_last=False
    )
    train_loader = list(train_loader)
    val_loader = list(val_loader)
    TRAIN_LOADER_LEN = len(train_loader)
    features = train_dataset.dim
    print('train loader length is [{}]'.format(TRAIN_LOADER_LEN))
    print('val loader length is [{}]'.format(len(val_loader)))

    # Load source model
    source_transform = create_transform(Config.num_flow_steps, Config.hidden_features, features)
    source_distribution = distributions.StandardNormal((features,))
    source_flow = flows.Flow(source_transform, source_distribution).to(device)
    source_path = os.path.join(ckpts_PATH, '{}-id{}-stu{}-best-val.t'.format(args.dataset, Config.data_id_source, Config.flow_id_source))
    source_flow.load_state_dict(torch.load(source_path))
    source_flow.cuda()
    source_flow.eval()
    source_dataset = myDataset(args.dataset, Config.data_id_source)
    source_loader = data.DataLoader(
        source_dataset,
        batch_size=Config.train_batch_size,
        shuffle=False,
        drop_last=False
    )
    source_prefetcher = pf.data_prefetcher(source_loader)

    # Create target stu model
    transform = create_transform(Config.num_flow_steps, Config.num_flow_steps, features)
    distribution = distributions.StandardNormal((features,))
    flow = flows.Flow(transform, distribution).to(device)
    n_params = utils.get_num_parameters(flow)
    print('There are {} trainable parameters in student model.'.format(n_params))
    print('Parameters total size is {} MB\n'.format(n_params * 4 / 1024 / 1024))

    ot = OT(device)
    optimizer = optim.Adam(flow.parameters(), lr=Config.learning_rate)
    if Config.anneal_learning_rate:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, np.ceil(Config.num_training_steps / TRAIN_LOADER_LEN),0)
    kd_crit = nn.MSELoss()
    best_val_score = -1e10
    prefetcher = pf.data_prefetcher(train_loader)

    num_training_steps = int(np.ceil(Config.num_training_steps / TRAIN_LOADER_LEN) * TRAIN_LOADER_LEN)
    print('num training steps is ', num_training_steps)

    for step in range(num_training_steps):
        if step % 10000 == 0:
            print('[{}] {}/400000  {}% has finished!'.format(datetime.datetime.now(), step, 100. * step / 400000))
        flow.train()
        batch = prefetcher.next()
        if batch.shape[0] != Config.train_batch_size:
            prefetcher = pf.data_prefetcher(train_loader)
            batch = prefetcher.next()

        log_density = flow.log_prob(batch)
        noise = flow.transform_to_noise(batch)
        loss = - torch.mean(log_density)

        if Config.transfer and step > Config.pretrain_step:
            source_batch = source_prefetcher.next()
            if source_batch.shape[0] != Config.train_batch_size:
                source_prefetcher = pf.data_prefetcher(source_loader)
                source_batch = source_prefetcher.next()
            source_noise = source_flow.transform_to_noise(source_batch)
            ot_loss = ot(noise, source_noise)
            loss += ot_loss
            if step % 10000 == 0:
                print('[{}] {}/400000  {}% has finished!'.format(datetime.datetime.now(), step, 100. * step / 400000))

        optimizer.zero_grad()
        loss.backward()
        if Config.grad_norm_clip_value is not None:
            clip_grad_norm_(flow.parameters(), Config.grad_norm_clip_value)
        optimizer.step()

        if (step + 1) % 5000 == 0:
            flow.eval()
            val_prefetcher = pf.data_prefetcher(val_loader)
            with torch.no_grad():
                running_val_log_density = 0
                while True:
                    val_batch = val_prefetcher.next()
                    if val_batch is None:
                        break

                    log_density_val = flow.log_prob(val_batch.to(device).detach())
                    mean_log_density_val = torch.mean(log_density_val).detach()
                    running_val_log_density += mean_log_density_val
                running_val_log_density /= len(val_loader)
                print('[{}] step now is [{:6d}] running_val_log_density is {:.4f}'.format(datetime.datetime.now(), step,
                                                                                          running_val_log_density),
                      end='')

            if running_val_log_density > best_val_score:
                best_val_score = running_val_log_density
                print('  ## New best! ##  ')
                path = os.path.join(ckpts_PATH,
                                    '{}-id{}-stu{}-best-val.t'.format(args.dataset, Config.data_id_target, Config.flow_id))
                torch.save(flow.state_dict(), path)
            else:
                print('')

        if (step + 1) % 20000 == 0:
            flow.eval()
            print(
                '[{}] save once. Step is {} best val score is {}'.format(datetime.datetime.now(), step, best_val_score))

        if Config.anneal_learning_rate and (step + 1) % TRAIN_LOADER_LEN == 0:
            scheduler.step()

def evaluate(args):
    import pickle
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    data, n, dim = ut.LoadTable(args.dataset, Config.data_id_target)
    DW = ut.DataWrapper(data, args.dataset)
    legal_lists = pickle.load(open(PROJECT_PATH + 'test_queries/queryvals_{}-{}_rng.pkl'.format(args.dataset, Config.data_id), 'rb'))
    oracle_cards = pickle.load(open(PROJECT_PATH + 'test_queries/cards_{}-{}_rng.pkl'.format(args.dataset, Config.data_id), 'rb'))
    legal_tensors = torch.Tensor(legal_lists).to('cuda')
    set_up_backend("torch", data_type="float32")
    distribution = distributions.StandardNormal((dim,))
    transform = create_transform()
    flow = flows.Flow(transform, distribution).to(device)
    path = os.path.join(ckpts_PATH, '{}-id{}-stu{}-best-val.t'.format(args.dataset, Config.data_id, Config.flow_id))
    flow.load_state_dict(torch.load(path))
    flow.cuda()
    flow.eval()
    n_params = utils.get_num_parameters(flow)
    print('There are {} trainable parameters in this model.'.format(n_params))
    print('Parameters total size is {} MB'.format(n_params * 4 / 1024 / 1024))

    f_batch_time = 0
    def f_batch(inp):
        global f_batch_time
        with torch.no_grad():
            inp = inp.cuda()
            print("【Example input】", inp[0, :])
            print("inp shape ", inp.shape)
            st = time.time()
            prob_list = flow.log_prob(inp)
            prob_list = torch.exp(prob_list)
            print("【max_prob】 ", prob_list.max())
            print("【median_prob】 ", prob_list.median())
            en = time.time()
            f_batch_time += en - st
            return prob_list

    if Config.REUSE_FROM_FILE == True:
        f = open(PROJECT_PATH + 'test_queries/' + '{}-{}.pickle'.format(args.dataset, Config.data_id), 'rb')
        target_map = pickle.load(f)
    z = DW.getLegalRangeQuery_JOB_light([[], [], []])
    z = torch.Tensor(z)
    print(z.shape)
    full_integration_domain = torch.Tensor(z)
    domain_starts = full_integration_domain[:, 0]
    domain_sizes = full_integration_domain[:, 1] - domain_starts

    if Config.REUSE_FROM_FILE == False:
        vegas = VEGAS()
        bigN = 1000000 * 40
        result = vegas.integrate(f_batch, dim=dim,
                                 N=bigN,
                                 integration_domain=full_integration_domain,
                                 use_warmup=True,
                                 use_grid_improve=True,
                                 max_iterations=40
                                 )
        print(result)
        result = result * DW.n
        print('result is ', result)
        target_map = vegas.map
        import pickle
        f = open(PROJECT_PATH + 'test_queries/' + '{}-{}.pickle'.format(args.dataset, Config.data_id), 'wb')
        pickle.dump(target_map, f)
        f.close()

    def getResult(n, N, num_iterations=3, alpha=0.5, beta=0.5):
        global f_batch_time
        """ n: batch size """
        z = BatchMulVEGAS()
        DIM = dim
        full_integration_domain = torch.Tensor(DIM * [[0, 1]])

        start_id = 0
        end_id = 0

        f_batch_time = 0
        st = time.time()
        results = []
        with torch.no_grad():
            while start_id < 2000:
                end_id = end_id + n
                if end_id > 2000:
                    end_id = 2000
                z.setValues(f_batch,
                            dim=DIM,
                            alpha=alpha,
                            beta=beta,
                            N=N,
                            n=end_id - start_id,
                            iterations=num_iterations,
                            integration_domains=legal_tensors[start_id:end_id],
                            rng=None,
                            seed=1234,
                            reuse_sample_points=True,
                            target_map=target_map,
                            target_domain_starts=domain_starts,
                            target_domain_sizes=domain_sizes,
                            )
                start_id = start_id + n
                results.append(z.integrate())

        en = time.time()
        total_time = en - st
        return total_time, results

    def testHyper(n, N, num_iterations, alpha, beta):
        with HiddenPrints():
            total_time, result = getResult(n=n,
                                           N=N,
                                           num_iterations=num_iterations,
                                           alpha=alpha,
                                           beta=beta)

            result = torch.cat(tuple(result))
            FULL_SIZE = torch.Tensor([DW.n])
            result = result * FULL_SIZE
            result = result.to('cpu')

            n_ = 2000
            oracle_list = oracle_cards.copy()

            err_list = BatchErrorMetrix(result.int(), oracle_list)

            total_query_time = total_time
            avg_per_query_time = 1000. * (total_query_time / n_)
            avg_f_batch_time = 1000. * f_batch_time / n_
            avg_vegas_time = avg_per_query_time - avg_f_batch_time

        print("********** total_n=[{}] batchn=[{}]  N=[{}]  nitr=[{}]  alpha=[{}]  beta=[{}] ******".format(n_, n, N,
                                                                                                            num_iterations,
                                                                                                            alpha,
                                                                                                            beta))
        print('@ Average per query          [{}] ms'.format(avg_per_query_time))
        print(' --  Average per query NF    [{}] ms'.format(avg_f_batch_time))
        print(' --  Average per query vegas [{}] ms'.format(avg_vegas_time))
        p50 = np.percentile(err_list, 50)
        p95 = np.percentile(err_list, 95)
        p99 = np.percentile(err_list, 99)
        pmax = np.max(err_list)
        print('Median [{:.3f}]  95th [{:.3f}]  99th [{:.3f}]  max [{:.3f}]'.format(np.percentile(err_list, 50),
                                                                                   np.percentile(err_list, 95),
                                                                                   np.percentile(err_list, 99),
                                                                                   np.max(err_list)))
        return p50, p95, p99, pmax

    alpha_list = [0.4]
    beta_list = [0.2]
    p50s = []
    p95s = []
    p99s = []
    pmaxs = []
    for alpha in alpha_list:
        for beta in beta_list:
            p50, p95, p99, pmax = testHyper(100, 16000, 4, alpha, beta)
            p50s.append(p50)
            p95s.append(p95)
            p99s.append(p99s)
            pmaxs.append(pmax)
            print(p50, p95, p99s, pmax)


def run(args):
    train(args)
    evaluate(args)