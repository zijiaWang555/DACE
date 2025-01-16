import numpy as np
import tensorflow as tf
import os
from ot_util import OT
from ALECE import ALECE
import math
from arg_parser import arg_parser

from ot_util import OT
from utils import FileViewer, file_utils, eval_utils, arg_parser_utils
from data_process import feature
global_seed = 42
np.random.seed(global_seed) 
def process_error_val(val):
    if val < 100:
        s = str(round(val, 2))
        terms = s.split('.')
        if len(terms) == 1:
            s += '.00'
        elif len(terms[1]) == 1:
            s += '0'

        return f'${s}$'

    if val >= 1e10:
        return '>$10^{10}$'
    if val < 1e5:
        int_val = int(val)
        s = format(int_val, ',d')
        return f'${s}$'
    exponent = int(math.log10(val))
    x = math.pow(10, exponent)
    a = val / x

    a_str = str(round(a,1))
    terms = a_str.split('.')
    if len(terms) == 1:
        a_str += '.0'
    return f'${a_str}$$\\cdot$$10^{exponent}$'

@tf.function  
def set_e_to_a(e, a):  
    e = tf.Variable(a)  
    return e

def train_with_batch(model, args, train_data, validation_data, curr_ckpt_step, test_data=None, q_error_dir=None,data_re=False):
    _train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
    _validation_dataset = tf.data.Dataset.from_tensor_slices(validation_data)

    train_labels = train_data[-1]
    validation_labels = validation_data[-1]
    n_batches = math.ceil(train_labels.shape[0] / args.batch_size)
    validation_n_batches = math.ceil(validation_labels.shape[0] / args.batch_size)
    ot_list = np.load('/data1/liuhaoran/ALECE/src/pred_teacher_outputs_job.npy')
    if model.require_init_train_step == True:
        for epoch in range(args.n_epochs):
            train_dataset = _train_dataset.shuffle(args.shuffle_buffer_size ,seed=global_seed).batch(args.batch_size)
            for batch in train_dataset.take(n_batches):
                model.init_train_step(batch)

    best_loss = 1e100
    loss = 0
    if curr_ckpt_step >= 0:
        validation_dataset = _validation_dataset.batch(args.batch_size)
        for batch in validation_dataset.take(validation_n_batches):
            batch_loss = model.eval_validation(batch)
            loss += batch_loss.numpy()
        best_loss = loss

    if best_loss > 1e50:
        print('best_loss = inf')
    else:
        print(f'best_loss = {best_loss}')
    
    import datetime
    import random
    batch_all = []
    batch_use = []
    lossa = []
    losst = []
    idd = 0
    pred_teacher_outputs = []

    for epoch in range(1, args.n_epochs + 1):
        model.e.assign_add(1)

        train_dataset = _train_dataset.shuffle(args.shuffle_buffer_size,seed=global_seed).batch(args.batch_size)
        batch_no = 0
        batch_temp = []
        wl = 0.02 + (epoch - 1) * (0.08 / 1499)

        T = 3.0
        alpha = 0.09
        for batch in train_dataset.take(n_batches):
            if random.random() < wl and len(batch_use) != 0 and data_re:
                batch = random.choice(batch_use)
            batch_temp.append(batch)
            (train_X, train_Q, train_weights, train_labels) = batch
            with tf.GradientTape() as tape:
                loss = model.build_loss(train_X, train_Q, train_weights, train_labels)
                pred_stu, _ = model.forward(train_X, train_Q, training=True)

                ot_random_index = np.random.choice(len(ot_list),len(pred_stu))
                ot_random_list = ot_list[ot_random_index]
                pred_stu_softmax = tf.nn.softmax(pred_stu / T)
                ot_softmax = tf.nn.softmax(ot_random_list / T)

                loss_ot = tf.keras.losses.KLDivergence()(ot_softmax, pred_stu_softmax) * (T * T)


                loss = loss + loss_ot * alpha

            gradients = tape.gradient(loss, model.attn_model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.attn_model.trainable_variables))
            batch_no += 1



        ####################################

        if model.e <900 and model.e >150 and data_re:
            for batch in batch_temp:
                batch_loss = model.eval_validation2(batch)
                batch_loss_T = model.eval_validation2_T(batch)
                loss  = batch_loss.numpy()
                lossT = batch_loss_T.numpy()
                lossa.append(loss)
                losst.append(lossT)
                batch_all.append(batch)
            
        if len(batch_all)>500 and data_re:
            length = len(lossa)  
            top_20_percent = length // 5  # 计算20%的长度  
            
            sorted_list = sorted(lossa, reverse=True)
            

            bsize = 2500
            top_20_percent_value = sorted_list[top_20_percent]  
            
            for ik,loss in enumerate(lossa):
                if loss > top_20_percent_value and losst[ik]<lossa[ik]:
                    if len(batch_use)<bsize:
                        batch_use.append(batch_all[ik])
                        idd = idd+1
                    else:
                        batch_use[idd%bsize]=batch_all[ik]
                        idd = idd+1
            lossa=[]
            losst=[]
            batch_all=[]



        ##########################################
        loss = 0
        if curr_ckpt_step >= 0 or epoch >= args.min_n_epochs:
            validation_dataset = _validation_dataset.batch(args.batch_size)
            for batch in validation_dataset.take(validation_n_batches):
                batch_loss = model.eval_validation(batch)
                loss += batch_loss.numpy()

            if loss < best_loss:
                ckpt_step = curr_ckpt_step + epoch
                model.save(ckpt_step)
                best_loss = loss
            print(f'{datetime.datetime.now()}Epoch-{epoch}, loss = {loss}, best_loss = {best_loss},e={model.e.numpy()},lenalldata={len(batch_all)}, lendata = {len(batch_use)}')
        else:
            print(f'Epoch-{epoch}')
        if epoch%1==0:
            eval_new(model, args, test_data)
def eval_new(model, args, test_data):
    batch_preds_list = []

    test_labels = test_data[-1]
    test_dataset = tf.data.Dataset.from_tensor_slices(test_data).batch(args.batch_size)
    n_batches = math.ceil(test_labels.shape[0] / args.batch_size)
    test_length = test_labels.shape[0]
    from datetime import datetime 
    a=datetime.now()  
    cu = 0
    for batch in test_dataset.take(n_batches):
        batch_preds = model.eval_test(batch)
        batch_preds_list.append(batch_preds.numpy())
        cu = cu+1
    b=datetime.now()
    print("用时：",(b-a).microseconds/test_length /1000)

    preds = np.concatenate(batch_preds_list, axis=0)
    preds = label_preds_to_card_preds(preds, args)

    labels = test_data[-1].numpy()
    labels = label_preds_to_card_preds(np.reshape(labels, [-1]), args)
    q_error = eval_utils.generic_calc_q_error(preds, labels)
    ########################
    p_error_test = np.sort(q_error)
    n = p_error_test.shape[0]')
    ratios = [0.5, 0.9, 0.95, 0.99]

    error_vals = []
    for ratio in ratios:
        idx = int(n * ratio)
        error_vals.append(p_error_test[idx])

    # print(error_vals)
    results = []
    for val in error_vals:
        results.append(process_error_val(val))
    result_str = ' & '.join(results)
    print(f'{args.data}-{args.wl_type}-{args.model}: {result_str}')
    ########################
    idexes = np.where(q_error < 10)[0]
    n = idexes.shape[0]
    print('ratio =', n / q_error.shape[0])


    return preds

def eval(model, args, test_data, q_error_dir):
    batch_preds_list = []

    test_labels = test_data[-1]
    test_dataset = tf.data.Dataset.from_tensor_slices(test_data).batch(args.batch_size)
    n_batches = math.ceil(test_labels.shape[0] / args.batch_size)
    test_length = test_labels.shape[0]
    from datetime import datetime
    a=datetime.now()
    cu = 0
    for batch in test_dataset.take(n_batches):
        batch_preds = model.eval_test(batch)
        batch_preds_list.append(batch_preds.numpy())
        cu = cu+1
    b=datetime.now()

    preds = np.concatenate(batch_preds_list, axis=0)
    preds = label_preds_to_card_preds(preds, args)

    labels = test_data[-1].numpy()
    labels = label_preds_to_card_preds(np.reshape(labels, [-1]), args)
    q_error = eval_utils.generic_calc_q_error(preds, labels)
    ########################
    p_error_test = np.sort(q_error)
    n = p_error_test.shape[0]
    ratios = [0.5, 0.9, 0.95, 0.99]

    error_vals = []
    for ratio in ratios:
        idx = int(n * ratio)
        # print(f'idx = {idx}')
        error_vals.append(p_error_test[idx])

    # print(error_vals)
    results = []
    for val in error_vals:
        results.append(process_error_val(val))
    result_str = ' & '.join(results)
    print(f'{args.data}-{args.wl_type}-{args.model}: {result_str}')
    ########################
    idexes = np.where(q_error < 10)[0]
    n = idexes.shape[0]
    print('ratio =', n / q_error.shape[0])

    if q_error_dir is not None:
        fname = model.model_name
        result_path = os.path.join(q_error_dir, fname + ".npy")
        np.save(result_path, preds)

    return preds


def _normalize(data, X_mean, X_std, nonzero_idxes):
    norm_data = (data - X_mean)
    norm_data[:, nonzero_idxes] /= X_std[nonzero_idxes]
    return norm_data


def normalizations(datas):
    X = datas[0]
    X_std = X.std(axis=0)
    nonzero_idxes = np.where(X_std > 0)[0]
    X_mean = X.mean(axis=0)
    norm_data = tuple(_normalize(data, X_mean, X_std, nonzero_idxes) for data in datas)
    return norm_data


def valid_datasets(datas):
    cards = datas[-1]
    valid_idxes = np.where(cards >= 0)[0]
    valid_datas = tuple(data[valid_idxes] for data in datas)
    return valid_datas


def organize_data(raw_data_i, args):
    features = raw_data_i[0]
    db_states = features[:, 0:args.histogram_feature_dim]
    query_part_features = features[:, args.histogram_feature_dim:]

    data = [db_states, query_part_features]
    data.extend(raw_data_i[1:])
    data = tuple(data)
    return data


def cards_to_labels(cards, args):
    card_min = np.min(cards)
    assert card_min >= 0
    dtype = np.float32
    if args.use_float64 == 1:
        dtype = np.float64

    cards += 1
    cards = cards.astype(dtype)
    if args.card_log_scale == 1:
        labels = np.log(cards) / args.scaling_ratio
    else:
        labels = cards
    labels = np.reshape(labels, [-1, 1])
    return labels.astype(dtype)


def label_preds_to_card_preds(preds, args):
    preds = np.reshape(preds, [-1])
    if args.card_log_scale:
        preds *= args.scaling_ratio
        preds = np.clip(preds, a_max=25, a_min=0)
        preds = np.exp(preds) - 1
    return preds


def data_compile(data, args):
    dtype = tf.float32
    if args.use_float64 == 1:
        dtype = tf.float64

    tf_data = tuple(tf.convert_to_tensor(x, dtype=dtype) for x in data)
    return tf_data


def load_data(args):
    print('Loading data...')

    workload_data = feature.load_workload_data_(args)
    ckpt_dir = arg_parser_utils.get_ckpt_dir(args)
    _, q_error_dir = arg_parser_utils.get_p_q_error_dir(args)

    FileViewer.detect_and_create_dir(ckpt_dir)
    FileViewer.detect_and_create_dir(q_error_dir)

    (train_features, train_cards, test_features, test_cards, meta_infos) = workload_data
    (histogram_feature_dim, query_part_feature_dim, join_pattern_dim, num_attrs) = meta_infos

    args.histogram_feature_dim = histogram_feature_dim
    args.query_part_feature_dim = query_part_feature_dim
    args.join_pattern_dim = join_pattern_dim
    args.num_attrs = num_attrs

    print('Processing data...')

    (train_features, train_cards) = valid_datasets((train_features, train_cards))

    train_labels = cards_to_labels(train_cards, args)

    _n_test = test_features.shape[0]
    (test_features, test_cards) = valid_datasets((test_features, test_cards))

    test_labels = cards_to_labels(test_cards, args)

    (train_features, test_features) = normalizations(
        (train_features, test_features)
    )

    #randomly select 10% of train data as validation data
    N_train = train_features.shape[0]
    shuffle_idxes = np.arange(0, N_train, dtype=np.int64)
    
    np.random.shuffle(shuffle_idxes)

    train_features = train_features[shuffle_idxes]
    train_labels = train_labels[shuffle_idxes]

    # split data into training and validation parts
    N_train = int(N_train * 0.9)

    validation_features = train_features[N_train:]
    validation_labels = train_labels[N_train:]

    train_features = train_features[0: N_train]
    train_labels = train_labels[0: N_train]

    label_mean = np.mean(train_labels)
    if args.use_loss_weights == 1:
        weights = train_labels / label_mean
        weights = np.reshape(weights, [-1])
        weights = np.clip(weights, a_min=1e-3, a_max=np.max(weights))
    else:
        weights = np.ones(shape=[train_labels.shape[0]], dtype=train_labels.dtype)

    train_data = (train_features, weights, train_labels)
    validation_data = (validation_features, validation_labels)
    # validation_data = (test_features, test_labels)
    test_data = (test_features, test_labels)

    train_data = organize_data(train_data, args)
    validation_data = organize_data(validation_data, args)
    test_data = organize_data(test_data, args)
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hrliu/.conda/envs/alece_env/lib/python3.8/site-packages/tensorrt_libs
    train_data = data_compile(train_data, args)
    validation_data = data_compile(validation_data, args)
    test_data = data_compile(test_data, args)

    return (train_data, validation_data, test_data), q_error_dir, ckpt_dir

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
def run():
    args = arg_parser.get_arg_parser()
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[args.gpu:args.gpu + 1], 'GPU')
    #动态分配内存
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    test_wl_type = args.test_wl_type
    FileViewer.detect_and_create_dir(args.experiments_dir)

    datasets, q_error_dir, ckpt_dir = load_data(args)
    (train_data, validation_data, test_data) = datasets

    model_name = f'{args.model}_{args.wl_type}'
    # ====================================================
    n = args.mlp_hidden_dim
    model = ALECE.ALECE(args)
    model.set_model_name(model_name)
    model.ckpt_init(ckpt_dir)
    ckpt_files = FileViewer.list_files(ckpt_dir)
    if len(ckpt_files) > 0:
        ckpt_step = model.restore().numpy()
    else:
        ckpt_step = -1
    print('ckpt_step =', ckpt_step)

    kd = False
    data_re = False
    if kd==True :

       model2 = ALECE.ALECE(args, trainable=False)
       model2.set_model_name("T1")
       model2.ckpt_init(path_model2)
       ckpt_files2 = FileViewer.list_files(path_model2)
       if len(ckpt_files2) > 0:
           ckpt_step2 = model2.restore().numpy()
       else:
           ckpt_step2 = -1
       print('ckpt_step =', ckpt_step2)

       #########################################################

       model.model2=model2



    training = ckpt_step < 0 or args.keep_train == 1
    if training:
        train_with_batch(model, args, train_data, validation_data, ckpt_step,test_data, q_error_dir,data_re)
        model.compile(train_data)

    ckpt_step = model.restore().numpy()
    assert ckpt_step >= 0
    preds = eval(model, args, test_data, q_error_dir)

    # ====================================================
    preds = preds.tolist()
    workload_dir = arg_parser_utils.get_workload_dir(args, test_wl_type)
    e2e_dir = os.path.join(args.experiments_dir, args.e2e_dirname)
    FileViewer.detect_and_create_dir(e2e_dir)
    train_wl_type_pre, test_wl_type_pre, pg_cards_path = arg_parser_utils.get_wl_type_pre_and_pg_cards_paths(args)

    if test_wl_type == 'static':
        path = os.path.join(e2e_dir, f'{args.model}_{args.data}_static.txt')
    else:
        path = os.path.join(e2e_dir, f'{args.model}_{args.data}_{train_wl_type_pre}_{test_wl_type_pre}.txt')

    lines = [(str(x) + '\n') for x in preds]
    file_utils.write_all_lines(path, lines)


