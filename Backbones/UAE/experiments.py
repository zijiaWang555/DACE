"""Experiment configurations.

EXPERIMENT_CONFIGS holds all registered experiments.

TEST_CONFIGS (defined at end of file) stores "unit tests": these are meant to
run for a short amount of time and to assert metrics are reasonable.

Experiments registered here can be launched using:

  >> python kd.py --run <config> [ <more configs> ]
  >> python kd.py  # Runs all tests in TEST_CONFIGS.
"""
import os

from ray import tune

EXPERIMENT_CONFIGS = {}
TEST_CONFIGS = {}

# Common config. Each key is auto set as an attribute (i.e. NeuroCard.<attr>)
# so try to avoid any name conflicts with members of that class.
BASE_CONFIG = {
    'cwd': os.getcwd(),
    'epochs_per_iteration': 1,
    'num_eval_queries_per_iteration': 100,
    'num_eval_queries_at_end': 2000,  # End of training.
    'num_eval_queries_at_checkpoint_load': 2000,  # Evaluate a loaded ckpt.
    'epochs': 10,
    'seed': None,
    'order_seed': None,
    'bs': 128,
    'order': None,
    'layers': 2,
    'fc_hiddens': 128,
    'warmups': 1000,
    'constant_lr': None,
    'lr_scheduler': None,
    'custom_lr_lambda': None,
    'optimizer': 'adam',
    'residual': True,
    'direct_io': True,
    'input_encoding': 'embed',
    'output_encoding': 'embed',
    'query_filters': [5, 12],
    'force_query_cols': None,
    'embs_tied': True,
    'embed_size': 32,
    'input_no_emb_if_leq': True,
    'resmade_drop_prob': 0.,

    # Multi-gpu data parallel training.
    'use_data_parallel': False,

    # If set, load this checkpoint and run eval immediately. No training. Can
    # be glob patterns.
    # Example:
    # 'checkpoint_to_load': tune.grid_search([
    #     'models/*52.006*',
    #     'models/*43.590*',
    #     'models/*42.251*',
    #     'models/*41.049*',
    # ]),
    'checkpoint_to_load': None,
    # Dropout for wildcard skipping.
    'disable_learnable_unk': False,
    'per_row_dropout': True,
    'dropout': 1,
    'table_dropout': False,
    'fixed_dropout_ratio': False,
    'asserts': None,
    'special_orders': 0,
    'special_order_seed': 0,
    'join_tables': [],
    'label_smoothing': 0.0,
    'compute_test_loss': False,

    # Column factorization.
    'factorize': False,
    'factorize_blacklist': None,
    'grouped_dropout': True,
    'factorize_fanouts': False,

    # Eval.
    'eval_psamples': [100, 1000, 10000],
    'eval_join_sampling': None,  # None, or #samples/query.

    # Transformer.
    'use_transformer': False,
    'transformer_args': {},

    # Checkpoint.
    'save_checkpoint_at_end': True,
    'checkpoint_every_epoch': False,

    # Experimental.
    '_save_samples': None,
    '_load_samples': None,
    'num_orderings': 1,
    'num_dmol': 0,

    # For dps
    'semi_train': False,
    'train_sample_num': 50,
}

JOB_LIGHT_BASE = {
    'dataset': 'imdb',
    'join_tables': [
        'cast_info', 'movie_companies', 'movie_info', 'movie_keyword', 'title',
        'movie_info_idx'
    ],
    'join_keys': {
        'cast_info': ['movie_id'],
        'movie_companies': ['movie_id'],
        'movie_info': ['movie_id'],
        'movie_keyword': ['movie_id'],
        'title': ['id'],
        'movie_info_idx': ['movie_id']
    },
    # Sampling starts at this table and traverses downwards in the join tree.
    'join_root': 'title',
    # Inferred.
    'join_clauses': None,
    'join_how': 'outer',
    # Used for caching metadata.  Each join graph should have a unique name.
    'join_name': 'job-light',
    # See datasets.py.
    'use_cols': 'simple',
    'seed': 0,
    'per_row_dropout': False,
    'table_dropout': True,
    'embs_tied': True,
    # Num tuples trained =
    #   bs (batch size) * max_steps (# batches per "epoch") * epochs.
    'epochs': 1,
    'bs': 128,
    'max_steps': 100,
    # Use this fraction of total steps as warmups.
    'warmups': 0.05,
    # Number of DataLoader workers that perform join sampling.
    'loader_workers': 8,
    # Options: factorized_sampler, fair_sampler (deprecated).
    'sampler': 'factorized_sampler',
    'sampler_batch_size': 256,
    'layers': 4,
    # Eval:
    'compute_test_loss': True,
    'queries_csv': './queries/job-light.csv',
    'num_eval_queries_per_iteration': 0,
    'num_eval_queries_at_end': 70,
    'eval_psamples': [4000],

    # Multi-order.
    'special_orders': 0,
    'order_content_only': True,
    'order_indicators_at_front': False,
}

FACTORIZE = {
    'factorize': True,
    'word_size_bits': 1,
    'grouped_dropout': True,
}

JOB_M = {
    'join_tables': [
        'title', 'aka_title', 'cast_info', 'complete_cast', 'movie_companies',
        'movie_info', 'movie_info_idx', 'movie_keyword', 'movie_link',
        'kind_type', 'comp_cast_type', 'company_name', 'company_type',
        'info_type', 'keyword', 'link_type'
    ],
    'join_keys': {
        'title': ['id', 'kind_id'],
        'aka_title': ['movie_id'],
        'cast_info': ['movie_id'],
        'complete_cast': ['movie_id', 'subject_id'],
        'movie_companies': ['company_id', 'company_type_id', 'movie_id'],
        'movie_info': ['movie_id'],
        'movie_info_idx': ['info_type_id', 'movie_id'],
        'movie_keyword': ['keyword_id', 'movie_id'],
        'movie_link': ['link_type_id', 'movie_id'],
        'kind_type': ['id'],
        'comp_cast_type': ['id'],
        'company_name': ['id'],
        'company_type': ['id'],
        'info_type': ['id'],
        'keyword': ['id'],
        'link_type': ['id']
    },
    'join_clauses': [
        'title.id=aka_title.movie_id',
        'title.id=cast_info.movie_id',
        'title.id=complete_cast.movie_id',
        'title.id=movie_companies.movie_id',
        'title.id=movie_info.movie_id',
        'title.id=movie_info_idx.movie_id',
        'title.id=movie_keyword.movie_id',
        'title.id=movie_link.movie_id',
        'title.kind_id=kind_type.id',
        'comp_cast_type.id=complete_cast.subject_id',
        'company_name.id=movie_companies.company_id',
        'company_type.id=movie_companies.company_type_id',
        'movie_info_idx.info_type_id=info_type.id',
        'keyword.id=movie_keyword.keyword_id',
        'link_type.id=movie_link.link_type_id',
    ],

    'all_join_clauses': [
            'title.id=aka_title.movie_id',
            'title.id=cast_info.movie_id',
            'title.id=complete_cast.movie_id',
            'title.id=movie_companies.movie_id',
            'title.id=movie_info.movie_id',
            'title.id=movie_info_idx.movie_id',
            'title.id=movie_keyword.movie_id',
            'title.id=movie_link.movie_id',

            'aka_title.movie_id=cast_info.movie_id',
            'aka_title.movie_id=complete_cast.movie_id',
            'aka_title.movie_id=movie_companies.movie_id',
            'aka_title.movie_id=movie_info.movie_id',
            'aka_title.movie_id=movie_info_idx.movie_id',
            'aka_title.movie_id=movie_keyword.movie_id',
            'aka_title.movie_id=movie_link.movie_id',

            'cast_info.movie_id=complete_cast.movie_id',
            'cast_info.movie_id=movie_companies.movie_id',
            'cast_info.movie_id=movie_info.movie_id',
            'cast_info.movie_id=movie_info_idx.movie_id',
            'cast_info.movie_id=movie_keyword.movie_id',
            'cast_info.movie_id=movie_link.movie_id',

            'complete_cast.movie_id=movie_companies.movie_id',
            'complete_cast.movie_id=movie_info.movie_id',
            'complete_cast.movie_id=movie_info_idx.movie_id',
            'complete_cast.movie_id=movie_keyword.movie_id',
            'complete_cast.movie_id=movie_link.movie_id',

            'movie_companies.movie_id=movie_info.movie_id',
            'movie_companies.movie_id=movie_info_idx.movie_id',
            'movie_companies.movie_id=movie_keyword.movie_id',
            'movie_companies.movie_id=movie_link.movie_id',

            'movie_info.movie_id=movie_info_idx.movie_id',
            'movie_info.movie_id=movie_keyword.movie_id',
            'movie_info.movie_id=movie_link.movie_id',

            'movie_info_idx.movie_id=movie_keyword.movie_id',
            'movie_info_idx.movie_id=movie_link.movie_id',

            'movie_keyword.movie_id=movie_link.movie_id',

            'title.kind_id=kind_type.id',
            'comp_cast_type.id=complete_cast.subject_id',
            'company_name.id=movie_companies.company_id',
            'company_type.id=movie_companies.company_type_id',
            'movie_info_idx.info_type_id=info_type.id',
            'keyword.id=movie_keyword.keyword_id',
            'link_type.id=movie_link.link_type_id',
        ],
    'join_root': 'title',
    'join_how': 'outer',
    'join_name': 'job-m',
    'use_cols': 'multi',
    'epochs': 10,
    'bs': 1000,
    'resmade_drop_prob': 0.1,
    'max_steps': 1000,
    'loader_workers': 8,
    'sampler': 'factorized_sampler',
    'sampler_batch_size': 128,
    'warmups': 0.15,
    # Eval:
    'compute_test_loss': False,
    'queries_csv': './queries/job-m.csv',
    'num_eval_queries_per_iteration': 0,
    'num_eval_queries_at_end': 113,
    'eval_psamples': [1000],
}

JOB_M_FACTORIZED = {
    'factorize': True,
    'factorize_blacklist': [],
    'factorize_fanouts': True,
    'word_size_bits': 14,
    'bs': 2048,
    'max_steps': 512,
    'epochs': 20,
    'epochs_per_iteration': 1,
}

### EXPERIMENT CONFIGS ###
# Run multiple experiments concurrently by using the --run flag, ex:
# $ ./kd.py --run job-light
EXPERIMENT_CONFIGS = {
    # JOB-light, NeuroCard base.
    'job-light': dict(
        dict(BASE_CONFIG, **JOB_LIGHT_BASE),
        **{
            'factorize': True,
            'grouped_dropout': True,
            'loader_workers': 4,
            'warmups': 0.05,  # Ignored.
            'lr_scheduler': tune.grid_search(['OneCycleLR-0.28']),
            'loader_workers': 4,
            'max_steps': tune.grid_search([500]),
            'epochs': 10,
            'num_eval_queries_per_iteration': 70,
            'input_no_emb_if_leq': False,
            'eval_psamples': [8000],
            'epochs_per_iteration': 1,
            'resmade_drop_prob': tune.grid_search([.1]),
            'label_smoothing': tune.grid_search([0]),
            'word_size_bits': tune.grid_search([11]),
        }),


    'job-light-ranges-mscn-workload': dict(
    dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **FACTORIZE)),
    **{
        # Train for a bit and checks that these metrics are reasonable.
        'use_cols': 'content',
        'num_eval_queries_per_iteration': 100,
        # 10M tuples total.
        'max_steps': tune.grid_search([500]),
        'epochs': 10,
        # Evaluate after every 1M tuples trained.
        'epochs_per_iteration': 1,
        'loader_workers': 4,
        'layers':4, 
        'input_no_emb_if_leq': False,
        'resmade_drop_prob': tune.grid_search([0]),
        'label_smoothing': tune.grid_search([0]),
        'fc_hiddens': 32,
        'embed_size': tune.grid_search([8]),
        'word_size_bits': tune.grid_search([14]),
        'table_dropout': False,
        'lr_scheduler': None,   
        'warmups': tune.grid_search([0.07]),
        'queries_csv': './queries/mscn_queries_neurocard_format.csv',
        'subqueries_csv': None,
        'job_light_queries_csv': './queries/job-light.csv',
        'semi_train': True,
        'q_weight': tune.grid_search([10]),
        'checkpoint_every_epoch': True,
        'eval_psamples': [1000, 8000],
        'bs':128,
        'train_queries': 10000,
        'test_queries': 0,
        'train_virtual_cols': True,
        'run_uaeq': False,
        'save_result_dir': "test_results/preds/",
        'save_model_dir': "test_results/models/"
        },
    ),
    'stu': dict(
    dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **FACTORIZE)),
    **{
        # Train for a bit and checks that these metrics are reasonable.
        'use_cols': 'content',
        'num_eval_queries_per_iteration': 70,
        # 10M tuples total.
        'max_steps': tune.grid_search([500]),
        'epochs': 10, 
        # Evaluate after every 1M tuples trained.
        'epochs_per_iteration': 1,
        'loader_workers': 4,
        'layers':4, 
        'input_no_emb_if_leq': False,
        'resmade_drop_prob': tune.grid_search([0]),
        'label_smoothing': tune.grid_search([0]),
        'fc_hiddens': 32,
        'embed_size': tune.grid_search([4]),
        'word_size_bits': tune.grid_search([14]),
        'table_dropout': False,
        'lr_scheduler': None,
        'warmups': tune.grid_search([0.07]),
        'queries_csv': './queries/mscn_queries_neurocard_format.csv',
        'subqueries_csv': None,
        'job_light_queries_csv': './queries/job-light.csv',
        'semi_train': True,
        'q_weight': tune.grid_search([10]),
        'checkpoint_every_epoch': True,
        'eval_psamples': [8000],
        'bs':512,
        'train_queries': 2000,
        'test_queries': 0 ,
        'train_virtual_cols': True,
        'run_uaeq': False,
        'save_result_dir': "test_results/preds/",
        'save_model_dir': "test_results/models/"
        },
    ),
    'teacher': dict(
    dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **FACTORIZE)),
    **{
        # Train for a bit and checks that these metrics are reasonable.
        'use_cols': 'content',
        'num_eval_queries_per_iteration': 100,
        # 10M tuples total.
        'max_steps': tune.grid_search([500]),
        'epochs': 10,
        # Evaluate after every 1M tuples trained.
        'epochs_per_iteration': 1,
        'loader_workers': 4,
        'layers':6, 
        'input_no_emb_if_leq': False,
        'resmade_drop_prob': tune.grid_search([0]),
        'label_smoothing': tune.grid_search([0]),
        'fc_hiddens': 128,
        'embed_size': tune.grid_search([16]),
        'word_size_bits': tune.grid_search([14]),
        'table_dropout': False,
        'lr_scheduler': None,
        'warmups': tune.grid_search([0.07]),
        'queries_csv': './queries/mscn_queries_neurocard_format.csv',
        'subqueries_csv': None,
        'job_light_queries_csv': './queries/job-light.csv',
        'semi_train': True,
        'q_weight': tune.grid_search([10]),
        'checkpoint_every_epoch': True,
        'eval_psamples': [8000],
        'bs':512,
        'train_queries': 2000,
        'test_queries': 0,
        'train_virtual_cols': True,
        'run_uaeq': False,
        'save_result_dir': "test_results/preds/",
        'save_model_dir': "test_results/models/"
        },
    ),

    'job-light-mscn-workload': dict(
    dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **FACTORIZE)),
    **{
        # Train for a bit and checks that these metrics are reasonable.
        'use_cols': 'simple',
        'num_eval_queries_per_iteration': 1000,
        # 10M tuples total.
        'max_steps': tune.grid_search([500]),
        'epochs': 10,
        # Evaluate after every 1M tuples trained.
        'epochs_per_iteration': 1,
        'loader_workers': 4,
        'input_no_emb_if_leq': False,
        'resmade_drop_prob': tune.grid_search([0]),
        'label_smoothing': tune.grid_search([0]),
        'fc_hiddens': 128,
        'embed_size': tune.grid_search([4]),
        'word_size_bits': tune.grid_search([14]),
        'table_dropout': False,
        'lr_scheduler': None,
        'bs':128,
        'warmups': tune.grid_search([0.07]),
        'queries_csv': './queries/job-light-new.csv',
        'subqueries_csv': None,
        'job_light_queries_csv': './queries/job-light.csv',
        'semi_train': True,
        'q_weight': tune.grid_search([10]),
        'checkpoint_every_epoch': True,
        'eval_psamples': [1000, 8000],
        'train_queries': 10000,
        'test_queries': 0,
        'train_virtual_cols': True,
        'run_uaeq': False,
        'save_result_dir': "test_results/preds/",
        'save_model_dir': "test_results/models/"
        },
    ),
    'job-m': dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
                  **JOB_M_FACTORIZED),
        **{
            'num_eval_queries_per_iteration': 100,
            'fc_hiddens':64,
            'layers':2,
            'epochs':10,
            'queries_csv': './queries/mscn_queries_neurocard_format_jobm.csv',
            'subqueries_csv': None,
            'job_light_queries_csv': './queries/job-m.csv',
            'semi_train': True,
            'q_weight': tune.grid_search([10]),
            'checkpoint_every_epoch': True,
            'eval_psamples': [1000, 8000],
            'train_queries': 2000,
            'test_queries': 0,
            'train_virtual_cols': True,
            'run_uaeq': False,
            'save_result_dir': "test_results/preds/",
            'save_model_dir': "test_results/models/"
        }), 
}

######  TEST CONFIGS ######
# These are run by default if you don't specify --run.

TEST_CONFIGS['test-job-light'] = dict(
    EXPERIMENT_CONFIGS['job-light'],
    **{
        # Train for a bit and checks that these metrics are reasonable.
        'epochs': 1,
        'asserts': {
            'fact_psample_8000_median': 4,
            'fact_psample_8000_p99': 50,
            'train_bits': 80,
        },
    })

TEST_CONFIGS['job-light-reload'] = dict(
    EXPERIMENT_CONFIGS['job-light'], **{
        'checkpoint_to_load': tune.grid_search([
            'models/job-light-pretrained.pt',
        ]),
        'eval_psamples': [512, 8000],
        'asserts': {
            'fact_psample_512_median': 1.7,
            'fact_psample_512_p99': 13.5,
            'fact_psample_8000_median': 1.7,
            'fact_psample_8000_p99': 10,
        },
    })

for name in TEST_CONFIGS:
    TEST_CONFIGS[name].update({'save_checkpoint_at_end': False})
EXPERIMENT_CONFIGS.update(TEST_CONFIGS)
