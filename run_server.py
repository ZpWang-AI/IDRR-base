# ===== prepare server_name, root_fold =====
SERVER_NAME = 'cu12_'
if SERVER_NAME in ['cu12_', 'cu13_', 'northern_']:
    ROOT_FOLD_IDRR = '/data/zpwang/IDRR/'
# elif SERVER_NAME == :
#     ROOT_FOLD_IDRR = ''
else:
    raise Exception('wrong ROOT_FOLD_IDRR')

import os
CODE_SPACE = ROOT_FOLD_IDRR+'IDRR-rank-v2-multi-stage/'
if __name__ == '__main__':
    os.chdir(CODE_SPACE)

# ===== import ===== !!! Don't import torch !!!
from arguments import CustomArgs, StageArgs


def server_base_args(test_setting=False, data_name='pdtb2', label_level='level1') -> CustomArgs:
    args = CustomArgs(test_setting=test_setting)
    
    args.version = SERVER_NAME+('test' if test_setting else 'base')
    args.server_name = SERVER_NAME
    
    # data
    args.data_name = data_name
    if data_name == 'pdtb2':
        args.data_path = ROOT_FOLD_IDRR+'CorpusData/PDTB2/pdtb2.csv'
    elif data_name == 'pdtb3':
        args.data_path = ROOT_FOLD_IDRR+'CorpusData/PDTB3/pdtb3_implicit.csv'
    elif data_name == 'conll':
        args.data_path = ROOT_FOLD_IDRR+'CorpusData/CoNLL16/'
    args.label_level = label_level
    
    # file path
    args.model_name_or_path = ROOT_FOLD_IDRR+'/plm_cache/models--roberta-base/snapshots/bc2764f8af2e92b6eb5679868df33e224075ca68'
    args.load_ckpt_dir = ROOT_FOLD_IDRR+'ckpt_fold'
    args.cache_dir = ''
    # args.output_dir = ROOT_FOLD_IDRR+'output_space/'
    args.output_dir = '/home/zpwang/IDRR/output_space/'  # TODO: consume lots of memory
    if test_setting:
        args.log_dir = ROOT_FOLD_IDRR+'log_space_test_rank-v2-multi-stage/'
    else:
        args.log_dir = ROOT_FOLD_IDRR+'log_space_rank-v2-multi-stage/'

    return args


def server_v1_args():
    args = server_base_args(test_setting=False,)
    
    args.cuda_cnt = 1
    args.secondary_label_weight = 0
    args.data_augmentation_secondary_label = False
    args.data_augmentation_connective_arg2 = False
    args.rank_data_sampler = 'shuffle'
    args.rank_dataset_size_multiplier = 1
    
    args.training_stages = [
        StageArgs(
            stage_name='rank',
            epochs=2,
            train_batch_size=1,
            eval_batch_size=1,
            eval_steps=800,
            log_steps=40,
            gradient_accumulation_steps=2,
            eval_per_epoch=-15,
            warmup_ratio=0.05,
            weight_decay=0.01,
            learning_rate=5e-6,
        ),
        StageArgs(
            stage_name='ft',
            epochs=5,
            train_batch_size=8,
            eval_batch_size=32,
            eval_steps=100,
            log_steps=10,
            gradient_accumulation_steps=1,
            eval_per_epoch=-4,
            warmup_ratio=0.05,
            weight_decay=0.01,
            learning_rate=5e-6,
        ),
    ]
    
    args.version = SERVER_NAME+'v1-arg'
    return args


def server_long_args(data_name='pdtb2', label_level='level1'):
    args = server_base_args(test_setting=False, data_name=data_name, label_level=label_level)
    args:CustomArgs
    
    args.version = SERVER_NAME+'long_best'
    
    args.cuda_cnt = 2
    args.secondary_label_weight = 0.5
    
    args.training_stages = [StageArgs(
        stage_name='ft',
        epochs=25,
        train_batch_size=32,
        eval_batch_size=32,
        eval_steps=800, log_steps=80,
        gradient_accumulation_steps=1,
        eval_per_epoch=4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=3e-5,
    )]
    return args


def server_dataAug_args(args=None, data_name='pdtb2'):
    if not args:
        args = server_base_args(test_setting=False, data_name=data_name)
    args.version = SERVER_NAME+'ConnDA'
    args.data_augmentation_connective_arg2 = True
    
    return args
    
    
if __name__ == '__main__':
    # ===== choose args =====
    todo_args = server_base_args(test_setting=True, data_name='pdtb2')
    # todo_args = server_base_args(test_setting=True, data_name='pdtb2', label_level='level2')
    # todo_args = server_base_args(test_setting=True, data_name='pdtb3')
    # todo_args = server_base_args(test_setting=True, data_name='conll')
    # todo_args = server_base_args()
    # todo_args = server_long_args()
    todo_args = server_v1_args()
    
    todo_args.prepare_gpu(target_mem_mb=10000)
    todo_args.complete_path(
        show_cur_time=True,
        show_data_name=False,
        show_label_level=False,
    )
    
    from main import main
    main(todo_args)
    pass