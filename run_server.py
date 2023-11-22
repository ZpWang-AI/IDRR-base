# ===== prepare server_name, root_fold =====
SERVER_NAME = 'cu12_'
if SERVER_NAME in ['cu12_', 'cu13_', 'northern_']:
    ROOT_FOLD_IDRR = '/data/zpwang/IDRR/'
# elif SERVER_NAME == :
#     ROOT_FOLD_IDRR = ''
else:
    raise Exception('wrong ROOT_FOLD_IDRR')

import os
BRANCH = 'main'
CODE_SPACE = ROOT_FOLD_IDRR+f'IDRR-{BRANCH}/'
if __name__ == '__main__':
    os.chdir(CODE_SPACE)

# ===== import ===== !!! Don't import torch !!!
from arguments import CustomArgs


def server_base_args(test_setting=False, data_name='pdtb2', label_level='level1') -> CustomArgs:
    args = CustomArgs(test_setting=test_setting)
    
    args.version = 'test' if test_setting else 'base'
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
    args.cache_dir = ''
    # args.output_dir = ROOT_FOLD_IDRR+'output_space/'
    args.output_dir = '/home/zpwang/IDRR/output_space/'  # TODO: consume lots of memory
    if test_setting:
        args.log_dir = ROOT_FOLD_IDRR+f'log_space_test_{BRANCH}/'
    else:
        args.log_dir = ROOT_FOLD_IDRR+f'log_space_{BRANCH}/'

    return args


def server_experiment_args(args=None):
    if args is None:
        args = server_base_args(test_setting=False)
    
    args.secondary_label_weight = 0.
    args.data_augmentation_secondary_label = False
    args.data_augmentation_connective_arg2 = False
    
    args.cuda_cnt = 1
    args.warmup_ratio = 0.05
    args.epochs = 5
    args.train_batch_size = 8
    args.eval_batch_size = 32
    args.eval_steps = 100
    args.log_steps = 10
    args.gradient_accumulation_steps = 1
    args.eval_per_epoch = -1
    
    args.weight_decay = 0.01
    args.learning_rate = 5e-6
    
    args.version = 'base'
    return args
    
    
if __name__ == '__main__':
    from main import main
    
    def experiment_once():
        todo_args = server_base_args(test_setting=True, data_name='pdtb2')
        # todo_args = server_base_args(test_setting=True, data_name='pdtb2', label_level='level2')
        # todo_args = server_base_args(test_setting=True, data_name='pdtb3')
        # todo_args = server_base_args(test_setting=True, data_name='conll')
        todo_args = server_experiment_args()
        
        # todo_args.prepare_gpu(target_mem_mb=10000)  # when gpu usage is low
        todo_args.prepare_gpu(target_mem_mb=-1)
        todo_args.complete_path(
            show_cur_time=True,
            show_server_name=False,
            show_data_name=False,
            show_label_level=False,
        )
        
        main(todo_args)
    
    def experiment_multi_times():
        # TODO: prepare gpu
        cuda_cnt = 2
        cuda_id = CustomArgs().prepare_gpu(target_mem_mb=10500, gpu_cnt=cuda_cnt) 
         
        for epoch in [5,10,20,30]:
            for mu in [5,10,20,30]:
                for batch_size in [32]:
                    todo_args = server_experiment_args()

                    # TODO: prepare args
                    todo_args.version = f'epoch{epoch}_lr{mu}mu_bs{batch_size}*2'
                    todo_args.epochs = epoch
                    todo_args.learning_rate = float(f'{mu}e-6') 
                    todo_args.train_batch_size = batch_size             
                    
                    todo_args.cuda_id = cuda_id
                    todo_args.cuda_cnt = cuda_cnt
                    todo_args.complete_path(
                        show_cur_time=True,
                        show_server_name=False,
                        show_data_name=False,
                        show_label_level=False,
                    )
                    
                    main(todo_args)

    experiment_once()
    # experiment_multi_times()
    pass