# ===== prepare server_name, root_fold =====
SERVER_NAME = 'cu12_'
if SERVER_NAME in ['cu12_', 'cu13_', 'northern_']:
    ROOT_FOLD_IDRR = '/data/zpwang/IDRR/'
# elif SERVER_NAME == :
#     ROOT_FOLD_IDRR = ''
else:
    raise Exception('wrong ROOT_FOLD_IDRR')

import os
os.chdir(ROOT_FOLD_IDRR+'IDRR-base/')

# ===== prepare gpu =====
from gpuManager import GPUManager
free_gpu_ids = GPUManager.get_some_free_gpus(gpu_cnt=1)
os.environ["CUDA_VISIBLE_DEVICES"] = free_gpu_ids
print(f'=== CUDA {free_gpu_ids} ===')

# ===== import =====
from arguments import CustomArgs
from corpusData import CustomCorpusData
from main import main


def server_base_args(test_setting=False, data_name='pdtb2', label_level='level1') -> CustomArgs:
    args = CustomArgs(test_setting=test_setting)
    
    args.version = SERVER_NAME+('test' if test_setting else 'base')
        
    args.data_name = data_name
    if data_name == 'pdtb2':
        args.data_path = ROOT_FOLD_IDRR+'CorpusData/PDTB2/pdtb2.csv'
    elif data_name == 'pdtb3':
        args.data_path = ROOT_FOLD_IDRR+'CorpusData/PDTB3/pdtb3_implicit.csv'
    elif data_name == 'conll':
        args.data_path = ROOT_FOLD_IDRR+'CorpusData/CoNLL16/'
    args.cuda_id = free_gpu_ids
    args.label_level = label_level
    
    args.model_name_or_path = ROOT_FOLD_IDRR+'/plm_cache/models--roberta-base/snapshots/bc2764f8af2e92b6eb5679868df33e224075ca68'
    args.load_ckpt_dir = ROOT_FOLD_IDRR+'ckpt_fold'
    args.cache_dir = ''
    # args.output_dir = ROOT_FOLD_IDRR+'output_space/'
    args.output_dir = '/home/zpwang/IDRR/output_space/'  # TODO: consume lots of memory
    if test_setting:
        args.log_dir = ROOT_FOLD_IDRR+'log_space_test/'
    else:
        args.log_dir = ROOT_FOLD_IDRR+'log_space/'

    return args


def server_long_args(data_name='pdtb2', label_level='level1'):
    args = server_base_args(test_setting=False, data_name=data_name, label_level=label_level)
    args:CustomArgs
    
    args.epochs = 25
    args.learning_rate = 3e-5
    args.warmup_ratio = 0.1
    
    args.version = 'cu12_long_bs256'
    args.train_batch_size = 16
    args.gradient_accumulation_steps = 16
    # args.eval_per_epoch = 4
    return args


def server_dataAug_args(args=None, data_name='pdtb2'):
    if not args:
        args = server_base_args(test_setting=False, data_name=data_name)
    args.version = SERVER_NAME+'dataAugmentation'
    args.data_augmentation_connective_arg2 = True
    
    return args

    
if __name__ == '__main__':
    main(server_base_args(test_setting=True, data_name='pdtb2'))
    # main(server_base_args(test_setting=True, data_name='pdtb2', label_level='level2'))
    # main(server_base_args(test_setting=True, data_name='pdtb3'))
    # main(server_base_args(test_setting=True, data_name='conll'))
    # main(server_dataAug_args())
    main(server_base_args())
    main(server_long_args())
    pass