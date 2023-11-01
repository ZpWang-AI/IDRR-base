
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
free_gpu_id = GPUManager.get_free_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpu_id)
print(f'=== CUDA {free_gpu_id} ===')

# ===== import =====
from arguments import CustomArgs
from main import main


def server_base_args(test_setting=False, data_name='pdtb2'):
    args = CustomArgs(test_setting=test_setting)
    
    args.version = SERVER_NAME+('test' if test_setting else 'base')
        
    args.data_name = data_name
    if data_name == 'pdtb2':
        args.data_path = ROOT_FOLD_IDRR+'CorpusData/PDTB2/pdtb2.csv'
    elif data_name == 'pdtb3':
        args.data_path = ROOT_FOLD_IDRR+'CorpusData/PDTB3/pdtb3_implicit.csv'
    elif data_name == 'conll':
        args.data_path = ROOT_FOLD_IDRR+'CorpusData/CoNLL16/'
    
    args.model_name_or_path = ROOT_FOLD_IDRR+'/plm_cache/models--roberta-base/snapshots/bc2764f8af2e92b6eb5679868df33e224075ca68'
    args.load_ckpt_dir = ROOT_FOLD_IDRR+'ckpt_fold'
    args.cache_dir = ''
    args.output_dir = ROOT_FOLD_IDRR+'output_space/'
    args.log_dir = ROOT_FOLD_IDRR+'log_space/'

    return args


def server_dataAug_args(data_name='pdtb2'):
    args = server_base_args(test_setting=False, data_name=data_name)
    args.version = SERVER_NAME+'dataAugmentation'
    args.data_augmentation = True
    
    return args

    
if __name__ == '__main__':
    main(server_base_args(test_setting=True, data_name='pdtb2'))
    main(server_base_args(test_setting=True, data_name='pdtb3'))
    main(server_base_args(test_setting=True, data_name='conll'))
    # main(server_dataAug_args())