
# ===== prepare server_name, root_fold =====
SERVER_NAME = 'cu12_'
if SERVER_NAME in ['cu12_', 'cu13_', 'northern_']:
    ROOT_FOLD_IDRR = '/data/zpwang/IDRR/'
# elif SERVER_NAME == :
#     ROOT_FOLD_IDRR = ''
else:
    raise Exception('wrong ROOT_FOLD_IDRR')

import os
CODE_SPACE = ROOT_FOLD_IDRR+'IDRR-rank-v1/'
if __name__ == '__main__':
    os.chdir(CODE_SPACE)

    # ===== prepare gpu =====
    from gpuManager import GPUManager
    free_gpu_id = GPUManager.get_free_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpu_id)
    print(f'=== CUDA {free_gpu_id} ===')

    # ===== import =====
    from arguments import CustomArgs
    from main import main


def server_base_args(test_setting=False):
    args = CustomArgs(test_setting=test_setting)
    
    args.version = SERVER_NAME+('test' if test_setting else 'rank')
    
    args.model_name_or_path = ROOT_FOLD_IDRR+'/plm_cache/models--roberta-base/snapshots/bc2764f8af2e92b6eb5679868df33e224075ca68'
    args.data_path = ROOT_FOLD_IDRR+'CorpusData/PDTB2/pdtb2.csv'
    args.load_ckpt_dir = ROOT_FOLD_IDRR+'ckpt_fold'
    args.cache_dir = ''
    # args.output_dir = ROOT_FOLD_IDRR+'output_space/'
    args.output_dir = '/home/zpwang/IDRR/output_space/'  # TODO: consume lots of memory
    args.log_dir = ROOT_FOLD_IDRR+'log_space_rank-v1/'
    args.rank_order_file = CODE_SPACE+'rank_order/rank_order1.json'

    return args


def server_rank_dataAug_args():
    args = server_base_args()
    args.version = SERVER_NAME+'rank_dataAugmentation'
    args.data_augmentation = True
    
    return args

    
if __name__ == '__main__':
    main(server_base_args(test_setting=True))
    # main(server_rank_dataAug_args())