ROOT_FOLD_IDRR = '/data/zpwang/IDRR/'
SERVER_NAME = 'cu12_'

import os
os.chdir(ROOT_FOLD_IDRR+'IDRR-base/')

# prepare gpu
from gpuManager import GPUManager
free_gpu_id = GPUManager.get_free_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpu_id)
print(f'=== CUDA {free_gpu_id} ===')

from arguments import CustomArgs
from main import main


def cu12_base_args():
    args = CustomArgs()
    
    args.version = SERVER_NAME+'rank'
    
    args.model_name_or_path = ROOT_FOLD_IDRR+'/plm_cache/models--roberta-base/snapshots/bc2764f8af2e92b6eb5679868df33e224075ca68'
    args.data_path = ROOT_FOLD_IDRR+'CorpusData/PDTB2/pdtb2.csv'
    args.load_ckpt_dir = ROOT_FOLD_IDRR+'ckpt_fold'
    args.cache_dir = ''
    args.output_dir = ROOT_FOLD_IDRR+'output_space/'
    args.log_dir = ROOT_FOLD_IDRR+'log_space/'
    args.rank_order_file = ROOT_FOLD_IDRR+'IDRR-base/rank_order/rank_order1.json'

    return args


def cu12_test_args():
    args = cu12_base_args()
    
    args.version = SERVER_NAME+'test'
    
    args.mini_dataset = True
    
    args.data_augmentation = False
    
    args.do_train = True
    args.do_eval = True
    args.training_iteration = 2
    args.save_ckpt = True

    # args.max_steps = -1
    args.train_batch_size = 8
    args.eval_batch_size = 8
    args.epochs = 2
    args.eval_steps = 4
    args.log_steps = 4
    
    args.rank_epochs = 2
    args.rank_gradient_accumulation_steps = 8
    args.rank_eval_steps = 2
    args.rank_log_steps = 1

    return args


def cu12_rank_dataAug_args():
    args = cu12_base_args()
    args.version = SERVER_NAME+'rank_dataAugmentation'
    args.data_augmentation = True
    
    return args

    
if __name__ == '__main__':
    # main(cu12_test_args())
    # main(cu12_base_args())
    main(cu12_rank_dataAug_args())