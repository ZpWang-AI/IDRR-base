from arguments import CustomArgs
from main import main


def local_test_args():
    args = CustomArgs()
    
    args.version = 'local_test'
    
    args.data_path = r'D:\0--data\projects\04.01-IDRR数据\IDRR-base\CorpusData\PDTB2\pdtb2.csv'
    args.load_ckpt_dir = r'D:\0--data\projects\04.01-IDRR数据\IDRR-base\output_space\2023-10-09-17-33-24_local_test__train_eval\training_iter_0\checkpoint_best_Acc'
    args.cache_dir = './plm_cache/'
    args.output_dir = './output_space/'
    args.log_dir = './log_space/'
    
    args.label_expansion_positive = 0.2
    args.label_expansion_negative = 0.2
    args.data_augmentation = True
    
    args.do_train = True
    args.do_eval = True
    args.training_iteration = 2
    
    args.train_batch_size = 8
    args.eval_batch_size = 8
    args.max_steps = 8
    args.epochs = 2
    args.eval_steps = 4
    args.log_steps = 4

    return args
    

def cu12_test_args():
    args = local_test_args()
    
    args.version = 'cu12_test'
    
    args.data_path = 'CorpusData/DRR_corpus/pdtb2.csv'
    args.load_ckpt_dir = './ckpt_fold'
    args.cache_dir = './plm_cache/'
    args.output_dir = './output_space/'
    args.log_dir = './log_space/'
    
    return args
    
    
if __name__ == '__main__':
    main(local_test_args())
    # main(cu12_test_args())