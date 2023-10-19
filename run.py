from arguments import CustomArgs
from main import main


def local_test_args():
    args = CustomArgs()
    
    args.version = 'local_test'
    
    args.mini_dataset = True
    
    args.data_path = r'D:\0--data\projects\04.01-IDRR数据\IDRR-base\CorpusData\PDTB2\pdtb2.csv'
    args.load_ckpt_dir = r'D:\0--data\projects\04.01-IDRR数据\IDRR-base\output_space\2023-10-17-09-55-22_local_test__train_eval\training_iteration_0\checkpoint_best_Acc'
    args.cache_dir = './plm_cache/'
    args.output_dir = './output_space/'
    args.log_dir = './log_space/'
    
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
    
    
if __name__ == '__main__':
    main(local_test_args())