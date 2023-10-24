from arguments import CustomArgs
from main import main


def local_test_args():
    args = CustomArgs(test_setting=True)
    
    args.version = 'local_test'
    
    args.data_path = './CorpusData/PDTB2/pdtb2.csv'
    args.load_ckpt_dir = 'ckpt_fold'
    args.cache_dir = './plm_cache/'
    args.output_dir = './output_space/'
    args.log_dir = './log_space/'
    # args.rank_gradient_accumulation_steps

    return args
    
    
if __name__ == '__main__':
    main(local_test_args())