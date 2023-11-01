from arguments import CustomArgs
from main import main


def local_test_args(data_name='pdtb2'):
    args = CustomArgs(test_setting=True)
    
    args.version = 'local_test'
    
    args.data_name = data_name
    if data_name == 'pdtb2':
        args.data_path = './CorpusData/PDTB2/pdtb2.csv'
    elif data_name == 'pdtb3':
        args.data_path = './CorpusData/PDTB3/pdtb3_implicit.csv'
    elif data_name == 'conll':
        args.data_path = './CorpusData/CoNLL16/'    
        
    args.load_ckpt_dir = 'ckpt_fold'
    args.cache_dir = './plm_cache/'
    args.output_dir = './output_space/'
    args.log_dir = './log_space/'

    return args
    
    
if __name__ == '__main__':
    main(local_test_args('pdtb2'))
    main(local_test_args('pdtb3'))
    main(local_test_args('conll'))