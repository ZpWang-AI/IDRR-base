from arguments import CustomArgs
from main import main


def local_test_args(data_name='pdtb2', label_level='level1'):
    args = CustomArgs(test_setting=True)
    
    args.version = 'test'
    args.server_name = 'local'
    
    args.data_name = data_name
    if data_name == 'pdtb2':
        args.data_path = './CorpusData/PDTB2/pdtb2.csv'
    elif data_name == 'pdtb3':
        args.data_path = './CorpusData/PDTB3/pdtb3_implicit.csv'
    elif data_name == 'conll':
        args.data_path = './CorpusData/CoNLL16/'  
    args.label_level = label_level  
    
    args.model_name_or_path = './plm_cache/models--roberta-base/snapshots/bc2764f8af2e92b6eb5679868df33e224075ca68/'
    args.cache_dir = './plm_cache/'
    args.output_dir = './output_space/'
    args.log_dir = './log_space/'

    return args
    
    
if __name__ == '__main__':
    main(local_test_args('pdtb2'))
    # main(local_test_args('pdtb2', label_level='level2'))
    # main(local_test_args('pdtb3'))
    # main(local_test_args('pdtb3', label_level='level2'))
    # main(local_test_args('conll'))
    pass