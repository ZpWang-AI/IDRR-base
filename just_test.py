import arguments, main


def main_test():
    args = arguments.Args()
    args.data_path = r'D:\0--data\projects\04.01-IDRR数据\IDRR-base\CorpusData\PDTB2\pdtb2.csv'
    args.load_ckpt_dir = r'D:\0--data\projects\04.01-IDRR数据\IDRR-base\ckpt\2023-10-06-09-44-33_test_train\ckpt-best_acc'

    args.train_or_test = 'train'
    args.batch_size = 8
    args.max_steps = 200
    args.epochs = 2
    args.eval_steps = 50
    args.log_steps = 10
    
    # args.train_or_test = 'test'

    main.main(args)
    
    

if __name__ == '__main__':
    main_test()