import arguments, main


def run_local_test():
    args = arguments.Args()
    
    args.version = 'local'
    args.data_path = r'D:\0--data\projects\04.01-IDRR数据\IDRR-base\CorpusData\PDTB2\pdtb2.csv'
    args.load_ckpt_dir = r'D:\0--data\projects\04.01-IDRR数据\IDRR-base\output_space\2023-10-06-19-56-37_local__train\checkpoint\best_acc'
    
    args.do_train = True
    args.do_eval = True
    args.train_batch_size = 8
    args.eval_batch_size = 8
    args.max_steps = 20
    args.epochs = 2
    args.eval_steps = 10
    args.log_steps = 10

    main.main(args)
    
    
if __name__ == '__main__':
    run_local_test()