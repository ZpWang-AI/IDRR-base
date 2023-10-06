import arguments, main


def main_test():
    args = arguments.Args()
    args.data_path = r'D:\0--data\projects\04.01-IDRR数据\IDRR-base\CorpusData\PDTB2\pdtb2.csv'

    args.train_or_test = 'train'
    args.max_steps = 200
    args.epochs = 2
    args.eval_steps = 50
    
    args.train_or_test = 'test'
    # args.load_ckpt_dir = './ckpt/2023-10-05-20-50-13_test/ckpt-best_acc/'

    main.main(args)
    
    

if __name__ == '__main__':
    main_test()