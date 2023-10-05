import argparse

from pathlib import Path as path


class Args:
    def __init__(self) -> None:
        for attr_name in dir(self):
            if attr_name.startswith('__') and attr_name.endswith('__'):
                continue
            attr_val = getattr(self, attr_name)
            if not callable(attr_val):
                setattr(self, attr_name, attr_val)

        path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        path(self.log_path).touch()
        path(self.output_dir).mkdir(parents=True, exist_ok=True)
        assert not path(self.data_path).exists(), 'wrong data path'

    
    ############################ Args
    
    train_or_test = 'train+test'
    label_level = 'level1'
    
    model_name_or_path = 'roberta-base'
    data_name = 'pdtb2'
    data_path = './CorpusData/PDTB-2.0/pdtb2.csv'
    log_path = './custom_log.log'
    output_dir = './ckpt'
    ckpt_fold = './ckpt/ckpt-best_acc'
    
    batch_size = 8
    eval_steps = 5
    epochs = 4
    
    seed = 2023
    warmup_ratio = 0.05
    weight_decay = 0.01
    learning_rate = 5e-6
        
    def get_from_argparse(self):
        parser = argparse.ArgumentParser("")
        parser.add_argument("--train_or_test", type=str, default='train+test', choices=['train', 'test', 'train+test'])
        parser.add_argument("--label_level", type=str, default='level1', choices=['level1', 'level2'])

        parser.add_argument("--model_name_or_path", default='roberta-base')
        parser.add_argument("--data_name", type=str, default= "pdtb2" )
        parser.add_argument("--data_path", type=str, default='/content/drive/MyDrive/IDRR/CorpusData/DRR_corpus/pdtb2.csv')
        # parser.add_argument("--cache_dir", type=str, default='')
        parser.add_argument("--output_dir", type=str, default="./ckpt/")
        parser.add_argument("--ckpt_fold", type=str, default="./ckpt/ckpt-best_acc")
        
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--eval_steps", type=int, default=5)
        parser.add_argument("--epochs", type=int, default=4)

        parser.add_argument("--seed", type=int, default=2023)
        parser.add_argument("--warmup_ratio", type=float, default=0.05)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--learning_rate", type=float, default=5e-6)
        
        args = parser.parse_args()
        for k, v in args.__dict__.items():
            setattr(self, k, v)
    
    def generate_script(self):
        # keep the same order as the args shown in the file
        keys_order = {k:-1 for k in self.__dict__}
        with open('./arguments.py', 'r', encoding='utf8')as f:
            sep_label = 0
            cnt = 0
            for line in f.readlines():
                if line.count('#') > 3 and 'Args' in line:
                    sep_label = 1
                    continue
                if sep_label:
                    for k in keys_order:
                        if k in line:
                            keys_order[k] = cnt
                            cnt += 1
                            break
                    if cnt >= len(keys_order):
                        break
                    
        script_string = ['python main.py']
        for k, v in sorted(self.__dict__.items(), key=lambda x:keys_order[x[0]]):
            script_string.append(f'    -- {k} {v}')
        script_string = ' \\\n'.join(script_string)
        print(script_string)
        # exit()


if __name__ == '__main__':
    sample_args = Args()
    sample_args.generate_script()