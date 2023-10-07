import os
import argparse

from pathlib import Path as path
from datetime import datetime
from typing import Any


class arg_bool:
    def __new__(cls, input_s: str) -> bool:
        return 't' in input_s.lower()
       

class Args:
    
    ############################ Args # Don't delete this line
    
    version = 'test'
    
    do_train = True
    do_eval = True
    label_level = 'level1'
    model_name_or_path = 'roberta-base'
    data_name = 'pdtb2'
    
    data_path = './CorpusData/PDTB-2.0/pdtb2.csv'
    output_dir = './ckpt'
    log_path = 'log.out'
    load_ckpt_dir = './ckpt/2023-10-05-21-04-47_test_train+test'
    
    label_expansion_positive = 0.0
    label_expansion_negative = 0.0
    data_augmentation = False
    
    epochs = 5
    max_steps = -1
    train_batch_size = 8
    eval_batch_size = 32
    eval_steps = 5
    log_steps = 5
    gradient_accumulation_steps = 1
    
    seed = 2023
    warmup_ratio = 0.05
    weight_decay = 0.01
    learning_rate = 5e-6
    
    def __init__(self) -> None:
        
        # === set default values below ===
        parser = argparse.ArgumentParser('zp')
        parser.add_argument("--version", type=str, default='colab')

        parser.add_argument("--do_train", type=arg_bool, default='True')
        parser.add_argument("--do_eval", type=arg_bool, default='False')
        parser.add_argument("--label_level", type=str, default='level1', choices=['level1', 'level2'])
        parser.add_argument("--model_name_or_path", default='roberta-base')
        parser.add_argument("--data_name", type=str, default= "pdtb2" )
        
        parser.add_argument("--label_expansion_positive", type=float, default=0)
        parser.add_argument("--label_expansion_negative", type=float, default=0)
        parser.add_argument("--data_augmentation", type=arg_bool, default='False')
        
        parser.add_argument("--data_path", type=str, default='/content/drive/MyDrive/IDRR/CorpusData/DRR_corpus/pdtb2.csv')
        parser.add_argument("--log_path", type=str, default='log.out')
        # parser.add_argument("--cache_dir", type=str, default='')
        parser.add_argument("--output_dir", type=str, default="./output_space/")
        parser.add_argument("--load_ckpt_dir", type=str, default='./ckpt_fold')
        
        parser.add_argument("--epochs", type=int, default=5)
        parser.add_argument("--max_steps", type=int, default=-1)
        parser.add_argument("--train_batch_size", type=int, default=32)
        parser.add_argument("--eval_batch_size", type=int, default=32)
        parser.add_argument("--eval_steps", type=int, default=100)
        parser.add_argument("--log_steps", type=int, default=10)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

        parser.add_argument("--seed", type=int, default=2023)
        parser.add_argument("--warmup_ratio", type=float, default=0.05)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--learning_rate", type=float, default=5e-6)
        
        args = parser.parse_args()
        for k, v in args.__dict__.items():
            setattr(self, k, v)
            
    def check_path(self):
        assert path(self.data_path).exists(), 'wrong data path'
        cur_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        train_eval_string = '_train'*self.do_train + '_test'*self.do_eval
        self.output_dir = os.path.join(self.output_dir, f'{cur_time}_{self.version}_{train_eval_string}')
        path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.log_path = os.path.join(self.output_dir, self.log_path)
        path(self.log_path).touch()
    
    def __iter__(self):
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
        
        return iter(sorted(self.__dict__.items(), key=lambda x:keys_order[x[0]]))
        
    def generate_script(self):
        script_string = ['python main.py']
        for k, v in list(self):
            script_string.append(f'    --{k} {v}')
        script_string = ' \\\n'.join(script_string)
        print(script_string)
        # exit()


if __name__ == '__main__':
    sample_args = Args()
    sample_args.generate_script()