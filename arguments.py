import os
import argparse

from pathlib import Path as path
from datetime import datetime


class arg_bool:
    def __new__(cls, input_s) -> bool:
        return 't' in str(input_s).lower()
       

class CustomArgs:
    
    ############################ Args # Don't modify this line
    
    version = 'colab'
    
    # base setting
    mini_dataset = False
    do_train = True
    do_eval = True
    training_iteration = 5
    save_ckpt = False
    label_level = 'level1'
    model_name_or_path = 'roberta-base'
    data_name = 'pdtb2'
    
    # path 
    data_path = '/content/drive/MyDrive/IDRR/CorpusData/DRR_corpus/pdtb2.csv'
    cache_dir = '/content/drive/MyDrive/IDRR/plm_cache'
    output_dir = './output_space/'
    log_dir = '/content/drive/MyDrive/IDRR/log_space'
    load_ckpt_dir = './ckpt_fold'
    
    # improvement
    loss_type = 'CELoss'
    secondary_label_weight = 0.5
    data_augmentation = False
    
    # epoch, batch, step
    epochs = 5
    max_steps = -1
    train_batch_size = 8
    eval_batch_size = 32
    eval_steps = 100
    log_steps = 10
    gradient_accumulation_steps = 1
    
    # seed, lr
    seed = 2023
    warmup_ratio = 0.05
    weight_decay = 0.01
    learning_rate = 5e-6
    
    # additional setting ( not shown in ArgumentParser ) # Don't modify this line
    trainset_size = -1
    devset_size = -1
    testset_size = -1
    cur_time = '2023-10-16-20-00-36'
    
    ############################ Args # Don't modify this line
    
    def __init__(self, test_setting=False) -> None:
        parser = argparse.ArgumentParser('zp')

        ############################ Args # Don't modify this line
        
        parser.add_argument("--version", type=str, default='colab')
        
        # base setting
        parser.add_argument("--mini_dataset", type=arg_bool, default=False)
        parser.add_argument("--do_train", type=arg_bool, default=True)
        parser.add_argument("--do_eval", type=arg_bool, default=True)
        parser.add_argument("--training_iteration", type=int, default=5)
        parser.add_argument("--save_ckpt", type=arg_bool, default=False)
        parser.add_argument("--label_level", type=str, default='level1')
        parser.add_argument("--model_name_or_path", type=str, default='roberta-base')
        parser.add_argument("--data_name", type=str, default='pdtb2')
        
        # path
        parser.add_argument("--data_path", type=str, default='/content/drive/MyDrive/IDRR/CorpusData/DRR_corpus/pdtb2.csv')
        parser.add_argument("--cache_dir", type=str, default='/content/drive/MyDrive/IDRR/plm_cache')
        parser.add_argument("--output_dir", type=str, default='./output_space/')
        parser.add_argument("--log_dir", type=str, default='/content/drive/MyDrive/IDRR/log_space')
        parser.add_argument("--load_ckpt_dir", type=str, default='./ckpt_fold')
        
        # improvement
        parser.add_argument("--loss_type", type=str, default='CELoss')
        parser.add_argument("--secondary_label_weight", type=float, default=0.5)
        parser.add_argument("--data_augmentation", type=arg_bool, default=False)
        
        # epoch, batch, step
        parser.add_argument("--epochs", type=int, default=5)
        parser.add_argument("--max_steps", type=int, default=-1)
        parser.add_argument("--train_batch_size", type=int, default=8)
        parser.add_argument("--eval_batch_size", type=int, default=32)
        parser.add_argument("--eval_steps", type=int, default=100)
        parser.add_argument("--log_steps", type=int, default=10)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
        
        # seed, lr
        parser.add_argument("--seed", type=int, default=2023)
        parser.add_argument("--warmup_ratio", type=float, default=0.05)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--learning_rate", type=float, default=5e-6)
        
        ############################ Args # Don't modify this line

        args = parser.parse_args()
        for k, v in args.__dict__.items():
            setattr(self, k, v)
            
        if test_setting:
            self.version = 'colab_test'
            self.mini_dataset = True
            self.data_augmentation = False
            self.training_iteration = 2
            self.train_batch_size = 8
            self.eval_batch_size = 8
            self.epochs = 2
            self.eval_steps = 4
            self.log_steps = 4
    
    def complete_path(self):
        self.cur_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # train_eval_string = '_train'*self.do_train + '_eval'*self.do_eval
        specific_fold_name = f'{self.cur_time}_{self.version}_{self.data_name}_{self.label_level}'
        self.output_dir = os.path.join(self.output_dir, specific_fold_name)
        self.log_dir = os.path.join(self.log_dir, specific_fold_name) 
    
    def recalculate_eval_log_steps(self, sample_per_eval=800, sample_per_log=80):
        real_batch_size = self.train_batch_size*self.gradient_accumulation_steps
        self.eval_steps = sample_per_eval // real_batch_size
        self.log_steps = sample_per_log // real_batch_size
        
    def check_path(self):
        # self.data_path
        # self.cache_dir
        # self.output_dir
        # self.log_dir
        # self.load_ckpt_dir
        
        assert path(self.data_path).exists(), 'wrong data path'
        
        if str(self.cache_dir) in ['None', '']:
            self.cache_dir = None
        else:
            path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            
        path(self.output_dir).mkdir(parents=True, exist_ok=True)
        path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        if not self.do_train and self.do_eval and not path(self.load_ckpt_dir).exists():
            raise Exception('no do_train and load_ckpt_dir does not exist')  
        
    def __iter__(self):
        # keep the same order as the args shown in the file
        keys_order = {k:-1 for k in self.__dict__}
        with open('./arguments.py', 'r', encoding='utf8')as f:
            sep_label = 0
            cnt = 0
            for line in f.readlines():
                if line.count('#') > 3 and 'Args' in line:
                    if sep_label:
                        break
                    else:
                        sep_label = 1
                    continue
                if sep_label:
                    for k in keys_order:
                        if k in line and keys_order[k] == -1:
                            keys_order[k] = cnt
                            cnt += 1
                            break
        
        return iter(sorted(self.__dict__.items(), key=lambda x:keys_order[x[0]]))
        
    def generate_script(self, file_path='./tmp/script.sh'):
        script_string = ['python main.py']
        for k, v in list(self):
            if k in ['cur_time']:
                continue
            script_string.append(f'    --{k} {v}')
        script_string = ' \\\n'.join(script_string)
        
        print(script_string)
        path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf8')as f:
            f.write(script_string)
        
    def generate_parser(self, file_path='./tmp/arguments.py'):
        with open('./arguments.py', 'r', encoding='utf8')as f:
            contents = f.readlines()
            
        parser_lines, pre_lines, post_lines = [], [], []
        sep_label = 0
        for line in contents:
            if (line.count('#') > 3 and 'Args' in line) or 'additional setting' in line:
                sep_label += 1
            if sep_label < 4:
                pre_lines.append(line)
            elif sep_label >= 5:
                post_lines.append(line)
                
            if sep_label != 1:
                continue
            
            if line.count('=') == 1:
                k, v = line.split('=')
                k, v = k.strip(), v.strip()
                if '\'' in v or '"' in v:
                    v_type = 'str'
                elif v in ['True', 'False']:
                    v_type = 'arg_bool'
                elif float(v) == int(float(v)):
                    v_type = 'int'
                else:
                    v_type = 'float'
                parser_lines.append(f'parser.add_argument("--{k}", type={v_type}, default={v})')
            else:
                parser_lines.append(line.strip())
                    
        path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf8')as f:
            f.write(''.join(pre_lines))
            for line in parser_lines:
                print(line)
                f.write(' '*8+line+'\n')
            f.write(''.join(post_lines))


if __name__ == '__main__':
    sample_args = CustomArgs(test_setting=False)
    # print(list(sample_args))
    # print(dict(sample_args))
    sample_args.generate_script()
    sample_args.generate_parser()