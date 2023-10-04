import os 
import torch
import pandas as pd 
import argparse

from pathlib import Path as path
from transformers import TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding, set_seed
from model_custom import BaselineModel
from datasets_custom import CustomDatasets
from utils import (get_logger,
                   create_file_or_fold,
                   compute_metrics, 
                   SaveBestModelCallback, 
                   )

os.environ['TOKENIZERS_PARALLELISM'] = 'false'



def train(args, dataset):
    # import pdb; pdb.set_trace()
    # Training arguments
    
    training_args = TrainingArguments(
        output_dir = args.output_dir, 
        optim='adamw_torch',
        evaluation_strategy = "steps", 
        eval_steps = args.eval_steps,
        learning_rate = args.learning_rate,
        logging_strategy='steps',
        logging_steps=1,
        lr_scheduler_type = 'linear',
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        num_train_epochs = args.epochs,
        weight_decay = args.weight_decay,
        warmup_ratio = args.warmup_ratio,
        save_strategy = 'no',

    )

    model = BaselineModel(args.model_name_or_path)
    save_callback = SaveBestModelCallback()
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=dataset.train_dataset,
        eval_dataset=dataset.dev_dataset, 
        tokenizer=dataset.tokenizer, 
        data_collator=dataset.data_collator,
        compute_metrics=compute_metrics,
        callbacks=[save_callback],
    )

    save_callback.trainer = trainer
    trainer.train()

    return dataset

def evaluate(args, dataset, model_path, metric_type):
    # import pdb; pdb.set_trace()
    # Training arguments
    training_args = TrainingArguments(
        output_dir = args.output_dir, 
        optim='adamw_torch',
        evaluation_strategy = "steps", 
        eval_steps = args.eval_steps,
        learning_rate = args.learning_rate,
        logging_strategy='steps',
        logging_steps=1,
        lr_scheduler_type = 'linear',
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        num_train_epochs = args.epochs,
        weight_decay = args.weight_decay,
        warmup_ratio = args.warmup_ratio,
        save_strategy = 'no',

    )


    model = BaselineModel(args.model_name_or_path)
    print(model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin')), strict=True))

    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=dataset.train_dataset,
        eval_dataset=dataset.dev_dataset, 
        tokenizer=dataset.tokenizer, 
        data_collator=dataset.data_collator,
        compute_metrics=compute_metrics,
    )

    metric_res = trainer.evaluate(dataset.test_dataset)
    for k, v in metric_res.items():
        if metric_type in k:
            print(f'{metric_type}: {v}')
            break


class Args:
    def __init__(self) -> None:
        create_file_or_fold(self.log_path)
        create_file_or_fold(self.output_dir)
        assert not path(self.data_path).exists(), 'wrong data path'

        for attr_name in dir(self):
            if attr_name.startswith('__') and attr_name.endswith('__'):
                continue
            attr_val = getattr(self, attr_name)
            if not callable(attr_val):
                setattr(self, attr_name, attr_val)
    
    ############################ Args
    
    train_or_test = 'train+test'
    
    model_name_or_path = 'roberta-base'
    data_name = 'pdtb2'
    data_path = './CorpusData/PDTB-2.0/pdtb2.csv'
    log_path = './custom_log.log'
    output_dir = './ckpt'
    
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

        parser.add_argument("--model_name_or_path", default='roberta-base')
        parser.add_argument("--data_name", type=str, default= "pdtb2" )
        parser.add_argument("--data_path", type=str, default="./data/pdtb2.csv")
        parser.add_argument("--output_dir", type=str, default="./ckpt/")
        parser.add_argument("--model_path", type=str, default="./ckpt/checkpoint-best-dev_acc")
        
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
        with open('./main.py', 'r', encoding='utf8')as f:
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
        exit()
    

if __name__ == '__main__':
    args = Args()
    args.data_path = '/content/drive/MyDrive/IDRR/CorpusData/DRR_corpus/pdtb2.csv'
    # args.generate_script()
    
    dataset = CustomDatasets(args.data_path, data_name=args.data_name)
    set_seed(args.seed)
    
    if args.train_or_test == 'train':
        train(args, dataset)
    elif args.train_or_test == 'test':
        evaluate(args, dataset, model_path='/home/destinylu/pdtb/baseline/ckpt/checkpoint-best-dev_acc', metric_type='acc')
    else:
        train(args)
        evaluate(args, dataset, model_path='/home/destinylu/pdtb/baseline/ckpt/checkpoint-best-dev_acc', metric_type='acc')
        