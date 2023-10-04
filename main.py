import os 
import torch
import pandas as pd 
import argparse
import logging

from pathlib import Path as path
from transformers import TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding, set_seed
from model_custom import BaselineModel
from datasets_custom import CustomDatasets
from utils import compute_metrics, SaveBestModelCallback

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def get_logger(log_file='custom_log.log', logger_name='custom_logger', print_output=False):
    # 创建一个logger
    logger = logging.getLogger(logger_name)

    # 设置全局级别为DEBUG
    logger.setLevel(logging.DEBUG)

    # 创建一个handler，用于写入日志文件
    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)
    if print_output:
        logger.addHandler(ch)
    return logger


def train(args):
    dataset = CustomDatasets(args.data_path)
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

    save_callback.add_trainer(trainer)
    trainer.train()

    return dataset

def eval(args, dataset, model_path, metric_type):
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
    seed = 2023
    model_name_or_path = 'roberta-base'
    data_type = 'pdtb2'
    data_path = './CorpusData/PDTB-2.0/pdtb2.csv'
    log_path = './custom_log.log'
    output_dir = './ckpt'
    
    batch_size = 8
    eval_steps = 5
    epochs = 4
    
    warmup_ratio = 0.05
    weight_decay = 0.01
    learning_rate = 5e-6
    
    def __init__(self) -> None:
        path(self.log_path).touch()
        path(self.output_dir).mkdir(exist_ok=True)
        assert not path(self.data_path).exists(), 'wrong data path'
        
    def get_from_argparse():
        parser = argparse.ArgumentParser("")
        parser.add_argument("--seed", type=int, default=2023)
        parser.add_argument("--model_name_or_path", default='roberta-base')
        parser.add_argument("--data_type",type=str, default= "pdtb2" )
        parser.add_argument("--data_path", type=str, default="./data/pdtb2.csv")
        parser.add_argument("--output_dir", type=str, default="./ckpt/")
        
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--eval_steps", type=int, default=5)
        parser.add_argument("--epochs", type=int, default=4)

        parser.add_argument("--warmup_ratio", type=float, default=0.05)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--learning_rate", type=float, default=5e-6)
        
        args = parser.parse_args()
        return args
    

if __name__ == '__main__':
    args = Args()
    
    set_seed(args.seed)
    train(args)
    dataset = CustomDatasets(args.data_path)
    eval(args, dataset, model_path='/home/destinylu/pdtb/baseline/ckpt/checkpoint-best-dev_acc', metric_type='acc')