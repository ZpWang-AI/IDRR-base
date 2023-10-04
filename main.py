import os 
import torch
import pandas as pd 
import argparse

from pathlib import Path as path
from transformers import TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding, set_seed

from utils import (get_logger,
                   create_file_or_fold,
                   compute_metrics, 
                   SaveBestModelCallback, 
                   )
from datasets_custom import CustomDatasets
from model_custom import BaselineModel
from arguments import Args

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def train(args:Args, dataset):
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

def evaluate(args:Args, dataset, metric_type):
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
    print(model.load_state_dict(torch.load(os.path.join(args.ckpt_fold, 'pytorch_model.bin')), strict=True))

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
        

if __name__ == '__main__':
    args = Args()
    # args.generate_script()
    
    logger = get_logger(
        log_file=args.log_path,
        print_output=True,
    )
    
    dataset = CustomDatasets(
        file_path=args.data_path,
        data_name=args.data_name,
        model_name_or_path=args.model_name_or_path,
        logger=logger,
    )
    set_seed(args.seed)
    
    if args.train_or_test == 'train':
        train(args, dataset, logger)
    elif args.train_or_test == 'test':
        evaluate(args, dataset, metric_type='acc')
    else:
        train(args)
        evaluate(args, dataset, metric_type='acc')
        