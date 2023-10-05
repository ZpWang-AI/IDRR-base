import os 
import torch
import pandas as pd 
import argparse

from pathlib import Path as path
from transformers import TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding, set_seed

from utils import (get_logger,
                   compute_metrics, 
                   SaveBestModelCallback, 
                   )
from datasets__ import CustomDatasets
from model__ import BaselineModel
from arguments import Args

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def train(args:Args, training_args:TrainingArguments, dataset, logger):
    model = BaselineModel(args.model_name_or_path)
    
    save_callback = SaveBestModelCallback(trainer=trainer, logger=logger)
    
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

    trainer.train()

def evaluate(args:Args, training_args:TrainingArguments, dataset, logger, metric_type=None):
    model = BaselineModel(args.model_name_or_path)
    
    logger.info(model.load_state_dict(torch.load(os.path.join(args.ckpt_fold, 'pytorch_model.bin')), strict=True))

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
        if metric_type:
            if metric_type in k:
                logger.info(f'{metric_type}: {v}')
                break
        else:
            logger.info(f'{k}: {v}')


def main():
    args = Args()
    args.get_from_argparse()
    set_seed(args.seed)
    
    training_args = TrainingArguments(
        output_dir = args.output_dir,
        # strategies of evaluation, logging, save
        evaluation_strategy = "steps", 
        eval_steps = args.eval_steps,
        logging_strategy='steps',
        logging_steps=5,
        save_strategy = 'no',
        # optimizer  
        optim='adamw_torch',
        lr_scheduler_type = 'linear',
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay,
        warmup_ratio = args.warmup_ratio,
        # epochs and batches 
        num_train_epochs = args.epochs, 
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
    )
    
    logger = get_logger(
        log_file=args.log_path,
        print_output=True,
    )
    
    dataset = CustomDatasets(
        file_path=args.data_path,
        data_name=args.data_name,
        label_level=args.label_level,
        model_name_or_path=args.model_name_or_path,
        logger=logger,
    )
    
    if args.train_or_test == 'train':
        train(args, training_args, dataset, logger)
    elif args.train_or_test == 'test':
        evaluate(args, training_args, dataset, logger)
    else:
        train(args)
        # evaluate(args, training_args, dataset, logger, metric_type='acc')
        evaluate(args, training_args, dataset, logger)
    

if __name__ == '__main__':
    main()