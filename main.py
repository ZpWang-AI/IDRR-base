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
from corpusDatasets import CustomDatasets
from model import BaselineModel, CustomModel
from arguments import Args

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def train(args:Args, training_args:TrainingArguments, model, dataset, logger):
    save_callback = SaveBestModelCallback(logger=logger)
    
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

def evaluate(args:Args, training_args:TrainingArguments, model, dataset, logger, metric_type=None):
    model_params_path = os.path.join(args.load_ckpt_dir, 'pytorch_model.bin')
    model_params = torch.load(model_params_path)
    logger.info(model.load_state_dict(model_params, strict=True))

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


def main(args:Args):
    args.check_path()
    set_seed(args.seed)
    
    training_args = TrainingArguments(
        output_dir = args.output_dir,
        # strategies of evaluation, logging, save
        evaluation_strategy = "steps", 
        eval_steps = args.eval_steps,
        logging_strategy='steps',
        logging_steps=args.log_steps,
        save_strategy = 'no',
        # optimizer  
        optim='adamw_torch',
        lr_scheduler_type = 'linear',
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay,
        warmup_ratio = args.warmup_ratio,
        # epochs and batches 
        num_train_epochs = args.epochs, 
        max_steps=args.max_steps,
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

    model = CustomModel(
        model_name_or_path=args.model_name_or_path,
        num_labels=len(dataset.label_map),
    )
    
    if 'train' in args.train_or_test:
        train(
            args=args,
            training_args=training_args,
            model=model,
            dataset=dataset,
            logger=logger,
        )
    if 'test' in args.train_or_test:
        evaluate(
            args=args,
            training_args=training_args,
            model=model,
            dataset=dataset,
            logger=logger,
            # metric_type='acc',
        )
    

if __name__ == '__main__':
    args = Args()
    main(args)