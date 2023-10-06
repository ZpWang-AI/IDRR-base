import os 
import torch
import pandas as pd 
import json

from pathlib import Path as path
from transformers import TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding, set_seed

from utils import (
    get_logger,
    compute_metrics, 
)
from callbacks import (
    SaveBestModelCallback,
    LogCallback,
)
from corpusDatasets import CustomCorpusDatasets
from model import BaselineModel, CustomModel
from arguments import Args

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def train(args:Args, training_args:TrainingArguments, model, dataset, logger):
    save_callback = SaveBestModelCallback(args=args, logger=logger)
    log_callback = LogCallback(args=args, logger=logger)
    
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=dataset.train_dataset,
        eval_dataset=dataset.dev_dataset, 
        tokenizer=dataset.tokenizer, 
        data_collator=dataset.data_collator,
        compute_metrics=compute_metrics,
        callbacks=[save_callback, log_callback],
    )
    save_callback.trainer = trainer

    train_output = trainer.train().metrics
    with open(path(args.output_dir)/'train_output.json', 'w', encoding='utf8')as f:
        json.dump(train_output, f, ensure_ascii=False, indent=2)
        

def evaluate(args:Args, training_args:TrainingArguments, model, dataset, logger, metric_type=None):
    model_params_path = os.path.join(args.load_ckpt_dir, 'pytorch_model.bin')
    model_params = torch.load(model_params_path)
    logger.info(model.load_state_dict(model_params, strict=True))
    
    log_callback = LogCallback(args=args, logger=logger)

    trainer = Trainer(
        model=model, 
        args=training_args, 
        tokenizer=dataset.tokenizer, 
        data_collator=dataset.data_collator,
        compute_metrics=compute_metrics,
        callbacks=[log_callback],
    )

    evaluate_output = trainer.evaluate(dataset.test_dataset)

    evaluate_output_string = json.dumps(evaluate_output, ensure_ascii=False, indent=2)
    with open(path(args.output_dir)/'evaluate_output.json', 'w', encoding='utf8')as f:
        f.write(evaluate_output_string)
        
    if metric_type:
        for k, v in evaluate_output.items():
            if metric_type in k:
                logger.info(f'{metric_type}: {v}')
                break
    else:
        logger.info('\n'+evaluate_output_string)


def main(args:Args):
    args.check_path()
    set_seed(args.seed)
    
    training_args = TrainingArguments(
        output_dir = args.output_dir,
        seed=args.seed,
        # strategies of evaluation, logging, save
        evaluation_strategy = "steps", 
        eval_steps = args.eval_steps,
        logging_strategy='steps',
        logging_steps=args.log_steps,
        save_strategy = 'no',
        # optimizer and lr_scheduler
        optim='adamw_torch',
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay,
        lr_scheduler_type = 'linear',
        warmup_ratio = args.warmup_ratio,
        # epochs and batches 
        num_train_epochs = args.epochs, 
        max_steps=args.max_steps,
        per_device_train_batch_size = args.train_batch_size,
        per_device_eval_batch_size = args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    logger = get_logger(
        log_file=args.log_path,
        print_output=True,
    )
    
    dataset = CustomCorpusDatasets(
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
    
    if args.do_train:
        train(
            args=args,
            training_args=training_args,
            model=model,
            dataset=dataset,
            logger=logger,
        )
    if args.do_eval:
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