import os 
import json
import shutil
import torch
import pandas as pd 
import numpy as np

from typing import *
from pathlib import Path as path
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, set_seed

from arguments import CustomArgs
from logger import CustomLogger
from corpusData import CustomCorpusData
from model import CustomModel
from metrics import ComputeMetrics
from callbacks import CustomCallback
from analyze import analyze_metrics_json

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def train_func(
    args:CustomArgs, 
    training_args:TrainingArguments, 
    logger:CustomLogger,
    dataset:CustomCorpusData, 
    model:CustomModel, 
    compute_metrics:ComputeMetrics,
):
    callback = CustomCallback(
        args=args, 
        logger=logger, 
        metric_names=compute_metrics.metric_names,
    )
    
    trainer = Trainer(
        model=model, 
        args=training_args, 
        tokenizer=dataset.tokenizer, 
        compute_metrics=compute_metrics,
        callbacks=[callback],
        data_collator=dataset.data_collator,
        
        train_dataset=dataset.train_dataset,
        eval_dataset=dataset.dev_dataset, 
    )
    callback.trainer = trainer

    train_output = trainer.train().metrics
    logger.log_json(train_output, 'train_output.json', log_info=True)
    
    if args.do_eval:
        callback.evaluate_testdata = True
        
        eval_metrics = {}
        for metric_ in compute_metrics.metric_names:
            load_ckpt_dir = path(args.output_dir)/f'checkpoint_best_{metric_}'
            if load_ckpt_dir.exists():
                evaluate_output = trainer.evaluate(eval_dataset=dataset.test_dataset)
                eval_metric_name = 'eval_'+metric_
                eval_metrics[eval_metric_name] = evaluate_output[eval_metric_name]
                
        logger.log_json(eval_metrics, 'eval_metric_score.json', log_info=True)                

    return trainer, callback


def evaluate_func(
    args:CustomArgs,
    training_args:TrainingArguments,
    logger:CustomLogger,
    dataset:CustomCorpusData,
    model:CustomModel,
    compute_metrics:ComputeMetrics,
):
    callback = CustomCallback(
        args=args,
        logger=logger,
        metric_names=compute_metrics.metric_names,
        evaluate_testdata=True,
    )

    trainer = Trainer(
        model=model, 
        args=training_args, 
        tokenizer=dataset.tokenizer, 
        compute_metrics=compute_metrics,
        callbacks=[callback],
        data_collator=dataset.data_collator,
    )
    callback.trainer = trainer
    
    evaluate_output = trainer.evaluate(dataset.test_dataset)
    logger.log_json(evaluate_output, 'evaluate_output.json', log_info=True)
        
    return trainer


def main_one_iteration(args:CustomArgs, training_iter_id=0):
    if not args.do_train and not args.do_eval:
        raise Exception('neither do_train nor do_eval')
    
    # === prepare === 
        
    if 1:
        # seed
        args.seed += training_iter_id
        set_seed(args.seed)
        # path
        train_fold_name = f'training_iteration_{training_iter_id}'
        args.output_dir = os.path.join(args.output_dir, train_fold_name)
        args.log_dir = os.path.join(args.log_dir, train_fold_name)
        args.check_path()
        
        training_args = TrainingArguments(
            output_dir = args.output_dir,
            
            # strategies of evaluation, logging, save
            evaluation_strategy = "steps", 
            eval_steps = args.eval_steps,
            logging_strategy = 'steps',
            logging_steps = args.log_steps,
            save_strategy = 'no',
            # save_strategy = 'epoch',
            # save_total_limit = 1,
            
            # optimizer and lr_scheduler
            optim = 'adamw_torch',
            learning_rate = args.learning_rate,
            weight_decay = args.weight_decay,
            lr_scheduler_type = 'linear',
            warmup_ratio = args.warmup_ratio,
            
            # epochs and batches 
            num_train_epochs = args.epochs, 
            max_steps = args.max_steps,
            per_device_train_batch_size = args.train_batch_size,
            per_device_eval_batch_size = args.eval_batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
        )
        
        logger = CustomLogger(
            log_dir=args.log_dir,
            logger_name=f'iter{training_iter_id}_logger',
            print_output=True,
        )

        dataset = CustomCorpusData(
            data_path=args.data_path,
            data_name=args.data_name,
            model_name_or_path=args.model_name_or_path,
            cache_dir=args.cache_dir,
            mini_dataset=args.mini_dataset,
            
            label_level=args.label_level,
            data_augmentation=args.data_augmentation,
        )
        args.trainset_size, args.devset_size, args.testset_size = map(len, [
            dataset.train_dataset, dataset.dev_dataset, dataset.test_dataset
        ])
        
        model = CustomModel(
            model_name_or_path=args.model_name_or_path,
            num_labels=dataset.num_labels,
            cache_dir=args.cache_dir,
            loss_type=args.loss_type,
        )
        
        compute_metrics = ComputeMetrics(label_list=dataset.label_list)
        
        train_evaluate_kwargs = {
            'args': args,
            'training_args': training_args,
            'model': model,
            'dataset': dataset,
            'compute_metrics': compute_metrics,
            'logger': logger,
        }
    
    logger.log_json(dict(args), 'hyperparams.json', log_info=False)

    # === train or evaluate ===
    
    if args.do_train:
        train_func(**train_evaluate_kwargs)
            
    elif args.do_eval:
        model_params_path = os.path.join(args.load_ckpt_dir, 'pytorch_model.bin')
        model_params = torch.load(model_params_path)
        logger.info( model.load_state_dict(model_params, strict=True) )

        evaluate_func(**train_evaluate_kwargs)
    
    # mv tensorboard ckpt to log_dir
    cnt = 0
    for dirpath, dirnames, filenames in os.walk(args.output_dir):
        if 'checkpoint' in dirpath:
            continue
        if 'runs' in dirpath:
            for filename in filenames:
                if 'events' in filename:
                    cur_file = path(dirpath)/filename
                    tensorboard_dir = path(args.log_dir)/'tensorboard'/str(cnt)
                    tensorboard_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(cur_file, tensorboard_dir)
                    cnt += 1

    if not args.save_ckpt:
        shutil.rmtree(args.output_dir)


def main(args:CustomArgs, training_iter_id=-1):
    """
    params:
        args: CustomArgs
        training_iter_id: int ( set t=args.training_iteration )
            -1: auto train t iterations
            0, 1, ..., t-1: train a specific iteration
            t: calculate average of metrics
    """
    from copy import deepcopy
    
    args.complete_path()
    args.check_path()
    main_logger = CustomLogger(args.log_dir, logger_name='main_logger', print_output=True)
    # print(training_iter_id)
    
    if training_iter_id < 0 or training_iter_id == 0:    
        main_logger.log_json(dict(args), log_file_name='hyperparams.json', log_info=True)
    
    if training_iter_id < 0:
        for _training_iter_id in range(args.training_iteration):
            main_one_iteration(deepcopy(args), training_iter_id=_training_iter_id)
    else:
        main_one_iteration(args, training_iter_id=training_iter_id)
    
    if training_iter_id < 0:
        # calculate average
        for json_file_name in [
            'best_metric_score.json', 
            'eval_metric_score.json',
            'train_output.json',
        ]:
            metric_analysis = analyze_metrics_json(args.log_dir, json_file_name, just_average=True)
            main_logger.log_json(metric_analysis, json_file_name, log_info=True)


if __name__ == '__main__':
    from run import local_test_args
    args = local_test_args()
    main(args)
    
    # args = CustomArgs()
    # main(args)