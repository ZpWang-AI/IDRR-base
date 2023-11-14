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

LOG_FILENAME_DICT = {
    'hyperparams': 'hyperparams.json',
    'best': 'best_metric_score.json',
    'dev': 'dev_metric_score.jsonl',
    'test': 'test_metric_score.json',
    'blind test': 'test_blind_metric_score.json',
    'loss': 'train_loss.jsonl',
    'output': 'train_output.json',
    
    'evaluate': 'evaluate_output.json'
}


def train_func(
    args:CustomArgs, 
    training_args:TrainingArguments, 
    logger:CustomLogger,
    data:CustomCorpusData, 
    model:CustomModel, 
    compute_metrics:ComputeMetrics,
):
    callback = CustomCallback(
        args=args, 
        logger=logger, 
        metric_names=compute_metrics.metric_names,
    )
    callback.best_metric_file_name = LOG_FILENAME_DICT['best']
    callback.dev_metric_file_name = LOG_FILENAME_DICT['dev']
    callback.train_loss_file_name = LOG_FILENAME_DICT['loss']
    
    trainer = Trainer(
        model=model, 
        args=training_args, 
        tokenizer=data.tokenizer, 
        compute_metrics=compute_metrics,
        callbacks=[callback],
        data_collator=data.data_collator,
        
        train_dataset=data.train_dataset,
        eval_dataset=data.dev_dataset, 
    )
    callback.trainer = trainer

    train_output = trainer.train().metrics
    logger.log_json(train_output, LOG_FILENAME_DICT['output'], log_info=True)
    
    if args.do_eval:
        callback.evaluate_testdata = True
        
        test_metrics = {}
        for metric_ in compute_metrics.metric_names:
            load_ckpt_dir = path(args.output_dir)/f'checkpoint_best_{metric_}'
            if load_ckpt_dir.exists():
                evaluate_output = trainer.evaluate(eval_dataset=data.test_dataset)
                test_metrics['test_'+metric_] = evaluate_output['eval_'+metric_]
                
        logger.log_json(test_metrics, LOG_FILENAME_DICT['test'], log_info=True)                

        if args.data_name == 'conll':
            test_metrics = {}
            for metric_ in compute_metrics.metric_names:
                load_ckpt_dir = path(args.output_dir)/f'checkpoint_best_{metric_}'
                if load_ckpt_dir.exists():
                    evaluate_output = trainer.evaluate(eval_dataset=data.blind_test_dataset)
                    test_metrics['test_'+metric_] = evaluate_output['eval_'+metric_]
                    
            logger.log_json(test_metrics, LOG_FILENAME_DICT['blind test'], log_info=True)    
                

    return trainer, callback


def evaluate_func(
    args:CustomArgs,
    training_args:TrainingArguments,
    logger:CustomLogger,
    data:CustomCorpusData,
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
        tokenizer=data.tokenizer, 
        compute_metrics=compute_metrics,
        callbacks=[callback],
        data_collator=data.data_collator,
    )
    callback.trainer = trainer
    
    evaluate_output = trainer.evaluate(data.test_dataset)
    logger.log_json(evaluate_output, LOG_FILENAME_DICT['evaluate'], log_info=True)
        
    return trainer


def main_one_iteration(args:CustomArgs, data:CustomCorpusData, training_iter_id=0):
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
            logger_name=f'{args.cur_time}_iter{training_iter_id}_logger',
            print_output=True,
        )
        
        model = CustomModel(
            model_name_or_path=args.model_name_or_path,
            num_labels=data.num_labels,
            cache_dir=args.cache_dir,
            loss_type=args.loss_type,
        )
        
        compute_metrics = ComputeMetrics(label_list=data.label_list)
        
        train_evaluate_kwargs = {
            'args': args,
            'training_args': training_args,
            'model': model,
            'data': data,
            'compute_metrics': compute_metrics,
            'logger': logger,
        }
    
    logger.log_json(dict(args), LOG_FILENAME_DICT['hyperparams'], log_info=False)

    # === train or evaluate ===
    
    if args.do_train:
        train_func(**train_evaluate_kwargs)
            
    elif args.do_eval:
        model_params_path = os.path.join(args.load_ckpt_dir, 'pytorch_model.bin')
        model_params = torch.load(model_params_path)
        logger.info( model.load_state_dict(model_params, strict=True) )

        evaluate_func(**train_evaluate_kwargs)
    
    # # mv tensorboard ckpt to log_dir
    # cnt = 0
    # for dirpath, dirnames, filenames in os.walk(args.output_dir):
    #     if 'checkpoint' in dirpath:
    #         continue
    #     if 'runs' in dirpath:
    #         for filename in filenames:
    #             if 'events' in filename:
    #                 cur_file = path(dirpath)/filename
    #                 tensorboard_dir = path(args.log_dir)/'tensorboard'/str(cnt)
    #                 tensorboard_dir.mkdir(parents=True, exist_ok=True)
    #                 shutil.copy(cur_file, tensorboard_dir)
    #                 cnt += 1

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
    
    data = CustomCorpusData(**dict(args))
    args.trainset_size, args.devset_size, args.testset_size = map(len, [
        data.train_dataset, data.dev_dataset, data.test_dataset
    ])
    args.recalculate_eval_log_steps()
    
    main_logger = CustomLogger(args.log_dir, logger_name=f'{args.cur_time}_main_logger', print_output=True)
    if training_iter_id < 0 or training_iter_id == 0:    
        main_logger.log_json(dict(args), log_file_name=LOG_FILENAME_DICT['hyperparams'], log_info=True)
    
    try:
        if training_iter_id < 0:
            for _training_iter_id in range(args.training_iteration):
                main_one_iteration(deepcopy(args), data=data, training_iter_id=_training_iter_id)
            if not args.save_ckpt:
                shutil.rmtree(args.output_dir)
        else:
            main_one_iteration(deepcopy(args), data=data, training_iter_id=training_iter_id)
    except Exception as e:
        import traceback
        
        error_file = main_logger.log_dir/'error.out'
        with open(error_file, 'w', encoding='utf8')as f:
            error_string = traceback.format_exc()
            f.write(error_string)
            print('\n', '='*20, '\n')
            print(error_string)
        exit(1)
    
    if training_iter_id < 0 or training_iter_id == args.training_iteration:
        # calculate average
        for json_file_name in [
            LOG_FILENAME_DICT['best'],
            LOG_FILENAME_DICT['test'],
            LOG_FILENAME_DICT['blind test'],
            LOG_FILENAME_DICT['output'],
        ]:
            metric_analysis = analyze_metrics_json(args.log_dir, json_file_name, just_average=True)
            if metric_analysis:
                main_logger.log_json(metric_analysis, json_file_name, log_info=True)


if __name__ == '__main__':
    from run import local_test_args
    args = local_test_args()
    main(args)
    
    # args = CustomArgs()
    # main(args)