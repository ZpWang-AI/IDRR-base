import os 
import json
import shutil
import torch
import pandas as pd 
import numpy as np

from typing import *
from pathlib import Path as path
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, set_seed

from utils import catch_and_record_error
from arguments import CustomArgs, StageArgs
from logger import CustomLogger
from corpusData import CustomCorpusData
from rankingData import RankingData
from model import CustomModel
from rankingModel import RankingModel
from metrics import ComputeMetrics
from rankingMetrics import RankingMetrics
from callbacks import CustomCallback
from analyze import analyze_metrics_json

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class LogFilenameDict:
    def __init__(self) -> None:
        self.dict = {
            'hyperparams': 'hyperparams.json',
            'best': 'best_metric_score.json',
            'dev': 'dev_metric_score.jsonl',
            'test': 'test_metric_score.json',
            'blind test': 'test_blind_metric_score.json',
            'loss': 'train_loss.jsonl',
            'output': 'train_output.json',
        }
        self.stage_id = ''
        self.stage_name = ''
        
        self.visited = set() 
    
    def set_stage(self, stage_num, stage_name):
        self.stage_id = 'stage'+str(stage_num).rjust(1, '0')
        self.stage_name = stage_name
    
    def __getitem__(self, key:Literal['hyperparams','best','dev','test',
                                      'blind test','loss','output']):
        if key == 'hyperparams':
            return self.dict[key]
        else:
            res = f'{self.stage_id}.{self.stage_name}.{self.dict[key]}'
            self.visited.add(res)
            return res
    
    def keys(self):
        return self.visited

LOG_FILENAME_DICT = LogFilenameDict()

        
def train_func(
    args:CustomArgs, 
    training_args:TrainingArguments, 
    logger:CustomLogger,
    data:Union[CustomCorpusData, RankingData], 
    model:RankingModel, 
    compute_metrics:ComputeMetrics,
):
    callback = CustomCallback(
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
    final_state_fold = path(training_args.output_dir)/'final'
    trainer.save_model(final_state_fold)
    
    # do test 
    callback.evaluate_testdata = True
    
    test_metrics = {}
    for metric_ in compute_metrics.metric_names:
        load_ckpt_dir = path(training_args.output_dir)/f'checkpoint_best_{metric_}'
        if load_ckpt_dir.exists():
            model.load_state_dict(torch.load(load_ckpt_dir/'pytorch_model.bin'))
            evaluate_output = trainer.evaluate(eval_dataset=data.test_dataset)
            test_metrics['test_'+metric_] = evaluate_output['eval_'+metric_]
            
    logger.log_json(test_metrics, LOG_FILENAME_DICT['test'], log_info=True)                
    model.load_state_dict(torch.load(final_state_fold/'pytorch_model.bin'))

    return trainer, callback


def main_one_iteration(args:CustomArgs, data:CustomCorpusData, training_iter_id=0):
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
        
        def prepare_training_args(output_dir, stage_args: StageArgs):
            return TrainingArguments(
                output_dir=output_dir,
                
                # strategies of evaluation, logging, save
                evaluation_strategy="steps", 
                eval_steps=stage_args.eval_steps,
                logging_strategy='steps',
                logging_steps=stage_args.log_steps,
                save_strategy='no',
                # save_strategy='epoch',
                # save_total_limit=1,
                
                # optimizer and lr_scheduler
                optim='adamw_torch',
                learning_rate=stage_args.learning_rate,
                weight_decay=stage_args.weight_decay,
                lr_scheduler_type='linear',
                warmup_ratio=stage_args.warmup_ratio,
                
                # epochs and batches 
                num_train_epochs=stage_args.epochs, 
                per_device_train_batch_size=stage_args.train_batch_size,
                per_device_eval_batch_size=stage_args.eval_batch_size,
                gradient_accumulation_steps=stage_args.gradient_accumulation_steps,
            )
            
        logger = CustomLogger(
            log_dir=args.log_dir,
            logger_name=f'{args.cur_time}_iter{training_iter_id}_logger',
            print_output=True,
        )
        
        for stage in args.training_stages:
            if 'rank' in stage.stage_name:
                ranking_data = RankingData(
                    corpus_data=data,
                    rank_order_file=args.rank_order_file,
                    data_sampler=args.rank_data_sampler,
                    balance_batch=args.rank_balance_batch,
                    balance_class=args.rank_balance_class,
                    fixed_sampling=args.rank_fixed_sampling,
                    dataset_size_multiplier=args.rank_dataset_size_multiplier,
                )
                break
        else:
            ranking_data:RankingData = None
        
        model = RankingModel(
            model_name_or_path=args.model_name_or_path,
            label_list=data.label_list,
            cache_dir=args.cache_dir,
            loss_type=args.loss_type,
            rank_loss_type=args.rank_loss_type,
        )
        
        compute_metrics = ComputeMetrics(label_list=data.label_list)
        ranking_metrics = RankingMetrics(num_labels=data.num_labels)
        
        train_evaluate_kwargs = {
            'args': args,
            'training_args': None,  # switch
            'model': model,
            'data': data,  # switch
            'compute_metrics': compute_metrics,  # switch
            'logger': logger,
        }
    
    logger.log_json(dict(args), LOG_FILENAME_DICT['hyperparams'], log_info=False)

    # === train ===
    
    def stage_rank(training_args:TrainingArguments):
        train_evaluate_kwargs['training_args'] = training_args
        train_evaluate_kwargs['data'] = ranking_data
        train_evaluate_kwargs['compute_metrics'] = ranking_metrics
        model.forward_fn = 'rank'
        train_func(**train_evaluate_kwargs)
    
    def stage_ft(training_args:TrainingArguments):
        train_evaluate_kwargs['training_args'] = training_args
        train_evaluate_kwargs['data'] = data
        train_evaluate_kwargs['compute_metrics'] = compute_metrics
        model.forward_fn = 'ft'
        train_func(**train_evaluate_kwargs)
    
    stage_dict = {
        'rank': stage_rank,
        'ft': stage_ft,
    }
    for stage_num, stage in enumerate(args.training_stages):
        LOG_FILENAME_DICT.set_stage(stage_num, stage.stage_name)
        training_args = prepare_training_args(
            output_dir=path(args.output_dir)/LOG_FILENAME_DICT.stage_id,
            stage_args=stage,
        )
        stage_dict[stage.stage_name](training_args)
    
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
    set_seed(args.seed)
    
    data = CustomCorpusData(
        data_path=args.data_path,
        data_name=args.data_name,
        model_name_or_path=args.model_name_or_path,
        cache_dir=args.cache_dir,
        label_level=args.label_level,
        secondary_label_weight=args.secondary_label_weight,
        mini_dataset=args.mini_dataset,
        data_augmentation_secondary_label=args.data_augmentation_secondary_label,
        data_augmentation_connective_arg2=args.data_augmentation_connective_arg2,
    )
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
                main_one_iteration(deepcopy(args), 
                                   data=data, 
                                   training_iter_id=_training_iter_id)
            if not args.save_ckpt:
                shutil.rmtree(args.output_dir)
        else:
            main_one_iteration(deepcopy(args), 
                               data=data, 
                               training_iter_id=training_iter_id)
    except Exception as e:
        error_file = main_logger.log_dir/'error.out'
        catch_and_record_error(error_file)
        exit(1)
    
    if training_iter_id < 0 or training_iter_id == args.training_iteration:
        # calculate average
        for json_file_name in LOG_FILENAME_DICT.keys():
            if json_file_name == LOG_FILENAME_DICT['hyperparams']:
                continue
            metric_analysis = analyze_metrics_json(args.log_dir, json_file_name, just_average=True)
            if metric_analysis:
                main_logger.log_json(metric_analysis, json_file_name, log_info=True)


if __name__ == '__main__':
    from run import local_test_args
    args = local_test_args()
    main(args)
    
    # args = CustomArgs()
    # main(args)