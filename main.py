import os 
import torch
import pandas as pd 
import json
import shutil

from pathlib import Path as path
from transformers import TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding, set_seed

from arguments import CustomArgs
from logger import CustomLogger
from callbacks import CustomCallback
from corpusDatasets import CustomCorpusDatasets
from model import BaselineModel, CustomModel
from metrics import ComputeMetrics

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def multi_training_iteration_decorator(main_func):
    def new_main_func(args:CustomArgs):
        args.complete_path()
        args.check_path()
        
        if not args.do_train and args.do_eval:
            return main_func(args)
        # if args.training_iteration == 1:
        #     return main_func(args)
        
        main_logger = CustomLogger(
            log_dir=args.log_dir,
            logger_name='main_logger',
            print_output='True'
        )
        main_logger.log_json(dict(args), 'hyperparams.json', log_info=True)
        
        init_output_dir = args.output_dir
        init_log_dir = args.log_dir
        
        for training_iter_id in range(args.training_iteration):
            args.seed += training_iter_id
            train_fold_name = f'training_iteration_{training_iter_id}'
            args.output_dir = os.path.join(init_output_dir, train_fold_name)
            args.log_dir = os.path.join(init_log_dir, train_fold_name)
            main_func(args)
        
        # TODO: calculate average, delete ckpt
        for json_file_name in ['best_metric_score.json', 'eval_metric_score.json', 'train_output.json']:
            average_metrics = main_logger.average_metrics_json(init_log_dir, json_file_name)
            main_logger.log_json(average_metrics, json_file_name, log_info=True)
            
    return new_main_func


def train_func(
    args:CustomArgs, 
    training_args:TrainingArguments, 
    dataset:CustomCorpusDatasets, 
    model:CustomModel, 
    compute_metrics:ComputeMetrics,
    logger:CustomLogger,
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
        data_collator=dataset.data_collator,
        compute_metrics=compute_metrics,
        callbacks=[callback],
        
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
    
    # return trainer
        

def evaluate_func(
    args:CustomArgs,
    training_args:TrainingArguments,
    dataset:CustomCorpusDatasets,
    model:CustomModel,
    compute_metrics:ComputeMetrics,
    logger:CustomLogger,
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
        data_collator=dataset.data_collator,
        compute_metrics=compute_metrics,
        callbacks=[callback],
    )
    callback.trainer = trainer
    
    evaluate_output = trainer.evaluate(dataset.test_dataset)
    logger.log_json(evaluate_output, 'evaluate_output.json', log_info=True)
        
    return trainer


# @multi_training_iteration_decorator
def main(args:CustomArgs):
    if not args.do_train and not args.do_eval:
        raise Exception('neither do_train nor do_eval')
    
    args.complete_path()
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
        print_output=True,
    )

    dataset = CustomCorpusDatasets(
        file_path=args.data_path,
        data_name=args.data_name,
        model_name_or_path=args.model_name_or_path,
        cache_dir=args.cache_dir,
        logger=logger,
        label_level=args.label_level,

        label_expansion_positive=args.label_expansion_positive,
        label_expansion_negative=args.label_expansion_negative,
        data_augmentation=args.data_augmentation,
    )
    args.trainset_size, args.devset_size, args.testset_size = map(len, [
        dataset.train_dataset, dataset.dev_dataset, dataset.test_dataset
    ])

    model = CustomModel(
        model_name_or_path=args.model_name_or_path,
        cache_dir=args.cache_dir,
        num_labels=len(dataset.label_map),
    )
    
    compute_metrics = ComputeMetrics(dataset.label_map)
    
    train_evaluate_kwargs = {
        'args': args,
        'training_args': training_args,
        'model': model,
        'dataset': dataset,
        'compute_metrics': compute_metrics,
        'logger': logger,
    }
    
    if args.do_train:
        logger.log_json(dict(args), 'hyperparams.json', log_info=True)
        init_output_dir = args.output_dir
        init_log_dir = args.log_dir
        
        for training_iter_id in range(args.training_iteration):
            # seed
            args.seed += training_iter_id
            set_seed(args.seed)
            training_args.seed = args.seed
            # path
            train_fold_name = f'training_iteration_{training_iter_id}'
            args.output_dir = os.path.join(init_output_dir, train_fold_name)
            args.log_dir = os.path.join(init_log_dir, train_fold_name)
            args.check_path()
            training_args.output_dir = args.output_dir
            logger.log_dir = args.log_dir
            # model
            model.initial_model()
            
            train_func(**train_evaluate_kwargs)        
        
        # TODO: calculate average, delete ckpt
        logger.log_dir = init_log_dir
        for json_file_name in ['best_metric_score.json', 'eval_metric_score.json', 'train_output.json']:
            average_metrics = logger.average_metrics_json(init_log_dir, json_file_name)
            logger.log_json(average_metrics, json_file_name, log_info=True)
            
    elif args.do_eval:
        model_params_path = os.path.join(args.load_ckpt_dir, 'pytorch_model.bin')
        model_params = torch.load(model_params_path)
        logger.info( model.load_state_dict(model_params, strict=True) )

        evaluate_func(**train_evaluate_kwargs)
    
    # mv output_dir/run/xxx/events.xxx log_dir/
    for dirpath, dirnames, filenames in os.walk(args.output_dir):
        if 'checkpoint' in dirpath:
            continue
        if 'runs' in dirpath:
            for filename in filenames:
                if 'events' in filename:
                    cur_file = path(dirpath)/filename
                    shutil.copy(cur_file, args.log_dir)

if __name__ == '__main__':
    args = CustomArgs()
    main(args)