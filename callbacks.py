import os
import json
import transformers

from pathlib import Path as path
from typing import *
from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from arguments import CustomArgs
from logger import CustomLogger
from corpusDatasets import CustomCorpusDatasets


class CustomCallback(TrainerCallback):
    def __init__(self, args:CustomArgs, logger:CustomLogger, metric_names:list, evaluate_testdata=False):
        super().__init__()
        
        self.trainer:Trainer = None
        self.dataset:CustomCorpusDatasets = None
        self.args = args
        self.logger = logger
        self.evaluate_testdata = evaluate_testdata
        
        # self.best_metrics_file = path(args.output_dir)/
        # self.eval_metrics_file = path(args.output_dir)/

        self.metric_names = metric_names
        self.best_metrics = {'best_'+m:-1 for m in self.metric_names}
        self.metric_map = {m:p for p, m in enumerate(self.metric_names)}

    def on_evaluate(self, args, state, control, metrics:Dict[str, float], **kwargs):
        if self.evaluate_testdata:
            return
        
        dev_metrics = {}
        for metric_name, metric_value in metrics.items():
            best_metric_name = metric_name.replace('eval_', 'best_')
            if best_metric_name not in self.best_metrics:
                continue
            dev_metrics[metric_name.replace('eval_', 'dev_')] = metric_value
            
            if metric_value > self.best_metrics[best_metric_name]:
                self.best_metrics[best_metric_name] = metric_value
                
                best_model_path = path(args.output_dir)/f'checkpoint_{best_metric_name}'
                self.trainer.save_model(best_model_path)
                if self.logger:
                    self.logger.info(f'{best_metric_name}: {metric_value}')
                    # self.logger.info(f"New best model saved to {best_model_path}")

        self.logger.log_json(self.best_metrics, 'best_metric_score.json', log_info=False)
        self.logger.log_jsonl(dev_metrics, 'dev_metric_score.jsonl', log_info=True)
        
    # def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     print('\n===== step begin =====\n')
    #     print(f'= args: \n{args}\n')
    #     print(f'= state: \n{state}\n')
    #     print(f'= control: \n{control}\n')
    #     print(f'= kwargs: \n{kwargs}\n')
    #     print('\n===== step begin =====\n')
    #     return super().on_step_begin(args, state, control, **kwargs)
    
    # def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     print('\n===== step end =====\n')
    #     print(f'= args: \n{args}\n')
    #     print(f'= state: \n{state}\n')
    #     print(f'= control: \n{control}\n')
    #     print(f'= kwargs: \n{kwargs}\n')
    #     print('\n===== step end =====\n')
    #     return super().on_step_end(args, state, control, **kwargs)
            
    