import os
import json
import transformers

from pathlib import Path as path
from typing import *
from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from arguments import CustomArgs
from logger import CustomLogger


class CustomCallback(TrainerCallback):
    def __init__(self, args:CustomArgs, logger:CustomLogger, metric_names:list, evaluate_testdata=False):
        super().__init__()
        
        self.trainer:Trainer = None
        self.args = args
        self.logger = logger
        self.evaluate_testdata = evaluate_testdata
        
        # self.best_metrics_file = path(args.output_dir)/
        # self.eval_metrics_file = path(args.output_dir)/

        self.metric_names = metric_names
        self.best_metrics = {'best_'+m:-1 for m in self.metric_names}
        self.metric_map = {m:p for p, m in enumerate(self.metric_names)}
    
    # def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     self.logger.log_json(dict(self.args), 'hyperparams.json', log_info=False)

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
        # eval_metrics_string = json.dumps(dev_metrics, ensure_ascii=False, indent=2)
        # self.logger.info('\n'+eval_metrics_string)
        
        # eval_metrics_list = sorted(dev_metrics.items(), key=lambda item:self.metric_map[item[0].replace('eval_', '')])
        # with open(self.eval_metrics_file, 'a', encoding='utf8')as f:
        #     f.write(','.join([str(v)for k,v in eval_metrics_list])+'\n')
    
    # def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     with open(self.eval_metrics_file, 'w', encoding='utf8')as f:
    #         f.write(','.join(self.metric_names)+'\n')
    
    # def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     with open(self.best_metrics_file, 'w', encoding='utf8')as f:
    #         json.dump(self.best_metrics, f, ensure_ascii=False, indent=2)
            
    