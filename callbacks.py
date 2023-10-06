import os
import json
import logging
import transformers

from pathlib import Path as path
from typing import *
from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from arguments import Args


class SaveBestModelCallback(TrainerCallback):
    def __init__(self, args:Args, logger:logging.Logger=None):
        super().__init__()
        
        self.args = args
        self.trainer = None
        self.logger = logger
        
        self.best_metrics_file = path(args.output_dir)/'best_metric_score.json'
        self.eval_metrics_file = path(args.output_dir)/'eval_metric_score.csv'
        
        self.metric_names = 'acc macro_f1 tem com con exp'.split()
        self.best_metrics = {'best_'+m:-1 for m in self.metric_names}
        self.metric_map = {m:p for p, m in enumerate(self.metric_names)}

    def on_evaluate(self, args, state, control, metrics:Dict[str, float], **kwargs):
        eval_metrics = {}
        for metric_name, metric_value in metrics.items():
            best_metric_name = metric_name.replace('eval_', 'best_')
            if best_metric_name not in self.best_metrics:
                continue
            eval_metrics[metric_name] = metric_value
            
            if metric_value > self.best_metrics[best_metric_name]:
                self.best_metrics[best_metric_name] = metric_value
                
                best_model_path = os.path.join(args.output_dir, f"ckpt-{best_metric_name}")
                self.trainer.save_model(best_model_path)
                if self.logger:
                    self.logger.info(f'{best_metric_name}: {metric_value}')
                    # self.logger.info(f"New best model saved to {best_model_path}")

        eval_metrics_string = json.dumps(eval_metrics, ensure_ascii=False, indent=2)
        self.logger.info('\n'+eval_metrics_string)
        
        eval_metrics_list = sorted(eval_metrics.items(), key=lambda item:self.metric_map[item[0].replace('eval_', '')])
        with open(self.eval_metrics_file, 'a', encoding='utf8')as f:
            f.write(','.join([str(v)for k,v in eval_metrics_list])+'\n')
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        with open(self.eval_metrics_file, 'w', encoding='utf8')as f:
            f.write(','.join(self.metric_names)+'\n')
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        with open(self.best_metrics_file, 'w', encoding='utf8')as f:
            json.dump(self.best_metrics, f, ensure_ascii=False, indent=2)


class LogCallback(TrainerCallback):
    def __init__(self, args:Args, logger:logging.Logger=None) -> None:
        super().__init__()
        
        self.args = args
        self.logger = logger
    
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        args_string = json.dumps(dict(self.args), ensure_ascii=False, indent=2)
        self.logger.info('\n'+args_string)
        
        hyper_file = path(self.args.output_dir)/'hyperparams.json'
        with open(hyper_file, 'w', encoding='utf8')as f:
            f.write(args_string)
            
    # def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     print(state)
    #     print(control)
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs, **kwargs):
        # self.logger.info(str(logs))
        pass
        