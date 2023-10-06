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
        self.save_metric_path = path(args.output_dir)/'best_metric_score.json'
        
        self.best_metric = {
            'best_acc' : -1,
            'best_macro_f1' : -1,
            'best_tem' : -1,
            'best_com' : -1,
            'best_con' : -1,
            'best_exp' : -1,
        }

    def on_evaluate(self, args, state, control, metrics:Dict[str, float], **kwargs):
        eval_metric = {}
        for metric_name, metric_value in metrics.items():
            best_metric_name = metric_name.replace('eval_', 'best_')
            if best_metric_name not in self.best_metric:
                continue
            eval_metric[metric_name] = metric_value
            
            if metric_value > self.best_metric[best_metric_name]:
                self.best_metric[best_metric_name] = metric_value
                
                best_model_path = os.path.join(args.output_dir, f"ckpt-{best_metric_name}")
                self.trainer.save_model(best_model_path)
                if self.logger:
                    self.logger.info(f'{best_metric_name}: {metric_value}')
                    # self.logger.info(f"New best model saved to {best_model_path}")

        metric_score_string = json.dumps(eval_metric, ensure_ascii=False, indent=2)
        self.logger.info('\n'+metric_score_string)
        with open(self.save_metric_path, 'w', encoding='utf8')as f:
            f.write(metric_score_string)


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
            
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print(kwargs)
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs, **kwargs):
        # self.logger.info(str(logs))
        pass
        