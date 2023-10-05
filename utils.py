import os
import numpy as np
import logging

from pathlib import Path as path
from typing import *
from transformers import TrainerCallback, TrainerState, TrainerControl, Trainer
from sklearn.metrics import f1_score, accuracy_score


def get_logger(log_file='custom_log.log', logger_name='custom_logger', print_output=False):
    # 创建一个logger
    logger = logging.getLogger(logger_name)

    # 设置全局级别为DEBUG
    logger.setLevel(logging.DEBUG)

    # 创建一个handler，用于写入日志文件
    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)
    if print_output:
        logger.addHandler(ch)
    return logger


class SaveBestModelCallback(TrainerCallback):
    def __init__(self, trainer:Trainer=None, logger:logging.Logger=None):
        super().__init__()
        
        self.trainer = trainer
        self.logger = logger
        self.best_metric = {
            'best_acc' : None,
            'best_macro_f1' : None,
            'best_tem' : None,
            'best_com' : None,
            'best_con' : None,
            'best_exp' : None
        }

    def on_evaluate(self, args, state, control, metrics:Dict[str, float], **kwargs):
        # 假设我们使用的是准确率作为评估指标，且越高越好
        
        for metric_name, metric_value in metrics.items():
            best_metric_name = metric_name.replace('eval_', 'best_')
            if best_metric_name not in self.best_metric:
                continue
            
            # 如果这次的评估结果好于历史最优结果，那么保存模型
            if self.best_metric[best_metric_name] is None or metric_value > self.best_metric[best_metric_name]:
                self.best_metric[best_metric_name] = metric_value
                
                best_model_path = os.path.join(args.output_dir, f"ckpt-{best_metric_name}")
                self.trainer.save_model(best_model_path)
                if self.logger:
                    self.logger.info(f"New best model saved to {best_model_path}")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # import pdb; pdb.set_trace()
    predictions = np.argmax(predictions, axis=1)
    if labels.ndim == 2:
        labels = np.argmax(labels, axis=1)
    
    res = {
        'acc': accuracy_score(labels, predictions),
        'macro_f1': f1_score(labels, predictions, average='macro'),
    }
    
    for i, target_type in enumerate('tem com con exp'.split()):
        _preds = (predictions == i).astype(int)
        _labels = (labels == i).astype(int)
        res[target_type] = f1_score(_labels, _preds)
    
    return res


    

