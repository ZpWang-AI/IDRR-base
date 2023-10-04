import os
import numpy as np
import logging

from pathlib import Path as path
from typing import *
from transformers import TrainerCallback, TrainerState, TrainerControl
from sklearn.metrics import f1_score, accuracy_score


def create_file_or_fold(file_or_fold_path):
    file_or_fold_path = path(file_or_fold_path)
    if file_or_fold_path.is_file():
        file_or_fold_path.parent.mkdir(parents=True, exist_ok=True)
        file_or_fold_path.touch()
    else:
        file_or_fold_path.mkdir(parents=True, exist_ok=True)


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
    def __init__(self):
        super().__init__()
        
        self.trainer = None
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
            if self.best_metric[best_metric_name] is None or self.best_metric[best_metric_name]:
                self.best_metric[best_metric_name] = metric_value
                
                best_model_path = os.path.join(args.output_dir, f"checkpoint-best-{metric_name.replace('eval', 'dev')}")
                self.trainer.save_model(best_model_path)
                print(f"New best model saved to {best_model_path}")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # import pdb; pdb.set_trace()
    predictions = np.argmax(predictions, axis=1)
    metrics = [None]*6
    metrics[5] = accuracy_score(labels, predictions)
    metrics[4] = f1_score(labels, predictions, average='macro')
    
    for i in range(4):
        _preds = (predictions == i).astype(int)
        _labels = (labels == i).astype(int)
        metrics[i] = f1_score(_labels, _preds)

    return {
        'acc': metrics[5],
        'macro_f1': metrics[4],
        'tem': metrics[0],
        'com': metrics[1],
        'con': metrics[2],
        'exp': metrics[3]
    }


    

