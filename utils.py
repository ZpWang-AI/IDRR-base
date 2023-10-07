import os
import time 
import logging
import json
import numpy as np

from pathlib import Path as path
from typing import *
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
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)
    if print_output:
        logger.addHandler(ch)
    return logger


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
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


def format_dict_values(dct, k):
    formatted_dict = {}
    for key, value in dct.items():
        if isinstance(value, float):
            formatted_value = f"{value:.{k}f}"
        else:
            formatted_value = value
        formatted_dict[key] = formatted_value
    return formatted_dict
        

if __name__ == '__main__':
    sample_dict = {
        "best_acc": 0.6686390532544378,
        "best_macro_f1": 0.5770237000192924,
        "best_tem": 0.41237113402061853,
        "best_com": 0.5945945945945947,
        "best_con": 0.591044776119403,
        "best_exp": 0.7608391608391608
    }
    sample_formatted_dict = format_dict_values(sample_dict, 4)
    for k, v in sample_formatted_dict.items():
        print(f'{k}: {v}')