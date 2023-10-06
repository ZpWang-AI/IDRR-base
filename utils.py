import os
import numpy as np
import logging
import json

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


    

