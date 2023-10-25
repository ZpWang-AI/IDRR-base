import os
import json
import numpy as np
import pandas as pd

from collections import defaultdict
from pathlib import Path as path


def analyze_metrics_json(log_dir, file_name, just_average=False):
    total_metrics = defaultdict(list)  # {metric_name: [values]}
    for dirpath, dirnames, filenames in os.walk(log_dir):
        if path(dirpath) == path(log_dir):
            continue
        for cur_file in filenames:
            if cur_file == file_name:
                with open(path(dirpath, cur_file), 'r', encoding='utf8')as f:
                    cur_metrics = json.load(f)
                for k, v in cur_metrics.items():
                    total_metrics[k].append(v)
    if not total_metrics:
        return {}
    metric_analysis = {}
    for k, v in total_metrics.items():
        if just_average:
            metric_analysis[k] = np.mean(v)
        else:
            metric_analysis[k] = {
                'cnt': len(v),
                'mean': np.mean(v),
                'variance': np.var(v),
                'min': np.min(v),
                'max': np.max(v),
                'range': np.max(v)-np.min(v),
                'error': np.std(v)/np.sqrt(len(v)),
            }
    return metric_analysis


def analyze_experiment_evaluations(target_log_folds, to_json_file=None, to_csv_file=None):
    results = []
    hypers = []
    for log_dir in target_log_folds:
        analysis = analyze_metrics_json(log_dir, 'eval_metric_score.json', just_average=False)
        results.append(analysis)
        
        hyper_path = path(log_dir)/'hyperparams.json'
        if hyper_path.exists():
            with open(hyper_path, 'r', encoding='utf8')as f:
                hyper = json.load(f)
            hypers.append(hyper)
        
    df_results = pd.DataFrame(results)
    df_hypers = pd.DataFrame(hypers)
    
    print(df_results)
    print('='*20)
    print(df_hypers)
    
    if to_json_file:
        pass
    
    if to_csv_file:
        df_results.to_csv(to_csv_file)
    

if __name__ == '__main__':
    target_log_folds = [
        r'D:\0--data\projects\04.01-IDRR数据\IDRR-base\log_space\2023-10-16-20-51-27_local_test__train_eval',
        r'D:\0--data\projects\04.01-IDRR数据\IDRR-base\log_space\2023-10-16-20-53-04_local_test__train_eval',
        r'D:\0--data\projects\04.01-IDRR数据\IDRR-base\log_space\2023-10-16-20-54-30_local_test__train_eval',
        r'D:\0--data\projects\04.01-IDRR数据\IDRR-base\log_space\2023-10-16-20-56-42_local_test__train_eval',
        r'D:\0--data\projects\04.01-IDRR数据\IDRR-base\log_space\2023-10-16-21-02-02_local_test__train_eval',
    ]
    analyze_experiment_evaluations(target_log_folds, to_csv_file='./tmp/analysis.csv')