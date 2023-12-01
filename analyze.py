import os
import json
import numpy as np
import pandas as pd
import datetime

from collections import defaultdict
from pathlib import Path as path


def analyze_metrics_json(log_dir, file_name, just_average=False):
    if path(file_name).suffix != '.json':
        return {}
    total_metrics = defaultdict(list)  # {metric_name: [values]}
    for dirpath, dirnames, filenames in os.walk(log_dir):
        if path(dirpath) == path(log_dir):
            continue
        for cur_file in filenames:
            if str(cur_file) == str(file_name):
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
                'tot': v,
                'cnt': len(v),
                'mean': np.mean(v),
                'variance': np.var(v),
                'std': np.std(v),
                'error': np.std(v)/np.sqrt(len(v)),
                'min': np.min(v),
                'max': np.max(v),
                'range': np.max(v)-np.min(v),
            }
    return metric_analysis


def format_analysis_value(value, format_metric=False, format_runtime=False, decimal_place=2):
    if not value:
        if format_metric:
            return '', ''
        return ''
    if format_metric:
        mean = '%.2f' % (value['mean']*100)
        error = '%.2f' % (value['error']*100)
        return mean, error
    elif format_runtime:
        return str(datetime.timedelta(seconds=int(value['mean'])))
    return f"{value['mean']:.{decimal_place}f}"


def dict_to_defaultdict(dic, default_type=dict):
    new_dic = defaultdict(default_type)
    new_dic.update(dic)
    return new_dic


def analyze_experiment_results(
    root_log_fold,
    target_csv_filename,
    hyperparam_keywords,
    hyperparam_filename,
    test_metric_filename,
    best_metric_filename,
    train_output_filename,
):
    results = []
    for log_dir in os.listdir(root_log_fold):
        log_dir = path(root_log_fold, log_dir)
        cur_result = {}

        hyper_path = path(log_dir, hyperparam_filename)
        if hyper_path.exists():
            with open(hyper_path, 'r', encoding='utf8')as f:
                hyperparams = json.load(f)
            for k, v in hyperparams.items():
                if k in hyperparam_keywords:
                    cur_result[k] = v
        
        test_analysis = analyze_metrics_json(log_dir, test_metric_filename)
        best_analysis = analyze_metrics_json(log_dir, best_metric_filename)
        train_output_analysis = analyze_metrics_json(log_dir, train_output_filename)
        test_analysis, best_analysis, train_output_analysis = map(dict_to_defaultdict, [
            test_analysis, best_analysis, train_output_analysis
        ])
        
        cur_result['acc'], cur_result['acc error'] = format_analysis_value(test_analysis['test_Acc'], format_metric=True)
        cur_result['f1'], cur_result['f1 error'] = format_analysis_value(test_analysis['test_Macro-F1'], format_metric=True)
        cur_result['epoch acc'] = format_analysis_value(best_analysis['best_epoch_Acc'])
        cur_result['epoch f1'] = format_analysis_value(best_analysis['best_epoch_Macro-F1'])
        cur_result['sample ps'] = format_analysis_value(train_output_analysis['train_samples_per_second'])
        cur_result['runtime'] = format_analysis_value(train_output_analysis['train_runtime'], format_runtime=True)
        
        results.append(cur_result)
    
    df_results = pd.DataFrame(results)
    # df_results.sort_values(by=['F1'], ascending=True)
    df_results.to_csv(target_csv_filename, encoding='utf-8')
    
    print(df_results)
    

if __name__ == '__main__':
    root_log_fold = './experiment_results/epoch_and_lr'
    target_csv_file = './experiment_results/epoch_and_lr.csv'
    hyperparam_keywords = 'log_dir version learning_rate epochs'.split()
    analyze_experiment_results(
        root_log_fold = root_log_fold,
        target_csv_filename=target_csv_file,
        hyperparam_keywords=hyperparam_keywords,
        hyperparam_filename='hyperparams.json',
        test_metric_filename='test_metric_score.json',
        best_metric_filename='best_metric_score.json',
        train_output_filename='train_output.json',
    )