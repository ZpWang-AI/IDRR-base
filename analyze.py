import os
import json
import numpy as np

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


def analyze_experiment_evaluations(target_log_folds, res_json=None, res_csv=None):
    pass


if __name__ == '__main__':
    print(analyze_metrics_json('./log_space/2023-10-16-21-25-10_local_test__train_eval/', 'eval_metric_score.json'))