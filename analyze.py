import os
import json
import numpy as np

from collections import defaultdict
from pathlib import Path as path


def analyze_metrics_json(root_dir, file_name):
    total_metrics = defaultdict(list)  # {metric_name: [values]}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for cur_file in filenames:
            if cur_file == file_name:
                with open(path(dirpath, cur_file), 'r', encoding='utf8')as f:
                    cur_metrics = json.load(f)
                for k, v in cur_metrics.items():
                    total_metrics[k].append(v)
    metric_analysis = {}
    for k, v in total_metrics.items():
        metric_analysis[k] = {
            'mean': np.mean(v),
            'variance':np.var(v),
            'range': np.max(v)-np.min(v),
        }
    return metric_analysis