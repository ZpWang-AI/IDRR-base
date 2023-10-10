import os
import time 
import logging
import json
import numpy as np

from pathlib import Path as path
from typing import *



import os
import shutil
import argparse


def move_folds(cur_dir, destination_dir):
    os.chdir(cur_dir)
    
    for dirpath, dirnames, filenames in os.walk('.'):
        if 'checkpoint' in dirpath:
            continue
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            destination_path = os.path.join(destination_dir, os.path.dirname(file_path))
            os.makedirs(destination_path, exist_ok=True)
            shutil.copy(file_path, destination_path)


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
    "eval_Acc": 0.6720841300191204,
    "eval_Macro-F1": 0.5813365675593336,
    "eval_Temporal": 0.38755151913046654,
    "eval_Comparison": 0.5987849573314485,
    "eval_Contingency": 0.5773303437365938,
    "eval_Expansion": 0.7616794500388254
    }
    sample_formatted_dict = format_dict_values(sample_dict, 4)
    for k, v in sample_formatted_dict.items():
        print(f'{k}: {v}')