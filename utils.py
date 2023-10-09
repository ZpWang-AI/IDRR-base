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