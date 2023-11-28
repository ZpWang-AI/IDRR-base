import os
import shutil
import time 
import logging
import json
import numpy as np
import traceback

from pathlib import Path as path
from typing import *


def catch_and_record_error(error_file):
    with open(error_file, 'w', encoding='utf8')as f:
        error_string = traceback.format_exc()
        f.write(error_string)
        print(f"\n{'='*10} ERROR {'='*10}\n")
        print(error_string)
        print(f"\n{'='*10} ERROR {'='*10}\n")


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