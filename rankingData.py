import json
import numpy as np
import pandas as pd
import torch

from typing import *
from itertools import cycle
from transformers import AutoTokenizer

from corpusData import CustomCorpusData, Dataset


class RankingDataset(Dataset):
    def __init__(self, arg1, arg2, tokenizer, labels, label_rec, dataset_size) -> None:
        super().__init__()
        
        self.arg1 = arg1
        self.arg2 = arg2
        self.tokenizer = tokenizer
        self.labels = labels
        self.label_rec = label_rec
        self.dataset_size = dataset_size
        
    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)
    
    def __len__(self):
        return self.dataset_size


class RankingData():
    def __init__(
        self, 
        corpus_data:CustomCorpusData,
        rank_order_file:str,
    ):
        self.corpus_data = corpus_data
        self.label_to_id = corpus_data.label_to_id
        self.tokenizer = corpus_data.tokenizer
        
        with open(rank_order_file, 'r', encoding='utf8')as f:
            rank_order_label = json.load(f)
        self.rank_order = {
            self.label_to_id(k):list(map(self.label_to_id, rank_order_label[k]))
            for k in rank_order_label
        }
        
        self.train_dataset = self.get_dataset(corpus_data.train_df, is_train=True)
        self.dev_dataset = self.get_dataset(corpus_data.dev_df, is_train=False)
        self.test_dataset = self.get_dataset(corpus_data.test_df, is_train=False)
    
    def get_dataset(self, df, is_train=False):
        arg1_list, arg2_list = [], []
        label_ids = []
        label_rec = [[]for _ in range(self.corpus_data.num_labels)]
        for p in range(df.shape[0]):
            row = df.iloc[p]
            arg1 = row.arg1
            arg2 = row.arg2
            conn1sense1 = row.conn1sense1
            
            primary_label = self.label_to_id(conn1sense1)
            if primary_label == -1:
                continue
            
            arg1_list.append(arg1)
            arg2_list.append(arg2)
            label_ids.append(primary_label)
            label_rec[primary_label].append(p)
        
         
    
    
if __name__ == '__main__':
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
