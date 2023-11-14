import json
import numpy as np
import pandas as pd
import torch
import random

from typing import *
from itertools import cycle
from transformers import AutoTokenizer

from corpusData import CustomCorpusData, Dataset


class RandomSampler:
    def __init__(self, label_rec, rank_order) -> None:
        self.label_rec = label_rec
        self.rank_order = rank_order
        
    def __call__(self, first_label, first_item=None):
        if first_item is not None:
            return [first_item]+[np.random.choice(self.label_rec[p]) 
                                 for p in self.rank_order[first_label]]
        else:
            return [np.random.choice(self.label_rec[p]) 
                    for p in [first_label]+self.rank_order[first_label]]
    

class RankingDataset(Dataset):
    def __init__(self, 
                 arg1, arg2, tokenizer, labels, 
                 label_rec, rank_order, 
                 balance_class=True,
                 fixed_sampling=True,
                 dataset_size=-1
                 ) -> None:
        super().__init__()
        
        self.arg1 = arg1
        self.arg2 = arg2
        self.tokenizer = tokenizer
        self.labels = labels
        self.random_sampler = RandomSampler(label_rec, rank_order)
        self.balance_class = balance_class
        self.num_labels = len(label_rec)
        self.fixed_sampling = fixed_sampling
        if self.fixed_sampling:
            self.pids_list = [self._get_pids(p)for p in range(dataset_size)]
        self.dataset_size = len(self.labels) if dataset_size < 0 else dataset_size
            
    def _get_pids(self, index):
        if self.balance_class:
            pids = self.random_sampler(np.random.randint(self.num_labels))
        else:
            pids = self.random_sampler(self.labels[index], first_item=index)
        return pids
        
    def __getitem__(self, index):
        if self.fixed_sampling:
            pids = self.pids_list[index]
        else:
            pids = self._get_pids(index)
        model_inputs = self.tokenizer(
            [[self.arg1[p],self.arg2[p]]for p in pids],
            add_special_tokens=True, 
            truncation='longest_first', 
            max_length=256,
        )
        
        model_inputs['label'] = self.labels[index]
    
        return model_inputs
        
    
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
