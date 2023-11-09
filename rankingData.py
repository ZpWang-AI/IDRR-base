import json
import numpy as np
import pandas as pd
import torch

from typing import *
from itertools import cycle
from transformers import AutoTokenizer

from corpusData import CustomCorpusData


class CustomData:
    def __init__(self, arg1, arg2, tokenizer, labels) -> None:
        super().__init__()
        
        self.arg1 = arg1
        self.arg2 = arg2
        self.tokenizer = tokenizer
        self.labels = np.array(labels)
        
    def __getitem__(self, indices) -> Any:
        arg_pair_list = [(self.arg1[p], self.arg2[p])for p in indices]
        model_inputs = self.tokenizer(
            arg_pair_list,
            add_special_tokens=True, 
            truncation='longest_first', 
            max_length=256,
            padding=True,
            return_tensors='pt',
        )
        
        model_inputs['labels'] = torch.IntTensor(self.labels[indices])
    
        return model_inputs
    
    def __len__(self):
        return len(self.arg1)
    
    
class CustomDataCollator:
    def __init__(
        self, 
        tokenizer,
        num_labels,
        rank_order,
        train_data:CustomData,
        dev_data:CustomData, 
        test_data:CustomData,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_labels = num_labels
        self.rank_order = rank_order
        
        self.data = [train_data, dev_data, test_data]
        self.id_iter = []
        for data_id, data in enumerate(self.data):
            self.id_iter.append([])
            for label_id in range(num_labels):
                cur_iter = [p for p in range(len(data)) if data.labels[p] == label_id]
                if not cur_iter:
                    cur_iter.append(0)
                    print(f'>> warning: not enough sample in dataset {data_id} about label {label_id}')
                
                np.random.shuffle(cur_iter)
                self.id_iter[data_id].append(cycle(cur_iter))
                
    def __call__(self, features):
        assert len(features) == 1, f'{len(features)}\n{features}'
        sample_id, data_id = features[0]
        data = self.data[data_id]
        label_id = data.labels[sample_id]
        sample_id_list = [sample_id]
        for p in self.rank_order[label_id]:
            sample_id_list.append(next(self.id_iter[data_id][p]))
        return data[sample_id_list]


class RankingDataset():
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
        
        self.train_dataset = [(p, 0) for p in range(corpus_data.train_df.shape[0])]
        self.dev_dataset = [(p, 1) for p in range(corpus_data.dev_df.shape[0])]
        self.test_dataset = [(p, 2) for p in range(corpus_data.test_df.shape[0])]
        
        self.data_collator = CustomDataCollator(
            tokenizer=self.tokenizer,
            num_labels=corpus_data.num_labels,
            rank_order=self.rank_order,
            train_data=self.get_data(corpus_data.train_df),
            dev_data=self.get_data(corpus_data.dev_df),
            test_data=self.get_data(corpus_data.test_df),
        )
    
    def get_data(self, df):
        # 'Relation', 'Section', 
        # 'Arg1_RawText', 'Arg2_RawText', 
        # 'Conn1', 'Conn2',
        # 'ConnHeadSemClass1', 'ConnHeadSemClass2',
        # 'Conn2SemClass1', 'Conn2SemClass2'
        arg1_list, arg2_list = [], []
        label_ids = []
        additional_label_ids = []
        for p in range(df.shape[0]):
            row = df.iloc[p]
            arg1 = row.Arg1_RawText
            arg2 = row.Arg2_RawText
            conn1 = row.Conn1
            conn2 = row.Conn2
            conn1sem1 = row.ConnHeadSemClass1
            conn1sem2 = row.ConnHeadSemClass2
            conn2sem1 = row.Conn2SemClass1
            conn2sem2 = row.Conn2SemClass2
            
            arg1_list.append(arg1)
            arg2_list.append(arg2)
            
            label_ids.append(self.label_to_id(conn1sem1))
            # cur_adds = [self.label_to_id(sense) 
            #             for sense in [conn1sem2, conn2sem1, conn2sem2]
            #             if not pd.isna(sense)]
            # additional_label_ids.append(cur_adds)
            
        labels = label_ids
        
        return CustomData(
            arg1=arg1_list,
            arg2=arg2_list,
            tokenizer=self.corpus_data.tokenizer,
            labels=labels,
        )
    
    
if __name__ == '__main__':
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    sample_corpus_data = CustomCorpusDataset(
        './CorpusData/PDTB2/pdtb2.csv',
        cache_dir='./plm_cache/',
        mini_dataset=False,
        data_augmentation=False,
    )
    sample_rank_dataset = RankingDataset(
        corpus_data=sample_corpus_data,
        rank_order_file='./rank_order/rank_order1.json'
    )
    sample_ids = [next(sample_rank_dataset.data_collator.id_iter[0][0]) for _ in range(10)]
    print(', '.join(map(str, sample_ids)))
    for p in sample_rank_dataset.train_dataset:
        sample_intput = sample_rank_dataset.data_collator([p])
        print(sample_intput)
        print(sample_intput['input_ids'].shape, sample_intput['attention_mask'].shape)
        break