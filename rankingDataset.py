import numpy as np
import pandas as pd

from typing import *
from itertools import cycle
from transformers import AutoTokenizer


class CustomData:
    def __init__(self, arg1, arg2, tokenizer, labels) -> None:
        super().__init__()
        
        self.arg1 = arg1
        self.arg2 = arg2
        self.tokenizer = tokenizer
        self.labels = labels
        
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
        self.id_iter = [[
                cycle([p for p in range(len(data)) if data.labels[p] == label_id])
                for label_id in range(num_labels)
            ]for data in self.data
        ]
        
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
        file_path,
        data_name='pdtb2',
        model_name_or_path='roberta-base',
        cache_dir='',
        label_level='level1',
        
        rank_order=None,
        mini_dataset=False,
        data_augmentation=False,
    ):
        assert data_name in ['pdtb2', 'pdtb3', 'conll']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        
        if label_level == 'level1':
            self.label_list = 'Temporal Comparison Contingency Expansion'.split()
        else:
            raise ValueError('wrong label_level')
        self.num_labels = len(self.label_list)
        self.label_map = {label:p for p, label in enumerate(self.label_list)}
        self.label_level = label_level

        df = pd.read_csv(
            file_path, 
            usecols=[
                'Relation', 'Section', 
                'Arg1_RawText', 'Arg2_RawText', 
                'Conn1', 'Conn2',
                'ConnHeadSemClass1', 'ConnHeadSemClass2',
                'Conn2SemClass1', 'Conn2SemClass2'],
            low_memory=False,
        )
        df = df[df['Relation'] == 'Implicit']
        train_df = df[~df['Section'].isin([0, 1, 21, 22, 23, 24])]
        dev_df = df[df['Section'].isin([0, 1])]
        test_df = df[df['Section'].isin([21, 22])]
            
        if data_augmentation:
            train_df = self.data_augmentation_df(train_df)
            
        if mini_dataset:
            train_df = train_df.iloc[:32]
            dev_df = dev_df.iloc[:16]
            test_df = test_df.iloc[:16]

        self.rank_order = {
            self.label_to_id(k):list(map(self.label_to_id, rank_order[k]))
            for k in rank_order
        }
        self.data_augmentation = data_augmentation
        
        self.train_dataset = [(p, 0) for p in range(train_df.shape[0])]
        self.dev_dataset = [(p, 1) for p in range(dev_df.shape[0])]
        self.test_dataset = [(p, 2) for p in range(test_df.shape[0])]
        
        self.data_collator = CustomDataCollator(
            tokenizer=self.tokenizer,
            num_labels=self.num_labels,
            rank_order=self.rank_order,
            train_data=self.get_data(train_df),
            dev_data=self.get_data(dev_df),
            test_data=self.get_data(test_df),
        )
    
    def label_to_id(self, sense):
        if self.label_level == 'level1':
            label_id = self.label_map[sense.split('.')[0]]
        else:
            raise ValueError('wrong label_level')
        return label_id
    
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
            cur_adds = [self.label_to_id(sense) 
                        for sense in [conn1sem2, conn2sem1, conn2sem2]
                        if not pd.isna(sense)]
            additional_label_ids.append(cur_adds)
            
        labels = label_ids
        
        return CustomData(
            arg1=arg1_list,
            arg2=arg2_list,
            tokenizer=self.tokenizer,
            labels=labels,
        )
                        
    def data_augmentation_df(self, df:pd.DataFrame):
        df2 = df.copy()
        df2['Arg2_RawText'] = df2['Conn1']+df2['Arg2_RawText']
        df3 = df.copy()
        df3.dropna(subset=['Conn2'], inplace=True)
        df3['Arg2_RawText'] = df3['Conn2']+df3['Arg2_RawText']
        df3['ConnHeadSemClass1'], df3['ConnHeadSemClass2'], df3['Conn2SemClass1'], df3['Conn2SemClass2'] = (
            df3['Conn2SemClass1'], df3['Conn2SemClass2'], df3['ConnHeadSemClass1'], df3['ConnHeadSemClass2']
        )
        return pd.concat([df, df2, df3], ignore_index=True)
    
    
if __name__ == '__main__':
    import os
    import json
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    with open('./rank_order/rank_order1.json', 'r', encoding='utf8')as f:
        rank_order = json.load(f)
    sample_dataset = RankingDataset(
        r'D:\0--data\projects\04.01-IDRR数据\IDRR-base\CorpusData\PDTB2\pdtb2.csv',
        data_name='pdtb2',
        model_name_or_path='roberta-base',
        cache_dir='./plm_cache/',
        label_level='level1',
        rank_order=rank_order,
        mini_dataset=False,
        data_augmentation=True,
    )
    for p in sample_dataset.train_dataset:
        sample_intput = sample_dataset.data_collator([p])
        print(sample_intput)
        print(sample_intput['input_ids'].shape, sample_intput['attention_mask'].shape)
        break