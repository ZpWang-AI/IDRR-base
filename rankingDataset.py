import numpy as np
import pandas as pd

from typing import *
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, arg1, arg2, tokenizer, labels) -> None:
        super().__init__()
        
        self.arg1 = arg1
        self.arg2 = arg2
        self.tokenizer = tokenizer
        self.labels = labels
        
    def __getitem__(self, index) -> Any:
        model_inputs = self.tokenizer(
            self.arg1[index], 
            self.arg2[index], 
            add_special_tokens=True, 
            truncation='longest_first', 
            max_length=256,
        )
        
        model_inputs['label'] = self.labels[index]
    
        return model_inputs
    
    def __len__(self):
        return len(self.arg1)
    
    
class CustomDataCollator:
    def __init__(self, tokenizer, rank_order, train_dataset, dev_dataset, test_dataset) -> None:
        self.tokenizer = tokenizer
        self.rank_order = rank_order
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        
    # def 
    
    def __call__(self, features):
        sample_id, dataset_id = features
        if dataset_id == 0:
            pass
        elif dataset_id == 1:
            pass
        else:
            pass
        exit()
        return features



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
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.sep_t = tokenizer.sep_token
        self.cls_t = tokenizer.cls_token
        self.tokenizer = tokenizer 
        
        if label_level == 'level1':
            self.label_list = 'Temporal Comparison Contingency Expansion'.split()
        else:
            raise ValueError('wrong label_level')
        self.num_labels = len(self.label_list)
        self.label_map = {label:p for p, label in enumerate(self.label_list)}

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

        self.label_level = label_level
        self.data_augmentation = data_augmentation
        
        self.train_dataset = [(p, 0) for p in range(train_df.shape[0])]
        self.dev_dataset = [(p, 1) for p in range(dev_df.shape[0])]
        self.test_dataset = [(p, 2) for p in range(test_df.shape[0])]
        
        self.data_collator = CustomDataCollator(tokenizer=tokenizer)
    
    def label_to_id(self, sense):
        if self.label_level == 'level1':
            label_id = self.label_map[sense.split('.')[0]]
        else:
            raise ValueError('wrong label_level')
        return label_id
    
    def get_dataset(self, df, is_train):
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
        
        return CustomDataset(
            arg1=arg1_list,
            arg2=arg2_list,
            tokenizer=self.tokenizer,
            labels=labels,
        )
                        
    def data_augmentation_df(self, df:pd.DataFrame):
        # 'Relation', 'Section', 
        # 'Arg1_RawText', 'Arg2_RawText', 
        # 'Conn1', 'Conn2',
        # 'ConnHeadSemClass1', 'ConnHeadSemClass2',
        # 'Conn2SemClass1', 'Conn2SemClass2'
        df2 = df.copy()
        df2['Arg2_RawText'] = df2['Conn1']+df2['Arg2_RawText']
        df3 = df.copy()
        df3.dropna(subset=['Conn2'], inplace=True)
        df3['Arg2_RawText'] = df3['Conn2']+df3['Arg2_RawText']
        return pd.concat([df, df2, df3], ignore_index=True)
    