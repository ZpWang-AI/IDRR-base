import numpy as np
import pandas as pd

from typing import *
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(
        self,
        df:pd.DataFrame, 
        tokenizer, 
        label_map,
        label_level,
        le_pos,
        le_neg,
    ) -> None:
        super().__init__()
        
        self.df = df
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.label_level = label_level
        self.le_pos = le_pos
        self.le_neg = le_neg
    
    def label_to_id(self, sense):
        if self.label_level == 'level1':
            label_id = self.label_map[sense.split('.')[0]]
        else:
            raise ValueError('wrong label_level')
        return label_id
    
    def generate_label(self, sense_list:List[str]):
        label = np.zeros(len(self.label_map))

        sense1 = sense_list[0]
        label[self.label_to_id(sense1)] += 1
        
        if self.le_pos:
            for sense_ in sense_list[1:]:
                if not pd.isna(sense_):
                    label[self.label_to_id(sense_)] += self.le_pos
        if self.le_neg:
            label -= self.le_neg
            
        return label
        
    def __getitem__(self, index):
        row = self.df.iloc[index]
        arg1 = row.Arg1_RawText
        arg2 = row.Arg2_RawText
        conn1sense1 = row.ConnHeadSemClass1
        conn1sense2 = row.ConnHeadSemClass2
        conn2sense1 = row.Conn2SemClass1
        conn2sense2 = row.Conn2SemClass2
        model_inputs = self.tokenizer(
            arg1, 
            arg2, 
            add_special_tokens=True, 
            truncation='longest_first', 
            max_length=256,
        )
        
        model_inputs['label'] = self.generate_label(
            [conn1sense1, conn1sense2, conn2sense1, conn2sense2]
        )
    
        return model_inputs
        
    def __len__(self):
        return self.df.shape[0]


class CustomCorpusDatasets():
    def __init__(
        self, 
        file_path,
        data_name='pdtb2',
        model_name_or_path='roberta-base',
        cache_dir='',
        logger=None,
        
        label_level='level1',
        label_expansion_positive=0,
        label_expansion_negative=0,
        data_augmentation=False,
    ):
        assert data_name in ['pdtb2', 'pdtb3', 'conll']
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.sep_t = tokenizer.sep_token
        self.cls_t = tokenizer.cls_token
        self.tokenizer = tokenizer 
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        if label_level == 'level1':
            label_list = 'Temporal Comparison Contingency Expansion'.split()
        else:
            raise ValueError('wrong label_level')
        self.label_map = {label:p for p, label in enumerate(label_list)}

        df = pd.read_csv(file_path, usecols=['Relation', 'Section', 
                                             'Arg1_RawText', 'Arg2_RawText', 
                                             'Conn1', 'Conn2',
                                             'ConnHeadSemClass1', 'ConnHeadSemClass2',
                                             'Conn2SemClass1', 'Conn2SemClass2'])
        df = df[df['Relation'] == 'Implicit']
        
        train_df = df[~df['Section'].isin([0, 1, 21, 22, 23, 24])]
        dev_df = df[df['Section'].isin([0, 1])]
        test_df = df[df['Section'].isin([21, 22])]
        if data_augmentation:
            train_df = self.data_augmentation_df(train_df)

        self.label_level = label_level
        self.le_pos = label_expansion_positive
        self.le_neg = label_expansion_negative
        self.data_augmentation = data_augmentation
        
        dataset_kwargs = {
            'tokenizer': tokenizer,
            'label_map': self.label_map,
            'label_level': label_level,
            'le_pos': label_expansion_positive,
            'le_neg': label_expansion_negative,
        }
        self.train_dataset = CustomDataset(
            df=train_df, 
            **dataset_kwargs,
        )
        self.dev_dataset = CustomDataset(
            df=dev_df, 
            **dataset_kwargs,
        )
        self.test_dataset = CustomDataset(
            df=test_df,
            **dataset_kwargs,
        )
        
        if logger is not None:
            logger.info('-' * 30)
            logger.info(f'Trainset Size: {len(self.train_dataset)}')
            logger.info(f'Devset Size: {len(self.dev_dataset)}')
            logger.info(f'Testset Size: {len(self.test_dataset)}')
            logger.info('-' * 30)
            
    def data_augmentation_df(self, df:pd.DataFrame):
        # 'Relation', 'Section', 
        # 'Arg1_RawText', 'Arg2_RawText', 
        # 'Conn1', 'Conn2',
        # 'ConnHeadSemClass1', 'ConnHeadSemClass2',
        # 'Conn2SemClass1', 'Conn2SemClass2'
        df2 = df.copy()
        df2['Arg2_RawText'] = df['Conn1']+df['Arg2_RawText']
        return pd.concat([df, df2], ignore_index=True)
    
    
if __name__ == '__main__':
    import os
    from logger import CustomLogger
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    sample_dataset = CustomCorpusDatasets(
        r'D:\0--data\projects\04.01-IDRR数据\IDRR-base\CorpusData\PDTB2\pdtb2.csv',
        data_name='pdtb2',
        label_level='level1',
        model_name_or_path='roberta-base',
        logger=CustomLogger('tmp', print_output=True),
        label_expansion_positive=True,
        label_expansion_negative=True,
    )
    for p in sample_dataset.train_dataset:
        print(p)
        break