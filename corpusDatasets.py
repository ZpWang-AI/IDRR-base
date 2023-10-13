import numpy as np
import pandas as pd

from typing import *
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import Dataset


class LabelManager:
    def __init__(
        self,
        label_ids:List[int],
        num_labels:int,
        additional_label_ids:Optional[List[List[int]]]=None,
        label_expansion_positive=0.0,
        label_expansion_negative=0.0,
        dynamic_positive=0.0,
        dynamic_negative=0.0,
        max_positive_limit=5.0,
        max_negative_limit=5.0,
    ) -> None:
        self.label_ids = label_ids
        self.num_labels = num_labels
        self.additional_label_ids = additional_label_ids
        self.label_expansion_positive = label_expansion_positive
        self.label_expansion_negative = label_expansion_negative
        self.dynamic_positive = dynamic_positive
        self.dynamic_negative = dynamic_negative
        self.max_positive_limit = max_positive_limit
        self.max_negative_limit = max_negative_limit

        self.initial_labels()
    
    def __getitem__(self, index):
        return self.label_vectors[index]
    
    def __len__(self):
        return len(self.label_ids)
    
    def initial_labels(self):
        self.label_vectors = np.eye(self.num_labels)[self.label_ids]
        
        if self.label_expansion_positive:
            if self.additional_label_ids is None:
                raise Exception('miss additional_label_ids')
            for p, add_ids in enumerate(self.additional_label_ids):
                for add_id in add_ids:
                    self.label_vectors[p][add_id] += self.label_expansion_positive
        
        if self.label_expansion_negative:
            self.label_vectors -= self.label_expansion_negative
        pass
        
    def update_labels(self, preds, labels):
        if not self.dynamic_positive and not self.dynamic_negative:
            return
        
        preds = np.argmax(preds, axis=1)
        labels = np.argmax(labels, axis=1)
        correct = preds == labels

        if self.dynamic_positive:
            pos_addition = (np.eye(self.num_labels)*self.dynamic_positive)[labels]
            pos_addition[correct] *= 0
            self.label_vectors += pos_addition
        if self.dynamic_negative:
            neg_addition = (np.eye(self.num_labels)*self.dynamic_negative)[preds]
            neg_addition[correct] *= 0
            self.label_vectors -= neg_addition
        
        self.label_vectors = np.clip(
            self.label_vectors,
            a_min=-self.max_negative_limit, 
            a_max=self.max_positive_limit
        )


class CustomDataset(Dataset):
    def __init__(self, arg1, arg2, tokenizer, label_manager) -> None:
        super().__init__()
        
        self.arg1 = arg1
        self.arg2 = arg2
        self.tokenizer = tokenizer
        self.label_manager = label_manager
        
    def __getitem__(self, index) -> Any:
        model_inputs = self.tokenizer(
            self.arg1[index], 
            self.arg2[index], 
            add_special_tokens=True, 
            truncation='longest_first', 
            max_length=256,
        )
        
        model_inputs['label'] = self.label_manager[index]
    
        return model_inputs
    
    def __len__(self):
        return len(self.arg1)


class CustomCorpusDatasets():
    def __init__(
        self, 
        file_path,
        data_name='pdtb2',
        model_name_or_path='roberta-base',
        cache_dir='',
        logger=None,
        mini_dataset=False,
        
        label_level='level1',
        label_expansion_positive=0.0,
        label_expansion_negative=0.0,
        dynamic_positive=0.0,
        dynamic_negative=0.0,
        max_positive_limit=5.0,
        max_negative_limit=5.0,
        data_augmentation=False,
    ):
        assert data_name in ['pdtb2', 'pdtb3', 'conll']
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.sep_t = tokenizer.sep_token
        self.cls_t = tokenizer.cls_token
        self.tokenizer = tokenizer 
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
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
        if mini_dataset:
            train_df = train_df.iloc[:32]
            dev_df = dev_df.iloc[:16]
            test_df = test_df.iloc[:16]
            
        if data_augmentation:
            train_df = self.data_augmentation_df(train_df)

        self.label_level = label_level
        self.label_expansion_positive = label_expansion_positive
        self.label_expansion_negative = label_expansion_negative
        self.dynamic_positive = dynamic_positive
        self.dynamic_negative = dynamic_negative
        self.max_positive_limit = max_positive_limit
        self.max_negative_limit = max_negative_limit
        self.data_augmentation = data_augmentation
        
        self.train_dataset = self.get_dataset(train_df, is_train=True)
        self.dev_dataset = self.get_dataset(dev_df, is_train=False)
        self.test_dataset = self.get_dataset(test_df, is_train=False)
        
        if logger is not None:
            logger.info('-' * 30)
            logger.info(f'Trainset Size: {len(self.train_dataset)}')
            logger.info(f'Devset Size: {len(self.dev_dataset)}')
            logger.info(f'Testset Size: {len(self.test_dataset)}')
            logger.info('-' * 30)
    
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
            conn1sense1 = row.ConnHeadSemClass1
            conn1sense2 = row.ConnHeadSemClass2
            conn2sense1 = row.Conn2SemClass1
            conn2sense2 = row.Conn2SemClass2
            
            arg1_list.append(arg1)
            arg2_list.append(arg2)
            
            label_ids.append(self.label_to_id(conn1sense1))
            cur_adds = [self.label_to_id(sense) 
                        for sense in [conn1sense2, conn2sense1, conn2sense2]
                        if not pd.isna(sense)]
            additional_label_ids.append(cur_adds)
            
        if is_train:
            label_manager = LabelManager(
                label_ids=label_ids,
                additional_label_ids=additional_label_ids,
                num_labels=self.num_labels,
                label_expansion_positive=self.label_expansion_positive,
                label_expansion_negative=self.label_expansion_negative,
                dynamic_positive=self.dynamic_positive,
                dynamic_negative=self.dynamic_negative,
                max_positive_limit=self.max_positive_limit,
                max_negative_limit=self.max_negative_limit,
            )
        else:
            label_manager = LabelManager(
                label_ids=label_ids,
                additional_label_ids=additional_label_ids,
                num_labels=self.num_labels,
                label_expansion_positive=0,
                label_expansion_negative=0,
                dynamic_positive=0,
                dynamic_negative=0,
                max_positive_limit=0,
                max_negative_limit=0,
            )
        
        return CustomDataset(
            arg1=arg1_list,
            arg2=arg2_list,
            tokenizer=self.tokenizer,
            label_manager=label_manager,
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
        data_augmentation=True,
    )
    for p in sample_dataset.train_dataset:
        print(p)
        break