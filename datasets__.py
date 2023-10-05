import os
import pandas as pd

from transformers import AutoTokenizer, DataCollatorWithPadding

from utils import get_logger

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class CustomDatasets():
    def __init__(self, file_path, data_name='pdtb2', label_level='level1', model_name_or_path='roberta-base', logger=None):
        assert data_name in ['pdtb2', 'pdtb3', 'conll']
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.sep_t = tokenizer.sep_token
        self.cls_t = tokenizer.cls_token
        self.tokenizer = tokenizer 
        
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

        self.train_dataset = self.data_tokenize(df[~df['Section'].isin([0, 1, 21, 22, 23, 24])])
        self.dev_dataset = self.data_tokenize(df[df['Section'].isin([0, 1])])
        self.test_dataset = self.data_tokenize(df[df['Section'].isin([21, 22])])

        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        if logger is not None:
            logger.info('-' * 30)
            logger.info(f'Trainset Size: {len(self.train_dataset)}')
            logger.info(f'Devset Size: {len(self.dev_dataset)}')
            logger.info(f'Testset Size: {len(self.test_dataset)}')
            logger.info('-' * 30)

    def data_tokenize(self, df_):
        dataset = []
        for row in df_.itertuples():
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
            
            label_id = self.label_map[conn1sense1.split('.')[0]]
            label = [0]*len(self.label_map)
            label[label_id] = 1
            model_inputs['label'] = label
        
            dataset.append(model_inputs)
            # print(label)
            # break

        return dataset
    
    
if __name__ == '__main__':
    sample_dataset = CustomDatasets(r'D:\0--data\projects\04.01-IDRR数据\IDRR-base\CorpusData\PDTB2\pdtb2.csv',
                                    data_name='pdtb2',
                                    label_level='level1',
                                    model_name_or_path='roberta-base',
                                    logger=get_logger())
    for p in sample_dataset.train_dataset:
        print(p)
        break