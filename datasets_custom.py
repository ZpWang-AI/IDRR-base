import pandas as pd

from transformers import AutoTokenizer, DataCollatorWithPadding


class CustomDatasets():
    def __init__(self, file_path, data_name='pdtb2', model_name_or_path='roberta-base', logger=None) -> None:
        assert data_name in ['pdtb2', 'pdtb3', 'conll']
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.sep_t = tokenizer.sep_token
        self.cls_t = tokenizer.cls_token
        self.tokenizer = tokenizer 
        
        self.label_map = {
            'Temporal': 0, 
            'Comparison': 1, 
            'Contingency': 2,
            'Expansion': 3,
        }

        df = pd.read_csv(file_path, usecols=['Relation', 'Section', 'Arg1_RawText', 'Arg2_RawText', 'ConnHeadSemClass1', 'ConnHeadSemClass2'])
        df = df[df['Relation'] == 'Implicit']

        self.train_dataset = self.data_tokenize(df[~df['Section'].isin([0, 1, 21, 22, 23, 24])])
        self.dev_dataset = self.data_tokenize(df[df['Section'].isin([0, 1])])
        self.test_dataset = self.data_tokenize(df[df['Section'].isin([21, 22])])

        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        if logger is not None:
            logger.info('-' * 30)
            logger.info('Trainset Size: {}'.format(len(self.train_dataset)))
            logger.info('Devset Size: {}'.format(len(self.dev_dataset)))
            logger.info('Testset Size: {}'.format(len(self.test_dataset)))
            logger.info('-' * 30)

    def data_tokenize(self, examples):
        dataset = []
        for arg1, arg2, sense in zip(examples['Arg1_RawText'], examples['Arg2_RawText'], examples['ConnHeadSemClass1']):
            model_inputs = self.tokenizer(
                arg1, 
                arg2, 
                add_special_tokens=True, 
                truncation='longest_first', 
                max_length=256
            )
            
            model_inputs['label'] = self.label_map[sense.split('.')[0]]
        
            dataset.append(model_inputs)

        return dataset