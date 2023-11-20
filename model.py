import numpy as np
import torch 
import torch.nn as nn 
import transformers

from transformers import (AutoConfig,
                          AutoTokenizer,
                          AutoModel,
                          AutoModelForSequenceClassification,
                          )


# make sure `labels.dim = 2`
class CELoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, logits:torch.Tensor, labels:torch.Tensor):
        """
        n = label categories
        logits: [batch size, n]
        labels: [batch size, n]
        
        formula = sum( labels * log(softmax(logits)) )
        """
        probs = torch.softmax(logits, dim=1)
        return -(labels*torch.log(probs)).sum(dim=1).mean()

        
class CustomModel(nn.Module):
    def __init__(
        self, 
        model_name_or_path,
        num_labels=4,
        cache_dir='',
        loss_type='CELoss',
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.cache_dir = cache_dir
        
        self.model:nn.Module = None
        self.model_config = None
        self.initial_model()
        
        if loss_type.lower() == 'celoss':
            self.loss_fn = CELoss()
        else:
            raise Exception('wrong loss_type')
    
    def initial_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path, 
            num_labels=self.num_labels,
            cache_dir=self.cache_dir
        )
        self.model_config = AutoConfig.from_pretrained(
            self.model_name_or_path, 
            num_labels=self.num_labels,
            cache_dir=self.cache_dir
        )
    
    def forward(self, input_ids, attention_mask, labels):
        model_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = model_outputs.logits
        loss = self.loss_fn(logits, labels)
        
        return {
            'logits': logits,
            'loss': loss,
        }        
        

if __name__ == '__main__':
    def demo_model():
        cache_dir = './plm_cache/'
        sample_model = CustomModel('roberta-base', num_labels=3, cache_dir=cache_dir)
        
        sample_tokenizer = AutoTokenizer.from_pretrained('roberta-base', cache_dir=cache_dir)
        sample_x = ['你好']*2+['hello world. Nice to see you']*2
        sample_x_token = sample_tokenizer(sample_x, padding=True, return_tensors='pt',)
        sample_y = torch.Tensor([
            [1, 0, 0],
            [1, 1, 0],
            [1, 0.5, 0],
            [1, 0.5, -1],
        ])
        res = sample_model(sample_x_token['input_ids'], sample_x_token['attention_mask'], sample_y)
        print(res)
        
    def demo_CELoss():
        y_pred = torch.tensor([[0.8, 0.5, 0.9, 0.4, 0.7],
                            [0.3, 0.6, 0.1, 0.7, 0.5]])
        y_true = torch.tensor([[1, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0]])       
        y_true2 = torch.tensor([0, 1]) 
        criterion1 = CELoss()
        criterion2 = nn.CrossEntropyLoss(reduction='mean')
        loss1 = criterion1(y_pred, y_true)
        loss2 = criterion2(y_pred, y_true2)
        print(loss1, loss2, sep='\n')
    
    demo_model()
    demo_CELoss()
    