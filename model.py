import torch 
import torch.nn as nn 
import transformers

from transformers import (AutoConfig,
                          AutoTokenizer,
                          AutoModel,
                          AutoModelForSequenceClassification,
                          )


class CustomLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self, logits, labels):
        probs = torch.softmax(logits, dim=1)
        zeros = torch.zeros_like(labels)
        positive_labels = torch.max(labels, zeros)
        negative_labels = torch.min(labels, zeros)
        return torch.sum(-positive_labels*torch.log(probs)+negative_labels*torch.log(1-probs))/labels.shape[0]
        
        
class CustomModel(nn.Module):
    def __init__(self, model_name_or_path, num_labels=4, cache_dir='', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.cache_dir = cache_dir
        
        self.model:nn.Module = None
        self.model_config = None
        self.initial_model()
        
        self.loss_fn = CustomLoss()
    
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
        

class BaselineModel(nn.Module):
    def __init__(self, model_name_or_path) -> None:
        super(BaselineModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model_config = AutoConfig.from_pretrained(model_name_or_path)
        self.classifer = nn.Linear(self.model_config.hidden_size, 4)
        # self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = CustomLoss()

    def forward(self, input_ids, attention_mask, labels):
        model_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_token_repre = model_outputs.pooler_output
        logits = self.classifer(cls_token_repre)
        
        # labels = torch.argmax(labels, dim=1)
        loss = self.loss_fn(logits, labels)

        return {
            'loss': loss,
            'logits': logits,
        }


if __name__ == '__main__':
    sample_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    sample_model = CustomModel('roberta-base', num_labels=3)
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