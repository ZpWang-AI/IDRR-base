import torch 
import torch.nn as nn 
import transformers

from transformers import (AutoConfig,
                          AutoTokenizer,
                          AutoModel,
                          AutoModelForSequenceClassification,
                          )


class CELoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, logits, labels):
        probs = torch.softmax(logits, dim=1)
        return -torch.sum(labels*torch.log(probs))/labels.shape[0]


class ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RankModel(nn.Module):
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
        
        self.model_config = None
        self.encoder:nn.Module = None
        self.classifier:nn.Module = None
        self.initial_model()
        
        if loss_type.lower() == 'celoss':
            self.loss_fn = CELoss()
        else:
            raise Exception('wrong loss_type')
        
    def initial_model(self):
        self.model_config = AutoConfig.from_pretrained(
            self.model_name_or_path, 
            num_labels=self.num_labels,
            cache_dir=self.cache_dir
        )
        self.encoder = AutoModel.from_pretrained(
            self.model_name_or_path, 
            num_labels=self.num_labels,
            cache_dir=self.cache_dir
        )
        self.classifier = ClassificationHead(self.model_config)
        
    def hot_start(self):
        pass
    
    def forward(self, input_ids, attention_mask, labels):
        model_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = model_outputs.logits
        loss = self.loss_fn(logits, labels)

        return {
            'logits': logits,
            'loss': loss,
        }        
     
        
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