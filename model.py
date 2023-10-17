import numpy as np
import torch 
import torch.nn as nn 
import transformers

from transformers import (AutoConfig,
                          AutoTokenizer,
                          AutoModel,
                          AutoModelForSequenceClassification,
                          )


class ListMLELoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self, scores, labels=None, k=None):
        if k is not None:
            sublist_ids = (torch.rand(size=(k,))*scores.shape[1]).long()
            scores = scores[:, sublist_ids]
            if labels is not None:
                labels = labels[:, sublist_ids]
        
        if labels is not None:
            _, sort_ids = labels.sort(descending=True, dim=-1)
            scores = scores.gather(dim=1, index=sort_ids)
            
        cumsums = scores.exp().flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
        loss = torch.log(cumsums+1e-10) - scores
        return loss.sum(dim=1).mean()


class ListNetLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, logits, labels):
        probs = torch.softmax(logits, dim=1)
        labels = torch.softmax(labels, dim=1)
        return -(labels*torch.log(probs)).sum(dim=1).mean()
    

class CELoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, logits, labels):
        probs = torch.softmax(logits, dim=1)
        return -(labels*torch.log(probs)).sum(dim=1).mean()


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
        label_list=(),
        cache_dir='',
        loss_type='CELoss',
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.model_name_or_path = model_name_or_path
        self.label_list = label_list
        self.num_labels = len(label_list)
        self.cache_dir = cache_dir
        
        self.model_config = None
        self.encoder:nn.Module = None
        self.classifier:nn.Module = None
        self.label_vectors:torch.Tensor = None
        
        self.forward_fn = self.forward_fine_tune
        if loss_type.lower() == 'celoss':
            self.loss_fn = CELoss()
        else:
            raise Exception('wrong loss_type')
        
        self.initial_model()
        
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
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, cache_dir=self.cache_dir)
        tokenized_labels = tokenizer(self.label_list, padding=True, return_tensors='pt')
        self.label_vectors = self.encoder(
            input_ids=tokenized_labels['input_ids'],
            attention_mask=tokenized_labels['attention_mask'],
        ).last_hidden_state[:,0,:]
    
    def forward_rank(self, input_ids, attention_mask, labels):
        hidden_state = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = hidden_state.pooler_output
        label_vector = self.label_vectors[np.argmax(labels[0])]
        
        
    
    def forward_fine_tune(self, input_ids, attention_mask, labels):
        hidden_state = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state, pooler_output
        logits = self.classifier(hidden_state.last_hidden_state)
        loss = self.loss_fn(logits, labels)
        return {
            'logits': logits,
            'loss': loss,
        }
        
    def forward(self, *args, **kwargs):
        return self.forward_fn(*args, **kwargs)  
     
        
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
        sample_model = RankModel('roberta-base', label_list=['good', 'bad', 'middle'], cache_dir=cache_dir)
        
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
        criterion1 = CELoss()
        criterion2 = nn.CrossEntropyLoss(reduction='mean')
        loss1 = criterion1(y_pred, y_true)
        loss2 = criterion2(y_pred, torch.argmax(y_true, dim=1))
        print(loss1, loss2)
    
    def demo_listMLE():
        y_pred = torch.tensor([[0.8, 0.5, 0.9, 0.4, 0.7],
                            [0.3, 0.6, 0.1, 0.7, 0.5]])
        y_true = torch.tensor([[1, 0, 0, 0, 1],
                            [0, 1, 1, 0, 0]])
        k = 3
        criterion = ListMLELoss()
        loss = criterion(y_pred, y_true, k)
        print(loss)
    
    demo_CELoss()
    # demo_listMLE()
    