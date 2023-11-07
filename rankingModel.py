import numpy as np
import torch 
import torch.nn as nn 
import transformers

from transformers import (AutoConfig,
                          AutoTokenizer,
                          AutoModel,
                          AutoModelForSequenceClassification,
                          )

from model import CELoss


class ListMLELoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self, scores:torch.Tensor, labels:torch.Tensor=None, k:int=None):
        """
        n = the number of the ranked samples
        scores: [batch size, n]
        labels: [batch size, n]
        if shape == [n]:
            unsqueeze and shape become [1, n]
        if k is not None:
            resample and choose `k` samples from the initial array (`scores` and `labels`)
        if labels is not None: 
            sort scores by labels from largest label to smallest label
        
        formula = softmax(x1, X) * softmax(x2, X-{x1}) * softmax(x3, X-{x1,x2}) * ...
        """
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        if labels is not None and labels.dim() == 1:
            labels = labels.unsqueeze(0)
        
        if k is not None:
            sublist_ids = (torch.rand(size=(k,))*scores.shape[1]).long()
            sublist_ids, _ = sublist_ids.sort()
            scores = scores[:, sublist_ids]
            if labels is not None:
                labels = labels[:, sublist_ids]
        
        if labels is not None:
            _, sort_ids = labels.sort(descending=True, dim=-1)
            scores = scores.gather(dim=1, index=sort_ids)
        
        cumsums = scores.exp().flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
        loss = torch.log(cumsums+1e-10) - scores
        loss = loss.sum(dim=1).mean()
        return loss


class ListNetLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, logits, labels):
        """
        n = label categories
        logits: [batch size, n]
        labels: [batch size, n]
        
        formula = sum( softmax(labels) * log(softmax(logits)) )
        """
        probs = torch.softmax(logits, dim=1)
        labels = torch.softmax(labels, dim=1)
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
        rank_loss_type='ListMLELoss',
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.model_name_or_path = model_name_or_path
        self.label_list = label_list
        self.num_labels = len(label_list)
        self.cache_dir = cache_dir
        
        self.model_config = None
        self.encoder:nn.Module = None
        self.classifier:nn.Module = None
        self.label_vectors:nn.Parameter = None
        self.initial_model()
        
        self.forward_fn = self.forward_fine_tune
        if loss_type.lower() == 'celoss':
            self.loss_fn = CELoss()
        else:
            raise Exception('wrong loss_type')
        if rank_loss_type.lower() == 'listmleloss':
            self.rank_loss_fn = ListMLELoss()
        else:
            raise Exception('wrong rank_loss_type')
        
        
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
        label_vectors = self.encoder(
            input_ids=tokenized_labels['input_ids'],
            attention_mask=tokenized_labels['attention_mask'],
        ).last_hidden_state[:,0,:]
        self.label_vectors = nn.Parameter(label_vectors)
    
    def forward_rank(self, input_ids, attention_mask, labels):
        hidden_state = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = hidden_state.pooler_output
        label_vector = self.label_vectors[torch.argmax(labels[0])]
        scores = (torch.stack([label_vector]*pooler_output.shape[0]) * pooler_output).sum(dim=1)
        loss = self.rank_loss_fn(scores)
        return {
            'logits': scores,
            'loss': loss,
        }
    
    def forward_fine_tune(self, input_ids, attention_mask, labels):
        hidden_state = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state, pooler_output
        logits = self.classifier(hidden_state.last_hidden_state)
        loss = self.loss_fn(logits, labels)
        return {
            'logits': logits,
            'loss': loss,
        }
        
    def forward(self, input_ids, attention_mask, labels):
        return self.forward_fn(input_ids, attention_mask, labels)
    
if __name__ == '__main__':
    def demo_model():
        cache_dir = './plm_cache/'
        sample_model = RankModel('roberta-base', label_list=['good', 'bad', 'middle', 'very good', 'very bad'], cache_dir=cache_dir)
        
        sample_tokenizer = AutoTokenizer.from_pretrained('roberta-base', cache_dir=cache_dir)
        sample_x = ['你好']*2+['hello world. Nice to see you']*2
        sample_x_token = sample_tokenizer(sample_x, padding=True, return_tensors='pt',)
        sample_y = torch.Tensor([
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0.5, 0, 0, 0],
            [1, 0.5, -1, 0, 0],
        ])
        res = sample_model(sample_x_token['input_ids'], sample_x_token['attention_mask'], sample_y)
        print(res)
        
        sample_model.forward_fn = sample_model.forward_rank
        res = sample_model(sample_x_token['input_ids'], sample_x_token['attention_mask'], sample_y)
        print(res)
        
    def demo_listMLE():
        y_pred = torch.tensor([[0.8, 0.5, 0.9, 0.4, 0.7],
                            [0.3, 0.6, 0.1, 0.7, 0.5]])
        y_true = torch.tensor([[1, 0, 0, 0, 1],
                            [0, 1, 1, 0, 0]])
        k = 3
        criterion = ListMLELoss()
        loss = criterion(y_pred, y_true, k)
        print(loss)
    
    demo_model()
    demo_listMLE()