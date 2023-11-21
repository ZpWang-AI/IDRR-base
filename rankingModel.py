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
            _, sort_ids = labels.sort(descending=True, dim=1)
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
    
    
class RankingModel(nn.Module):
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
        
        self.forward_fn = 'ft'
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
        label_vectors = self.encoder(**tokenized_labels).last_hidden_state[:,0,:]
        self.label_vectors = nn.Parameter(label_vectors)
        
    def forward(self, input_ids, attention_mask, labels):
        if self.forward_fn == 'ft':
            hidden_state = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            logits = self.classifier(hidden_state.last_hidden_state)
            loss = self.loss_fn(logits, labels)
            return {
                'logits': logits,
                'loss': loss,
            }            
        elif self.forward_fn == 'rank':
            # input, attention (bs, num, dim)
            rank_batch_size, num_labels, emb_dim = input_ids.shape  
            hidden_state = self.encoder(
                input_ids=input_ids.reshape(-1, emb_dim),
                attention_mask=attention_mask.reshape(-1, emb_dim),
            )
            # output (bs*num, dim2)
            pooler_output = hidden_state.pooler_output
            # labels (bs, num)
            labels = torch.argmax(labels, dim=1)
            # label_vector (bs*num, dim2)
            # repeat_interleave: (1,2,3) -> (1,1,2,2,3,3)
            label_vector = self.label_vectors[labels.repeat_interleave(num_labels)] 
            # scores (bs, num)
            scores = (pooler_output*label_vector).sum(dim=1)
            scores = scores.reshape(-1, num_labels)
            # loss (1)
            loss = self.rank_loss_fn(scores)
            return {
                'logits': scores,
                'loss': loss,
            }
        elif self.forward_fn == 'rank+ft':
            rank_batch_size, num_labels, emb_dim = input_ids.shape  
            hidden_state = self.encoder(
                input_ids=input_ids.reshape(-1, emb_dim),
                attention_mask=attention_mask.reshape(-1, emb_dim),
            )
            # rank
            pooler_output = hidden_state.pooler_output
            labels = torch.argmax(labels, dim=1)
            label_vector = self.label_vectors[labels.repeat_interleave(num_labels)] 
            scores = (pooler_output*label_vector).sum(dim=1)
            scores = scores.reshape(-1, num_labels)
            loss_rank = self.rank_loss_fn(scores)
            # ft
            logits = self.classifier(hidden_state.last_hidden_state)
            loss_ft = self.loss_fn(logits, labels)
            
            loss = loss_rank+loss_ft
            return {
                'logits': logits,
                'loss': loss,
            }
        else:
            raise ValueError('wrong forward_fn')
        
        
if __name__ == '__main__':
    def demo_model():
        cache_dir = './plm_cache/'
        model_name_or_path = 'roberta-base'
        model_name_or_path = './plm_cache/models--roberta-base/snapshots/bc2764f8af2e92b6eb5679868df33e224075ca68/'
        sample_model = RankingModel(model_name_or_path, cache_dir=cache_dir,
                                    label_list=['good', 'bad', 'middle', 'very good', 'very bad'])
        
        sample_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        sample_x = ['你好']*2+['hello world. Nice to see you']*2
        sample_x_token = sample_tokenizer(sample_x, padding=True, return_tensors='pt',)
        sample_y = torch.Tensor([
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0.5, 0, 0, 0],
            [1, 0.5, -1, 0, 0],
        ])
        print([(k,v.shape) for k,v in sample_x_token.items()])
        print(sample_y.shape)
        res = sample_model(sample_x_token['input_ids'], sample_x_token['attention_mask'], sample_y)
        print(res)
        
        sample_model.forward_fn = 'rank'
        sample_x_token['input_ids'] = sample_x_token['input_ids'].unsqueeze(0).repeat((10,1,1))
        sample_x_token['attention_mask'] = sample_x_token['attention_mask'].unsqueeze(0).repeat((10,1,1))
        sample_y = sample_y.argmax(dim=1).unsqueeze(0).repeat((10,1))
        print([(k,v.shape) for k,v in sample_x_token.items()])
        print(sample_y.shape)
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
    # demo_listMLE()