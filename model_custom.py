import torch 
import torch.nn as nn 

from transformers import AutoModel, AutoConfig


class BaselineModel(nn.Module):
    def __init__(self, model_name_or_path) -> None:
        super(BaselineModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model_config = AutoConfig.from_pretrained(model_name_or_path)
        self.classifer = nn.Linear(self.model_config.hidden_size, 4)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels):
        model_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_token_repre = model_outputs.pooler_output
        logits = self.classifer(cls_token_repre)
        loss = self.loss_fn(logits, labels)

        return {
            'loss': loss,
            'logits': logits,
        }

