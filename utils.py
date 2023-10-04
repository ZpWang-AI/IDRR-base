import os
import numpy as np

from typing import *
from transformers import TrainerCallback, TrainerState, TrainerControl
from sklearn.metrics import f1_score, accuracy_score


class SaveBestModelCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        
        self.trainer = None
        self.best_metric = {
            'best_acc' : None,
            'best_macro_f1' : None,
            'best_tem' : None,
            'best_com' : None,
            'best_con' : None,
            'best_exp' : None
        }

    def on_evaluate(self, args, state, control, metrics:Dict[str, float], **kwargs):
        # 假设我们使用的是准确率作为评估指标，且越高越好
        
        for metric_name, metric_value in metrics.items():
            best_metric_name = metric_name.replace('eval_', 'best_')
            if best_metric_name not in self.best_metric:
                continue
            
            # 如果这次的评估结果好于历史最优结果，那么保存模型
            if self.best_metric[best_metric_name] is None or self.best_metric[best_metric_name]:
                self.best_metric[best_metric_name] = metric_value
                
                best_model_path = os.path.join(args.output_dir, f"checkpoint-best-{metric_name.replace('eval', 'dev')}")
                self.trainer.save_model(best_model_path)
                print(f"New best model saved to {best_model_path}")
    
    def add_trainer(self, trainer):
        self.trainer = trainer



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # import pdb; pdb.set_trace()
    predictions = np.argmax(predictions, axis=1)
    metrics = [None]*6
    metrics[5] = accuracy_score(labels, predictions)
    metrics[4] = f1_score(labels, predictions, average='macro')
    
    for i in range(4):
        _preds = (predictions == i).astype(int)
        _labels = (labels == i).astype(int)
        metrics[i] = f1_score(_labels, _preds)

    return {
        'acc': metrics[5],
        'macro_f1': metrics[4],
        'tem': metrics[0],
        'com': metrics[1],
        'con': metrics[2],
        'exp': metrics[3]
    }


    

