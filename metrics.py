import numpy as np

from sklearn.metrics import f1_score, accuracy_score


class ComputeMetrics:
    def __init__(self, label_map:dict) -> None:
        self.label_map = label_map
        self.metric_names = ['Acc', 'Macro-F1']+list(label_map.keys()) 
        
    def __call__(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        if labels.ndim == 2:
            labels = np.argmax(labels, axis=1)
        
        res = {
            'Acc': accuracy_score(labels, predictions),
            'Macro-F1': f1_score(labels, predictions, average='macro'),
        }
        
        for target_type, i in self.label_map.items():
            res[target_type] = f1_score(predictions==i, labels==i)
        
        return res