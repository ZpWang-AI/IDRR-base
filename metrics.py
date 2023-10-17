import numpy as np

from sklearn.metrics import f1_score, accuracy_score


class ComputeMetrics:
    def __init__(self, label_list:list) -> None:
        self.label_list = label_list
        self.metric_names = ['Acc', 'Macro-F1']+label_list
        
    def __call__(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        if labels.ndim == 2:
            labels = np.argmax(labels, axis=1)
        
        res = {
            'Acc': accuracy_score(labels, predictions),
            'Macro-F1': f1_score(labels, predictions, average='macro'),
        }
        
        for i, target_type in enumerate(self.label_list):
            res[target_type] = f1_score(predictions==i, labels==i)
        
        return res