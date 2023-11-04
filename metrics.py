import numpy as np

from sklearn.metrics import f1_score, accuracy_score


class ComputeMetrics:
    def __init__(self, label_list:list) -> None:
        self.label_list = label_list
        self.metric_names = ['Acc', 'Macro-F1']+label_list
        
    def __call__(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        predictions = np.eye(len(self.label_list))[predictions]
        
        res = {
            # 'Acc': accuracy_score(labels, predictions),
            'Acc': np.sum(predictions*labels)/len(predictions),
            'Macro-F1': f1_score(labels, predictions, average='macro', zero_division=0),
        }
        
        for i, target_type in enumerate(self.label_list):
            res[target_type] = f1_score(predictions[:,i], labels[:,i], zero_division=0)
        
        return res
    
    
if __name__ == '__main__':
    sample_pred = np.random.random(20)
    sample_label = np.random.random(20)
    print(sample_pred.reshape((-1, 4)))
    