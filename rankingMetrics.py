import numpy as np


class RankMetrics:
    def __init__(self, num_labels:int) -> None:
        self.num_labels = num_labels
        self.metric_names = ['rank_Acc']
        pass
    
    def __call__(self, eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.reshape((-1, self.num_labels))
        
        res = {
            'rank_Acc': np.mean( np.argmax(predictions, axis=1) == 0)
        }
        return res