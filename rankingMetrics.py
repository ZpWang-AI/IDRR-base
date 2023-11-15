import numpy as np


class RankingMetrics:
    def __init__(self, num_labels:int) -> None:
        self.num_labels = num_labels
        self.metric_names = ['rank_Acc']
        pass
    
    def __call__(self, eval_pred):
        predictions, labels = eval_pred
        
        res = {
            'rank_Acc': np.mean( np.argmax(predictions, axis=1) == 0)
        }
        return res