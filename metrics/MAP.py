import numpy as np 
from sklearn.metrics import average_precision_score

class MAP():
    def __init__(self):
        super().__init__()
        
    def scoring(self, y_true, y_pred=None, y_score=None):
        # >> scoring(y_true,y_pred=None,y_score=None): Initialize the data transformation method.
        # >> - y_true: Ground-truth labels.
        # >> - y_pred: Hard labels for model predictions.
        # >> - y_score: Soft labels for model predictions.
        
        y_true = y_true.tolist()
        y_score = y_score.tolist()
        

        _, num_classes = np.shape(y_true)

        y_pred = np.array(y_score)
        y_true = np.array(y_true)
        y_true = np.array(y_true == 1, dtype=np.float32) # convert from -1 / 1 format to 0 / 1 format

        average_precision_list = [] #For Macro label-level mAP

        for j in range(num_classes):
            average_precision_list.append(compute_avg_precision(y_true[:, j], y_pred[:, j]))

        return 100.0 * float(np.mean(average_precision_list))


def check_inputs(targs, preds):
    '''
    Helper function for input validation.
    '''

    assert (np.shape(preds) == np.shape(targs))
    assert type(preds) is np.ndarray
    assert type(targs) is np.ndarray
    assert (np.max(preds) <= 1.0) and (np.min(preds) >= 0.0)
    assert (np.max(targs) <= 1.0) and (np.min(targs) >= 0.0)
    assert (len(np.unique(targs)) <= 2)

def compute_avg_precision(targs, preds):
    
    '''
    Compute average precision.
    
    Parameters
    targs: Binary targets.
    preds: Predicted probability scores.
    '''
    
    check_inputs(targs,preds)
    
    if np.all(targs == 0):
        # If a class has zero true positives, we define average precision to be zero.
        metric_value = 0.0
    else:
        metric_value = average_precision_score(targs, preds)
    
    return metric_value