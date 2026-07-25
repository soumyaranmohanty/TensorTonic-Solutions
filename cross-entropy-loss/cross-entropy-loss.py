import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    
    loss = 0
    N = len(y_true)
    y_pred = np.array(y_pred)
    top_pred = y_pred[np.arange(N), y_true]
    loss=-(1/N)*(np.sum(np.log(top_pred)))

    return loss