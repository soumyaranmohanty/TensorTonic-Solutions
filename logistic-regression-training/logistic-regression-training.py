import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    
    
    num_samples, num_features = X.shape
    w = np.zeros(num_features)
    b = 0.0
    
    for _ in range(steps):
    
        #print(np.dot(X, w))
        z =  np.dot(X, w) + b
        #print(z)
        ypred = _sigmoid(z)

        dw = (np.dot(X.T, ypred-y))/num_samples
        db = np.mean(ypred-y)
        
        w = w - lr*dw
        b = b - lr*db
    
    return (w,b)