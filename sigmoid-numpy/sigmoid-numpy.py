import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    np_arr = np.array(x)
    # print(np_arr)
    return 1/(1+np.exp(-np_arr))
    