import numpy as np
def normalize_data(x):
    return(x-np.mean(x)) /np.std(x)