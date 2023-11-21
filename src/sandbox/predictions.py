import numpy as np

def binary_classification(Y):
    return np.where(Y > 0.5, 1, 0)