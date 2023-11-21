import numpy as np

# Cross-Entropy
# For binary classification
class BinaryCrossentropy:
    # (-1 / m * sum(yln(a) + (1 - y)ln(1 - a)))
    @staticmethod
    def forward(AL, Y):
        return np.squeeze(-1 / Y.shape[1] * np.sum(np.dot(Y, np.log(AL).T) + np.dot(1 - Y, np.log(1 - AL).T)))
    
    # (-y/a + (1 - y)/(1 - a))
    @staticmethod
    def backward(AL, Y):
        return -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
