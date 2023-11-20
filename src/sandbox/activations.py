import numpy as np

class Sigmoid:
    # (1 / 1 + e^-z)
    @staticmethod
    def forward(Z):
        return 1 / (1 + np.exp(-Z)), Z
    
    # s(z) * (1 - s(z))
    @staticmethod
    def backward(dA, cache):
        s = 1/(1+np.exp(-cache))
        return dA * s * (1-s)

class Relu():
    # max(0,z)
    @staticmethod
    def forward(Z):
        return np.maximum(0, Z), Z
    
    # 0 if z <= 0, 1 otherwise
    @staticmethod
    def backward(dA, cache):
        dZ = np.array(dA, copy=True) 
        dZ[cache <= 0] = 0
        return dZ
