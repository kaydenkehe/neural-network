import numpy as np

class Linear:
    def __init__(self, k=1):
        self.k = k

    # kz
    def forward(self, Z):
        return self.k * Z

    def backward(self, Z):
        return self.k

class Sigmoid:
    def __init__(self):
        pass

    # (1 / 1 + e^-z)
    def forward(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    # s(z) * (1 - s(z))
    def backward(self, Z):
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)

class ReLU():
    def __init__(self):
        pass

    # max(0,z)
    def forward(self, Z):
        return np.maximum(0, Z)
    
    # 0 if z <= 0, 1 otherwise
    def backward(self, Z):
        return np.where(Z <= 0, 0, 1)
