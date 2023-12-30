# Handle conditional imports
def configure_imports(cuda):
    global np
    np = __import__('cupy' if cuda else 'numpy')

'''
Every activation class includes three methods:
    - __init__: Initialize activation parameters (if applicable)
    - forward: Compute neuron activation
    - backward: If dA is supplied, compute derivative of cost w.r.t. un-activated neuron value (Z)
                If dA is not supplied, compute derivative of activation w.r.t. Z
'''

# Used in regression output layers
class Linear:

    def __init__(self, k=1):
        self.k = k

    # kz
    def forward(self, Z):
        return self.k * Z

    # k
    def backward(self, Z, dA=1):
        dZ = self.k
        return dA * dZ

# Maps output to [0, 1] (probability)
# Used in binary classification output layers
class Sigmoid:

    def __init__(self, c=1):
        self.c = c

    # 1 / (1 + e^-cz)
    def forward(self, Z):
        return 1 / (1 + np.exp(-self.c * Z))
    
    # c * s(z) * (1 - s(z))
    def backward(self, Z, dA=1):
        s = self.forward(Z)
        dZ = self.c * s * (1 - s)
        return dA * dZ

# Each output is [0, 1] and sums to 1
# Used in multi-class classification output layers
class Softmax:
    
        def __init__(self):
            pass
    
        # e^z / sum(e^z)
        def forward(self, Z):
            e = np.exp(Z) # Subtract max to avoid nan values
            return e / np.sum(e, axis=1, keepdims=True)
        
        '''
        Softmax is basically always used with categorical crossentropy,
        and when taken together, the derivative of the cost w.r.t. the un-activated last layer is just AL - Y.
        However, in the interest of modularity and allowing softmax and categorical crossentropy to be used separately,
        I have opted to include both of their individual derivatives.
        '''

        # Vectorized softmax Jacobian calculation taken from user Michael Bay on Stack Overflow
        # https://stackoverflow.com/questions/36279904/softmax-derivative-in-numpy-approaches-0-implementation
        # s(z) * (diracdelta - s(z)) (Jacobian)
        def backward(self, Z, dA=None):
            shape = Z.shape
            s = self.forward(Z)
            diag = s.reshape(shape[0], -1, 1) * np.diag(np.ones(shape[1])) # Calculate diagonal (vectorized equivalent to np.diag(s))
            outer = np.matmul(s.reshape(shape[0], -1, 1), s.reshape(shape[0], 1, -1)) # Calculate outer products (vectorized equivalent to np.outer(s, s))
            jacobian = diag - outer # Calculate Jacobian

            # If dA is supplied, return dJ/dZ, otherwise, return dA/dZ
            if dA is None: jacobian
            else: return np.matmul(dA.reshape(shape[0], 1, -1), jacobian).reshape(shape)

# Rectified Linear Units
# Most common hidden layer activation
class ReLU():

    def __init__(self):
        pass

    # max(0,z)
    def forward(self, Z):
        return np.maximum(0, Z)
    
    # 0 if z <= 0, 1 if z > 0
    def backward(self, Z, dA=1):
        dZ = np.where(Z <= 0, 0, 1)
        return dA * dZ

# Offshoot of ReLU, attempts to combat vanishing gradient
class LeakyReLU():

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    # max(0,z) + alpha * min(0,z)
    def forward(self, Z):
        return np.maximum(0, Z) + self.alpha * np.minimum(0, Z)
    
    # 0 if z <= 0, 1 otherwise
    def backward(self, Z, dA=1):
        dZ = np.where(Z <= 0, self.alpha, 1)
        return dA * dZ
    
class Tanh():

    def __init__(self):
        pass

    # (e^z - e^-z) / (e^z + e^-z)
    def forward(self, Z):
        return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    
    # 1 - tanh^2(z)
    def backward(self, Z, dA=1):
        dZ = 1 - self.forward(Z) ** 2
        return dA * dZ

# Exponential Linear Units
class ELU():

    def __init__(self, alpha=1):
        self.alpha = alpha

    # alpha * (e^z - 1) if z <= 0, z otherwise
    def forward(self, Z):
        return np.where(Z < 0, self.alpha * (np.exp(Z) - 1), Z)
    
    # alpha * e^z if z < 0, 1 otherwise
    def backward(self, Z, dA=1):
        dZ = np.where(Z < 0, self.alpha * np.exp(Z), 1)
        return dA * dZ
    
# Scaled Exponential Linear Units
class SELU():

    def __init__(self, alpha=1, scale=1):
        self.alpha = alpha
        self.scale = scale

    # scale * alpha * (e^z - 1) if z <= 0, scale * z otherwise
    def forward(self, Z):
        return self.scale * np.where(Z < 0, self.alpha * (np.exp(Z) - 1), Z)
    
    # scale * alpha * e^z if z < 0, scale otherwise
    def backward(self, Z, dA=1):
        dZ = self.scale * np.where(Z < 0, self.alpha * np.exp(Z), 1)
        return dA * dZ
    
# Sigmoid Linear Units
class SLU():

    def __init__(self):
        pass

    # (z / 1 + e^-z)
    def forward(self, Z):
        return Z / (1 + np.exp(-Z))
    
    # sigmoid(z) * (1 + z * sigmoid(-z))
    def backward(self, Z, dA=1):
        dZ = Sigmoid().forward(Z) * (1 + Z * Sigmoid().forward(-Z))
        return dA * dZ
    
class Softplus():

    def __init__(self):
        pass

    # ln(1 + e^z)
    def forward(self, Z):
        return np.log(1 + np.exp(Z))
    
    # 1 / (1 + e^-z)
    def backward(self, Z, dA=1):
        dZ = 1 / (1 + np.exp(-Z))
        return dA * dZ

class Softsign():

    def __init__(self):
        pass

    # z / (1 + |z|)
    def forward(self, Z):
        return Z / (1 + np.abs(Z))
    
    # 1 / (1 + |z|)^2
    def backward(self, Z, dA=1):
        dZ = 1 / (1 + np.abs(Z)) ** 2
        return dA * dZ
    
class BentIdentity():

    def __init__(self):
        pass

    # ((sqrt(z^2 + 1) - 1) / 2) + z
    def forward(self, Z):
        return ((np.sqrt(Z ** 2 + 1) - 1) / 2) + Z
    
    # (z / (2 * sqrt(z^2 + 1))) + 1
    def backward(self, Z, dA=1):
        dZ = (Z / (2 * np.sqrt(Z ** 2 + 1))) + 1
        return dA * dZ

class Gaussian():

    def __init__(self):
        pass

    # e^-z^2
    def forward(self, Z):
        return np.exp(-Z ** 2)
    
    # -2z * e^-z^2
    def backward(self, Z, dA=1):
        dZ = -2 * Z * np.exp(-Z ** 2)
        return dA * dZ

class Arctan():

    def __init__(self):
        pass

    # arctan(z)
    def forward(self, Z):
        return np.arctan(Z)
    
    # 1 / (z^2 + 1)
    def backward(self, Z, dA=1):
        dZ = 1 / (Z ** 2 + 1)
        return dA * dZ

class PiecewiseLinear():

    def __init__(self, alpha=1):
        self.alpha = alpha

    # -1 if x <= -alpha, x / alpha if -alpha < x < alpha, 1 if x >= alpha
    def forward(self, Z):
        return np.where(Z <= -self.alpha, -1, np.where(Z >= self.alpha, 1, Z / self.alpha))
    
    # 0 if x <= -alpha, 1 / alpha if -alpha < x < alpha, 0 if x >= alpha
    def backward(self, Z, dA=1):
        dZ = np.where(Z <= -self.alpha, 0, np.where(Z >= self.alpha, 0, 1 / self.alpha))
        return dA * dZ
