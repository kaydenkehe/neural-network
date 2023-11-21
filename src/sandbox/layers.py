import numpy as np

class Dense():
    def __init__(self, units, activation):
        self.units = units
        self.activation = activation

    # Calculate linear step of neuron activation (W^T * A + b)
    def forward(self, A_prev, W, b):
        Z = np.dot(W, A_prev) + b # Compute Z
        A = self.activation.forward(Z) # Compute A using the given activation function
        cache = ((A_prev, W, b), Z) # Cache for backprop

        return A, cache
    
    # Find derivative with respect to weights, biases, and activations for a particular layer
    def backward(self, dA, cache):
        linear_cache, Z = cache
        dZ = dA * self.activation.backward(Z) # Evaluate dZ using the derivative of activation function
        A_prev, W, _ = linear_cache
        m = A_prev.shape[1]

        dW = 1 / m * np.dot(dZ, A_prev.T) # Calculate derivative with respect to weights
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True) # Calculate derivative with respect to biases
        dA_prev = np.dot(W.T, dZ) # Calculate derivative with respect to the activation of the previous layer

        return dA_prev, dW, db
