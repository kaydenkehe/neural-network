from sandbox import initializers

# Handle conditional imports
def configure_imports(cuda):
    global np
    np = __import__('cupy' if cuda else 'numpy')

'''
Every layer class includes three methods:
    - __init__: Initialize layer parameters
    - forward: Compute neuron activation
    - backward: Compute derivative of cost w.r.t. weights, biases, and activations
'''

'''
Dense / fully connected layer

The dense layer is the most basic layer in a neural network
Every neuron in a dense layer is connected to every neuron in the layer preceding it
'''
class Dense():
    def __init__(self, units, activation, initializer=initializers.glorot_uniform):
        self.trainable = True
        self.units = units
        self.activation = activation
        self.initializer = initializer

    # Calculate layer neuron activations
    def forward(self, A_prev, W, b):
        Z = np.dot(A_prev, W) + b # Compute Z
        A = self.activation.forward(Z) # Compute A using the given activation function

        return A, Z
    
    # Find derivative with respect to weights, biases, and activations for a particular layer
    def backward(self, dA, A_prev, W, b, Z):
        m = A_prev.shape[0]
        dZ = dA * self.activation.backward(Z) # Evaluate dZ using the derivative of activation function
        dW = 1 / m * np.dot(A_prev.T, dZ) # Calculate derivative with respect to weights
        db = 1 / m * np.sum(dZ, axis=0, keepdims=True) # Calculate derivative with respect to biases
        dA_prev = np.dot(dZ, W.T) # Calculate derivative with respect to the activation of the previous layer

        return dA_prev, dW, db

'''
Dropout layer

Combats overfitting by randomly setting activations to 0
Helps prevent co-adaptation of neurons
'''
class Dropout():
    def __init__(self, rate):
        self.trainable = False
        self.rate = rate
        self.mask = None

    # Randomly set activations to 0
    def forward(self, A, W, b):
        self.mask = np.where(np.random.rand(*A.shape) > self.rate, 1, 0)
        A = np.multiply(A, self.mask)

        return A, None
    
    # Set activation derivatives to 0 if they were set to 0 during forward propagation
    def backward(self, dA, A_prev, W, b, Z):
        dA_prev = np.multiply(dA, self.mask)

        return dA_prev, None, None
