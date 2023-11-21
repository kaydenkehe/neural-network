import numpy as np

class Model:
    layers = []
    parameters = []

    def add(self, layer):
        self.layers.append(layer)

    # Predict given input values and weights / biases
    def predict(self, X, type):
        if type == 'binary_classification':
            p = np.zeros((1, X.shape[1])) # Empty row vector for outputs
            probabilities, _ = self.model_forward(X) # Model outputs
            for i in range(0, probabilities.shape[1]):
                p[0, i] = 1 if probabilities[0, i] > 0.5 else 0 # Transform prediction into 1 or 0
            return p

    def configure(self, cost_type, learning_rate = 0.0075, epochs = 3000):
        self.cost_type = cost_type
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self, X, Y, verbose=False):
        self.initialize_parameters(input_size=X.shape[0]) # Initialize random parameters
        costs = []

        # Loop through epochs
        for i in range(self.epochs):
            AL, caches = self.model_forward(X) # Forward propagate
            cost = self.cost_type.forward(AL, Y) # Calculate cost
            grads = self.model_backward(AL, Y, caches, self.cost_type) # Calculate gradient
            self.update_parameters(grads, self.learning_rate) # Update weights and biases
            
            if verbose and i % 100 == 0 or i == self.epochs - 1:
                print("Cost after epoch {}: {}".format(i, np.squeeze(cost))) # Optional, output progress

            if i % 100 == 0 or i == self.epochs:
                costs.append(cost) # Update costs list

    # Forward propagate through model
    def model_forward(self, X):
        caches = []
        A = X

        # Loop through hidden layers, calculating activations
        for layer in range(len(self.layers)):
            A, cache = self.layers[layer].forward(A, **self.parameters[layer])
            caches.append(cache)

        return A, caches

    # Find derivative with respect to each activation, weight, and bias
    def model_backward(self, AL, Y, caches, cost):
        grads = [None] * len(self.layers)
        dA_prev = cost.backward(AL, Y.reshape(AL.shape)) # Find derivative of cost with respect to final activation
        
        # Find dA, dW, and db for all layers
        for layer in reversed(range(len(self.layers))):
            cache = caches[layer]
            dA_prev, dW, db = self.layers[layer].backward(dA_prev, cache)
            grads[layer] = {'dW': dW, 'db': db}
            
        return grads

    # Update parameters using gradient
    def update_parameters(self, grads, learning_rate):
        for layer in range(len(self.layers)):
            self.parameters[layer]['b'] -= learning_rate * grads[layer]['db'] # Update biases for layer
            self.parameters[layer]['W'] -= learning_rate * grads[layer]['dW'] # Update weights for layer

    # Initialize weights and biases
    def initialize_parameters(self, input_size):
        # We start on layer -1 to handle the parameters connecting the input layer to the first hidden layer
        self.parameters = [{
            'W': np.random.randn(self.layers[layer + 1].units, self.layers[layer].units if layer != -1 else input_size) / np.sqrt(self.layers[layer].units if layer != -1 else input_size), # Gaussian random dist for weights
            'b': np.zeros((self.layers[layer + 1].units, 1)) # Zeros for biases
        } for layer in range(-1, len(self.layers) - 1)]
        