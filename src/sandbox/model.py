import numpy as np

class Model:
    layers = []
    parameters = []
    

    def add(self, layer):
        self.layers.append(layer)


    # Predict given input values and weights / biases
    # TODO: Add support for other types of output
    def predict(self, X, type):
        if type == 'binary_classification':
            p = np.zeros((1, X.shape[1])) # Empty row vector for outputs
            probabilities, _ = self.model_forward(X) # Model outputs
            for i in range(0, probabilities.shape[1]):
                p[0, i] = 1 if probabilities[0, i] > 0.5 else 0 # Transform prediction into 1 or 0
            return p


    def train(self, X, Y, cost_type, learning_rate = 0.0075, epochs = 3000, print_cost=False):
        # TODO: Add support for different layer types
        layer_dims = [len(X)] + [layer.units for layer in self.layers]
        self.initialize_parameters(layer_dims) # Initialize random parameters
        costs = []

        # Loop through epochs
        for i in range(epochs):
            AL, caches = self.model_forward(X) # Forward propagate
            cost = cost_type.forward(AL, Y) # Calculate cost
            grads = self.model_backward(AL, Y, caches, cost_type) # Calculate gradient
            self.update_parameters(grads, learning_rate) # Update weights and biases
            
            if print_cost and i % 100 == 0 or i == epochs - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost))) # Optional, output progress

            if i % 100 == 0 or i == epochs:
                costs.append(cost) # Update costs list


    # Forward propagate through model
    def model_forward(self, X):
        caches = []
        A = X

        # Loop through hidden layers, calculating activations
        for layer in range(len(self.layers)):
            A, cache = self.layers[layer].forward(A, self.parameters[layer]['W'], self.parameters[layer]['b'])
            caches.append(cache)

        return A, caches


    # Find derivative with respect to each activation, weight, and bias
    def model_backward(self, AL, Y, caches, cost):
        grads = [None] * len(self.layers)
        Y = Y.reshape(AL.shape)
        dA_prev = cost.backward(AL, Y) # Find derivative of cost with respect to final activation
        
        # Find dA, dW, and db for all layers
        for layer in reversed(range(len(self.layers))):
            current_cache = caches[layer]
            dA_prev, dW, db = self.layers[layer].backward(dA_prev, current_cache)
            grads[layer] = {'dW': dW, 'db': db}
            
        return grads


    # Update parameters using gradient
    def update_parameters(self, grads, learning_rate):
        for layer in range(len(self.layers)): # Loop through weights and biases
            self.parameters[layer]['b'] -= learning_rate * grads[layer]['db'] # Update biases for layer
            self.parameters[layer]['W'] -= learning_rate * grads[layer]['dW'] # Update weights for layer
            

    # Initialize weights and biases
    def initialize_parameters(self, layer_dims):
        np.random.seed(1)
        self.parameters = [None] * len(self.layers)

        for layer in range(len(self.layers)):
            self.parameters[layer] = {
                'W': np.random.randn(layer_dims[layer + 1], layer_dims[layer]) / np.sqrt(layer_dims[layer]), # Gaussian random dist for weights
                'b': np.zeros((layer_dims[layer + 1], 1)) # Zeros for biases
            }
