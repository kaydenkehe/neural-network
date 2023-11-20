import numpy as np

class Model:
    layers = []
    parameters = {}
    

    def add(self, layer):
        self.layers.append(layer)


    # Predict given input values and weights / biases
    # TODO: Add support for other types of output
    def predict(self, X):
        p = np.zeros((1, X.shape[1])) # Empty row vector for outputs
        probabilities, _ = self.model_forward(X) # Model outputs
        for i in range(0, probabilities.shape[1]):
            p[0, i] = 1 if probabilities[0, i] > 0.5 else 0 # Transform prediction into 1 or 0
        return p


    def train(self, X, Y, cost_function, learning_rate = 0.0075, epochs = 3000, print_cost=False):
        # TODO: Add support for different layer types
        layer_dims = [len(X)] + [layer.units for layer in self.layers]
        self.initialize_parameters(layer_dims) # Initialize random parameters
        costs = []

        # Loop through epochs
        for i in range(epochs):
            AL, caches = self.model_forward(X) # Forward propagate
            cost = cost_function.forward(AL, Y) # Calculate cost
            grads = self.model_backward(AL, Y, caches, cost_function) # Calculate gradient
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
            A, cache = self.layers[layer].forward(A, self.parameters['W' + str(layer + 1)], self.parameters['b' + str(layer + 1)])
            caches.append(cache)

        return A, caches


    # Find derivative with respect to each activation, weight, and bias
    def model_backward(self, AL, Y, caches, cost):
        grads = {}
        Y = Y.reshape(AL.shape)
        dA_prev = cost.backward(AL, Y) # Find derivative of cost with respect to final activation
        
        # Find dA, dW, and db for all layers
        for layer in reversed(range(len(self.layers))):
            current_cache = caches[layer]
            dA_prev, dW_temp, db_temp = self.layers[layer].backward(dA_prev, current_cache)
            grads['dA' + str(layer)] = dA_prev
            grads['dW' + str(layer + 1)] = dW_temp
            grads['db' + str(layer + 1)] = db_temp
            
        return grads


    # Update parameters using gradient
    def update_parameters(self, grads, learning_rate):
        for l in range(len(self.parameters) // 2): # Loop through weights and biases
            self.parameters['b' + str(l + 1)] -= learning_rate * grads['db' + str(l + 1)] # Update biases for layer
            self.parameters['W' + str(l + 1)] -= learning_rate * grads['dW' + str(l + 1)] # Update weights for layer
            

    # Initialize weights and biases
    def initialize_parameters(self, layer_dims):
        np.random.seed(1)
        self.parameters = {}

        for layer in range(1, len(self.layers) + 1):
            self.parameters['W' + str(layer)] = np.random.randn(layer_dims[layer], layer_dims[layer - 1]) / np.sqrt(layer_dims[layer - 1]) # Gaussian random dist for weights
            self.parameters['b' + str(layer)] = np.zeros((layer_dims[layer], 1)) # Zeros for biases
