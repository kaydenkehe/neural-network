import numpy as np

class Model:
    layers = [] # Each item is a layer object
    parameters = [] # Each item is a dictionary with a 'W' and 'b' matrix for the weights and biases respectively
    caches = [] # Each item is a dictionary with the 'A_prev', 'W', 'b', and 'Z' values for the layer
    costs = [] # Each item is the cost for the epoch

    def add(self, layer):
        self.layers.append(layer)

    # Predict given input values and weights / biases
    def predict(self, X, prediction_type=lambda x: x):
        prediction = self.model_forward(X) # Model outputs
        return prediction_type(prediction)

    # Configure model parameters
    def configure(self, cost_type, learning_rate = 0.0075, epochs = 3000):
        self.cost_type = cost_type
        self.learning_rate = learning_rate
        self.epochs = epochs

    # Train model
    def train(self, X, Y, verbose=False):
        self.initialize_parameters(input_size=X.shape[0]) # Initialize random parameters
        self.costs = []

        # Loop through epochs
        for i in range(self.epochs + 1):
            AL = self.model_forward(X) # Forward propagate
            cost = self.cost_type.forward(AL, Y) # Calculate cost
            grads = self.model_backward(AL, Y, self.cost_type) # Calculate gradient
            self.update_parameters(grads, self.learning_rate) # Update weights and biases
            self.costs.append(cost) # Update costs list
            
            if verbose and i % 100 == 0 or i == self.epochs:
                print("Cost after epoch {}: {}".format(i, np.squeeze(cost))) # Optional, output progress

    # Forward propagate through model
    def model_forward(self, A):
        self.caches = []

        # Loop through hidden layers, calculating activations
        for layer in range(len(self.layers)):
            A_prev = A
            A, Z = self.layers[layer].forward(A_prev, **self.parameters[layer])
            self.caches.append({
                'A_prev': A_prev,
                "W": self.parameters[layer]['W'],
                "b": self.parameters[layer]['b'],
                "Z": Z
            })

        return A

    # Find derivative with respect to each activation, weight, and bias
    def model_backward(self, AL, Y, cost):
        grads = [None] * len(self.layers)
        dA_prev = cost.backward(AL, Y.reshape(AL.shape)) # Find derivative of cost with respect to final activation
        
        # Find dA, dW, and db for all layers
        for layer in reversed(range(len(self.layers))):
            cache = self.caches[layer]
            dA_prev, dW, db = self.layers[layer].backward(dA_prev, **cache)
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
        