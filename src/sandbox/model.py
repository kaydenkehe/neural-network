import inspect
import json
from sandbox import initializers, optimizers
from prettytable import PrettyTable

# Handle conditional imports
def configure_imports(cuda):
    global np
    np = __import__('cupy' if cuda else 'numpy')

class Model:

    # Initialize model
    def __init__(self, cuda=False):
        self.layers = [] # Each item is a layer object
        self.parameters = [] # Each item is a dictionary with a 'W' and 'b' matrix for the weights and biases respectively
        self.caches = [] # Each item is a dictionary with the 'A_prev', 'W', 'b', and 'Z' values for the layer - Used in backprop
        self.costs = [] # Each item is the cost for the epoch

        # Configure all scripts to run on either CuPy or NumPy
        import sandbox
        for module, _ in inspect.getmembers(sandbox, inspect.ismodule):
            exec(f'sandbox.{module}.configure_imports(cuda)')

    # Add layer to model
    def add(self, layer):
        self.layers.append(layer)

    # Configure model settings
    def configure(self, cost_type, input_size, optimizer=optimizers.SGD(), initializer=initializers.he):
        self.cost_type = cost_type
        self.initializer = initializer
        self.optimizer = optimizer

        self.initialize_parameters(input_size) # Initialize parameters
        self.parameters = self.optimizer.configure(self.parameters, self.layers) # Add optimizer velocities, etc.

    # Train model
    def train(self, X, Y, learning_rate=0.001, epochs=1000, batch_size=None, verbose=False):
        self.optimizer.learning_rate = learning_rate
        m = X.shape[0]
        X, Y = X.T, Y.T # Transpose X and Y to match shape of weights and biases
        if not batch_size: batch_size = m # Default to batch GD

        # Loop through epochs
        for i in range(1, epochs + 1):
            # Shuffle data, split into batches
            X, Y = self.shuffle(X, Y)
            X_batches = np.array_split(X, m // batch_size, axis=1)
            Y_batches = np.array_split(Y, m // batch_size, axis=1)

            # Loop through batches
            for X_batch, Y_batch in zip(X_batches, Y_batches):
                AL = self.forward(X_batch) # Forward propagate
                cost = self.cost_type.forward(AL, Y_batch) # Calculate cost
                grad = self.backward(AL, Y_batch) # Calculate gradient
                self.parameters = self.optimizer.update(self.parameters, self.layers, grad) # Update weights and biases

                self.costs.append(cost) # Update costs list
            if verbose and (i % (epochs // 10) == 0 or i == epochs):
                print(f"Cost on epoch {i}: {round(cost.item(), 5)}") # Optional, output progress

    # Initialize weights and biases
    def initialize_parameters(self, input_size):
        # Initialize parameters using initializer function
        layer_sizes = [input_size] + [layer.units for layer in self.layers if layer.trainable]
        self.parameters = self.initializer(layer_sizes)

        # For non-trainable layers, we want to set the parameters to empty arrays
        for layer in range(len(self.layers)):
            if not self.layers[layer].trainable:
                self.parameters.insert(layer, {'W': np.array([]), 'b': np.array([])})

    # Shuffle data
    def shuffle(self, X, Y):
        assert X.shape[1] == Y.shape[1]
        permutation = np.random.permutation(X.shape[1])
        return X[:, permutation], Y[:, permutation]

    # Forward propagate through model
    def forward(self, A, train=True):
        self.caches = []

        # Exclude non-trainable layers (like dropout) when not training
        layers = [i for i in range(len(self.layers)) if self.layers[i].trainable or train]
        
        # Loop through hidden layers, calculating activations
        for layer in layers:
            A_prev = A
            A, Z = self.layers[layer].forward(A_prev, self.parameters[layer]['W'], self.parameters[layer]['b'])
            self.caches.append({
                'A_prev': A_prev,
                "W": self.parameters[layer]['W'],
                "b": self.parameters[layer]['b'],
                "Z": Z
            })

        return A

    # Find derivative with respect to each activation, weight, and bias
    def backward(self, AL, Y):
        grad = [None] * len(self.layers)
        dA_prev = self.cost_type.backward(AL, Y.reshape(AL.shape)) # Find derivative of cost with respect to final activation
        
        # Find dA, dW, and db for all layers
        for layer in reversed(range(len(self.layers))):
            cache = self.caches[layer]
            dA_prev, dW, db = self.layers[layer].backward(dA_prev, **cache)
            grad[layer] = {'dW': dW, 'db': db}
            
        return grad

    # Predict given input
    def predict(self, X):
        return self.forward(X.T, train=False).T

    # Save parameters to JSON file
    def save(self, name='parameters.json', dir=''):
        jsonified_params = [
            {'W': layer['W'].tolist(), 'b': layer['b'].tolist()}
            for layer in [layer for layer in self.parameters if layer['W'].size != 0] # Exclude empty arrays (dropout layers, for example)
        ]
        with open(dir + name, 'w') as file:
            json.dump(jsonified_params, file)

    # Load parameters from JSON file
    def load(self, name='parameters.json', dir=''):
        with open(dir + name, 'r') as file:
            jsonified_params = json.load(file)
        self.parameters = [{'W': np.array(layer['W']), 'b': np.array(layer['b'])} for layer in jsonified_params]

    # Print model summary
    def summary(self, print_table=True):
        # Get number of parameters in each layer
        num_params = [layer['W'].size + layer['b'].size for layer in self.parameters]

        # Create table
        table = PrettyTable(['Layer type', 'Parameters'])
        for idx, layer in enumerate(self.layers):
            table.add_row([type(layer).__name__, num_params[idx]])

        # Print summary
        if print_table:
            print(table)
            print("Total parameters:", sum(num_params))

        return sum(num_params)
