# Handle conditional imports
def configure_imports(cuda):
    global np
    np = __import__('cupy' if cuda else 'numpy')

# SGD optimizer (default)
class SGD:

    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def configure(self, parameters, layers):
        return parameters

    def update(self, parameters, layers, grad):    
        for layer in range(len(layers)):
                if layers[layer].trainable:
                    parameters[layer]['W'] -= self.learning_rate * grad[layer]['dW'] # Update weights
                    parameters[layer]['b'] -= self.learning_rate * grad[layer]['db'] # Update biases

        return parameters

# Momentum optimizer
class Momentum:

    def __init__(self, learning_rate=0.001, beta=0.9):
        self.learning_rate = learning_rate
        self.beta = beta

    def configure(self, parameters, layers):
        for layer in range(len(layers)):
            if layers[layer].trainable:
                parameters[layer]['VdW'] = np.zeros(parameters[layer]['W'].shape) # Initialize weight velocities
                parameters[layer]['Vdb'] = np.zeros(parameters[layer]['b'].shape) # Initialize bias velocities

        return parameters

    def update(self, parameters, layers, grad):
        for layer in range(len(layers)):
                if layers[layer].trainable:
                    parameters[layer]['VdW'] = self.beta * parameters[layer]['VdW'] + (1 - self.beta) * grad[layer]['dW'] # Update weight velocities
                    parameters[layer]['Vdb'] = self.beta * parameters[layer]['Vdb'] + (1 - self.beta) * grad[layer]['db'] # Update bias velocities
                    parameters[layer]['W'] -= self.learning_rate * parameters[layer]['VdW'] # Update weights
                    parameters[layer]['b'] -= self.learning_rate * parameters[layer]['Vdb'] # Update biases

        return parameters

# RMSProp optimizer (adaptive learning rates)
class RMSProp:
         
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon

    def configure(self, parameters, layers):
        for layer in range(len(layers)):
            if layers[layer].trainable:
                parameters[layer]['SdW'] = np.zeros(parameters[layer]['W'].shape) # Initialize weight LR scalers
                parameters[layer]['Sdb'] = np.zeros(parameters[layer]['b'].shape) # Initialize bias LR scalers

        return parameters

    def update(self, parameters, layers, grad):
        for layer in range(len(layers)):
                if layers[layer].trainable:
                    parameters[layer]['SdW'] = self.beta * parameters[layer]['SdW'] + (1 - self.beta) * np.square(grad[layer]['dW']) # Update weight velocities
                    parameters[layer]['Sdb'] = self.beta * parameters[layer]['Sdb'] + (1 - self.beta) * np.square(grad[layer]['db']) # Update bias velocities
                    parameters[layer]['W'] -= self.learning_rate * grad[layer]['dW'] / (np.sqrt(parameters[layer]['SdW']) + self.epsilon) # Update weights
                    parameters[layer]['b'] -= self.learning_rate * grad[layer]['db'] / (np.sqrt(parameters[layer]['Sdb']) + self.epsilon) # Update biases

        return parameters

# Adam optimizer (momentum with adaptive learning rates)
class Adam:
         
        def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
            self.learning_rate = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
    
        def configure(self, parameters, layers):
            for layer in range(len(layers)):
                if layers[layer].trainable:
                    parameters[layer]['VdW'] = np.zeros(parameters[layer]['W'].shape) # Initialize weight velocities
                    parameters[layer]['Vdb'] = np.zeros(parameters[layer]['b'].shape) # Initialize bias velocities
                    parameters[layer]['SdW'] = np.zeros(parameters[layer]['W'].shape) # Initialize weight LR scalers
                    parameters[layer]['Sdb'] = np.zeros(parameters[layer]['b'].shape) # Initialize bias LR scalers
    
            return parameters
    
        def update(self, parameters, layers, grad):
            for layer in range(len(layers)):
                    if layers[layer].trainable:
                        parameters[layer]['VdW'] = self.beta1 * parameters[layer]['VdW'] + (1 - self.beta1) * grad[layer]['dW'] # Update weight velocities
                        parameters[layer]['Vdb'] = self.beta1 * parameters[layer]['Vdb'] + (1 - self.beta1) * grad[layer]['db'] # Update bias velocities
                        parameters[layer]['SdW'] = self.beta2 * parameters[layer]['SdW'] + (1 - self.beta2) * np.square(grad[layer]['dW']) # Update weight velocities
                        parameters[layer]['Sdb'] = self.beta2 * parameters[layer]['Sdb'] + (1 - self.beta2) * np.square(grad[layer]['db']) # Update bias velocities
                        parameters[layer]['W'] -= self.learning_rate * parameters[layer]['VdW'] / (np.sqrt(parameters[layer]['SdW']) + self.epsilon) # Update weights
                        parameters[layer]['b'] -= self.learning_rate * parameters[layer]['Vdb'] / (np.sqrt(parameters[layer]['Sdb']) + self.epsilon) # Update biases
    
            return parameters
