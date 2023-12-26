# Handle conditional imports
def configure_imports(cuda):
    global np
    np = __import__('cupy' if cuda else 'numpy')

'''
Every optimizer class includes three methods:
    - __init__: Initialize optimizer parameters
    - configure: Add necessary additional parameters to existing model parameters (e.g. velocities, learning rate scalers)
    - update: Update additional parameters and model parameters
'''

'''
SGD (Stochastic Gradient Descent) optimizer

SGD is the default / traditional optimizer
It updates parameters using the gradient and learning rate alone
'''
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

'''
Momentum optimizer

The basic momentum optimizer smooths the path of the parameters down the cost function
Instead of updating parameters based on a single gradient, it updates parameters based on a rolling average of the gradients from every timestep

In a physical sense, using the ball rolling down a hill analogy (the hill being the cost function and the ball being the parameters),
as opposed to SGD, which updates the ball's position based on the current slope of the hill,
momentum updates the velocity of the ball, and the velocity updates the ball's position
'''
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

'''
AdaGrad (Adaptive Gradient) optimizer

At a high level, AdaGrad implements learning rate decay by: 
- Assigning every parameter its own learning rate
- Penalizing the learning rate of parameters that have substantially changed

The intuition here being that if a parameter has undergone substantial change,
it's probably approaching a local minimum and should be updated more slowly

The epsilon term helps ensure that we never divide by zero
'''
class AdaGrad:
         
    def __init__(self, learning_rate=0.001, epsilon=1e-8):
        self.learning_rate = learning_rate
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
                    parameters[layer]['SdW'] += np.square(grad[layer]['dW']) # Update weight LR scalers
                    parameters[layer]['Sdb'] += np.square(grad[layer]['db']) # Update bias LR scalers
                    
                    parameters[layer]['W'] -= self.learning_rate * grad[layer]['dW'] / (np.sqrt(parameters[layer]['SdW']) + self.epsilon) # Update weights
                    parameters[layer]['b'] -= self.learning_rate * grad[layer]['db'] / (np.sqrt(parameters[layer]['Sdb']) + self.epsilon) # Update biases

        return parameters

'''
RMSProp (Root Mean Squared Propagation) optimizer

RMSProp essentially assigns each parameter a learning rate based on a rolling average of squared gradients
The epsilon term helps ensure that we never divide by zero

RMSProp is an offshoot of Adagrad, using a decaying average instead of a cumulative sum
'''
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
                    parameters[layer]['SdW'] = self.beta * parameters[layer]['SdW'] + (1 - self.beta) * np.square(grad[layer]['dW']) # Update weight LR scalers
                    parameters[layer]['Sdb'] = self.beta * parameters[layer]['Sdb'] + (1 - self.beta) * np.square(grad[layer]['db']) # Update bias LR scalers
                    
                    parameters[layer]['W'] -= self.learning_rate * grad[layer]['dW'] / (np.sqrt(parameters[layer]['SdW']) + self.epsilon) # Update weights
                    parameters[layer]['b'] -= self.learning_rate * grad[layer]['db'] / (np.sqrt(parameters[layer]['Sdb']) + self.epsilon) # Update biases

        return parameters

'''
Adam (Adaptive Moment Estimation) optimizer

Adam combines momentum and RMSProp, gaining the benefits from both approaches
Adam is the most widely used optimizer in DL
The epsilon term helps ensure that we never divide by zero
'''
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
                        parameters[layer]['SdW'] = self.beta2 * parameters[layer]['SdW'] + (1 - self.beta2) * np.square(grad[layer]['dW']) # Update weight LR scalers
                        parameters[layer]['Sdb'] = self.beta2 * parameters[layer]['Sdb'] + (1 - self.beta2) * np.square(grad[layer]['db']) # Update bias LR scalers
                        
                        parameters[layer]['W'] -= self.learning_rate * parameters[layer]['VdW'] / (np.sqrt(parameters[layer]['SdW']) + self.epsilon) # Update weights
                        parameters[layer]['b'] -= self.learning_rate * parameters[layer]['Vdb'] / (np.sqrt(parameters[layer]['Sdb']) + self.epsilon) # Update biases
    
            return parameters
