import inspect
import numpy as np
import sandbox

# Handle CuPy import
def import_cupy():
    global np
    np = __import__('cupy')

# Handle CuPy import for all modules
def configure_cupy():
    for module, _ in inspect.getmembers(sandbox, inspect.ismodule):
        exec(f'sandbox.{module}.import_cupy()')

# Gradient checking
def gradient_check(model, X, Y, epsilon=1e-4):

    # Calculate actual gradient
    AL = model.forward(X)
    grad = model.backward(AL, Y)

    # Declare empty arrays for gradient and gradient approximation
    num_params = model.summary(print_table=False)
    grad_arr, grad_aprox_arr = np.zeros(num_params), np.zeros(num_params)

    # Loop through every parameter
    iter = 0
    for layer in range(len(model.layers)):
        for param_type in ['W', 'b']:
            for param in range(model.parameters[layer][param_type].shape[0]):

                # Calculate the cost with the parameter shifted by +epsilon
                model.parameters[layer][param_type][param][0] += epsilon
                AL_pe = model.forward(X)
                cost_pe = model.cost_type.forward(AL_pe, Y)

                # Calculate the cost with the parameter shifted by -epsilon
                model.parameters[layer][param_type][param][0] -= 2 * epsilon
                AL_ne = model.forward(X)
                cost_ne = model.cost_type.forward(AL_ne, Y)

                # Reset the parameter
                model.parameters[layer][param_type][param][0] += epsilon

                # Calculate the approximate parameter derivative w.r.t. cost
                grad_aprox_arr[iter] = (cost_pe - cost_ne) / (2 * epsilon)

                # Append actual gradient value to list (allows for Euclidean distance in later step)
                grad_arr[iter] = grad[layer]['d' + param_type][param][0]

                iter += 1

            # Return representation of distance between actual and approximate gradients
            return np.linalg.norm(grad_arr - grad_aprox_arr) / (np.linalg.norm(grad_arr) + np.linalg.norm(grad_aprox_arr))

# 1 if output > 0.5, 0 otherwise
def binary_round(Y):
    return np.squeeze(np.where(Y > 0.5, 1, 0))

# Calculate accuracy for binary or multiclass classification
def evaluate(Y_pred, Y):
    Y_pred = binary_round(Y_pred) if Y_pred.shape[1] == 1 else argmax(Y_pred)
    Y = binary_round(Y) if Y.shape[1] == 1 else argmax(Y)
    return np.mean(np.where(Y_pred == Y, 1, 0))

# One-hot encode labels
def one_hot(Y, num_classes):
    return np.eye(num_classes)[Y.reshape(-1)]

# Argmax - return index of highest value in each row
def argmax(Y):
    return np.squeeze(np.argmax(Y, axis=1))
