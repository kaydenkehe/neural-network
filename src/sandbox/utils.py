# Handle conditional imports
def configure_imports(cuda):
    global np
    np = __import__('cupy' if cuda else 'numpy')

# Gradient checking (assuring that all the calculus is correct)
# TODO: Update to account for different arrangements of weights (e.g., for convolutional layers)
# TODO: Vectorize, somehow(?)
def gradient_check(model, X, Y, epsilon=1e-4):
    # Calculate actual gradient
    AL = model.forward(X.T)
    grad = model.backward(AL, Y.T)

    # Empty arrays for gradient and gradient approximation
    num_params = model.summary(print_table=False)
    grad_arr, grad_aprox_arr = np.zeros(num_params), np.zeros(num_params)

    # Loop through every parameter
    iter = 0
    for layer in range(len(model.layers)):
        for param_type in ['W', 'b']:
            for param in range(model.parameters[layer][param_type].shape[0]):

                # Calculate the cost with the parameter shifted by +epsilon
                model.parameters[layer][param_type][param][0] += epsilon
                AL_pe = model.forward(X.T)
                cost_pe = model.cost_type.forward(AL_pe, Y.T)

                # Calculate the cost with the parameter shifted by -epsilon
                model.parameters[layer][param_type][param][0] -= 2 * epsilon
                AL_ne = model.forward(X.T)
                cost_ne = model.cost_type.forward(AL_ne, Y.T)

                # Reset the parameter
                model.parameters[layer][param_type][param][0] += epsilon

                # Calculate the approximate parameter derivative w.r.t. cost
                grad_aprox_arr[iter] = (cost_pe - cost_ne) / (2 * epsilon)

                # Append actual gradient value to list (allows for Euclidean distance in later step)
                grad_arr[iter] = (grad[layer]['d' + param_type][param][0])

                iter += 1

            # Return representation of distance between actual and approximate gradients
            return np.linalg.norm(grad_arr - grad_aprox_arr) / np.linalg.norm(grad_arr + grad_aprox_arr)

# 1 if output > 0.5, 0 otherwise
def binary_round(Y):
    return np.where(Y > 0.5, 1, 0)
