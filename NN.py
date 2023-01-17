from os import chdir, path
from h5py import File
from PIL import Image
import numpy as np

chdir(path.dirname(path.abspath(__file__)))


# --- ACTIVATIONS, COSTS, DERIVATIVES ---

# Sigmoid (1 / (1 + e^-z))
def sigmoid(Z): return 1 / (1 + np.exp(-Z)), Z
def sigmoid_backward(dA, cache):
    s = 1/(1+np.exp(-cache))
    return dA * s * (1-s)

# ReLU (max(0,z))
def relu(Z): return np.maximum(0, Z), Z
def relu_backward(dA, cache):
    dZ = np.array(dA, copy=True) 
    dZ[cache <= 0] = 0
    return dZ

# Cross-Entropy (-1 / m * sum(yln(a) + (1 - y)ln(1 - a)))
def crossentropy(AL, Y): return np.squeeze(-1 / Y.shape[1] * np.sum(np.dot(Y, np.log(AL).T) + np.dot(1 - Y, np.log(1 - AL).T)))
def crossentropy_backward(AL, Y): return -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))


# --- MISC FUNCTIONS ---


# Load input data
def load_data():
    train_dataset = File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # Train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # Train set labels

    test_dataset = File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # Test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # Test set labels

    classes = np.array(test_dataset["list_classes"][:]) # List of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    # Flatten and normalize
    train_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T / 255
    test_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T / 255
    
    return train_x, train_set_y_orig, test_x, test_set_y_orig, classes

# Predict given input values and weights / biases
def predict(X, parameters):
    p = np.zeros((1,X.shape[1])) # Empty row vector for outputs
    probabilities, _ = model_forward(X, parameters) # Model outputs
    for i in range(0, probabilities.shape[1]): p[0, i] = 1 if probabilities[0, i] > 0.5 else 0 # Transform prediction into 1 or 0
    return p


# --- MODEL FUNCTIONS ---


# Initialize weights and biases
def initialize_parameters(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) # Gaussian random dist for weights
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1)) # Zeros for biases
        
    return parameters 

# Calculate linear step of neuron activation (W.T * A + b)
def linear_activation_forward(A_prev, W, b, activation):
    Z = np.dot(W, A_prev) + b # Compute Z
    A, activation_cache = eval(activation)(Z) # Compute A using the given activation function
    cache = ((A_prev, W, b), activation_cache) # Cache for backprop

    return A, cache

# Forward propagation
def model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2 # Number of layers (excluding input)
    
    # Loop through hidden layers, calculating activations
    for l in range(1, L):
        A, cache = linear_activation_forward(A, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
        caches.append(cache)
    
    # Calculate activation(s) for output layer
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
    caches.append(cache)

    return AL, caches

# Find derivative with respect to weights, biases, and activations for a particular layer
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    dZ = eval(activation + '_backward')(dA, activation_cache) # Evaluate dZ using the derivative of sigmoid or relu
    A_prev, W, _ = linear_cache
    m = A_prev.shape[1]

    dW = 1 / m * np.dot(dZ, A_prev.T) # Calculate derivative with respect to weights
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True) # Calculate derivative with respect to biases
    dA_prev = np.dot(W.T, dZ) # Calculate derivative with respect to the activation of the previous layer

    return dA_prev, dW, db

# Find derivative with respect to each activation, weight, and bias
def model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) 
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = crossentropy_backward(AL, Y) # Find derivative of cost with respect to final activation
    
    # Find dAL, dWL, and dbL (partial derivatives for the final layer)
    current_cache = caches[L - 1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, 'sigmoid')
    grads['dA' + str(L-1)] = dA_prev_temp
    grads['dW' + str(L)] = dW_temp
    grads['db' + str(L)] = db_temp
    
    # Find dA, dW, and db for all hidden layers
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev_temp, current_cache, 'relu')
        grads['dA' + str(l)] = dA_prev_temp
        grads['dW' + str(l + 1)] = dW_temp
        grads['db' + str(l + 1)] = db_temp
        
    return grads

# Update parameters using gradient
def update_parameters(params, grads, learning_rate):
    parameters = params.copy()
    for l in range(len(parameters) // 2): # Loop through weights and biases
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)] # Update weights for layer
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)] # Update biases for layer
        
    return parameters

# Train model parameters
def train_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    parameters = initialize_parameters(layers_dims) # Initialize random parameters
    costs = []

    # Loop through iterations
    for i in range(0, num_iterations):
        AL, caches = model_forward(X, parameters) # Forward propagate
        cost = crossentropy(AL, Y) # Calculate cost
        grads = model_backward(AL, Y, caches) # Calculate gradient
        parameters = update_parameters(parameters, grads, learning_rate) # Update weights and biases
        
        if print_cost and i % 100 == 0 or i == num_iterations - 1: print("Cost after iteration {}: {}".format(i, np.squeeze(cost))) # Optional, output progress
        if i % 100 == 0 or i == num_iterations: costs.append(cost) # Update costs list
    
    return parameters, costs


# --- USE MODEL ---


train_x, train_y, test_x, test_y, classes = load_data() # Load data
layers_dims = [12288, 20, 7, 5, 1] # Create model dimensions, each item in list represents node count at that layer (l = 0 to l = L)

parameters, costs = train_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True) # Train model on data, get optimized parameters

pred_train = predict(train_x, parameters) # Get model accuracy on training data
print("Training Accuracy: "  + str(np.sum((pred_train == train_y)/train_x.shape[1])))
pred_test = predict(test_x, parameters) # Get model accuracy on testing data
print("Testing Accuracy: "  + str(np.sum((pred_test == test_y)/test_x.shape[1])))


# Use model on custom image
my_image = 'not_cat.jpg' 
my_label_y = [0] # 1: cat, 0: non-cat
num_px = 64

image = np.array(Image.open(my_image).resize((num_px, num_px))) / 255 # Resize and normalize image, cast to NumPy array
image = image.reshape((1, num_px * num_px * 3)).T # Flatten image array
my_predicted_image = predict(image, parameters) # Predict custom image
print("Custom Image Prediction: " + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8"))
