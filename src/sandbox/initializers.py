# Handle conditional imports
def configure_imports(cuda):
    global np
    np = __import__('cupy' if cuda else 'numpy')

# Initialize weights and biases to zeros
# Bad - leads to symmetry, many neurons learn same features
def zeros(layer_sizes):
    return [{
        'W': np.zeros((layer_sizes[layer + 1], layer_sizes[layer])), # Zeros for weights
        'b': np.zeros((layer_sizes[layer + 1], 1)) # Zeros for biases
    } for layer in range(len(layer_sizes) - 1)]

# Initialize weights and biases to ones
# Bad - leads to symmetry, many neurons learn same features
def ones(layer_sizes):
    return [{
        'W': np.ones((layer_sizes[layer + 1], layer_sizes[layer])), # Ones for weights
        'b': np.ones((layer_sizes[layer + 1], 1)) # Ones for biases
    } for layer in range(len(layer_sizes) - 1)]

# Initialize weights and biases to normal random values
def normal(layer_sizes):
    return [{
        'W': np.random.randn(layer_sizes[layer + 1], layer_sizes[layer]),
        'b': np.random.randn(layer_sizes[layer + 1], 1)
    } for layer in range(len(layer_sizes) - 1)]

# Initialize weights and biases to uniform random values
def uniform(layer_sizes):
    return [{
        'W': np.random.uniform(-1, 1, (layer_sizes[layer + 1], layer_sizes[layer])),
        'b': np.random.uniform(-1, 1, (layer_sizes[layer + 1], 1))
    } for layer in range(len(layer_sizes) - 1)]

# Initialize weights and biases using Glorot normal initialization ()
def glorot_normal(layer_sizes):
    return [{
        'W': np.random.randn(layer_sizes[layer + 1], layer_sizes[layer]) * np.sqrt(2 / (layer_sizes[layer] + layer_sizes[layer + 1])),
        'b': np.zeros((layer_sizes[layer + 1], 1))
    } for layer in range(len(layer_sizes) - 1)]

# The gist of the Xavier and He initializer functions are from Andrew Ng's course,
# but these seem to be at odds with some other things I've read online,
# so I may update them at a later date.

# Initialize weights and biases using Xavier (a.k.a. Glorot) initialization
# Good for Sigmoid, Tanh
def xavier(layer_sizes):
    return [{
        'W': np.random.randn(layer_sizes[layer + 1], layer_sizes[layer]) / np.sqrt(layer_sizes[layer]),
        'b': np.zeros((layer_sizes[layer + 1], 1))
    } for layer in range(len(layer_sizes) - 1)]

# Initialize weights and biases using He initialization
# Good for ReLU
def he(layer_sizes):
    return [{
        'W': np.random.randn(layer_sizes[layer + 1], layer_sizes[layer]) * np.sqrt(2 / layer_sizes[layer]),
        'b': np.zeros((layer_sizes[layer + 1], 1))
    } for layer in range(len(layer_sizes) - 1)]
