# Handle conditional imports
def configure_imports(cuda):
    global np
    np = __import__('cupy' if cuda else 'numpy')

# Initialize parameters as zeros
# Bad - leads to symmetry, many neurons learn same features
def zeros(layer_sizes):
    return [{
        'W': np.zeros((layer_sizes[layer + 1], layer_sizes[layer])), # Zeros for weights
        'b': np.zeros((layer_sizes[layer + 1], 1)) # Zeros for biases
    } for layer in range(len(layer_sizes) - 1)]

# Initialize parameters as ones
# Bad - leads to symmetry, many neurons learn same features
def ones(layer_sizes):
    return [{
        'W': np.ones((layer_sizes[layer + 1], layer_sizes[layer])), # Ones for weights
        'b': np.ones((layer_sizes[layer + 1], 1)) # Ones for biases
    } for layer in range(len(layer_sizes) - 1)]

# Initialize parameters to normal random values
def normal(layer_sizes):
    return [{
        'W': np.random.randn(layer_sizes[layer + 1], layer_sizes[layer]),
        'b': np.random.randn(layer_sizes[layer + 1], 1)
    } for layer in range(len(layer_sizes) - 1)]

# Initialize parameters to uniform random values
def uniform(layer_sizes):
    return [{
        'W': np.random.uniform(-1, 1, (layer_sizes[layer + 1], layer_sizes[layer])),
        'b': np.random.uniform(-1, 1, (layer_sizes[layer + 1], 1))
    } for layer in range(len(layer_sizes) - 1)]

# Initialize parameters using Glorot (Xavier) normal initialization
def glorot_normal(layer_sizes):
    return [{
        'W': np.random.randn(layer_sizes[layer + 1], layer_sizes[layer]) * np.sqrt(2 / (layer_sizes[layer] + layer_sizes[layer + 1])),
        'b': np.zeros((layer_sizes[layer + 1], 1))
    } for layer in range(len(layer_sizes) - 1)]

# Glorot / Xavier uniform initialization
def glorot_uniform(layer_sizes):
    return [{
        'W': np.random.uniform(-1, 1, (layer_sizes[layer + 1], layer_sizes[layer])) * np.sqrt(2 / (layer_sizes[layer] + layer_sizes[layer + 1])),
        'b': np.zeros((layer_sizes[layer + 1], 1))
    } for layer in range(len(layer_sizes) - 1)]

def he_normal(layer_sizes):
    return [{
        'W': np.random.randn(layer_sizes[layer + 1], layer_sizes[layer]) * np.sqrt(2 / layer_sizes[layer]),
        'b': np.zeros((layer_sizes[layer + 1], 1))
    } for layer in range(len(layer_sizes) - 1)]

def he_uniform(layer_sizes):
    return [{
        'W': np.random.uniform(-1, 1, (layer_sizes[layer + 1], layer_sizes[layer])) * np.sqrt(2 / layer_sizes[layer]),
        'b': np.zeros((layer_sizes[layer + 1], 1))
    } for layer in range(len(layer_sizes) - 1)]
