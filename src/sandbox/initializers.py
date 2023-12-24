# Handle conditional imports
def configure_imports(cuda):
    global np
    np = __import__('cupy' if cuda else 'numpy')

# Initialize parameters as zeros
# Bad - leads to symmetry, many neurons learn same features
def zeros(prev_layer_size, layer_size):
    return {
        'W': np.zeros((layer_size, prev_layer_size)), # Zeros for weights
        'b': np.zeros((layer_size, 1)) # Zeros for biases
    }

# Initialize parameters as ones
# Bad - leads to symmetry, many neurons learn same features
def ones(prev_layer_size, layer_size):
    return {
        'W': np.ones((layer_size, prev_layer_size)), # Ones for weights
        'b': np.ones((layer_size, 1)) # Ones for biases
    }

# Initialize parameters following normal random distribution
def normal(prev_layer_size, layer_size):
    return {
        'W': np.random.randn(layer_size, prev_layer_size), # Random normal for weights
        'b': np.zeros((layer_size, 1)) # Zeros for biases
    }

# Initialize parameters following uniform random distribution
def uniform(prev_layer_size, layer_size):
    return {
        'W': np.random.uniform(-1, 1, (layer_size, prev_layer_size)), # Random uniform for weights
        'b': np.zeros((layer_size, 1)) # Zeros for biases
    }

# Initialize parameters following Glorot / Xavier normal distribution
def glorot_normal(prev_layer_size, layer_size):
    return {
        'W': np.random.randn(layer_size, prev_layer_size) * np.sqrt(2 / (layer_size + prev_layer_size)), # Glorot normal for weights
        'b': np.zeros((layer_size, 1)) # Zeros for biases
    }

# Initialize parameters following Glorot / Xavier uniform distribution
def glorot_uniform(prev_layer_size, layer_size):
    limit = np.sqrt(6 / (layer_size + prev_layer_size))
    return {
        'W': np.random.uniform(-limit, limit, (layer_size, prev_layer_size)), # Glorot uniform for weights
        'b': np.zeros((layer_size, 1)) # Zeros for biases
    }

# Initialize parameters following He normal distribution
def he_normal(prev_layer_size, layer_size):
    return {
        'W': np.random.randn(layer_size, prev_layer_size) * np.sqrt(2 / prev_layer_size), # He normal for weights
        'b': np.zeros((layer_size, 1)) # Zeros for biases
    }

# Initialize parameters following He uniform distribution
def he_uniform(prev_layer_size, layer_size):
    limit = np.sqrt(6 / prev_layer_size)
    return {
        'W': np.random.uniform(-limit, limit, (layer_size, prev_layer_size)), # He uniform for weights
        'b': np.zeros((layer_size, 1)) # Zeros for biases
    }
