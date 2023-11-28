import cupy as np # Use Cuda
from h5py import File
from os import chdir, path
from sandbox import model, layers, activations, predictions

chdir(path.dirname(path.abspath(__file__)))

# Create model
model = model.Model(cuda=True)
model.add(layers.Dense(units=20, activation=activations.ReLU()))
model.add(layers.Dense(units=7, activation=activations.ReLU()))
model.add(layers.Dense(units=5, activation=activations.ReLU()))
model.add(layers.Dense(units=1, activation=activations.Sigmoid()))

# Load pre-trained parameters
model.load('parameters.json')

# Load test data
test_dataset = File('test_catvnoncat.h5', 'r')
test_x = np.array(test_dataset['test_set_x'][:]) # Test set features
test_y = np.array(test_dataset['test_set_y'][:]) # Test set labels
test_y = test_y.reshape((test_y.shape[0], 1))
test_x = test_x.reshape(test_x.shape[0], -1) / 255

# Evaluate model
pred_test = model.predict(test_x, prediction_type=predictions.binary_classification) # Get model accuracy on testing data
print('Testing Accuracy: '  + str(np.sum((pred_test == test_y)/test_x.shape[0])))
