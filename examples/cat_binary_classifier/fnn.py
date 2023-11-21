import numpy as np
from h5py import File
from os import chdir, path
from PIL import Image
from sandbox import model, layers, activations, costs

chdir(path.dirname(path.abspath(__file__)))

# Load data
train_dataset = File('train_catvnoncat.h5', 'r')
train_x = np.array(train_dataset['train_set_x'][:]) # Train set features
train_y = np.array(train_dataset['train_set_y'][:]) # Train set labels

test_dataset = File('test_catvnoncat.h5', 'r')
test_x = np.array(test_dataset['test_set_x'][:]) # Test set features
test_y = np.array(test_dataset['test_set_y'][:]) # Test set labels

classes = np.array(test_dataset['list_classes'][:]) # List of classes

train_y = train_y.reshape((1, train_y.shape[0]))
test_y = test_y.reshape((1, test_y.shape[0]))

# Flatten and normalize
train_x = train_x.reshape(train_x.shape[0], -1).T / 255
test_x = test_x.reshape(test_x.shape[0], -1).T / 255

# Create and train model
np.random.seed(1)

model = model.Model()
model.add(layers.Dense(units=20, activation=activations.ReLU()))
model.add(layers.Dense(units=7, activation=activations.ReLU()))
model.add(layers.Dense(units=5, activation=activations.ReLU()))
model.add(layers.Dense(units=1, activation=activations.SLU()))

model.configure(learning_rate=0.0075, epochs=2500, cost_type=costs.BinaryCrossentropy())
model.train(train_x, train_y, verbose=True)

# Assess model accuracy
pred_train = model.predict(train_x, type='binary_classification') # Get model accuracy on training data
print('Training Accuracy: '  + str(np.sum((pred_train == train_y)/train_x.shape[1])))
pred_test = model.predict(test_x, type='binary_classification') # Get model accuracy on testing data
print('Testing Accuracy: '  + str(np.sum((pred_test == test_y)/test_x.shape[1])))
print(pred_test)

# Use model on custom image
my_image = 'cat.jpg' 
my_label_y = [1] # 1: cat, 0: non-cat
num_px = 64

image = np.array(Image.open(my_image).resize((num_px, num_px))) / 255 # Resize and normalize image, cast to NumPy array
image = image.reshape((1, num_px * num_px * 3)).T # Flatten image array
my_predicted_image = model.predict(image, type='binary_classification') # Predict custom image
print('Custom Image Prediction: ' + classes[int(np.squeeze(my_predicted_image)),].decode('utf-8'))
