# Neural Network

This repo contains a neural network built from scratch in Python. This project is more of a learning exercise for myself than anything else, and I intend to continue adding features as I find time / motivation / when I get bored.<br>`/src/sandbox` contains the source code, `/examples` contains (you guessed it) examples of model usage, and `gradient_check.ipynb` contains gradient checking code to ensure the model is working correctly.

## Setup

Install the package: `python -m pip install -e <path to /src>`

## Usage

Import required packages:
```{python}
from sandbox import activations, costs, initializers, layers, model, utils
```

Create the model:
```{python}
model = model.Model(cuda=True)
```

Add layers to the model:
```{python}
model.add(layers.Dense(units=20, activation=activations.ReLU()))
model.add(layers.Dense(units=7, activation=activations.ReLU()))
model.add(layers.Dense(units=5, activation=activations.ReLU()))
model.add(layers.Dense(units=1, activation=activations.Sigmoid()))
```

Configure the model:
```{python}
model.configure(
    cost_type=costs.BinaryCrossentropy(),
    initializer=initializers.Initializers.he,
    input_size=train_x.shape[1]
)
```

Train the model:
```{python}
model.train(
  train_x,
  train_y,
  learning_rate=0.01,
  epochs=100,
  batch_size=32
  verbose=True
)
```

Predict with the model:
```{python}
prediction = model.predict(image)
```

Save / load model parameters:
```{python}
model.save(name='parameters.json', dir='')
model.load(name='parameters.json', dir='')
```

Print model summary:
```{python}
model.summary()
```

## Utilities

Helper functions found in `\src\sandbox\utils.py`:
- `gradient_check(model, X, Y, epsilon=1e-4)` - Assess the correctness of the gradient calculation. Returns normalized Euclidean distance between the actual and approximated gradient.
- `binary_round(Y)` - Rounds binary classification output to 0 or 1.

## Examples:

**Binary Classification**
  - cat_classifier
  - pizza_classifier 
  - point_classifier

**Regression**
  - mpg_estimator

## Features

**Activation Functions**
- Linear
- Sigmoid
- ReLU
- Tanh
- ELU (Exponential Linear Units)
- SELU (Scaled Exponential Linear Units)
- SLU (Sigmoid Linear Units)
- Softplus
- Softsign
- BentIdentity
- Gaussian
- Arctan
- PiecewiseLinear

**Cost Functions**
- BinaryCrossentropy
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)

**Layer Types**
- Dense
- Dropout

## (Hopefully) Upcoming Features

- Momentum-based optimizers
- Multiclass classification

## Shorthand Notation

Most (if not all) of the shorthand notation in the code is taken from Andrew Ng.
- X - Inputs
- Y - Labels
- A - Activated neuron values
- AL - Output layer values 
- Z - Pre-activation neuron values
- W, b - Weight matrix, bias vector
- d<W, b, A, AL> - Derivative of value with respect to cost
- m - Number of training samples

## Notes

- The first hidden layer here is layer zero, as opposed to the convention used by Andrew Ng, where the input layer is layer zero.
- This documentation is mostly for myself - There's no reason to use this code over something like TensorFlow, Jax, or PyTorch.
- This project was originally based on the notation and content taught by Andrew Ng in his Deep Learning Specialization course, but it will be transitioning to follow the conventions from in Ovidiu Calin's Deep Learning Architectures book (ISBN 978-3-030-36723-7, https://www.amazon.com/Deep-Learning-Architectures-Mathematical-Approach/dp/3030367207/) if I ever get back around to working on it.
-  The structure of the project is heavily inspired by TensorFlow, as you can see in the example usage.
