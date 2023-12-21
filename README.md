# Neural Network

This repo contains a neural network built from scratch in Python. This project is more of a learning exercise for myself than anything else, and I intend on continuing to add features as I find time / motivation / when I get bored.

## Setup

Install the package: `python -m pip install -e <path to /src>`

## Usage

Import required packages:
```{python}
from sandbox import model, layers, activations, costs, utils
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
    learning_rate=0.0075,
    epochs=2500,
    cost_type=costs.BinaryCrossentropy(),
    initializer=utils.Initializers.he
)
```

Train the model:
```{python}
model.train(train_x, train_y, verbose=True)
```

Predict with the model:
```{python}
prediction = model.predict(image, prediction_type=utils.Predictions.binary_classification)
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
- Heaviside
- Signum
- ELU (Exponential Linear Units)
- SELU (Scaled Exponential Linear Units)
- SLU (Sigmoid Linear Units)
- Softplus
- Softsign
- BentIdentity
- Gaussian
- Arctan
- PiecewiseLinear
- DoubleExponential

**Cost Functions**
- BinaryCrossentropy
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)

**Layer Types**
- Dense
- Dropout

## (Hopefully) Upcoming Features

- Gradient checking
- Momentum-based optimizers
- Multiclass classification
- Mini-batch GD, SGD

## Shorthand Notation

Most (if not all) of the shorthand notation in the code is taken from Andrew Ng.
- X - Inputs
- Y - Labels
- A - Activated neuron values
- AL - Output layer values 
- Z - Pre-activation neuron values
- W, b - Weight matrix, bias vector
- d<W, b, A, AL> - Derivative of value with respect to cost

## Notes

- The first hidden layer here is considered to be layer zero, as opposed to the convention used by Andrew Ng, where the input layer is layer zero.
- This documentation is mostly for myself - There's no reason to use this code over something like TensorFlow, Jax, or PyTorch.
- This project was originally based on the notation and content taught by Andrew Ng in his Deep Learning Specialization course, but it will be transitioning to follow the conventions from in Ovidiu Calin's Deep Learning Architectures book (ISBN 978-3-030-36723-7, https://www.amazon.com/Deep-Learning-Architectures-Mathematical-Approach/dp/3030367207/) if I ever get back around to working on it.
-  The structure of the project is heavily inspired by TensorFlow, as you can see in the example usage.
