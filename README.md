# Neural Network

This repo contains a neural network built from scratch in Python. This project is more of a learning exercise for myself than anything else, and I intend to continue adding features as I find time / motivation / when I get bored.<br>`/src/sandbox` contains the source code, `/examples` contains (you guessed it) examples of model usage, and `gradient_check.ipynb` contains gradient checking code to ensure the model is working correctly.

## Setup

Install the package: `python -m pip install -e <path to /src>`

## Usage

Import required packages:
```{python}
from sandbox import activations, costs, initializers, layers, model, optimizers, utils
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
model.add(layers.Dense(units=1, activation=activations.Sigmoid(), initializer=initializers.he_uniform))
```

Configure the model:
```{python}
model.configure(
    cost_type=costs.BinaryCrossentropy(),
    optimizer=optimizers.Adam(),
    input_size=train_x.shape[1]
)
```

Train the model:
```{python}
model.train(
  train_x,
  train_y,
  epochs,
  learning_rate=0.001,
  batch_size=m,
  verbose=True
)
```

Predict with the model:
```{python}
prediction = model.predict(input)
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
- `configure_cuda` - Configure all modules to use CuPy instead of NumPy, running on Cuda cores.
- `gradient_check(model, X, Y, epsilon=1e-4)` - Assess the correctness of the gradient calculation. Low returned values (less than epsilon squared) indicate that the gradient was computed correctly.
- `binary_round(Y)` - Rounds binary classification output.
- `evaluate(Y_pred, Y)` - Given labels and predictions, return proportion of correct prediction, works for binary or multiclass classification.
- `one_hot(Y, num_classes)` - One-hot encodes labels.
- `argmax(Y)` - Return argmax of labels (index of highest value in each sample prediction).

## Examples:

**Binary Classification**
  - cat_classifier
  - pizza_classifier 
  - point_classifier

**Multiclass Classification**
  - mnist_classifier

**Regression**
  - mpg_estimator

## Features

**Activation Functions**
- Linear
- Sigmoid
- Softmax
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
- CategoricalCrossentropy
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)

**Initializers**
- zeros
- ones
- normal
- uniform
- glorot_normal
- glorot_uniform (default)
- he_normal
- he_uniform

**Layer Types**
- Dense
- Dropout

**Optimizers**
- SGD (default)
- Momentum
- RMSProp
- Adam
- AdaGrad

## Upcoming Features / Changes

- Make NumPy automatic
- Reinforcement learning example
- Numerical stability improvements
- Add layer output shape to summary
- 2D convolutional layers

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
