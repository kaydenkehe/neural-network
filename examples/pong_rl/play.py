from game import Game
from os import chdir, path
from sandbox import activations, costs, initializers, layers, model, optimizers, utils

# Set current working directory to the directory of this file
chdir(path.dirname(path.abspath(__file__)))

# Create the model and load pre-trained parameters
pong = model.Model()
pong.add(layers.Dense(units=2, activation=activations.ReLU()))
pong.add(layers.Dense(units=1, activation=activations.Sigmoid()))
pong.load('pong.json')

# Create and run the game
game = Game(model=pong)
game.run()
