import numpy as np
from gann.ga import Darwin  

from gann.nn.activations import Activation
from gann.nn.losses import Loss
from gann.nn.listeners import Listener, TwoDimensional_Graph

from gann.nn.layers import Layer
from gann.nn.net import Net

print('Activations availables:', Activation.all())
# Testing for Activation functions
# Activation.test()

print('Losses availables:', Loss.all())
# Testing for Loss functions
#Loss.test()

print('Listeners availables:', Listener.all())
# Testing for Loss functions
Listener.test()

n = Net(Loss.get('MSE'), 0.05)