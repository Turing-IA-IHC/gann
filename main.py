import numpy as np

from gann import Params as gann_params
#gann_params.activate_gpu_mode()

from gann.ga import Darwin

from gann.nn.activations import Activation
from gann.nn.losses import Loss
from gann.nn.listeners import Listener, Keep_Progress, TwoDimensional_Graph

from gann.nn.layers import Layer, Dense
from gann.nn.net import Net

#print('Activations availables:', Activation.all())
# Testing for Activation functions
# Activation.test()

#print('Losses availables:', Loss.all())
# Testing for Loss functions
#Loss.test()

#print('Listeners availables:', Listener.all())
# Testing for Loss functions
#Listener.test()

# """ Testing for Layer Dense functions
from sklearn.datasets import make_circles
np.random.seed(42)
n = 500
clases = 2
X, Y = make_circles(n_samples=n, factor=0.4, noise=0.1)
Y = Y[:,np.newaxis]

best_model = Net(loss_func='mse', layers=[Dense(8, 'Sigmoid'), Dense(4, 'Sigmoid'), Dense(1, 'Sigmoid')],  
  listener=Keep_Progress())
best_model.compile(clases)
#best_model.info()

import cupy as cp
#X = cp.asarray(X)
best_model.train(X, Y, 1500)

best_model.info()
tdg = TwoDimensional_Graph(X,Y, best_model)
for i in zip(best_model.listener.losses, best_model.listener.lrs):
  tdg.append_data(i[0], i[1])
tdg.show()
# """

""" Testing for Layer Conv2 functions

from gann.nn import Net
from gann.nn.layers import Conv2D, Flatten, Dense
from gann.nn.activations import ReLU

# Crear la red neuronal
network = Net(loss_func='mse')

# Agregar las capas a la red
network.append_layer(Conv2D(filters=32, kernel_size=(3, 3), stride=(1, 1), padding='same', act_function=ReLU()))
network.append_layer(Conv2D(filters=64, kernel_size=(3, 3), stride=(1, 1), padding='same', act_function=ReLU()))
network.append_layer(Flatten())
network.append_layer(Dense(qty_neurons=128, act_function=ReLU()))
network.append_layer(Dense(qty_neurons=64, act_function=ReLU()))
network.append_layer(Dense(qty_neurons=10, act_function=ReLU()))

# Compilar la red
network.compile(input_shape=(500, 500, 3))

# Mostrar información de la red
network.info()

# """