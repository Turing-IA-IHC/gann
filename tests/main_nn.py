import sys
sys.path.insert(0, "../gann")

import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

from gann.nn import Net
from gann.nn.layers import Dense, Conv2D
from gann.nn.listeners import Keep_Progress, TwoDimensional_Graph

np.random.seed(42)
n = 500
clases = 2
X, Y = make_circles(n_samples=n, factor=0.4, noise=0.1)
Y = Y[:,np.newaxis]

#@title Plot the data of test
fig, ax = plt.subplots(figsize=(3, 3), layout='constrained')
plt.scatter(X[Y[:,0]==1,0], X[Y[:,0]==1,1], c='salmon')
plt.scatter(X[Y[:,0]==0,0], X[Y[:,0]==0,1], c='skyblue')
plt.axis('equal')
#plt.show()

Conv2D(filters=8, kernel_size=(3, 3), activation='ReLU').info()
Conv2D(filters=32, kernel_size=(3, 3), activation='ReLU', input_shape=(32, 32, 3)).info()

Dense(8, 'Sigmoid').info()
Dense(8, 'Sigmoid', input_shape=2).info()
Dense(8, 'Sigmoid', input_shape=(2,)).info()

#@title Create the net
best_model = Net(loss_func='mse', 
  layers=[
      Dense(8, 'Sigmoid'), 
      Dense(4, 'Sigmoid'), 
      Dense(1, 'Sigmoid')], 
  listener=Keep_Progress())
best_model.compile(2) # 2 inputs axis X and Y
best_model.summary()

#@title Train the net
best_model.fit(X, Y, 1000)

#@title Plot results
best_model.info()
tdg = TwoDimensional_Graph(X,Y, best_model)
for i in zip(best_model.listener.losses, best_model.listener.lrs):
  tdg.append_data(i[0], i[1])
tdg.show()


