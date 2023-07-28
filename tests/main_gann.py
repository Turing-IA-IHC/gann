import sys
sys.path.insert(0, "../gann")

#@title general imports
import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import time

#@title gann imports
from gann.ga import Individual, Darwin_G0
from gann.nn import Net
from gann.nn.layers import Dense
from gann.nn.listeners import Keep_Progress, TwoDimensional_Graph

#@title Initial parameters
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
plt.show()

# Create a basic evolutive model
db = Darwin_G0(2, 1, X=X, Y=Y, limit_population=16, steps=500, listener=Keep_Progress())

# Create random population
db.population_create(4)

# Create and add some custom individuals
i1 = Net(loss_func='mse', layers=[Dense(8, 'Sigmoid'), Dense(4, 'Sigmoid'), Dense(1, 'Sigmoid')], listener=Keep_Progress())
i1.compile(clases)
db.population_add(Individual(i1))
i2 = Net(loss_func='mse', layers=[Dense(7, 'Sigmoid'), Dense(4, 'Sigmoid'), Dense(1, 'Sigmoid')], listener=Keep_Progress())
i2.compile(clases)
db.population_add(Individual(i2))
i3 = Net(loss_func='mse', layers=[Dense(8, 'Sigmoid'), Dense(3, 'Sigmoid'), Dense(1, 'Sigmoid')], listener=Keep_Progress())
i3.compile(clases)
db.population_add(Individual(i3))

# Show the population info
db.population_info()

# Run the evolutive process
start_time = time.time()
db.start_evolution(7)
end_time = time.time()
print('Time evolving:', round(end_time - start_time, 2), 'seconds')

# Show the best model
db.population_sort()
best_model = db.population[0].ADN
print('Best model adapted:', round(db.population[0].adaptation, 3))
best_model.info()
tdg = TwoDimensional_Graph(X,Y, best_model)
for i in zip(best_model.listener.losses, best_model.listener.lrs):
  tdg.append_data(i[0], i[1])
tdg.show()