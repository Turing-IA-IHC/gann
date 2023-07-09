import numpy as np
import copy

from gann.ga import Individual
from gann.ga import Darwin

from gann.nn.activations import Activation
from gann.nn.losses import Loss
from gann.nn.layers import Layer, Dense
from gann.nn.net import Net
from gann.nn.listeners import Listener

class Darwin_G0(Darwin):
  """ Evolutive model to use basic neuronal net """

  def individual_create(self, structure:list, loss_func:Loss|str, learning_rate:float=0.05, listener:Listener=None):
    """ Allow create simple individual """
    model = Net(loss_func, learning_rate, listener=listener)
    for l in structure:
      model.append_layer(l)
    model.compile(self.input_dim)
    return Individual(model)

  def individual_random(self):
    """ Create a random individual """
    loss_func = Loss.get(np.random.choice(Loss.all()))
    learning_rate = np.random.choice([0.1, 0.05, 0.02, 0.01])
    size = np.random.randint(2, 11)
    structure = []
    for s in range(size):
      act_function = np.random.choice(Activation.all())
      qty_neurons = np.random.randint(1, 129)
      l = Layer.create_one('Dense', qty_neurons=qty_neurons, act_function=act_function)
      structure.append(l)    
    structure.append(Dense(self.output_dim, 'Sigmoid'))
    listener = self.kwargs['listener'] if 'listener' in self.kwargs else None
    return self.individual_create(structure, loss_func, learning_rate, listener)
  
  def individual_mutate(self, individual:Individual|int):
    """ Modify chromosome to test other combinations """
    chang_index = False
    if type(individual) == type(1):
      indiv = self.population[individual]
      chang_index = True
    else:
      indiv = individual
    
    # 0: Nothing, 1:Replace, 2:Insert, 3:Remove, 4:Change Activation, 
    # 5: change Loss, 6: change Learning rate
    t = np.random.randint(1, 7)
    #mt = ['Nothing', 'Replace', 'Insert', 'Remove', 'Change Activation', 'change Loss', 'change Learning rate']
    #print('Muting type:', mt[t])

    if t < 5:
      m = np.random.randint(0, len(indiv.ADN.layers()) - 1)
      l1 = np.random.choice(Layer.all())    
      act_function = np.random.choice(Activation.all())
      qty_neurons = np.random.randint(1, 129)
      l = Layer.create_one(l1, qty_neurons=qty_neurons, act_function=act_function) # New layer
    
      if t == 1:
        indiv.ADN.change_layer(m, l)
      elif t == 2:
        indiv.ADN.insert_layer(m, l)
      elif t == 3 and len(indiv.ADN.layers()) > 2:
        indiv.ADN.remove_layer(m)
      elif t == 4:
        indiv.ADN.get_layer(m).change_af = act_function

    elif t == 5:
      loss_func = Loss.get(np.random.choice(Loss.all()))
      indiv.ADN.change_lf(loss_func)

    elif t == 6:
      learning_rate = np.random.choice([0.1, 0.05, 0.02, 0.01])
      indiv.ADN.change_lr(learning_rate)

    indiv.ADN.compile(self.input_dim)
    if chang_index:
      self.population[individual] = indiv
    return indiv

  def individual_fit(self, index:int):
    """ Train individual """
    self.steps = self.kwargs['steps'] if 'steps' in self.kwargs else 500    
    self.population[index].ADN.train(self.X, self.Y, self.steps)

  def individual_cross(self, parent1:int, parent2:int):
    """ Cross two parents to generate two childs """
    indiv1 = self.population[parent1]
    indiv2 = self.population[parent2]

    cut1 = np.random.randint(1, len(indiv1.ADN.layers()) - 1)    
    cut2 = np.random.randint(1, len(indiv2.ADN.layers()) - 1)
    
    child1 = []
    child2 = []
    
    for i in range(0, cut1):
        child1.append(copy.deepcopy(indiv1.ADN.layers()[i]))
    for i in range(0, cut2):
        child2.append(copy.deepcopy(indiv2.ADN.layers()[i]))
    for i in range(cut1, len(indiv1.ADN.layers())):
        child2.append(copy.deepcopy(indiv1.ADN.layers()[i]))
    for i in range(cut2, len(indiv2.ADN.layers())):
        child1.append(copy.deepcopy(indiv2.ADN.layers()[i]))

    child1 = self.individual_create(child1, indiv1.ADN.loss, indiv1.ADN.lr, copy.deepcopy(indiv1.ADN.listener))
    child2 = self.individual_create(child2, indiv2.ADN.loss, indiv2.ADN.lr, copy.deepcopy(indiv2.ADN.listener))
    child1.ADN.compile(self.input_dim)
    child2.ADN.compile(self.input_dim)

    return child1, child2
  
  def individual_info(self, index:int):
    """ Shows individual info """
    self.population[index].ADN.mini_info()

  def evaluate(self, population):
    """ Check objetive function """
    # extra point for less params
    min_params = 10000
    min_index = -1
    for i, p in enumerate(self.population):
      p.adaptation = 1 - p.ADN.loss_val
      if np.isnan(p.adaptation):
        p.adaptation = -1
      if p.adaptation > 0.99:
        p.reached = True
      if p.ADN.param_count() < min_params:
        min_params = p.ADN.param_count()
        min_index = i
    self.population[min_index].adaptation = self.population[min_index].adaptation + 0.05