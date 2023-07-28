import numpy as np
from abc import ABC, abstractmethod
from gann.ga import Individual

# TODO: Add a listener to show the evolution
# TODO: Add option to receive a list layer and type of layers availables

class Darwin(ABC):
  """ Class to represent the genetic algorithm """

  def __init__(self, input_dim, output_dim, population:list=None, 
               objetive_function:callable=None, limit_population:int=16, X=None, Y=None, **kwargs):
    """
      Create a class for evolutive 
      input_dim: size, shape, layer or other thing to define input strutcture
      output_dim: size, shape, layer or other thing to define output strutcture
      objetive_function: function wich evaluate the adaptation of individuals
      limit_population: max size of population
    """
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.population = [] if population == None else population
    self.objetive_function = objetive_function if objetive_function != None else self.evaluate
    self.limit_population = limit_population
    self.X = X
    self.Y = Y
    self.kwargs = kwargs
  
  @abstractmethod
  def individual_create(self): pass
  @abstractmethod
  def individual_random(self): pass
  @abstractmethod
  def individual_mutate(self, individual:Individual|int): pass
  @abstractmethod
  def individual_fit(self, index:int): pass
  @abstractmethod
  def individual_cross(self, parent1:int, parent2:int): pass
  @abstractmethod
  def individual_info(self, index:int): pass

  @abstractmethod
  def evaluate(self, population): pass


  def population_add(self, indiv:Individual):
    """ Allow add a new individual to population """
    self.population.append(indiv)

  def population_create(self, size:int=4):
    """ Allow create a new population of size x (min=4) """
    size = max(4, size)
    self.population = []
    for i in range(size):
      self.population.append(self.individual_random())

  def population_sort(self):
    """ Sort by adaptation - best (1.0) to worst (0.0) """
    for i in range(0, len(self.population) - 1):
      for j in range(i + 1, len(self.population)):
        if self.population[i].adaptation < self.population[j].adaptation:
          aux = self.population[i]
          self.population[i] = self.population[j]
          self.population[j] = aux

  def population_cross(self):
    """ Cross best parents """
    self.population_sort()
    # The first parent pairs with the following 4
    for i in range(1, min(5, len(self.population))):
      h1, h2 = self.individual_cross(0, i)
      self.population.insert(0, h1)
      self.population.insert(0, h2)
    # The second parent pairs with the following 2
    for i in range(2, min(4, len(self.population))):
      h1, h2 = self.individual_cross(1, i)
      self.population.insert(0, h1)
      self.population.insert(0, h2)
    # Cross between the first 2 parents to generate 2 children that will be mutated
    h1, h2 = self.individual_cross(0, 1)
    self.population.insert(0, h1)
    self.population.insert(0, h2)
    self.individual_mutate(1)
    self.individual_mutate(0)

  def population_reduce(self):
    """ Limit the number of individuals in the population """
    #self.population_sort()
    self.population = self.population[0: min(self.limit_population, len(self.population))]
    for i in reversed(range(len(self.population))):
      if np.isnan(self.population[i].adaptation) or self.population[i].adaptation < 0:
        self.population.remove(self.population[i])

  def population_info(self):
    """ Shows population info """
    for i in range(len(self.population)):
      print('{:>2}) Adaptation:{:5} => '.format(i, round(self.population[i].adaptation, 4)), end='')
      self.individual_info(i)

  def start_evolution(self, generations:int=100):
    """ Perform the evolution for all individuals """
    print('Start evolution...')

    reached = False
    for g in range(generations):
      print('\t\t ==== Gen:{:4} Size:{:3} ===='.format(g, len(self.population)))
      for i in range(len(self.population)):
        self.individual_fit(i)
      self.evaluate(self.population)

      # Check if objetive was reached
      for p in self.population:
        if p.reached:
          reached = True
          print('\n ******** Objetive reached ******** \n')
          self.population_sort()
          self.population_info()
          break
      if reached: break

      self.population_sort()
      print('')
      self.population_info()
      self.population_cross()
      self.population_reduce()

    print('End evolution...')