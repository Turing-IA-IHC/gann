import numpy as np
from abc import ABC, abstractmethod


"""
Listener class

Listener is an abstract class that allow show data while training
"""
class Listener(ABC):
  """ An Listener allow show data while training """

  _all = []

  def __init__(self, X=None, Y=None, model=None):
    self.reset_data()
    self.X = X
    self.Y = Y
    self.model = model

  @staticmethod
  def append(listen):
    """ Add a new Listener fuction to available list. If name exist will be replaced"""
    if not isinstance(listen, Listener):
      raise Exception('Type error: Listener.append() only accept Listener objects. Received: {}'.format(type(listen)))
    
    for i, af in enumerate(Listener._all):
      if af.name.lower() == listen.name.lower():
        Listener._all[i] = listen
        return
    Listener._all.append(listen)
  
  @staticmethod
  def get(name:str):
    """ Return Listener function object by name """
    for li in Listener._all:
      if li.name.lower() == str(name).lower():
        return li
    raise Exception('El Listener ({}) no existe. Consulte Listener.all() o agregue uno nuevo con Listener.append(listener)'.format(
        name
    ))

  @staticmethod
  def all():
    """ List of Listener function availables """
    names = [l.name for l in Listener._all]
    return names
  
  def append_data(self, loss, lr=None, prediction=None): 
    """ Allow add data to show """
    self.losses.append(loss)
    self.lrs.append(lr)
    self.predictions.append(prediction)
    self.iter = self.iter + 1

  def reset_data(self): 
    """ Allow clen all data """
    self.losses = []
    self.lrs = []
    self.predictions = []
    self.iter = 0

  @abstractmethod
  def _get_name(self): pass
  @property
  def name(self): return self._get_name()
  
  @abstractmethod
  def test(funct_names:list=[]):
    """ Test all Activation functions """
    funct_names = Listener.all() if len(funct_names) == 0 else funct_names
    for name in funct_names:
      print('Testing:', name)
      l = Listener.get(name)
      l.test()
      print('')
    
  @abstractmethod
  def show(self): pass
