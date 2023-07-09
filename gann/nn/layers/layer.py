import numpy as np
from abc import ABC, abstractmethod
from ..activations import Activation
from ..losses import Loss

"""
Abstract class for representate any Layer of net

Methods:
  change_af(af:Activation|str): Allow change the Activation function
  append(lt:str, create:callable): Add a new Layer type to available list. If name exist will be replaced
  all(): List of Layers type availables
  create_one(name:str, **kwargs): Create a Layer by name
  create(**kwargs): Create a Layer
  compile(qty_inputs): Prepare Layer for use
  param_count(): Return the number of parameters
  info(index=''): Return information about Layer
  predict(X): Return the predicted value
  delta(lf:Loss, is_out:bool, my_predict, delta_next=None, w_next=None, Y=None): Return the delta value
  update_weights(my_predict, delta, learning_rate:float): Update the weights

Properties:
  name: Name of Layer
  qty_outputs: Number of outputs
  qty_inputs: Number of inputs
  act: Activation function
"""
class Layer(ABC):
  """ Class for representate any Layer of net """

  _all = {}

  def __init__(self, af:Activation|str):
    if type(af) == type('') or type(af) == np.str_:
      af = Activation.get(af)
    self.af = af

  def change_af(self, af:Activation|str):
    """ Allow change the Activation function """
    if type(af) == type('') or type(af) == np.str_:
      af = Activation.get(af)
    self.af = af

  @staticmethod
  def append(lt:str, create:callable):
    """ Add a new Layer type to available list. If name exist will be replaced """
    Layer._all[lt] = create

  @staticmethod
  def all():
    """ List of Layers type availables """
    return list(Layer._all.keys())
  
  @staticmethod
  def create_one(name:str, **kwargs):
    return Layer._all[name](**kwargs)

  @abstractmethod
  def create(**kwargs): pass

  @abstractmethod
  def compile(self, qty_inputs): pass
  @abstractmethod
  def param_count(self): pass
  @abstractmethod
  def info(self, index=''): pass

  @abstractmethod
  def predict(self, X): pass
  @abstractmethod
  def delta(self, lf:Loss, is_out:bool, my_predict, delta_next=None, w_next=None, Y=None): pass
  @abstractmethod
  def update_weights(self, my_predict, delta, learning_rate:float): pass

  @abstractmethod
  def _get_name(self): pass
  @property
  def name(self): return self._get_name()
  @abstractmethod
  def _get_qty_outputs(self): pass
  @property
  def qty_outputs(self): return self._get_qty_outputs()
  @abstractmethod
  def _get_qty_inputs(self): pass
  @property
  def qty_inputs(self): return self._get_qty_inputs()
    
  def _get_act(self): return self.af
  act = property(_get_act)