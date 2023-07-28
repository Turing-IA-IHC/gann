import numpy as np
from abc import ABC, abstractmethod
from ..activations import Activation

class Layer(ABC):
  """ Class for representate any Layer of net """

  _all = {}

  def __init__(self, activation:Activation|str, input_shape=None):
    """ Constructor of Layer """
    self.activation = activation
    self._input_shape = None
    self._output_shape = None
    self._name = 'Layer'  
  
  def info(self, index='', show=True):
    """
    Shows complete information of layer
    """
    if show:
      msg = ' {:>2} {:<7} => Activation: {:<10} Output: {:<25} Params:{:<12}'.format(
        index, self.name, self.activation.name, str(self._output_shape), self.param_count())
      print('-' * 80)
      print(msg)
      print('-' * 80)
    else:
      msg = '{:>2} {:<10} {:<12} {:<25} {:<12}'.format(
        index, self.name, self.activation.name, str(self._output_shape), self.param_count())
    return msg

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
  def compile(self, input_shape): pass
  @abstractmethod
  def predict(self, X): pass
  @abstractmethod
  def delta(self, my_predict, delta_next, w_next): pass
  @abstractmethod
  def update_weights(self, delta, input, learning_rate:float): pass
  @abstractmethod
  def param_count(self): pass

  def _set_activation(self, activation:Activation|str):
    """ Set the activation function of layer """
    if type(activation) == type('') or type(activation) == np.str_:
      activation = Activation.get(activation)
    self._activation = activation
  def _set_input_shape(self, value):
    """ Set the input shape of layer """
    self._input_shape = value
  def _set_output_shape(self, value):
    """ Set the output shape of layer """
    self._output_shape = value
  def _set_name(self, value):
    """ Set the name of layer """
    self._name = value
  @abstractmethod
  def _get_name(self): pass
  name = property(
      lambda self: self._get_name(),
      lambda self, value: self._set_name(value),
      doc=""" Name of type layer """
    )
  input_shape = property(
      lambda self: self._input_shape, 
      lambda self, value: self._set_input_shape(value),
      doc=""" Input shape of layer """
    )
  output_shape = property(
      lambda self: self._output_shape, 
      lambda self, value: self._set_output_shape(value),
      doc=""" Output shape of layer """
    )
  activation = property(
      lambda self: self._activation,
      lambda self, value: self._set_activation(value),
      doc=""" Activation function of layer """
    )
