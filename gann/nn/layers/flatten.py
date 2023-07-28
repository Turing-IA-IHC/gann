import numpy as np
from ..activations import Activation, Lineal
from ..losses import Loss
from .layer import Layer

class Flatten(Layer):
  """
  Flatten class

  This class represents a Flatten layer in a neural network.
  It flattens the input data into a 1D vector.

  Example:
    flatten = Flatten()
    flatten.compile(input_shape=(10, 10, 3))
    flatten.info()
  """

  def _get_name(self): return 'Flatten'

  def __init__(self, activation:Activation|str=None, input_shape=None):
    super().__init__(activation if activation is not None else Lineal())
    self.b = None
    self.w = None
    
    if input_shape is not None:
      self.compile(input_shape)

  def create(**kwargs):
    return Flatten()

  def compile(self, input_shape):
    self.input_shape = input_shape
    self.output_shape = (None, np.prod(input_shape[1:]))

  def param_count(self):
    return 0

  def predict(self, X):
    """ Do the inference """
    self.w = X
    return X.reshape((X.shape[0], -1))

  def delta(self, my_predict, delta_next, w_next):
    """ Calculate the Derivative to performe backward propagation """    
    d = (delta_next @ w_next).reshape((-1,) + self.input_shape[1:])
    #sp = (-1,) + self.input_shape[1:]
    #d = (delta_next @ w_next).reshape(sp[::-1])
    return d

  def update_weights(self, delta, input, learning_rate: float):
    """ Update the weights """
    pass

#Layer.append('Flatten', Flatten.create)