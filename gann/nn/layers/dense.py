import numpy as np
from .layer import Layer

from ..activations import Activation
from ..losses import Loss

class Dense(Layer):
  """
    Dense class

    This class is a layer of neuronal net, this layer is a fully connected layer
    and allow to create a layer with a qty of neurons and a activation function.

    Example:
      dense = Dense(2, 'Sigmoid')
      dense.compile(2)
      dense.info()      
  """

  def _get_name(self): return 'Dense'

  def __init__(self, neurons:int, activation:Activation|str, input_shape=None):
    super().__init__(activation)
    self.neurons = neurons
    self.b = None
    self.w = None
    if input_shape is not None:
      self.compile(input_shape)

  def create(**kwargs):
    """ Crea a Layer type Dense """
    return Dense(kwargs['neurons'], kwargs['activation'])

  def compile(self, input_shape):
    """ Prepare the layer for use """

    if not isinstance(input_shape, tuple) and not isinstance(input_shape, int):
      raise Exception('Layer', self.name, 'expected tuple (..., connections) or int of connections and received', input_shape)
    
    channels_shape = input_shape[-1] if isinstance(input_shape, tuple) else input_shape

    if self.input_shape == None:
      self.w = np.random.rand(self.neurons, channels_shape)
      self.b = np.random.rand(self.neurons)
    else: # This is thinking in mutations
      self.w = np.resize(self.w, (self.neurons, channels_shape))
      self.b = np.resize(self.b, (self.neurons))
    
    self.input_shape = input_shape
    self.output_shape = (None, self.neurons)

  def predict(self, X):
    """ Do the inference """
    baux = np.expand_dims(self.b, axis=0)
    z = X @ self.w.T + baux
    a = self.activation.cal(z)
    return a
  
  def delta(self, my_predict, delta_next, w_next):
    """ Calculate the Derivative to performe backward propagation """
    d = delta_next @ w_next * self.activation.der(my_predict)
    return d

  def update_weights(self, delta, input, learning_rate:float):
    """ Update layer weights next delta is expected """
    self.b = self.b - (np.mean(delta) * learning_rate)
    self.w = self.w - (delta.T @ input * learning_rate)

  def param_count(self):
    """ Return the number of parameters in the layer """
    if self.w is None:
      return 0
    return self.w.size + self.b.size
  
Layer.append('Dense', Dense.create)