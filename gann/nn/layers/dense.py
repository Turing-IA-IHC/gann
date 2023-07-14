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
  def _get_qty_outputs(self): return self.qty_neurons
  def _get_qty_inputs(self): return self._qty_inputs

  def __init__(self, qty_neurons:int, act_function:Activation|str):
    super().__init__(act_function)
    self.qty_neurons = qty_neurons
    self._qty_inputs = None
    self.b = np.random.rand(1)[0]
    #self.w = np.random.rand(self.qty_neurons, self._qty_inputs)
    #self.compile(self.qty_inputs)

  def create(**kwargs):
    """ Crea a Layer type Dense """
    return Dense(kwargs['qty_neurons'], kwargs['act_function'])

  def compile(self, qty_inputs):
    #self.w = np.random.rand(self.qty_neurons, self.qty_inputs)

    if self._qty_inputs == None:
      self._qty_inputs = qty_inputs
      self.w = np.random.rand(self.qty_neurons, self._qty_inputs)
    else: 
      self._qty_inputs = qty_inputs
      self.w = np.resize(self.w, (self.qty_neurons, self.qty_inputs))

  def param_count(self):
    return self.w.shape[0] * self.w.shape[1] + 1

  def info(self, index=''):
    """
    Shows complete information of layer
    """
    print('====================================================================')
    print('\t\t\t\t Layer (Dense):', index)

    print('\tBias: {:<10}'.format(round(self.b, 3)), 
          'Act func: {:<10}'.format(self.act.name),
          'Params:', self.param_count())
    print(' N# ', end='')
    for i in range(self.w.shape[1]):
      print('| w{:<3}'.format(i), end='')
    print('')
    for idx, w in enumerate(self.w):
      print(' {:<3}'.format(idx), end='')
      for val in w:
        print('|{:<5}'.format(round(val, 3)), end='')
      print('')
    print('====================================================================')
  
  def predict(self, X):
    """ Do the inference """
    z = X @ self.w.T + self.b
    a = self.af.cal(z)
    return a
  
  def delta(self, lf:Loss, is_out:bool, my_predict, delta_next=None, w_next=None, Y=None):
    """ Calculate the Derivative to performe backward propagation """
    d = None
    if is_out:
      d = lf.der(my_predict, Y) * self.af.der(my_predict)
    else:
      d = delta_next @ w_next * self.af.der(my_predict)
    return d

  def update_weights(self, delta, input, learning_rate:float):
    """ Update layer weights next delta is expected """
    self.b = self.b - (np.mean(delta) * learning_rate)
    self.w = self.w - (delta.T @ input * learning_rate)

Layer.append('Dense', Dense.create)