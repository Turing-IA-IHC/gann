import cupy as cp
import numpy as np

from .layer import Layer
from ..activations import Activation
from ..losses import Loss

class Dense_GPU(Layer):
  """
  Dense class

  This class is a layer of a neural net. It is a fully connected layer that allows creating a layer
  with a certain number of neurons and an activation function.

  Example:
   dense = Dense_GPU(2, 'Sigmoid')
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
    self.b = cp.random.rand(1)[0]
    #self.w = cp.random.rand(self.qty_neurons, self._qty_inputs)

  @staticmethod
  def create(**kwargs):
    """ Create a Layer of type Dense_GPU """
    return Dense_GPU(kwargs['qty_neurons'], kwargs['act_function'])

  def compile(self, qty_inputs):
    if self._qty_inputs == None:
      self._qty_inputs = qty_inputs
      self.w = cp.random.rand(self.qty_neurons, self._qty_inputs)
    else: 
      self._qty_inputs = qty_inputs
      self.w = cp.resize(self.w, (self.qty_neurons, self.qty_inputs))

  def param_count(self):
    return self.w.size + 1

  def info(self, index=''):
    """
    Shows complete information of the layer
    """
    print('====================================================================')
    print('\t\t\t\t Layer (Dense GPU):', index)

    print('\tBias: {:<10}'.format(round(float(cp.asnumpy(self.b)), 3)), 
          'Act func: {:<10}'.format(self.act.name),
          'Params:', self.param_count())
    print(' N# ', end='')
    for i in range(self.w.shape[1]):
        print('| w{:<3}'.format(i), end='')
    print('')
    for idx, row in enumerate(cp.asnumpy(self.w)):
        print(' {:<3}'.format(idx), end='')
        for val in row:
            print('|{:<5}'.format(round(val, 3)), end='')
        print('')
    print('====================================================================')

 
  def predict(self, X):
    """ Do the inference """
    X = cp.asarray(X)
    z = cp.dot(X, cp.transpose(self.w)) + self.b
    a = self.af.cal(cp.asnumpy(z))
    return cp.asarray(a)

  def delta(self, lf:Loss, is_out:bool, my_predict, delta_next=None, w_next=None, Y=None):
    """ Calculate the derivative to perform backward propagation """
    if is_out:
      delta = lf.der(my_predict, Y) * self.af.der(my_predict)
    else:
      delta = delta_next @ w_next * self.af.der(my_predict)
    return delta

  def update_weights(self, delta, input, learning_rate:float):
    """ Update the layer weights """
    delta = cp.asarray(delta)
    input = cp.asarray(input)
    self.b = self.b - (cp.mean(delta) * learning_rate)
    self.w = self.w - (cp.dot(cp.transpose(delta), input) * learning_rate)

Layer.append('Dense', Dense_GPU.create)
