import numpy as np
from .layer import Layer
from ..activations import Activation
from ..losses import Loss

try:
  import cupy as cp
  has_gpu = True
except ImportError:
  has_gpu = False


class Dense(Layer):
  """
  Dense class

  This class represents a fully connected layer in a neural network. It allows you to create a layer with a specific
  number of neurons and an activation function.

  Example:
    dense = Dense(2, 'Sigmoid')
    dense.compile(2)
    dense.info()
  """

  def _get_name(self):
    return 'Dense'

  def _get_qty_outputs(self):
    return self.qty_neurons

  def _get_qty_inputs(self):
    return self._qty_inputs

  def __init__(self, qty_neurons: int, act_function: Activation | str):
    super().__init__(act_function)
    self.qty_neurons = qty_neurons
    self._qty_inputs = None
    self.b = None
    self.w = None

  @staticmethod
  def create(**kwargs):
    """
    Create a Layer of type Dense.

    Args:
      **kwargs: Keyword arguments for initializing the layer.

    Returns:
      Dense: A new instance of Dense layer.
    """
    return Dense(kwargs['qty_neurons'], kwargs['act_function'])

  def compile(self, qty_inputs):
    """
    Prepare the layer for use.

    Args:
      qty_inputs (int): Number of inputs to the layer.
    """
    if self._qty_inputs is None:
      self._qty_inputs = qty_inputs
      if has_gpu:
        self.b = cp.random.rand(1)[0]
        self.w = cp.random.rand(self.qty_neurons, self._qty_inputs)
      else:
        self.b = np.random.rand(1)[0]
        self.w = np.random.rand(self.qty_neurons, self._qty_inputs)
    else:
      self._qty_inputs = qty_inputs
      if has_gpu:
        self.w = cp.resize(
          self.w, (self.qty_neurons, self._qty_inputs))
      else:
        self.w = np.resize(
          self.w, (self.qty_neurons, self._qty_inputs))

  def param_count(self):
    """
    Return the number of parameters in the layer.

    Returns:
      int: Number of parameters in the layer.
    """
    return self.w.size + 1

  def info(self, index=''):
    """
    Print complete information about the layer.

    Args:
        index (str): Index of the layer.
    """
    print('====================================================================')
    print('\t\t\t\t Layer (Dense):', index)

    print('\tBias: {:<10}'.format(
        round(cp.asnumpy(self.b) if has_gpu else self.b, 3) if has_gpu else round(self.b, 3)),
        'Act func: {:<10}'.format(self.act.name),
        'Params:', self.param_count())
    print(' N# ', end='')
    for i in range(self.w.shape[1]):
        print('| w{:<3}'.format(i), end='')
    print('')
    for idx, w in enumerate(self.w):
        print(' {:<3}'.format(idx), end='')
        for val in w:
            print('|{:<5}'.format(
                round(cp.asnumpy(val) if has_gpu else val, 3) if has_gpu else round(val, 3)), end='')
        print('')
    print('====================================================================')

  def predict(self, X):
    """
    Perform the forward pass and return the predicted value.

    Args:
      X (ndarray): Input data.

    Returns:
      ndarray: Predicted value.
    """
    if has_gpu:
      z = cp.dot(X, cp.transpose(self.w)) + self.b
      a = self.af.cal(cp.asnumpy(z))
      return cp.asarray(a)
    else:
      z = np.dot(X, np.transpose(self.w)) + self.b
      return self.af.cal(z)

  def delta(self, lf: Loss, is_out: bool, my_predict, delta_next=None, w_next=None, Y=None):
    """
    Calculate the delta value.

    Args:
      lf (Loss): Loss function.
      is_out (bool): Whether the layer is an output layer.
      my_predict (ndarray): Predicted value of the layer.
      delta_next (ndarray, optional): Delta value from the next layer.
      w_next (ndarray, optional): Weights from the next layer.
      Y (ndarray, optional): Ground truth labels.

    Returns:
      ndarray: Delta value.
    """
    if has_gpu:
      my_predict = cp.asnumpy(my_predict)
      delta_next = cp.asnumpy(delta_next)
      w_next = cp.asnumpy(w_next)
      Y = cp.asnumpy(Y)

    if is_out:
      delta = lf.der(my_predict, Y) * self.af.der(my_predict)
    else:
      delta
