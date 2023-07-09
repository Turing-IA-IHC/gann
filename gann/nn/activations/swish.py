import numpy as np
from .activation import Activation

"""
Swish activation function

f(x) = x * sigmoid(1.5 * x)
"""
class Swish(Activation):
  sigmoid_func = lambda x: 1 / (1 + np.exp(np.minimum(-x, 512)))
  swish_func = lambda x: x * Swish.sigmoid_func(1.5 * x)
  def _get_name(self): return 'Swish'
  def _get_cal(self): return lambda x : x * Swish.sigmoid_func(1.5 * x)
  def _get_der(self): return lambda x : Swish.swish_func(x) + Swish.sigmoid_func(1.5 * x) * (1 - Swish.swish_func(x))
af = Swish()
Activation.append(af)