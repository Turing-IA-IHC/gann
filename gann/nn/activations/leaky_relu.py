import numpy as np
from .activation import Activation

"""
Leaky ReLU activation function

f(x) = max(0.01 * x, x)
"""
class LeakyReLU(Activation):
  def _get_name(self): return 'LeakyRelu'
  def _get_cal(self): return lambda x: np.maximum(0.01 * x, x)
  def _get_der(self): return lambda x: 1. * (x > 0)
af = LeakyReLU()
Activation.append(af)