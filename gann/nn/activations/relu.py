import numpy as np
from .activation import Activation

"""
ReLU activation function

f(x) = max(0, x)
"""
class ReLU(Activation):
  def _get_name(self): return 'ReLU'
  def _get_cal(self): return lambda x: np.maximum(0, x) 
  def _get_der(self): return lambda x: 1. * (x > 0)
af = ReLU()
Activation.append(af)