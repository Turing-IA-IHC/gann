import numpy as np
from .activation import Activation

"""
Tanh activation function

f(x) = tanh(x)
"""
class Tanh(Activation):
  def _get_name(self): return 'Tanh'
  def _get_cal(self): return lambda x: np.tanh(x)
  def _get_der(self): return lambda x: 1 - np.tanh(x)**2
af = Tanh()
Activation.append(af)