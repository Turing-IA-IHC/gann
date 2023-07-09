import numpy as np
from .activation import Activation

"""
Sigmoid activation function

f(x) = 1 / (1 + e^(-x))
"""
class Sigmoid(Activation):
  def _get_name(self): return 'Sigmoid'
  def _get_cal(self): return lambda x: 1 / (1 + np.exp(-x)) #np.minimum(-x, 512)))
  def _get_der(self): return lambda x: x * (1 - x)
af = Sigmoid()
Activation.append(af)