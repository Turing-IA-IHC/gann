import numpy as np
from .activation import Activation

"""
Softmax activation function

f(x) = e^(x - max(x)) / sum(e^(x - max(x)))
"""
class Softmax(Activation):
  def softmax_d(self, x):
    # Calcular softmax de entrada
    softmax_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) / np.sum(np.exp(x - np.max(x, axis=-1, keepdims=True)), axis=-1, keepdims=True)
    # Calcular la derivada de softmax
    derivative = softmax_x * (1 - softmax_x)
    return derivative

  def _get_name(self): return 'Softmax'
  def _get_cal(self): return lambda x: np.exp(x - np.max(x, axis=-1, keepdims=True)) / np.sum(np.exp(x - np.max(x, axis=-1, keepdims=True)), axis=-1, keepdims=True)
  def _get_der(self): return self.softmax_d
af = Softmax()
Activation.append(af)