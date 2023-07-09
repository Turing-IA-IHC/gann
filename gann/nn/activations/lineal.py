from .activation import Activation

"""
Lineal activation function

f(x) = x
"""
class Lineal(Activation):
  def _get_name(self): return 'Lineal'
  def _get_cal(self): return lambda x: x
  def _get_der(self): return lambda x: x
af = Lineal()
Activation.append(af)