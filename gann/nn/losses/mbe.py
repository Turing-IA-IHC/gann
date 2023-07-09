import numpy as np
from .loss import Loss

"""
Mean Bias Error loss function

L = 1/N * sum(predictions - targets)
"""
class MBE(Loss): # Error de sesgo medio
  def _get_name(self): return 'MBE'
  def _get_cal(self): return lambda predictions, targets: np.mean(predictions - targets)
  def _get_der(self): return lambda predictions, targets: np.ones_like(predictions)
lf = MBE()
#Loss.append(lf) # No se agrega por no ser una buena medida del error general