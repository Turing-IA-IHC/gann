import numpy as np
from .loss import Loss

"""
Mean Squared Error loss function

L = 1/N * sum((predictions - targets)^2)
"""
class MSE(Loss): # Error cuadrático medio / Pérdida cuadrática / Pérdida L2
  def _get_name(self): return 'MSE'
  def _get_cal(self): return lambda predictions, targets: np.mean((predictions - targets) ** 2)
  def _get_der(self): return lambda predictions, targets: (predictions - targets)
lf = MSE()
Loss.append(lf)