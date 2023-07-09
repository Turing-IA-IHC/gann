import numpy as np
from .loss import Loss

"""
Root Mean Square Error loss function

L = sqrt(1/N * sum((predictions - targets)^2))
"""
class RMSE(Loss): # Root Mean Square Error (RMSE)
  def _get_name(self): return 'RMSE'
  def _get_cal(self): return lambda predictions, targets: np.sqrt(np.mean((predictions - targets) ** 2))
  def _get_der(self): return lambda predictions, targets: 1 / 2 * (predictions - targets)
lf = RMSE()
Loss.append(lf)