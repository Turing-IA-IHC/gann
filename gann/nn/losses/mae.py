import numpy as np
from .loss import Loss

"""
Mean Absolute Error loss function

L = 1/N * sum(|predictions - targets|)
"""
class MAE(Loss): # Error Absoluto Medio / Pérdida L1
  def _get_name(self): return 'MAE'
  def _get_cal(self): return lambda predictions, targets: np.mean(np.abs(predictions - targets))
  def _get_der(self): return lambda predictions, targets: (predictions - targets) / (np.abs(predictions - targets) + 1e8)
lf = MAE()
Loss.append(lf)