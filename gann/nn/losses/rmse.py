import numpy as np
from ...params import Params
from .loss import Loss

try:
  if Params.gpu_activated():
    import cupy as cp
except ImportError:
  pass

class RMSE(Loss):
  """
  Root Mean Square Error loss function

  L = sqrt(1/N * sum((predictions - targets)^2))
  """
  def _get_name(self): return 'RMSE'  
  def _get_cal(self): return lambda predictions, targets: np.sqrt(np.mean((predictions - targets) ** 2))
  def _get_der(self):return lambda predictions, targets: 1 / 2 * (predictions - targets)

class RMSE_GPU(Loss):
  """
  Root Mean Square Error loss function
  GPU version

  L = sqrt(1/N * sum((predictions - targets)^2))
  """
  def _get_name(self): return 'RMSE'
  def _get_cal(self): return lambda predictions, targets: cp.sqrt(cp.mean((predictions - targets) ** 2))
  def _get_der(self): return lambda predictions, targets: 1 / 2 * (predictions - targets)

if Params.gpu_activated():
  lf = RMSE_GPU()
else:
  lf = RMSE()

Loss.append(lf)