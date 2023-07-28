import numpy as np
from ...params import Params
from .loss import Loss

try:
  if Params.gpu_activated():
    import cupy as cp
except:
  pass

class MSE(Loss):
  """
  Mean Squared Error loss function / Loss L2

  L = 1/N * sum((predictions - targets)^2)
  """
  def _get_name(self): return 'MSE'
  def _get_cal(self): return lambda predictions, targets: np.mean((predictions - targets) ** 2)
  def _get_der(self): return lambda predictions, targets: (predictions - targets)

class MSE_GPU(Loss):
  """
  Mean Squared Error loss function / Loss L2
  GPU version

  L = 1/N * sum((predictions - targets)^2)
  """
  def _get_name(self): return 'MSE'
  def _get_cal(self): return lambda predictions, targets: cp.mean((cp.asarray(predictions) - cp.asarray(targets)) ** 2)
  def _get_der(self): return lambda predictions, targets: (cp.asarray(predictions) - cp.asarray(targets))

if Params.gpu_activated():
  lf = MSE_GPU()
else:
  lf = MSE()
Loss.append(lf)