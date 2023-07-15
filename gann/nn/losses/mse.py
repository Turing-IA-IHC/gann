import numpy as np
from ...params import Params
from .loss import Loss

try:
  import cupy as cp
except:
  pass

"""
Mean Squared Error loss function

L = 1/N * sum((predictions - targets)^2)
"""
class MSE(Loss): # Error cuadrático medio / Pérdida cuadrática / Pérdida L2
  def _get_name(self): return 'MSE'
  def _get_cal(self): return lambda predictions, targets: np.mean((predictions - targets) ** 2)
  def _get_der(self): return lambda predictions, targets: (predictions - targets)

"""
Mean Squared Error loss function

L = 1/N * sum((predictions - targets)^2)
"""
class MSE_GPU(Loss):
  def _get_name(self): return 'MSE'
  def _get_cal(self): return lambda predictions, targets: cp.mean((cp.asarray(predictions) - cp.asarray(targets)) ** 2)
  def _get_der(self): return lambda predictions, targets: (cp.asarray(predictions) - cp.asarray(targets))

if Params.gpu_actived():
  lf = MSE_GPU()
else:
  lf = MSE()
Loss.append(lf)