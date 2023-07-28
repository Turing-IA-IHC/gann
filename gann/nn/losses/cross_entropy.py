import numpy as np
from ...params import Params
from .loss import Loss

try:
  if Params.gpu_activated():
    import cupy as cp
except ImportError:
  pass

class CrossEntropy(Loss):
  """
  Cross entropy loss function

  L = -1/N * sum(targets * log(predictions))
  """
  sigm = lambda x: 1 / (1 + np.exp(-x))  
  def _get_name(self): return 'CrossEntropy'
  def _get_cal(self): return lambda predictions, targets: -np.mean(CrossEntropy.sigm(targets) * np.log(CrossEntropy.sigm(predictions)))
  def _get_der(self):return lambda predictions, targets: -targets / predictions

class CrossEntropy_GPU(Loss):
  """
  Cross entropy loss function
  GPU version

  L = -1/N * sum(targets * log(predictions))
  """
  sigm = lambda x: 1 / (1 + cp.exp(-x))  
  def _get_name(self): return 'CrossEntropy'
  def _get_cal(self): return lambda predictions, targets: -cp.mean(CrossEntropy_GPU.sigm(targets) * cp.log(CrossEntropy_GPU.sigm(predictions)))  
  def _get_der(self): return lambda predictions, targets: -targets / predictions

if Params.gpu_activated():
  lf = CrossEntropy_GPU()
else:
  lf = CrossEntropy()

Loss.append(lf)