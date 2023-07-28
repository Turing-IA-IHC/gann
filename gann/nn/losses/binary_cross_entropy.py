import numpy as np
from ...params import Params
from .loss import Loss

try:
  if Params.gpu_activated():
    import cupy as cp
except ImportError:
  pass

class BinaryCrossEntropy(Loss):
  """
  Binary Cross-Entropy loss function

  L = -1/N * sum(targets * log(predictions) + (1 - targets) * log(1 - predictions))
  """
  clip = lambda x: np.clip(x, 1e-10, 1.0 - 1e-10)
  
  def _cal(self, predictions, targets):
    predictions = BinaryCrossEntropy.clip(predictions)
    targets = BinaryCrossEntropy.clip(targets)
    return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
  
  def _get_name(self): return 'BinaryCrossEntropy'
  def _get_cal(self): return self._cal
  def _get_der(self): return lambda predictions, targets: (predictions - targets) / BinaryCrossEntropy.clip(predictions * (1 - predictions))

class BinaryCrossEntropy_GPU(Loss):
  """
  Binary Cross-Entropy loss function
  GPU version

  L = -1/N * sum(targets * log(predictions) + (1 - targets) * log(1 - predictions))
  """
  clip = lambda x: cp.clip(x, 1e-10, 1.0 - 1e-10)
  
  def _cal(self, predictions, targets):
    predictions = BinaryCrossEntropy_GPU.clip(predictions)
    targets = BinaryCrossEntropy_GPU.clip(targets)
    return -cp.mean(targets * cp.log(predictions) + (1 - targets) * cp.log(1 - predictions))
  
  def _get_name(self): return 'BinaryCrossEntropy'
  def _get_cal(self): return self._cal
  def _get_der(self): return lambda predictions, targets: (predictions - targets) / BinaryCrossEntropy_GPU.clip(predictions * (1 - predictions))

if Params.gpu_activated():
  lf = BinaryCrossEntropy_GPU()
else:
  lf = BinaryCrossEntropy()

Loss.append(lf)