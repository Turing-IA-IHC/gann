import numpy as np
from .loss import Loss

"""
Binary Cross-Entropy loss function

L = - 1/N * sum(targets * log(predictions) + (1 - targets) * log(1 - predictions))
"""
class BinaryCrossEntropy(Loss): # binary cross-entropy loss
  clip = lambda x: np.clip(x, 1e-10, 1.0 - 1e-10)
  def _cal(self, predictions, targets):    
    predictions = BinaryCrossEntropy.clip(predictions)
    targets = BinaryCrossEntropy.clip(targets)
    return - np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))

  def _get_name(self): return 'BinaryCrossEntropy'
  def _get_cal(self): return self._cal
  def _get_der(self): return lambda predictions, targets: (predictions - targets) / BinaryCrossEntropy.clip(predictions * (1 - predictions))
lf = BinaryCrossEntropy()
Loss.append(lf)