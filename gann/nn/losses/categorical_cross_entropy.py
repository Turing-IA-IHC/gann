import numpy as np
from .loss import Loss

"""
  Catetorical Cross Entropy loss function

  L = - 1/N * sum(targets * log(predictions))
"""
class CategoricalCrossEntropy(Loss): # categorical cross-entropy loss
  clip = lambda x: np.clip(x, 1e-10, 1.0 - 1e-10)
  def _cal(self, predictions, targets):
    predictions = CategoricalCrossEntropy.clip(predictions)
    targets = CategoricalCrossEntropy.clip(targets)
    return - np.mean(targets * np.log(predictions))

  def _get_name(self): return 'CategoricalCrossEntropy'
  def _get_cal(self): return self._cal
  def _get_der(self): return lambda predictions, targets: (predictions - targets) / CategoricalCrossEntropy.clip(predictions)
lf = CategoricalCrossEntropy()
Loss.append(lf)