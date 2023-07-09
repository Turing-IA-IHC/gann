import numpy as np
from .loss import Loss

"""
cross entropy loss function

L = -1/N * sum(targets * log(predictions))
"""
class CrossEntropy(Loss): # Pérdida de entropía cruzada / Probabilidad de registro negativo (analisis propio)
  sigm = lambda x: 1 / (1 + np.exp(-x))
  def _get_name(self): return 'CrossEntropy'
  def _get_cal(self): return lambda predictions, targets: -np.mean(CrossEntropy.sigm(targets) * np.log(CrossEntropy.sigm(predictions)))
  def _get_der(self): return lambda predictions, targets: -targets / predictions
lf = CrossEntropy()
Loss.append(lf)