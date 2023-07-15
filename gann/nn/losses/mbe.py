import numpy as np
from ...params import Params
from .loss import Loss

try:
    import cupy as cp
except ImportError:
    pass

class MBE(Loss):
    """
    Mean Bias Error loss function

    L = 1/N * sum(predictions - targets)
    """
    def _get_name(self): return 'MBE'
    def _get_cal(self): return lambda predictions, targets: np.mean(predictions - targets)
    def _get_der(self): return lambda predictions, targets: np.ones_like(predictions)

class MBE_GPU(Loss):
    """
    Mean Bias Error loss function
    GPU version

    L = 1/N * sum(predictions - targets)
    """
    def _get_name(self): return 'MBE'    
    def _get_cal(self): return lambda predictions, targets: cp.mean(cp.asarray(predictions) - cp.asarray(targets))
    def _get_der(self): return lambda predictions, targets: cp.ones_like(predictions)

if Params.gpu_activated():
    lf = MBE_GPU()
else:
    lf = MBE()

# Loss.append(lf) # Not add because is not a good loss function