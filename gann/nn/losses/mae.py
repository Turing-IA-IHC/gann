import numpy as np
from ...params import Params
from .loss import Loss

try:
    import cupy as cp
except ImportError:
    pass

class MAE(Loss):
    """
    Mean Absolute Error loss function

    L = 1/N * sum(|predictions - targets|)
    """
    def _get_name(self): return 'MAE'    
    def _get_cal(self): return lambda predictions, targets: np.mean(np.abs(predictions - targets))    
    def _get_der(self): return lambda predictions, targets: (predictions - targets) / (np.abs(predictions - targets) + 1e-8)

class MAE_GPU(Loss):
    """
    Mean Absolute Error loss function
    GPU version

    L = 1/N * sum(|predictions - targets|)
    """
    def _get_name(self): return 'MAE'
    def _get_cal(self): return lambda predictions, targets: cp.mean(cp.abs(cp.asarray(predictions) - cp.asarray(targets)))    
    def _get_der(self): return lambda predictions, targets: (cp.asarray(predictions) - cp.asarray(targets)) / (cp.abs(cp.asarray(predictions) - cp.asarray(targets)) + 1e-8)

if Params.gpu_activated():
    lf = MAE_GPU()
else:
    lf = MAE()

Loss.append(lf)