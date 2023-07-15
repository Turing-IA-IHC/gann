from .loss import Loss

from ...params import Params

if Params.gpu_activated():
  from .mse import MSE_GPU as MSE
  from.mae import MAE_GPU as MAE
  from .rmse import RMSE_GPU as RMSE
  from .binary_cross_entropy import BinaryCrossEntropy_GPU as BinaryCrossEntropy
  from .cross_entropy import CrossEntropy_GPU as CrossEntropy
  from .categorical_cross_entropy import CategoricalCrossEntropy_GPU as CategoricalCrossEntropy    
else:
  from .mse import MSE
  from .mae import MAE
  from .rmse import RMSE
  from .binary_cross_entropy import BinaryCrossEntropy
  from .cross_entropy import CrossEntropy
  from .categorical_cross_entropy import CategoricalCrossEntropy