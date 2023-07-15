from .layer import Layer

from ...params import Params

if Params.gpu_actived():  
  from .dense_gpu import Dense_GPU as Dense
else:
  from .dense import Dense