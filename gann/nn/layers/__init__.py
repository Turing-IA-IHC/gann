from .layer import Layer

from ...params import Params

if Params.gpu_activated():  
  from .dense_gpu import Dense_GPU as Dense
  from .flatten_gpu import Flatten_GPU as Flatten
else:
  from .dense import Dense
  from .flatten import Flatten
  from .conv2d import Conv2D