import numpy as np
from .layer import Layer

from ..activations import Activation
from ..losses import Loss

class Conv2D(Layer):
  """
  Conv2D layer class

  This class represents a Conv2D layer in a neural network.

  Methods:
   create(filters: int, kernel_size: int, stride: int, padding: int, activation: Activation|str): Create a Conv2d layer
   compile(input_shape: int): Prepare the layer for use
   param_count(): Return the number of parameters in the layer
   info(index=''): Return information about the layer
   predict(X: ndarray): Return the predicted value
   delta(lf: Loss, is_out: bool, my_predict, delta_next=None, w_next=None, Y=None): Return the delta value
   update_weights(my_predict, delta, learning_rate: float): Update the weights

  Properties:
   name: Name of the layer
   qty_outputs: Number of outputs
   input_shape: Number of inputs
   act: Activation function of the layer
  """
  
  def _get_name(self): return 'Conv2D'

  def __init__(self, filters: int, kernel_size: int|tuple, activation: Activation|str, 
               input_shape=None, stride: int=1, padding: int=0):
    super().__init__(activation)
    self.filters = filters
    self.kernel_size_x = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
    self.kernel_size_y = kernel_size[1] if isinstance(kernel_size, tuple) else kernel_size

    self.stride = stride
    self.padding_x = padding[0] if isinstance(padding, tuple) else padding
    self.padding_y = padding[1] if isinstance(padding, tuple) else padding

    self.w = None
    self.b = None
    if input_shape is not None:
      self.compile(input_shape)

  @staticmethod
  def create(**kwargs):
    """Create a Conv2d layer"""
    return Conv2D(kwargs['filters'], kwargs['kernel_size'], kwargs['activation'])

  def compile(self, input_shape: int):
    """Prepare the layer for use"""

    if not isinstance(input_shape, tuple) or len(input_shape) < 3:
      raise Exception('Layer', self.name, 'expected tuple (..., height, width, channels) and received', input_shape)
    
    height, width, channels_shape = input_shape[-3:]

    if self.input_shape == None:
      self.w = np.random.randn(self.filters, self.kernel_size_x, self.kernel_size_y, channels_shape)
      self.b = np.random.randn(self.filters)
    else: # This is thinking in mutations
      self.w = np.resize(self.w, (self.filters, self.kernel_size_x, self.kernel_size_y, channels_shape))
      self.b = np.resize(self.b, (self.filters))
    
    self.input_shape = input_shape
    
    output_height = int(((height - self.kernel_size_x + 2 * self.padding_x) / self.stride) + 1)
    output_width = int(((width - self.kernel_size_y + 2 * self.padding_y) / self.stride) + 1)
    self.output_shape = (None, output_height, output_width, self.filters)

  def predict(self, X: np.ndarray):
    #batch_size, height, width, channels = X.shape
    batch_size = X.shape[0]
    output_height = self.output_shape[1]
    output_width = self.output_shape[2]
    output = np.zeros((batch_size, output_height, output_width, self.filters))

    X = np.pad(X, ((0, 0), (self.padding_x, self.padding_x), (self.padding_y, self.padding_y), (0, 0)), mode='constant')

    for i in range(output_height):
      for j in range(output_width):
        h_start = i * self.stride
        h_end = h_start + self.kernel_size_x
        w_start = j * self.stride
        w_end = w_start + self.kernel_size_y
        X_slice = X[:, h_start:h_end, w_start:w_end, :]

        for f in range(self.filters):
          output[:, i, j, f] = np.sum(X_slice * np.expand_dims(self.w[f, :, :, :], axis=0), axis=(1, 2, 3)) + self.b[f]

    return self.activation.cal(output)

  def param_count(self):
    """ Return the number of parameters in the layer """
    if self.w is None:
      return 0
    return self.w.size + self.b.size
  
  def delta(self, my_predict, delta_next, w_next):
    batch_size, height, width, num_filters = my_predict.shape
    delta = np.zeros_like(my_predict)
    
    if delta_next.shape == w_next.shape:
      delta = delta_next * w_next
      delta *= self.activation.der(my_predict)
      return delta    

    px = (my_predict.shape[1] - delta_next.shape[1]) // 2
    py = (my_predict.shape[2] - delta_next.shape[2]) // 2
    delta_padded = np.pad(delta_next, ((0, 0), (px, px), (py, py), (0, 0)), mode='constant')

    for h in range(height):
      for w in range(width):
        h_start = h * self.stride
        h_end = h_start + self.kernel_size_x
        w_start = w * self.stride
        w_end = w_start + self.kernel_size_y
        delta_slice = delta_padded[:, h_start:h_end, w_start:w_end, :]

        for f in range(self.filters):
          xx = delta_padded[:, w, h, f][:, None, None, None] * w_next[f, :, :, :]
          #delta[:, h_start:h_end, w_start:w_end, :] += xx
          delta[:, h, w, :] += np.sum(delta_padded[:, w, h, f][:, None, None, None] * np.expand_dims(w_next[f, :, :, :], axis=0), axis=(1, 2, 3))
          

    delta *= self.activation.der(my_predict)
    return delta
  
  def update_weights(self, my_predict, delta, learning_rate: float):
    """ Update layer weights next delta is expected """
    pass


#Layer.append('Conv2D', Conv2D.create)