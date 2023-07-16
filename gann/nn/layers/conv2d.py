import numpy as np
from .layer import Layer

from ..activations import Activation
from ..losses import Loss

class Conv2D(Layer):
  """
  Conv2D layer class

  This class represents a Conv2D layer in a neural network.

  Methods:
   create(filters: int, kernel_size: int, stride: int, padding: int, act_function: Activation|str): Create a Conv2d layer
   compile(qty_inputs: int): Prepare the layer for use
   param_count(): Return the number of parameters in the layer
   info(index=''): Return information about the layer
   predict(X: ndarray): Return the predicted value
   delta(lf: Loss, is_out: bool, my_predict, delta_next=None, w_next=None, Y=None): Return the delta value
   update_weights(my_predict, delta, learning_rate: float): Update the weights

  Properties:
   name: Name of the layer
   qty_outputs: Number of outputs
   qty_inputs: Number of inputs
   act: Activation function of the layer
  """
  
  def _get_name(self): return 'Conv2D'
  def _get_qty_outputs(self): return self.filters
  def _get_qty_inputs(self): return self._qty_inputs
  def _get_act(self): return self.af

  def __init__(self, filters: int, kernel_size: int, stride: int, padding: int, act_function: Activation|str):
    super().__init__(act_function)
    self.filters = filters
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self._qty_inputs = None
    self.W = None
    self.b = None

  @staticmethod
  def create(filters: int, kernel_size: int, stride: int, padding: int, act_function: Activation|str):
    """Create a Conv2d layer"""
    return Conv2D(filters, kernel_size, stride, padding, act_function)

  def compile(self, qty_inputs: int):
    self._qty_inputs = qty_inputs
    kernel_size_x = self.kernel_size[0] if isinstance(self.kernel_size, tuple) else self.kernel_size
    kernel_size_y = self.kernel_size[1] if isinstance(self.kernel_size, tuple) else self.kernel_size
    qty_inputs = qty_inputs[-1] if isinstance(qty_inputs, tuple) else qty_inputs
    self.W = np.random.randn(self.filters, kernel_size_x, kernel_size_y, qty_inputs)
    self.b = np.random.randn(self.filters)

  def param_count(self):
    return np.prod(self.W.shape) + self.b.shape[0]

  def info(self, index=''):
    print('====================================================================')
    print(f'\t\t\t\t Layer (Conv2d): {index}')
    print(f' Filters: {self.filters}')
    print(f' Kernel size: {self.kernel_size}')
    print(f' Stride: {self.stride}')
    print(f' Padding: {self.padding}')
    print(f' Act func: {self.act.name}')
    print(f' Params: {self.param_count()}')
    print('====================================================================')

  def predict(self, X: np.ndarray):
    batch_size, height, width, _ = X.shape
    output_height = int(((height - self.kernel_size + 2 * self.padding) / self.stride) + 1)
    output_width = int(((width - self.kernel_size + 2 * self.padding) / self.stride) + 1)
    output = np.zeros((batch_size, output_height, output_width, self.filters))

    if self.padding > 0:
      X = np.pad(X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')

    for i in range(output_height):
      for j in range(output_width):
        h_start = i * self.stride
        h_end = h_start + self.kernel_size
        w_start = j * self.stride
        w_end = w_start + self.kernel_size
        X_slice = X[:, h_start:h_end, w_start:w_end, :]

        for f in range(self.filters):
          output[:, i, j, f] = np.sum(X_slice * self.W[f, :, :, :], axis=(1, 2, 3)) + self.b[f]

    return self.af.cal(output)

  def delta(self, lf: Loss, is_out: bool, my_predict, delta_next=None, w_next=None, Y=None):
    batch_size, height, width, _ = my_predict.shape
    output_height = int(((height - self.kernel_size + 2 * self.padding) / self.stride) + 1)
    output_width = int(((width - self.kernel_size + 2 * self.padding) / self.stride) + 1)
    delta = np.zeros_like(my_predict)

    if is_out:
      delta = lf.der(my_predict, Y) * self.af.der(my_predict)
    else:
      for i in range(output_height):
        for j in range(output_width):
          h_start = i * self.stride
          h_end = h_start + self.kernel_size
          w_start = j * self.stride
          w_end = w_start + self.kernel_size
          X_slice = my_predict[:, h_start:h_end, w_start:w_end, :]

          for f in range(self.filters):
            delta[:, h_start:h_end, w_start:w_end, :] += (
                delta_next[:, i, j, f][:, None, None, None] * w_next[f, :, :, :])

      delta *= self.af.der(my_predict)

    return delta

  def update_weights(self, my_predict, delta, learning_rate: float):
    batch_size, height, width, _ = my_predict.shape
    output_height = int(((height - self.kernel_size + 2 * self.padding) / self.stride) + 1)
    output_width = int(((width - self.kernel_size + 2 * self.padding) / self.stride) + 1)
    dW = np.zeros_like(self.W)
    db = np.zeros_like(self.b)

    if self.padding > 0:
      my_predict = np.pad(my_predict, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')

    for i in range(output_height):
      for j in range(output_width):
        h_start = i * self.stride
        h_end = h_start + self.kernel_size
        w_start = j * self.stride
        w_end = w_start + self.kernel_size
        X_slice = my_predict[:, h_start:h_end, w_start:w_end, :]

        for f in range(self.filters):
          dW[f, :, :, :] += np.sum(X_slice * delta[:, i, j, f][:, None, None, None], axis=0)
          db[f] += np.sum(delta[:, i, j, f])

    self.W -= learning_rate * dW
    self.b -= learning_rate * db

#Layer.append('Conv2D', Conv2D.create)