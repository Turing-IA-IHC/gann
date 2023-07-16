import cupy as cp
from .layer import Layer

class Flatten_GPU(Layer):
  """
  Flatten layer class (GPU version)

  This class represents a flatten layer in a neural network.
  It flattens the input tensor into a 1D array.

  Example:
    flatten = Flatten_GPU()
    flatten.compile(input_shape=(10, 20, 3))
  """

  def _get_name(self): return 'Flatten'
  def _get_qty_outputs(self): return self.qty_outputs
  def _get_qty_inputs(self): return self.qty_inputs

  def __init__(self):
    super().__init__(act_function=None)
    self.qty_outputs = None
    self.qty_inputs = None

  def create(**kwargs):
    """ Create a Flatten_GPU layer """
    return Flatten_GPU()

  def compile(self, input_shape):
    self.qty_inputs = input_shape[0] * input_shape[1] * input_shape[2]
    self.qty_outputs = self.qty_inputs

  def param_count(self):
    return 0

  def info(self, index=''):
    print('=================================================')
    print('\t\t\t Layer (Flatten GPU):', index)
    print(' Flatten layer')
    print(' Input shape:', self.qty_inputs)
    print(' Output shape:', self.qty_outputs)
    print('=================================================')

  def predict(self, X):
    return cp.reshape(X, (X.shape[0], -1))

  def delta(self, lf, is_out, my_predict, delta_next=None, w_next=None, Y=None):
    #return cp.zeros_like(my_predict)
    return None

  def update_weights(self, delta, input, learning_rate):
    pass

#Layer.append('Flatten', Flatten_GPU.create)