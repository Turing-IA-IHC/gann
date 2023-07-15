import numpy as np
from ..activations import Lineal
from ..losses import Loss
from .layer import Layer

class Flatten(Layer):
    """
    Flatten class

    This class represents a Flatten layer in a neural network.
    It flattens the input data into a 1D vector.

    Example:
        flatten = Flatten()
        flatten.compile(input_shape=(10, 10, 3))
        flatten.info()
    """

    def _get_name(self): return 'Flatten'
    def _get_qty_outputs(self): return np.prod(self.input_shape)
    def _get_qty_inputs(self): return self.input_shape

    def __init__(self):
        super().__init__(Lineal())
        self.input_shape = None

    def create(**kwargs):
        return Flatten()

    def compile(self, input_shape):
        self.input_shape = input_shape

    def param_count(self):
        return 0

    def info(self, index=''):
        print('======================================')
        print('\t\t\t\t Layer (Flatten):', index)
        print(' Input shape:', self.input_shape)
        print(' Output shape:', (self.input_shape[0], np.prod(self.input_shape)))
        print('======================================')

    def predict(self, X):
        return X.reshape((X.shape[0], -1))

    def delta(self, lf: Loss, is_out: bool, my_predict, delta_next=None, w_next=None, Y=None):
        return None
        #return delta_next.reshape(my_predict.shape)

    def update_weights(self, delta, input, learning_rate: float):
        pass

Layer.append('Flatten', Flatten.create)