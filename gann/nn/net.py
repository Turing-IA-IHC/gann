import numpy as np
import pickle
from tqdm import tqdm 

from gann.params import Params
from gann.nn.losses import Loss
from gann.nn.layers import Layer
from gann.nn.listeners import Listener

try:
  import cupy as cp
except:
  pass

class Net():
  """
    Net class

    This class is a container of layers, and allow to create a neuronal net
    with a list of layers.

    Example:
      net = Net('MSE', 0.05)
      net.append_layer(Layer.create_one('Dense', 2, 'Sigmoid'))
      net.append_layer(Layer.create_one('Dense', 1, 'Sigmoid'))
      net.compile(2)
      net.summary()
      net.fit(X, Y, 1000)
      net.predict(X)
  """

  def __init__(self, loss_func:Loss|str, learning_rate:float=0.05, layers=None, listener:Listener|str=None):
    self.change_lf(loss_func)
    self.change_lr(learning_rate)
    self.change_listener(listener)
    self._layers = [] if layers == None else layers
    self._input_shape = None
    self._loss_val = 0.0
  
  def add(self, l):
    """ Add a Layer to list """
    if not isinstance(l, Layer):
      raise Exception('Error de tipo se esperaba Layer y recibió', type(l))
    self._layers.append(l)

  def layers(self):
    """ List layer in the net """
    return self._layers
  
  def get_layer(self, index:int)->Layer:
    """ Return Layer in index position """
    return self._layers[index]

  def change_layer(self, index:int, l:Layer):
    """ Allow change a layer by other """
    self._layers[index] = l

  def insert_layer(self, index:int, l:Layer):
    """ Allow insert a layer in index position """
    self._layers.insert(index, l)

  def remove_layer(self, index:int):
    """ Allow to delete a layer in index """
    self._layers.remove(self._layers[index])

  def compile(self, input_shape):
    """ Runs compile of all layers sequiential """
    self.input_shape = input_shape
    self._layers[0].compile(input_shape)
    for i, l in enumerate(self._layers[1:], 1):
      l.compile(self._layers[i - 1].output_shape)
  
  def param_count(self):
    """ Returns the params quantity """
    params = 0
    for l in self._layers:
      params = params + l.param_count()
    return params
    
  def _set_input_shape(self, value):
    """ Set the input shape of net """
    self._input_shape = value
  
  input_shape = property(
      lambda self: self._input_shape, 
      lambda self, value: self._set_input_shape(value),
      doc=""" Input shape of net """
    )
  
  def change_lf(self, lf:Loss|str):
    """ Allow change the Loss function """
    if type(lf) == type(''):
      lf = Loss.get(lf)
    self._lf = lf    
  def _get_loss(self): return self._lf
  loss = property(_get_loss)
  
  def change_lr(self, lr:float):
    """ Allow change the Learning rate """
    self._lr = lr    
  def _get_lr(self): return self._lr
  lr = property(_get_lr)

  def change_listener(self, listener:Listener|str):
    """ Allow change the Listener """
    if type(listener) == type(''):
      listener = Listener.get(listener)
    if listener != None and not isinstance(listener, Listener):
      raise Exception('Error expected Listener and received', type(listener))
    self._listener = listener
  def _get_listener(self): return self._listener
  listener = property(_get_listener)
    
  def change_loss_val(self, loss_val:float):
    """ Allow change the Loss value """
    self._loss_val = loss_val    
  def _get_loss_val(self): return self._loss_val
  loss_val = property(_get_loss_val)
  
  def info(self, detailed=False):    
    """
    Shows full Net structure
    """
    print('====================================================================')    
    return

    print('Inputs: {:<4} Outputs: {:<4} lr: {:<5}  loss ({}): {:<5}  Params: {}'.format(
        str(self.input_shape),
        str(self._layers[-1].output_shape),
        self.lr,
        self.loss.name,
        np.round(self.loss_val, 3),
        self.param_count()
      )
    )

    if not detailed:
      print('')
      print(' Layer : Type     | in  : out : Act func  : Params')

    for idx, hl in enumerate(self._layers):
      if detailed:
        hl.info(idx)
      else:
        print(' {:^5} : {:<9}|{:^5}:{:^5}: {:<9} : {:<10}'.format(
            idx, hl.name , str(hl.input_shape), hl.output_shape, hl.activation.name, hl.param_count())
        )
    
    if not detailed:
      print('')

    print('====================================================================')
    
  def mini_info(self, detailed=False):    
    """
    Shows simple Net structure
    """
    s = [
        '{} ({})'.format(l.output_shape, l.act.name) if detailed else l.output_shape 
         for l in self._layers
         ]
    print('In: {:<2} -> {} -> {:>5}: {} | lr: {}'.format(
        self.input_shape ,s, self.loss.name, 
        round(self.loss_val, 5), round(self.lr, 5))
    )

  def summary(self, detailed=False):
    """ Shows full Net structure """
    print('=' * 80)
    print('Inputs: {:<4} Outputs: {:<4} Params: {}'.format(
        str(self.input_shape),
        str(self._layers[-1].output_shape),
        self.param_count()
      )
    )
    print('Loss ({}): {:<5} Learning rate: {:<5}'.format(
        self.loss.name, np.round(self.loss_val, 3), self.lr
      )
    )
    print('{:>2} {:<10} {:<12} {:<25} {:<12}'.format('Id', 'Name', 'Activation', 'Output', 'Params'))
    print('-' * 80)
    for i, l in enumerate(self._layers):
      print(l.info(i, show=False))
    print('=' * 80)

  def save(self, file_name):
    """ Save model """
    file = open(file_name, 'wb')
    pickle.dump(self, file)
    file.close()
  
  @staticmethod
  def load(file_name):
    """ Load model """
    file = open(file_name, 'rb')
    model_loaded = pickle.load(file)
    file.close()
    return model_loaded

  def predict(self, X):
    """ Perfom the prediction (Forward propagation) """
    a = X
    for l in self._layers:
      a = l.predict(a)
    return a

  def fit(self, X, Y, epochs:int=1000, lf:Loss=None, lr:float=None, listener:Listener=None):
    """ Perform backward propagation and gradient descendent """

    if Params.gpu_activated():
      X = cp.asarray(X)

    if lf != None: self.change_lf(lf)
    if lr != None: self.change_lr(lr)

    listener = listener if listener != None else self.listener

    t = tqdm(range(epochs), desc='Training ', disable=listener!=None and listener.name != 'Keep Progress')
    outputs = None
    for epoch in t:
      # Fordward propagation
      outputs = [self._layers[0].predict(X)]
      for i, l in enumerate(self._layers[1:], start=1):
        outputs.append(l.predict(outputs[i - 1]))
      
      # Keep w before update
      _w = self._layers[-1].w
      # Backward propagation / Gradient descendent
      # Calculate output delta
      deltas = [self.loss.der(outputs[-1], Y) * self._layers[-1].activation.der(outputs[-1])]
      # Update weights
      self._layers[-1].update_weights(deltas[0], outputs[-2], self.lr)

      for i, l in reversed(list(enumerate(self._layers[1:-1], start=1))):
        deltas.insert(0, l.delta(outputs[i], delta_next=deltas[0], w_next=_w))
        _w = l.w
        l.update_weights(deltas[0], outputs[i-1], self.lr)
      
      deltas.insert(0, self._layers[0].delta(outputs[0], delta_next=deltas[0], w_next=_w))
      self._layers[0].update_weights(deltas[0], X, self.lr) # Último paso contra la entrada      
      self.change_loss_val(self.loss.cal(outputs[-1], Y))

      if listener != None:
        listener.append_data(self.loss_val, self.lr, outputs[-1])
        if (epoch + 1) % 50==0:
          listener.show()
      
      if listener == None or listener.name == 'Keep Progress':
        t.set_description('Training ({:5}:{:5})'.format(self.loss.name, np.round(self.loss_val, 3)))
        t.refresh()

      if self.loss_val < self.lr:
        self.change_lr(self.lr * 0.9)

      #time.sleep(0.5)
    #return self.loss_val, self.lr, outputs[-1]