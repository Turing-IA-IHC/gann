import numpy as np
from .listener import Listener

class Keep_Progress(Listener):
  """ Class for keeps the progress of the training """

  def _get_name(self): return 'Keep Progress'
    
  def test(self):
    """ Presents an example to show """
    losses = [0.2136,0.1527,0.1121,0.0351,0.0050]
    lrs = [0.05,0.05,0.04,0.04,0.03]
    adaptations = [0.5,0.6,0.7,0.8,0.95]
    for i in range(len(losses)):
      self.append_data(losses[i], lrs[i], adaptations[i])
    self.Y = self.Y if self.Y != None else np.array([[1],[0],[1],[0]])
    self.X = self.X if self.X != None else np.array([[0,0],[1,0],[0,1],[1,1]])

    self.show()

  def show(self):
    """ Show cached data """
    return
    print('Iteration:', self.iter, 'Loss:', round(self.losses[-1],3), 'Lr:', round(self.lrs[-1],3), end='\r')
    try:
      from IPython.display import clear_output
      clear_output(wait=True)
    except ImportError:
      pass
    
li = Keep_Progress()
Listener.append(li)
