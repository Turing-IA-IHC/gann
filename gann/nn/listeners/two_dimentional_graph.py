import numpy as np
from .listener import Listener

class TwoDimensional_Graph(Listener):
  """ Class for draw 2d graph """

  def _get_name(self): return '2-Dimensional Graph'
    
  def test(self):
    """ Presents an example of this graph """    
    losses = [0.2136,0.1527,0.1121,0.0351,0.0050]
    lrs = [0.05,0.05,0.04,0.04,0.03]
    adaptations = [0.5,0.6,0.7,0.8,0.95]
    for i in range(len(losses)):
      self.append_data(losses[i], lrs[i], adaptations[i])
    self.Y = self.Y if self.Y != None else np.array([[1],[0],[1],[0]])
    self.X = self.X if self.X != None else np.array([[0,0],[1,0],[0,1],[1,1]])

    self.show()

  def show(self):
    """ This is called to draw graph """
    try:
      import matplotlib.pyplot as plt
    except ImportError:
        # Que hacer si el módulo no se puede importar
        print('exec pip install matplotlib')
        return

    res = 50
    _x0 = np.linspace(-1.5, 1.5, res)
    _x1 = np.linspace(-1.5, 1.5, res)    
    _y = np.zeros((res,res))

    if self.model != None:
      for i0, x0 in enumerate(_x0):
        for i1, x1 in enumerate(_x1):
          _y[i0, i1] = self.model.predict(np.array([[x0, x1]]))

    fig, axs = plt.subplots(1, 2, figsize=(9, 3), layout='constrained', gridspec_kw={'width_ratios': [3, 4]})

    axs[0].pcolormesh(_x0, _x1, _y, cmap='coolwarm')
    #axs[0].axis('equal')
    axs[0].set(title='Iteración: {}'.format(self.iter), aspect=1)
    axs[0].scatter(self.X[self.Y[:,0]==1,0], self.X[self.Y[:,0]==1,1], c='salmon')
    axs[0].scatter(self.X[self.Y[:,0]==0,0], self.X[self.Y[:,0]==0,1], c='skyblue')
    #plt.show()

    #fig, ax = plt.subplots(figsize=(3, 3), layout='constrained')    
    axs[1].plot(range(len(self.losses)), self.losses, label='Loss: ' + str(round(self.losses[-1], 5)))
    axs[1].plot(range(len(self.lrs)), self.lrs, label='Lr: ' + str(round(self.lrs[-1], 5 )))
    axs[1].set(title='Loss and Lr')  # Add a title to the axes.
    axs[1].legend()  # Add a legend.
    plt.show()

    try:
      from IPython.display import clear_output
      clear_output(wait=True)
    except ImportError:
      pass

li = TwoDimensional_Graph()
Listener.append(li)
