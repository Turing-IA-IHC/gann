import numpy as np
from abc import ABC, abstractmethod

class Loss(ABC):
  """ An Loss function allow calculate difference between target and predicted """

  _all = []

  @staticmethod
  def append(funct):
    """ Add a new Loss fuction to available list. If name exist will be replaced"""
    if not isinstance(funct, Loss):
      raise Exception('funct Error de tipo se esperaba Loss y recibió', type(funct))
    
    for i, af in enumerate(Loss._all):
      if af.name.lower() == funct.name.lower():
        Loss._all[i] = funct
        return
    Loss._all.append(funct)
  
  @staticmethod
  def get(name:str):
    """ Return Loss function object by name """
    for af in Loss._all:
      if af.name.lower() == str(name).lower():
        return af
    raise Exception('La función ({}) no existe. Consulte Loss.all() o agregue una nueva con Loss.append(funct)'.format(
        name
    ))

  @staticmethod
  def all():
    """ List of Loss function availables """
    names = [af.name for af in Loss._all]
    return names

  def test(funct_names:list=[]):
    """ Test all Loss functions """
    try:
      import matplotlib.pyplot as plt
    except ImportError:
        # Que hacer si el módulo no se puede importar
        print('exec pip install matplotlib')
        return

    funct_names = Loss.all() if len(funct_names) == 0 else funct_names
    data = np.linspace(-2, 2, 11)
    hh = np.linspace(int(-len(data) / 2), int(len(data) / 2), len(data))
    pp = [
          data,
          data + 1.0, 
          #data - 1.0, 
          (np.random.rand(len(data)) * 4) - 2
        ]
    pn = [
          'Same',
          'Data + 1.0', 
          #'Data - 1.0',
          'Random'
        ]
    bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']

    for i, p in enumerate(pp):
      fig, ax = plt.subplots(figsize=(5, 3), layout='constrained')

      plt.axhline(y=0, color="black", linestyle="--")
      plt.axvline(x=0, color="black", linestyle="--")

      plt.plot(hh, data, label='Y-target')
      plt.plot(hh, p, label='Y-predicted')

      fns = []
      vals = []
      for nf in funct_names:
        lf = Loss.get(nf)
        fns.append(nf)
        vals.append(float(lf.cal(p, data)))

      plt.bar(fns, vals, label=fns, color=bar_colors)

      ax.set_title('Loss functions ({})'.format(pn[i]))
      ax.legend()
      plt.show()

  @abstractmethod
  def _get_name(self): pass
  @property
  def name(self): return self._get_name()

  @abstractmethod
  def _get_cal(self): pass
  @property
  def cal(self): return self._get_cal()

  @abstractmethod
  def _get_der(self): pass
  @property
  def der(self): return self._get_der()