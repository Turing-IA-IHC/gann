import numpy as np
from abc import ABC, abstractmethod

class Activation(ABC):
  """ 
    Activation class

    This class is a container of activation functions.
    An Activation function allow modify the lineality of a data.

    Example:
      af = Activation.get('Sigmoid')
      data = np.linspace(-2, 2, 11)
      print(af.cal(data))      
  """

  _all = []

  @staticmethod
  def append(funct):
    """ Add a new Activation fuction to available list. If name exist will be replaced"""
    if not isinstance(funct, Activation):
      raise Exception('funct Error de tipo se esperaba Activation y recibió', type(funct))
    
    for i, af in enumerate(Activation._all):
      if af.name.lower() == funct.name.lower():
        Activation._all[i] = funct
        return
    Activation._all.append(funct)
  
  @staticmethod
  def get(name:str):
    """ Return Activation function object by name """
    for af in Activation._all:
      if af.name.lower() == str(name).lower():
        return af
    raise Exception('La función ({}) no existe. Consulte Activation.all() o agregue una nueva con Activation.append(funct)'.format(
        name
    ))
  @staticmethod
  def all():
    """ List of Activation function availables """
    names = [af.name for af in Activation._all]
    return names

  def test(funct_names:list=[]):
    """ Test all Activation functions """
    
    try:
      import matplotlib.pyplot as plt
    except ImportError:
        # Que hacer si el módulo no se puede importar
        print('exec pip install matplotlib')
        return

    fig, ax = plt.subplots(figsize=(5, 3), layout='constrained')

    plt.axhline(y=0, color="black", linestyle="--")
    plt.axvline(x=0, color="black", linestyle="--")

    funct_names = Activation.all() if len(funct_names) == 0 else funct_names
    data = np.linspace(-2, 2, 11)
    hh = np.linspace(int(-len(data) / 2), int(len(data) / 2), len(data))
    plt.plot(hh, data, label='Base')

    for nf in funct_names:
      af = Activation.get(nf)
      plt.plot(hh, af.cal(data), label=af.name)

    ax.set_title('Activation functions')
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