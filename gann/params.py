class Params:
  _gpu_activated = False

  @staticmethod
  def activate_gpu_mode():
    try:
      import cupy as cp
      Params._gpu_activated = True
      print('GPU mode activated.')
    except ImportError:
      Params._gpu_activated = False
      print('GPU mode not activated (Cupy not installed).')
  
  @staticmethod
  def deactivate_gpu_mode():
    Params._gpu_activated = False
    print('GPU mode deactivated. Could be required to restart the kernel.')
  
  @staticmethod
  def gpu_activated(): return Params._gpu_activated