class Individual():
  """ Class to representate an indivudual """
  def __init__(self, ADN):
    self.ADN = ADN
    self.adaptation = 0.0
    self.reached = False