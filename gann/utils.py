"""
This file contains utility functions for the GANN project.
"""

import numpy as np

def one_hot_encode(Y: np.ndarray, qty_classes: int) -> np.ndarray:
  """
  One hot encode the Y data

  Parameters:
      Y: The Y data
      qty_classes: The number of classes

  Returns:
      The one hot encoded Y data
  """
  Y = Y.reshape(-1)
  Y_one_hot = np.zeros((Y.size, qty_classes))
  Y_one_hot[np.arange(Y.size), Y] = 1
  return Y_one_hot

def to_categorical(Y: np.ndarray) -> np.ndarray:
  """
  One hot encode the Y data

  Parameters:
      Y: The Y data

  Returns:
      The one hot encoded Y data
  """
  return one_hot_encode(Y, np.max(Y) + 1)

def one_hot_decode(Y: np.ndarray) -> np.ndarray:
  """
  One hot decode the Y data

  Parameters:
      Y: The Y data

  Returns:
      The one hot decoded Y data
  """
  return np.argmax(Y, axis=1)

def to_class(Y: np.ndarray) -> np.ndarray:
  """
  One hot decode the Y data

  Parameters:
      Y: The Y data

  Returns:
      The one hot decoded Y data
  """
  return one_hot_decode(Y)
