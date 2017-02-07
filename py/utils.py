"""
DUMMY FILE

Will contain functions to interact with auto-encoder.

Nathanael Romano
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Dataset(object):

  def __init__(self):
    self.data = np.random.randn(1000, 700)
    self.data = (self.data < -2.0) * 1.0
    self.dimension = 700

  def next_batch(self, batchSize):
    np.random.shuffle(self.data)
    return self.data[:batchSize, :]


def load_data():
  return Dataset()
