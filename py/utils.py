"""
Data handling functions.

Nathanael Romano
"""

import tensorflow as tf
import numpy as np
import cPickle


NOTES_PATH = "/scratch/users/naromano/deep-patient/shah/embedding_patients_cui.pkl"
XVAL_PATH = "/scratch/users/kjung/ehr-repr-learn/data/x.val.txt"
XTEST_PATH = "/scratch/users/kjung/ehr-repr-learn/data/x.test.txt"
YVAL_PATH = "/scratch/users/kjung/ehr-repr-learn/data/y.val.txt"
YTEST_PATH = "/scratch/users/kjung/ehr-repr-learn/data/y.test.txt"
YTRAIN_PATH = YVAL_PATH
XTRAIN_PATH = XVAL_PATH
DIMENSION = 8930
NOTES_DIM = 300



class Dataset(object):

  def __init__(self, use_notes=True, **kwargs):
    self.notes_path = kwargs.get("notes", NOTES_PATH)
    self.xtrain_path = kwargs.get("xtrain", XTRAIN_PATH)
    self.xval_path = kwargs.get("xval", XVAL_PATH)
    self.xtest_path = kwargs.get("xtest", XTEST_PATH)
    self.ytrain_path = kwargs.get("ytrain", YTRAIN_PATH)
    self.yval_path = kwargs.get("yval", YVAL_PATH)
    self.ytest_path = kwargs.get("ytest", YTEST_PATH)

    self.dimension = kwargs.get("dim", DIMENSION)
    with open(self.ytrain_path, "r") as f:
        self.labels = [int(lab.replace('"', "")) for lab in
                f.readline().strip().split(" ")[3:]]

    self.use_notes = use_notes
    self.notes_dim = kwargs.get("notes_dim", NOTES_DIM)

    self.patient_ids = {}
    self._index_in_epochs = 0
    self._epochs_completed = 0

    # Load training set
    self.load_set("train")
    if use_notes:
        self.dimension += self.notes_dim
        self.load_notes("train")


  def load_set(self, set_name):
    x = np.loadtxt(
      getattr(self, "x" + set_name + "_path"),
      skiprows=1
    )
    y = np.loadtxt(
      getattr(self, "y" + set_name + "_path"),
      skiprows=1,
      usecols=[0] + range(80)[3:]
    )

    # Save patient ids and mapping to row id
    ids = y[:, 0]
    self.patient_ids[set_name] = dict(zip(ids, range(len(ids))))

    if self.use_notes:
        # Pad x to add encoding dim
        x = np.pad(x, ((0, 0), (0, self.notes_dim)), mode='constant')

    setattr(self, "x" + set_name, x)
    setattr(self, "y" + set_name, y)


  def load_notes(self, set_name):
    with open(self.notes_path, "r") as f:
        while True:
            try:
                pid, encoding = cPickle.load(f)
                if pid in self.patient_ids[set_name]:
                    getattr(self, "x" +
                            set_name)[self.patient_ids[set_name][pid],
                            self.dimension-self.notes_dim:] = encoding
            except EOFError:
                break


  def next_batch(self, batchSize, useLabels=False):
    start = self._index_in_epochs
    self._index_in_epochs += batchSize

    if self._index_in_epochs >= self.xtrain.shape[0]:
        self._epochs_completed += 1
        perm = np.arange(self.xtrain.shape[0])
        np.random.shuffle(perm)
        self.xtrain = self.xtrain[perm, :]
        self.ytrain = self.ytrain[perm, :]
        start = 0
        self._index_in_epochs = batchSize

    end = self._index_in_epochs
    if useLabels:
        return self.xtrain[start:end, :], self.ytrain[start:end, 1:]
    else:
        return self.xtrain[start:end, :]



def load_data(use_notes=True, **kwargs):
  return Dataset(use_notes, **kwargs)



if __name__ == "__main__":
    # Test
    data = load_data(use_notes=True, xtrain=XVAL_PATH, ytrain=YVAL_PATH)

