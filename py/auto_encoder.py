"""
Deep Patient implementation

Nathanael Romano
"""

import argparse

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils import load_data


ENCODE_IDX = 0

# Hyperparameters
NUM_ENCODERS = 3
NOISE_LEVEL = 0.05
H = 500
LR = 0.01



class StackedEncoders(object):
  """
  Convenience class to fit a greddy layer-wise stack of denoising autoencoder
  to data.
  """

  def __init__(self, input_dim, hidden_units=H, noise_level=NOISE_LEVEL,
    learning_rate=LR, num_encoders=NUM_ENCODERS, batch_size=512, verbose=0):
    """
    Initializes model.
    """
    self.input_dim = input_dim
    self.hidden_units = hidden_units
    self.learning_rate = learning_rate
    self.noise_level = noise_level
    self.num_encoders = num_encoders
    self.verbose = verbose
    self.batch_size = batch_size

    self.input_placeholders = []
    self.weights = []
    self.encode_biases = []
    self.decode_biases = []
    self.ops = []
    self.encode_ops = []

    tf.reset_default_graph()
    self.sess = tf.InteractiveSession()

    self._add_placeholders()
    for i in xrange(self.num_encoders):
      if i == 0:
        in_dim = self.input_dim
      else:
        in_dim = self.hidden_units

      self._add_encoder(self.input_placeholders[i],
        in_dim, self.hidden_units)

    self._add_encode_ops()
    self.saver = tf.train.Saver()

    self.sess.run(tf.global_variables_initializer())


  def train(self, dataset, num_iter, encoder_id):
    """
    Greedy layer-wise training.
    """
    encode, decode, cost, train = self.ops[encoder_id]
    if self.verbose >= 1:
      print
      print "Training encoder #%d" % (encoder_id)

    for i in xrange(num_iter):
      batch = dataset.next_batch(self.batch_size)
      # Encode input with previously trained layer
      if encoder_id > 0:
        batch = self.encode(batch, encoder_id)
      # Add noise
      batch = self.corrupt(batch)

      # Training step
      train.run(
        feed_dict={self.input_placeholders[encoder_id]: batch}
      )

      # Print current loss
      if (i + 1) % 100 == 0:
        if self.verbose >= 2:
          loss = cost.eval(
              feed_dict={self.input_placeholders[encoder_id]: batch}
          )
          print "Step %d for encoder %d:" % (i + 1, encoder_id), loss


  def encode(self, data, num_encoders=2):
    """
    Uses the trained encoder to encode some data.

    Can encode partially if num_encoders if specified.
    """
    assert data.shape[1] == self.input_dim
    op = self.encode_ops[num_encoders]
    return op.eval(
      feed_dict={self.input_placeholders[0]: data}
    )


  def corrupt(self, data):
    """
    Input noise.
    """
    n_masked = int(data.shape[1] * self.noise_level)

    for i in xrange(data.shape[0]):
      mask = np.random.randint(0, data.shape[1], n_masked)
      data[:, mask] = 0

    return data


  def save(self, path):
    """
    Saves model.
    """
    self.saver.save(self.sess, path)


  def load(self, path):
    """
    Loads a model.
    """
    self.saver.restore(self.sess, path)


  def close_session(self):
    """
    Closes TF session.
    """
    self.sess.close()


  def _add_placeholders(self):
    """
    Adds input placeholders.
    """
    # Create placeholders
    self.input_placeholders = [
      tf.placeholder(tf.float32, shape=[None, self.input_dim]),
      tf.placeholder(tf.float32, shape=[None, self.hidden_units]),
      tf.placeholder(tf.float32, shape=[None, self.hidden_units]),
    ]


  def _add_encoder(self, input, input_dim, hidden_dim):
    """
    Create an autoencoder with one hidden layer of given dimension, to be 
    trained greedily.
    """
    # Initialize parameters
    w_ = self._xavier_init(input_dim, hidden_dim)
    b1 = tf.Variable(tf.zeros([hidden_dim]))
    b2 = tf.Variable(tf.zeros([input_dim]))

    self.weights.append(w_)
    self.encode_biases.append(b1)
    self.decode_biases.append(b2)

    # Encode and decode
    encode = tf.nn.sigmoid(tf.add(tf.matmul(input, w_), b1))
    decode = tf.nn.sigmoid(tf.add(tf.matmul(encode, tf.transpose(w_)), b2))

    # Objective function and optimizer
    cost = tf.reduce_mean(
      - tf.reduce_sum(input * tf.log(decode) + (1 - input) * tf.log(1 - decode), 
        axis=1)
    )
    train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)
    self.ops.append((encode, decode, cost, train))


  def _add_encode_ops(self):
    """
    Creates encode operations (which takes the raw data as input). The last one
    is the real encoding.
    """
    for ops_id in xrange(self.num_encoders):
      x = self.input_placeholders[0]

      for encoder in xrange(ops_id):
        x = tf.nn.sigmoid(
          tf.add(tf.matmul(x, self.weights[encoder]), 
            self.encode_biases[encoder])
        )
      self.encode_ops.append(x)


  def _xavier_init(self, input_dim, output_dim):
    """
    Creates a weight matrix, initialized with Xavier initialization.

    Returns:
      Tensorflow variable correctly initialized.
    """
    eps = np.sqrt(6. / (input_dim + output_dim))
    return tf.Variable(tf.random_uniform([input_dim, output_dim], -eps, eps))



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Parameters')
  parser.add_argument('-n', '--numIter', type=int, default=500)
  parser.add_argument('-l', '--lr', type=float, default=LR)
  parser.add_argument('-b', '--batchSize', type=int, default=512)
  parser.add_argument('-v', '--verbose', type=int, default=2)
  args = parser.parse_args()
  
  # Load data
  if args.verbose >= 1:
      print "Loading data..."

  dataset = load_data()
  dim = dataset.dimension
      
  dae = StackedEncoders(dim, learning_rate=args.lr, batch_size=args.batchSize,
    verbose=args.verbose)

  # # Greedy layer-wise training
  # for i in xrange(NUM_ENCODERS):
  #  dae.train(dataset, args.numIter, i)

  dae.load('../data/dae.small.ckpt')

  # encoded = dae.encode(dataset.xtrain)
  # np.savetxt("/scratch/users/naromano/deep-patient/shah/x.train.encoded.small.txt", 
  #        encoded,
  #        delimiter=",")

  dataset.load_set("test")
  dataset.load_notes("test")

  encoded = dae.encode(dataset.xtest)
  np.savetxt("/scratch/users/naromano/deep-patient/shah/x.test.encoded.small.txt",
          encoded,
          delimiter=",")

  dae.close_session()

