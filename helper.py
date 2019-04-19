import os
from termcolor import colored
import tensorflow as tf
import numpy as np
import gym
import tensorlayer as tl

import load_policy
import tf_util

def envname(envstr):
  d = {
    "ant": "Ant-v2",
    "half_cheetah": "HalfCheetah-v2",
    "hopper": "Hopper-v2",
    "humanoid": "Humanoid-v2",
    "reacher": "Reacher-v2",
    "walker": "Walker2d-v2",
  }
  return d[envstr]

def input_output_shape(envstr):
  d = {
    "ant": [111, 8],
    "half_cheetah": [17, 6],
    "hopper": [11, 3],
    "humanoid": [376, 17],
    "reacher": [11, 2],
    "walker": [17, 6],
  }
  return d[envstr]

def expert_policy_file(envstr):
  _envname = envname(envstr)
  return "experts/%s.pkl" % _envname

def checkpoint_path(envstr, prefix=''):
  _envname = envname(envstr)
  return "checkpoints/%s%s.ckpt" % (prefix, _envname)

def print_returns_stats(returns, color='green'):
  print(colored('returns: %s' % returns, color))
  print(colored('mean return: %s' % np.mean(returns), color))
  print(colored('std of return: %s' % np.std(returns), color))

def batch_to_seq(h, nbatch, nsteps, flat=False):
  if flat:
    h = tf.reshape(h, [nbatch, nsteps])
  else:
    h = tf.reshape(h, [nbatch, nsteps, -1])
  return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps, value=h)]

def seq_to_batch(h, flat = False):
  shape = h[0].get_shape().as_list()
  if not flat:
    assert(len(shape) > 1)
    nh = h[0].get_shape()[-1].value
    return tf.reshape(tf.concat(axis=1, values=h), [-1, nh])
  else:
    return tf.reshape(tf.stack(values=h, axis=1), [-1])

def ortho_init(scale=1.0):
  def _ortho_init(shape, dtype, partition_info=None):
    #lasagne ortho init for tf
    shape = tuple(shape)
    if len(shape) == 2:
      flat_shape = shape
    elif len(shape) == 4: # assumes NHWC - Num_samples x Height x Width x Channels
      flat_shape = (np.prod(shape[:-1]), shape[-1])
    else:
      raise NotImplementedError
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v # pick the one with the correct shape
    q = q.reshape(shape)
    return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
  return _ortho_init

def lstm(xs, s, scope, nh, init_scale=1.0):
  """LSTM Cell.

  Arguments:
    xs: List/sequence of input tensors.
    s: State tensor (memory in LSTM, contains h[t-1] and c[t-1]).
    scope: Variable scope.
    nh: Size of hidden layer.
    init_scale: Scalar value used to initialize weight using ortho_init.

  Returns:
    Tensor, output of softmax transformation.
  """
  nbatch, nin = [v.value for v in xs[0].get_shape()]
  with tf.variable_scope(scope):
    wx = tf.get_variable("wx", [nin, nh*4], initializer=ortho_init(init_scale))
    wh = tf.get_variable("wh", [nh, nh*4], initializer=ortho_init(init_scale))
    b = tf.get_variable("b", [nh*4], initializer=tf.constant_initializer(0.0))

  c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
  for idx, x in enumerate(xs):
    z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
    i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
    i = tf.nn.sigmoid(i)
    f = tf.nn.sigmoid(f)
    o = tf.nn.sigmoid(o)
    u = tf.tanh(u)
    c = f*c + i*u
    h = o*tf.tanh(c)
    xs[idx] = h
  s = tf.concat(axis=1, values=[c, h])
  return xs, s

def fc(x, nh, *, init_scale=1.0, init_bias=0.0):
  nin = x.get_shape()[1].value
  w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
  b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
  return tf.matmul(x, w)+b

def build_model(input_dim, output_dim):
  H = 64 # Size of hidden layer

  input_ph  = tf.placeholder(tf.float32, shape=[None, input_dim])
  output_ph = tf.placeholder(tf.float32, shape=[None, output_dim])

  mean_v = tf.get_variable(name='mean', dtype=tf.float32, shape=[input_dim], trainable=False)
  stdev_v = tf.get_variable(name='stdev', dtype=tf.float32, shape=[input_dim], trainable=False)

  mean_stacked  = tf.reshape(tf.tile(mean_v, tf.shape(input_ph)[0:1]), tf.shape(input_ph))
  stdev_stacked = tf.reshape(tf.tile(stdev_v, tf.shape(input_ph)[0:1]), tf.shape(input_ph))

  W0 = tf.get_variable(name='W0', shape=[input_dim, H], initializer=tf.initializers.random_normal)
  b0 = tf.get_variable(name='b0', shape=[H], initializer=tf.zeros_initializer)

  W1 = tf.get_variable(name='W1', shape=[H, H], initializer=tf.initializers.random_normal)
  b1 = tf.get_variable(name='b1', shape=[H], initializer=tf.zeros_initializer)

  W2 = tf.get_variable(name='W2', shape=[H, output_dim], initializer=tf.initializers.random_normal)
  b2 = tf.get_variable(name='b2', shape=[output_dim], initializer=tf.zeros_initializer)

  # Build model
  input_norm = (input_ph - mean_stacked) / (stdev_stacked + 1e-6)
  layer = tf.matmul(input_norm, W0) + b0
  layer = tf.nn.relu(layer)
  # layer = tl.activation.swish(layer)
  layer = tf.layers.dropout(layer)
  layer = tf.matmul(layer, W1) + b1
  layer = tf.nn.relu(layer)
  # layer = tl.activation.swish(layer)
  layer = tf.layers.dropout(layer)
  layer = tf.matmul(layer, W2) + b2

  output_pred = layer

  # create loss
  mse = tf.reduce_mean(0.5 * tf.square(output_pred - output_ph))
  # create optimizer
  opt = tf.train.AdamOptimizer().minimize(mse)

  d = {
    'input_ph': input_ph,
    'output_ph': output_ph,
    'mean_v': mean_v,
    'stdev_v': stdev_v,
    'output_pred': output_pred,
    'mse': mse,
    'opt': opt,
    'input_norm': input_norm,
  }
  return d

def build_lstm_model(input_dim, output_dim, scope):
  H_fc = 64   # Size of FC hidden layer
  H_lstm = 64 # Size of LSTM layer

  input_ph = tf.placeholder(tf.float32, shape=[None, input_dim])
  output_ph = tf.placeholder(tf.float32, shape=[None, output_dim])

  # Store state of LSTM to be feed in subsequent call (Ct, Ht)
  S = tf.placeholder(tf.float32, shape=[1, 2*H_lstm])

  mean_v = tf.get_variable(name='mean', dtype=tf.float32, shape=[input_dim], trainable=False)
  stdev_v = tf.get_variable(name='stdev', dtype=tf.float32, shape=[input_dim], trainable=False)

  mean_stacked  = tf.reshape(tf.tile(mean_v, tf.shape(input_ph)[0:1]), tf.shape(input_ph))
  stdev_stacked = tf.reshape(tf.tile(stdev_v, tf.shape(input_ph)[0:1]), tf.shape(input_ph))

  input_norm  = (input_ph - mean_stacked) / (stdev_stacked + 1e-6)
  layer       = fc(input_norm, H_fc)
  layer       = tf.nn.relu(layer)
  layer       = tf.layers.dropout(layer)
  layer       = fc(layer, H_fc)           # Shape: batch x 64
  layer       = tf.nn.relu(layer)
  layer       = tf.layers.dropout(layer)  # Shape: batch x 64
  xs          = batch_to_seq(layer, 1, 32)
  h5, snew    = lstm(xs, S, )
  h           = seq_to_batch(h5)
  layer       = fc(h, H_fc)

  initial_state = np.zeros(S.shape.as_list(), dtype=float)

  output_pred = layer

  # create loss
  mse = tf.reduce_mean(0.5 * tf.square(output_pred - output_ph))
  # create optimizer
  opt = tf.train.AdamOptimizer().minimize(mse)

  d = {
    'input_ph': input_ph,
    'output_ph': output_ph,
    'mean_v': mean_v,
    'stdev_v': stdev_v,
    'S': S,
    'output_pred': output_pred,
    'mse': mse,
    'opt': opt,
    'input_norm': input_norm,
    'state': snew,
    'initial_state': initial_state,
  }
  return d

def mean_and_stdev(data):
  mean = np.mean(data, axis=0)
  stdev = np.std(data, axis=0)
  return mean, stdev

def run_expert(envstr, num_rollouts=1, save=False, debug=False):
  policy_fn = load_policy.load_policy(expert_policy_file(envstr))

  with tf.Session():
    tf_util.initialize()

    env = gym.make(envname(envstr))
    max_steps = env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(num_rollouts):
      if debug:
        print('iter', i)
      obs = env.reset()
      done = False
      totalr = 0.
      steps = 0
      while not done:
        action = policy_fn(obs[None,:])
        observations.append(obs)
        actions.append(action[0])
        obs, r, done, _ = env.step(action)
        totalr += r
        steps += 1
        if debug and steps % 100 == 0:
          print("%i/%i"%(steps, max_steps))
        if steps >= max_steps:
          break
      returns.append(totalr)

    print_returns_stats(returns, color='blue')

    expert_data = {'observations': np.array(observations),
                   'actions': np.array(actions)}

    if save:
      save_data_as_pickle(expert_data, expert_policy_file(envstr))

    return expert_data

def ask_expert_actions(envstr, observations):
  policy_fn = load_policy.load_policy(expert_policy_file(envstr))
  actions = []
  for obs in observations:
    action = policy_fn(obs[None, :])
    actions.append(action[0])

  return actions

def save_data_as_pickle(data, path):
  with open(path, 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
