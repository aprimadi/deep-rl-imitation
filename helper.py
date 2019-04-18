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
