import os
import pickle
import tensorflow as tf
import numpy as np
import gym
from termcolor import colored
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import load_policy
import tf_util
import helper

def train_bc(sess, data, model=None, curr_epoch=None, epochs=1, batch_size=32, debug=False):
  mean, stdev = helper.mean_and_stdev(data['observations'])

  if model:
    input_ph, output_ph, mean_ph, stdev_ph, output_pred, mse, opt = model
  else:
    input_dim = len(data['observations'][0])
    output_dim = len(data['actions'][0])
    input_ph, output_ph, mean_ph, stdev_ph, output_pred, mse, opt = \
      helper.build_model(input_dim, output_dim)

    sess.run(tf.global_variables_initializer())

  saver = tf.train.Saver()

  # run training
  n_inputs = len(data['observations'])
  if debug:
    print(colored('n_inputs: %d' % n_inputs, 'red'))
  for epoch in range(epochs):
    for i in range(1_000):
      indices = np.random.randint(n_inputs, size=batch_size)

      _mean = np.tile(mean, (32, 1))
      _stdev = np.tile(stdev, (32, 1))

      input_batch = data['observations'][indices]
      output_batch = data['actions'][indices]

      _, mse_run = sess.run([opt, mse], feed_dict={input_ph: input_batch, output_ph: output_batch, mean_ph: _mean, stdev_ph: _stdev})

    if curr_epoch:
      print('epoch: {0:03d} mse: {1:.4f}'.format(curr_epoch, mse_run))
    else:
      print('epoch: {0:03d} mse: {1:.4f}'.format(epoch, mse_run))
    saver.save(sess, '/tmp/model.ckpt')

  policy_fn = tf_util.function([input_ph], output_pred, givens={mean_ph: mean[None, :], stdev_ph: stdev[None, :]})
  return policy_fn

def run_bc(sess, envstr, policy_fn, num_rollouts=1, debug=False, stats=True):
  env = gym.make(helper.envname(envstr))
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
      # action, = sess.run([output_pred], feed_dict={input_ph: obs[None, :]})
      action = policy_fn(obs[None, :])
      # print('run_bc action shape:', action.shape)
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

  if stats:
    helper.print_returns_stats(returns)

  data = {
    'observations': np.array(observations),
    'actions': np.array(actions),
    'returns': returns
  }
  return data

def compare_bc_on_multiple_envs():
  envs = ["ant", "half_cheetah", "hopper", "humanoid", "reacher", "walker"]
  epochs = 200
  num_rollouts = 10

  for env in envs:
    print(colored("ENV: %s" % env, 'green'))
    data = helper.run_expert(env, num_rollouts=num_rollouts)
    with tf.Session():
      with tf.variable_scope(env):
        sess = tf.get_default_session()
        policy_fn = train_bc(sess, data, epochs=epochs)

        run_bc(sess, env, policy_fn, num_rollouts=num_rollouts)

        print("==============================================================")

def graph_hyperparameter():
  env = "half_cheetah"

  num_rollouts_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  # num_rollouts_list = [5, 10]
  epochs = 200

  rewards = []
  for num_rollouts in num_rollouts_list:
    print(colored("num_rollouts: %d" % num_rollouts, "red"))
    data = helper.run_expert(env, num_rollouts=num_rollouts)
    with tf.Session():
      with tf.variable_scope("%s-%d" % (env, num_rollouts)):
        sess = tf.get_default_session()
        policy_fn = train_bc(sess, data, epochs=epochs)

        data = run_bc(sess, env, policy_fn, num_rollouts=10)
        rewards.append(np.mean(data['returns']))

  df = pd.DataFrame({
    'rollouts': pd.Series(num_rollouts_list),
    'rewards': pd.Series(rewards),
  })
  sns.lmplot('rollouts', 'rewards', data=df, fit_reg=False)
  plt.show()

def main():
  compare_bc_on_multiple_envs()
  # graph_hyperparameter()

if __name__ == '__main__':
  main()
