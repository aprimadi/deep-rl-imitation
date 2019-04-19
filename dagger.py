import argparse
import os
import pickle
import tensorflow as tf
import numpy as np
import gym
from termcolor import colored
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import load_policy
import tf_util
import helper
import bc

def merge_data(d1, d2):
  data = {}
  data['observations'] = np.concatenate((d1['observations'], d2['observations']))
  data['actions'] = np.concatenate((d1['actions'], d2['actions']))
  return data

def dagger(env, num_rollouts=1, epochs=1):
  data = helper.run_expert(env, num_rollouts=num_rollouts)
  input_dim = len(data['observations'][0])
  output_dim = len(data['actions'][0])
  model = helper.build_model(input_dim, output_dim)

  sess = tf.get_default_session()
  sess.run(tf.global_variables_initializer())

  rewards = []

  os.makedirs('checkpoints', exist_ok=True)

  for epoch in range(epochs):
    checkpoint_path = None
    if epoch == epochs - 1:
      checkpoint_path = helper.checkpoint_path(env, 'dagger-')
    policy_fn = bc.train_bc(sess, data, model=model, curr_epoch=epoch, epochs=1, checkpoint_path=checkpoint_path)

    _data = bc.run_bc(sess, env, policy_fn, num_rollouts=num_rollouts, stats=False)
    _data['actions'] = helper.ask_expert_actions(env, _data['observations'])
    rewards.append(_data['returns'])
    data = merge_data(data, _data)

  return policy_fn, rewards

def run_dagger(epochs=200, num_rollouts=10):
  envs = ["ant", "half_cheetah", "hopper", "humanoid", "reacher", "walker"]

  for env in envs:
    run_dagger_single_env(env, num_rollouts=num_rollouts, epochs=epochs)

def run_dagger_single_env(env, num_rollouts=10, epochs=200):
  print(colored("ENV: %s" % env, 'green'))
  with tf.Session():
    with tf.variable_scope(env):
      sess = tf.get_default_session()
      policy_fn, rewards = dagger(env, num_rollouts=num_rollouts, epochs=epochs)
      bc.run_bc(sess, env, policy_fn, num_rollouts=10)

      return rewards

def run_bc_single_env(env, num_rollouts=10, epochs=200):
  print(colored("ENV: %s" % env, 'green'))
  data = helper.run_expert(env, num_rollouts=num_rollouts)

  input_dim = len(data['observations'][0])
  output_dim = len(data['actions'][0])
  model = helper.build_model(input_dim, output_dim)

  rewards = []
  with tf.Session():
    with tf.variable_scope(env):
      sess = tf.get_default_session()
      sess.run(tf.global_variables_initializer())
      for epoch in range(epochs):
        policy_fn = bc.train_bc(sess, data, model=model, epochs=1, curr_epoch=epoch)

        _data = bc.run_bc(sess, env, policy_fn, num_rollouts=num_rollouts)

        rewards.append(_data['returns'])

      return rewards


def compare_dagger_with_bc(epochs=200, num_rollouts=10):
  env = "walker"

  s_epochs = []
  s_rewards = []
  s_algorithms = []

  dagger_rewards = run_dagger_single_env(env, num_rollouts=num_rollouts, epochs=epochs)
  bc_rewards = run_bc_single_env(env, num_rollouts=num_rollouts, epochs=epochs)

  for e in range(epochs):
    epoch = e + 1
    for r in bc_rewards[e]:
      s_epochs.append(epoch)
      s_rewards.append(r)
      s_algorithms.append('BC')
  for e in range(epochs):
    epoch = e + 1
    for r in dagger_rewards[e]:
      s_epochs.append(epoch)
      s_rewards.append(r)
      s_algorithms.append('DAgger')

  df = pd.DataFrame({
    'epoch': pd.Series(s_epochs),
    'reward': pd.Series(s_rewards),
    'algorithm': pd.Series(s_algorithms),
  })
  sns.lineplot(x="epoch", y="reward", hue="algorithm", style="algorithm", data=df)
  plt.show()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('task', type=str, help='run_all|compare_dagger_vs_bc')
  parser.add_argument('--epochs', type=int, default=200)
  parser.add_argument('--num_rollouts', type=int, default=10)
  args = parser.parse_args()

  if args.task == "run_all":
    run_dagger(epochs=args.epochs, num_rollouts=args.num_rollouts)
  elif args.task == "compare_dagger_vs_bc":
    compare_dagger_with_bc(epochs=args.epochs, num_rollouts=args.num_rollouts)
  else:
    print("Unsupported task")

if __name__ == '__main__':
  main()
