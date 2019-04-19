import os
import tensorflow as tf
import numpy as np
import gym

import helper
import dagger
import tf_util

def train(sess, data, model, curr_epoch, batch_size=32, debug=False, checkpoint_path=None):
  obs = None
  for _data in data:
    if obs is None:
      obs = _data['observations']
    else:
      obs = np.concatenate((obs, _data['observations']))
  mean, stdev = helper.mean_and_stdev(obs)
  empty_action = np.array([0] * 17)

  m = model
  input_ph, output_ph     = m['input_ph'], m['output_ph']
  mean_v, stdev_v         = m['mean_v'], m['stdev_v']
  output_pred, mse, opt   = m['output_pred'], m['mse'], m['opt']
  S, initial_state, state = m['S'], m['initial_state'], m['state']

  mean_v.load(mean, session=sess)
  stdev_v.load(stdev, session=sess)

  if checkpoint_path:
    saver = tf.train.Saver()

  for _data in data:
    idx = 0
    lstm_state = initial_state
    if len(_data['observations']) % 32 > 0:
      rep = 32 - (len(_data['observations']) % 32)
      mean_stk = np.tile(mean[None, :], [rep, 1])
      output_stk = np.tile(empty_action[None, :], [rep, 1])
      _data['observations'] = np.concatenate((mean_stk, _data['observations']))
      _data['actions'] = np.concatenate((output_stk, _data['actions']))
    while idx < len(_data['observations']):
      input_batch = _data['observations'][idx : idx+batch_size]
      output_batch = _data['actions'][idx : idx+batch_size]

      _, mse_run, lstm_state = sess.run([opt, mse, state], feed_dict={input_ph: input_batch, output_ph: output_batch, S: lstm_state})

      idx += batch_size

  print('epoch: {0:03d} mse: {1:.4f}'.format(curr_epoch, mse_run))
  if checkpoint_path:
    saver.save(sess, checkpoint_path)

  policy_fn = tf_util.function([input_ph, S], [output_pred, state])
  return policy_fn, initial_state, mean

def run(sess, envstr, policy_fn, initial_state, mean, num_rollouts=1,
        debug=False, stats=True):
  env = gym.make(helper.envname(envstr))
  max_steps = env.spec.timestep_limit

  returns = []
  observations = []
  actions = []
  lstm_states = []

  obs = env.reset()
  done = False
  totalr = 0.
  steps = 0
  while not done:
    if steps - 31 >= 0:
      _o = observations[steps-31:steps]
      _obs = np.concatenate((np.array(_o), obs[None, :]))
    else:
      rep = 32 - steps - 1
      mean_stk = np.tile(mean[None, :], [rep, 1])
      _o = np.array(observations[0 : steps])
      if steps > 0:
        _obs = np.concatenate((mean_stk, _o, obs[None, :]))
      else:
        _obs = np.concatenate((mean_stk, obs[None, :]))

    if steps - 32 >= 0:
      lstm_state = lstm_states[steps - 32]
    else:
      lstm_state = initial_state

    action, lstm_state = policy_fn(_obs, lstm_state)
    observations.append(obs)
    actions.append(action[len(action)-1])
    lstm_states.append(lstm_state)
    obs, r, done, _ = env.step(action[len(action)-1])
    totalr += r
    steps += 1
    if debug and steps % 100 == 0:
      print("%i/%i" % (steps, max_steps))
    if steps >= max_steps:
      break
  returns.append(totalr)

  if stats:
    helper.print_returns_stats(returns)

  data = {
    'observations': np.array(observations),
    'actions': np.array(actions),
    'returns': returns,
  }
  return data

def dagger_lstm(env, num_rollouts=1, epochs=1):
  data = helper.run_expert(env, num_rollouts=num_rollouts)
  input_dim = len(data['observations'][0])
  output_dim = len(data['actions'][0])
  model = helper.build_lstm_model(input_dim, output_dim)

  sess = tf.get_default_session()
  sess.run(tf.global_variables_initializer())

  data = [data]
  rewards = []

  os.makedirs('checkpoints', exist_ok=True)

  for epoch in range(epochs):
    checkpoint_path = None
    if epoch == epochs-1:
      checkpoint_path = helper.checkpoint_path(env, 'dagger-lstm-')
    policy_fn, initial_state, mean = train(sess, data, model=model, curr_epoch=epoch, checkpoint_path=checkpoint_path)

    _data = run(sess, env, policy_fn, initial_state, mean,
                num_rollouts=num_rollouts, stats=False)
    _data['actions'] = helper.ask_expert_actions(env, _data['observations'])
    rewards.append(_data['returns'])
    data.append(_data)

  return policy_fn, rewards

def run_dagger_lstm(num_rollouts=10, epochs=200):
  envs = ["humanoid"]

  for env in envs:
    run_dagger_lstm_single_env(env, num_rollouts=num_rollouts, epochs=epochs)

def run_dagger_lstm_single_env(env, num_rollouts=10, epochs=200):
  with tf.Session() as sess:
    with tf.variable_scope(env):
      policy_fn, rewards = dagger_lstm(env, num_rollouts=num_rollouts, epochs=epochs)
      run(sess, env, policy_fn, num_rollouts=10)

      return rewards

def main():
  run_dagger_lstm()

if __name__ == '__main__':
  main()
