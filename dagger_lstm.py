import os
import tensorflow as tf

import helper
import dagger

def train(sess, data, model, curr_epoch, batch_size=32, debug=False, checkpoint_path=None):
  obs = None:
  for _data in data:
    if obs is None:
      obs = _data['observations']
    else:
      obs = np.concatenate((obs, _data['observations']))
  mean, stdev = helper.mean_and_stdev(obs)

  m = model
  input_ph, output_ph   = m['input_ph'], m['output_ph']
  mean_v, stdev_v       = m['mean_v'], m['stdev_v']
  output_pred, mse, opt = m['output_pred'], m['mse'], m['opt']
  S, initial_state      = m['S'], m['initial_state']

  mean_v.load(mean, session=sess)
  stdev_v.load(stdev, session=sess)

  if checkpoint_path:
    saver = tf.train.Saver()

  for _data in data:
    idx = 0
    lstm_state = initial_state
    while idx < len(_data['observations']):
      input_batch = _data['observations'][idx : idx+batch_size]
      output_batch = _data['actions'][idx : idx+batch_size]

      _, mse_run, lstm_state = sess.run([opt, mse, S], feed_dict={input_ph: input_batch, output_ph: output_batch, S: lstm_state})

      print('epoch: {0:03d} mse: {1:.4f}'.format(curr_epoch, mse_run))

      idx += batch_size
    if checkpoint_path:
      saver.save(sess, checkpoint_path)

def dagger_with_lstm(env, num_rollouts=1, epochs=1):
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
    policy_fn = train(sess, data, model=model, curr_epoch=epoch, epochs=1, checkpoint_path=checkpoint_path)

    _data = run(sess, env, policy_fn, num_rollouts=num_rollouts, stats=False)
    _data['actions'] = helper.ask_expert_actions(env, _data['observations'])
    rewards.append(_data['returns'])
    data.append(_data)

  return policy_fn, rewards

def run_dagger_lstm(num_rollouts=10, epochs=200):
  envs = ["ant", "half_cheetah", "hopper", "humanoid", "reacher", "walker"]

  for env in envs:
    run_dagger_single_env(env, num_rollouts=num_rollouts, epochs=epochs)

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
