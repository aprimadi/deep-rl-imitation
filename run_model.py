import argparse
import tensorflow as tf
import gym

import helper
import tf_util

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('env', type=str)
  parser.add_argument('--model_checkpoint', type=str)
  parser.add_argument('--render', type=bool, default=True)
  parser.add_argument('--max_timesteps', type=int)
  parser.add_argument('--num_rollouts', type=int, default=10)
  args = parser.parse_args()

  with tf.Session() as sess:
    with tf.variable_scope(args.env):
      input_dim, output_dim = helper.input_output_shape(args.env)
      model = helper.build_model(input_dim, output_dim)
      input_ph, output_pred = model['input_ph'], model['output_pred']

      policy_fn = tf_util.function([input_ph], output_pred)

      if args.model_checkpoint:
        checkpoint_path = args.model_checkpoint
      else:
        checkpoint_path = helper.checkpoint_path(args.env)

      saver = tf.train.Saver()
      saver.restore(sess, checkpoint_path)

      env = gym.make(helper.envname(args.env))
      max_steps = args.max_timesteps or env.spec.timestep_limit

      returns = []
      observations = []
      actions = []
      for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0
        steps = 0
        while not done:
          action = policy_fn(obs[None, :])
          observations.append(obs)
          actions.append(action)
          obs, r, done, _ = env.step(action)
          totalr += r
          steps += 1
          if args.render:
            env.render()
          if steps >= max_steps:
            break
        returns.append(totalr)

      helper.print_returns_stats(returns)

if __name__ == '__main__':
  main()
