import argparse

import helper

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('envname', type=str)
  parser.add_argument('--model_checkpoint', type=str)
  parser.add_argument('--render', action='store_true')
  parser.add_argument('--max_timesteps', type=int)
  parser.add_argument('--num_rollouts', type=int, default=20)
  args = parser.parse_args()

  with tf.Session() as sess:
    model = helper.build_model()

    saver = tf.train.Saver()
    saver.restore(sess, helper.checkpoint_path(args.envname))

    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    obserations = []
    actions = []
    for i in range(args.num_rollouts):
      print('iter', i)
      obs = env.reset()
      done = False
      totalr = 0
      steps = 0
      while not done:
        pass

if __name__ == '__main__':
  main()
