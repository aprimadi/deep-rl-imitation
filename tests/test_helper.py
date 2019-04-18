import pytest
import tensorflow as tf
import numpy as np

import helper

def test_envname():
  assert helper.envname("ant") == "Ant-v2"
  assert helper.envname("half_cheetah") == "HalfCheetah-v2"

def test_expert_policy_file():
  assert helper.expert_policy_file("ant") == "experts/Ant-v2.pkl"

def test_checkpoint_path():
  assert helper.checkpoint_path("ant") == "checkpoints/Ant-v2.ckpt"

def test_build_model():
  with tf.Session():
    with tf.variable_scope("test_build_model"):
      m = helper.build_model(11, 3)
      assert m != None

def test_build_model_input_norm():
  with tf.Session() as sess:
    with tf.variable_scope("test_build_model_input_norm"):
      m = helper.build_model(5, 3)
      input_ph, input_norm = m['input_ph'], m['input_norm']
      mean_v, stdev_v = m['mean_v'], m['stdev_v']
      sess.run(tf.global_variables_initializer())
      mean_v.load([0.5, 0.5, 0.5, 0.5, 0.5], session=sess)
      stdev_v.load([1, 1, 1, 1, 1], session=sess)
      values, = sess.run([input_norm], feed_dict={input_ph: [
        [1, 1, 1, 1, 1],
        [1, 2, 3, 4, 5],
      ]})
      expected_values = [[
        [0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 1.5, 2.5, 3.5, 4.5],
      ]]
      assert (values - np.array(expected_values) < 0.01).all()

if __name__ == '__main__':
  test_envname()
  test_expert_policy_file()
  test_checkpoint_path()
  test_build_model()
  test_build_model_input_norm()
