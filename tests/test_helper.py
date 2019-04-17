import pytest

import helper

def test_envname():
  assert helper.envname("ant") == "Ant-v2"
  assert helper.envname("half_cheetah") == "HalfCheetah-v2"

def test_expert_policy_file():
  assert helper.expert_policy_file("ant") == "experts/Ant-v2.pkl"

def test_checkpoint_path():
  assert helper.checkpoint_path("ant") == "checkpoints/Ant-v2.ckpt"

def test_build_model():
  model = helper.build_model(11, 3)
  assert model != None

if __name__ == '__main__':
  test_envname()
  test_expert_policy_file()
  test_checkpoint_path()