# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for valan.r2r.env."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import numpy as np
import tensorflow.compat.v2 as tf
from valan.framework import common
from valan.framework import hparam

from valan.r2r import constants
from valan.r2r import env
from valan.r2r import env_config
from valan.r2r import image_features_pb2

FLAGS = flags.FLAGS


class EnvTest(tf.test.TestCase):

  def setUp(self):
    super(EnvTest, self).setUp()
    tf.enable_v2_behavior()
    self.data_dir = FLAGS.test_srcdir + (
        'valan/r2r/testdata')

    self._env_config = hparam.HParams(
        problem='R2R',
        scan_base_dir=self.data_dir,
        data_base_dir=self.data_dir,
        vocab_file='vocab.txt',
        images_per_pano=36,
        max_conns=14,
        image_encoding_dim=2048 + 4,
        image_features_dir=os.path.join(self.data_dir, 'image_features'),
        instruction_len=50,
        max_agent_actions=6,
        reward_fn=env_config.RewardFunction.get_reward_fn('distance_to_goal'))
    self._runtime_config = common.RuntimeConfig(task_id=0, num_tasks=1)

    self._env = env.R2REnv(
        data_sources=['R2R_small_split'],
        runtime_config=self._runtime_config,
        env_config=self._env_config)

    # For deterministic behavior in test
    np.random.seed(0)

  def _get_pano_id(self, pano_name, scan_id):
    return self._env._scan_info[scan_id].pano_name_to_id[pano_name]

  def testDenseDtw(self):
    scan_id = 0  # testdata only has single scan 'gZ6f7yhEvPG'
    scan_info = self._env._scan_info[scan_id]
    path_names = [
        'ba27da20782d4e1a825f0a133ad84da9', '47d8a8282c1c4a7fb3eeeacc45e9d959',
        '0ee20663dfa34b438d48750ddcd7366c'
    ]
    path = [self._get_pano_id(name, scan_id) for name in path_names]
    distance_0_1 = self._env.get_distance(path[0], path[1], scan_id)
    distance_0_2 = self._env.get_distance(path[0], path[2], scan_id)
    distance_1_2 = self._env.get_distance(path[1], path[2], scan_id)
    expected_dtw_rewards = [((distance_0_1 + distance_0_2) - distance_1_2),
                            (distance_1_2 - 0.)]
    # Step through the trajectory and verify the rewards.
    for i in range(len(expected_dtw_rewards)):
      path_history = path[:i + 1]
      next_pano = path[i + 1]
      dtw = env_config.dense_dtw(path_history, next_pano, path_names, False,
                                 scan_info)
      self.assertAlmostEqual(dtw, expected_dtw_rewards[i])

  def testReset(self):
    scan_id = 0  # testdata only has single scan 'gZ6f7yhEvPG'
    verify_env_output(
        self,
        self._env.reset(),
        expected_reward=0,
        expected_done=False,
        expected_info='',
        expected_time_step=0,
        expected_path_id=1304,
        expected_pano_name='80929af5cf234ae38ac3a2a4e60e4342',
        expected_heading=6.101,
        expected_pitch=0.,
        expected_scan_id=scan_id,
        expected_oracle_action=self._get_pano_id(
            'ba27da20782d4e1a825f0a133ad84da9', scan_id))

  def testStep(self):
    scan_id = 0  # testdata only has single scan 'gZ6f7yhEvPG'
    _ = self._env.reset()
    golden_path = [
        'ba27da20782d4e1a825f0a133ad84da9', '47d8a8282c1c4a7fb3eeeacc45e9d959',
        '46cecea0b30e4786b673f5e951bf82d4'
    ]
    # Step through the trajectory and verify the env_output.
    for i, action in enumerate(
        [self._get_pano_id(p, scan_id) for p in golden_path]):
      expected_time_step = i + 1
      expected_heading, expected_pitch = self._env._get_heading_pitch(
          action, scan_id, expected_time_step)
      if i + 1 < len(golden_path):
        expected_oracle_action = self._get_pano_id(golden_path[i + 1], scan_id)
      else:
        expected_oracle_action = constants.STOP_NODE_ID
      verify_env_output(
          self,
          self._env.step(action),
          expected_reward=1,  #  Moving towards goal.
          expected_done=False,
          expected_info='',
          expected_time_step=expected_time_step,
          expected_path_id=1304,
          expected_pano_name=golden_path[i],
          expected_heading=expected_heading,
          expected_pitch=expected_pitch,
          expected_scan_id=scan_id,
          expected_oracle_action=expected_oracle_action)

    # Stop at goal pano. Terminating the episode results in resetting the
    # observation to next episode.
    verify_env_output(
        self,
        self._env.step(constants.STOP_NODE_ID),
        expected_reward=4,  # reached goal and stopped
        expected_done=True,  # end of episode
        expected_info='',
        # observation for next episode.
        expected_time_step=0,
        expected_path_id=1304,
        expected_pano_name='80929af5cf234ae38ac3a2a4e60e4342',
        expected_heading=6.101,
        expected_pitch=0.,
        expected_scan_id=scan_id,
        expected_oracle_action=self._get_pano_id(
            'ba27da20782d4e1a825f0a133ad84da9', scan_id))

  def testGetDistance(self):
    scan_id = 0  # testdata only has single scan 'gZ6f7yhEvPG'
    self.assertAlmostEqual(
        self._env.get_distance(2, 5, scan_id), 2.9398794637365659)

  def testGetRelEnc(self):
    init_heading = np.pi / 2.
    init_pitch = np.pi
    enc = np.array([
        np.sin(init_heading),
        np.cos(init_heading),
        np.sin(init_pitch),
        np.cos(init_pitch), 5., 6., 7., 8.
    ])
    final_heading = np.pi
    final_pitch = 0.
    expected_enc = np.array([
        np.sin(init_heading - final_heading),
        np.cos(init_heading - final_heading),
        np.sin(init_pitch - final_pitch),
        np.cos(init_pitch - final_pitch), 5., 6., 7., 8.
    ])
    np.testing.assert_array_almost_equal(
        np.reshape(expected_enc, (1, 8)),
        env.get_rel_enc(np.reshape(enc, [1, 8]), final_heading, final_pitch))

  def testRandom(self):
    scan_id = 0  # testdata only has single scan 'gZ6f7yhEvPG'
    scan_info = self._env._scan_info[scan_id]
    path_names = [
        'ba27da20782d4e1a825f0a133ad84da9', '47d8a8282c1c4a7fb3eeeacc45e9d959',
        '0ee20663dfa34b438d48750ddcd7366c'
    ]
    path = [self._get_pano_id(name, scan_id) for name in path_names]
    path_history = path[0]
    next_pano = path[1]
    np.random.seed(0)
    expected_random_rwd = np.random.uniform(-1, 1)
    np.random.seed(0)
    random_rwd = env_config.random_reward(path_history, next_pano, path_names,
                                          False, scan_info)
    self.assertLessEqual(random_rwd, 1)
    self.assertEqual(random_rwd, expected_random_rwd)

  def testRandomGoal(self):
    scan_id = 0  # testdata only has single scan 'gZ6f7yhEvPG'
    scan_info = self._env._scan_info[scan_id]
    path_names = [
        'ba27da20782d4e1a825f0a133ad84da9', '47d8a8282c1c4a7fb3eeeacc45e9d959',
        '0ee20663dfa34b438d48750ddcd7366c'
    ]
    path = [self._get_pano_id(name, scan_id) for name in path_names]
    path_history = path[:1]
    next_pano = path[1]
    goal_random_rwd = env_config.goal_plus_random_reward(
        path_history, next_pano, path_names, False, scan_info)
    self.assertLessEqual(goal_random_rwd, 4)


def _get_encoding(filename, expected_size):
  protos = []
  # Pretend to read more than 1 values and then verify there is exactly 1.
  for record in tf.data.TFRecordDataset([filename]).take(2):
    protos.append(record.numpy())
  assert len(protos) == 1
  parsed_record = image_features_pb2.ImageFeatures()
  parsed_record.ParseFromString(protos[0])
  enc = np.array(parsed_record.value)
  if np.size(enc) != expected_size:
    enc = np.pad(enc, (0, expected_size - np.size(enc)), mode='constant')
  return enc


def verify_env_output(test, env_output, expected_reward, expected_done,
                      expected_info, expected_time_step, expected_path_id,
                      expected_pano_name, expected_heading, expected_pitch,
                      expected_scan_id, expected_oracle_action):
  test.assertEqual(expected_reward, env_output.reward)
  test.assertEqual(expected_done, env_output.done)
  test.assertEqual(expected_info, env_output.info)
  obs = env_output.observation
  test.assertEqual(expected_time_step, obs[constants.TIME_STEP])
  test.assertEqual(expected_path_id, obs[constants.PATH_ID])
  test.assertEqual(
      test._env._scan_info[expected_scan_id]
      .pano_name_to_id[expected_pano_name], obs[constants.PANO_ID])
  test.assertEqual(expected_heading, obs[constants.HEADING])
  test.assertEqual(expected_pitch, obs[constants.PITCH])
  test.assertEqual(expected_scan_id, obs[constants.SCAN_ID])
  test.assertEqual(expected_oracle_action, obs[constants.ORACLE_NEXT_ACTION])
  instr_len = (obs[constants.INS_TOKEN_IDS] > 0).sum()
  test.assertEqual(instr_len, obs[constants.INS_LEN])
  valid_conn_mask = obs[constants.CONN_IDS] >= 0
  np.testing.assert_array_equal(valid_conn_mask, obs[constants.VALID_CONN_MASK])
  raw_pano_enc = _get_encoding(
      os.path.join(test._env_config.image_features_dir,
                   '{}_viewpoints_proto'.format(expected_pano_name)), 2052 * 36)
  np.testing.assert_array_almost_equal(
      env.get_rel_enc(
          np.reshape(raw_pano_enc, (36, 2052)), expected_heading,
          expected_pitch), obs[constants.PANO_ENC])
  raw_conn_enc = _get_encoding(
      os.path.join(test._env_config.image_features_dir,
                   '{}_connections_proto'.format(expected_pano_name)),
      2052 * 14)
  np.testing.assert_array_almost_equal(
      env.get_rel_enc(
          np.reshape(raw_conn_enc, (14, 2052)), expected_heading,
          expected_pitch), obs[constants.CONN_ENC])


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
