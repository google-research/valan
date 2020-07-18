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
from valan.r2r import env_ndh
from valan.r2r import env_ndh_config as env_config
from valan.r2r import env_test

FLAGS = flags.FLAGS


class NDHEnvTest(tf.test.TestCase):

  def setUp(self):
    super(NDHEnvTest, self).setUp()
    self.data_dir = FLAGS.test_srcdir + (
        'valan/r2r/testdata')

    self.reward_fn_type = 'distance_to_goal'
    self._env_config = hparam.HParams(
        problem='NDH',
        history='all',
        path_type='trusted_path',
        max_goal_room_panos=4,
        scan_base_dir=self.data_dir,
        data_base_dir=os.path.join(self.data_dir, 'NDH'),
        vocab_file='vocab.txt',
        images_per_pano=36,
        max_conns=14,
        image_encoding_dim=64,
        direction_encoding_dim=256,
        image_features_dir=os.path.join(self.data_dir, 'image_features'),
        instruction_len=50,
        max_agent_actions=6,
        reward_fn_type=self.reward_fn_type,
        reward_fn=env_config.RewardFunction.get_reward_fn(self.reward_fn_type))
    self._runtime_config = common.RuntimeConfig(task_id=0, num_tasks=1)

    self._env = env_ndh.NDHEnv(
        data_sources=['small_split'],
        runtime_config=self._runtime_config,
        env_config=self._env_config)

    # For deterministic behavior in test
    np.random.seed(0)

  def _get_pano_id(self, pano_name, scan_id):
    return self._env._scan_info[scan_id].pano_name_to_id[pano_name]

  def testStepToGoalRoom(self):
    self.reward_fn_type = 'distance_to_room'
    self._env_config = hparam.HParams(
        problem='NDH',
        history='all',
        path_type='trusted_path',
        max_goal_room_panos=4,
        scan_base_dir=self.data_dir,
        data_base_dir=os.path.join(self.data_dir, 'NDH'),
        vocab_file='vocab.txt',
        images_per_pano=36,
        max_conns=14,
        image_encoding_dim=64,
        direction_encoding_dim=256,
        image_features_dir=os.path.join(self.data_dir, 'image_features'),
        instruction_len=50,
        max_agent_actions=6,
        reward_fn_type=self.reward_fn_type,
        reward_fn=env_config.RewardFunction.get_reward_fn(self.reward_fn_type))
    self._runtime_config = common.RuntimeConfig(task_id=0, num_tasks=1)

    self._env = env_ndh.NDHEnv(
        data_sources=['small_split'],
        runtime_config=self._runtime_config,
        env_config=self._env_config)

    scan_id = 0  # testdata only has single scan 'gZ6f7yhEvPG'
    _ = self._env.reset()
    golden_path = [
        'ba27da20782d4e1a825f0a133ad84da9',
        '47d8a8282c1c4a7fb3eeeacc45e9d959',  # in the goal room
        '0ee20663dfa34b438d48750ddcd7366c'  # in the goal room
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
      expected_reward = 1 if i <= 1 else 0
      env_test.verify_env_output(
          self,
          self._env.step(action),
          expected_reward=expected_reward,  #  Moving towards goal.
          expected_done=False,
          expected_info='',
          expected_time_step=expected_time_step,
          expected_path_id=318,
          expected_pano_name=golden_path[i],
          expected_heading=expected_heading,
          expected_pitch=expected_pitch,
          expected_scan_id=scan_id,
          expected_oracle_action=expected_oracle_action)

    # Stop at goal pano. Terminating the episode results in resetting the
    # observation to next episode.
    env_test.verify_env_output(
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

  def testGetAllPaths(self):

    def _get_all_paths(history):
      return env_ndh._get_all_paths_ndh(
          data_sources=['small_split'],
          data_base_dir=os.path.join(self.data_dir, 'NDH'),
          vocab_file='vocab.txt',
          fixed_instruction_len=50,
          history=history,
          path_type='trusted_path')

    # <PAD> is 0, <UNK> is 1, <NAV> is 3, <ORA> is 4, <TAR> is 5.
    all_paths = _get_all_paths('none')
    self.assertEqual(0, all_paths[0]['instruction_len'])
    np.testing.assert_array_equal(all_paths[0]['instruction_token_ids'],
                                  [0] * 50)

    all_paths = _get_all_paths('target')
    self.assertEqual(2, all_paths[0]['instruction_len'])
    np.testing.assert_array_equal(all_paths[0]['instruction_token_ids'],
                                  [5, 66] + [0] * 48)

    all_paths = _get_all_paths('oracle_ans')
    self.assertEqual(9, all_paths[0]['instruction_len'])
    np.testing.assert_array_equal(
        all_paths[0]['instruction_token_ids'],
        [
            4, 87, 91, 86, 97, 121, 66,  # ora_ans
            5, 66,  # target
        ] + [0] * 41)

    all_paths = _get_all_paths('nav_q_oracle_ans')
    self.assertEqual(18, all_paths[0]['instruction_len'])
    np.testing.assert_array_equal(
        all_paths[0]['instruction_token_ids'],
        [
            3, 254, 88, 122, 1, 90, 87, 91, 89,  # nav_q
            4, 87, 91, 86, 97, 121, 66,  # ora_ans
            5, 66,  # target
        ] + [0] * 32)

    all_paths = _get_all_paths('all')
    self.assertEqual(32, all_paths[0]['instruction_len'])
    np.testing.assert_array_equal(
        all_paths[0]['instruction_token_ids'],
        [
            3, 254, 88, 142, 97, 118, 90, 221, 91, 87, 91, 89,  # nav_q
            4, 299,  # ora_ans
            3, 254, 88, 122, 1, 90, 87, 91, 89,  # nav_q
            4, 87, 91, 86, 97, 121, 66,  # ora_ans
            5, 66,  # target
        ] + [0] * 18)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
