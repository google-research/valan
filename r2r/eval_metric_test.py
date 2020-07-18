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

"""Tests for valan.r2r.eval_metric."""

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
from valan.r2r import eval_metric


FLAGS = flags.FLAGS


class EvalMetricTest(tf.test.TestCase):

  def setUp(self):
    super(EvalMetricTest, self).setUp()

    self.data_dir = FLAGS.test_srcdir + (
        'valan/r2r/testdata')

    self._env_config = hparam.HParams(
        problem='R2R',
        scan_base_dir=self.data_dir,
        data_base_dir=self.data_dir,
        vocab_dir=self.data_dir,
        vocab_file='vocab.txt',
        images_per_pano=36,
        max_conns=14,
        image_encoding_dim=64,
        image_features_dir=os.path.join(self.data_dir, 'image_features'),
        instruction_len=50,
        max_agent_actions=6,
        reward_fn=env_config.RewardFunction.get_reward_fn('distance_to_goal'))

    self._runtime_config = common.RuntimeConfig(task_id=0, num_tasks=1)
    self._env = env.R2REnv(
        data_sources=['R2R_small_split'],
        runtime_config=self._runtime_config,
        env_config=self._env_config)

    # scan: gZ6f7yhEvPG
    # Path: 1, 3, 7, 5, 2
    self._golden_path = [1, 4, 6, 2]
    self._scan_id = 0  # testdata has single scan only 'gZ6f7yhEvPG'
    self._env_list = [
        common.EnvOutput(
            reward=0,
            done=None,
            observation={
                constants.PANO_ID: 1,
                constants.GOLDEN_PATH: self._golden_path,
                constants.GOAL_PANO_ID: 2,
                constants.SCAN_ID: self._scan_id,
                constants.GOAL_ROOM_PANOS: [6, 2, constants.INVALID_NODE_ID]
            },
            info=None),
        common.EnvOutput(
            reward=1,
            done=None,
            observation={
                constants.PANO_ID: 3,
                constants.GOLDEN_PATH: self._golden_path,
                constants.GOAL_PANO_ID: 2,
                constants.SCAN_ID: self._scan_id,
                constants.GOAL_ROOM_PANOS: [6, 2, constants.INVALID_NODE_ID]
            },
            info=None),
        common.EnvOutput(
            reward=1,
            done=None,
            observation={
                constants.PANO_ID: 7,
                constants.GOLDEN_PATH: self._golden_path,
                constants.GOAL_PANO_ID: 2,
                constants.SCAN_ID: self._scan_id,
                constants.GOAL_ROOM_PANOS: [6, 2, constants.INVALID_NODE_ID]
            },
            info=None),
        common.EnvOutput(
            reward=1,
            done=None,
            observation={
                constants.PANO_ID: 5,
                constants.GOLDEN_PATH: self._golden_path,
                constants.GOAL_PANO_ID: 2,
                constants.SCAN_ID: self._scan_id,
                constants.GOAL_ROOM_PANOS: [6, 2, constants.INVALID_NODE_ID]
            },
            info=None),
        common.EnvOutput(
            reward=1,
            done=False,
            observation={
                constants.PANO_ID: 2,
                constants.GOLDEN_PATH: self._golden_path,
                constants.GOAL_PANO_ID: 2,
                constants.SCAN_ID: self._scan_id,
                constants.GOAL_ROOM_PANOS: [6, 2, constants.INVALID_NODE_ID]
            },
            info=None),
        common.EnvOutput(
            reward=4,  # success
            done=True,  # end of episode
            # next episode's observation.
            observation={
                constants.PANO_ID: 11,
                constants.GOLDEN_PATH: self._golden_path,
                constants.GOAL_PANO_ID: 2,
                constants.SCAN_ID: self._scan_id,
                constants.GOAL_ROOM_PANOS: [6, 2, constants.INVALID_NODE_ID]
            },
            info=None),
    ]
    self._action_list = [3, 7, 5, 2, 0]

  def test_num_steps_before_stop(self):
    # Stop at first action.
    self.assertEqual(
        0,
        eval_metric.get_num_steps_before_stop(
            action_list=[0],
            env_output_list=self._env_list[:2],
            environment=self._env))
    # Stop after taking 4 actions.
    self.assertEqual(
        2,
        eval_metric.get_num_steps_before_stop(
            action_list=self._action_list[:2],
            env_output_list=self._env_list[:3],
            environment=self._env))
    self.assertEqual(
        4,
        eval_metric.get_num_steps_before_stop(
            action_list=self._action_list[:-1],
            env_output_list=self._env_list[:-1],
            environment=self._env))
    # Last stop action is not counted towards number of steps taken.
    self.assertEqual(
        4,
        eval_metric.get_num_steps_before_stop(
            action_list=self._action_list,
            env_output_list=self._env_list,
            environment=self._env))

  def test_navigation_error(self):
    result = eval_metric.get_navigation_error(self._action_list, self._env_list,
                                              self._env)
    self.assertEqual(result, 0.0)
    result = eval_metric.get_navigation_error(self._action_list[:-2],
                                              self._env_list[:-2], self._env)
    self.assertAlmostEqual(result, 2.9398794637365659)

  def test_dtw(self):
    result = eval_metric.get_dtw(self._action_list, self._env_list, self._env)
    expected_result = (
        self._env.get_distance(1, 3, self._scan_id) +
        self._env.get_distance(4, 7, self._scan_id) +
        self._env.get_distance(6, 5, self._scan_id))
    golden_path_length = eval_metric._get_path_length(self._golden_path,
                                                      self._scan_id, self._env)
    self.assertAlmostEqual(result, expected_result / golden_path_length)

  def test_norm_dtw(self):
    result = eval_metric.get_norm_dtw(self._action_list, self._env_list,
                                      self._env)
    dtw = (
        self._env.get_distance(1, 3, self._scan_id) +
        self._env.get_distance(4, 7, self._scan_id) +
        self._env.get_distance(6, 5, self._scan_id))
    expected_result = np.exp(-1. * dtw / (3. * len(self._golden_path)))
    self.assertAlmostEqual(result, expected_result)

  def test_sdtw(self):
    # If not successful, sdtw is always zero.
    result = eval_metric.get_sdtw(self._action_list[:2], self._env_list[:3],
                                  self._env)
    self.assertEqual(0., result)
    # If successful, sdtw is equal to norm_dtw.
    result = eval_metric.get_sdtw(self._action_list, self._env_list, self._env)
    dtw = (
        self._env.get_distance(1, 3, self._scan_id) +
        self._env.get_distance(4, 7, self._scan_id) +
        self._env.get_distance(6, 5, self._scan_id))
    expected_result = np.exp(-1. * dtw / (3. * len(self._golden_path)))
    self.assertAlmostEqual(result, expected_result)

  def test_cls(self):
    result = eval_metric.get_cls(self._action_list, self._env_list, self._env)
    self.assertAlmostEqual(result, 0.36422316181585335)

  def test_success_rate(self):

    result = eval_metric.get_success_rate(self._action_list, self._env_list,
                                          self._env)
    self.assertEqual(result, 1.0)
    result = eval_metric.get_success_rate(self._action_list[:2],
                                          self._env_list[:3], self._env)
    self.assertEqual(result, 0.0)

  def test_spl(self):
    # Shortest path from node_id=1 to node_id=2 is of length 5.34029839056
    # while length of path 1 --> 3 --> 7 --> 5 --> 2 is 9.78930294617
    result = eval_metric.get_spl(self._action_list, self._env_list, self._env)
    self.assertAlmostEqual(result, 5.34029839056 / 9.78930294617)
    result = eval_metric.get_spl(self._action_list[:2], self._env_list[:3],
                                 self._env)
    self.assertEqual(result, 0.0)

  def test_path_length(self):
    result = eval_metric.get_path_length(self._action_list, self._env_list,
                                         self._env)
    self.assertAlmostEqual(result, 9.7893029461736418)

  def test_oracle_success(self):

    result = eval_metric.get_oracle_success(self._action_list, self._env_list,
                                            self._env)
    self.assertEqual(result, 1.0)

  def test_undisc_episode_reward(self):
    result = eval_metric.get_undisc_episode_reward(self._action_list,
                                                   self._env_list, self._env)
    self.assertEqual(8., result)

  def test_visualization_image(self):
    result = eval_metric.get_visualization_image(self._action_list,
                                                 self._env_list, self._env)
    self.assertEqual(3, result[0].shape[2])  # always an RGB

  def test_goal_progress(self):
    result = eval_metric.get_goal_progress(self._action_list, self._env_list,
                                           self._env)
    start_dis = self._env.get_distance(1, 6, self._scan_id)
    end_dis = self._env.get_distance(2, 2, self._scan_id)
    self.assertEqual(result, start_dis - end_dis)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
