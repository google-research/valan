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

"""Tests for valan.r2r.agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import numpy as np
import tensorflow.compat.v2 as tf
from valan.framework import common
from valan.framework import hparam
from valan.framework import utils

from valan.r2r import agent
from valan.r2r import agent_config
from valan.r2r import constants
from valan.r2r import env
from valan.r2r import env_config

FLAGS = flags.FLAGS


class AgentR2RTest(tf.test.TestCase):

  def setUp(self):
    super(AgentR2RTest, self).setUp()
    self.num_panos = 36
    self.image_feature_size = 64 + 256
    self.num_actions = 14
    self.time_step = 3
    self.batch_size = 1
    done = np.array([[True], [False], [True]])
    self._test_environment = common.EnvOutput(
        reward=None,
        done=done,
        observation={
            constants.HEADING: np.zeros([self.time_step, self.batch_size, 1]),
            constants.PITCH: np.zeros([self.time_step, self.batch_size, 1]),
            constants.PANO_ENC:
                tf.random.normal([
                    self.time_step, self.batch_size, self.num_panos,
                    self.image_feature_size
                ]),
            constants.CONN_ENC:
                tf.random.normal([
                    self.time_step, self.batch_size, self.num_actions,
                    self.image_feature_size
                ]),
            constants.PREV_ACTION_ENC:
                tf.random.normal([
                    self.time_step, self.batch_size, self.image_feature_size
                ]),
            constants.INS_TOKEN_IDS:
                np.array([
                    [[3, 6, 1, 0, 0]],
                    [[3, 6, 1, 0, 0]],
                    [[3, 6, 1, 0, 0]],
                ]),
            constants.VALID_CONN_MASK:
                np.array([
                    [[1.0] * 14],
                    [[1.0] * 5 + [0.0] * 9],
                    [[1.0] * 2 + [0.0] * 12],
                ])
        },
        info='')

  def test_call_r2r(self):
    self._agent = agent.R2RAgent(agent_config.get_r2r_agent_config())
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
        direction_encoding_dim=256,
        image_features_dir=os.path.join(self.data_dir, 'image_features'),
        instruction_len=50,
        max_agent_actions=6,
        reward_fn=env_config.RewardFunction.get_reward_fn('distance_to_goal'))

    self._runtime_config = common.RuntimeConfig(task_id=0, num_tasks=1)
    self._env = env.R2REnv(
        data_sources=['R2R_small_split'],
        runtime_config=self._runtime_config,
        env_config=self._env_config)

    env_output = self._env.reset()
    observation = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                        env_output.observation)
    initial_agent_state = self._agent.get_initial_state(
        observation, batch_size=1)
    # Agent always expects time,batch dimensions. First add and then remove.
    env_output = utils.add_time_batch_dim(env_output)
    agent_output, _ = self._agent(env_output, initial_agent_state)

    self.assertEqual(agent_output.policy_logits.shape, [1, 1, 14])
    self.assertEqual(agent_output.baseline.shape, [1, 1])

    initial_agent_state = ([
        (tf.random.normal([self.batch_size,
                           512]), tf.random.normal([self.batch_size, 512])),
        (tf.random.normal([self.batch_size,
                           512]), tf.random.normal([self.batch_size, 512]))
    ], tf.random.normal([self.batch_size, 5, 512]))
    agent_output, _ = self._agent(self._test_environment, initial_agent_state)

    self.assertEqual(agent_output.policy_logits.shape,
                     [self.time_step, self.batch_size, 14])
    self.assertEqual(agent_output.baseline.shape,
                     [self.time_step, self.batch_size])

  def test_call_ndh(self):
    self._agent = agent.R2RAgent(agent_config.get_ndh_agent_config())
    self.data_dir = FLAGS.test_srcdir + (
        'valan/r2r/testdata')

    self._env_config = hparam.HParams(
        problem='NDH',
        history='all',
        path_type='trusted_path',
        max_goal_room_panos=4,
        scan_base_dir=self.data_dir,
        data_base_dir=self.data_dir,
        vocab_dir=self.data_dir,
        problem_path=os.path.join(self.data_dir, 'NDH'),
        vocab_file='vocab.txt',
        images_per_pano=36,
        max_conns=14,
        image_encoding_dim=64,
        direction_encoding_dim=256,
        image_features_dir=os.path.join(self.data_dir, 'image_features'),
        instruction_len=50,
        max_agent_actions=6,
        reward_fn=env_config.RewardFunction.get_reward_fn('distance_to_goal'))

    self._runtime_config = common.RuntimeConfig(task_id=0, num_tasks=1)
    self._env = env.R2REnv(
        data_sources=['R2R_small_split'],
        runtime_config=self._runtime_config,
        env_config=self._env_config)

    env_output = self._env.reset()
    observation = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                        env_output.observation)
    initial_agent_state = self._agent.get_initial_state(
        observation, batch_size=1)
    # Agent always expects time,batch dimensions. First add and then remove.
    env_output = utils.add_time_batch_dim(env_output)
    agent_output, _ = self._agent(env_output, initial_agent_state)

    self.assertEqual(agent_output.policy_logits.shape, [1, 1, 14])
    self.assertEqual(agent_output.baseline.shape, [1, 1])

    initial_agent_state = ([
        (tf.random.normal([self.batch_size,
                           512]), tf.random.normal([self.batch_size, 512])),
        (tf.random.normal([self.batch_size,
                           512]), tf.random.normal([self.batch_size, 512]))
    ], tf.random.normal([self.batch_size, 5, 512]))
    agent_output, _ = self._agent(self._test_environment, initial_agent_state)

    self.assertEqual(agent_output.policy_logits.shape,
                     [self.time_step, self.batch_size, 14])
    self.assertEqual(agent_output.baseline.shape,
                     [self.time_step, self.batch_size])


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
