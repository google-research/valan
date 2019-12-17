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

"""Tests for valan.r2r.discriminator_agent."""

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
from valan.r2r import agent_config
from valan.r2r import constants
from valan.r2r import discriminator_agent
from valan.r2r import env
from valan.r2r import env_config

FLAGS = flags.FLAGS


class DiscriminatorTest(tf.test.TestCase):

  def setUp(self):
    super(DiscriminatorTest, self).setUp()
    self.data_dir = FLAGS.test_srcdir + (
        'valan/r2r/testdata')
    self._env_config = hparam.HParams(
        problem='R2R',
        base_path=self.data_dir,
        vocab_file='vocab.txt',
        images_per_pano=36,
        max_conns=14,
        image_encoding_dim=2052,
        image_features_dir=os.path.join(self.data_dir, 'image_features'),
        instruction_len=50,
        max_agent_actions=6,
        reward_fn=env_config.RewardFunction.get_reward_fn('distance_to_goal'))

    self._runtime_config = common.RuntimeConfig(task_id=0, num_tasks=100)

    self._env = env.R2REnv(
        data_sources=['small_split'],
        runtime_config=self._runtime_config,
        env_config=self._env_config)
    self.num_panos = 36
    self.image_feature_size = 2052
    self.num_actions = 14
    self.time_step = 3
    self.batch_size = 1
    done = np.array([[True], [False], [True]])
    done = np.reshape(done, [3, 1])
    self._test_environment = common.EnvOutput(
        reward=0,
        done=done,
        observation={
            constants.IS_START:
                np.array([[True], [False], [True]]),
            constants.DISC_MASK:
                np.array([[True], [False], [True]]),
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
            constants.INS_TOKEN_IDS:
                np.array([[[3, 6, 1, 0, 0]], [[3, 6, 1, 0, 0]],
                          [[3, 6, 1, 0, 0]]]),
            constants.VALID_CONN_MASK:
                np.array([[[True] * 14], [[True] * 5 + [False] * 9],
                          [[True] * 2 + [False] * 12]])
        },
        info='')
    self._agent = discriminator_agent.DiscriminatorAgent(
        agent_config.get_r2r_agent_config())

  def test_call(self):
    env_output = self._env.reset()
    observation = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                        env_output.observation)
    initial_agent_state = self._agent.get_initial_state(
        observation, batch_size=1)
    # Agent always expects time,batch dimensions. First add and then remove.
    env_output = utils.add_time_batch_dim(env_output)
    agent_output, _ = self._agent(env_output, initial_agent_state)
    initial_agent_state = ([
        (tf.random.normal([self.batch_size,
                           512]), tf.random.normal([self.batch_size, 512])),
        (tf.random.normal([self.batch_size,
                           512]), tf.random.normal([self.batch_size, 512]))
    ], tf.random.normal([self.batch_size, 5, 512]))
    agent_output, _ = self._agent(self._test_environment, initial_agent_state)
    self.assertEqual(agent_output.policy_logits.shape, [3, 1, 1])


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
