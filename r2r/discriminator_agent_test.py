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
from absl.testing import parameterized
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


class DiscriminatorTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(DiscriminatorTest, self).setUp()
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
        project_decoder_input_states=True,
        use_all_final_states=False,
        reward_fn=env_config.RewardFunction.get_reward_fn('distance_to_goal'))

    self._runtime_config = common.RuntimeConfig(task_id=0, num_tasks=100)
    self._env = env.R2REnv(
        data_sources=['R2R_small_split'],
        runtime_config=self._runtime_config,
        env_config=self._env_config)
    self.num_panos = 36
    self.image_feature_size = 64
    self.direction_encoding_dim = 256
    self.num_actions = 14
    self.time_step = 3
    self.batch_size = 2
    done = np.array([[False, True], [True, False], [True, False]])
    self._test_environment = common.EnvOutput(
        reward=0,
        done=done,
        observation={
            constants.PATH_ID:  # Shape = [time, batch]
                np.array([[2, 1], [0, 1], [0, 1]]),
            constants.IS_START:  # Shape = [time, batch]
                np.array([[False, True], [True, False], [False, False]]),
            constants.DISC_MASK:  # Shape = [time, batch]
                np.array([[False, True], [True, True], [True, True]]),
            constants.PANO_ENC:  # Shape = [time, batch, num_panos, featur_size]
                tf.random.normal([
                    self.time_step, self.batch_size, self.num_panos,
                    self.image_feature_size + self.direction_encoding_dim
                ]),
            constants.CONN_ENC:
                # Shape = [time, batch, num_actions, feature_size]
                tf.random.normal([
                    self.time_step, self.batch_size, self.num_actions,
                    self.image_feature_size + self.direction_encoding_dim
                ]),
            constants.PREV_ACTION_ENC:
                # Shape = [time, batch, feature_size]
                tf.random.normal([
                    self.time_step, self.batch_size,
                    self.image_feature_size + self.direction_encoding_dim
                ]),
            constants.NEXT_GOLDEN_ACTION_ENC:
                # Shape = [time, batch, feature_size]
                tf.random.normal([
                    self.time_step, self.batch_size,
                    self.image_feature_size + self.direction_encoding_dim
                ]),
            constants.INS_TOKEN_IDS:  # Shape = [time, batch, token_len]
                np.array([[[5, 3, 2, 1, 0], [3, 4, 5, 6, 1]],
                          [[3, 6, 1, 0, 0], [3, 4, 5, 6, 1]],
                          [[3, 6, 1, 0, 0], [3, 4, 5, 6, 1]]]),
            constants.INS_LEN:  # Shape = [time, batch]
                np.tile(np.array([[3]]), [self.time_step, self.batch_size]),
            constants.VALID_CONN_MASK:
                # Shape = [time, batch, num_connections]
                np.tile(
                    np.array([[[True] * 14], [[True] * 5 + [False] * 9],
                              [[True] * 2 + [False] * 12]]),
                    [1, self.batch_size, 1]),
            constants.LABEL:
                # Shape = [time, batch]
                np.array([[False, False], [True, False], [True, False]])
        },
        info='')
    self._agent_config = agent_config.get_r2r_agent_config()

  def _get_agent(self,
                 agent_type,
                 init_with_text_state=True,
                 avg_all_img_states=False,
                 embed_action=True):
    self._agent_config.add_hparam('init_image_enc_with_text_state',
                                  init_with_text_state)
    self._agent_config.add_hparam('average_image_states_of_all_steps',
                                  avg_all_img_states)
    self._agent_config.embed_action = embed_action
    if agent_type == 'default':
      self._agent = discriminator_agent.DiscriminatorAgent(
          self._agent_config)
    elif agent_type == 'v2':
      self._agent = discriminator_agent.DiscriminatorAgentV2(
          self._agent_config)
    else:
      raise ValueError('agent_type must be `default` or `v2`.')

  def test_call(self):
    self._get_agent('default')
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
    self.assertEqual(agent_output.policy_logits.shape, [3, self.batch_size])

  @parameterized.named_parameters(
      ('Regular_mask', False, True, False),
      ('Regular_mask_init0', False, False, False),
      ('Regular_mask_avg_states', False, False, True),
      ('Zero_mask', True, True, False),
      ('Zero_mask_avg_states', True, True, True),)
  def test_call_v2(self, zero_mask, init_with_text_state, avg_all_img_states):
    self._get_agent('v2', init_with_text_state, avg_all_img_states)
    if zero_mask:
      disc_mask = np.tile(np.array([[False]] * 3), [1, self.batch_size])
      observation = self._test_environment.observation
      observation[constants.DISC_MASK] = disc_mask
      self._test_environment._replace(observation=observation)

    env_output = self._env.reset()
    observation = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                        env_output.observation)
    initial_agent_state = self._agent.get_initial_state(
        observation, batch_size=1)
    # Tests batch_size =1 for actor's scenario.
    # Agent always expects time,batch dimensions. First add and then remove.
    env_output = utils.add_time_batch_dim(env_output)
    agent_output, _ = self._agent(env_output, initial_agent_state)
    # Output shape = [time, batch, ...]
    self.assertEqual(agent_output.policy_logits['similarity'].shape, [1, 1, 1])
    self.assertEqual(agent_output.policy_logits['labels'].shape, [1, 1])
    self.assertEqual(agent_output.baseline.shape, [1, 1])

    # Remove time-batch dims for single env_output (i.e., batch=1, timestep=1).
    env_output, agent_output = utils.remove_time_batch_dim(env_output,
                                                           agent_output)
    self.assertEqual(agent_output.policy_logits['similarity'].shape, [1])
    self.assertEqual(agent_output.policy_logits['labels'].shape, [])
    self.assertEqual(agent_output.baseline.shape, [])

    # Tests with custom states and env.
    initial_input_state = [(tf.random.normal([self.batch_size, 512]),
                            tf.random.normal([self.batch_size, 512])),
                           (tf.random.normal([self.batch_size, 512]),
                            tf.random.normal([self.batch_size, 512]))]
    text_enc_output = tf.random.normal([self.batch_size, 5, 512])
    initial_agent_state = (initial_input_state,
                           (text_enc_output,
                            tf.nest.map_structure(tf.identity,
                                                  initial_input_state)))

    agent_output, _ = self._agent(self._test_environment, initial_agent_state)

    # Note that the agent_output has an extra time dim.
    self.assertEqual(agent_output.policy_logits['similarity'].shape,
                     [1, self.batch_size, self.batch_size])
    self.assertEqual(agent_output.policy_logits['similarity_mask'].shape,
                     [1, self.batch_size, self.batch_size])
    self.assertEqual(agent_output.policy_logits['labels'].shape,
                     [1, self.batch_size])
    self.assertEqual(agent_output.baseline.shape, [1, self.batch_size])

    if zero_mask:
      expected_similarity_mask = [[[True, True], [True, True]]]
      expected_labels = [[0.0, 0.0]]
    else:
      expected_similarity_mask = [[[True, True], [True, True]]]
      expected_labels = [[1.0, 0.0]]
    self.assertAllEqual(agent_output.policy_logits['similarity_mask'],
                        expected_similarity_mask)
    self.assertAllEqual(agent_output.policy_logits['labels'], expected_labels)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
