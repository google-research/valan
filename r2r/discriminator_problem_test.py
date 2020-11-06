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

"""Tests for valan.r2r.discriminator_problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
from absl.testing import parameterized
import tensorflow.compat.v2 as tf
from valan.framework import common
from valan.framework import hparam
from valan.r2r import agent_config
from valan.r2r import constants
from valan.r2r import discriminator_problem
from valan.r2r import env
from valan.r2r import env_config

FLAGS = flags.FLAGS


class DiscriminatorProblemTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(DiscriminatorProblemTest, self).setUp()
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

    self._runtime_config = common.RuntimeConfig(task_id=0, num_tasks=100)
    self._env = env.R2REnv(
        data_sources=['R2R_small_split'],
        runtime_config=self._runtime_config,
        env_config=self._env_config)

    self._agent_config = agent_config.get_r2r_agent_config()
    self._agent_config.add_hparam('init_image_enc_with_text_state', True)
    self._agent_config.add_hparam('average_image_states_of_all_steps', False)
    self._agent_config.embed_action = True

  @parameterized.named_parameters(
      ('default', 'default', True),
      ('default_perturbed', 'default', False),
      ('v2', 'v2', True),
      ('v2_perturbed', 'v2', False),
  )
  def test_select_actor_action(self, agent_type, use_golden_path=True):
    problem = discriminator_problem.DiscriminatorProblem(
        self._runtime_config,
        mode='train',
        data_sources=['R2R_small_split'],
        agent_config=self._agent_config,
        env_config=self._env_config)
    problem.get_environment()

    if not use_golden_path:
      # Tests repeated start and end nodes as well as disjoint node.
      problem._env._paths[0]['label'] = 0
      perturbed_path = [
          '29b20fa80dcd4771974303c1ccd8953f',
          '29b20fa80dcd4771974303c1ccd8953f',
          '46cecea0b30e4786b673f5e951bf82d4',
          '0ee20663dfa34b438d48750ddcd7366c', '0ee20663dfa34b438d48750ddcd7366c'
      ]
      problem._env._paths[0]['path'] = perturbed_path

    env_output = problem._env.reset()

    scan_id = env_output.observation[constants.SCAN_ID]
    pano_name_to_id = problem._env._scan_info[scan_id].pano_name_to_id
    golden_path = [
        pano_name_to_id[name] for name in problem._env._paths[0]['path']
    ]
    obs_golden_path = [
        x for x in env_output.observation[constants.GOLDEN_PATH] if x != -1
    ]
    self.assertEqual(obs_golden_path, golden_path)

    action_vals = []
    path_history = []
    for i in range(len(golden_path)):
      action = problem.select_actor_action(env_output, None)
      action_vals.append(action.action_val)
      path_history.append(env_output.observation[constants.PANO_ID])
      if i == 0:
        self.assertTrue(env_output.observation[constants.IS_START])
      else:
        self.assertFalse(env_output.observation[constants.IS_START])
      # Update to the next step.
      env_output = problem._env.step(action.action_val)
    self.assertEqual(path_history, golden_path)
    self.assertEqual(action_vals, golden_path[1:] + [0])


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
