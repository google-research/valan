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

"""Base class for testing Touchdown and CrowdDriving envs.

Both envs follow almost identical API, and this is to avoid duplicate tests
and ensure future compatibility.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
from valan.framework import utils
from valan.streetview_common import streetview_constants


class EnvTestBase(tf.test.TestCase):
  """Base class for streetview based valan environment tests."""

  def setUp(self):
    super(EnvTestBase, self).setUp()
    self._data_sources = ['train']
    # Tell PyType about these attributes
    self._env = None
    self._env_config = None
    self._pano_env = None
    self._pano_env_config = None
    # Actually initialize the attributes
    self._env, self._env_config = self._setup_env_and_config(
        self._data_sources, panoramic_actions=False)
    self._pano_env, self._pano_env_config = (
        self._setup_env_and_config(self._data_sources, panoramic_actions=True))
    # So that env reset resets to the first environment always
    np.random.randint = lambda low, high: low

  def _setup_env_and_config(self, data_sources, panoramic_actions=False):
    raise NotImplementedError('_setup_env needs to be overridden')

  def _verify_env_output(self, env_output, expected_reward, expected_done,
                         expected_distance_to_goal, expected_timestep,
                         expected_feature_shape):
    self.assertEqual(expected_reward, env_output.reward)
    self.assertEqual(expected_done, env_output.done)
    self.assertEqual(expected_distance_to_goal,
                     env_output.info.distance_to_goal)
    obs = env_output.observation
    # Pano features have shape (1, 8, 2048)
    self.assertEqual(expected_feature_shape,
                     obs[streetview_constants.IMAGE_FEATURES].shape)
    self.assertEqual(expected_timestep, obs[streetview_constants.TIMESTEP])
    nav_text_len = (obs[streetview_constants.NAV_TEXT] > 0).sum()
    self.assertEqual(nav_text_len, obs[streetview_constants.NAV_TEXT_LENGTH])

  def _execute_actions_get_observations(self, some_env, action_sequence):
    """Executes the action sequence on the env and returns observations."""
    out0 = some_env.reset()
    env_outputs = [out0]
    for action in action_sequence:
      out = some_env.step(action)
      if not out.done:
        obs_action = out.observation[streetview_constants.PREV_ACTION_IDX]
        assert obs_action == action, (
            'Executed action {} does not match observed action: {}'.format(
                action, obs_action))
      env_outputs.append(out)
    env_outputs_stacked = utils.stack_nested_tensors(env_outputs)
    return env_outputs_stacked

  def _assert_all_in_range(self, name, items, min_incl, max_incl):
    """For each element in items, checks that it's in the stated range."""
    for i, item in enumerate(items):
      assert min_incl <= item <= max_incl, (
          '{} item number {} value {} out of range [{}, {}]'.format(
              name, i, item, min_incl, max_incl))

  def test_stackable_observations(self):
    """Tests whether an action sequence can be executed without errors."""
    actions = [self._env.ACTION_STR_TO_IDX['forward'] for _ in range(2)]
    actions.append(self._env.ACTION_STR_TO_IDX['stop'])
    _ = self._execute_actions_get_observations(self._env, actions)

  def test_stackable_observations_panoramic(self):
    """Tests whether a panoramic action sequence can be executed w/o errors."""
    actions = [int(self._env_config.panoramic_action_bins/2) for _ in range(2)]
    actions.append(self._env_config.panoramic_action_bins)
    _ = self._execute_actions_get_observations(self._pano_env, actions)

  def test_golden_action_range_panoramic(self):
    """Tests whether oracle actions are in the correct range."""
    
    for _ in range(2):
      self._pano_env.reset()
      self._assert_all_in_range('Oracle pano action',
                                self._pano_env._golden_actions,
                                0,
                                self._pano_env_config.panoramic_action_bins)

      last_action = self._pano_env._golden_actions[-1]
      assert (last_action == self._pano_env_config.panoramic_action_bins), (
          'Last action expected stop ({}), but found {}'.format(
              self._pano_env_config.panoramic_action_bins, last_action))


if __name__ == '__main__':
  tf.test.main()
