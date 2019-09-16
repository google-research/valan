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

"""Tests for valan.framework.base_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf
from valan.framework import base_agent
from valan.framework import common

_OBS_KEY_0 = 'encoding0'
_OBS_KEY_1 = 'encoding1'


class MockAgent(base_agent.BaseAgent):
  """Mock agent for testing."""

  def __init__(self, total_timesteps, batch_size, init_state, done):
    super(MockAgent, self).__init__(name='mock_agent')
    self._total_timesteps = total_timesteps
    self._batch_size = batch_size
    assert np.shape(init_state)[0] == self._batch_size
    self._current_state = init_state
    self._init_state_size = np.shape(self._current_state)[1]
    assert np.shape(done) == (total_timesteps, batch_size)
    self._done = done
    self._timestep = None

  def reset_timestep(self):
    self._timestep = 0

  def _get_initial_state(self, observation, batch_size):
    return tf.zeros([batch_size, self._init_state_size])

  def _torso(self, observation):
    # Verify observation.
    np.testing.assert_equal((self._total_timesteps * self._batch_size, 50),
                            observation[_OBS_KEY_1].shape)
    np.testing.assert_array_almost_equal(
        np.ones((self._total_timesteps * self._batch_size, 50)),
        observation[_OBS_KEY_1])
    np.testing.assert_equal((self._total_timesteps * self._batch_size, 50),
                            observation[_OBS_KEY_0].shape)
    np.testing.assert_array_almost_equal(
        np.zeros((self._total_timesteps * self._batch_size, 50)),
        observation[_OBS_KEY_0])
    return tf.concat([observation[_OBS_KEY_1], observation[_OBS_KEY_0]], axis=1)

  def _neck(self, torso_output, state):
    # Verify state. It could have been reset if done was true.
    expected_state = np.copy(self._current_state.numpy())
    done = self._done[self._timestep]
    for i, d in enumerate(done):
      if d:
        expected_state[i] = np.zeros(self._init_state_size)
    np.testing.assert_array_almost_equal(expected_state, state.numpy())
    # Verify torso_output
    expected_torso_output = np.concatenate([
        np.ones(shape=(self._batch_size, 50)),
        np.zeros(shape=(self._batch_size, 50))
    ],
                                           axis=1)
    np.testing.assert_array_almost_equal(expected_torso_output,
                                         torso_output.numpy())
    self._timestep += 1
    self._current_state = state + 1
    return (tf.ones([self._batch_size, 6]) * self._timestep,
            self._current_state)

  def _head(self, neck_output):
    # Verify neck_output
    np.testing.assert_equal((self._total_timesteps * self._batch_size, 6),
                            neck_output.shape)
    arrays = []
    for i in range(self._total_timesteps):
      arrays.append(np.ones((self._batch_size, 6)) * (i + 1))
    expected_neck_output = np.concatenate(arrays, axis=0)
    np.testing.assert_array_almost_equal(expected_neck_output,
                                         neck_output.numpy())
    return common.AgentOutput(
        policy_logits=tf.zeros(
            shape=[self._total_timesteps * self._batch_size, 4]),
        baseline=tf.ones(shape=[self._total_timesteps * self._batch_size]))


class BaseAgentTest(tf.test.TestCase):

  def testWithMockAgent(self):
    total_timesteps = 3
    batch_size = 4
    done = np.array([[True, True, False, False], [True, False, True, False],
                     [False, False, False, False]])
    init_state = tf.constant([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                              [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                              [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                              [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]])

    env_output = common.EnvOutput(
        reward=None,
        done=done,
        observation={
            _OBS_KEY_1: tf.ones([total_timesteps, batch_size, 50]),
            _OBS_KEY_0: tf.zeros([total_timesteps, batch_size, 50]),
        },
        info=None)
    agent = MockAgent(3, 4, init_state, done)
    agent.reset_timestep()
    agent_output, final_state = agent(env_output, init_state)
    np.testing.assert_array_almost_equal(
        np.zeros((total_timesteps, batch_size, 4)), agent_output.policy_logits)
    np.testing.assert_array_almost_equal(
        np.ones((total_timesteps, batch_size)), agent_output.baseline)
    expected_final_state = np.array([[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                                     [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                                     [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                                     [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]])
    np.testing.assert_array_almost_equal(expected_final_state,
                                         final_state.numpy())

  def testWithMockAgent_DoneAllFalse(self):
    total_timesteps = 3
    batch_size = 4
    done = np.array([[False, False, False, False], [False, False, False, False],
                     [False, False, False, False]])
    init_state = tf.constant([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                              [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                              [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                              [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]])

    env_output = common.EnvOutput(
        reward=None,
        done=done,
        observation={
            _OBS_KEY_1: tf.ones([total_timesteps, batch_size, 50]),
            _OBS_KEY_0: tf.zeros([total_timesteps, batch_size, 50]),
        },
        info=None)
    agent = MockAgent(3, 4, init_state, done)
    agent.reset_timestep()
    agent_output, final_state = agent(env_output, init_state)
    np.testing.assert_array_almost_equal(
        np.zeros((total_timesteps, batch_size, 4)), agent_output.policy_logits)
    np.testing.assert_array_almost_equal(
        np.ones((total_timesteps, batch_size)), agent_output.baseline)
    expected_final_state = np.array([[4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
                                     [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                                     [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
                                     [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]])
    np.testing.assert_array_almost_equal(expected_final_state,
                                         final_state.numpy())


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
