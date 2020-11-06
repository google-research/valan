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

"""Utilities for testing."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf
from valan.framework import base_agent
from valan.framework import base_env
from valan.framework import common
from valan.framework import problem_type


def assert_matches_spec(specs, tensor_list):
  """Assert that a list of tensors matches the given TensorSpecs."""
  # Weirdly `tf.nest.pack_sequence_as` doesn't fail if tensor_list doesn't
  # conform to the specs type. So first pack the sequence, then explicitly
  # check the compatibility of each tensor with the corresponding spec.
  packed_tensors = tf.nest.pack_sequence_as(specs, tensor_list)
  packed_tensors = tf.nest.map_structure(tf.convert_to_tensor, packed_tensors)

  def is_compatible(sp, tensor):
    assert sp.is_compatible_with(
        tensor), 'TensorSpec {} is not compatible with tensor {}'.format(
            sp, tensor)

  tf.nest.map_structure(is_compatible, specs, packed_tensors)


class MockEnv(base_env.BaseEnv):
  """Mock environment for testing."""

  def __init__(self, state_space_size, unroll_length=1):
    self._state_space_size = state_space_size
    # Creates simple dynamics (T stands for transition):
    #   states = [0, 1, ... len(state_space_size - 1)] + [STOP]
    #   actions = [-1, 1]
    #   T(s, a) = s + a  iff (s + a) is a valid state
    #           = STOP   otherwise
    self._action_space = [-1, 1]
    self._current_state = None
    self._env_spec = common.EnvOutput(
        reward=tf.TensorSpec(shape=[unroll_length + 1], dtype=tf.float32),
        done=tf.TensorSpec(shape=[unroll_length + 1], dtype=tf.bool),
        observation={
            'f1':
                tf.TensorSpec(
                    shape=[unroll_length + 1, 4, 10], dtype=tf.float32),
            'f2':
                tf.TensorSpec(
                    shape=[unroll_length + 1, 7, 10, 2], dtype=tf.float32)
        },
        info=tf.TensorSpec(shape=[unroll_length + 1], dtype=tf.string))

  @property
  def env_spec(self):
    return self._env_spec

  def _get_current_observation(self):
    return {
        'f1': np.zeros((4, 10), dtype=np.float32),
        'f2': np.ones((7, 10, 2), dtype=np.float32) * self._current_state
    }

  def _reset(self):
    self._current_state = 0  # always start at state=0
    return common.EnvOutput(
        reward=0.,
        done=False,
        observation=self._get_current_observation(),
        info='')

  def _step(self, action):
    assert action in self._action_space
    new_state = self._current_state + action
    done = False
    if new_state < 0 or new_state == self._state_space_size:
      new_state = -100  # STOP
      done = True
    self._current_state = new_state
    return common.EnvOutput(
        reward=float(action),
        done=done,
        observation=self._get_current_observation(),
        info='')


class MockAgent(base_agent.BaseAgent):
  """Mock agent for testing."""

  def __init__(self, unroll_length=1):
    super(MockAgent, self).__init__(name='mock_agent')
    self._state_size = 5
    # This matches the action space of MockEnv
    self._action_space_size = 2
    self._logits_layer = tf.keras.layers.Dense(
        self._action_space_size,
        kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    self._agent_spec = common.AgentOutput(
        policy_logits=tf.TensorSpec(
            shape=[unroll_length + 1, 2], dtype=tf.float32),
        baseline=tf.TensorSpec(shape=[unroll_length + 1], dtype=tf.float32))

  @property
  def agent_spec(self):
    return self._agent_spec

  def _get_initial_state(self, env_output, batch_size):
    return tf.zeros([batch_size, self._state_size])

  def _torso(self, observation):
    # Verify shapes of observation.
    first_dim = observation['f1'].shape.as_list()[0]
    np.testing.assert_equal((first_dim, 4, 10), observation['f1'].shape)
    np.testing.assert_equal((first_dim, 7, 10, 2), observation['f2'].shape)
    return tf.ones(shape=[first_dim, 50])

  def _neck(self, torso_output, state):
    return tf.ones([tf.shape(torso_output)[0], 6]), state + 1

  def _head(self, neck_output):
    return common.AgentOutput(
        policy_logits=self._logits_layer(neck_output),
        baseline=tf.ones(shape=[tf.shape(neck_output)[0]]))


class MockProblem(problem_type.ProblemType):
  """Mock problem type."""

  def __init__(self, unroll_length=1):
    self._env = MockEnv(state_space_size=4, unroll_length=unroll_length)
    self._agent = MockAgent(unroll_length=unroll_length)
    self._actor_output_spec = common.ActorOutput(
        initial_agent_state=tf.TensorSpec(shape=[5], dtype=tf.float32),
        env_output=self._env.env_spec,
        agent_output=self._agent.agent_spec,
        actor_action=common.ActorAction(
            chosen_action_idx=tf.TensorSpec(
                shape=[unroll_length + 1], dtype=tf.int32),
            oracle_next_action_idx=tf.TensorSpec(
                shape=[unroll_length + 1], dtype=tf.int32),
            action_val=tf.TensorSpec(
                shape=[unroll_length + 1], dtype=tf.int32),
            log_prob=tf.TensorSpec(
                shape=[unroll_length + 1], dtype=tf.float32)),
        loss_type=tf.TensorSpec(shape=[], dtype=tf.int32),
        info=tf.TensorSpec(shape=[], dtype=tf.string),
    )

  def get_actor_output_spec(self):
    return self._actor_output_spec

  def get_environment(self):
    return self._env

  def get_agent(self):
    return self._agent

  def get_optimizer(self, learning_rate):
    return tf.keras.optimizers.SGD(learning_rate=learning_rate)

  def create_summary(self, step, info):
    pass

  def get_study_loss_types(self):
    return [common.CE_LOSS, common.AC_LOSS]

  def get_episode_loss_type(self, iterations):
    return common.AC_LOSS

  def get_actor_info(self, final_step_env_output, episode_reward_sum,
                     episode_num_steps):
    return ''

  def select_actor_action(self, env_output, agent_output):
    # Always selects action=1 by default.
    action_idx = 1
    action_val = 1
    oracle_next_action_idx = 1
    return common.ActorAction(
        chosen_action_idx=action_idx,
        oracle_next_action_idx=oracle_next_action_idx,
        action_val=action_val,
        log_prob=0.0)

  def eval(self, action_list, env_list):
    return {'result': 1000.0}
