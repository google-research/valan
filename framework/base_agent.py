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

"""Abstract agent class."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import abc
import six
import tensorflow.compat.v2 as tf
from valan.framework import common
from valan.framework import utils


@six.add_metaclass(abc.ABCMeta)
class BaseAgent(tf.keras.Model):
  """Abstract agent class.

  Subclasses are expected to implement the following methods (see method
  docstring below for information about the args):
  1. _get_initial_state: returns state of an agent at the beginning of an
    episode.
  2. _torso: any pre-processing logic that gets parallelized for all timesteps.
    This is called exactly once per episode before call to `_neck`.
  3. _neck: called once per each timestep in the episode.
  4. _head: any post-processing logic that gets parallelized for all timesteps.
    This is called exactly once per episode after call to `_neck`.
  """

  def __init__(self, name):
    super(BaseAgent, self).__init__(name=name)

  def get_initial_state(self, observation, batch_size):
    """Returns initial state for `self._neck`.

    The initial state returned is used as the state for the beginning of an
    episode and passed on to `self._neck`.

    Args:
      observation: A nested structure from `EnvOutput` tuple. The individual
        tensors are expected to have correct batch dimension.
      batch_size: An int scalar tensor.

    Returns:
      A tensor or nested structure containing initial state of the agent at the
      beginning of an episode.
    """
    return self._get_initial_state(observation, batch_size)

  def call(self, env_output, initial_neck_state):
    """Runs the entire episode given time-major tensors.

    Args:
      env_output: An `EnvOutput` tuple with following expectations:
        reward - Unused
        done - A boolean tensor of shape  [num_timesteps, batch_size].
        observation - A nested structure with individual tensors that have first
          two dimensions equal to [num_timesteps, batch_size]
        info - Unused
      initial_neck_state: A tensor or nested structure with individual tensors
        that have first dimension equal to batch_size and no time dimension.

    Returns:
      An `AgentOutput` tuple with individual tensors that have first two
        dimensions equal to [num_timesteps, batch_size]
    """
    neck_output_list, neck_state = self._unroll_neck_steps(
        env_output, initial_neck_state)

    # Stack all time steps together in the 0th dim for all tensors in output.
    head_input = tf.nest.map_structure(lambda *tensors: tf.stack(tensors),
                                       *neck_output_list)
    head_output = utils.batch_apply(self._head, head_input)
    assert isinstance(head_output, common.AgentOutput)
    return head_output, neck_state

  def _get_reset_state(self, observation, done, default_state):
    """Resets the state wherever marked in `done` tensor.

    Consider the following example with num_timesteps=2, batch_size=3,
    state_size=1:
      default_state (batch_size, state_size) = [[5.], [5.], [5.]]
      done (num_timesteps, batch_size) = [[True, True, False],
                                          [False, True, False]]
      observation (num_timesteps, batch_size, 1) = [[[1.], [2.], [3.]],
                                                    [[4.], [5.], [6.]]]
      self.get_initial_state implements `observation + 10`.
    then returned tensor will be of shape (num_timesteps, batch_size,
    state_size) and its value will be:
      [[[11.], [12.], [0.]],
       [[0.],  [15.], [0.]]]
    where state values are replaced by call to `self.get_initial_state` wherever
    done=True. Note that the state values where done=False are set to zeros and
    are expected not to be used by the caller.

    Args:
      observation: A nested structure with individual tensors that have first
        two dimensions equal to [num_timesteps, batch_size].
      done: A boolean tensor of shape  [num_timesteps, batch_size].
      default_state: A tensor or nested structure with individual tensors that
        have first dimension equal to batch_size and no time dimension.

    Returns:
      A structure similar to `default_state` except that all tensors in the
      returned structure have an additional leading dimension equal to
      num_timesteps.
    """
    reset_indices = tf.compat.v1.where(tf.equal(done, True))

    def _get_reset_state_indices():
      reset_indices_obs = tf.nest.map_structure(
          lambda t: tf.gather_nd(t, reset_indices), observation)
      # shape: [num_indices_to_reset, ...]
      reset_indices_state = self.get_initial_state(
          reset_indices_obs, batch_size=tf.shape(reset_indices)[0])
      # Scatter tensors in `reset_indices_state` to shape: [num_timesteps,
      # batch_size, ...]
      return tf.nest.map_structure(
          lambda reset_tensor: tf.scatter_nd(  
              indices=reset_indices,
              updates=reset_tensor,
              shape=done.shape.as_list() + reset_tensor.shape.as_list()[1:]),
          reset_indices_state)

    # A minor optimization wherein if all elements in `done` are False, we
    # simply return a structure with zeros tensors of correct shape.
    return tf.cond(
        tf.greater(tf.size(reset_indices), 0),
        _get_reset_state_indices,
        lambda: tf.nest.map_structure(  
            lambda t: tf.zeros(         
                shape=done.shape.as_list() + t.shape.as_list()[1:],
                dtype=t.dtype),
            default_state))

  def _unroll_neck_steps(self, env_output, initial_state):
    """Unrolls all timesteps and returns a list of outputs and a final state."""
    unused_reward, done, observation, unused_info = env_output
    # Add current time_step and batch_size.
    self._current_num_timesteps = tf.shape(done)[0]
    self._current_batch_size = tf.shape(done)[1]

    torso_output = utils.batch_apply(self._torso, observation)

    # shape: [num_timesteps, batch_size, ...], where the trailing dimensions are
    # same as trailing dimensions of `neck_state`.
    neck_state = initial_state
    reset_state = self._get_reset_state(observation, done, neck_state)
    neck_output_list = []
    for timestep, d in enumerate(tf.unstack(done)):
      neck_input = utils.get_row_nested_tensor(torso_output, timestep)
      # If the episode ended, the neck state should be reset before the next
      # step.
      curr_timestep_reset_state = utils.get_row_nested_tensor(
          reset_state, timestep)
      neck_state = tf.nest.map_structure(
          lambda reset_state, state: tf.compat.v1.where(d, reset_state, state),  
          curr_timestep_reset_state, neck_state)
      neck_output, neck_state = self._neck(neck_input, neck_state)
      neck_output_list.append(neck_output)
    return neck_output_list, neck_state

  @abc.abstractmethod
  def _get_initial_state(self, observation, batch_size):
    """Returns initial state for `self._neck`.

    The initial state returned is used as the state for the beginning of an
    episode and passed on to `self._neck`.

    Args:
      observation: A nested structure from `EnvOutput` tuple. The individual
        tensors are expected to have correct batch dimension.
      batch_size: An int scalar tensor.

    Returns:
      A tensor or nested structure containing initial state of the agent at the
      beginning of an episode.
    """

  @abc.abstractmethod
  def _torso(self, observation):
    """Contains any pre-processing logic before call to `self._neck`.

    This method must only contain the logic that can be applied to all timesteps
    simultaneously in a batched call. This is called exactly once per episode.

    Args:
      observation: A nested structure from `EnvOutput` tuple. The tensors in
        nested structure have first dimension equal to `num_timesteps *
        batch_size`.

    Returns:
      A tensor or nested structure of tensors with first dimension equal to
      `num_timesteps * batch_size`. The returned tensors are reshaped to
      [num_timesteps, batch_size, ...], unstacked along time dimension and
      passed as arg to `self._neck`.
    """

  @abc.abstractmethod
  def _neck(self, torso_output, state):
    """Called once per timestep for every episode.

    Args:
      torso_output: A tensor or nested structure of tensors with shape
        [batch_size, ...] which is the output of `self._torso` unstacked along
        time dimension.
      state: A tensor or nested structure of tensors with shape [batch_size,
        ...] containing either recurrent state from last timestep or initial
        state if the episode was reset.

    Returns:
      A 2-tuple:
        neck_output: A tensor or nested structure of tensors with shape
          [batch_size, ...]. The returned tensors are stacked to create tensors
          of shape [num_timesteps, batch_size, ...] which are then passed as arg
          to `self._head`.
        neck_state: A recurrent state.
    """

  @abc.abstractmethod
  def _head(self, neck_output):
    """Contains any post-processing after call to `self._neck`.

    This method must only contain the logic that can be applied to all timesteps
    simultaneously in a batched call. This is called exactly once per episode.

    Args:
      neck_output: A tensor or nested structure of tensors with shape
        [num_timesteps * batch_size, ...] which is the stacked output of calls
        to `self._neck` for entire episode.

    Returns:
      An `AgentOutput` tuple:
        policy_logits: A tensor of shape [num_timesteps * batch_size,
          num_actions] containing logits for each of the actions.
        baseline: A tensor of shape [num_timesteps * batch_size] containing
          value of state, V(s), at each timestep.
    """
