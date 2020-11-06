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

"""Abstract Environment class for Python environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
from valan.framework import common


@six.add_metaclass(abc.ABCMeta)
class BaseEnv(object):
  """Abstract Environment class for Python environments."""

  def __init__(self):
    self._current_env_output = common.EnvOutput(
        reward=None, done=True, observation=None, info=None)

  def get_current_env_output(self):
    """Returns the current environment output."""
    return self._current_env_output

  def reset(self):
    """Resets and returns the internal state.

    Calling this methods starts a new episode.

    NOTE: Subclasses must NOT override this method directly. Subclasses
    implement `_reset()` which will be called by this method. The output of
    `_reset()` will be cached and made available through
    `get_current_env_output()`.

    Returns:
      An `EnvOutput` tuple. See common.py for details about the tuple.
    """
    self._current_env_output = self._reset()
    assert isinstance(self._current_env_output, common.EnvOutput)
    return self._current_env_output

  def step(self, action):
    """Updates and returns the internal state after taking the action.

    If the environment returned `done=True` in previous step, `action` is
    ignored and internal state is reset starting a new episode.

    NOTE: Subclasses must NOT override this method directly. Subclasses
    implement `_step()` which will be called by this method. The output of
    `_step()` will be cached and made available through
    `get_current_env_output()`.

    Args:
      action: A numpy array or a nested dict/list/tuple of numpy arrays.

    Returns:
      An `EnvOutput` tuple. See common.py for details about the tuple.
    """
    self._current_env_output = self._step(action)
    assert isinstance(self._current_env_output, common.EnvOutput)
    if self._current_env_output.done:

      self._current_env_output = self._current_env_output._replace(
          observation=self._reset().observation)
    return self._current_env_output

  def get_state(self):
    """Returns an object to be used in conjunction with set_state.

    Returns:
      An object containing sufficient context to restore the current
      environment state later in the same episode.
    """
    return self._current_env_output

  def set_state(self, state):
    """Restore a previous state from the same episode, to support planning.

    Note that set_state and get_state are not expected to work across episodes
    (e.g., after reset). Subclasses should override get_stage and set_stage if
    there is additional state that needs to be restored, e.g. path history.

    Args:
      state: An object returned by get_state.
    """
    self._current_env_output = state

  @abc.abstractmethod
  def _step(self, action):
    """Updates and returns the internal state after taking the action.

    See `step(self, action)` docstring for more details.

    Args:
      action: A numpy array or a nested dict/list/tuple of numpy arrays.

    Returns:
      An `EnvOutput` tuple. See common.py for details about the tuple.
    """

  @abc.abstractmethod
  def _reset(self):
    """Resets and returns the internal state.

    See `reset(self)` docstring for more details.

    Returns:
      An `EnvOutput` tuple. See common.py for details about the tuple.
    """
