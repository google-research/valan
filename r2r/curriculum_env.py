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

"""Curriculum environment class for R2R problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from valan.r2r import curriculum_env_config as curr_env_config_lib
from valan.r2r import env


class CurriculumR2REnv(env.R2REnv):
  """Curriculum environment class for R2R problem."""

  def __init__(self,
               data_sources,
               runtime_config,
               curriculum_env_config=None):
    """Initialize an instance of CurriculumR2REnv.

    Initialize R2REnv (super class). Sort all paths from easy to hard with
    respect to the given sort_key.

    Args:
      data_sources: A list of strings. The paths from 'R2R_{}.json'.format(
        source) are cached for each of the source in data_sources.
      runtime_config: An instance of `common.RuntimeConfig`.
      curriculum_env_config: Optional. If None, defaults to config specified in
        lookfar/r2r/curriculum_env_config.py.
    """
    if curriculum_env_config is None:
      curriculum_env_config = curr_env_config_lib.get_default_curriculum_env_config(
          'constant-50-1')
    super(CurriculumR2REnv, self).__init__(data_sources, runtime_config,
                                           curriculum_env_config)

    # Sort the paths from easy to hard.
    self._sorted_paths = sorted(self._paths,
                                key=curriculum_env_config.sort_key,
                                reverse=curriculum_env_config.reverse)

    # Parse the method into curriculum parameters.
    method_name, arg1, arg2 = curriculum_env_config.method.split('-')
    method_name = method_name.lower()     # avoid typos of method_name
    assert method_name in ['constant', 'adaptive']
    if method_name == 'constant':
      initial_path_num = float(arg1)
      self._increment = float(arg2)
    elif method_name == 'adaptive':
      initial_path_num = float(arg1)
      expected_iters = float(arg2)
      self._increment = (
          max(0., self.num_paths - initial_path_num) / expected_iters)

    # Start with top `initial_path_num paths` in self._sorted_paths.
    # It is the expected number of paths since (A) It is a float number. (B) It
    # might exceed the number of all paths.
    self._expected_num_paths = initial_path_num
    self._paths = self._sorted_paths[:int(self._expected_num_paths)]

  def _get_next_idx(self, current_idx):
    """Get the next data idx in the environment."""
    return random.randint(0, self.num_paths - 1)

  def _reset(self):
    """Reset the environment.

    For each calling of _reset, add self._increment amount of paths to
    the environment. It would call super()._reset for the next datum.
    The index of the datum depends on self._get_next_idx.

    Returns:
      An instance of common.EnvOutput, which is the initial observation.
    """
    # Add harder paths into envs.
    self._expected_num_paths += self._increment
    expected_num_paths = int(self._expected_num_paths)
    if (self.num_paths < expected_num_paths and
        self.num_paths < len(self._sorted_paths)):
      self._paths.extend(self._sorted_paths[self.num_paths:expected_num_paths])

    return super(CurriculumR2REnv, self)._reset()
