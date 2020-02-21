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

"""Tests for valan.r2r.curriculum_env."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow.compat.v2 as tf

from valan.r2r import constants
from valan.r2r import curriculum_env
from valan.r2r import curriculum_env_config as curriculum_env_config_lib
from valan.r2r import env_test

FLAGS = flags.FLAGS


def cmp_path_len(path1, path2):
  return len(path1['path']) <= len(path2['path'])


def cmp_inst_len(path1, path2):
  return path1['instruction_len'] <= path2['instruction_len']


def cmp_path_len_inst_len(path1, path2):
  path_len1 = len(path1['path'])
  path_len2 = len(path2['path'])
  inst_len1 = path1['instruction_len']
  inst_len2 = path2['instruction_len']
  if path_len1 < path_len2:
    return True
  elif path_len1 == path_len2 and inst_len1 <= inst_len2:
    return True
  return False


def cmp_inst_len_path_len(path1, path2):
  inst_len1 = path1['instruction_len']
  inst_len2 = path2['instruction_len']
  path_len1 = len(path1['path'])
  path_len2 = len(path2['path'])
  if inst_len1 < inst_len2:
    return True
  elif inst_len1 == inst_len2 and path_len1 <= path_len2:
    return True
  return False


name2cmp = {
    'path_len': cmp_path_len,
    'inst_len': cmp_inst_len,
    'path_len,inst_len': cmp_path_len_inst_len,
    'inst_len,path_len': cmp_inst_len_path_len,
}


class CurriculumEnvTest(env_test.EnvTest):

  def _check_sorted(self, sort_cmp):
    for easy_path, hard_path in zip(self._env._paths, self._env._paths[1:]):
      self.assertTrue(sort_cmp(easy_path, hard_path))

  def testCurriculumSorting(self):
    # Check whether every sort_key option is effective.
    key_names = [
        'path_len',
        'inst_len',
        'path_len,inst_len',
        'inst_len,path_len',
    ]
    sort_keys = [
        curriculum_env_config_lib.KEY_PATHLEN,
        curriculum_env_config_lib.KEY_INSTLEN,
        curriculum_env_config_lib.KEY_PATHLEN_INSTLEN,
        curriculum_env_config_lib.KEY_INSTLEN_PATHLEN,
    ]
    for key_name, sort_key in zip(key_names, sort_keys):
      config = curriculum_env_config_lib.get_default_curriculum_env_config(
          'constant-100-1', self._env_config)
      config.sort_key = sort_key

      # Key function takes one input and converts it to the key.
      self._env = curriculum_env.CurriculumR2REnv(
          data_sources=['R2R_small_split'],
          runtime_config=self._runtime_config,
          curriculum_env_config=config)
      # Compare function takes two inputs and determine their relationship.
      sort_cmp = name2cmp[key_name]
      self._check_sorted(sort_cmp)

  def _check_paths_order(self):
    for path, sorted_path in zip(self._env._paths, self._env._sorted_paths):
      self.assertEqual(path[constants.PATH_ID], sorted_path[constants.PATH_ID])

  def testCurriculumBuilding(self):
    config = curriculum_env_config_lib.get_default_curriculum_env_config(
        'constant-1-1', self._env_config)
    self._env = curriculum_env.CurriculumR2REnv(
        data_sources=['R2R_small_split'],
        runtime_config=self._runtime_config,
        curriculum_env_config=config)

    # Since initially only 1 path is put in the environment, the length of
    # paths should be 1.
    self.assertLen(self._env._paths, 1)
    for i in range(2, 7):
      _ = self._env.reset()
      self._check_paths_order()
      self._check_sorted(name2cmp['path_len,inst_len'])
      self.assertLen(self._env._paths, i)
    for _ in range(100):
      _ = self._env.reset()
      self._check_paths_order()
      self._check_sorted(name2cmp['path_len,inst_len'])
      self.assertLen(self._env._paths, 6)

  def testCurriculumConstantIncrement(self):
    config = curriculum_env_config_lib.get_default_curriculum_env_config(
        'constant-1-1.5', self._env_config)
    self._env = curriculum_env.CurriculumR2REnv(
        data_sources=['R2R_small_split'],
        runtime_config=self._runtime_config,
        curriculum_env_config=config)
    self.assertLen(self._env._paths, 1)
    for i in range(1, 4):
      _ = self._env.reset()
      self._check_paths_order()
      self._check_sorted(name2cmp['path_len,inst_len'])
      self.assertLen(self._env._paths, int(1 + 1.5 * i))
    for _ in range(100):
      _ = self._env.reset()
      self._check_paths_order()
      self._check_sorted(name2cmp['path_len,inst_len'])
      self.assertLen(self._env._paths, 6)

  def testCurriculumAdaptiveIncrement(self):
    config = curriculum_env_config_lib.get_default_curriculum_env_config(
        'adaptive-1-4', self._env_config)
    self._env = curriculum_env.CurriculumR2REnv(
        data_sources=['R2R_small_split'],
        runtime_config=self._runtime_config,
        curriculum_env_config=config)
    self.assertLen(self._env._paths, 1)

    self.assertEqual(self._env._increment, (6. - 1.) / 4)
    for i in range(1, 4):
      _ = self._env.reset()
      self._check_paths_order()
      self._check_sorted(name2cmp['path_len,inst_len'])
      self.assertLen(self._env._paths, int(1 + 1.25 * i))
    for _ in range(100):
      _ = self._env.reset()
      self._check_paths_order()
      self._check_sorted(name2cmp['path_len,inst_len'])
      self.assertLen(self._env._paths, 6)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
