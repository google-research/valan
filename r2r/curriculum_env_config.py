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

"""Default configuration used for curriculum R2R environment."""

import random
from valan.framework import hparam

from valan.r2r import env_config as env_config_lib

# Keys in sorting the paths.
KEY_PATHLEN = lambda path: len(path['path'])
KEY_INSTLEN = lambda path: path['instruction_len']
KEY_PATHLEN_INSTLEN = lambda path: (KEY_PATHLEN(path), KEY_INSTLEN(path))
KEY_INSTLEN_PATHLEN = lambda path: (KEY_INSTLEN(path), KEY_PATHLEN(path))
KEY_RANDOM = lambda path: random.random()   # Used in testing.


# Default config.
DEFAULT_CURRICULUM_ENV_CONFIG = dict(
    env_config_lib.DEFAULT_ENV_CONFIG,
    sort_key=KEY_PATHLEN_INSTLEN,
    reverse=False,
    method=''
)


def get_default_curriculum_env_config(method, env_config=None):
  """Get default curriculum env config.

  Args:
    method: The method used in curriculum learning.
    env_config: Optional. The env config. If None, use the default env
      config file. Default, None.

  Returns:
    A curriculum env config.
  """
  if env_config is None:
    env_config = env_config_lib.get_default_env_config()
  config_updates = dict(
      env_config.values(),
      method=method
  )
  curriculum_env_config = DEFAULT_CURRICULUM_ENV_CONFIG.copy()
  curriculum_env_config.update(config_updates)
  config = hparam.HParams(**curriculum_env_config)
  return config
