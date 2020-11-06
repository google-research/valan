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

"""Default configuration used for Touchdown environment."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from valan.framework import hparam

EXAMPLE_ENV_CONFIG = {
    """Example of env config expected by StreetViewEnv and its subclasses.

    The fields in this config should be filled according to the specific
    StreetView dataset (Touchdown or CrowdDriving). This file serves to list
    the fields expected in the config by StreetViewEnv.
    """

    # Care needs to be taken with respect to the location of image features.
    # These features are loaded from disk on demand and can slow down the actors
    # if the data is located in a far-off cell.
    # Cells with copy of touchdown data: lu, el, qo
    'image_features_dir': 'undefined',
    # The following paths are not that critical as they are read exactly once
    # at the beginning and cached.
    'data_base_path': 'undefined',
    'vocab_file': 'undefined',
    'nodes_file': 'undefined',
    'links_file': 'undefined',
    # Max number of tokens across all the instructions in the dataset after
    # word-piece tokenization: 550
    'instruction_len': 550,
    # Maximum timesteps an episode can run for before the
    # environment raises a done signal.
    'max_agent_actions': 55,
}


def get_default_env_config():
  """Returns default config using values from dict `EXAMPLE_ENV_CONFIG`."""
  config = hparam.HParams(**EXAMPLE_ENV_CONFIG)
  return config
