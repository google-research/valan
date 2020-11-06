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

import os

from valan.framework import hparam

# Note: change the base path to a filesystem you have access to.
BASE_PATH = '/usr/local/home/touchdown'
FEATURES_RESNET = 'features_resnet_50_h-45_v-45_height-1_fov-60.sstable-*-of-00500'
FEATURES_STARBURST_PANORAMIC = (
    'features_starburst_v4_h-20_v-20_height-3_fov-30.sstable-*-of-01000')
FEATURES_BOTTLENECK_PANORAMIC = (
    'features_bottleneck_7_h-20_v-20_height-3_fov-30.sstable-*-of-01000')

DEFAULT_ENV_CONFIG = {

    # Care needs to be taken with respect to the location of image features.
    # These features are loaded from disk on demand and can slow down the actors
    # if the data is located in a far-off cell.
    # Cells with copy of touchdown data: lu, el, qo, iz
    #
    # Commented out are the old 1x8x2048 features from Howard Chen
    # 'image_features_path':
    #    os.path.join(BASE_PATH, 'features3_sstable/features-*-of-00010'),
    'legacy_image_features_path':
        os.path.join(BASE_PATH, 'features-dev', FEATURES_RESNET),
    'pano_image_features_path':
        os.path.join(BASE_PATH, 'features-dev', FEATURES_BOTTLENECK_PANORAMIC),
    'image_features_path': 'defined in get_default_env_config',
    # Assuming smaller features.
    'feature_height': 1,
    'feature_width': 8,
    'feature_channels': 2048,

    # The following paths are not that critical as they are read exactly once
    # at the beginning and cached.
    'data_base_path':
        os.path.join(BASE_PATH, 'data'),
    'vocab_file':
        os.path.join(BASE_PATH, 'merged_vocab.txt'),
    'nodes_file':
        os.path.join(BASE_PATH, 'graph/nodes.txt'),
    'links_file':
        os.path.join(BASE_PATH, 'graph/links.txt'),
    # From touchdown paper: maximum timesteps an episode can run for before the
    # environment raises a done signal.
    'max_agent_train_actions': 55,
    'max_agent_test_actions': 50,

    'big_model': False,

    'panoramic_agent': False,
    'panoramic_action_space': False,
    'panoramic_action_bins': 18,
    'pano_heading_window': 40,

    # Max number of tokens across all the instructions in the dataset after
    # word-piece tokenization: 550
    'instruction_tok_len': 550,

    # The following values are set in `get_default_env_config`.
    'max_agent_actions': -1,
    'mode': 'train',

    'base_yaw_angle': 157.5
}


def get_default_env_config(mode):
  """Returns default config using values from dict `DEFAULT_ENV_CONFIG`."""
  config = hparam.HParams(**DEFAULT_ENV_CONFIG)
  config.mode = mode
  if mode == 'train':
    config.max_agent_actions = config.max_agent_train_actions
  else:
    config.max_agent_actions = config.max_agent_test_actions
  config.image_features_path = (config.pano_image_features_path
                                if config.panoramic_action_space else
                                config.legacy_image_features_path)
  return config
