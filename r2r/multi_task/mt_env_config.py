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


"""Env Config for R2R+NDH tasks."""

from valan.framework import hparam
from valan.r2r import constants
from valan.r2r import env_config
from valan.r2r import env_ndh_config

DEFAULT_ENV_CONFIG = {
    'data_base_dir': '',
    'scan_base_dir': '',
    'vocab_file': 'r2rndh_trainval_vocab.txt',
    'image_features_dir': '',
    'image_encoding_dim': -1,
    'images_per_pano': 36,
    'max_conns': 14,
    'instruction_len': 240,
    'vln_reward_fn_type': constants.REWARD_DISTANCE_TO_GOAL,
    'max_agent_actions': 20,
    'max_goal_room_panos':
        35,  # must be larger than the largest number in the dataset (34)
    # NDH-specific only
    'history':  # none | target | oracle_ans | nav_q_oracle_ans | all
        'all',
    'path_type': 'trusted_path',  # trusted_path | planner_path | player_path
    'ndh_reward_fn_type': constants.REWARD_DISTANCE_TO_ROOM,
    # The following are set in `get_default_env_config`.
    'vln_reward_fn': '',
    'ndh_reward_fn': '',
}


def get_default_env_config():
  """Returns default env config."""
  config = hparam.HParams(**DEFAULT_ENV_CONFIG)
  config.vln_reward_fn = env_config.RewardFunction.get_reward_fn(
      config.vln_reward_fn_type)
  config.ndh_reward_fn = env_ndh_config.RewardFunction.get_reward_fn(
      config.ndh_reward_fn_type)
  return config
