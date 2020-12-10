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

"""Default configuration used for R2R environment."""
from absl import flags

import numpy as np
from valan.framework import eval_metric
from valan.framework import hparam
from valan.r2r import constants

FLAGS = flags.FLAGS

# Default fixed params for env.
DEFAULT_ENV_CONFIG = {
    # Base dir for scan data, eg., `scans`, `connections`.
    'scan_base_dir': '',  # Can be updated by FLAGS.scan_base_dir.
    # Base dir for input JSON and vocab files.
    'data_base_dir': '',  # Can be updated by FLAGS.data_base_dir.
    # Base dir for vocab.
    'vocab_dir': '',  # Can be updated by FLAGS.vocab_dir.
    'vocab_file': 'vocab.txt',
    # Number of image pathes for each panorama, typically 36.
    'images_per_pano': 36,
    'max_conns': 14,
    # Max number of instruction tokens.
    'instruction_len': 50,
    'max_agent_actions': 12,
    'reward_fn_type': constants.REWARD_DISTANCE_TO_GOAL,
    # Field of View used to generate image features.
    'fov': 0.17,
    # Dir containing pre-generated image features.
    'image_features_dir': '',  # Can be updated by FLAGS.image_features_dir.
    # Image feature dimension size. 1792 for EfficientNet B4.
    'image_encoding_dim': 1792,
    # Direction encoding dimension size.
    'direction_encoding_dim': 256,  # Must be a multiple of 8.
    # The following are set in `get_default_env_config`.
    'reward_fn': '',
}


### Different functions go here. All methods accept the following 4 args:
#  path_history: A list of integers specifying pano ids (source) until the
#    current step.
#  next_pano: An integer specifying next pano id (destination).
#  golden_path: A list containing string names of panos on the golden path.
#  end_of_episode: True if this is the last transition in the episode.
#  scan_info: A `ScanInfo` tuple. See constants.py.
def distance_to_goal(path_history, next_pano, golden_path, end_of_episode,
                     scan_info):
  """Rewards an agent based on how close it gets to the goal node.

  If d(p, g) is the distance of pano `p` from goal node `g`, then

  r(p1 --> p2) = 4  if end_of_episode and agent stopped correctly
               = -4 if end_of_episode and agent did not stop correctly
               = clip(d(p1, g) - d(p2, g), max=1, min=-1) otherwise

  Args:
    path_history: See above.
    next_pano: See above.
    golden_path: See above.
    end_of_episode: See above.
    scan_info: See above.

  Returns:
    A scalar float immediate reward for the transition
    current_pano --> next_pano.
  """
  current_pano = path_history[-1]
  goal_pano_name = golden_path[-1]
  if end_of_episode:
    # If episode ended due to STOP node, then last valid node is
    # `current_pano`.
    last_node_id = (
        next_pano if next_pano != constants.STOP_NODE_ID else current_pano)
    last_node_name = scan_info.pano_id_to_name[last_node_id]

    return 4. if last_node_name == goal_pano_name else -4.
  current_pano_name = scan_info.pano_id_to_name[current_pano]
  next_pano_name = scan_info.pano_id_to_name[next_pano]
  delta_distance = scan_info.graph.get_distance(
      current_pano_name, goal_pano_name) - scan_info.graph.get_distance(
          next_pano_name, goal_pano_name)
  return min(1., max(-1., delta_distance))


def dense_dtw(path_history, next_pano, golden_path, end_of_episode, scan_info):
  """Rewards an agent based on the difference in DTW after going to nex_pano.

  Args:
    path_history: See above.
    next_pano: See above.
    golden_path: See above.
    end_of_episode: See above.
    scan_info: See above.

  Returns:
    A scalar float immediate reward for the transition
    current_pano --> next_pano.
  """
  del end_of_episode
  if next_pano in [constants.STOP_NODE_ID, constants.INVALID_NODE_ID]:
    return 0.0
  observed_pano_ids = path_history + [next_pano]
  observed_pano_names = [
      scan_info.pano_id_to_name[pano] for pano in observed_pano_ids
  ]

  dtw_matrix = eval_metric.get_dtw_matrix(observed_pano_names, golden_path,
                                          scan_info.graph.get_distance)

  num_obs_panos = len(observed_pano_names)
  num_golden_panos = len(golden_path)
  previous_dtw = dtw_matrix[num_obs_panos - 1][num_golden_panos]
  current_dtw = dtw_matrix[num_obs_panos][num_golden_panos]

  return previous_dtw - current_dtw


def random_reward(path_history, next_pano, golden_path, end_of_episode,
                  scan_info):
  """Rewards by sampling a random value in (-1, 1) from a uniform distribution.

  Args:
    path_history: See above.
    next_pano: See above.
    golden_path: See above.
    end_of_episode: See above.
    scan_info: See above.

  Returns:
    A scalar float immediate reward sampled from a uniform dist for the
    transition current_pano --> next_pano.
  """
  del path_history, next_pano, golden_path, end_of_episode, scan_info
  return np.random.uniform(-1, 1)


def goal_plus_random_reward(path_history, next_pano, golden_path,
                            end_of_episode, scan_info):
  """Rewards an agent based on the difference in DTW after going to nex_pano.

  Args:
    path_history: See above.
    next_pano: See above.
    golden_path: See above.
    end_of_episode: See above.
    scan_info: See above.

  Returns:
    A scalar float immediate reward for the transition
    current_pano --> next_pano.
  """
  goal_rwd = distance_to_goal(path_history, next_pano, golden_path,
                              end_of_episode, scan_info)
  random_rwd = np.random.uniform(-1, 1)
  return goal_rwd + random_rwd


class RewardFunction(object):
  """Specifies the RL reward function."""

  ### Registration happens here.
  _REWARD_FN_REGISTRY = {
      constants.REWARD_DISTANCE_TO_GOAL: distance_to_goal,
      constants.REWARD_DENSE_DTW: dense_dtw,
      constants.REWARD_RANDOM: random_reward,
      constants.REWARD_GOAL_RANDOM: goal_plus_random_reward,
  }

  @staticmethod
  def get_reward_fn(reward_fn_type):
    if reward_fn_type not in RewardFunction._REWARD_FN_REGISTRY:
      raise ValueError(
          'Unsupported reward function type: %s. Please use one of %s or '
          'add your reward function to the registry in this file' %
          (reward_fn_type, RewardFunction._REWARD_FN_REGISTRY.keys()))
    return RewardFunction._REWARD_FN_REGISTRY[reward_fn_type]


def get_default_env_config():
  """Returns default config using values from dict `DEFAULT_ENV_CONFIG`."""
  config = hparam.HParams(**DEFAULT_ENV_CONFIG)
  config.reward_fn = RewardFunction.get_reward_fn(config.reward_fn_type)

  # Update directories if set in FLAGS.
  if FLAGS.scan_base_dir:
    config.scan_base_dir = FLAGS.scan_base_dir
  if FLAGS.data_base_dir:
    config.data_base_dir = FLAGS.data_base_dir
  if FLAGS.vocab_dir:
    config.vocab_dir = FLAGS.vocab_dir
  if FLAGS.vocab_file:
    config.vocab_file = FLAGS.vocab_file
  if FLAGS.image_features_dir:
    config.image_features_dir = FLAGS.image_features_dir
  return config
