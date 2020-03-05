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

from valan.framework import hparam

from valan.r2r import constants
from valan.r2r import env_config as env_config_lib

NDH_ENV_CONFIG = {
    'history':  # none | target | oracle_ans | nav_q_oracle_ans | all
        'oracle_ans',
    'path_type':
        'trusted_path',  # trusted_path | planner_path | player_path
    'max_goal_room_panos':
        35,  # must be larger than the largest number in the dataset (34)
    'data_base_dir':
        '',
    'scan_base_dir':  # Base dir for scan data, e.g., 'scans/', 'connections/'.
        '',
    # keep the task-specific data separate from the simulator data
    # making it more scalable
    'vocab_file':
        'trainval_vocab.txt',
    'images_per_pano':
        36,
    'max_conns':
        14,
    'instruction_len':
        50,
    'max_agent_actions':
        20,  # to be studied (check the distribution of the NDH data)
    'reward_fn_type':
        constants.REWARD_DISTANCE_TO_ROOM,
    'fov':
        0.2,
    'image_features_dir':
        '',
    'image_encoding_dim':
        2048,
    # The following are set in `get_default_env_config`.
    'reward_fn':
        '',
}


def distance_to_room(path_history, next_pano, golden_path, end_of_episode,
                     scan_info, goal_room_panos):
  """Rewards an agent based on how close it gets to the goal room.

  If d(p, g) is the distance of pano `p` from goal room `g`, then

  r(p1 --> p2) = 4  if end_of_episode and agent stopped in the goal room
               = -4 if end_of_episode and agent did not stop in the room
               = clip(d(p1, g) - d(p2, g), max=1, min=-1) otherwise

  Args:
    path_history: A list of integers specifying pano ids (source) until the
      current step.
    next_pano: An integer specifying next pano id (destination).
    golden_path: A list containing string names of panos on the golden path. Not
      used in this function.
    end_of_episode: True if this is the last transition in the episode.
    scan_info: A `ScanInfo` tuple. See constants.py.
    goal_room_panos: A list containing string names of panos in the goal room.

  Returns:
    A scalar float immediate reward for the transition
    current_pano --> next_pano.
  """
  del golden_path
  current_pano = path_history[-1]
  if end_of_episode:
    # If episode ended due to STOP node, then last valid node is
    # `current_pano`.
    last_node_id = (
        next_pano if next_pano != constants.STOP_NODE_ID else current_pano)
    last_node_name = scan_info.pano_id_to_name[last_node_id]
    return 4. if last_node_name in goal_room_panos else -4.
  current_pano_name = scan_info.pano_id_to_name[current_pano]
  next_pano_name = scan_info.pano_id_to_name[next_pano]

  def get_distance_to_goal_room(pano_name):
    return min([
        scan_info.graph.get_distance(pano_name, goal_pano_name)
        for goal_pano_name in goal_room_panos
    ])

  delta_distance = get_distance_to_goal_room(
      current_pano_name) - get_distance_to_goal_room(next_pano_name)
  return min(1., max(-1., delta_distance))


def distance_to_goal(path_history, next_pano, golden_path, end_of_episode,
                     scan_info, goal_room_panos):
  del goal_room_panos
  return env_config_lib.distance_to_goal(path_history, next_pano, golden_path,
                                         end_of_episode, scan_info)


def dense_dtw(path_history, next_pano, golden_path, end_of_episode, scan_info,
              goal_room_panos):
  del goal_room_panos
  return env_config_lib.dense_dtw(path_history, next_pano, golden_path,
                                  end_of_episode, scan_info)


class RewardFunction(object):
  """Specifies the RL reward function."""

  ### Registration happens here.
  _REWARD_FN_REGISTRY = {
      constants.REWARD_DISTANCE_TO_GOAL: distance_to_goal,
      constants.REWARD_DISTANCE_TO_ROOM: distance_to_room,
      constants.REWARD_DENSE_DTW: dense_dtw,
  }

  @staticmethod
  def get_reward_fn(reward_fn_type):
    if reward_fn_type not in RewardFunction._REWARD_FN_REGISTRY:
      raise ValueError(
          'Unsupported reward function type: %s. Please use one of %s or '
          'add your reward function to the registry in this file' %
          (reward_fn_type, RewardFunction._REWARD_FN_REGISTRY.keys()))
    return RewardFunction._REWARD_FN_REGISTRY[reward_fn_type]


def get_ndh_env_config():
  """Returns default NDH config using values from dict `NDH_ENV_CONFIG`."""
  # Input settings.
  history = NDH_ENV_CONFIG['history']
  if history == 'none':
    NDH_ENV_CONFIG['instruction_len'] = 1  # [<EOS>] fixed length.
  elif history == 'target':
    NDH_ENV_CONFIG['instruction_len'] = 3  # [<TAR> target <EOS>] fixed length.
  elif history == 'oracle_ans':
    # 16.16+/-9.67 ora utt len, 35.5 at x2 stddevs. 71 is double that.
    NDH_ENV_CONFIG['instruction_len'] = 50
  elif history == 'nav_q_oracle_ans':
    # 11.24+/-6.43 [plus Ora avg], 24.1 at x2 std.
    # 71+48 ~~ 120 per QA doubles both.
    NDH_ENV_CONFIG['instruction_len'] = 120
  else:  # i.e., 'all'
    # 4.93+/-3.21 turns -> 2.465+/-1.605 Q/A.
    # 5.67 at x2 std. Call it 6 (real max 13).
    NDH_ENV_CONFIG['instruction_len'] = 240

  config = hparam.HParams(**NDH_ENV_CONFIG)
  config.reward_fn = RewardFunction.get_reward_fn(config.reward_fn_type)
  return config
