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


"""Environment for multi-task learning -- NDH and VLN tasks on Matterport.

Note that this environment may only be used by training actors as evaluation
actors can still evaluate on each task independently.
"""

import numpy as np
from valan.framework import common

from valan.r2r import constants
from valan.r2r import env
from valan.r2r import env_ndh
from valan.r2r.multi_task import mt_env_config

INF = 1e9


class MTEnv(env.R2REnv):
  """Env class for VLN+NDH tasks."""

  def __init__(self, runtime_config, env_config=None):
    assert isinstance(runtime_config, common.RuntimeConfig)

    env_config = env_config or mt_env_config.get_default_env_config()

    vln_paths = env._get_all_paths(  
        # hard-code train data sources as we use this env for train actors only.
        data_sources=['R2R_train', 'R2R_data_augmentation'],
        data_base_dir=env_config.data_base_dir,
        vocab_dir=env_config.vocab_dir,
        vocab_file=env_config.vocab_file,
        fixed_instruction_len=env_config.instruction_len)
    ndh_paths = env_ndh._get_all_paths_ndh(  
        # hard-code train data sources as we use this env for train actors only.
        data_sources=['NDH_train'],
        data_base_dir=env_config.data_base_dir,
        vocab_file=env_config.vocab_file,
        fixed_instruction_len=env_config.instruction_len,
        history=env_config.history,
        path_type=env_config.path_type)
    all_paths = vln_paths + ndh_paths
    self._compute_vln_reward = env_config.vln_reward_fn
    self._compute_ndh_reward = env_config.ndh_reward_fn
    self._max_goal_room_panos = env_config.max_goal_room_panos
    self._setup(['R2R_train', 'R2R_data_augmentation', 'NDH_train'],
                runtime_config, env_config, all_paths)

  def _get_next_idx(self, current_idx):
    """Get the next data idx in the environment."""
    return (current_idx + 1) % len(self._paths)

  def _reset(self):
    """Reset the environment with new data.

    Returns:
      A instance of common.EnvOutput, which is the initial Observation.
    """
    self._current_idx = self._get_next_idx(self._current_idx)
    current_scan_id = self._paths[self._current_idx]['scan_id']
    current_pano_id = self._scan_info[current_scan_id].pano_name_to_id[
        self._paths[self._current_idx]['path'][0]]
    self._path_history = [current_pano_id]
    return common.EnvOutput(
        reward=np.float32(0.0),
        done=False,
        observation=self._get_current_observation(current_pano_id,
                                                  current_scan_id, 0),
        info='')

  def _step(self, action):
    """Updates the state using the provided action.

    Sets `done=True` if either this action corresponds to stop node or the
    budget for the current episode is exhausted.

    Args:
      action: An integer specifying the next pano id.

    Returns:
      A tuple `EnvOutput`.
    """
    # First check this is a valid action.
    assert action >= 0
    current_observations = self.get_current_env_output().observation
    current_pano_id = current_observations[constants.PANO_ID]
    current_scan_id = current_observations[constants.SCAN_ID]
    current_time_step = current_observations[constants.TIME_STEP]
    assert action in self._scan_info[current_scan_id].conn_ids[current_pano_id]
    next_pano_id = action
    self._path_history.append(next_pano_id)
    done = False
    if (next_pano_id == constants.STOP_NODE_ID or
        current_time_step == self._max_actions_per_episode):
      done = True

    problem_type = self._paths[self._current_idx]['problem_type']
    if problem_type == constants.PROBLEM_VLN:
      reward = np.float32(
          self._compute_vln_reward(
              path_history=self._path_history[:-1],
              next_pano=next_pano_id,
              golden_path=self._paths[self._current_idx]['path'],
              end_of_episode=done,
              scan_info=self._scan_info[current_scan_id]))
    elif problem_type == constants.PROBLEM_NDH:
      reward = np.float32(
          self._compute_ndh_reward(
              path_history=self._path_history[:-1],
              next_pano=next_pano_id,
              golden_path=self._paths[self._current_idx]['path'],
              end_of_episode=done,
              scan_info=self._scan_info[current_scan_id],
              goal_room_panos=self._paths[self._current_idx]['end_panos']))
    else:
      raise ValueError('Invalid problem_type: {}.'.format(problem_type))
    return common.EnvOutput(
        reward=reward,
        done=done,
        observation=self._get_current_observation(next_pano_id, current_scan_id,
                                                  current_time_step + 1),
        info='')

  def _get_current_observation(self, pano_id, scan_id, time_step):
    obs = super(MTEnv, self)._get_current_observation(pano_id, scan_id,
                                                      time_step)

    problem_type = self._paths[self._current_idx]['problem_type']
    if problem_type == constants.PROBLEM_VLN:
      goal_room_panos = [-50]  # random invalid id.
    elif problem_type == constants.PROBLEM_NDH:
      goal_room_panos = [
          self._scan_info[scan_id].pano_name_to_id[pano_name]
          for pano_name in self._paths[self._current_idx]['end_panos']
      ]
    else:
      raise ValueError('Invalid problem_type: {}.'.format(problem_type))

    def _pad_with_invalid(room_panos):
      padding = max(0, self._max_goal_room_panos + 1 - len(room_panos))
      return room_panos + [constants.INVALID_NODE_ID] * padding

    padded_goal_room_panos = _pad_with_invalid(goal_room_panos)
    obs[constants.GOAL_ROOM_PANOS] = padded_goal_room_panos
    obs[constants.PROBLEM_TYPE] = problem_type
    return obs
