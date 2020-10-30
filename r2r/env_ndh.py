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

"""Environment class for R2R problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import os
import numpy as np

import tensorflow.compat.v2 as tf
from valan.framework import common

from valan.r2r import constants
from valan.r2r import env
from valan.r2r import env_ndh_config as default_env_config


class NDHEnv(env.R2REnv):
  """Environment class for NDH problem.

  An object of this class holds two main types of data:
  1. The graph of the underlying environment(s) - If running in distributed
     mode, every instance loads only selected scans and the connectivity graph
     of those scans.
  2. The paths from the selected environment(s). These are read from
     corresponding JSON files and filtered to contain only the selected
     environments.
  """

  def __init__(self, data_sources, runtime_config, env_config=None):
    """Initializes an instance of NDHEnv.

    Args:
      data_sources: A list of strings. The paths from '{}.json'.format( source)
        are cached for each of the source in data_sources.
      runtime_config: An instance of `common.RuntimeConfig`.
      env_config: Optional. If None, defaults to config specified in
        lookfar/r2r/env_config.py.
    """
    assert isinstance(runtime_config, common.RuntimeConfig)
    env_config = env_config if env_config else _get_default_env_config()
    self._env_config = env_config
    self._add_direction_encs = (
        env_config.add_direction_encs
        if hasattr(env_config, 'add_direction_encs') else True)

    all_paths = _get_all_paths_ndh(
        data_sources=data_sources,
        data_base_dir=env_config.data_base_dir,
        vocab_file=env_config.vocab_file,
        fixed_instruction_len=env_config.instruction_len,
        history=env_config.history,
        path_type=env_config.path_type)

    self._max_goal_room_panos = env_config.max_goal_room_panos
    self._compute_reward = env_config.reward_fn

    self._setup(data_sources, runtime_config, env_config, all_paths)

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

    reward = np.float32(
        self._compute_reward(
            path_history=self._path_history[:-1],
            next_pano=next_pano_id,
            golden_path=self._paths[self._current_idx]['path'],
            end_of_episode=done,
            scan_info=self._scan_info[current_scan_id],
            goal_room_panos=self._paths[self._current_idx]['end_panos']))

    return common.EnvOutput(
        reward=reward,
        done=done,
        observation=self._get_current_observation(next_pano_id, current_scan_id,
                                                  current_time_step + 1),
        info='')

  def _get_current_observation(self, pano_id, scan_id, time_step):
    obs = super(NDHEnv, self)._get_current_observation(pano_id, scan_id,
                                                       time_step)

    # add the pano ids of in the goal room
    goal_room_panos = [
        self._scan_info[scan_id].pano_name_to_id[pano_name]
        for pano_name in self._paths[self._current_idx]['end_panos']
    ]

    def _pad_with_invalid(room_panos):
      padding = max(0, self._max_goal_room_panos + 1 - len(room_panos))
      return room_panos + [constants.INVALID_NODE_ID] * padding

    padded_goal_room_panos = _pad_with_invalid(goal_room_panos)
    obs[constants.GOAL_ROOM_PANOS] = padded_goal_room_panos
    return obs


def _get_default_env_config():
  return default_env_config.get_ndh_env_config()


def _get_all_paths_ndh(data_sources, data_base_dir, vocab_file,
                       fixed_instruction_len, history, path_type):
  """Returns list of all paths from the given `data_sources`."""
  vocab_filepath = os.path.join(data_base_dir, vocab_file)
  vocab = env.load_vocab(vocab_filepath)
  filenames = [
      os.path.join(data_base_dir, '{}.json'.format(source))
      for source in data_sources
  ]
  processed_paths = []
  for filename in filenames:
    raw = [p for p in json.load(tf.io.gfile.GFile(filename))]
    for entry in raw:
      ins_tokens = []
      if history == 'none':
        pass
      elif history == 'target' or len(entry['dialog_history']) <= 0:
        ins_tokens = ['<TAR>'] + env.Tokenizer.split_sentence(entry['target'])
      elif history == 'oracle_ans':
        ora_a = entry['dialog_history'][-1][
            'message']  # i.e., the last oracle utterance.
        ins_tokens = ['<ORA>'] + env.Tokenizer.split_sentence(ora_a) + [
            '<TAR>'
        ] + env.Tokenizer.split_sentence(entry['target'])
      elif history == 'nav_q_oracle_ans':
        nav_q = entry['dialog_history'][-2]['message']
        ora_a = entry['dialog_history'][-1]['message']
        ins_tokens = ['<NAV>'] + env.Tokenizer.split_sentence(nav_q) + [
            '<ORA>'
        ] + env.Tokenizer.split_sentence(ora_a) + [
            '<TAR>'
        ] + env.Tokenizer.split_sentence(entry['target'])
      elif history == 'all':
        for turn in entry['dialog_history']:
          if turn['role'] == 'navigator':
            ins_tokens += ['<NAV>'] + env.Tokenizer.split_sentence(
                turn['message'])
          else:
            ins_tokens += ['<ORA>'] + env.Tokenizer.split_sentence(
                turn['message'])
        ins_tokens += ['<TAR>'] + env.Tokenizer.split_sentence(entry['target'])

      e = copy.copy(entry)
      e['instruction_token_ids'], e['instruction_len'] = env.get_token_ids(
          ins_tokens, fixed_instruction_len, vocab, is_tokenized=True)

      if path_type == 'trusted_path':
        # The trusted path is either the planner_path or
        # the player_path depending on whether the player_path
        # contains the planner_path goal
        # (e.g., stricter planner oracle success of player_path
        # indicates we can 'trust' it,
        # otherwise we fall back to the planner path for supervision).
        # Hypothesize that this will combine the strengths of
        # good human exploration with the known good, if
        # short, routes the planner uses.
        planner_goal = e['planner_path'][-1]
        if planner_goal in e['player_path'][1:]:
          # player walked through planner goal (did not start on it)
          e['path'] = e['player_path'][:]  # trust the player.
        else:
          e['path'] = e['planner_path'][:]  # trust the planner.
      elif not path_type:
        # No path is available (except starting pano) in test dataset.
        e['path'] = [e['start_pano']['pano']]
        # end_panos doesn't matter as we don't compute reward.
        e['end_panos'] = [e['start_pano']['pano']]
      else:
        e['path'] = e[path_type]
      # comparable to the R2R simulator
      e['path_id'] = e['inst_idx']
      e['heading'] = e['start_pano']['heading']
      e['problem_type'] = constants.PROBLEM_NDH

      processed_paths.append(e)
  return processed_paths
