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

"""Environment base class for StreetView-based navigation problems.

https://arxiv.org/abs/1811.12354

Supports either discrete actions {Left, Right, Forward, Stop} or panoramic
actions (-180 to 180 degrees or 'stop'). If panoramic actions are used, this
environment represents them as bins from 0 to K, where bins 0 to K-1 represent
angles -180 + i*360/K (0 <= i <= K-1), and K-th bin indicates the stop action.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections

from absl import logging
from language.bert import tokenization
import numpy as np
from valan.framework import base_env
from valan.framework import common
from valan.streetview_common import graph_loader
from valan.streetview_common import streetview_constants


GraphState = graph_loader.GraphState
Entry = graph_loader.Entry
StepInfo = collections.namedtuple('StepInfo', ('distance_to_goal'))


class StreetViewEnv(base_env.BaseEnv):
  """Abstract environment class for StreetView problems.

  An object of this class holds two main data structures:
  1. The graph of the underlying StreetView environment.
  2. The paths from JSON files depending on the source of data.
  """

  ACTION_STR_TO_IDX = {'forward': 0, 'left': 1, 'right': 2, 'stop': 3}
  ACTION_IDX_TO_STR = {v: k for k, v in ACTION_STR_TO_IDX.items()}

  _DEFAULT_STEP_REWARD = -0.2
  _INF = 1e9

  def __init__(self,
               data_sources,
               env_config,
               debug_writer=None,
               worker_idx=0):
    """Construct the StreetView environment.

    Args:
      data_sources: A list of strings containing one or more of 'train', 'dev'
        and 'test'.
      env_config: The configuration object containing settings for the current
        run.
      debug_writer: Class for writing execution traces to dir.
      worker_idx: Restrict regions that this worker explores to those
        where worker_idx % region_idx == 0. Default 0 means include all regions.
        region_idx is defined as the order in a sorted list of region names.
    """
    super(StreetViewEnv, self).__init__()

    # Logging execution traces if enabled
    self._run_writer = debug_writer

    self._env_config = env_config

    # Action space
    self._panoramic_actions = env_config.panoramic_action_space
    self._panoramic_action_bins = env_config.panoramic_action_bins

    # Attributes prefixed by _all* are dictionaries indexed by region name
    self._all_regions = self._select_regions(env_config, worker_idx)
    self._all_graphs = self._init_graphs(env_config, self._all_regions)

    # Load entry sequences for each region, and activate the current region
    self._all_entry_sequences = self._init_entry_sequences(
        data_sources, env_config, self._all_regions)
    logging.info('Loaded %d regions with data sources %s',
                 len(self._all_entry_sequences), data_sources)
    for region_name, entry_sequence in self._all_entry_sequences.items():
      logging.info('Region %s has %d entries', region_name, len(entry_sequence))

    self._all_feature_loaders = self._init_feature_loaders(
        env_config, self._all_regions)

    # Using BERT word piece tokenizer.
    self._tokenizer = tokenization.FullTokenizer(
        vocab_file=env_config.vocab_file, do_lower_case=True)
    # Constants from env_config.
    self._max_actions_per_episode = env_config.max_agent_actions
    self._instruction_tok_len = env_config.instruction_tok_len

    # Constant parameter
    self._base_yaw_angle = env_config.base_yaw_angle

    # Class members.
    # Variables that identify current instruction (assigned in reset)
    self._graph = None
    self._current_region = ''
    self._current_sequence_idx = -1
    self._current_entry_idx = 0

    # State variables
    self._frame_count = 0
    self._goal_pano_id = -1
    self._graph_state = GraphState(-1, 0., 0., 0.)
    self._distance_to_goal = 0.

    # Information about current instruction
    self._golden_actions = None
    self._golden_path = None

  def _select_regions(self, env_config, worker_idx=0):
    """Returns a list of regions that this worker should traverse."""
    raise NotImplementedError('_select_regions should have been overridden')

  def _init_graphs(self, env_config, worker_regions):
    """Returns a dict of (region name, graph) that this env can traverse."""
    raise NotImplementedError('_init_graphs should have been overridden')

  def _init_entry_sequences(self, data_sources, env_config, worker_regions):
    """Returns a dict of lists of Entry that represent all annotations."""
    raise NotImplementedError(
        '_init_entry_sequences should have been overridden')

  def _init_feature_loaders(self, env_config, worker_regions):
    """Returns a dict of functions that take a panoid and return an ndarray."""
    raise NotImplementedError(
        '_init_feature_loaders should have been overridden')

  def _pano_heading_to_action_bin(self, heading_degrees):
    """Converts action heading to a bin index."""
    # 11-degrees should be rounded up to the 20-degree bin
    bin_size = 360 / self._panoramic_action_bins
    action_bin = int((180 + heading_degrees + bin_size / 2) *
                     float(self._panoramic_action_bins) / 360) % (
                         self._panoramic_action_bins)
    return action_bin

  def _pano_action_bin_to_heading(self, action_bin):
    """Converts bin index (int) to an action heading."""
    assert action_bin < self._panoramic_action_bins
    heading_yaw = action_bin * (360.0 / self._panoramic_action_bins) - 180
    # Make sure that int(float(int(heading_yaw))) would not round integers down
    heading_yaw += 1e-9
    return heading_yaw

  def _convert_golden_actions(self, golden_actions):
    """Convert heading angles / strings to discrete action bins."""
    if self._panoramic_actions:
      # Convert each angle to bin index, except for stop action
      golden_actions = [h if h == 'stop' else
                        int(self._pano_heading_to_action_bin(h))
                        for h in golden_actions]
      # Convert stop action to bin index
      golden_actions = [int(self._panoramic_action_bins) if h == 'stop'
                        else int(h)
                        for h in golden_actions]
    else:
      golden_actions = [self.ACTION_STR_TO_IDX[a] for a in golden_actions]
    return golden_actions

  def _reset(self):
    self._frame_count = 0

    # Switch to a new, random region among this worker's regions
    region_idx = np.random.randint(0, len(self._all_regions))
    self._current_region = self._all_regions[region_idx]
    self._graph = self._all_graphs[self._current_region]

    # Switch to a new, random instruction sequence from this region
    self._current_sequence_idx = np.random.randint(
        0, len(self._all_entry_sequences[self._current_region]))
    current_entry_sequence = (
        self._all_entry_sequences[
            self._current_region][self._current_sequence_idx])

    # Switch to a new, random instruction from this instruction sequence

    self._current_entry_idx = np.random.randint(
        0, len(current_entry_sequence))
    current_entry = current_entry_sequence[self._current_entry_idx]

    # The graph uses different action representations depending on action space
    # Convert to integer indices - the unifying representation used here
    golden_actions, self._golden_path = self._graph.get_golden_actions(
        current_entry)
    self._golden_actions = self._convert_golden_actions(golden_actions)

    self._goal_pano_id = current_entry.route[-1]

    self._graph_state = GraphState(current_entry.route[0],
                                   current_entry.start_heading, 0, 0)
    self._distance_to_goal = self.shortest_path_length(
        self._graph_state.pano_id, self._goal_pano_id)

    if self._run_writer:
      self._run_writer.reset()
      self._run_writer.log_run_data({
          'region': self._current_region,
          'route_id': current_entry.route_id,
          'segment_idx': self._current_entry_idx})

    return common.EnvOutput(
        reward=np.float32(0.),
        done=False,
        # We randomly choose prev_action_idx at the beginning of every episode.

        observation=self._get_current_observation(
            prev_action=np.random.choice(
                self._panoramic_action_bins + 1 if self._panoramic_actions else
                streetview_constants.NUM_DISCRETE_ACTIONS)),
        info=self._get_step_info())

  def shortest_path_length(self, start_pano_id, end_pano_id):
    return self._graph.shortest_path_length(start_pano_id, end_pano_id)

  def _step(self, action):
    """Steps the environment.

    Args:
      action:
        If not using panoramic actions, this is an integer index 0 to 3
        If using panoramic actions, this is an index 0 to K, where
          values 0 to K-1 are angle bins from -180 to 180, and K is the stop
          action

    Returns:
      The next environment output.
    """
    self._frame_count += 1
    # Update distance to goal. Note that at every step, self._distance_to_goal
    # is the distance to goal from the pano we reached in the prev step.
    self._distance_to_goal = self.shortest_path_length(
        self._graph_state.pano_id, self._goal_pano_id)

    if not isinstance(action, np.ndarray):
      action = np.array(action, dtype=np.int64)

    # Convert action idx to representation used by the graph
    if self._panoramic_actions:
      # go_towards is 'stop' or heading angle in degrees
      go_towards = ('stop' if action >= self._panoramic_action_bins
                    else self._pano_action_bin_to_heading(action))
    else:
      # go_towards is left, right, forward or stop.
      go_towards = self.ACTION_IDX_TO_STR[int(action)]

    reward = 0.
    done = False
    if go_towards == 'stop':
      # If the action is to stop.

      reward = 1. if self._goal_pano_id == self._graph_state.pano_id else -1.
      done = True
    else:
      # Else, take the step.
      next_graph_state = self._graph.get_next_graph_state(self._graph_state,
                                                          go_towards)

      if len(self._graph.nodes[next_graph_state.pano_id].neighbors) < 2:
        # stay still when running into the boundary of the graph
        logging.info('At the border (number of neighbors < 2). Did not go %s.',
                     str(go_towards))
        done = True
      else:
        prev_state_potential = self.shortest_path_length(
            self._graph_state.pano_id, self._goal_pano_id)
        cur_state_potential = self.shortest_path_length(
            next_graph_state.pano_id, self._goal_pano_id)

        dist_reward = prev_state_potential - cur_state_potential
        reward = self._DEFAULT_STEP_REWARD + dist_reward

        self._graph_state = next_graph_state

    if self._frame_count > self._max_actions_per_episode:
      done = True

    observation = self._get_current_observation(prev_action=action)

    # Log transition:
    if self._run_writer:
      self._run_writer.log_action(action, go_towards)
      self._run_writer.log_state(self._graph_state)
      self._run_writer.log_observation(observation)
      if done:
        self._run_writer.write()

    return common.EnvOutput(
        reward=reward,
        done=done,
        observation=observation,
        info=self._get_step_info())

  def _get_step_info(self):
    return StepInfo(self._distance_to_goal)

  def _get_current_observation(self, prev_action):
    oracle_next_action = np.int64(-1)

    if self._frame_count < len(self._golden_actions):
      oracle_next_action = self._golden_actions[self._frame_count]

    # Make sure the types are consistent and as expected
    prev_action = int(prev_action)
    oracle_next_action = int(oracle_next_action)


    pad_length = max(
        0, streetview_constants.MAX_GOLDEN_PATH_LENGTH - len(self._golden_path))
    self._golden_path = self._golden_path + [
        streetview_constants.INVALID_PANO_ID
    ] * pad_length


    return {
        streetview_constants.PANO_ID: self._graph_state.pano_id,
        streetview_constants.IMAGE_FEATURES: (
            self._get_image_feature(self._graph_state)),
        streetview_constants.HEADING: float(self._graph_state.heading),
        streetview_constants.LATITUDE: float(self._graph_state.latitude),
        streetview_constants.LONGITUDE: float(self._graph_state.longitude),
        streetview_constants.TIMESTEP: self._frame_count,
        streetview_constants.PREV_ACTION_IDX: prev_action,
        streetview_constants.ORACLE_NEXT_ACTION: oracle_next_action,
        streetview_constants.GOLDEN_PATH: self._golden_path,
    }

  def _get_image_feature(self, graph_state):
    """Retrieve image features for current pano."""
    # Expensive sstable call.
    image_feature = self._all_feature_loaders[self._current_region](
        graph_state.pano_id)

    # rotate the pano feature so the middle is the agent's heading direction
    # `shift_angle` is essential for adjusting to the correct heading
    # please include the following in your own `get_image_feature` function
    shift_angle = self._base_yaw_angle + self._graph.nodes[
        graph_state.pano_id].pano_yaw_angle - graph_state.heading

    # Pano feature shape for Touchdown (1, 8, 2048)
    width = image_feature.shape[1]
    shift = int(width * shift_angle / 360)
    image_feature = np.roll(image_feature, shift, axis=1)

    return image_feature

  def _get_text_feature(self, text):
    word_list = self._tokenizer.tokenize(text)
    # 0 vocab id can not be used since we are masking for RNN performance.
    # Therefore, shift all vocab ids by 1.
    token_embeddings = np.asarray(pad_or_truncate(
        [x + 1 for x in self._tokenizer.convert_tokens_to_ids(word_list)],
        self._instruction_tok_len))
    tokenized_length = np.asarray(
        min(len(word_list), self._instruction_tok_len))
    return token_embeddings, tokenized_length


def pad_or_truncate(x, desired_length, pad_num=0):
  if len(x) >= desired_length:
    return x[:desired_length]
  else:
    return x + [pad_num] * (desired_length - len(x))
