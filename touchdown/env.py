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

"""Environment class for Touchdown problem.

https://arxiv.org/abs/1811.12354
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import functools
import json
import os

import numpy as np
import tensorflow.compat.v2 as tf

from valan.streetview_common import graph_loader
from valan.streetview_common import streetview_constants
from valan.streetview_common import streetview_env


class TouchdownEnv(streetview_env.StreetViewEnv):
  """Environment class for Touchdown problem.

  An object of this class holds two main data structures:
  1. The graph of the underlying Touchdown environment (inherited).
  2. The paths from JSON files depending on the source of data.
  """

  def _load_golden_paths(self, input_file):
    """Builds golden path entries from the touchdown json files."""
    entry_sequences = []
    with tf.io.gfile.GFile(input_file, 'r') as f:
      for line in f:
        entry = json.loads(line)
        entry_sequences.append([
            streetview_env.Entry(entry['route_panoids'], entry['start_heading'],
                                 entry['end_heading'], entry['navigation_text'],
                                 entry['route_id'])])
    return entry_sequences

  def _select_regions(self, env_config, worker_idx=0):
    """Touchdown only has a single region."""
    return ['nyc_touchdown']

  def _init_graphs(self, env_config, worker_regions):
    """Returns a list of graphs that this env can traverse."""
    assert len(worker_regions) == 1
    return {worker_regions[0]: graph_loader.GraphLoader(
        env_config.nodes_file,
        env_config.links_file,
        panoramic_actions=self._panoramic_actions,
        pano_heading_window=env_config.pano_heading_window).construct_graph()}

  def _init_entry_sequences(self, data_sources, env_config, worker_regions):
    """Loads all annotations from Touchdown JSON files."""
    entry_sequences = []
    for data_source in data_sources:
      entry_sequences += self._load_golden_paths(
          os.path.join(env_config.data_base_path,
                       '{}.json'.format(data_source)))
    assert len(worker_regions) == 1
    return {worker_regions[0]: entry_sequences}

  def _extract_features_from_table_value(self, im_table, panoid):
    """Grabs element for panoid from feature table and returns it as ndarray."""
    filename = im_table[panoid]
    image_feature = np.load(
        filename, dtype=np.float32).reshape(self._env_config.feature_height,
                                            self._env_config.feature_width,
                                            self._env_config.feature_channels)
    return image_feature

  def _init_feature_loaders(self, env_config, worker_regions):
    """Returns a dict of functions that take a panoid and return an ndarray."""
    filenames = tf.io.gfile.glob(env_config.image_features_path)
    assert len(worker_regions) == 1

    im_table = {}
    for filename in filenames:
      # Remove .npy to get panoid.
      panoid = os.path.basename(filename)[:-4]
      im_table[panoid] = filename

    return {worker_regions[0]: functools.partial(
        self._extract_features_from_table_value, im_table)}

  def _get_current_observation(self, prev_action):
    """Adds entities from map layers to returned observations."""
    # Get the common parts of the observation
    observation = super(TouchdownEnv, self)._get_current_observation(
        prev_action)

    raw_text = self._all_entry_sequences[self._current_region][
        self._current_sequence_idx][self._current_entry_idx].navigation_text

    # Compute text features:
    text_ids, length = self._get_text_feature(raw_text)
    observation[streetview_constants.NAV_TEXT] = text_ids
    observation[streetview_constants.NAV_TEXT_LENGTH] = length
    observation[streetview_constants.NAV_TEXT_RAW] = raw_text

    return observation
