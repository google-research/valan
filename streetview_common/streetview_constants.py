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

"""Constants shared between Touchdown and CrowdDriving datasets."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections

NUM_DISCRETE_ACTIONS = 4
MAX_GOLDEN_PATH_LENGTH = 128

RAW_IMAGE = 'raw_image'
IMAGE_FEATURES = 'image_features'
NAV_TEXT = 'nav_text'
NAV_TEXT_RAW = 'nav_text_raw'
NAV_TEXT_EMBEDDED = 'nav_text_embedded'
HEADING = 'heading'
LATITUDE = 'latitude'
LONGITUDE = 'longitude'
NAV_TEXT_LENGTH = 'nav_text_length'
TIMESTEP = 'timestep'
PREV_ACTION_IDX = 'previous_action_idx'
ORACLE_NEXT_ACTION = 'oracle_next_action'
MAP_CHAR_IDX_TO_TOK = 'char_idx_to_tok'
GOLDEN_PATH = 'golden_path'
PANO_ID = 'pano_id'
INVALID_PANO_ID = '-1'

BaselineAgentParams = collections.namedtuple(
    'BaselineAgentParams',
    ['VOCAB_SIZE',
     'INSTRUCTION_LSTM_DIM',
     'TEXT_EMBED_DIM',
     'TIMESTEP_EMBED_DIM',
     'ACTION_EMBED_DIM',
     'MAX_AGENT_ACTIONS',
     'L2_SCALE']
)


PanoramicAgentParams = collections.namedtuple(
    'PanoramicAgentParams',
    ['VOCAB_SIZE',
     'FEATURE_H',
     'FEATURE_W',
     'FEATURE_C',
     'LINGUNET_H',
     'LINGUNET_G',
     'INSTRUCTION_LSTM_DIM',
     'GROUNDING_EMBEDDING_DIM',
     'TIME_LSTM_DIM',
     'TEXT_EMBED_DIM',
     'TIMESTEP_EMBED_DIM',
     'ACTION_EMBED_DIM',
     'MAX_AGENT_ACTIONS',
     'L2_SCALE']
)
