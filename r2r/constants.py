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

"""Shared constants."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Keys in observation dict returned by environment.
TIME_STEP = 'time_step'
PATH_ID = 'path_id'
PANO_ID = 'pano_id'
HEADING = 'heading'
PITCH = 'pitch'
SCAN_ID = 'scan_id'
INS_TOKEN_IDS = 'instruction_token_ids'
INS_LEN = 'instruction_len'
CONN_IDS = 'connection_ids'
VALID_CONN_MASK = 'valid_connections_mask'
PANO_ENC = 'pano_encoding'
CONN_ENC = 'connection_encoding'
PREV_ACTION_ENC = 'previous_action_encoding'
NEXT_GOLDEN_ACTION_ENC = 'next_golden_action_encoding'
ORACLE_NEXT_ACTION = 'oracle_next_action'
GOAL_PANO_ID = 'goal_pano_id'
GOLDEN_PATH = 'golden_path'
OBSERVED_PATH = 'observed_path'
GOAL_ROOM_PANOS = 'goal_room_pano_ids'  # only for NDH task
LABEL = 'label'
IS_START = 'is_start'
DISC_MASK = 'disc_mask'
PROBLEM_TYPE = 'problem_type'

# reward types
REWARD_DISTANCE_TO_ROOM = 'distance_to_room'
REWARD_DISTANCE_TO_GOAL = 'distance_to_goal'
REWARD_DENSE_DTW = 'dense_dtw'
REWARD_RANDOM = 'random_reward'
REWARD_GOAL_RANDOM = 'goal_plus_random'

# Environment constants.
# We use two special ids:
#  0: is for STOP_NODE. Every node has a connection to this node since the
#     agent can decide to stop at any node.
# -1: is for invalid node. We use this node for padding whenever dense tensors
#     are created. There should never be a transition involving this node.
STOP_NODE_ID = 0
STOP_NODE_NAME = 'STOP_NODE'
INVALID_NODE_ID = -1
INVALID_NODE_NAME = 'INVALID_NODE'

# Problem Types
PROBLEM_VLN = 0
PROBLEM_NDH = 1

# Other constants
PAD_TOKEN = '<PAD>'
OOV_TOKEN = '<UNK>'

# Tuples
ScanInfo = collections.namedtuple('ScanInfo', [
    'pano_name_to_id', 'pano_id_to_name', 'pano_enc', 'pano_heading',
    'pano_pitch', 'conn_ids', 'conn_enc', 'conn_heading', 'conn_pitch',
    'graph'
])
