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

"""Shared constants for Touchdown problem.

The remainder of constants is defined streetview_common subdir.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from valan.streetview_common import streetview_constants


TD_BASELINE_AGENT_PARAMS = streetview_constants.BaselineAgentParams(
    # Actual vocab size is 4280, we add 1 as vocab_id=0 can not be used since we
    # are masking for RNN performance.
    VOCAB_SIZE=4280 + 1,
    INSTRUCTION_LSTM_DIM=256,
    TEXT_EMBED_DIM=32,
    TIMESTEP_EMBED_DIM=32,
    ACTION_EMBED_DIM=16,
    MAX_AGENT_ACTIONS=55,
    L2_SCALE=0.0,
    )


TD_PANO_AGENT_PARAMS = streetview_constants.PanoramicAgentParams(
    VOCAB_SIZE=4280 + 1,
    INSTRUCTION_LSTM_DIM=256,
    GROUNDING_EMBEDDING_DIM=256,  # Unused in Touchdown
    FEATURE_H=3,
    FEATURE_W=18,
    # FEATURE_C=64,  # Starburst v4
    FEATURE_C=606,  # Bottleneck v7
    # FEATURE_H=1,
    # FEATURE_W=8,
    # FEATURE_C=2048,  # ResNet50
    LINGUNET_H=64,
    LINGUNET_G=16,
    TIME_LSTM_DIM=32,
    TEXT_EMBED_DIM=32,
    TIMESTEP_EMBED_DIM=32,
    ACTION_EMBED_DIM=16,
    MAX_AGENT_ACTIONS=55,
    L2_SCALE=0.0,
    )

TD_PANO_AGENT_PARAMS_BIG = streetview_constants.PanoramicAgentParams(
    VOCAB_SIZE=30522+1,
    INSTRUCTION_LSTM_DIM=256,
    GROUNDING_EMBEDDING_DIM=256,
    FEATURE_H=3,
    FEATURE_W=18,
    FEATURE_C=606,
    LINGUNET_H=128,
    LINGUNET_G=64,
    TIME_LSTM_DIM=32,
    TEXT_EMBED_DIM=32,
    TIMESTEP_EMBED_DIM=32,
    ACTION_EMBED_DIM=16,
    MAX_AGENT_ACTIONS=55,
    L2_SCALE=0.0,
    )


