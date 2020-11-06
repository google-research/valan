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

"""Default configurations used for R2R and NDH agents."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from valan.framework import hparam

R2R_AGENT_CONFIG = {
    'pretrained_embed_path': '',
    'oov_bucket_size': 1,
    'vocab_size': 1082,  # Set this according to the vocab file.
    'word_embed_dim': 300,
    'l2_scale': 0.0,
    'dropout': 0.0,
    'concat_context': False,
    'layernorm': False,
    'embed_action': True,
}

NDH_AGENT_CONFIG = {
    'pretrained_embed_path': None,
    'oov_bucket_size': 1,
    'vocab_size': 1268,
    'word_embed_dim': 300,
}


def get_r2r_agent_config():
  """Returns default config using values from dict `R2R_AGENT_CONFIG`."""
  config = hparam.HParams(**R2R_AGENT_CONFIG)
  return config


def get_ndh_agent_config():
  """Returns default config using values from dict `NDH_AGENT_CONFIG`."""
  config = hparam.HParams(**NDH_AGENT_CONFIG)
  return config
