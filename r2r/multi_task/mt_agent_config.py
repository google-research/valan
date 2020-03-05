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

from valan.framework import hparam

AGENT_CONFIG = {
    'pretrained_embed_path': None,
    'oov_bucket_size': 1,
    'vocab_size': 1633,  # size of common VLN+NDH tasks' vocab.
    'word_embed_dim': 300,
    # If True, use separate text encoders for NDH and VLN tasks.
    'use_separate_encoders': False,
    # Classifier related
    'classify_instructions': False,
    'classify_scans': True,
    'classifier_dropout': 0.5,
}


def get_agent_config():
  """Returns default config using values from dict `R2R_AGENT_CONFIG`."""
  config = hparam.HParams(**AGENT_CONFIG)
  return config
