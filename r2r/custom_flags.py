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

"""Additional hyperparam flags specific to this problem."""

from absl import flags

FLAGS = flags.FLAGS

# Settings for env_config.py
flags.DEFINE_string(
    'scan_base_dir', '',
    'Base dir for Matterport scan data, e.g., scans, connections.')
flags.DEFINE_string('data_base_dir', '', 'Base dir for input JSON files.')
flags.DEFINE_string('vocab_dir', '', 'Base dir for input vocab files.')
flags.DEFINE_string('image_features_dir', '',
                    'Dir containing pre-generated image features.')
flags.DEFINE_string('vocab_file', '', 'File name for vocabulary.')
