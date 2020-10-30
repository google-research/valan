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
from absl import logging

FLAGS = flags.FLAGS

# Clean duplicate flags before new declarations to avoid conflicts.
_KEYS_TO_CLEAN = ['mode']
for key in _KEYS_TO_CLEAN:
  
  if key in FLAGS._flags():
    FLAGS.__delattr__(key)
    logging.warn('Override FLAG: %s. No action needed if intended.', key)


# Overwrite `mode` from hyperparam_flags.
flags.DEFINE_string('mode', 'train', 'Job mode.')

flags.DEFINE_boolean('panoramic_actions', True,
                     'Use panoramic action space instead of discrete.')


logging.info('Added problem-specific flags from `custom_flags.py`.')
