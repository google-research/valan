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

"""Learner configs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('save_checkpoint_secs', 600,
                     'Checkpoint save duration in seconds.')
flags.DEFINE_integer('total_environment_frames', int(1e9),
                     'Total environment frames to train for.')
flags.DEFINE_enum('agent_device', 'CPU', ['CPU', 'GPU', 'TPU'],
                  'The device on which to run the agent model.')
flags.DEFINE_integer('queue_capacity', 0, 'Capacity of the learner queue')

flags.DEFINE_integer('batch_size', 1, 'Batch size for training.')

# Loss settings.
flags.DEFINE_float('entropy_cost', 0.0001, 'Entropy cost/multiplier.')
flags.DEFINE_float('baseline_cost', .5, 'Baseline cost/multiplier.')
flags.DEFINE_float('discounting', 0.0, 'Discounting factor.')
flags.DEFINE_enum('reward_clipping', '', ['', 'abs_one', 'soft_asymmetric'],
                  'Reward clipping.')

# Optimizer settings.
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate.')
flags.DEFINE_float('lr_decay_rate', 1., 'decay rate for learning rate.')
flags.DEFINE_float('lr_decay_steps', 0., 'decay steps for learning rate.')
