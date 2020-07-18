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
flags.DEFINE_integer('max_ckpt_to_keep', 5, 'Max num of ckpt to keep.')

# Loss settings.
flags.DEFINE_float('entropy_cost', 0.0001, 'Entropy cost/multiplier.')
flags.DEFINE_float('baseline_cost', .5, 'Baseline cost/multiplier.')
flags.DEFINE_float('discounting', 0.0, 'Discounting factor.')
flags.DEFINE_enum('reward_clipping', '', ['', 'abs_one', 'soft_asymmetric'],
                  'Reward clipping.')
flags.DEFINE_float('focal_loss_gamma', 2.0,
                   'The gamma (power) factor for focal loss.')
flags.DEFINE_float(
    'focal_loss_alpha', 0.5, 'Alpha factor for focal loss. Positive labels are '
    'weighted by alpha, negative labels are weighted by (1 - alpha); 0.5 means '
    'positive and negative are equally weighted.')
flags.DEFINE_float('focal_loss_normalizer', 0.1, 'Normalizer for focal loss.')

# Discriminator loss settings.
flags.DEFINE_bool(
    'use_batch_and_ce_losses', True, 'If set, then combine batch softmax loss'
    'with classification CE loss for the discriminator when `loss_type` is '
    'batch_loss.'
)
flags.DEFINE_float('disc_batch_loss_scale', 1.0, 'Scale multiplier for '
                   'discriminator batch softmax loss when used with CE loss.')

# Optimizer settings.
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate.')
flags.DEFINE_float('lr_decay_rate', 1., 'decay rate for learning rate.')
flags.DEFINE_float('lr_decay_steps', 0., 'decay steps for learning rate.')
flags.DEFINE_float(
    'gradient_clip_norm', 0.,
    'Caps gradients to this value before applying to variables.')
