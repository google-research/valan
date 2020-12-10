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

"""Hyperparam flags."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

FLAGS = flags.FLAGS

# Common task settings.
flags.DEFINE_string('logdir', '/tmp/agent', 'TensorFlow log directory.')
flags.DEFINE_string('problem', '', 'Problem type, e.g., R2R | NDH | R2R+NDH')
flags.DEFINE_integer('task', 0, 'Task id.')
flags.DEFINE_integer('num_tasks', 40, 'Number of sharded tasks.')
flags.DEFINE_integer('unroll_length', 10, 'Unroll length in agent steps.')
flags.DEFINE_string(
    'server_address', '',
    'Actors use server_address to connect to a learner. Eval actors use '
    'server_address to connect to an eval aggregator. Learners and eval '
    'aggregators use server_address to bind to a listen address.')


# Actor settings.
flags.DEFINE_string(
    'agent_type', 'R2R',
    'Agent type for a given problem, e.g., the R2R problem has two types of '
    'agents, R2R and MT.')
flags.DEFINE_string(
    'data_source', '',
    'A comma-separated list of sources to read the data from.'
    'This is usually name(s) of the splits from which the actor'
    ' reads the data (e.g., train, val_seen etc).')
flags.DEFINE_enum('mode', 'train', ['train', 'eval', 'predict'],
                  'A string representing mode.')
flags.DEFINE_string(
    'curriculum', '', 'Optional. Currently accept two curriculum methods:\n'
    'constant-X-Y: X is the number of initial paths\n'
    '              Y is the increment at each step\n'
    'adaptive-X-Y: X is the number of initial paths\n'
    '              Y is expected steps before all paths'
    ' are added\n'
    'Both X and Y could either be float or int.')
flags.DEFINE_integer(
    'sync_agent_every_n_steps', 1,
    'Synchronize agent variable values every n steps. Increasing this will '
    'reduce bandwidth consumption at the cost of training on older policies.')


# Aggregator settings.
flags.DEFINE_string('aggregator_prefix', 'agg_default',
                    'A string representing mode.')


# Learner settings.
flags.DEFINE_integer('save_checkpoint_secs', 600,
                     'Checkpoint save duration in seconds.')
flags.DEFINE_integer('total_environment_frames', int(1e9),
                     'Total environment frames to train for.')
flags.DEFINE_enum('agent_device', 'CPU', ['CPU', 'GPU', 'TPU'],
                  'The device on which to run the agent model.')
flags.DEFINE_integer('queue_capacity', 0, 'Capacity of the learner queue')
flags.DEFINE_integer('batch_size', 1, 'Batch size for training.')
flags.DEFINE_integer('max_ckpt_to_keep', 5, 'Max num of ckpt to keep.')
flags.DEFINE_string('warm_start_ckpt', None,
                    'Ckpt path to warm start from scratch.')


# Optimizer settings.
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate.')
flags.DEFINE_float('lr_decay_rate', 1., 'decay rate for learning rate.')
flags.DEFINE_float('lr_decay_steps', 0., 'decay steps for learning rate.')
flags.DEFINE_float(
    'gradient_clip_norm', 0.,
    'Caps gradients to this value before applying to variables.')


# Loss settings.
flags.DEFINE_float('entropy_cost', 0.0001, 'Entropy cost/multiplier.')
flags.DEFINE_float('baseline_cost', .5, 'Baseline cost/multiplier.')
flags.DEFINE_float('discounting', 0.0, 'Discounting factor.')
flags.DEFINE_enum('reward_clipping', '', ['', 'abs_one', 'soft_asymmetric'],
                  'Reward clipping.')
flags.DEFINE_float('focal_loss_gamma', 2.0,
                   'The gamma (power) factor for Focal Loss.')
flags.DEFINE_float(
    'focal_loss_alpha', 0.5, 'Alpha factor for Focal Loss. Positive labels are '
    'weighted by alpha, negative labels are weighted by (1 - alpha); 0.5 means '
    'positive and negative are equally weighted.')
flags.DEFINE_float('focal_loss_normalizer', 0.1, 'Normalizer for Focal Loss.')
flags.DEFINE_bool(
    'use_batch_and_ce_losses', True, 'If set, then combine contrastive batch'
    'loss with classification CE loss for the discriminator when `loss_type` is'
    'batch_loss.'
)
flags.DEFINE_float('disc_batch_loss_scale', 1.0, 'Scale multiplier for '
                   'discriminator batch softmax loss when used with CE loss.')


# Eval actor settings for running without an aggregator.
flags.DEFINE_string('aggregation_save_path', '',
                    'File to save aggregated eval results.')
flags.DEFINE_string(
    'checkpoint_path', '',
    'Optional eval checkpoint path. Required when running without aggregator.')
flags.DEFINE_string(
    'save_dir', '',
    'Eval output save dir. Required when running without aggregator.')
flags.DEFINE_string(
    'file_prefix', 'valan_score_test',
    'File identifier prefix for eval output. Required when running without '
    'aggregator.')
