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

"""Tests for valan.r2r.language_reward_net."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags

import tensorflow.compat.v2 as tf

from valan.r2r import language_reward_net

FLAGS = flags.FLAGS


class LanguageRewardNetTest(tf.test.TestCase):

  def setUp(self):
    super(LanguageRewardNetTest, self).setUp()
    self.reward_net = language_reward_net.LanguageRewardNet(100, 2)
    self.instruction_encoder = self.reward_net._instruction_encoder
    self.image_encoder = self.reward_net._image_encoder
    self.data_dir = FLAGS.test_srcdir + (
        'valan/r2r/testdata')

  def test_get_instruction_from_token_ids(self):
    instruction = 'walk to the table on the left.'
    fixed_instruction_len = 30
    vocab_path = os.path.join(self.data_dir, 'vocab.txt')
    instruction_token_ids, _ = self.reward_net.instruction_to_token_ids(
        instruction,
        fixed_instruction_len,
        vocab_path)
    self.assertLen(instruction_token_ids[0], fixed_instruction_len)
    # instruction_token_ids_2, _ self.reward_net._get_token_ids

  def test_instruction_encoder(self):
    tokens = tf.constant([[3, 4, 5, 1, 6, 0]])
    text_encoding, lstm_state = self.instruction_encoder(tokens)
    self.assertEqual(text_encoding.shape, [1, 6, 512])
    self.assertEqual(lstm_state[0][0].shape, [1, 512])
    self.assertEqual(lstm_state[0][1].shape, [1, 512])

  def test_individual_image_encoder(self):
    batch_size = 8
    lstm_space_size = 512
    num_panos = 2
    image_feature_size = 64

    image_features = tf.random.normal(
        [batch_size, num_panos, image_feature_size])
    states = [(tf.random.normal([batch_size, lstm_space_size]),
               tf.random.normal([batch_size, lstm_space_size])),
              (tf.random.normal([batch_size, lstm_space_size]),
               tf.random.normal([batch_size, lstm_space_size]))]
    output_hidden_state, next_state = self.image_encoder(image_features,
                                                         states)
    self.assertEqual(output_hidden_state.shape, [batch_size, lstm_space_size])
    self.assertEqual(len(states), len(next_state))

if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
