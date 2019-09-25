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

"""Tests for valan.r2r.image_encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from valan.r2r import image_encoder


class ImageEncoderTest(tf.test.TestCase):

  def test_encoding(self):
    batch_size = 8
    attention_space_size = 128
    lstm_space_size = 256
    num_panos = 5
    image_feature_size = 64
    num_hidden_layers = 2

    encoder = image_encoder.ImageEncoder(
        attention_space_size,
        lstm_space_size,
        num_hidden_layers=num_hidden_layers)
    image_features = tf.random.normal(
        [batch_size, num_panos, image_feature_size])
    states = [(tf.random.normal([batch_size, lstm_space_size]),
               tf.random.normal([batch_size, lstm_space_size])),
              (tf.random.normal([batch_size, lstm_space_size]),
               tf.random.normal([batch_size, lstm_space_size]))]
    output_hidden_state, next_state = encoder(image_features, states)
    # LSTM size.
    self.assertEqual(output_hidden_state.shape, [batch_size, lstm_space_size])
    self.assertEqual(len(states), len(next_state))
    for i in range(len(states)):
      self.assertEqual(states[i][0].shape, next_state[i][0].shape)
      self.assertEqual(states[i][1].shape, next_state[i][1].shape)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
