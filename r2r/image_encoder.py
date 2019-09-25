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

"""Text encoder class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS


class ImageEncoder(tf.keras.Model):
  """Encode text using glove embedding and Bi-LSTM."""

  def __init__(self, attention_space_size, num_lstm_units, num_hidden_layers=1):
    super(ImageEncoder, self).__init__()
    # Projection layers to do attention pooling.
    self._projection_hidden_layer = tf.keras.layers.Dense(attention_space_size)
    self._projection_image_feature = tf.keras.layers.Dense(attention_space_size)

    self._cells = []
    for _ in range(num_hidden_layers):
      self._cells.append(tf.keras.layers.LSTMCell(num_lstm_units))
    self.history_context_encoder = tf.keras.layers.StackedRNNCells(self._cells)

    self.attention = tf.keras.layers.Attention(use_scale=True)

  def call(self, image_features, current_lstm_state):
    """Function call.

    Args:
      image_features: A tensor with shape[batch_size, num_views,
        feature_vector_length]
      current_lstm_state: A list of (state_c, state_h) tuple

    Returns:
      next_hidden_state: Hidden state vector [batch_size, lstm_space_size],
        current steps's LSTM output.
      next_lstm_state: Same shape as current_lstm_state.
    """
    # Attention-based visual-feature pooling. Pool the visual features of
    # shape [batch_size, num_views, feature_vector_length] to
    # [batch_size, attention_space_size].

    # LSTM state is a tuple (h, c) and `current_lstm_state` is a list of such
    # tuples. We use last LSTM layer's `h` to attention-pool current step's
    # image features.
    previous_step_lstm_output = current_lstm_state[-1][0]
    # [batch_size, 1, lstm_space_size]
    hidden_state = tf.expand_dims(previous_step_lstm_output, axis=1)
    # [batch_size, 1, attention_space_size]
    x = self._projection_hidden_layer(hidden_state)
    # [batch_size, num_view, attention_space_size]
    y = self._projection_image_feature(image_features)

    # v_t has shape[batch_size, 1, attention_space_size], representing the
    # current visual context.
    v_t = self.attention([x, y])


    v_t = tf.squeeze(v_t, axis=1)
    next_lstm_output, next_state = self.history_context_encoder(
        v_t, current_lstm_state)

    return (next_lstm_output, next_state)

