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


class ImageEncoder(tf.keras.layers.Layer):
  """Encode text using glove embedding and Bi-LSTM."""

  def __init__(self,
               attention_space_size,
               num_lstm_units,
               num_hidden_layers=1,
               l2_scale=0.0,
               dropout=0.0,
               concat_context=False,
               layernorm=False,
               mode=None,
               name=None,
               use_attention_pooling=True):
    super(ImageEncoder, self).__init__(name=name if name else 'image_encoder')
    self._use_attention_pooling = use_attention_pooling
    if self._use_attention_pooling:
      # Projection layers to do attention pooling.
      self._projection_hidden_layer = tf.keras.layers.Dense(
          attention_space_size, name='project_hidden')
      self._projection_image_feature = tf.keras.layers.Dense(
          attention_space_size, name='project_feature')
    else:
      self._dense_pooling_layer = tf.keras.layers.Dense(
          num_lstm_units, name='dense_pooling')

    self._cells = []
    for layer_id in range(num_hidden_layers):
      self._cells.append(
          tf.keras.layers.LSTMCell(
              num_lstm_units,
              kernel_regularizer=tf.keras.regularizers.l2(l2_scale),
              recurrent_regularizer=tf.keras.regularizers.l2(l2_scale),
              name='lstm_layer_{}'.format(layer_id)))
    self.history_context_encoder = tf.keras.layers.StackedRNNCells(self._cells)

    self.attention = tf.keras.layers.Attention(use_scale=True, name='attention')

    # Context dropout and layernorm layers.
    self._use_layernorm = layernorm
    self._context_dropout = tf.keras.layers.Dropout(dropout)
    self._context_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    # Projection layernorm.
    self._hidden_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self._image_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    # If concat_context is set to True, then concat lstm_output with context.
    self._concat_context = concat_context
    self._dense = tf.keras.layers.Dense(num_lstm_units, activation='tanh')

    if dropout > 0.0 and not mode:
      raise ValueError(
          '`mode` must be set to train/eval/predict when using dropout.')
    self._is_training = True if mode == 'train' else False

  def _attention_pooling(self, image_features, current_lstm_state):
    """Returns the input tensor for subsequent LSTM using attention pooling.

    Args:
      image_features: A tensor with shape[batch_size, num_views,
        feature_vector_length]
      current_lstm_state: A list of (state_c, state_h) tuple.

    Returns:
      A pooled visual feature tensor of shape [batch_size, lstm_space_size].
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
    if self._use_layernorm:
      x = self._hidden_layernorm(x)
    # [batch_size, num_view, attention_space_size]
    y = self._projection_image_feature(image_features)
    if self._use_layernorm:
      y = self._image_layernorm(y)

    # v_t has shape[batch_size, 1, attention_space_size], representing the
    # current visual context.
    v_t = self.attention([x, y])
    v_t = tf.squeeze(v_t, axis=1)
    return v_t

  def _dense_pooling(self, image_features):
    """Flattens and projects all views of pano features into LSTM hidden dim."""
    batch_size = image_features.shape[0]
    flat_image_features = tf.reshape(image_features, [batch_size, -1])
    v_t = self._dense_pooling_layer(flat_image_features)
    return v_t

  def call(self, image_features, current_lstm_state, prev_action=None):
    """Function call.

    Args:
      image_features: A tensor with shape[batch_size, num_views,
        feature_vector_length]
      current_lstm_state: A list of (state_c, state_h) tuple
      prev_action: Optional tensor with shape[batch_size, feature_dim]

    Returns:
      next_hidden_state: Hidden state vector [batch_size, lstm_space_size],
        current steps's LSTM output.
      next_lstm_state: Same shape as current_lstm_state.
    """
    if self._use_attention_pooling:
      v_t = self._attention_pooling(image_features, current_lstm_state)
    else:
      v_t = self._dense_pooling(image_features)

    if prev_action is not None:
      # Shape = [batch_size, attention_space_size + full_feature_dim]
      v_t = tf.concat([v_t, prev_action], -1)
    v_t = self._context_dropout(v_t, training=self._is_training)
    if self._use_layernorm:
      v_t = self._context_layernorm(v_t)
    next_lstm_output, next_state = self.history_context_encoder(
        v_t, current_lstm_state)

    # Concat context vector to lstm output if concat_context=True.
    if self._concat_context:
      # Shape: [batch_size, 1, attention_space_size]
      next_output = tf.concat([v_t, next_lstm_output], axis=-1)
      next_output = self._dense(next_output)
    else:
      next_output = next_lstm_output

    return (next_output, next_state)
