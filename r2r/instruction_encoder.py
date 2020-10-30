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
from absl import logging
import numpy as np

import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS


class InstructionEncoder(tf.keras.Model):
  """Encode text using glove embedding and Bi-LSTM."""

  def __init__(self,
               num_hidden_layers,
               output_dim,
               pretrained_embed_path,
               oov_bucket_size,
               vocab_size=1082,
               word_embed_dim=300,
               l2_scale=0.0,
               dropout=0.0,
               layernorm=False,
               mode=None,
               use_bert_embeddings=False,
               name=None):
    super(InstructionEncoder,
          self).__init__(name=name if name else 'ins_encoder')
    self._l2_scale = l2_scale
    self._use_bert_embeddings = use_bert_embeddings
    if self._use_bert_embeddings:
      self._input_projection = tf.keras.layers.Dense(
          output_dim, name='bert_emb_projection')
      logging.info('Instruction Encoder uses Bert embeddings.')
    else:
      self._word_embeddings = self._get_embedding_layer(pretrained_embed_path,
                                                        oov_bucket_size,
                                                        vocab_size,
                                                        word_embed_dim)
    self._bi_lstm = self._get_bi_lstm_encoder(num_hidden_layers, output_dim)
    # Input dropout and layernorm layers.
    self._use_layernorm = layernorm
    self._input_dropout = tf.keras.layers.Dropout(dropout, seed=42)
    self._input_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    # State dropout and layernorm.
    self._state_h_dropout = tf.keras.layers.Dropout(dropout, seed=42)
    self._state_h_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self._state_c_dropout = tf.keras.layers.Dropout(dropout, seed=42)
    self._state_c_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    if dropout > 0.0 and not mode:
      raise ValueError(
          '`mode` must be set to train/eval/predict when using dropout. Got: {}'
          .format(mode))
    self._is_training = True if mode == 'train' else False

  def _get_embedding_layer(self, pretrained_embed_path, oov_buckets_size,
                           vocab_size, embed_dim):
    """Get word embedding layer.

    Args:
      pretrained_embed_path: Pretrained glove embedding path.
      oov_buckets_size: Out-of-vocabularies bucket size.
      vocab_size: vocabulary size (used if pretrained_embed_path is None).
      embed_dim: the dimension of word embeddings ( used if
        pretrained_embed_path is None).

    Returns:
      A tf.keras.layers.Embedding instance.
    """
    if pretrained_embed_path:
      with tf.io.gfile.GFile(pretrained_embed_path, 'rb') as f:
        floats_np = np.load(f)
      vocab_size = floats_np.shape[0]
      embed_dim = floats_np.shape[1]
      # Initialize word embeddings
      init_tensor = tf.constant(floats_np)
      oov_init = tf.compat.v1.truncated_normal_initializer(stddev=0.01)(
          shape=(oov_buckets_size, embed_dim), dtype=tf.float32)
      init_tensor = tf.concat([init_tensor, oov_init], axis=0)
    else:
      init_tensor = tf.compat.v1.truncated_normal_initializer(stddev=0.01)(
          shape=(vocab_size + oov_buckets_size, embed_dim), dtype=tf.float32)

    embeddings_initializer = tf.constant_initializer(init_tensor.numpy())
    # Now the init_tensor should have shape
    # [vocab_size+_OOV_BUCKETS_SIZE, embed_dim]
    return tf.keras.layers.Embedding(
        vocab_size + oov_buckets_size,
        embed_dim,
        embeddings_initializer=embeddings_initializer,
        mask_zero=True,
        name='embedding')

  def _get_bi_lstm_encoder(self, num_hidden_layers, hidden_dim):
    """Get Bi-LSTM encoder.

    Args:
      num_hidden_layers: Number of stacked layers.
      hidden_dim: The hidden size of LSTM.

    Returns:
      A list of 2N+1 elements. The first element is output of all timesteps
        and the others are LSTM state. The first N are forward states and the
        last N are backward states.
    """
    self._cells = []
    for layer_id in range(num_hidden_layers):

      self._cells.append(
          tf.keras.layers.LSTMCell(
              hidden_dim,
              kernel_regularizer=tf.keras.regularizers.l2(self._l2_scale),
              recurrent_regularizer=tf.keras.regularizers.l2(self._l2_scale),
              name='lstm_layer_{}'.format(layer_id)))

    self._cells_rnn = tf.keras.layers.RNN(
        self._cells, return_sequences=True, return_state=True)
    return tf.keras.layers.Bidirectional(self._cells_rnn, merge_mode='concat')

  def call(self, input_tensor):
    """Function call of instruction encoder.

    Args:
      input_tensor: tf.int64 tensor with shape [batch_size, max_seq_length]
        padded with some pad token id. If `use_bert_emb`, then tf.float32 tensor
        of shape [batch_size, max_seq_length, embedding_dim].

    Returns:
      A tuple<output, states>.
      Output: a tf.float32 tensor with shape [batch_size, max_seq_length,
        output_dim].
      State: A list which has num_hidden_layer elements. Every elements is
        a (state_c, state_h) tuple. This is concatenated last forward and
        backward states of each LSTM layer.
    """
    if self._use_bert_embeddings:
      tf.debugging.assert_rank(input_tensor, 3)
      embedding = self._input_projection(input_tensor)
    else:
      # tf.float32 [batch_size, max_seq_length, word_embedding_dim]
      embedding = self._word_embeddings(input_tensor)

    # Input dropout and layernorm.
    embedding = self._input_dropout(embedding, training=self._is_training)
    if self._use_layernorm:
      embedding = self._input_layernorm(embedding)

    # The result is a list of 2N+1 elements:
    # first element - output of all timesteps with dimension
    # [batch_size, time, hidden_dim]
    # The remaining 2N elements are final LSTM states of each of the N hidden
    # layers. The first N elements are forward states, the last N are backward
    # states.
    bilstm_result = self._bi_lstm(embedding)

    # tf.float32 [batch_size, max_seq_length, hidden_dim]
    output = bilstm_result[0]

    # a list (2 * num_hidden_layers) of states.
    states = bilstm_result[1:]

    num_states = len(states)
    encoder_states = []
    for idx in range(int(num_states / 2)):
      # every element in result is LSTMStateTuple, the first one is h and second
      # one is c.
      state_h = tf.concat(
          [states[idx][0], states[idx + int(num_states / 2)][0]], axis=1)
      # Add dropout and layernorm.
      state_h = self._state_h_dropout(state_h, training=self._is_training)
      if self._use_layernorm:
        state_h = self._state_h_layernorm(state_h)
      state_c = tf.concat(
          [states[idx][1], states[idx + int(num_states / 2)][1]], axis=1)
      # Add dropout and layernorm.
      state_c = self._state_c_dropout(state_c, training=self._is_training)
      if self._use_layernorm:
        state_c = self._state_c_layernorm(state_c)

      encoder_states.append((state_h, state_c))
    return (output, encoder_states)
