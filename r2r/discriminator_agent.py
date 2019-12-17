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

"""Discriminator Agent code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from valan.framework import base_agent
from valan.framework import common
from valan.framework import utils
from valan.r2r import constants
from valan.r2r import image_encoder
from valan.r2r import instruction_encoder

_INFINITY = 1e9


class DiscriminatorAgent(base_agent.BaseAgent):
  """R2R Agent."""

  def __init__(self, config):
    """Initialize R2R Agent."""
    super(DiscriminatorAgent, self).__init__(name='discriminator_r2r')

    self._instruction_encoder = instruction_encoder.InstructionEncoder(
        num_hidden_layers=2,
        output_dim=256,
        pretrained_embed_path=config.pretrained_embed_path,
        oov_bucket_size=config.oov_bucket_size,
        vocab_size=config.vocab_size,
        word_embed_dim=config.word_embed_dim,
    )
    self._image_encoder = image_encoder.ImageEncoder(
        256, 512, num_hidden_layers=2)
    self.affine_a = tf.Variable(1.0, dtype=tf.float32, trainable=True)
    self.affine_b = tf.Variable(0.0, dtype=tf.float32, trainable=True)

  def _get_initial_state(self, observation, batch_size):
    text_token_ids = observation[constants.INS_TOKEN_IDS]
    text_enc_outputs, final_state = self._instruction_encoder(text_token_ids)
    return (final_state, text_enc_outputs)

  def _torso(self, observation):
    # For R2R, torso does not do anything but pass all the variables.
    outputs = {}
    outputs[constants.PANO_ENC] = observation[constants.PANO_ENC]
    outputs[constants.INS_TOKEN_IDS] = observation[constants.INS_TOKEN_IDS]
    outputs[constants.IS_START] = observation[constants.IS_START]
    outputs[constants.DISC_MASK] = observation[constants.DISC_MASK]
    return outputs

  def _neck(self, torso_outputs, state):
    current_lstm_state, text_enc_outputs = state
    image_features = tf.cast(torso_outputs[constants.PANO_ENC], tf.float32)
    lstm_output, next_lstm_state = self._image_encoder(image_features,
                                                       current_lstm_state)
    # <tf.float32>[batch_size, 1, hidden_dim]
    lstm_output = tf.expand_dims(lstm_output, axis=1)
    # The next_lstm_state are ListWrappers. In order to make it consistent with
    # get_initial_state, we convert them to tuple.
    result_state = []
    for one_state in next_lstm_state:
      result_state.append((one_state[0], one_state[1]))
    torso_outputs['visual_feature'] = lstm_output
    torso_outputs['text_feature'] = text_enc_outputs
    return (torso_outputs, (result_state, text_enc_outputs))

  def _head(self, neck_outputs):

    # <tf.float32>[time * batch_size, 1, hidden_dim]
    visual_feature = neck_outputs['visual_feature']
    # <tf.float32>[time * batch_size, num_tokens, hidden_dim]
    text_feature = neck_outputs['text_feature']

    # <tf.float32>[time, batch_size, 1, hidden_dim]
    visual_feature = tf.reshape(
        visual_feature,
        [self._current_num_timesteps, self._current_batch_size] +
        visual_feature.shape[1:].as_list())

    # <tf.float32>[batch_size, time, hidden_dim]
    visual_feature = tf.squeeze(visual_feature, axis=2)
    visual_feature = tf.transpose(visual_feature, [1, 0, 2])

    first_true = utils.get_first_true_column(
        tf.reshape(neck_outputs[constants.DISC_MASK],
                   [self._current_num_timesteps, self._current_batch_size]))

    # <tf.float32>[batch_size, num_tokens, hidden_dim]
    text_feature = tf.cond(
        tf.keras.backend.any(first_true),
        lambda: tf.boolean_mask(text_feature, tf.reshape(first_true, [-1])),
        lambda: tf.reshape(text_feature, [
            self._current_num_timesteps, self._current_batch_size
        ] + text_feature.shape[1:].as_list())[0, :, :, :])
    # visual_feature = tf.nn.l2_normalize(visual_feature, axis=2)
    # text_feature = tf.nn.l2_normalize(text_feature, axis=2)

    # <tf.float32>[batch_size, time, num_tokens]
    alpha_i_j = tf.matmul(visual_feature,
                          tf.transpose(text_feature, perm=[0, 2, 1]))
    # <tf.float32>[batch_size, time, num_tokens]
    ealpha_i_j = tf.exp(alpha_i_j)
    sum_i_j = tf.tile(
        tf.expand_dims(tf.reduce_sum(ealpha_i_j, 2), 2),
        [1, 1, tf.shape(ealpha_i_j)[2]])
    mask = tf.cast(
        tf.transpose(
            tf.reshape(neck_outputs[constants.DISC_MASK],
                       [self._current_num_timesteps, self._current_batch_size]),
            perm=[1, 0]), tf.float32)
    # <tf.float32>[batch, time, num_tokens]
    c_i_j = tf.divide(ealpha_i_j, sum_i_j)
    # <tf.float32>[batch, time]
    score = tf.reduce_sum(c_i_j * alpha_i_j, 2)

    escore = tf.exp(-1 * score) * mask
    sum_escore = tf.tile(
        tf.expand_dims(tf.reduce_sum(escore, 1), 1), [1, tf.shape(escore)[1]])
    score_weight = tf.divide(escore, sum_escore)
    similarities = tf.reduce_sum(mask * score * score_weight, 1)
    similarities = tf.expand_dims(similarities, axis=0)
    # [time_step, batch_size]
    similarities = tf.tile(similarities, [self._current_num_timesteps, 1])

    # Apply an affine transform.
    similarities = similarities * self.affine_a + self.affine_b

    output_a = tf.reshape(tf.convert_to_tensor(self.affine_a), [1, 1])
    output_b = tf.reshape(tf.convert_to_tensor(self.affine_b), [1, 1])

    output_a = tf.tile(output_a,
                       [self._current_num_timesteps, self._current_batch_size])
    output_b = tf.tile(output_b,
                       [self._current_num_timesteps, self._current_batch_size])

    return common.AgentOutput(
        policy_logits=similarities, baseline=(output_a, output_b))
