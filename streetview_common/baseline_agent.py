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

"""Baseline Touchdown agent."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from absl import logging
import tensorflow.compat.v2 as tf
from valan.framework import base_agent
from valan.framework import common
from valan.streetview_common import streetview_constants

from tensorflow.python.client import device_lib  


class StreetviewAgent(base_agent.BaseAgent):
  """Modeling code for Touchdown navigation agent."""

  def __init__(self, num_actions, instruction_len, params, debug_writer=None):
    super(StreetviewAgent, self).__init__(name='agent_touchdown')
    self._debug_writer = debug_writer

    devices = device_lib.list_local_devices()
    logging.info('Agent Found devices: %s', devices)
    gpu_devices = [d for d in devices if d.device_type == 'GPU']
    logging.info('Agent Found GPU devices: %s', devices)
    if gpu_devices:
      self._fast_device = gpu_devices[0].name
    else:
      self._fast_device = devices[0].name
    logging.info('Using supplementary device: %s', self._fast_device)

    self._debug_writer = debug_writer

    self._action_embedder = tf.keras.layers.Embedding(
        input_dim=num_actions, output_dim=params.ACTION_EMBED_DIM)

    self._timestep_embedder = tf.keras.layers.Embedding(
        input_dim=params.MAX_AGENT_ACTIONS + 1,
        output_dim=params.TIMESTEP_EMBED_DIM)

    self._core = tf.keras.layers.LSTMCell(
        256,
        kernel_regularizer=tf.keras.regularizers.l2(params.L2_SCALE),
        recurrent_regularizer=tf.keras.regularizers.l2(params.L2_SCALE),
    )

    self._dense_img = tf.keras.layers.Dense(256, activation=tf.nn.sigmoid)
    self._dense_img_extra = tf.keras.layers.Dense(256, activation=tf.nn.sigmoid)
    self._dense_text = tf.keras.layers.Dense(2048)

    self._text_embedder = tf.keras.layers.Embedding(
        input_dim=params.VOCAB_SIZE,
        output_dim=params.TEXT_EMBED_DIM,
        input_length=instruction_len,
        mask_zero=True)

    self._instruction_rnn_layer = tf.keras.layers.RNN(
        tf.keras.layers.LSTMCell(
            params.INSTRUCTION_LSTM_DIM,
            kernel_regularizer=tf.keras.regularizers.l2(params.L2_SCALE),
            recurrent_regularizer=tf.keras.regularizers.l2(params.L2_SCALE),
        ),
        return_sequences=True)

    self._policy_logits = tf.keras.layers.Dense(
        num_actions, name='policy_logits')
    self._baseline = tf.keras.layers.Dense(1, name='baseline')

  def _get_initial_state(self, observation, batch_size):
    instruction = observation[streetview_constants.NAV_TEXT]
    length = observation[streetview_constants.NAV_TEXT_LENGTH]
    instruction_embedded = self._instruction(instruction, length)
    core_state = self._core.get_initial_state(
        batch_size=batch_size, dtype=tf.float32)
    return (instruction_embedded, core_state)

  def _instruction(self, instruction, length):
    """Processing of the language instructions."""
    embedding = self._text_embedder(instruction)

    # Pad to make sure there is at least one output.
    padding = tf.cast(tf.equal(tf.shape(embedding)[1], 0), dtype=tf.int32)
    embedding = tf.pad(embedding, [[0, 0], [0, padding], [0, 0]])

    output = self._instruction_rnn_layer(embedding)
    output = tf.reverse_sequence(output, length, seq_axis=1)[:, 0]
    output = tf.keras.layers.Flatten()(output)
    return output

  def _torso(self, observation):
    conv_out = observation[streetview_constants.IMAGE_FEATURES]
    heading = observation[streetview_constants.HEADING]
    last_action = observation[streetview_constants.PREV_ACTION_IDX]

    conv_out = tf.cast(conv_out, tf.float32)

    img_encoding = self._dense_img_extra(self._dense_img(conv_out))
    img_encoding = tf.keras.layers.Flatten()(img_encoding)

    heading = tf.expand_dims(heading, -1)
    last_action_embedded = self._action_embedder(last_action)

    torso_output = tf.concat([heading, last_action_embedded, img_encoding],
                             axis=1)
    timestep_embedded = self._timestep_embedder(
        observation[streetview_constants.TIMESTEP])
    return {
        'neck_input': torso_output,
        streetview_constants.TIMESTEP: timestep_embedded,
    }

  def _neck(self, torso_outputs, state):
    instruction_embedded, core_state = state
    torso_output = tf.concat(
        [torso_outputs['neck_input'], instruction_embedded], axis=1)
    core_output, core_state = self._core(torso_output, core_state)
    neck_output = tf.concat(
        [core_output, torso_outputs[streetview_constants.TIMESTEP]], axis=-1)
    return neck_output, (instruction_embedded, core_state)

  def _head(self, neck_outputs):
    logits = self._policy_logits(neck_outputs)
    if self._debug_writer:
      self._debug_writer.log_named_tensor(
          'action_logits', logits.numpy())
    value = tf.squeeze(self._baseline(neck_outputs), axis=-1)
    return common.AgentOutput(policy_logits=logits, baseline=value)
