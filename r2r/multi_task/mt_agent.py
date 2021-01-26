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

"""Multi-tasking env-agnostic agent code."""

import tensorflow.compat.v2 as tf
from valan.framework import base_agent
from valan.framework import common

from valan.r2r import constants
from valan.r2r import image_encoder
from valan.r2r import instruction_encoder

_INFINITY = 1e9
_LAMBDA_VAL = 1.0


@tf.custom_gradient
def grad_reverse(x):
  y = tf.identity(x)

  def custom_grad(dy):
    return -dy * _LAMBDA_VAL

  return y, custom_grad


class GradReverseLayer(tf.keras.layers.Layer):
  """Gradient reversal layer."""

  def call(self, x):
    return grad_reverse(x)


class MTEnvAgAgent(base_agent.BaseAgent):
  """Multitasking env-agnostic Agent."""

  def __init__(self, config, mode=None):
    """Initialize the Agent."""
    super(MTEnvAgAgent, self).__init__(name='agent_mt_envag')

    self._ins_classifier = None
    self._scan_classifier = None
    if config.classify_instructions:
      self._ins_classifier = self._get_ins_classifier(config.classifier_dropout)
    if config.classify_scans:
      self._scan_classifier = self._get_scan_classifier(
          config.classifier_dropout)

    self._instruction_encoder = instruction_encoder.InstructionEncoder(
        num_hidden_layers=2,
        output_dim=256,
        pretrained_embed_path=config.pretrained_embed_path,
        oov_bucket_size=config.oov_bucket_size,
        vocab_size=config.vocab_size,
        word_embed_dim=config.word_embed_dim,
        mode=mode,
    )

    self._ndh_instruction_encoder = None
    if config.use_separate_encoders:
      self._ndh_instruction_encoder = instruction_encoder.InstructionEncoder(
          num_hidden_layers=2,
          output_dim=256,
          pretrained_embed_path=config.pretrained_embed_path,
          oov_bucket_size=config.oov_bucket_size,
          vocab_size=config.vocab_size,
          word_embed_dim=config.word_embed_dim,
          mode=mode,
      )

    self._image_encoder = image_encoder.ImageEncoder(
        256, 512, num_hidden_layers=2, mode=mode)

    # Text attention.
    self._text_attention_size = 512
    self._text_attention_project_hidden = tf.keras.layers.Dense(
        self._text_attention_size, name='text_attention_project_hidden')
    self._text_attention_project_text = tf.keras.layers.Dense(
        self._text_attention_size, name='text_attention_project_text')
    self._text_attention = tf.keras.layers.Attention(
        use_scale=True, name='text_attention')

    # Visual attention.
    self._visual_attention_size = 256
    self._visual_attention_project_ctext = tf.keras.layers.Dense(
        self._visual_attention_size, name='vis_attention_project_ctext')
    self._visual_attention_project_feature = tf.keras.layers.Dense(
        self._visual_attention_size, name='vis_attention_project_feature')
    self._visual_attention = tf.keras.layers.Attention(
        use_scale=True, name='vis_attention')

    # Action predictor projection.
    self._action_projection_size = 256
    self._project_feature = tf.keras.layers.Dense(
        self._action_projection_size, name='action_layer_project_feature')
    self._project_action = tf.keras.layers.Dense(
        self._action_projection_size, name='action_layer_project_action')
    # Dot product over the last dimension.
    self._dot_product = tf.keras.layers.Dot(axes=2)

    # Value network.
    self._value_network = self._get_value_network()

  def _get_ins_classifier(self, dropout):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(GradReverseLayer())  # reverse the gradient passing
    model.add(tf.keras.layers.Dense(32))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dense(2))  # 2 for number of tasks.
    return model

  def _get_scan_classifier(self, dropout):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(GradReverseLayer())  # reverse the gradient passing
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dense(61))  # 61 for number of scans in training.
    return model

  def _get_value_network(self):
    model = tf.keras.Sequential(name='value_net')
    model.add(tf.keras.layers.Dense(64, name='hidden_layer'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dense(1, name='logit_layer'))
    return model

  def _get_initial_state(self, observation, batch_size):
    text_token_ids = observation[constants.INS_TOKEN_IDS]
    text_enc_outputs, final_state = self._instruction_encoder(text_token_ids)
    if self._ndh_instruction_encoder is not None:
      ndh_text_enc_outputs, ndh_final_state = self._ndh_instruction_encoder(
          text_token_ids)
      mask = tf.equal(observation[constants.PROBLEM_TYPE],
                      constants.PROBLEM_VLN)
      text_enc_outputs = tf.nest.map_structure(
          lambda x, y: tf.compat.v1.where(mask, x, y), text_enc_outputs,
          ndh_text_enc_outputs)
      final_state = tf.nest.map_structure(
          lambda x, y: tf.compat.v1.where(mask, x, y), final_state,
          ndh_final_state)

    if self._ins_classifier is not None:
      # Concatenate all hidden layers' state vectors. Use state.h
      ins_classifier_logits = self._ins_classifier(
          tf.concat([s[0] for s in final_state], axis=1))
    else:
      ins_classifier_logits = tf.zeros(shape=(batch_size, 2))
    return (final_state, text_enc_outputs, ins_classifier_logits)

  def _torso(self, observation):
    # The torso does not do anything but pass all the variables.
    outputs = {}
    outputs[constants.PANO_ENC] = observation[constants.PANO_ENC]
    outputs[constants.INS_TOKEN_IDS] = observation[constants.INS_TOKEN_IDS]
    outputs[constants.CONN_ENC] = observation[constants.CONN_ENC]
    outputs[constants.VALID_CONN_MASK] = observation[constants.VALID_CONN_MASK]

    return outputs

  def _neck(self, torso_outputs, state):
    current_lstm_state, text_enc_outputs, ins_classifier_logits = state
    image_features = tf.cast(torso_outputs[constants.PANO_ENC], tf.float32)
    lstm_output, next_lstm_state = self._image_encoder(image_features,
                                                       current_lstm_state)

    lstm_output = tf.expand_dims(lstm_output, axis=1)

    # c_text has shape [batch_size, 1, self._text_attention_size]
    c_text = self._text_attention([
        self._text_attention_project_hidden(lstm_output),
        self._text_attention_project_text(text_enc_outputs)
    ])
    # The next_lstm_state are ListWrappers. In order to make it consistent with
    # get_initial_state, we convert them to tuple.
    result_state = []
    for one_state in next_lstm_state:
      result_state.append((one_state[0], one_state[1]))
    torso_outputs['hidden_state'] = lstm_output
    torso_outputs['c_text'] = c_text
    torso_outputs['ins_classifier_logits'] = ins_classifier_logits
    return (torso_outputs, (result_state, text_enc_outputs,
                            ins_classifier_logits))

  def _head(self, neck_outputs):
    # The shape of hidden_state is [batch_size * time, 1, hidden_size]
    hidden_state = neck_outputs['hidden_state']
    if self._scan_classifier is not None:
      scan_classifier_logits = self._scan_classifier(hidden_state)
    else:
      scan_classifier_logits = tf.zeros(shape=(tf.shape(hidden_state)[0], 61))
    image_features = tf.cast(neck_outputs[constants.PANO_ENC], tf.float32)

    # c_visual has shape [batch_size * time, 1, self._c_visual_attention_size]
    c_visual = self._visual_attention([
        self._visual_attention_project_ctext(neck_outputs['c_text']),
        self._visual_attention_project_feature(image_features),
    ])

    # Concatenating the h_t, c_text and c_visual as described in RCM paper.
    input_feature = tf.concat([hidden_state, neck_outputs['c_text'], c_visual],
                              axis=2)
    connection_encoding = neck_outputs[constants.CONN_ENC]
    connection_encoding = tf.cast(connection_encoding, tf.float32)
    logits = self._dot_product([
        self._project_feature(input_feature),
        self._project_action(connection_encoding)
    ])
    # The final shape of logits is [batch_size * time, num_connections]
    logits = tf.squeeze(logits, axis=1)
    # mask out invalid connections.
    valid_conn_mask = tf.cast(neck_outputs[constants.VALID_CONN_MASK],
                              tf.float32)
    logits += (1. - valid_conn_mask) * -_INFINITY
    value = self._value_network(tf.squeeze(neck_outputs['c_text'], axis=1))
    value = tf.squeeze(value, axis=1)
    return common.AgentOutput(
        policy_logits=logits,
        baseline={
            'value': value,
            'ins_classifier_logits': neck_outputs['ins_classifier_logits'],
            'scan_classifier_logits': scan_classifier_logits,
        })
