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

import itertools
from absl import logging
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

  def __init__(self, config, mode=None, name=None):
    """Initialize R2R Agent."""
    super(DiscriminatorAgent,
          self).__init__(name=name if name else 'discriminator_r2r')
    use_bert_emb = (
        config.use_bert_emb if hasattr(config, 'use_bert_emb') else False)
    self._instruction_encoder = instruction_encoder.InstructionEncoder(
        num_hidden_layers=2,
        output_dim=256,
        pretrained_embed_path=config.pretrained_embed_path,
        oov_bucket_size=config.oov_bucket_size,
        vocab_size=config.vocab_size,
        word_embed_dim=config.word_embed_dim,
        l2_scale=config.l2_scale,
        dropout=config.dropout,
        layernorm=config.layernorm,
        use_bert_embeddings=use_bert_emb,
        mode=mode)

    # If False, the text and image encoders are independent from each other.
    self._init_with_text_state = (
        config.init_image_enc_with_text_state if hasattr(
            config, 'init_image_enc_with_text_state') else True)
    self._embed_prev_action = config.embed_prev_action if hasattr(
        config, 'embed_prev_action') else False
    self._embed_next_action = config.embed_next_action if hasattr(
        config, 'embed_next_action') else False
    self._use_attn_pooling = config.use_attn_pooling if hasattr(
        config, 'use_attn_pooling') else True
    image_enc_attention_dim = (
        config.image_enc_attention_dim
        if hasattr(config, 'image_enc_attention_dim') else 256)
    image_enc_hidden_dim = (
        config.image_enc_hidden_dim
        if hasattr(config, 'image_enc_hidden_dim') else 512)
    self._image_encoder = image_encoder.ImageEncoder(
        attention_space_size=image_enc_attention_dim,
        num_lstm_units=image_enc_hidden_dim,
        num_hidden_layers=2,
        l2_scale=config.l2_scale,
        dropout=config.dropout,
        concat_context=config.concat_context,
        layernorm=config.layernorm,
        mode=mode,
        use_attention_pooling=self._use_attn_pooling)

    # Learnable projection of initial decoder state from instruction encoder.
    self._project_decoder_input_states = (
        config.project_decoder_input_states if hasattr(
            config, 'project_decoder_input_states') else False)
    if self._project_decoder_input_states:
      self._encoder_projection = tf.keras.layers.Dense(
          4 * 512, name='encoder_projection')

    self.affine_a = tf.Variable(1.0, dtype=tf.float32, trainable=True)
    self.affine_b = tf.Variable(0.0, dtype=tf.float32, trainable=True)

  def _get_initial_state(self, observation, batch_size):
    text_token_ids = observation[constants.INS_TOKEN_IDS]
    text_enc_outputs, final_state = self._instruction_encoder(text_token_ids)
    if self._init_with_text_state:
      if self._project_decoder_input_states:
        flat_state = tf.concat(list(itertools.chain(*final_state)), -1)
        start_state = tf.split(self._encoder_projection(flat_state), 4, -1)
        start_state = [(start_state[0], start_state[1]),
                       (start_state[2], start_state[3])]
      else:
        start_state = final_state
    else:
      # Initialize image encoder from scratch.
      start_state = tf.nest.map_structure(tf.zeros_like, final_state)
    return (start_state, text_enc_outputs)

  def _torso(self, observation):
    # For R2R, torso does not do anything but pass all the variables.
    outputs = {}
    outputs[constants.PANO_ENC] = observation[constants.PANO_ENC]
    outputs[constants.INS_TOKEN_IDS] = observation[constants.INS_TOKEN_IDS]
    outputs[constants.IS_START] = observation[constants.IS_START]
    outputs[constants.DISC_MASK] = observation[constants.DISC_MASK]
    outputs[constants.PATH_ID] = observation[constants.PATH_ID]

    # Shape = [batch, feature_size]
    outputs['prev_action_feature'] = observation[constants.PREV_ACTION_ENC]
    outputs['next_action_feature'] = observation[
        constants.NEXT_GOLDEN_ACTION_ENC]
    return outputs

  def _neck(self, torso_outputs, state):
    current_lstm_state, text_enc_outputs = state
    image_features = tf.cast(torso_outputs[constants.PANO_ENC], tf.float32)

    if self._embed_prev_action and self._embed_next_action:
      # Concat both prev and next action features.
      # Shape = [batch, 2 * feature_size]
      action_feature = tf.concat([
          torso_outputs['prev_action_feature'],
          torso_outputs['next_action_feature']
      ],
                                 axis=1)
      action_feature = tf.cast(action_feature, tf.float32)
      logging.info(
          'Concatenating prev and next action embeddings to pano embeddings.')
    elif self._embed_prev_action:
      # Shape = [batch_size, feature_size]
      action_feature = tf.cast(torso_outputs['prev_action_feature'], tf.float32)
      logging.info('Concatenating prev action embeddings to pano embeddings.')
    elif self._embed_next_action:
      # Shape = [batch_size, feature_size]
      action_feature = tf.cast(torso_outputs['next_action_feature'], tf.float32)
      logging.info('Concatenating next action embeddings to pano embeddings.')
    else:
      action_feature = None

    lstm_output, next_lstm_state = self._image_encoder(
        image_features, current_lstm_state, prev_action=action_feature)
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
    # Shape : [time * batch]
    path_ids = neck_outputs[constants.PATH_ID]
    path_ids = tf.transpose(
        tf.reshape(path_ids,
                   [self._current_num_timesteps, self._current_batch_size]))

    # <tf.float32>[time * batch_size, 1, hidden_dim]
    visual_feature = neck_outputs['visual_feature']
    # <tf.float32>[time * batch_size, num_tokens, hidden_dim]
    raw_text_feature = tf.reshape(
        neck_outputs['text_feature'],
        [self._current_num_timesteps, self._current_batch_size] +
        neck_outputs['text_feature'].shape[1:].as_list())
    # Shape = [batch_size, time, num_tokens, hidden_dim]
    raw_text_feature = tf.transpose(raw_text_feature, perm=[1, 0, 2, 3])

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
    first_true = tf.transpose(first_true)

    # Sanity Check: path_ids are consistent for first_true and last_true.
    last_true = utils.get_last_true_column(
        tf.reshape(neck_outputs[constants.DISC_MASK],
                   [self._current_num_timesteps, self._current_batch_size]))
    last_true = tf.transpose(last_true)
    path_ids_first_true = tf.cond(
        tf.keras.backend.any(first_true),
        lambda: tf.boolean_mask(path_ids, first_true),
        lambda: path_ids[:, 0])
    path_ids_last_true = tf.cond(
        tf.keras.backend.any(last_true),
        lambda: tf.boolean_mask(path_ids, last_true),
        lambda: path_ids[:, 0])
    tf.debugging.assert_equal(path_ids_first_true, path_ids_last_true)

    # <tf.float32>[batch_size, num_tokens, hidden_dim]
    text_feature = tf.cond(
        tf.keras.backend.any(first_true),
        lambda: tf.boolean_mask(raw_text_feature, first_true),
        lambda: raw_text_feature[:, 0, :, :])

    text_feature_last_true = tf.cond(
        tf.keras.backend.any(last_true),
        lambda: tf.boolean_mask(raw_text_feature, last_true),
        lambda: raw_text_feature[:, 0, :, :])
    tf.debugging.assert_equal(text_feature, text_feature_last_true)
    visual_feature = tf.nn.l2_normalize(visual_feature, axis=2)
    text_feature = tf.nn.l2_normalize(text_feature, axis=2)

    # <tf.float32>[batch_size, time, num_tokens]
    alpha_i_j = tf.matmul(visual_feature,
                          tf.transpose(text_feature, perm=[0, 2, 1]))
    # <tf.float32>[batch, time, num_tokens]
    c_i_j = tf.nn.softmax(alpha_i_j)
    # <tf.float32>[batch_size, time, num_tokens]
    mask = tf.cast(
        tf.transpose(
            tf.reshape(neck_outputs[constants.DISC_MASK],
                       [self._current_num_timesteps, self._current_batch_size]),
            perm=[1, 0]), tf.float32)

    # <tf.float32>[batch, time]
    score = tf.reduce_sum(c_i_j * alpha_i_j, 2)

    # Compute softmin(x) = softmax(-x)
    # Use stable softmax since softmax(x) = softmax(x+c) for any constant c.
    # Here we use constant c = max(-x).
    negative_score = -1.0 * score
    escore = tf.exp(negative_score - tf.reduce_max(negative_score)) * mask
    sum_escore = tf.tile(
        tf.expand_dims(tf.reduce_sum(escore, 1), 1), [1, tf.shape(escore)[1]])
    score_weight = tf.divide(escore, sum_escore)

    similarities = tf.reduce_sum(mask * score * score_weight, 1)
    similarities = tf.expand_dims(similarities, axis=0)
    # shape: [time * batch_size]
    similarities = tf.reshape(
        tf.tile(similarities, [self._current_num_timesteps, 1]), [-1])

    # Apply an affine transform.
    similarities = similarities * self.affine_a + self.affine_b

    output_a = tf.reshape(tf.convert_to_tensor(self.affine_a), [1, 1])
    output_b = tf.reshape(tf.convert_to_tensor(self.affine_b), [1, 1])

    # shape: [time * batch]
    output_a = tf.reshape(
        tf.tile(output_a,
                [self._current_num_timesteps, self._current_batch_size]), [-1])
    output_b = tf.reshape(
        tf.tile(output_b,
                [self._current_num_timesteps, self._current_batch_size]), [-1])

    return common.AgentOutput(
        policy_logits=similarities, baseline=(output_a, output_b))


class DiscriminatorAgentV2(DiscriminatorAgent):
  """Discriminator Agent Version 2.

  This version encodes each modality (instruction and image) into a single
  vector before comparison, unlike in `DiscriminatorAgent` where each modality
  is encoded into a sequence of feature vectors.

  Major components, e.g., instruction encoder and image encoder, are shared with
  the parent class.
  """

  def __init__(self, config, mode=None, name='discriminator_v2'):
    """Initializes Discriminator Agent."""
    # Initializes self._instruction_encoder, self._image_encoder, decoder input
    # projection (optional), and the affine layer for similarity.
    super(DiscriminatorAgentV2, self).__init__(
        config, mode=mode, name=name)
    # If set, average image output from all steps, else only use the last step.
    self._average_image_states_of_all_steps = (
        config.average_image_states_of_all_steps if hasattr(
            config, 'average_image_states_of_all_steps') else False)
    # If set, then all states (i.e., [h, c]) from all layers are used as output.
    # Else, only use the last layer's h state.
    self._use_all_final_states = (
        config.use_all_final_states
        if hasattr(config, 'use_all_final_states') else False)

    # Projection layer for the output of instruction_encoder.
    self._instruction_feature_projection = tf.keras.layers.Dense(
        512, name='instruction_projection')

    # Projection layer for the output of image_encoder.
    self._image_feature_projection = tf.keras.layers.Dense(
        512, name='image_projection')

    # Affine transformation for batch loss. Offset is not needed as softmax is
    # shift invariant.
    self.similarity_scaler = tf.Variable(
        1.0, dtype=tf.float32, trainable=True, name='similarity_scaler')

  def _get_final_projection(self, projection_fn, final_states):
    """Gets the final projected feature for computing similarity."""
    if self._use_all_final_states:
      # Flatten all states.
      flat_state = tf.concat(list(itertools.chain(*final_states)), -1)
      result = projection_fn(flat_state)
    else:
      # Use only the h-state of the last layer.
      result = projection_fn(final_states[-1][0])
    return result

  def _get_initial_state(self, observation, batch_size):
    """Returns encoded instruction features and initial states for image enc."""
    text_token_ids = observation[constants.INS_TOKEN_IDS]
    enc_outputs, final_state = self._instruction_encoder(text_token_ids)
    # text_enc_outputs packs the sequence of text features and the final state.
    text_enc_outputs = (enc_outputs,
                        tf.nest.map_structure(tf.identity, final_state))

    if self._init_with_text_state:
      logging.info('Initializing image encoder with text states.')
      # Initilaize the image encoder with instruction states.
      if self._project_decoder_input_states:
        flat_state = tf.concat(list(itertools.chain(*final_state)), -1)
        start_state = tf.split(self._encoder_projection(flat_state), 4, -1)
        start_state = [(start_state[0], start_state[1]),
                       (start_state[2], start_state[3])]
      else:
        start_state = final_state
        return (final_state, text_enc_outputs)
    else:
      # Initialize image encoder from scratch.
      start_state = tf.nest.map_structure(tf.zeros_like, final_state)
      logging.info('Initializing image encoder from scratch.')
    return (start_state, text_enc_outputs)

  def _neck(self, torso_outputs, state):
    """Returns RNN state given one step of input observation and prev state."""
    current_lstm_state, text_enc_outputs = state
    # torso_outputs: dict.
    image_features = tf.cast(torso_outputs[constants.PANO_ENC], tf.float32)

    if self._embed_prev_action and self._embed_next_action:
      # Concat both prev and next action features.
      # Shape = [batch, 2 * feature_size]
      action_feature = tf.concat([
          torso_outputs['prev_action_feature'],
          torso_outputs['next_action_feature']
      ],
                                 axis=1)
      action_feature = tf.cast(action_feature, tf.float32)
      logging.info(
          'Concatenating prev and next action embeddings to pano embeddings.')
    elif self._embed_prev_action:
      # Shape = [batch_size, feature_size]
      action_feature = tf.cast(torso_outputs['prev_action_feature'], tf.float32)
      logging.info('Concatenating prev action embeddings to pano embeddings.')
    elif self._embed_next_action:
      # Shape = [batch_size, feature_size]
      action_feature = tf.cast(torso_outputs['next_action_feature'], tf.float32)
      logging.info('Concatenating next action embeddings to pano embeddings.')
    else:
      action_feature = None

    lstm_output, next_lstm_state = self._image_encoder(
        image_features, current_lstm_state, prev_action=action_feature)
    # <tf.float32>[batch_size, 1, hidden_dim]
    lstm_output = tf.expand_dims(lstm_output, axis=1)
    # The next_lstm_state are ListWrappers. In order to make it consistent with
    # get_initial_state, we convert them to tuple.
    result_state = []
    for one_state in next_lstm_state:
      result_state.append((one_state[0], one_state[1]))
    torso_outputs['visual_feature'] = lstm_output
    torso_outputs['visual_state'] = result_state
    torso_outputs['text_feature'] = text_enc_outputs[0]
    torso_outputs['text_state'] = text_enc_outputs[1]
    return (torso_outputs, (result_state, text_enc_outputs))

  def _head(self, env_output, neck_outputs):
    disc_mask = tf.reshape(
        neck_outputs[constants.DISC_MASK],
        [self._current_num_timesteps, self._current_batch_size])
    # Get first_true time step for text states as it's the same for all steps
    # in a path.
    # Shape = [time, batch] for both disc_mask and first_true
    first_true = utils.get_first_true_column(disc_mask)
    # Transpose to [batch, time] to ensure correct batch order for boolean_mask.
    first_true = tf.transpose(first_true, perm=[1, 0])

    # Transpose a list of n_lstm_layers (h, c) states to batch major.
    raw_text_state = tf.nest.map_structure(
        lambda t: tf.transpose(t, perm=[1, 0, 2]), neck_outputs['text_state'])
    tf.debugging.assert_equal(
        raw_text_state[0][0].shape,
        [self._current_batch_size, self._current_num_timesteps, 512])
    # Take the first step's text state since it's the same for all steps.
    # Selected state has shape [batch, hidden]
    text_state = self._select_by_mask(raw_text_state, first_true)

    # Projected shape: [batch, hidden_dim].
    text_feature = self._get_final_projection(
        self._instruction_feature_projection, text_state)

    # Get last_true mask for image states, i.e., state at end of sequence.
    # Shape = [time, batch] for both disc_mask and last_true
    last_true = utils.get_last_true_column(disc_mask)
    last_true = tf.transpose(last_true, perm=[1, 0])
    # Sanity check: ensure the first and last text states in a path are same.
    text_state_last_true = self._select_by_mask(raw_text_state, last_true)
    tf.debugging.assert_equal(text_state[-1][0], text_state_last_true[-1][0])

    # Transpose image states, a list of (h, c) states, into batch major. Each
    # state has shape [batch, time_step, hidden_dim]
    raw_image_state = tf.nest.map_structure(
        lambda t: tf.transpose(t, perm=[1, 0, 2]), neck_outputs['visual_state'])
    if self._average_image_states_of_all_steps:
      # Shape = [batch, time_step, 1]
      float_disc_mask = tf.expand_dims(
          tf.cast(tf.transpose(disc_mask), tf.float32), axis=2)
      # Shape of each reduced state: [batch, hidden_dim]
      image_state = tf.nest.map_structure(
          lambda x: tf.reduce_mean(x * float_disc_mask, 1), raw_image_state)
    else:
      # Selected state has shape [batch, hidden_dim].
      image_state = self._select_by_mask(raw_image_state, last_true)
    # Projected shape: [batch, hidden].
    visual_feature = self._get_final_projection(
        self._image_feature_projection, image_state)

    # Normalize features.
    visual_feature = tf.nn.l2_normalize(visual_feature, axis=-1)
    text_feature = tf.nn.l2_normalize(text_feature, axis=-1)

    # Select path_ids for current batch.
    # Transposed shape = [batch, time].
    raw_path_ids = tf.transpose(env_output.observation[constants.PATH_ID])
    # Shape = [batch].
    path_ids = self._select_by_mask(raw_path_ids, first_true)
    # Asserts first true and last true are referring to the same path.
    path_ids_last_true = self._select_by_mask(raw_path_ids, last_true)
    tf.debugging.assert_equal(path_ids, path_ids_last_true)

    # Shape = [time, batch]
    raw_labels = tf.cast(env_output.observation['label'], tf.float32)
    raw_labels = tf.transpose(raw_labels)
    # Shape = [batch]
    labels = self._select_by_mask(raw_labels, first_true)
    tf.debugging.assert_equal(labels,
                              self._select_by_mask(raw_labels, last_true))
    # Add time dimension as required by actor. Shape = [1, batch]
    labels = tf.expand_dims(labels, axis=0)

    # Shape: [batch, batch]
    similarity = tf.matmul(visual_feature,
                           tf.transpose(text_feature, perm=[1, 0]))
    # Add time dim as required by actor. Shape = [1, batch, batch]
    similarity = tf.expand_dims(similarity, axis=0)

    # Make similarity mask to exclude multiple positive matching labels
    diag_mask = tf.eye(self._current_batch_size, dtype=tf.bool)
    # path_id mask where matching col-row pairs are 1 except diagnal pairs.
    rows = tf.tile(
        tf.reshape(path_ids, [self._current_batch_size, 1]),
        [1, self._current_batch_size])
    cols = tf.tile(
        tf.reshape(path_ids, [1, self._current_batch_size]),
        [self._current_batch_size, 1])
    path_id_mask = tf.logical_and(
        tf.equal(rows, cols), tf.logical_not(diag_mask))
    # Filter the mask by label. Positive labels are 1.
    row_labels = tf.tile(
        tf.reshape(labels, [self._current_batch_size, 1]),
        [1, self._current_batch_size])
    col_labels = tf.tile(
        tf.reshape(labels, [1, self._current_batch_size]),
        [self._current_batch_size, 1])
    label_mask = tf.logical_and(tf.cast(row_labels, tf.bool),
                                tf.cast(col_labels, tf.bool))

    # M[i, j]=0 (i!=j) if path_id_mask[i,j] is True and label_mask[i, j] is True
    similarity_mask = tf.logical_not(tf.logical_and(path_id_mask, label_mask))
    # Add timestep dim as required by actor. Shape = [1, batch, batch]
    similarity_mask = tf.expand_dims(similarity_mask, axis=0)

    # Computes logits by transforming similarity from [-1, 1] to unbound.
    # Shape: [time, batch, batch]
    similarity_logits = self.similarity_scaler  * similarity

    output_logits = {'similarity': similarity_logits,
                     'similarity_mask': similarity_mask,
                     'labels': labels}

    # Logits for classification loss. Shape = [time, batch]
    classification_logits = (
        self.affine_a * tf.linalg.diag_part(similarity) + self.affine_b)

    return common.AgentOutput(policy_logits=output_logits,
                              baseline=classification_logits)

  @tf.function
  def call(self, env_output, initial_state):
    """Runs the entire episode given time-major tensors.

    Args:
      env_output: An `EnvOutput` tuple with following expectations:
        reward - Unused
        done - A boolean tensor of shape  [num_timesteps, batch_size].
        observation - A nested structure with individual tensors that have first
          two dimensions equal to [num_timesteps, batch_size]
        info - Unused
      initial_state: A tensor or nested structure with individual tensors that
        have first dimension equal to batch_size and no time dimension.

    Returns:
      An `AgentOutput` tuple with individual tensors of size [time_step,
      batch_size, ...].
      The neck state of the last time step of the sequence (not the last step of
        the path). Each tensor of the state has shape [batch_size, hidden_dim].
    """
    neck_output_list, neck_state = self._unroll_neck_steps(
        env_output, initial_state)
    # Tensor shapes: [time, batch, ...].
    neck_output_tensor = tf.nest.map_structure(
        lambda *tensors: tf.stack(tensors), *neck_output_list)

    head_output = self._head(env_output, neck_output_tensor)
    assert isinstance(head_output, common.AgentOutput)
    return head_output, neck_state

  def _select_by_mask(self, tensor_structure, mask):
    """Selects only the True elements in each tensor according to mask.

    Args:
      tensor_structure: scalar | tuple | dict | other nested structures, shape
        must be [batch, time_step, ...].
      mask: boolean tensor to be applied to tensor_structure. Shape must be:
        [batch, time_step]

    Returns:
      A mask filtered tensor of shape [None, ...]. Note that applying
        boolean_mask results in None shape in the 0th dim b/c its actual size
        depends on element values of the mask.
    """
    
    def _apply_mask(x):
      result = tf.cond(
          tf.math.reduce_any(mask),
          # Select True element along columns (time dimension).
          lambda: tf.boolean_mask(x, mask),
          # Take the 0th col if mask is all False
          lambda: tf.gather(x, 0, axis=1))
      return result
    selected = tf.nest.map_structure(_apply_mask, tensor_structure)
    return selected
