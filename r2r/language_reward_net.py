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

"""Language conditioned reward network.

This is a model that takes in the natural language instruction and an action
(in our case, an action is represented as a pair of panoramas) and produces a
reward. This model is to be called during the RL phase for each action that the
agent takes to give a corresponding reward.

This model is to be pretrained on (language, action) samples from R2R and
possibly other datasets.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
from absl import logging
import numpy as np

import tensorflow.compat.v2 as tf

from valan.framework import image_features_pb2

from valan.r2r import general_utils
from valan.r2r import image_encoder
from valan.r2r import instruction_encoder

FLAGS = flags.FLAGS


class LanguageRewardNet(tf.keras.Model):
  """Classify (language, action) pairs as related or unrelated.

  Main components are an instruction encoder, an action encoder (here a sequence
  of consecutive panos is an action), an MLP that transform the
  concatenation of (language, action) to smaller dimensions, and a classifier
  that produces a distribution over 2 classes corresponding to whether the
  language and action pairs are related or unrelated. The text and visual
  encoders (with attention) are the same as in the VLN model, which should
  allow better transfer with the actual VLN model at RL time when we need to get
  a reward value.
  """

  def __init__(self, hidden_dim, n_classes, mode='train'):
    super(LanguageRewardNet, self).__init__()
    self.config = self._get_default_config()
    # Text encoder is an LSTM over GLoVe with attention
    self._instruction_encoder = self._get_instruction_encoder()
    # Image encoder is an LSTM over Starburst with attention
    self._image_encoder = self._get_image_encoder()
    # Classifier is an MLP (2 hidden layers with ReLU activation) for 2 classes
    self._mlp_layer = self._get_mlp_layer(hidden_dim, n_classes)
    # Text attention.
    self._text_attention_size = 512
    self._text_attention_project_hidden = tf.keras.layers.Dense(
        self._text_attention_size)
    self._text_attention_project_text = tf.keras.layers.Dense(
        self._text_attention_size)
    self._text_attention = tf.keras.layers.Attention(use_scale=True)
    # Visual attention.
    self._visual_attention_size = 256
    self._visual_attention_project_ctext = tf.keras.layers.Dense(
        self._visual_attention_size)
    self._visual_attention_project_feature = tf.keras.layers.Dense(
        self._visual_attention_size)
    self._visual_attention = tf.keras.layers.Attention(use_scale=True)
    if mode == 'predict':
      restore_from_ckpt(self.config['ckpt_dirpath'])

  def _get_default_config(self):
    """Information needed to pick up ckpt and produce reward for a sample."""
    return {
        'vocab_file':
            'trainval_vocab.txt',
        'base_dir':
            '',
        'instruction_length':
            52,
        'pretrained_embed_path':
            '',
        'oov_bucket_size':
            1,
        'pano_dir':
            '',
        'image_path':
            '',
        'ckpt_dirpath':
            '',
    }

  def _get_mlp_layer(self, hidden_dim, n_classes):
    """Get MLP layer to produce a distribution over n classes.

    Args:
      hidden_dim: Dimension of first layer that takes in (instruction, action)
        sequence
      n_classes: Dimension of second layer that produces a distribution over 2
        classes (RELATED, UNRELATED)

    Returns:
      layers: A tf.keras.Sequential layer corresponding to input dimensions plus
      ReLU activation.
    """
    layers = tf.keras.Sequential()
    layers.add(tf.keras.layers.Dense(hidden_dim))
    layers.add(tf.keras.layers.ReLU())
    layers.add(tf.keras.layers.Dense(n_classes))
    layers.add(tf.keras.layers.Softmax())
    return layers

  def _get_instruction_encoder(self):
    """Return instruction encoder module."""
    return instruction_encoder.InstructionEncoder(
        num_hidden_layers=2,
        output_dim=256,
        pretrained_embed_path=self.config['pretrained_embed_path'],
        oov_bucket_size=self.config['oov_bucket_size'])

  def _get_image_encoder(self):
    """Return image encoder module."""
    return image_encoder.ImageEncoder(
        attention_space_size=256,
        num_lstm_units=512,
        num_hidden_layers=2)

  def _get_image_features(self, filename):
    """Get image features by loading from the filename."""
    protos = []
    # Pretend to read more than 1 values and then verify there is exactly 1.
    for record in tf.data.TFRecordDataset([filename]).take(2):
      protos.append(record.numpy())
    assert len(protos) == 1
    parsed_record = image_features_pb2.ImageFeatures()
    parsed_record.ParseFromString(protos[0])
    return parsed_record

  def _load_vocab(self, vocab_file):
    """Loads a vocabulary file into a dictionary.

    Args:
      vocab_file: File name to read R2R vocabulary.

    Returns:
      vocab: R2R vocabulary dictionary object.
    """
    with tf.io.gfile.GFile(vocab_file) as f:
      tokens = f.readlines()
    tokens = [token.strip() for token in tokens]
    vocab = {k: idx for idx, k in enumerate(tokens)}
    logging.info('Read vocabulary with %d tokens.', len(vocab))
    return vocab

  def encode_instruction(self, instruction_token_ids):
    """Takes in a batch of token ids and uses a BiLSTM to get an encoding.

    Args:
      instruction_token_ids: A tensor of type int64 for token ids.

    Returns:
      A tuple outputs: <encoding, state>.
      encoding: a tf.float32 tensor with shape [batch_size, max_seq_length,
        output_dim].
      state: A list which has num_hidden_layer elements. Every elements is
        a (state_c, state_h) tuple. This is concatenated last forward and
        backward states of each LSTM layer.
    """
    outputs = self._instruction_encoder(instruction_token_ids)
    # text_encoding, final_state = outputs
    return outputs

  def encode_action(self, current_state, action_panos):
    """Takes in action representations and encodes with a BiLSTM.

    The current state is the state passed by the text encoder.

    Args:
      current_state: A list of (state_c, state_h) tuples.
      action_panos: A tensor of type int64 for the previous pano ids.

    Returns:
      A tuple: <encoding, state>
      next_hidden_state: Hidden state vector [batch_size, lstm_space_size],
        current steps's LSTM output.
      next_lstm_state: Same shape as current_lstm_state.
    """
    outputs = self._image_encoder(action_panos, current_state)
    # action_encoding, next_state = outputs
    return outputs

  def encode_pano(self, pano_name, pano_path, num_views):
    """Takes in a pano name and returns image features.

    Args:
      pano_name: String specifying the file name of the pano.
      pano_path: Path to directory that contains panoramic images.
      num_views: Number of viewpoints.

    Returns:
      image_features: Image features for the input panoramic image.
    """
    image_features = self._get_image_features(
        os.path.join(pano_path, '{}_viewpoints_proto'.format(pano_name)))

    return np.reshape(image_features.value, (1, num_views, -1))

  def predict_class(self, text_token_ids, action_panos):
    """Takes in an instruction and action and returns classifier outputs.

    Args:
      text_token_ids: Tensor of token indices for the input instruction.
      action_panos: Tensor of concatenated image panoramas.

    Returns:
      (class_outputs, predictions): Output of last layer of MLP and prediction.
    """
    text_enc_outputs, current_lstm_state = self.encode_instruction(
        text_token_ids)
    lstm_output, next_lstm_state = self.encode_action(current_lstm_state,
                                                      action_panos)
    lstm_output = tf.expand_dims(lstm_output, axis=1)
    batch_size = text_enc_outputs.shape[0]

    # c_text has shape [batch_size, 1, self._text_attention_size]
    c_text = self._text_attention([
        self._text_attention_project_hidden(lstm_output),
        self._text_attention_project_text(text_enc_outputs)
    ])
    # convert ListWrapper output of next_lstm_state to tuples
    result_state = []
    for one_state in next_lstm_state:
      result_state.append((one_state[0], one_state[1]))

    hidden_state = lstm_output
    c_visual = self._visual_attention([
        self._visual_attention_project_ctext(c_text),
        self._visual_attention_project_feature(action_panos),
    ])

    input_feature = tf.concat([hidden_state, c_text, c_visual], axis=2)
    class_outputs = self._mlp_layer(input_feature)
    class_outputs = tf.reshape(class_outputs, (batch_size, 2))
    predictions = tf.argmax(class_outputs, axis=-1)
    return (class_outputs, predictions)

  def loss_function(self, true_labels, logits):
    """Takes in logits and true labels and makes a prediction.

    Args:
      true_labels: a tensor of shape [batch_size, n_classes]
      logits: output of MLP layer, a tensor of shape [batch_size, n_classes].

    Returns:
      loss: a scalar cross entropy loss.
    """
    loss = tf.keras.losses.categorical_crossentropy(true_labels, logits)
    return loss

  def accuracy(self, true_labels, predictions):
    """Takes in predictions and true labels and returns accuracy.

    Args:
      true_labels: a tensor of shape [batch_size, n_classes]
      predictions: a tensor of shape [batch_size, 1].

    Returns:
      loss: a scalar cross entropy loss.
    """
    true_labels = tf.keras.backend.argmax(true_labels)
    metric = tf.keras.metrics.Accuracy()
    accuracy = metric.update_state(true_labels, predictions)
    return accuracy

  def produce_reward(self, instruction, action_panos):
    """Takes in concatenated encoding, transforms and produces reward.

    Args:
      instruction: A tensor of type int64 for token ids.
      action_panos: A np array for the action taken (concatenated panoramas).

    Returns:
      A scalar reward value.
    """
    # take in instruction instead of token ids
    vocab_path = os.path.join(self.config['base_dir'],
                              self.config['vocab_file'])
    fixed_instruction_len = self.config['instruction_length']
    instr_token_ids, _ = self.instruction_to_token_ids(instruction,
                                                       fixed_instruction_len,
                                                       vocab_path)
    action_panos = tf.convert_to_tensor(action_panos, dtype=tf.float32)

    mlp_outputs = self.predict_class(instr_token_ids, action_panos)
    return mlp_outputs[0]

  def get_optimizer(self, learning_rate):
    """Get optimizer for the model.

    When picking up a pretrained model, the optimizer and model class need to
    be passed in.

    Args:
      learning_rate: Learning rate for the model.

    Returns:
      optimizer: a tf.keras optimizer
    """
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)

  def instruction_to_token_ids(self, instruction, fixed_instruction_len,
                               vocab_path):
    """Converts natural language instruction to indexed token ids.

    Can be used to test arbitrary instructions (string of words) to see the
    reward value given, by calling private load_vocab and tokeniser functions.

    Args:
      instruction: Natural language instruction (String).
      fixed_instruction_len: Max number of tokens (integer).
      vocab_path: Path to vocab file (String).

    Returns:
      instruction_token_ids: Tensor of type int64.
      instruction_len: Number of tokens (integer).
    """
    vocab = self._load_vocab(vocab_path)
    token_ids, instruction_len = general_utils.get_token_ids(
        instruction, fixed_instruction_len, vocab)
    instruction_token_ids = tf.convert_to_tensor(
        [token_ids], dtype=tf.float32)
    return instruction_token_ids, instruction_len


# Library functions to restore from and save checkpoints
def restore_from_ckpt(ckpt_dir, **kwargs):
  """Sets up checkpointing and restore checkpoint if available."""
  checkpoint_prefix = os.path.join(ckpt_dir, 'model.ckpt')
  ckpt = tf.train.Checkpoint(**kwargs)

  manager = tf.train.CheckpointManager(ckpt, checkpoint_prefix, max_to_keep=5,
                                       keep_checkpoint_every_n_hours=6)
  if manager.latest_checkpoint:
    logging.info('Restoring from checkpoint: %s', manager.latest_checkpoint)
    ckpt.restore(manager.latest_checkpoint)
  return manager


def save_model_checkpoint(self, model, ckpt_dirpath, optimizer, iterations):
  """Save model checkpoint to directory."""
  ckpt_manager = self.restore_from_ckpt(ckpt_dirpath, model=model,
                                        optimizer=optimizer)
  ckpt_manager.save(checkpoint_number=iterations)
