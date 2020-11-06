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

"""2-layer LingUNet model customized for Touchdown SDR task."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function


import tensorflow.compat.v2 as tf
from valan.framework import utils


class BidirectionalRNN(tf.keras.Model):
  """Bidirectional RNN with concatenated/reduced outoputs."""

  def __init__(self, hidden_size=600, dropout=0.5, reduce_states=True):
    super(BidirectionalRNN, self).__init__()
    self.reduce_states = reduce_states
    self.forward_rnn = tf.keras.layers.LSTM(
        hidden_size,
        dropout=dropout,
        return_sequences=True)
    self.backward_rnn = tf.keras.layers.LSTM(
        hidden_size,
        dropout=dropout,
        return_sequences=True)

  def call(self, inputs, lengths, training):
    seq_mask = tf.sequence_mask(
        lengths, inputs.shape[1], dtype=tf.dtypes.float32)
    forward_outputs = self.forward_rnn(inputs, training=training)
    reversed_inputs = tf.reverse_sequence(inputs, lengths, seq_axis=1)
    backward_outputs = self.backward_rnn(reversed_inputs, training=training)
    backward_outputs = tf.reverse_sequence(
        backward_outputs, lengths, seq_axis=1)
    outputs = tf.concat([forward_outputs, backward_outputs], axis=-1)
    outputs = outputs * tf.expand_dims(seq_mask, -1)

    if self.reduce_states:
      outputs = tf.reduce_mean(outputs, axis=1)

    return outputs


class LingUNet(tf.keras.Model):
  """2-layer LingUNet model for Touchdown SDR task.

  This assumes dimensions for panos features to be: 128x100x464 (464 = 8*58)

  This should be obtained by slicing each pano into eight images, and projecting
  each image from a equirectangular projection to a perspective projection.

  Each of the eight images is of size 800x460 -> passed to ResNet18 -> 4th last
  layer features of size 128x100x58 -> concat 8 ways -> 128x100x464.
  """

  def __init__(self,
               vocab_size=30000,  # BERT vocab is ~27k
               num_channels=128,
               embedding_size=300,
               max_num_tokens=128):
    super(LingUNet, self).__init__(name="LingUNet")

    self.num_channels = num_channels
    self.text_embedder = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        input_length=max_num_tokens,
        mask_zero=True)

    self.rnn = BidirectionalRNN()

    self.num_text_channels = 1 * 1 * num_channels * num_channels
    self.dense_k1 = tf.keras.layers.Dense(self.num_text_channels)
    self.dense_k2 = tf.keras.layers.Dense(self.num_text_channels)

    act = tf.keras.layers.ReLU()
    self.conv1 = tf.keras.layers.Conv2D(
        filters=num_channels, kernel_size=(5, 5), strides=1,
        padding="same", use_bias=True, activation=act)
    self.conv2 = tf.keras.layers.Conv2D(
        filters=num_channels, kernel_size=(5, 5), strides=1,
        padding="same", use_bias=True, activation=act)

    self.deconv1 = tf.keras.layers.Convolution2DTranspose(
        filters=num_channels, kernel_size=(5, 5), strides=1,
        padding="same", use_bias=True, activation=act)
    self.deconv2 = tf.keras.layers.Convolution2DTranspose(
        filters=num_channels, kernel_size=(5, 5), strides=1,
        padding="same", use_bias=True, activation=act)

    self.dense1 = tf.keras.layers.Dense(
        self.num_channels, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(
        self.num_channels, activation=tf.nn.relu)
    self.out_dense = tf.keras.layers.Dense(
        1, use_bias=False, activation=tf.keras.activations.linear)

  def call(self,
           image_embed,
           instructions,
           instruction_lengths,
           training=False):
    assert self.num_channels == image_embed.shape[3]

    text_embed = self.text_embedder(instructions)
    text_embed = self.rnn(text_embed, instruction_lengths, training)
    text_embed_1, text_embed_2 = tf.split(text_embed, 2, axis=-1)
    batch_size = text_embed.shape[0]

    # Compute 1x1 convolution weights
    kern1 = self.dense_k1(text_embed_1)
    kern2 = self.dense_k2(text_embed_2)
    kern1 = tf.reshape(kern1, (
        batch_size, 1, 1, self.num_channels, self.num_channels))
    kern2 = tf.reshape(kern2, (
        batch_size, 1, 1, self.num_channels, self.num_channels))

    f0 = image_embed
    f1 = self.conv1(f0)
    f2 = self.conv2(f1)

    # Filter encoded image features to produce language-conditioned features
    #

    g1 = utils.parallel_conv2d(f1, kern1, 1, "SAME")
    g2 = utils.parallel_conv2d(f2, kern2, 1, "SAME")

    h2 = self.deconv2(g2)
    h2_g1 = tf.concat([h2, g1], axis=3)  # Assuming NHWC

    h1 = self.deconv1(h2_g1)

    out1 = self.dense1(h1)
    out2 = self.dense2(out1)
    out = tf.squeeze(self.out_dense(out2), -1)

    out_flat = tf.reshape(out, [batch_size, -1])
    # So that the output forms a prob distribution.
    out_flat = tf.nn.softmax(out_flat)
    return out_flat
