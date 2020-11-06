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

"""4-Layer lingunet with input and output spatial dimensions at 3x18 and 1x18.

The intended use-case for this LingUNet implementation is to map a 3x18
panoramic StreetView feature map to a 1x18 output.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function


import tensorflow.compat.v2 as tf
from valan.framework import utils

assert_shape = utils.assert_shape



class LingUNet4(tf.keras.Model):
  """Implementation of a 4-layer LingUNet model."""

  def __init__(self,
               in_channels,
               out_channels,
               embedding_size,
               conv_channels,
               filter_channels):
    super(LingUNet4, self).__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.embedding_size = embedding_size
    self.conv_channels = conv_channels
    self.filter_channels = filter_channels

    act = tf.keras.layers.LeakyReLU(0.01)
    self.conv1 = tf.keras.layers.Conv2D(
        filters=conv_channels, kernel_size=(3, 3), strides=(1, 2),
        padding="valid", use_bias=True, activation=act)
    self.conv2 = tf.keras.layers.Conv2D(
        filters=conv_channels, kernel_size=(1, 3), strides=(1, 2),
        padding="valid", use_bias=True, activation=act)
    self.conv3 = tf.keras.layers.Conv2D(
        filters=conv_channels, kernel_size=(1, 3), strides=(1, 2),
        padding="valid", use_bias=True, activation=act)
    self.conv4 = tf.keras.layers.Conv2D(
        filters=conv_channels, kernel_size=(1, 3), strides=(1, 2),
        padding="valid", use_bias=True, activation=act)

    self.deconv4 = tf.keras.layers.Convolution2DTranspose(
        filters=conv_channels, kernel_size=(1, 3), strides=(1, 2),
        padding="valid", use_bias=True, activation=act)
    self.deconv3 = tf.keras.layers.Convolution2DTranspose(
        filters=conv_channels, kernel_size=(1, 3), strides=(1, 2),
        padding="valid", use_bias=True, activation=act)
    self.deconv2 = tf.keras.layers.Convolution2DTranspose(
        filters=conv_channels, kernel_size=(1, 3), strides=(1, 2),
        padding="valid", use_bias=True, activation=act)
    self.deconv1 = tf.keras.layers.Convolution2DTranspose(
        filters=out_channels, kernel_size=(1, 3), strides=(1, 2),
        padding="valid", use_bias=True, activation=act)

    self.dense1 = tf.keras.layers.Dense(conv_channels * filter_channels)
    self.dense2 = tf.keras.layers.Dense(conv_channels * filter_channels)
    self.dense3 = tf.keras.layers.Dense(conv_channels * filter_channels)
    self.dense4 = tf.keras.layers.Dense(conv_channels * filter_channels)


    self.norm1 = tf.keras.layers.LayerNormalization(axis=(1, 2, 3))
    self.norm2 = tf.keras.layers.LayerNormalization(axis=(1, 2, 3))
    self.norm3 = tf.keras.layers.LayerNormalization(axis=(1, 2, 3))
    self.norm4 = tf.keras.layers.LayerNormalization(axis=(1, 2, 3))

    self.denorm2 = tf.keras.layers.LayerNormalization(axis=(1, 2, 3))
    self.denorm1 = tf.keras.layers.LayerNormalization(axis=(1, 2, 3))

  def __call__(self, input_image, text_embed, training=False):
    batch_size = text_embed.shape[0]

    assert_shape((batch_size, 3, 18, self.in_channels), input_image.shape)
    assert_shape((batch_size, self.embedding_size), text_embed.shape)

    # Compute 1x1 convolution weights
    # 1 is channel (embedding dim) axis
    kern1 = tf.math.l2_normalize(self.dense1(text_embed), axis=1)
    kern2 = tf.math.l2_normalize(self.dense2(text_embed), axis=1)
    kern3 = tf.math.l2_normalize(self.dense3(text_embed), axis=1)
    kern4 = tf.math.l2_normalize(self.dense4(text_embed), axis=1)
    kern1 = tf.reshape(kern1, (
        batch_size, 1, 1, self.conv_channels, self.filter_channels))
    kern2 = tf.reshape(kern2, (
        batch_size, 1, 1, self.conv_channels, self.filter_channels))
    kern3 = tf.reshape(kern3, (
        batch_size, 1, 1, self.conv_channels, self.filter_channels))
    kern4 = tf.reshape(kern4, (
        batch_size, 1, 1, self.conv_channels, self.filter_channels))

    # Perform series of convolutions to encode image features
    # x0 = tf.pad(input_image, [[0,0],[0,0],[1,1],[0,0]], mode='REFLECT')
    x0 = utils.circular_pad(input_image, axis=2, padding=[1, 1])
    assert_shape((batch_size, 3, 20, self.in_channels), x0.shape)

    h1 = self.norm1(self.conv1(x0))
    assert_shape((batch_size, 1, 9, self.conv_channels), h1.shape)
    x1 = utils.circular_pad(h1, axis=2, padding=[1, 1])
    assert_shape((batch_size, 1, 11, self.conv_channels), x1.shape)

    h2 = self.norm2(self.conv2(x1))
    assert_shape((batch_size, 1, 5, self.conv_channels), h2.shape)
    x2 = utils.circular_pad(h2, axis=2, padding=[1, 1])
    assert_shape((batch_size, 1, 7, self.conv_channels), x2.shape)

    h3 = self.norm3(self.conv3(x2))
    assert_shape((batch_size, 1, 3, self.conv_channels), h3.shape)
    x3 = h3  # Being explicit

    h4 = self.norm4(self.conv4(x3))
    assert_shape((batch_size, 1, 1, self.conv_channels), h4.shape)

    # Filter encoded image features to produce language-conditioned features
    g1 = utils.parallel_conv2d(h1, kern1, 1, "SAME")
    assert_shape((batch_size, 1, 9, self.filter_channels), g1.shape)
    g2 = utils.parallel_conv2d(h2, kern2, 1, "SAME")
    assert_shape((batch_size, 1, 5, self.filter_channels), g2.shape)
    g3 = utils.parallel_conv2d(h3, kern3, 1, "SAME")
    assert_shape((batch_size, 1, 3, self.filter_channels), g3.shape)
    g4 = utils.parallel_conv2d(h4, kern4, 1, "SAME")
    assert_shape((batch_size, 1, 1, self.filter_channels), g4.shape)

    # Deconvolutions shoud look like this.
    # Example deconvolving from width=3 to witdh=5
    #         X X X
    #        /  |  \      <-. Transposed Conv with Stride=2, Padding=VALID
    #      /|\ /|\ /|\    <`
    #     X X X X X X X
    #     - | | | | | -   <- Cut off the margin that doesn't correspond to
    #       X X X X X        kernel centers. Recover input dimension.
    #
    # Perform deconvolutions to reconstruct the output
    gd4 = g4  # Being explicit
    d3 = self.deconv4(gd4)
    assert_shape((batch_size, 1, 3, self.conv_channels), d3.shape)
    gd3 = tf.concat([d3, g3], axis=3)  # 0-batch, 1-height, 2-width, 3-channel

    d2 = self.denorm2(self.deconv3(gd3))
    d2 = d2[:, :, 1:-1, :]  # Crop off the margin
    assert_shape((batch_size, 1, 5, self.conv_channels), d2.shape)
    gd2 = tf.concat([d2, g2], axis=3)

    d1 = self.denorm1(self.deconv2(gd2))
    d1 = d1[:, :, 1:-1, :]  # Crop off the margin
    assert_shape((batch_size, 1, 9, self.conv_channels), d1.shape)
    gd1 = tf.concat([d1, g1], axis=3)

    d0 = self.deconv1(gd1)
    out = d0[:, :, 0:-1, :]  # Crop off the right margin only
    assert_shape((batch_size, 1, 18, self.out_channels), out.shape)
    return out
