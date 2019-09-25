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

"""Tests for valan.r2r.glove_encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from valan.r2r import instruction_encoder


class InstructionEncoderTest(tf.test.TestCase):

  def test_call_r2r(self):
    encoder = instruction_encoder.InstructionEncoder(
        num_hidden_layers=2,
        output_dim=256,
        pretrained_embed_path='',
        oov_bucket_size=1)
    # Initialize tokens
    tokens = tf.constant([[3, 4, 5, 1, 6, 0]])
    result = encoder(tokens)
    self.assertEqual(result[0].shape, [1, 6, 512])
    self.assertEqual(result[1][0][0].shape, [1, 512])
    self.assertEqual(result[1][0][1].shape, [1, 512])
    self.assertEqual(result[1][1][0].shape, [1, 512])
    self.assertEqual(result[1][1][1].shape, [1, 512])

  def test_call_ndh(self):
    encoder = instruction_encoder.InstructionEncoder(
        num_hidden_layers=2,
        output_dim=256,
        pretrained_embed_path=None,
        oov_bucket_size=1,
        vocab_size=1082,
        word_embed_dim=300)

    tokens = tf.constant([[3, 4, 5, 1, 6, 0]])
    result = encoder(tokens)
    self.assertEqual(result[0].shape, [1, 6, 512])
    self.assertEqual(result[1][0][0].shape, [1, 512])
    self.assertEqual(result[1][0][1].shape, [1, 512])
    self.assertEqual(result[1][1][0].shape, [1, 512])
    self.assertEqual(result[1][1][1].shape, [1, 512])


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
