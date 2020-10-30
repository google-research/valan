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


"""Tests for valan.framework.loss_fns."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow.compat.v2 as tf
from valan.framework import common
from valan.framework import hyperparam_flags  
from valan.framework import loss_fns
from valan.framework.focal_loss import focal_loss_from_logits

FLAGS = flags.FLAGS


class LossFnsTest(tf.test.TestCase):

  def setUp(self):
    super(LossFnsTest, self).setUp()
    # Shape = [batch, batch].
    self._similarity_logits = tf.constant(
        [[-0.5, -0.3, 0.8], [-0.3, 0.4, 0.7], [0.8, 0.7, 1.0]],
        dtype=tf.float32)
    # Shape = [batch, batch].
    self._similarity_mask = tf.constant([[True, True, False],
                                         [True, True, True],
                                         [False, True, True]])
    # Shape = [batch].
    self._labels = tf.constant([0., 1.0, 1.0], dtype=tf.float32)
    # Shape = [batch].
    self._baseline_logits = tf.constant([-0.4, -.1, 0.9], dtype=tf.float32)
    output_logits = {
        'similarity': self._similarity_logits,
        'similarity_mask': self._similarity_mask,
        'labels': self._labels
    }
    agent_output = common.AgentOutput(
        policy_logits=output_logits, baseline=self._baseline_logits)
    # Add time dim as required by actor that shape must be [time, batch, ...]
    self._agent_output = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                               agent_output)

  def test_get_discriminator_batch_loss(self):
    batch_size = self._labels.shape[0]
    loss = loss_fns.get_discriminator_batch_loss(
        self._agent_output, None, None, None, None, None, None, None,
        num_steps=0)
    self.assertEqual(loss.shape, [batch_size])

    # Expected batch loss.
    diag_logits = tf.linalg.diag_part(self._similarity_logits)
    self.assertAllClose(diag_logits, [-0.5, 0.4, 1.0])
    row_softmax = tf.exp(diag_logits) / tf.reduce_sum(
        tf.exp(self._similarity_logits) *
        tf.cast(self._similarity_mask, tf.float32),
        axis=1)
    row_loss = -tf.math.log(row_softmax)
    col_softmax = tf.exp(diag_logits) / tf.reduce_sum(
        tf.exp(self._similarity_logits) *
        tf.cast(self._similarity_mask, tf.float32),
        axis=0)
    col_loss = -tf.math.log(col_softmax)
    expected_batch_loss = (row_loss + col_loss) / 2.0 * tf.squeeze(
        self._agent_output.policy_logits['labels'], 0)
    # Expected cross entropy focal loss.
    fl, _ = focal_loss_from_logits(
        self._baseline_logits,
        self._labels,
        alpha=FLAGS.focal_loss_alpha,
        gamma=FLAGS.focal_loss_gamma, normalizer=FLAGS.focal_loss_normalizer)
    expected_total_loss = expected_batch_loss + fl
    self.assertEqual(expected_batch_loss[0], 0.0)
    self.assertAllClose(expected_total_loss, loss)
    self.assertEqual(self._agent_output.baseline.shape, [1, 3])


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
