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


"""Tests for valan.framework.focal_loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf
from valan.framework.focal_loss import focal_loss_from_logits


class FocalLossTest(tf.test.TestCase):

  def test_focal_loss(self):
    alpha = 0.5
    gamma = 2.0
    logits = np.array([[-30.], [-50.], [30.0], [50.0]])
    labels = np.array([[0], [0], [1], [1]])
    loss, ce = focal_loss_from_logits(logits, labels, alpha, gamma)
    self.assertEqual(loss.shape, ce.shape)
    np.testing.assert_almost_equal(loss.numpy().mean(), 0.0)
    np.testing.assert_almost_equal(ce.numpy().mean(), 0.0)

    logits = np.array([[0.], [0.], [-0.0], [-0.0]])
    probs = 1.0 / (1.0 + np.exp(-logits))
    ce_expected = -labels * np.log(probs) - (1.0 - labels) * np.log(1. - probs)
    _, ce = focal_loss_from_logits(logits, labels, alpha, gamma)
    np.testing.assert_almost_equal(ce.numpy().mean(), ce_expected)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
