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


"""Focal loss for imbalanced labels.

C.F.: https://arxiv.org/abs/1708.02002
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import tensorflow.compat.v2 as tf


def focal_loss_from_logits(logits, class_labels, alpha, gamma, normalizer=1.0):
  """Compute focal loss for given `logits` and ground truth labels.

  Focal loss = (1 - p_t)^r * Cross_Entropy * alpha / normalizer
  where p_t is the prediction probability for each class (0 AND 1), alpha is
  the weighting factor, gamma (r) is the modulator power, and the normalizer
  scales the loss uniformly.

  For positive class, the modulator is:
    (1 - p_t)^r = (1 - sigmoid(X))^r = = exp(- r * X - r * log(1 + exp(-X)));
  for negative class:
    (1 - p_t)^r = exp(-r * log(1 + exp(-X)))
  Thus the general form is:
    (1 - p_t)^r = exp(-r * X * Y - r * log(1 + exp(-X))),
  where -X is the negative logits and Y is the numerical GT class labels (0/1).

  Args:
    logits: A float tensor of same size as `class_labels`.
    class_labels: A int tensor of the same size as logits for binary labels.
    alpha: A float scalar scaling factor for the positive class; for the
      negative class the factor is (1- alpha).
    gamma: A float scalar modulating factor.
    normalizer: A float scalar normalizing factor. normalizer should be > 0.

  Returns:
    focal loss: a tensor of per example focal loss, of the same shape as logits.
    cross_entropy loss: a tensor of same shape as logits that can be used for
      log_pplex.
  """
  tf.debugging.assert_equal(tf.shape(class_labels), tf.shape(logits))
  labels = tf.cast(class_labels, tf.float32)
  logits = tf.cast(logits, tf.float32)
  cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                          logits=logits)

  neg_logits = -1.0 * logits

  modulator = tf.exp(gamma * labels * neg_logits -
                     gamma * tf.math.log1p(tf.exp(neg_logits)))
  loss = modulator * cross_entropy
  alpha_weighted_loss = tf.where(tf.equal(labels, 1.0), alpha * loss,
                                 (1.0 - alpha) * loss)
  if normalizer <= 0.0:
    raise ValueError('Normalizer in focal_loss must be >= 0.0')
  return alpha_weighted_loss / normalizer, cross_entropy
