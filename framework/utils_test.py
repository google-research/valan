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

"""Tests for valan.framework.utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import numpy as np
import tensorflow.compat.v2 as tf
from valan.framework import utils

FLAGS = flags.FLAGS


class UtilsTest(tf.test.TestCase):

  def testStackNestedTensors(self):
    t1 = {
        'a': tf.constant([1., 2., 3.]),
        'b': tf.constant([[1., 1.], [2., 2.]])
    }
    t2 = {
        'a': tf.constant([4., 5., 6.]),
        'b': tf.constant([[3., 3.], [4., 4.]])
    }
    result = utils.stack_nested_tensors([t1, t2])
    np.testing.assert_array_almost_equal(
        np.array([[1., 2., 3.], [4., 5., 6.]]), result['a'].numpy())
    np.testing.assert_array_almost_equal(
        np.array([[[1., 1.], [2., 2.]], [[3., 3.], [4., 4.]]]),
        result['b'].numpy())

  def testTimeBatchDim(self):
    x = tf.ones(shape=(2, 3))
    y = tf.ones(shape=(2, 3, 4))

    x, y = utils.add_time_batch_dim(x, y)
    np.testing.assert_equal((1, 1, 2, 3), x.shape)
    np.testing.assert_equal((1, 1, 2, 3, 4), y.shape)

    x, y = utils.remove_time_batch_dim(x, y)
    np.testing.assert_equal((2, 3), x.shape)
    np.testing.assert_equal((2, 3, 4), y.shape)

  def testGetRowNestedTensor(self):
    x = {
        'a': tf.constant([[0., 0.], [1., 1.]]),
        'b': {
            'b_1': tf.ones(shape=(2, 3))
        }
    }
    result = utils.get_row_nested_tensor(x, 1)
    np.testing.assert_array_almost_equal(
        np.array([1., 1.]), result['a'].numpy())
    np.testing.assert_array_almost_equal(
        np.array([1., 1., 1.]), result['b']['b_1'].numpy())

  def testBatchApply(self):
    time_dim = 4
    batch_dim = 5
    inputs = {
        'a': tf.zeros(shape=(time_dim, batch_dim)),
        'b': {
            'b_1': tf.ones(shape=(time_dim, batch_dim, 9, 10)),
            'b_2': tf.ones(shape=(time_dim, batch_dim, 6)),
        }
    }

    def f(tensors):
      np.testing.assert_array_almost_equal(
          np.zeros(shape=(time_dim * batch_dim)), tensors['a'].numpy())
      np.testing.assert_array_almost_equal(
          np.ones(shape=(time_dim * batch_dim, 9, 10)),
          tensors['b']['b_1'].numpy())
      np.testing.assert_array_almost_equal(
          np.ones(shape=(time_dim * batch_dim, 6)), tensors['b']['b_2'].numpy())

      return tf.ones(shape=(time_dim * batch_dim, 2))

    result = utils.batch_apply(f, inputs)
    np.testing.assert_array_almost_equal(
        np.ones(shape=(time_dim, batch_dim, 2)), result.numpy())

  def testGatherFromDict(self):
    one_d_tensor_dict = {
        0: tf.ones(shape=(5)) * 0.,
        1: tf.ones(shape=(5)) * 1.,
        2: tf.ones(shape=(5)) * 2.,
        3: tf.ones(shape=(5)) * 3.,
    }
    choice = tf.constant([3, 0, 1, 1, 2])
    np.testing.assert_array_almost_equal(
        np.array([3., 0., 1., 1., 2.]),
        utils.gather_from_dict(one_d_tensor_dict, choice))

    choice = tf.constant([1, 1, 1, 1, 1])
    np.testing.assert_array_almost_equal(
        np.array([1., 1., 1., 1., 1.]),
        utils.gather_from_dict(one_d_tensor_dict, choice))

    one_d_tensor_dict = {
        'a': tf.ones(shape=(5)) * 0.,
        'b': tf.ones(shape=(5)) * 1.,
        'c': tf.ones(shape=(5)) * 2.,
        'd': tf.ones(shape=(5)) * 3.,
    }
    choice = tf.constant(['a', 'b', 'c', 'd', 'b'])
    np.testing.assert_array_almost_equal(
        np.array([0., 1., 2., 3., 1.]),
        utils.gather_from_dict(one_d_tensor_dict, choice))

    two_d_tensor_dict = {
        0: tf.ones(shape=(5, 2)) * 0.,
        1: tf.ones(shape=(5, 2)) * 1.,
        2: tf.ones(shape=(5, 2)) * 2.,
        3: tf.ones(shape=(5, 2)) * 3.,
    }
    choice = tf.constant([3, 0, 1, 1, 2])
    np.testing.assert_array_almost_equal(
        np.array([[3., 3.], [0., 0.], [1., 1.], [1., 1.], [2., 2.]]),
        utils.gather_from_dict(two_d_tensor_dict, choice))

  def testReadWriteSpecs(self):
    logdir = FLAGS.test_tmpdir
    specs = {
        'a': tf.TensorSpec(shape=(2, 3), dtype=tf.float32),
        'b': {
            'b_1': tf.TensorSpec(shape=(5,), dtype=tf.string),
            'b_2': tf.TensorSpec(shape=(5, 6), dtype=tf.int32),
        }
    }
    utils.write_specs(logdir, specs)
    # Now read and verify
    specs_read = utils.read_specs(logdir)

    def _check_equal(sp1, sp2):
      self.assertEqual(sp1, sp2)

    tf.nest.map_structure(_check_equal, specs, specs_read)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
