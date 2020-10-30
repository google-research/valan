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

  def testValanScoreAggregator(self):
    aggregator = utils.NavigationScorePickleAggregator()
    test_pickle_dir = ('third_party/py/valan/framework/test_data/'
                       'valan_score_data')
    score_dict, avg_scores = aggregator.get_aggregated_scores([test_pickle_dir])

    expected_score_dict = {
        'sr': [
            1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0
        ],
        'spl': [
            1.0, 0.0, 0.0, 1.0, 0.9274571878218345, 0.8525430083025657, 0.0,
            0.5164113132891832, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.8049224140850629, 0.0, 0.9347018491822641, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.7384204595770543,
            0.8701826558907835, 0.5587272563891355, 0.0, 0.0, 0.0, 0.0, 1.0,
            0.9858843639849288, 0.9471505692881894, 0.0, 0.0, 0.787739595665917,
            0.0, 0.5769936704765138, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0
        ],
        'sdtw': [
            0.8143809465707129, 0.0, 0.0, 0.34652914566434756,
            0.9307189274260722, 0.851673534723788, 0.0, 0.5454058755221346, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.8659635431629552, 0.0,
            0.9267488153774507, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8146770423283999,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.6745882226815845,
            0.6795208279601076, 0.767625043353138, 0.0, 0.0, 0.0, 0.0,
            0.8491946181966382, 0.9110337934666316, 0.6311807335468579, 0.0,
            0.0, 0.8287387929048695, 0.0, 0.764382484639884, 1.0, 0.0,
            0.8974247542540588, 0.0, 0.0, 0.9012774429521982, 0.0,
            0.7593372024970849, 0.0
        ],
        'ndtw': [
            0.8143809465707129, 0.2272123998209866, 0.19634187396371375,
            0.34652914566434756, 0.9307189274260722, 0.851673534723788,
            0.1768055533114191, 0.5454058755221346, 0.21729893225938526,
            0.5345611720432003, 0.5140702173740033, 0.24137453801931877,
            0.7450253749378951, 0.37844997556397114, 0.8659635431629552,
            0.3165631741998121, 0.9267488153774507, 0.1753459346878747,
            0.45050064578300264, 0.6174358348430163, 0.03869477332934748,
            0.6176465056501452, 0.8146770423283999, 0.14278273954879245,
            0.06473660102360958, 0.6390397032356736, 0.2827823118148731,
            0.6942450994784002, 1.0, 0.6745882226815845, 0.6795208279601076,
            0.767625043353138, 0.6056022017119623, 0.6353562169498947,
            0.5923411059166225, 0.12805681586813164, 0.8491946181966382,
            0.9110337934666316, 0.6311807335468579, 0.5328702568049911,
            0.18237619915120534, 0.8287387929048695, 0.548872206628709,
            0.764382484639884, 1.0, 0.27355855948686475, 0.8974247542540588,
            0.5695903592773883, 0.29429130228237876, 0.9012774429521982,
            0.26888623480054047, 0.7593372024970849, 0.15872607466984578
        ]
    }

    expected_avg_scores = {
        'sr': 0.39622641509433965,
        'spl': 0.3490780064896874,
        'sdtw': 0.3162339952307343,
        'ndtw': 0.5438083517295451
    }

    self.assertAlmostEqual(score_dict, expected_score_dict)
    self.assertAlmostEqual(avg_scores, expected_avg_scores)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
