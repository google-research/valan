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

"""Tests for valan.framework.learner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from absl import flags
from seed_rl import grpc
import tensorflow.compat.v2 as tf
from valan.framework import learner
from valan.framework import testing_utils
from valan.framework import utils

FLAGS = flags.FLAGS


class LearnerTest(tf.test.TestCase):

  def setUp(self):
    super(LearnerTest, self).setUp()
    # Remove existing local testing dir if exists.
    if tf.io.gfile.isdir(FLAGS.test_tmpdir):
      tf.io.gfile.rmtree(FLAGS.test_tmpdir)

  def testRunLearner(self):
    FLAGS.unroll_length = 6
    FLAGS.batch_size = 2
    logdir = FLAGS.test_tmpdir
    mock_problem = testing_utils.MockProblem(unroll_length=FLAGS.unroll_length)
    actor_output_spec = mock_problem.get_actor_output_spec()
    utils.write_specs(logdir, actor_output_spec)

    # Create dummy tensors with the right structure.
    zero_actor_output = tf.nest.map_structure(
        lambda sp: tf.zeros(shape=sp.shape, dtype=sp.dtype), actor_output_spec)

    server_address = 'unix:/tmp/learner_test_grpc'
    hparams = {}
    hparams['logdir'] = logdir
    hparams['warm_start_ckpt'] = None
    hparams['final_iteration'] = 5
    hparams['iter_frame_ratio'] = FLAGS.batch_size * FLAGS.unroll_length

    # Create a learner in a background thread. (Otherwise this call would
    # block.)
    thread = threading.Thread(
        target=learner.run_with_address,
        args=(mock_problem, server_address, hparams))
    thread.start()

    # Creating a client blocks until the learner responds.
    client = grpc.Client(server_address)

    # Send a number of enqueue RPCs to the learner.
    for i in range(FLAGS.batch_size * hparams['final_iteration']):
      client.enqueue(tf.nest.flatten(zero_actor_output))  # pytype: disable=attribute-error
    # Make sure the above `for` loop has completed successfully.
    self.assertEqual(i, FLAGS.batch_size * hparams['final_iteration'] - 1)

    # The learner should terminate after a fixed number of iterations.
    thread.join()


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
