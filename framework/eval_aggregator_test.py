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

"""Tests for valan.framework.eval_aggregator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import threading

from absl import flags
from seed_rl import grpc
import tensorflow.compat.v2 as tf
from valan.framework import common
from valan.framework import eval_aggregator

FLAGS = flags.FLAGS


class EvalAggregatorTest(tf.test.TestCase):

  def test_update_summary(self):
    key_one = 'first_metric'
    key_two = 'second_metric'
    init_metrics = {key_one: 1.0, key_two: 2.0}
    added_metric = {key_one: 2.0, key_two: 3.0}
    expected_metric = {key_one: 3.0, key_two: 5.0}
    init_summary = eval_aggregator.StepSummaries(
        step=10, count=10, metrics_sum=init_metrics)
    output_summary = eval_aggregator.update_summary(init_summary, added_metric)
    self.assertEqual(output_summary.count, 11)
    self.assertEqual(output_summary.metrics_sum, expected_metric)

  def test_run_eval_aggregator_many_times(self):
    server_address = 'unix:/tmp/eval_aggregator_test_grpc'
    hparams = {}
    hparams['logdir'] = os.path.join(FLAGS.test_tmpdir, 'mode')
    hparams['num_samples'] = 10

    # Create an eval aggregator in a background thread. (Otherwise this call
    # would block.)
    thread = threading.Thread(
        target=eval_aggregator.run_with_address, args=(server_address, hparams))
    thread.start()

    # Creating a client blocks until the aggregator responds.
    client = grpc.Client(server_address)

    # Send a number of eval_enqueue RPCs to the aggregator.
    for i in range(hparams['num_samples'] + 1):
      msg = pickle.dumps({common.STEP: i / 2, 'eval/a_number': 1})
      client.eval_enqueue(msg)  # pytype: disable=attribute-error

    # The aggregator should terminate after num_samples RPCs. Wait for it to
    # exit.
    thread.join()


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
