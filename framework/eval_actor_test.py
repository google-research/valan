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

"""Tests for valan.framework.eval_actor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

from absl import flags
from seed_rl import grpc
import tensorflow.compat.v2 as tf
from tensorflow.io import gfile
from valan.framework import eval_actor
from valan.framework import hyperparam_flags  
from valan.framework import testing_utils

FLAGS = flags.FLAGS


def _get_ckpt_manager(ckpt_dir, **kwargs):
  checkpoint_prefix = os.path.join(ckpt_dir, 'model.ckpt')
  ckpt = tf.train.Checkpoint(**kwargs)
  manager = tf.train.CheckpointManager(
      ckpt, checkpoint_prefix, max_to_keep=5, keep_checkpoint_every_n_hours=6)
  return manager


class EvalActorTest(tf.test.TestCase):

  def test_run_eval_actor_once(self):
    hparams = {}
    hparams['max_iter'] = 1
    hparams['num_episodes_per_iter'] = 5
    hparams['logdir'] = os.path.join(FLAGS.test_tmpdir, 'model')

    mock_problem = testing_utils.MockProblem(unroll_length=FLAGS.unroll_length)
    agent = mock_problem.get_agent()
    ckpt_manager = _get_ckpt_manager(hparams['logdir'], agent=agent)
    checkpoint_path = ckpt_manager.save(checkpoint_number=0)

    # Create a no-op gRPC server that responds to Aggregator RPCs.
    server_address = 'unix:/tmp/eval_actor_test_grpc'
    server = grpc.Server([server_address])

    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
    def eval_enqueue(_):
      return []

    # Test 01. Eval with aggregator.
    server.bind(eval_enqueue)
    server.start()
    eval_actor.run_evaluation(mock_problem, server_address, hparams)

    # Test 02. Eval without aggregator.
    hparams['task_id'] = 0000
    hparams['max_iter'] = 1
    eval_actor.run_evaluation(
        mock_problem, None, hparams,
        checkpoint_path, FLAGS.test_tmpdir, file_prefix='mock_test',
        test_mode=True)
    with gfile.GFile(
        os.path.join(FLAGS.test_tmpdir, 'test_data_0.p'), 'rb') as fp:
      eval_result = pickle.load(fp)
    self.assertEqual(eval_result['mock_test_0_0_0000']['result'], 1000.0)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
