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

"""Tests for valan.framework.actor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from seed_rl import grpc
import tensorflow.compat.v2 as tf
from valan.framework import actor
from valan.framework import testing_utils

FLAGS = flags.FLAGS


class ActorTest(tf.test.TestCase):

  def testRunActorOnce(self):
    FLAGS.unroll_length = 6
    mock_problem = testing_utils.MockProblem(unroll_length=FLAGS.unroll_length)

    hparams = {}
    hparams['max_iter'] = 1
    hparams['sync_agent_every_n_steps'] = 1

    # Create a no-op gRPC server that responds to Learner RPCs.
    flat_specs = [
        tf.TensorSpec.from_spec(s, str(i)) for i, s in enumerate(
            tf.nest.flatten(mock_problem.get_actor_output_spec()))
    ]

    server_address = 'unix:/tmp/actor_test_grpc'
    server = grpc.Server([server_address])

    @tf.function(input_signature=[])
    def variable_values():
      return []

    @tf.function(input_signature=flat_specs)
    def enqueue(*tensor_list):
      testing_utils.assert_matches_spec(flat_specs, tensor_list)
      return []

    server.bind(variable_values)
    server.bind(enqueue)

    server.start()

    actor.run_with_learner(mock_problem, server_address, hparams)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
