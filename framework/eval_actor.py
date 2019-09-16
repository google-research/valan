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

"""Evaluation actor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

import os
import pickle
import time

from absl import flags
from absl import logging
from seed_rl import grpc
import tensorflow.compat.v2 as tf
from typing import Text
from valan.framework import actor_config  
from valan.framework import common
from valan.framework import learner_config  
from valan.framework import problem_type as framework_problem_type
from valan.framework import utils

FLAGS = flags.FLAGS


def run_with_aggregator(problem_type, aggregator_address: Text, hparams):
  """Run evaluation actor with given problem_type, aggregator and hparams.

  Args:
    problem_type: An instance of `framework_problem_type.ProblemType`.
    aggregator_address: The aggregator address to which we will send data for
      batching.
    hparams: A dict containing hyperparameter settings.
  """
  assert isinstance(problem_type, framework_problem_type.ProblemType)
  env = problem_type.get_environment()
  agent = problem_type.get_agent()
  env_output = env.reset()

  agent_state = agent.get_initial_state(
      utils.add_batch_dim(env_output.observation), batch_size=1)
  # Agent always expects time,batch dimensions.
  _, _ = agent(utils.add_time_batch_dim(env_output), agent_state)

  logging.info('Connecting to aggregator %s', aggregator_address)
  aggregator = grpc.Client(aggregator_address)

  iter_steps = 0
  latest_checkpoint_path = ''
  while hparams['max_iter'] == -1 or iter_steps < hparams['max_iter']:
    logging.info('Iteration %d of %d', iter_steps + 1, hparams['max_iter'])
    checkpoint_directory = os.path.join(hparams['logdir'], 'model.ckpt')
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_directory)
    if checkpoint_path == latest_checkpoint_path or not checkpoint_path:
      logging.info(
          'Waiting for next checkpoint. Previously evaluated checkpoint %s',
          latest_checkpoint_path)
      time.sleep(30)
      continue

    ckpt = tf.train.Checkpoint(agent=agent)
    ckpt.restore(checkpoint_path)
    latest_checkpoint_path = checkpoint_path
    logging.info('Evaluating latest checkpoint - %s', latest_checkpoint_path)

    step = int(latest_checkpoint_path.split('-')[-1])
    logging.debug('Step %d', step)

    for i in range(hparams['num_episodes_per_iter']):
      logging.debug('Episode number %d of %d', i + 1,
                    hparams['num_episodes_per_iter'])
      action_list = []
      env_output_list = [env_output]
      while True:
        env_output = utils.add_time_batch_dim(env_output)
        agent_output, agent_state = agent(env_output, agent_state)
        env_output, agent_output = utils.remove_time_batch_dim(
            env_output, agent_output)

        _, action_val = problem_type.select_actor_action(
            env_output, agent_output)
        env_output = env.step(action_val)

        action_list.append(action_val)
        env_output_list.append(env_output)

        if env_output.done:
          eval_result = problem_type.eval(action_list, env_output_list)
          # eval_result is a dict.
          eval_result[common.STEP] = step
          aggregator.eval_enqueue(pickle.dumps(eval_result))  # pytype: disable=attribute-error
          break
      iter_steps += 1


def run(problem_type, max_iter=-1, num_episodes_per_iter=45):
  """Runs the eval_actor with the given problem type.

  Args:
    problem_type: An instance of `framework_problem_type.ProblemType`.
    max_iter: Number of iterations after which the actor must stop. The default
      value of -1 means run indefinitely.
    num_episodes_per_iter: Number of episodes to execute per iteration.
  """

  assert isinstance(problem_type, framework_problem_type.ProblemType)
  tf.enable_v2_behavior()
  hparams = {}
  hparams['logdir'] = FLAGS.logdir
  # In the eval actor, max_iter is only for use unit tests. (It is the learner's
  # job to terminate execution.)
  hparams['max_iter'] = max_iter
  hparams['num_episodes_per_iter'] = num_episodes_per_iter

  run_with_aggregator(problem_type, FLAGS.server_address, hparams)
