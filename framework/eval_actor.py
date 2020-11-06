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

from __future__ import print_function

import os
import pickle
import time

from typing import Text
from absl import flags
from absl import logging
from seed_rl import grpc
import tensorflow.compat.v2 as tf
from tensorflow.io import gfile

from valan.framework import common
from valan.framework import problem_type as framework_problem_type
from valan.framework import utils


FLAGS = flags.FLAGS


def run_evaluation(
    problem_type, aggregator_address: Text, hparams,
    checkpoint_path=None, save_dir=None, file_prefix=None,
    test_mode=False):
  """Run evaluation actor.

  In without-aggregator mode:
    Saves the eval results from all workers in pickles named in the format
    <FILE_PREFIX>_<TASK_ID>_<STEP>_<PATH_ID>.p
    The aggregation that avoids async aggregation is done with
    .utils.NavigationScorePickleAggregator().

  Args:
    problem_type: An instance of `framework_problem_type.ProblemType`.
    aggregator_address: The aggregator address to which we will send data for
      batching.
    hparams: A dict containing hyperparameter settings.
    checkpoint_path: Required only when aggregator_address is None, i.e.,
      running without the aggregator. Path to a saved checkpoint for eval
      (usually the best performing).
    save_dir: Dir to save eval results (in pickles). Required when not using
      the aggregator.
    file_prefix: Run identifier. Required when not using the aggregator.
    test_mode: Unittest flag.
  """
  with_aggregator = False if aggregator_address is None else True
  assert isinstance(problem_type, framework_problem_type.ProblemType)
  env = problem_type.get_environment()
  agent = problem_type.get_agent()
  env_output = env.reset()

  agent_state = agent.get_initial_state(
      utils.add_batch_dim(env_output.observation), batch_size=1)
  # Agent always expects time,batch dimensions.
  _, _ = agent(utils.add_time_batch_dim(env_output), agent_state)

  iter_steps = 0
  if with_aggregator:
    logging.info('Connecting to aggregator %s', aggregator_address)
    aggregator = grpc.Client(aggregator_address)
    latest_checkpoint_path = ''
  else:
    step = -1

  while hparams['max_iter'] == -1 or iter_steps < hparams['max_iter']:
    logging.info('Iteration %d of %d', iter_steps + 1, hparams['max_iter'])

    if with_aggregator:
      checkpoint_directory = os.path.join(hparams['logdir'], 'model.ckpt')
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_directory)
      if checkpoint_path == latest_checkpoint_path or not checkpoint_path:
        logging.info(
            'Waiting for next checkpoint. Previously evaluated checkpoint %s',
            latest_checkpoint_path)
        time.sleep(30)
        continue
      latest_checkpoint_path = checkpoint_path
      step = int(latest_checkpoint_path.split('-')[-1])
      logging.info('Evaluating latest checkpoint - %s', latest_checkpoint_path)
    else:
      step += 1
      logging.info('Evaluating chosen checkpoint - %s', checkpoint_path)
      eval_result_dict = dict()

    ckpt = tf.train.Checkpoint(agent=agent)
    ckpt.restore(checkpoint_path)

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

        action = problem_type.select_actor_action(env_output, agent_output)
        env_output = env.step(action.action_val)

        action_list.append(action.action_val)
        env_output_list.append(env_output)

        if env_output.done:
          eval_result = problem_type.eval(action_list, env_output_list)
          # `eval_result` is a dict.
          eval_result[common.STEP] = step

          if with_aggregator:
            aggregator.eval_enqueue(pickle.dumps(eval_result))  # pytype: disable=attribute-error

          break

      if not with_aggregator:
        if test_mode:
          path_id = '0000'
        else:
          path_id = env_output_list[0].observation['path_id']
        prefix_key = f"{file_prefix}_{hparams['task_id']}_{step}_{path_id}"
        eval_result_dict[prefix_key] = eval_result

    if not with_aggregator:
      if test_mode:
        data_source = 'test_data'
      else:
        data_source = FLAGS.data_source
      file_name = data_source + '_' + str(hparams['task_id']) + '.p'
      path = os.path.join(save_dir, file_name)
      with gfile.GFile(path, 'wb') as fp:
        pickle.dump(eval_result_dict, fp)

    iter_steps += 1


def run(problem_type, max_iter=-1, num_episodes_per_iter=45, task_id=None):
  """Runs the eval_actor with the given problem type.

  Args:
    problem_type: An instance of `framework_problem_type.ProblemType`.
    max_iter: Number of iterations after which the actor must stop. The default
      value of -1 means run indefinitely.
    num_episodes_per_iter: Number of episodes to execute per iteration.
    task_id: Task identifier used for saving aggregated results to disk. Used
      only when running without the aggregator.
  """

  assert isinstance(problem_type, framework_problem_type.ProblemType)
  tf.enable_v2_behavior()
  hparams = {}
  hparams['logdir'] = FLAGS.logdir
  # In the eval actor, max_iter is only for use unit tests. (It is the learner's
  # job to terminate execution.)
  hparams['max_iter'] = max_iter
  hparams['num_episodes_per_iter'] = num_episodes_per_iter
  hparams['task_id'] = task_id

  run_evaluation(
      problem_type, FLAGS.server_address, hparams,
      FLAGS.checkpoint_path, FLAGS.save_dir, FLAGS.file_prefix)
