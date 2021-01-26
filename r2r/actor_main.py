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

"""Wrapper of actor and eval_actor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
from valan.framework import actor
from valan.framework import common
from valan.framework import eval_actor
from valan.framework import hyperparam_flags  
from valan.r2r import custom_flags  
from valan.r2r import ndh_problem
from valan.r2r import r2r_problem
from valan.r2r.multi_task import mt_problem


FLAGS = flags.FLAGS


def main(_):
  logging.info('Total shards: %d; Current shard index: %d', FLAGS.num_tasks,
               FLAGS.task)
  runtime_config = common.RuntimeConfig(
      task_id=FLAGS.task, num_tasks=FLAGS.num_tasks)
  data_sources = FLAGS.data_source.split(',')
  aggregator_prefix = '_'.join(data_sources)

  # Get problem instance.
  if FLAGS.problem == 'R2R':
    problem = r2r_problem.R2RProblem(
        runtime_config,
        mode=FLAGS.mode,
        data_sources=data_sources,
        curriculum=FLAGS.curriculum,
        agent_type=FLAGS.agent_type)
  elif FLAGS.problem == 'NDH':
    problem = ndh_problem.NDHProblem(
        runtime_config,
        mode=FLAGS.mode,
        data_sources=data_sources,
        agent_type=FLAGS.agent_type)
  elif FLAGS.problem == 'R2R+NDH':
    # Multi-task problem-type during training only. Use task-specific problems
    # during eval.
    if FLAGS.mode != 'train':
      raise ValueError('Multi-tasking is only supported for training. '
                       'Use task-specific problems during eval.')
    problem = mt_problem.MTProblem(runtime_config, mode=FLAGS.mode)
  else:
    raise ValueError('Unsupported problem type encountered: {}'.format(
        FLAGS.problem))

  logging.info('Current mode is %s', FLAGS.mode)
  if FLAGS.mode == 'train':
    logging.info('Running train actor...')
    actor.run(problem)
  else:
    logging.info('Running eval actor...')
    eval_actor.run(
        problem,
        # Evaluate each path in the dataset exactly once.
        num_episodes_per_iter=problem.get_environment().num_paths,
        task_id=runtime_config.task_id)


if __name__ == '__main__':
  app.run(main)
