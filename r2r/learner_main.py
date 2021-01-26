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

"""Learner main function."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
from valan.framework import common
from valan.framework import hyperparam_flags  
from valan.framework import learner
from valan.r2r import custom_flags  
from valan.r2r import ndh_problem
from valan.r2r import r2r_problem
from valan.r2r.multi_task import mt_problem

FLAGS = flags.FLAGS


def main(_):
  # Pseudo config. Will not be used in learner.
  runtime_config = common.RuntimeConfig(task_id=0, num_tasks=1)
  if FLAGS.problem == 'R2R':
    problem = r2r_problem.R2RProblem(
        runtime_config,
        mode=FLAGS.mode,
        data_sources=None,
        agent_type=FLAGS.agent_type)
  elif FLAGS.problem == 'NDH':
    problem = ndh_problem.NDHProblem(
        runtime_config,
        mode=FLAGS.mode,
        data_sources=None,
        agent_type=FLAGS.agent_type)
  elif FLAGS.problem == 'R2R+NDH':
    problem = mt_problem.MTProblem(runtime_config, mode=FLAGS.mode)
  else:
    raise ValueError('Unsupported problem type encountered: {}'.format(
        FLAGS.problem))

  logging.info('Begin running learner...')
  learner.run(problem)


if __name__ == '__main__':
  app.run(main)
