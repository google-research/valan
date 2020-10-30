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

"""Definition of R2R problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

import collections
import pickle

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from valan.framework import common
from valan.framework import problem_type
from valan.r2r import agent
from valan.r2r import agent_config
from valan.r2r import constants
from valan.r2r import curriculum_env
from valan.r2r import curriculum_env_config as curriculum_env_config_lib
from valan.r2r import env
from valan.r2r import env_config as env_config_lib
from valan.r2r import eval_metric
from valan.r2r.multi_task import mt_agent
from valan.r2r.multi_task import mt_agent_config


R2RDebugInfo = collections.namedtuple(
    'R2RDebugInfo', ['episode_undisc_reward', 'episode_num_steps', 'num_paths'])


class R2RProblem(problem_type.ProblemType):
  """Problem type for R2R."""

  def __init__(self,
               runtime_config,
               mode,
               data_sources,
               curriculum='',
               agent_type='r2r'):
    self._runtime_config = runtime_config
    self._mode = mode
    self._data_sources = data_sources
    self._curriculum = curriculum

    if agent_type == 'r2r':
      self._agent = agent.R2RAgent(agent_config.get_r2r_agent_config())
    elif agent_type == 'mt':
      self._agent = mt_agent.MTEnvAgAgent(mt_agent_config.get_agent_config())
    else:
      raise ValueError('Invalid agent_type: {}'.format(agent_type))

    self._prob_ac = 0.5
    self._env = None
    self._loss_type = None
    self._eval_dict = self._get_eval_dict()

  def _get_eval_dict(self):
    return {
        'eval/success_rate':
            eval_metric.get_success_rate,
        'eval/navigation_error':
            eval_metric.get_navigation_error,
        'eval/path_length':
            eval_metric.get_path_length,
        'eval/oracle_success':
            eval_metric.get_oracle_success,
        'eval/num_steps_before_stop':
            eval_metric.get_num_steps_before_stop,
        'eval/spl':
            eval_metric.get_spl,
        'eval/undiscounted_episode_reward':
            eval_metric.get_undisc_episode_reward,
        'eval/cls':
            eval_metric.get_cls,
        'eval/dtw':
            eval_metric.get_dtw,
        'eval/norm_dtw':
            eval_metric.get_norm_dtw,
        'eval/sdtw':
            eval_metric.get_sdtw,
        'eval/' + common.VISUALIZATION_IMAGES:
            eval_metric.get_visualization_image,
    }

  def get_environment(self):
    if not self._env:
      assert self._data_sources, 'data_sources must be non-empty.'
      if self._curriculum:
        # See actor_main.py and curriculum_env.py for the argument options.
        self._env = curriculum_env.CurriculumR2REnv(
            data_sources=self._data_sources,
            runtime_config=self._runtime_config,
            curriculum_env_config=
            curriculum_env_config_lib.get_default_curriculum_env_config(
                self._curriculum)
        )
      else:
        self._env = env.R2REnv(
            data_sources=self._data_sources,
            runtime_config=self._runtime_config,
            env_config=env_config_lib.get_default_env_config())
    return self._env

  def get_agent(self):
    return self._agent

  def get_optimizer(self, learning_rate):
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)

  def create_summary(self, step, info):
    sum_episode_reward = 0.
    sum_episode_num_steps = 0.
    num_infos = 0

    num_paths_list = []
    for infos in [pickle.loads(t.numpy()) for t in info]:
      for episode_undisc_reward, episode_num_steps, num_paths in infos:
        sum_episode_reward += episode_undisc_reward
        sum_episode_num_steps += episode_num_steps
        num_paths_list.append(num_paths)
        num_infos += 1

    if num_infos:
      tf.summary.scalar(
          'train_debug/episode_undiscounted_reward',
          sum_episode_reward / num_infos,
          step=step)
      tf.summary.scalar(
          'train_debug/episode_num_steps',
          sum_episode_num_steps / num_infos,
          step=step)
      # Log the number of paths for analyzing curriculum learning.
      tf.summary.scalar(
          'train_debug/env_num_paths_mean',
          sum(num_paths_list) / num_infos,
          step=step)
      tf.summary.scalar(
          'train_debug/env_num_paths_maximum',
          max(num_paths_list),
          step=step)

  def get_actor_info(self, final_step_env_output, episode_reward_sum,
                     episode_num_steps):
    return R2RDebugInfo(episode_reward_sum, episode_num_steps,
                        self._env.num_paths)

  def get_study_loss_types(self):
    return [common.AC_LOSS, common.CE_LOSS]

  def get_episode_loss_type(self, iterations):
    self._loss_type = np.random.choice([common.AC_LOSS, common.CE_LOSS],
                                       p=[self._prob_ac, 1. - self._prob_ac])
    return self._loss_type

  def select_actor_action(self, env_output, agent_output):
    oracle_next_action = env_output.observation[constants.ORACLE_NEXT_ACTION]
    oracle_next_action_indices = tf.where(
        tf.equal(env_output.observation[constants.CONN_IDS],
                 oracle_next_action))
    oracle_next_action_idx = tf.reduce_min(oracle_next_action_indices)
    assert self._mode, 'mode must be set.'
    if self._mode == 'train':
      if self._loss_type == common.CE_LOSS:
        # This is teacher-forcing mode, so choose action same as oracle action.
        action_idx = oracle_next_action_idx
      elif self._loss_type == common.AC_LOSS:
        # Choose next pano from probability distribution over next panos
        action_idx = tfp.distributions.Categorical(
            logits=agent_output.policy_logits).sample()
      else:
        raise ValueError('Unsupported loss type {}'.format(self._loss_type))
    else:
      # In non-train modes, choose greedily.
      action_idx = tf.argmax(agent_output.policy_logits, axis=-1)
    action_val = env_output.observation[constants.CONN_IDS][action_idx]
    return common.ActorAction(
        chosen_action_idx=int(action_idx.numpy()),
        oracle_next_action_idx=int(oracle_next_action_idx.numpy())), int(
            action_val.numpy())

  def eval(self, action_list, env_output_list):
    result = {}
    for key, fn in self._eval_dict.items():
      score = fn(action_list, env_output_list, self._env)
      result[key] = score
    return result
