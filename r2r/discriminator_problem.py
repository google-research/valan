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

"""Definition of Discriminator problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

import collections

import tensorflow.compat.v2 as tf
from valan.framework import common
from valan.framework import problem_type
from valan.r2r import agent_config
from valan.r2r import constants
from valan.r2r import discriminator_agent
from valan.r2r import env
from valan.r2r import env_config as env_config_lib
from valan.r2r import eval_metric


R2RDebugInfo = collections.namedtuple(
    'R2RDebugInfo', ['episode_undisc_reward', 'episode_num_steps', 'num_paths'])


class DiscriminatorProblem(problem_type.ProblemType):
  """Mock problem type."""

  def __init__(self, runtime_config, mode, data_sources):
    self._runtime_config = runtime_config
    self._mode = mode
    self._data_sources = data_sources

    self._env = None
    self._loss_type = None
    self._eval_dict = self._get_eval_dict()
    self._agent = discriminator_agent.DiscriminatorAgent(
        agent_config.get_r2r_agent_config())

  def _get_eval_dict(self):
    return {
        'eval/' + common.AUC: eval_metric.get_score_label,
    }

  def get_environment(self):
    if not self._env:
      assert self._data_sources, 'data_sources must be non-empty.'
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
    del step, info

  def get_actor_info(self, final_step_env_output, episode_reward_sum,
                     episode_num_steps):
    del final_step_env_output, episode_reward_sum, episode_num_steps

  def get_study_loss_types(self):
    return [common.DCE_LOSS]

  def get_episode_loss_type(self, iterations):
    return common.DCE_LOSS

  def select_actor_action(self, env_output, agent_output):
    # Agent_output is unused here.
    oracle_next_action = env_output.observation[constants.ORACLE_NEXT_ACTION]
    oracle_next_action_indices = tf.where(
        tf.equal(env_output.observation[constants.CONN_IDS],
                 oracle_next_action))
    oracle_next_action_idx = tf.reduce_min(oracle_next_action_indices)
    assert self._mode, 'mode must be set.'
    action_idx = oracle_next_action_idx
    action_val = env_output.observation[constants.CONN_IDS][action_idx]
    return common.ActorAction(
        chosen_action_idx=int(action_idx.numpy()),
        oracle_next_action_idx=int(oracle_next_action_idx.numpy())), int(
            action_val)

  def eval(self, action_list, env_output_list, agent_output=None):
    result = {}
    for key, fn in self._eval_dict.items():
      score = fn(action_list, env_output_list, agent_output, self._env)
      result[key] = score
    return result

  def postprocessing(self, env_output):
    observation = env_output.observation
    # [time_step, 1]
    is_start = observation[constants.IS_START].numpy()
    cnt = 0
    mask = []
    for i in range(is_start.shape[0]):
      cnt += is_start[i]
      if cnt == 1:
        mask.append(True)
      else:
        mask.append(False)
    mask = tf.reshape(tf.convert_to_tensor(mask), is_start.shape)
    observation[constants.DISC_MASK] = mask
    env_output = env_output._replace(observation=observation)
    return env_output
