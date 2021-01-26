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

"""Definition of VLN + NDH problems together for TRAIN only."""

import collections
import pickle

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from valan.framework import common
from valan.framework import problem_type

from valan.r2r import constants
from valan.r2r.multi_task import mt_agent
from valan.r2r.multi_task import mt_agent_config
from valan.r2r.multi_task import mt_env


DebugInfo = collections.namedtuple(
    'DebugInfo', ['episode_undisc_reward', 'episode_num_steps', 'task_type'])


class MTProblem(problem_type.ProblemType):
  """Problem type for both VLN and NDH combined for TRAIN only."""

  def __init__(self, runtime_config, mode):
    self._runtime_config = runtime_config

    self._agent = mt_agent.MTEnvAgAgent(
        mt_agent_config.get_agent_config(), mode=mode)
    self._prob_ac = 0.5
    self._env = None
    self._loss_type = None

  def get_environment(self):
    if not self._env:
      self._env = mt_env.MTEnv(self._runtime_config)
    return self._env

  def get_agent(self):
    return self._agent

  def get_optimizer(self, learning_rate):
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)

  def create_summary(self, step, info):
    sum_episode_reward = 0.
    sum_episode_num_steps = 0.
    num_vln_episodes = 0.
    num_infos = 0

    for infos in [pickle.loads(t.numpy()) for t in info]:
      for episode_undisc_reward, episode_num_steps, env_type in infos:
        sum_episode_reward += episode_undisc_reward
        sum_episode_num_steps += episode_num_steps
        num_vln_episodes += env_type
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
      tf.summary.scalar(
          'train_debug/fraction_vln_episodes',
          num_vln_episodes / num_infos,
          step=step)

  def get_actor_info(self, final_step_env_output, episode_reward_sum,
                     episode_num_steps):
    is_vln = final_step_env_output.observation[
        constants.PROBLEM_TYPE] == constants.PROBLEM_VLN
    return DebugInfo(episode_reward_sum, episode_num_steps,
                     1 if is_vln else 0)

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
    if self._loss_type == common.CE_LOSS:
      # This is teacher-forcing mode, so choose action same as oracle action.
      action_idx = oracle_next_action_idx
    elif self._loss_type == common.AC_LOSS:
      # Choose next pano from probability distribution over next panos
      action_idx = tfp.distributions.Categorical(
          logits=agent_output.policy_logits).sample()
    else:
      raise ValueError('Unsupported loss type {}'.format(self._loss_type))
    action_val = env_output.observation[constants.CONN_IDS][action_idx]
    policy_logprob = tf.nn.log_softmax(agent_output.policy_logits)
    return common.ActorAction(
        chosen_action_idx=int(action_idx.numpy()),
        oracle_next_action_idx=int(oracle_next_action_idx.numpy()),
        action_val=int(action_val.numpy()),
        log_prob=float(policy_logprob[action_idx].numpy()))

  def eval(self, action_list, env_output_list):
    # Shouldn't be called.
    pass
