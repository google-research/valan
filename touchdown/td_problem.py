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

"""Definition of Touchdown problem."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import pickle

from absl import flags
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from valan.framework import common
from valan.framework import eval_metric as base_eval_metric
from valan.framework import problem_type
from valan.framework import utils
from valan.streetview_common import baseline_agent
from valan.streetview_common import streetview_constants
from valan.touchdown import constants
from valan.touchdown import env
from valan.touchdown import env_config

FLAGS = flags.FLAGS

TdDebugInfo = collections.namedtuple(
    'TdDebugInfo',
    ('episode_reward_sum', 'episode_num_steps', 'navigation_error'))
# Defined in TD paper.
_SUCCESS_THRESHOLD = 2.


class TouchdownProblem(problem_type.ProblemType):
  """Problem definition for Touchdown."""

  def __init__(self, runtime_config, mode, data_sources, logdir=''):
    self._runtime_config = runtime_config
    self._mode = mode
    self._data_sources = data_sources
    self._logdir = logdir
    self._env = None
    self._debug_writer = None
    self._env_config = env_config.get_default_env_config(self._mode)
    self._loss_type = None
    assert self._env_config.max_agent_actions > 0, ('max_agent_actions must be '
                                                    'greater than zero')

    self._agent = self._pick_agent()

  def _pick_agent(self):
    return baseline_agent.StreetviewAgent(
        num_actions=self.get_action_set_dim(),
        instruction_len=self._env_config.instruction_tok_len,
        params=constants.TD_BASELINE_AGENT_PARAMS)

  def get_environment(self):
    if not self._env:
      self._env = env.TouchdownEnv(
          data_sources=self._data_sources,
          env_config=self._env_config,
          worker_idx=self._runtime_config.task_id)
    return self._env

  def get_agent(self):
    return self._agent

  def get_optimizer(self, learning_rate):
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)

  def create_summary(self, step, info):
    sum_episode_reward = 0.
    sum_episode_num_steps = 0.
    sum_navigation_error = 0.
    episodes_with_success = 0.
    num_infos = 0

    for infos in [pickle.loads(t.numpy()) for t in info]:
      for episode_reward, episode_num_steps, navigation_error in infos:
        sum_episode_reward += episode_reward
        sum_episode_num_steps += episode_num_steps
        sum_navigation_error += navigation_error
        episodes_with_success += navigation_error < _SUCCESS_THRESHOLD
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
          'train_debug/navigation_error',
          sum_navigation_error / num_infos,
          step=step)
      tf.summary.scalar(
          'train_debug/task_completion_rate',
          episodes_with_success / num_infos,
          step=step)

  def get_actor_info(self, final_step_env_output, episode_reward_sum,
                     episode_num_steps):
    return TdDebugInfo(episode_reward_sum, episode_num_steps,
                       final_step_env_output.info.distance_to_goal)

  def get_study_loss_types(self):
    return [common.CE_LOSS]

  def get_episode_loss_type(self, iterations):
    self._loss_type = common.CE_LOSS
    return self._loss_type

  def select_actor_action(self, env_output, agent_output):
    assert self._mode, 'mode must be set for selecting action in actor.'
    oracle_next_action = env_output.observation[
        streetview_constants.ORACLE_NEXT_ACTION]
    if self._mode == 'train':
      if self._loss_type == common.CE_LOSS:
        # This is teacher-forcing mode, so choose action same as oracle action.
        action_idx = oracle_next_action
      elif self._loss_type == common.AC_LOSS:
        action_idx = tfp.distributions.Categorical(
            logits=agent_output.policy_logits).sample()
    else:
      # In non-train modes, choose greedily.
      action_idx = tf.argmax(agent_output.policy_logits, axis=-1)

    # Return ActorAction with the action to be passed to the env step function.
    policy_logprob = tf.nn.log_softmax(agent_output.policy_logits)
    return common.ActorAction(
        chosen_action_idx=int(action_idx.numpy()),
        oracle_next_action_idx=int(
            oracle_next_action.numpy()),
        action_val=action_idx.numpy(),
        log_prob=float(policy_logprob[action_idx].numpy()))

  def does_use_panoramic_actions(self):
    return self._env_config.panoramic_action_space

  def get_action_set_dim(self):
    if self.does_use_panoramic_actions():
      return self._env_config.panoramic_action_bins + 1
    else:
      return streetview_constants.NUM_DISCRETE_ACTIONS

  def _get_sed_score(self, success_rate, golden_path, agent_path):
    if success_rate == 0.:
      return 0.

    levenshtein_distance = utils.levenshtein(golden_path, agent_path)
    levenshtein_distance_normalized = levenshtein_distance / (
        len(golden_path) * len(agent_path))
    return 1 - levenshtein_distance_normalized

  def _get_dtw_score(self, success_rate, golden_path, agent_path):
    distance_fn = self._env.shortest_path_length
    dtw_matrix = base_eval_metric.get_dtw_matrix(agent_path, golden_path,
                                                 distance_fn)
    dtw = dtw_matrix[len(agent_path)][len(golden_path)]
    pln_dtw = dtw / len(golden_path)
    ndtw = tf.math.exp(-1. * dtw / (_SUCCESS_THRESHOLD * len(golden_path)))
    sdtw = ndtw if success_rate else 0.
    return pln_dtw, ndtw, sdtw

  def _get_golden_path(self, env_output_list):
    golden_path = env_output_list[0].observation[
        streetview_constants.GOLDEN_PATH]
    golden_path = [
        pano for pano in golden_path
        if pano not in [streetview_constants.INVALID_PANO_ID]
    ]
    return golden_path

  def _get_agent_path(self, env_output_list):
    agent_path = [
        env_output.observation[streetview_constants.PANO_ID]
        for env_output in env_output_list
    ]
    return agent_path

  def eval(self, action_list, env_output_list):
    episode_reward = sum([env_output.reward for env_output in env_output_list])
    episode_num_steps = len(env_output_list)

    golden_path = self._get_golden_path(env_output_list)
    agent_path = self._get_agent_path(env_output_list)

    navigation_error = env_output_list[-1].info.distance_to_goal
    success_rate = 1. if navigation_error < _SUCCESS_THRESHOLD else 0.
    metrics = {}
    metrics['eval/episode_undiscounted_reward'] = episode_reward
    metrics['eval/episode_num_steps'] = episode_num_steps
    metrics['eval/navigation_error'] = navigation_error
    metrics['eval/success_rate'] = success_rate
    dtw, ndtw, sdtw = self._get_dtw_score(success_rate, golden_path, agent_path)
    metrics['eval/dtw'] = dtw
    metrics['eval/ndtw'] = ndtw
    metrics['eval/sdtw'] = sdtw
    metrics['eval/sed'] = self._get_sed_score(success_rate, golden_path,
                                              agent_path)
    return metrics
