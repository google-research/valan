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

from __future__ import print_function

import collections

import tensorflow.compat.v2 as tf
from valan.framework import common
from valan.framework import problem_type
from valan.r2r import agent_config as agent_config_lib
from valan.r2r import constants
from valan.r2r import discriminator_agent
from valan.r2r import env
from valan.r2r import env_config as env_config_lib
from valan.r2r import eval_metric


R2RDebugInfo = collections.namedtuple(
    'R2RDebugInfo', ['episode_undisc_reward', 'episode_num_steps', 'num_paths'])


class DiscriminatorProblem(problem_type.ProblemType):
  """Mock problem type."""

  def __init__(self, runtime_config, mode, data_sources, agent_config=None,
               env_config=None):
    self._runtime_config = runtime_config
    self._mode = mode
    self._data_sources = data_sources

    self._env_config = (
        env_config if env_config else env_config_lib.get_default_env_config())
    self._env = None
    self._loss_type = None
    self._eval_dict = self._get_eval_dict()

    agent_config = (
        agent_config
        if agent_config else agent_config_lib.get_r2r_agent_config())
    agent_type = (
        agent_config.agent_type
        if hasattr(agent_config, 'agent_type') else 'default')
    if agent_type == 'default':
      self._agent = discriminator_agent.DiscriminatorAgent(
          agent_config, mode=mode)
    elif agent_type == 'v2':
      self._agent = discriminator_agent.DiscriminatorAgentV2(
          agent_config, mode=mode)

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
          env_config=self._env_config)
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

  def select_actor_action(self, env_output, unused_agent_output):
    """Returns the next ground truth action pano id."""
    time_step = env_output.observation[constants.TIME_STEP]
    current_pano_id = env_output.observation[constants.PANO_ID]
    golden_path = env_output.observation[constants.GOLDEN_PATH]
    golden_path_len = sum(
        [1 for pid in golden_path if pid != constants.INVALID_NODE_ID])

    # Sanity check: ensure pano id is on the golden path.
    if current_pano_id != golden_path[time_step]:
      raise ValueError(
          'Current pano id does not match that in golden path: {} vs. {}'
          .format(current_pano_id, golden_path[time_step]))

    if (time_step == golden_path_len - 1 or
        current_pano_id == constants.STOP_NODE_ID):
      next_golden_pano_id = constants.STOP_NODE_ID
    else:
      next_golden_pano_id = golden_path[time_step + 1]

    try:
      unused_action_idx = tf.where(
          tf.equal(env_output.observation[constants.CONN_IDS],
                   next_golden_pano_id))
    except ValueError:
      # Current and next panos are not connected, use idx for invalid node.
      unused_action_idx = unused_action_idx = tf.where(
          tf.equal(env_output.observation[constants.CONN_IDS],
                   constants.INVALID_NODE_ID))
    unused_action_idx = tf.cast(tf.reduce_min(unused_action_idx), tf.int32)
    return common.ActorAction(
        chosen_action_idx=unused_action_idx.numpy(),
        oracle_next_action_idx=unused_action_idx.numpy(),
        action_val=int(next_golden_pano_id),
        log_prob=0)

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
