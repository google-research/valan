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

"""Abstract problem class."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class ProblemType(object):
  """Abstract class to hold problem-specific logic."""

  @abc.abstractmethod
  def get_environment(self):
    """Returns environment object of type `BaseEnv`."""

  @abc.abstractmethod
  def get_agent(self):
    """Returns agent object of type `BaseAgent`."""

  @abc.abstractmethod
  def get_optimizer(self, learning_rate):
    """Returns a tf.keras Optimizer.

    Args:
      learning_rate: A float tensor or tf.keras LearningRateSchedule object if a
        decay schedule is set for learning rate. This can be passed directly to
        the ctors of tf.keras optimizers to create optimizer object.

    Returns:
      A tf.keras Optimizer.
    """

  @abc.abstractmethod
  def create_summary(self, step, info):
    """Create any necessary summary.

    Called in learner after each batch. It can print any summaries to either
    tensorboard or standard logs.

    Args:
      step: A scalar whose value is equal to (num_iterations * batch_size *
        unroll_length).
      info: Actor info for the current step. This is returned by
        `self.get_actor_info`.
    """

  @abc.abstractmethod
  def get_study_loss_types(self):
    """Returns all loss types used in the current study.

    Called in learner exactly once to determine all the loss types used in the
    study. The learner then creates a TF graph that has loss functions
    corresponding to all the returned loss types. Valid loss types are defined
    in `common.py`

    Returns:
      An iterable of loss types.
    """

  def get_episode_loss_type(self, iterations):
    """Returns loss type to be used for current episode.

    Called in actor exactly once per episode. This information is passed on by
    the actor to the learner. The inheriting problems can use this to:
      1. create learning schedules for different loss types, e.g., cross-entropy
          loss for first K steps and actor critic-loss afterwards.
      2. customize action selection in actor based on the loss type used for the
          episode.

    Args:
      iterations: The count of total iterations so far.

    Returns:
      A single loss type to be used for the current episode.
    """

  @abc.abstractmethod
  def get_actor_info(self, final_step_env_output, episode_reward_sum,
                     episode_num_steps):
    """Returns a tensor or nested structure containing debug information.

    Called in actor after each episode ends. This is then sent to learner which
    calls `self.create_summary` with the value returned by this method.

    Args:
      final_step_env_output: An instance of `EnvOutput`. The environment output
        for the last step in the episode.
      episode_reward_sum: A float containing the total reward of this episode.
      episode_num_steps: A float containing the total number of steps in this
        episode.

    Returns:
      A tensor or nested structure.
    """

  @abc.abstractmethod
  def select_actor_action(self, env_output, agent_output):
    """Selects action taken by the actor.

    Called after each step in the episode.

    Args:
      env_output: An instance of `EnvOutput`. The environment output of the
        current step.
      agent_output: An instance of `AgentOutput`. The agent output at the
        current step which contains policy_logits.

    Returns:
      actor_action: An instance of `ActorAction` that has:
        chosen_action_idx: int32 index of the action chosen. The index is
          expected to be in range [0, agent_output.policy_logits.shape[-1]).
        oracle_next_action_idx: int32 index of the oracle action. The index is
          expected to be in range [0, agent_output.policy_logits.shape[-1]).
        action_val: An int representing raw action that is then accepted by
          `step` method of environment.
        log_prob: A float representing the log-probability of the action under
          the agent's policy, if appropriate.
    """

  @abc.abstractmethod
  def eval(self, action_list, env_output_list, agent_output=None):
    """Computes eval metrics for the episode.

    Note that size of the `env_output_list` list is 1 more than the size of
    `action_list` list. The env_output_list[i] is the input to `agent` at
    timestep=i and action_list[i] is the action chosen by actor at timestep=i.

    The final element in `env_output_list` is the env output at the end of
    episode. It must contain the final step's reward, done must be `True` and
    there might be `info` if specified in the environment class. The observation
    of this env output will contain observation of the first step of the next
    episode.

    Args:
      action_list: List of actions taken during the episode.
      env_output_list: List of `EnvOutput` objects for the episode.
      agent_output: A tensor, which is the output of agents.

    Returns:
      A dict where the keys are string and values are floating points.
    """

  def postprocessing(self, action_output):
    """Postprocessing the action_output before sending it to learner.

    In most cases, we don't need to do any postprocessing, so just leave it
    identical.

    Args:
      action_output: An ActionOutput object with shape [time_step, ...]

    Returns:
      action_output: An ActionOutput object after postprocessing.
    """
    return action_output
