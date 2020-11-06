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

"""Common collections for all problems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Tuple containing state of the environment.
#  reward: a scalar float specifying the immediate reward for the step.
#  done: a boolean indicating if this is the end of the current episode.
#  observation: a numpy array or nested dict/list/tuple of numpy arrays
#    containing information about current internal state.
#  info: a numpy array or nested dict/list/tuple of numpy arrays containing
#    any debug information.
EnvOutput = collections.namedtuple("EnvOutput",
                                   ["reward", "done", "observation", "info"])

# Tuple containing output of the agent.
#  policy_logits: The logits of all possible actions.
#  baseline: The value of given state.
AgentOutput = collections.namedtuple("AgentOutput",
                                     ["policy_logits", "baseline"])

# Tuple containing runtime configuration for distributed setup.
#   task_id: A unique id assigned to each task
#   num_tasks: Total tasks performing the same functionality.
RuntimeConfig = collections.namedtuple("RuntimeConfig",
                                       ["task_id", "num_tasks"])

# Tuple containing information about action taken by actor.
#   chosen_action_idx: An int32 specifying the index of the chosen action at the
#     current timestep.
#   oracle_next_action_idx: An int32 specifying the index of the action at the
#     next timestep that oracle would have chosen.
#   action_val: An int32 specifying the pano id of the chosen action.
#   log_prob: Float specifying the policy log-probability of the chosen action.
ActorAction = collections.namedtuple(
    "ActorAction", ["chosen_action_idx", "oracle_next_action_idx", "action_val",
                    "log_prob"])

# Tuple containing output of the actor which is then read by learner.
#  initial_agent_state: a tensor containing previous episode's final agent
#    state. This may be used to initialize learner agent's initial state at
#    the beginning of a batch. This tensor doesn't have time or batch dimension.
#  env_output: A `EnvOutput` tuple for all the steps in the episode. The nested
#    tensors have first dimension equal to number of timesteps.
#  agent_output: A `AgentOutput` tuple for all steps in the episode. The nested
#    tensors have first dimension equal to number of timesteps.
#  action_action: A `ActorAction` tuple for all steps in the episode. The nested
#    tensors have first dimension equal to number of timesteps.
#  loss_type: A scalar int tensor denoting the type of loss to be used by
#    learner on the enqueued episode.
#  info: Any debug information to be passed on to the learner.
ActorOutput = collections.namedtuple("ActorOutput", [
    "initial_agent_state", "env_output", "agent_output", "actor_action",
    "loss_type", "info"
])

# Tuple contaiming information for aggregator summaries.
StepSummaries = collections.namedtuple("StepSummaries",
                                       ("step", "count", "metrics_sum"))

# Tuple containing agent and environment state, for use in planning.
#  score: a scalar float for comparing the value of planning states.
#  agent_output: A `AgentOutput` tuple.
#  agent_state: A tensor containing the agent state. This tensor doesn't have
#    time or batch dimension.
#  env_output: A `EnvOutput` tuple.
#  env_state: An object containing the environment state from get_state.
#  action_history: A list with an `ActorAction` for each step in the episode.
PlanningState = collections.namedtuple("PlanningState", [
    "score", "agent_output", "agent_state", "env_output", "env_state",
    "action_history"])

# Different loss types supported in the framework.
# actor-critic loss
AC_LOSS = 0
# cross-entropy loss
CE_LOSS = 1
# Discriminative model Cross-Entropy loss
DCE_LOSS = 2
# Discriminative model focal loss
DCE_FOCAL_LOSS = 3
# Discriminator Batch Softmax loss
DISC_BATCH_LOSS = 4

STEP = "__reserved__step"

# Special field for visualization images.
VISUALIZATION_IMAGES = "visualization_images"
AUC = "auc"
