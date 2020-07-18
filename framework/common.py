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

from absl import flags

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
ActorAction = collections.namedtuple(
    "ActorAction", ["chosen_action_idx", "oracle_next_action_idx"])

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

# Flags common for both actor and learner.

flags.DEFINE_string("logdir", "/tmp/agent", "TensorFlow log directory.")
flags.DEFINE_integer("unroll_length", 10, "Unroll length in agent steps.")
flags.DEFINE_string(
    "server_address", "",
    "Actors use server_address to connect to a learner. Eval actors use "
    "server_address to connect to an eval aggregator. Learners and eval "
    "aggregators use server_address to bind to a listen address.")
