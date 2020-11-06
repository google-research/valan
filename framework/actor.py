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

"""Actor implementation for VALAN.

This implementation is inspired from IMPALA (https://arxiv.org/abs/1802.01561).

To use the actor, create the appropriate problem_type and call the `run` method.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import pickle
import time
from typing import Any, Dict, Optional, Text

from absl import flags
from absl import logging
from seed_rl import grpc
from six.moves import range
import tensorflow.compat.v2 as tf
from valan.framework import common
from valan.framework import hyperparam_flags  
from valan.framework import problem_type as framework_problem_type
from valan.framework import utils


FLAGS = flags.FLAGS


def add_time_dimension(s: tf.TensorSpec):
  return tf.TensorSpec([FLAGS.unroll_length + 1] + s.shape.as_list(), s.dtype)


def _write_tensor_specs(initial_agent_state: Any,
                        env_output: common.EnvOutput,
                        agent_output: common.AgentOutput,
                        actor_action: common.ActorAction,
                        loss_type: Optional[int] = common.AC_LOSS):
  """Writes tensor specs of ActorOutput tuple to disk.

  Args:
    initial_agent_state: A tensor or nested structure of tensor without any time
      or batch dimensions.
    env_output: An instance of `EnvOutput` where individual tensors don't have
      time and batch dimensions.
    agent_output: An instance of `AgentOutput` where individual tensors don't
      have time and batch dimensions.
    actor_action: An instance of `ActorAction`.
    loss_type: A scalar int denoting the loss type.
  """
  actor_output = common.ActorOutput(
      initial_agent_state,
      env_output,
      agent_output,
      actor_action,
      loss_type,
      info='')
  specs = tf.nest.map_structure(tf.convert_to_tensor, actor_output)
  specs = tf.nest.map_structure(tf.TensorSpec.from_tensor, specs)
  env_output = tf.nest.map_structure(add_time_dimension, specs.env_output)
  agent_output = tf.nest.map_structure(add_time_dimension, specs.agent_output)
  actor_action = tf.nest.map_structure(add_time_dimension, specs.actor_action)
  specs = specs._replace(
      env_output=env_output,
      agent_output=agent_output,
      actor_action=actor_action)
  utils.write_specs(FLAGS.logdir, specs)


def run_with_learner(problem_type: framework_problem_type.ProblemType,
                     learner_address: Text, hparams: Dict[Text, Any],
                     num_retries: int = 10):
  """Runs actor with the given learner address and problem type.

  Args:
    problem_type: An instance of `framework_problem_type.ProblemType`.
    learner_address: The network address of a learner exposing two methods:
      `variable_values`: which returns latest value of trainable variables.
      `enqueue`: which accepts nested tensors of type `ActorOutput` tuple.
    hparams: A dict containing hyperparameter settings.
    num_retries: number of retries when the learner is closed.
  """
  env = problem_type.get_environment()
  agent = problem_type.get_agent()
  env_output = env.reset()
  initial_agent_state = agent.get_initial_state(
      utils.add_batch_dim(env_output.observation), batch_size=1)
  # Agent always expects time,batch dimensions. First add and then remove.
  env_output = utils.add_time_batch_dim(env_output)
  agent_output, _ = agent(env_output, initial_agent_state)
  env_output, agent_output = utils.remove_time_batch_dim(
      env_output, agent_output)
  actor_action = common.ActorAction(
      chosen_action_idx=tf.zeros([], dtype=tf.int32),
      oracle_next_action_idx=tf.zeros([], dtype=tf.int32),
      action_val=tf.zeros([], dtype=tf.int32),
      log_prob=tf.zeros([], dtype=tf.float32))
  # Remove batch_dim from returned agent's initial state.
  initial_agent_state = tf.nest.map_structure(lambda t: tf.squeeze(t, 0),
                                              initial_agent_state)

  # Write TensorSpecs the learner can use for initialization.
  logging.info('My task id is %d', FLAGS.task)
  if FLAGS.task == 0:
    _write_tensor_specs(initial_agent_state, env_output, agent_output,
                        actor_action)

  # gRPC Client creation blocks until the server responds to an RPC. Since the
  # server blocks at startup looking for TensorSpecs, and will not respond to
  # gRPC calls until these TensorSpecs are written, client creation must happen
  # after the actor writes TensorSpecs in order to prevent a deadlock.
  logging.info('Connecting to learner: %s', learner_address)
  client = grpc.Client(learner_address)

  iter_steps = 0
  num_steps = 0
  sum_reward = 0.
  # add batch_dim
  agent_state = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                      initial_agent_state)

  retry = 0
  iterations = 0
  while iter_steps < hparams['max_iter'] or hparams['max_iter'] == -1:
    logging.info('Iteration %d of %d', iter_steps + 1, hparams['max_iter'])
    # Get fresh parameters from the trainer.
    var_dtypes = [v.dtype for v in agent.trainable_variables]
    # trainer also adds `iterations` to the list of variables -- which is a
    # counter tracking number of iterations done so far.
    var_dtypes.append(tf.int64)
    new_values = []
    if iter_steps % hparams['sync_agent_every_n_steps'] == 0:
      new_values = client.variable_values()  # pytype: disable=attribute-error
    if new_values:
      logging.debug('Fetched variables from learner.')
      iterations = new_values[-1].numpy()
      updated_agent_vars = new_values[:-1]
      assert len(updated_agent_vars) == len(agent.trainable_variables)
      for x, y in zip(agent.trainable_variables, updated_agent_vars):
        x.assign(y)

    infos = []
    # Unroll agent.
    # Every episode sent by actor includes previous episode's final agent
    # state and output as well as final environment output.
    initial_agent_state = tf.nest.map_structure(lambda t: tf.squeeze(t, 0),
                                                agent_state)
    env_outputs = [env_output]
    agent_outputs = [agent_output]
    actor_actions = [actor_action]
    loss_type = problem_type.get_episode_loss_type(iterations)

    for i in range(FLAGS.unroll_length):
      logging.debug('Unroll step %d of %d', i + 1, FLAGS.unroll_length)
      # Agent expects time,batch dimensions in `env_output` and batch
      # dimension in `agent_state`. `agent_state` already has batch_dim.
      env_output = utils.add_time_batch_dim(env_output)
      agent_output, agent_state = agent(env_output, agent_state)

      env_output, agent_output = utils.remove_time_batch_dim(
          env_output, agent_output)

      actor_action = problem_type.select_actor_action(env_output, agent_output)

      env_output = env.step(actor_action.action_val)

      env_outputs.append(env_output)
      agent_outputs.append(agent_output)
      actor_actions.append(actor_action)
      num_steps += 1
      sum_reward += env_output.reward

      if env_output.done:
        infos.append(
            problem_type.get_actor_info(env_output, sum_reward, num_steps))
        num_steps = 0
        sum_reward = 0.

    processed_env_output = problem_type.postprocessing(
        utils.stack_nested_tensors(env_outputs))

    actor_output = common.ActorOutput(
        initial_agent_state=initial_agent_state,
        env_output=processed_env_output,
        agent_output=utils.stack_nested_tensors(agent_outputs),
        actor_action=utils.stack_nested_tensors(actor_actions),
        loss_type=tf.convert_to_tensor(loss_type, tf.int32),
        info=pickle.dumps(infos))
    flattened = tf.nest.flatten(actor_output)

    try:
      # NOTE: must disable the following to avoid pytype error.
      client.enqueue(flattened)  # pytype: disable=attribute-error
      retry = 0  # Reset retry.
    except tf.errors.UnavailableError:
      if retry > num_retries:
        raise ConnectionError('Learner unavailable. Lost connection.')
      retry += 1
      logging.warn('Sever unavailable. Wait 60secs (retry %s out of %s).',
                   retry, num_retries)
      time.sleep(60)
      # Reconnect to learner.
      client = grpc.Client(learner_address)
    iter_steps += 1


def run(problem_type: framework_problem_type.ProblemType):
  """Runs the actor with the given problem type."""
  tf.enable_v2_behavior()
  hparams = {}
  # In the actor, max_iter is only for use unit tests. (It is the learner's job
  # to terminate execution.)
  hparams['max_iter'] = -1  # -1 means infinite
  hparams['sync_agent_every_n_steps'] = FLAGS.sync_agent_every_n_steps
  run_with_learner(problem_type, FLAGS.server_address, hparams)
