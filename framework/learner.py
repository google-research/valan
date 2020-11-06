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

"""Learner implementation for VALAN.

This implementation is inspired from IMPALA (https://arxiv.org/abs/1802.01561).

To use the learner, create the appropriate problem_type and call the `run`
method.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import copy
import functools
import math
import os
import time
from typing import Any, Dict, List, Text

from absl import flags
from absl import logging
from seed_rl import grpc
import tensorflow as tf
from valan.framework import base_agent
from valan.framework import common
from valan.framework import hyperparam_flags  
from valan.framework import log_ckpt_restoration
from valan.framework import loss_fns
from valan.framework import problem_type as framework_problem_type
from valan.framework import utils

from tensorflow.python.client import device_lib  

FLAGS = flags.FLAGS



def _convert_uint8_to_bfloat16(ts: Any):
  """Casts uint8 to bfloat16 if input is uint8.

  Args:
    ts: any tensor or nested tensor structure, such as EnvOutput.

  Returns:
    Converted structure.
  """
  return tf.nest.map_structure(
      lambda t: tf.cast(t, tf.bfloat16) if t.dtype == tf.uint8 else t, ts)


def _maybe_restore_from_ckpt(ckpt_dir: Text, **kwargs):
  """Sets up checkpointing and restore checkpoint if available."""
  checkpoint_prefix = os.path.join(ckpt_dir, 'model.ckpt')
  ckpt = tf.train.Checkpoint(**kwargs)
  manager = tf.train.CheckpointManager(
      ckpt,
      checkpoint_prefix,
      max_to_keep=FLAGS.max_ckpt_to_keep,
      keep_checkpoint_every_n_hours=6)
  if manager.latest_checkpoint:
    logging.info('Restoring from checkpoint: %s', manager.latest_checkpoint)
    ckpt.restore(manager.latest_checkpoint)
  return manager


def _warm_start_from_ckpt(warm_start_ckpt_path: Text, training_ckpt_dir: Text,
                          model):
  """Initializes model weights from given ckpt and logs restoration status."""
  # Sets up a ckpt to restore weights only without optimizer.
  warmstart_ckpt = tf.train.Checkpoint(model=model)
  logging.info('Warm-starting from checkpoint (w/o optimizer): %s',
               warm_start_ckpt_path)
  # Use a weak assertion (fails when nothing is matched) and get the
  # restoration status object.
  status = warmstart_ckpt.restore(
      warm_start_ckpt_path).assert_nontrivial_match()

  # Save restoration details to file.
  if not tf.io.gfile.isdir(training_ckpt_dir):
    logging.warning(
        'Dir not found. Skip saving ckpt restoration information to: %s',
        training_ckpt_dir)
  else:
    log_ckpt_restoration.log_status(status, training_ckpt_dir)


def _create_server(
    listen_address: Text,
    specs: common.ActorOutput,
    agent: base_agent.BaseAgent,
    queue: tf.queue.QueueBase,
    extra_variables: List[tf.Variable],
):
  """Starts server for communicating with actor(s).

  The learner server exposes the following two methods for the actor:
    enqueue: actors are expected to call this server method to submit their
      trajectories for learner. This method accepts a nested structure of
      tensors of type `ActorOutput`.
    variable_values: actors can call this server method to get the latest value
      of trainable variables as well as variables in `extra_variables`.

  Args:
    listen_address: The network address on which to listen.
    specs: A nested structure where each element is either a tensor or a
      TensorSpec.
    agent: An instance of `BaseAgent`.
    queue: An instance of `tf.queue.QueueBase`.
    extra_variables: A list of variables other than `agent.trainable_variables`
      to be sent via `variable_values` method.

  Returns:
    A server object.
  """
  logging.info('Creating gRPC server on address %s', listen_address)
  server = grpc.Server([listen_address])
  flat_specs = [
      tf.TensorSpec.from_spec(s, str(i))
      for i, s in enumerate(tf.nest.flatten(specs))
  ]

  @tf.function(input_signature=flat_specs)
  def enqueue(*tensors: common.ActorOutput):
    queue.enqueue(tensors)
    return []

  server.bind(enqueue)

  @tf.function(input_signature=[])
  def variable_values():
    all_vars = copy.copy(agent.trainable_variables)
    all_vars += extra_variables
    return all_vars

  server.bind(variable_values)

  return server


def _transpose_batch(specs: common.ActorOutput, *actor_outputs):
  """Transpose a batch from the dataset into time-major order."""
  time_major_fn = lambda t: tf.nest.map_structure(utils.transpose_batch_time, t)
  actor_outputs = tf.nest.pack_sequence_as(specs, actor_outputs)
  actor_outputs = actor_outputs._replace(
      env_output=time_major_fn(actor_outputs.env_output),
      agent_output=time_major_fn(actor_outputs.agent_output),
      actor_action=time_major_fn(actor_outputs.actor_action))

  actor_outputs = actor_outputs._replace(
      env_output=_convert_uint8_to_bfloat16(actor_outputs.env_output))
  # tf.Dataset treats list leafs as tensors, so we need to flatten and repack.
  return tf.nest.flatten(actor_outputs)


def run_with_address(
    problem_type: framework_problem_type.ProblemType,
    listen_address: Text,
    hparams: Dict[Text, Any],
):
  """Runs the learner with the given problem type.

  Args:
    problem_type: An instance of `framework_problem_type.ProblemType`.
    listen_address: The network address on which to listen.
    hparams: A dict containing hyperparameter settings.
  """
  devices = device_lib.list_local_devices()
  logging.info('Found devices: %s', devices)
  devices = [d for d in devices if d.device_type == FLAGS.agent_device]
  assert devices, 'Could not find a device of type %s' % FLAGS.agent_device
  agent_device = devices[0].name
  logging.info('Using agent device: %s', agent_device)

  # Initialize agent, variables.
  specs = utils.read_specs(hparams['logdir'])
  flat_specs = [
      tf.TensorSpec.from_spec(s, str(i))
      for i, s in enumerate(tf.nest.flatten(specs))
  ]
  queue_capacity = FLAGS.queue_capacity or FLAGS.batch_size * 10
  queue = tf.queue.FIFOQueue(
      queue_capacity,
      [t.dtype for t in flat_specs],
      [t.shape for t in flat_specs],
  )
  agent = problem_type.get_agent()
  # Create dummy environment output of shape [num_timesteps, batch_size, ...].
  env_output = tf.nest.map_structure(
      lambda s: tf.zeros(  
          list(s.shape)[0:1] + [FLAGS.batch_size] + list(s.shape)[1:], s.dtype),
      specs.env_output)
  init_observation = utils.get_row_nested_tensor(env_output.observation, 0)
  init_agent_state = agent.get_initial_state(
      init_observation, batch_size=FLAGS.batch_size)
  env_output = _convert_uint8_to_bfloat16(env_output)
  with tf.device(agent_device):
    agent(env_output, init_agent_state)

    # Create optimizer.

    if FLAGS.lr_decay_steps > 0 and FLAGS.lr_decay_rate < 1.:
      lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
          initial_learning_rate=FLAGS.learning_rate,
          decay_steps=FLAGS.lr_decay_steps,
          decay_rate=FLAGS.lr_decay_rate)
    else:
      lr_schedule = FLAGS.learning_rate
    optimizer = problem_type.get_optimizer(lr_schedule)
    # NOTE: `iterations` is a non-trainable variable which is managed by
    # optimizer (created inside optimizer as well as incremented by 1 on every
    # call to optimizer.minimize).
    iterations = optimizer.iterations
    study_loss_types = problem_type.get_study_loss_types()

  @tf.function
  def train_step(iterator):
    """Training StepFn."""

    def step_fn(actor_output):
      """Per-replica StepFn."""
      actor_output = tf.nest.pack_sequence_as(specs, actor_output)
      (initial_agent_state, env_output, actor_agent_output, actor_action,
       loss_type, info) = actor_output
      with tf.GradientTape() as tape:
        loss = loss_fns.compute_loss(
            study_loss_types=study_loss_types,
            current_batch_loss_type=loss_type,
            agent=agent,
            agent_state=initial_agent_state,
            env_output=env_output,
            actor_agent_output=actor_agent_output,
            actor_action=actor_action,
            num_steps=iterations)
      grads = tape.gradient(loss, agent.trainable_variables)
      if FLAGS.gradient_clip_norm > 0.:
        for i, g in enumerate(grads):
          if g is not None:
            grads[i] = tf.clip_by_norm(g, FLAGS.gradient_clip_norm)
      grad_norms = {}
      for var, grad in zip(agent.trainable_variables, grads):
        # For parameters which are initialized but not used for loss
        # computation, gradient tape would return None.
        if grad is not None:
          grad_norms[var.name] = tf.norm(grad)
      optimizer.apply_gradients(zip(grads, agent.trainable_variables))
      return info, grad_norms

    return step_fn(next(iterator))

  if hparams['warm_start_ckpt']:
    _warm_start_from_ckpt(
        warm_start_ckpt_path=hparams['warm_start_ckpt'],
        training_ckpt_dir=hparams['logdir'],
        model=agent)

  ckpt_manager = _maybe_restore_from_ckpt(
      hparams['logdir'], agent=agent, optimizer=optimizer)
  server = _create_server(
      listen_address, specs, agent, queue, extra_variables=[iterations])
  logging.info('Starting gRPC server')
  server.start()

  dataset = tf.data.Dataset.from_tensors(0).repeat(None)
  dataset = dataset.map(lambda _: queue.dequeue())
  dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
  # Transpose each batch to time-major order. This is relatively slow, so do
  # this work outside of the training loop.
  dataset = dataset.map(functools.partial(_transpose_batch, specs))
  dataset = dataset.apply(tf.data.experimental.copy_to_device(agent_device))
  with tf.device(agent_device):
    dataset = dataset.prefetch(1)
    iterator = iter(dataset)

  # Execute learning and track performance.
  summary_writer = tf.summary.create_file_writer(
      hparams['logdir'], flush_millis=20000, max_queue=1000)
  last_ckpt_time = time.time()
  with summary_writer.as_default():
    last_log_iterations = iterations.numpy()
    last_log_num_env_frames = iterations * hparams['iter_frame_ratio']
    last_log_time = time.time()
    while iterations < hparams['final_iteration']:
      logging.info('Iteration %d of %d', iterations + 1,
                   hparams['final_iteration'])

      # Save checkpoint at specified intervals or if no previous ckpt exists.
      current_time = time.time()
      if (current_time - last_ckpt_time >= FLAGS.save_checkpoint_secs or
          not ckpt_manager.latest_checkpoint):
        ckpt_manager.save(checkpoint_number=iterations)
        last_ckpt_time = current_time

      with utils.WallTimer() as wt:
        with tf.device(agent_device):
          info, grad_norms = train_step(iterator)
      tf.summary.scalar(
          'steps_summary/step_seconds', wt.duration, step=iterations)
      norm_summ_family = 'grad_norms/'
      for name, norm in grad_norms.items():
        tf.summary.scalar(norm_summ_family + name, norm, step=iterations)

      if current_time - last_log_time >= 120:
        num_env_frames = iterations.numpy() * hparams['iter_frame_ratio']
        num_frames_since = num_env_frames - last_log_num_env_frames
        num_iterations_since = iterations.numpy() - last_log_iterations
        elapsed_time = time.time() - last_log_time
        tf.summary.scalar(
            'steps_summary/num_environment_frames_per_sec',
            tf.cast(num_frames_since, tf.float32) / elapsed_time,
            step=iterations)
        tf.summary.scalar(
            'steps_summary/num_iterations_per_sec',
            tf.cast(num_iterations_since, tf.float32) / elapsed_time,
            step=iterations)
        tf.summary.scalar('queue_size', queue.size(), step=iterations)
        tf.summary.scalar(
            'learning_rate',
            optimizer._decayed_lr(var_dtype=tf.float32),  
            step=iterations)
        last_log_num_env_frames, last_log_iterations, last_log_time = (
            num_env_frames, iterations.numpy(), time.time())
        logging.info('Number of environment frames: %d', num_env_frames)

      problem_type.create_summary(step=iterations, info=info)

  # Finishing up.
  ckpt_manager.save(checkpoint_number=iterations)
  queue.close()
  server.shutdown()


def run(problem_type: framework_problem_type.ProblemType):
  """Runs the learner with the given problem type."""

  iter_frame_ratio = FLAGS.batch_size * FLAGS.unroll_length
  final_iteration = int(
      math.ceil(FLAGS.total_environment_frames / iter_frame_ratio))
  hparams = {}
  hparams['logdir'] = FLAGS.logdir
  hparams['iter_frame_ratio'] = iter_frame_ratio
  hparams['final_iteration'] = final_iteration
  hparams['warm_start_ckpt'] = FLAGS.warm_start_ckpt
  run_with_address(problem_type, FLAGS.server_address, hparams)
