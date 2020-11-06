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

"""Different loss functions supported by VALAN framework."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from absl import flags
import tensorflow.compat.v2 as tf
from valan.framework import common
from valan.framework import utils
from valan.framework import vtrace
from valan.framework.focal_loss import focal_loss_from_logits

FLAGS = flags.FLAGS


def _compute_baseline_loss(advantages, step):
  # Loss for the baseline, summed over the time dimension. Multiply by 0.5 to
  # match the standard update rule:
  #   d(loss) / d(baseline) = advantage
  baseline_cost = .5 * tf.square(advantages)
  tf.summary.scalar(
      'loss/baseline_cost', tf.reduce_mean(baseline_cost), step=step)
  return baseline_cost


def _compute_entropy_loss(logits, step):
  policy = tf.nn.softmax(logits)
  log_policy = tf.nn.log_softmax(logits)
  entropy = -tf.reduce_mean(-policy * log_policy, axis=-1)
  tf.summary.scalar('loss/entropy', tf.reduce_mean(entropy), step=step)
  return entropy


def _compute_policy_gradient_loss(logits, actions, advantages, step):
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=actions, logits=logits)
  advantages = tf.stop_gradient(advantages)
  policy_gradient_loss = cross_entropy * advantages
  tf.summary.scalar(
      'loss/policy_gradient_loss',
      tf.reduce_mean(policy_gradient_loss),
      step=step)
  return policy_gradient_loss


def get_ac_loss(learner_agent_output, env_output, actor_agent_output,
                actor_action, reward_clipping, discounting, baseline_cost,
                entropy_cost, num_steps):
  """Computes actor-critic loss.

  Args:
    learner_agent_output: A nested structure of type `AgentOutput`. The tensors
      are expected to have shape [num_timesteps, batch, ....]
    env_output: A nested structure of type `EnvOutput`. The tensors are expected
      to have shape [num_timesteps, batch, ...].
    actor_agent_output: A nested structure of type `AgentOutput`. The tensors
      are expected to have shape [num_timesteps, batch, ....]
    actor_action: An instance of `ActorAction` containing indices of the actions
      chosen by actor. The total number of actions available to actor at any
      point is equal to actor_agent_output.policy_logits.shape()[-1].
    reward_clipping: A string denoting the clipping strategy to be applied to
      rewards. An empty string means no clipping is applied.
    discounting: The discount factor.
    baseline_cost: A multiplier for baseline loss.
    entropy_cost: A multiplier for entropy.
    num_steps: An int to be used as step arg for summaries.

  Returns:
    A tensor of shape [num_timesteps - 1, batch_size] which contains the
    computed actor-critic loss per timestep per element.
  """
  # Use last baseline value (from the value function) to bootstrap.
  bootstrap_value = learner_agent_output.baseline[-1]

  # At this point, the environment outputs at time step `t` are the inputs
  # that lead to the learner_outputs at time step `t`. After the following
  # shifting, the actions in actor_agent_output and learner_outputs at time step
  # `t` is what leads to the environment outputs at time step `t`.
  actor_agent_output = tf.nest.map_structure(lambda t: t[1:],
                                             actor_agent_output)
  rewards, done, _, _ = tf.nest.map_structure(lambda t: t[1:], env_output)
  actor_action_idx = actor_action.chosen_action_idx[1:]
  learner_agent_output = tf.nest.map_structure(lambda t: t[:-1],
                                               learner_agent_output)

  clipped_rewards = rewards
  if reward_clipping == 'abs_one':
    clipped_rewards = tf.clip_by_value(rewards, -1, 1)
  elif reward_clipping == 'soft_asymmetric':
    squeezed = tf.tanh(rewards / 5.0)
    # Negative rewards are given less weight than positive rewards.
    clipped_rewards = tf.where(rewards < 0, .3 * squeezed, squeezed) * 5.

  discounts = tf.cast(~done, tf.float32) * discounting

  # Compute V-trace returns and weights.
  vtrace_returns = vtrace.from_logits(
      behaviour_policy_logits=actor_agent_output.policy_logits,
      target_policy_logits=learner_agent_output.policy_logits,
      actions=actor_action_idx,
      discounts=discounts,
      rewards=clipped_rewards,
      values=learner_agent_output.baseline,
      bootstrap_value=bootstrap_value)

  pg_advantages = vtrace_returns.pg_advantages
  v_advantages = vtrace_returns.vs - learner_agent_output.baseline
  tf.summary.histogram('pg_advantages', pg_advantages, step=num_steps)
  tf.summary.histogram('vs', vtrace_returns.vs, step=num_steps)
  tf.summary.histogram('baseline', learner_agent_output.baseline,
                       step=num_steps)
  tf.summary.histogram('v_advantages', v_advantages, step=num_steps)

  # Compute loss as a weighted sum of the baseline loss, the policy gradient
  # loss and an entropy regularization term.
  pg_loss = _compute_policy_gradient_loss(
      learner_agent_output.policy_logits,
      actor_action_idx,
      pg_advantages,
      step=num_steps)
  baseline_loss = _compute_baseline_loss(v_advantages, step=num_steps)
  entropy = _compute_entropy_loss(
      learner_agent_output.policy_logits, step=num_steps)

  total_loss = pg_loss + baseline_cost * baseline_loss + entropy_cost * entropy
  tf.summary.scalar('loss/ac_loss', tf.reduce_mean(total_loss), step=num_steps)
  return total_loss


def get_cross_entropy_loss(learner_agent_output, env_output, actor_agent_output,
                           actor_action, reward_clipping, discounting,
                           baseline_cost, entropy_cost, num_steps):
  """Computes cross entropy loss."""
  del env_output
  del actor_agent_output
  del reward_clipping
  del discounting
  del baseline_cost
  del entropy_cost

  # Align learner output and actor output.
  # NOTE that for a tensor of num_timesteps=3, learner output has output at
  # timesteps [t1, t2, t3] while actor output has output at timesteps [t0, t1,
  # t2]. Hence the need to align before computing cross-entropy loss.
  policy_logits = learner_agent_output.policy_logits[:-1]
  target_actions = actor_action.oracle_next_action_idx[1:]

  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=target_actions, logits=policy_logits)
  tf.summary.scalar(
      'loss/cross_entropy_loss', tf.reduce_mean(cross_entropy), step=num_steps)
  return cross_entropy


def _get_discriminator_logits(learner_agent_output, env_output,
                              actor_agent_output, actor_action, reward_clipping,
                              discounting, baseline_cost, entropy_cost,
                              num_steps):
  """Discriminator loss."""
  del actor_agent_output
  del actor_action
  del reward_clipping
  del discounting
  del baseline_cost
  del entropy_cost

  first_true = utils.get_first_true_column(env_output.observation['disc_mask'])
  # Shape of output_logits:[time, batch].
  output_logits = learner_agent_output.policy_logits
  # Shape of output_logits:[batch].
  output_logits = tf.boolean_mask(output_logits, first_true)
  output_affine_a, output_affine_b = learner_agent_output.baseline

  # Get the first true.
  labels = tf.cast(env_output.observation['label'], tf.float32)
  tf.summary.scalar(
      'labels/mean_labels before masking',
      tf.reduce_mean(labels),
      step=num_steps)
  # Shape of labels:[batch].
  labels = tf.boolean_mask(labels, first_true)

  positive_label = tf.equal(labels, tf.constant(1.0))
  positive_logits = tf.boolean_mask(output_logits, positive_label)
  tf.summary.histogram(
      'distribution/sigmoid_positive_logits',
      tf.sigmoid(positive_logits),
      step=num_steps)
  tf.summary.histogram(
      'distribution/positive_logits', positive_logits, step=num_steps)

  negative_label = tf.equal(labels, tf.constant(0.0))
  negative_logits = tf.boolean_mask(output_logits, negative_label)
  tf.summary.histogram(
      'distribution/sigmoid_negative_logits',
      tf.sigmoid(negative_logits),
      step=num_steps)
  tf.summary.histogram(
      'distribution/negative_logits', negative_logits, step=num_steps)
  tf.summary.scalar(
      'labels/positive_label_ratio',
      tf.reduce_mean(tf.cast(positive_label, tf.float32)),
      step=num_steps)
  tf.summary.scalar(
      'affine_transform/a', tf.reduce_mean(output_affine_a), step=num_steps)
  tf.summary.scalar(
      'affine_transform/b', tf.reduce_mean(output_affine_b), step=num_steps)
  # Shape: [batch]
  return labels, output_logits


def get_discriminator_loss(learner_agent_output, env_output, actor_agent_output,
                           actor_action, reward_clipping, discounting,
                           baseline_cost, entropy_cost, num_steps):
  """Discriminator loss."""
  labels, output_logits = _get_discriminator_logits(
      learner_agent_output, env_output, actor_agent_output, actor_action,
      reward_clipping, discounting, baseline_cost, entropy_cost, num_steps)
  cross_entropy = tf.nn.weighted_cross_entropy_with_logits(
      labels=labels, logits=output_logits, pos_weight=5)
  return cross_entropy


def get_discriminator_focal_loss(learner_agent_output, env_output,
                                 actor_agent_output, actor_action,
                                 reward_clipping, discounting, baseline_cost,
                                 entropy_cost, num_steps):
  """Discriminator focal loss."""
  # Check if learner_agent_output has logits and labels prepared already,
  # otherwise filter and prepare the logits and labels.
  if (isinstance(learner_agent_output.policy_logits, dict) and
      'labels' in learner_agent_output.policy_logits and
      tf.is_tensor(learner_agent_output.baseline)):
    # Shape: [batch]
    labels = learner_agent_output.policy_logits['labels']
    output_logits = learner_agent_output.baseline
    tf.debugging.assert_equal(tf.shape(labels), tf.shape(output_logits))
    loss_tag = 'loss/focal_loss'
  else:
    # labels and output_logits have shape: [batch].
    labels, output_logits = _get_discriminator_logits(
        learner_agent_output, env_output, actor_agent_output, actor_action,
        reward_clipping, discounting, baseline_cost, entropy_cost, num_steps)
    loss_tag = 'loss/focal_loss (w/ softmin_softmax)'

  # Shape = [batch]
  fl, ce = focal_loss_from_logits(
      output_logits, labels, alpha=FLAGS.focal_loss_alpha,
      gamma=FLAGS.focal_loss_gamma, normalizer=FLAGS.focal_loss_normalizer)
  tf.summary.scalar(loss_tag, tf.reduce_mean(fl), step=num_steps)
  tf.summary.scalar(
      'loss/CE (reference only)', tf.reduce_mean(ce), step=num_steps)
  tf.summary.scalar(
      'labels/num_labels_per_batch', tf.size(labels), step=num_steps)
  tf.summary.scalar(
      'labels/mean_labels', tf.reduce_mean(labels), step=num_steps)
  return fl


def get_discriminator_batch_loss(learner_agent_output, env_output,
                                 unused_actor_agent_output, unused_actor_action,
                                 unused_reward_clipping, unused_discounting,
                                 unused_baseline_cost,
                                 unused_entropy_cost, num_steps):
  """Discriminator batch softmax loss with mask."""
  # Remove the time_step dimension for each tensor in the result.
  learner_agent_output = tf.nest.map_structure(lambda t: tf.squeeze(t, axis=0),
                                               learner_agent_output)
  result = learner_agent_output.policy_logits  # dict

  # Compute softmax.
  # Use stable softmax: softmax(x) = softmax(x+c) for any constant c.
  # Here we use constant c = max(-x).
  # Shape of similarity and similarity_mask: [batch, batch].
  row_max = tf.reduce_max(result['similarity'], axis=1, keepdims=True)
  masked_row_exp = tf.exp(result['similarity'] - row_max) * tf.cast(
      result['similarity_mask'], tf.float32)
  summed_rows = tf.reduce_sum(masked_row_exp, axis=1)  # Shape=[batch]
  # log(softmax_i). Shape = [batch]
  loss_by_row = -(tf.linalg.diag_part(result['similarity']) -
                  tf.squeeze(row_max, 1)) + tf.math.log(summed_rows)
  loss_by_row = loss_by_row * result['labels']

  col_max = tf.reduce_max(result['similarity'], axis=0, keepdims=True)
  masked_col_exp = tf.exp(result['similarity'] - col_max) * tf.cast(
      result['similarity_mask'], tf.float32)
  summed_cols = tf.reduce_sum(masked_col_exp, axis=0)  # Shape=[batch]
  tf.debugging.assert_equal(summed_cols.shape, summed_rows.shape)
  # log(softmax_j). Shape = [batch]
  loss_by_col = -(tf.linalg.diag_part(result['similarity']) -
                  tf.squeeze(col_max, 0)) + tf.math.log(summed_cols)
  loss_by_col = loss_by_col * result['labels']

  # Shape = [batch]
  loss = (loss_by_row + loss_by_col) / 2.0

  tf.summary.scalar('loss/batch_softmax', tf.reduce_mean(loss), step=num_steps)
  tf.summary.scalar('labels/num_positive_labels',
                    tf.reduce_sum(result['labels']), step=num_steps)
  tf.summary.scalar('labels/batch_loss_positive_label_ratio',
                    tf.reduce_mean(result['labels']), step=num_steps)
  # Add classification loss if set in FLAGS. Shape = [batch].
  if FLAGS.use_batch_and_ce_losses:
    classification_loss = get_discriminator_focal_loss(
        learner_agent_output, env_output, unused_actor_agent_output,
        unused_actor_action, unused_reward_clipping, unused_discounting,
        unused_baseline_cost, unused_entropy_cost, num_steps)
    # Shape = [batch].
    loss = classification_loss + loss * FLAGS.disc_batch_loss_scale
  return loss


LOSS_FNS_REGISTRY = {
    common.AC_LOSS: get_ac_loss,
    common.CE_LOSS: get_cross_entropy_loss,
    common.DCE_LOSS: get_discriminator_loss,
    common.DCE_FOCAL_LOSS: get_discriminator_focal_loss,
    common.DISC_BATCH_LOSS: get_discriminator_batch_loss,
}


def compute_loss(study_loss_types, current_batch_loss_type, agent, agent_state,
                 env_output, actor_agent_output, actor_action, num_steps):
  """Computes loss using given loss type of the batch.

  Args:
    study_loss_types: An iterable of all loss types used in the present study.
    current_batch_loss_type: A tensor of shape [batch_size] with each value a
      valid loss type for the study.
    agent: An instance of `BaseAgent`.
    agent_state: The initial state of the agent.
    env_output: A nested structure of type `EnvOutput`. The tensors are expected
      to have shape [num_timesteps, batch, ...].
    actor_agent_output: A nested structure of type `AgentOutput`. The tensors
      are expected to have shape [num_timesteps, batch, ....]
    actor_action: An instance of `ActorAction` containing indices of the actions
      chosen by actor. The total number of actions available to actor at any
      point is equal to actor_agent_output.policy_logits.shape()[-1].
    num_steps: An int to be used as step arg for summaries.

  Returns:
    A scalar tensor with computed loss.
  """
  learner_agent_output, _ = agent(env_output, agent_state)
  losses_dict = {}

  for loss_type in study_loss_types:
    losses_dict[loss_type] = LOSS_FNS_REGISTRY[loss_type](
        learner_agent_output, env_output, actor_agent_output, actor_action,
        FLAGS.reward_clipping, FLAGS.discounting, FLAGS.baseline_cost,
        FLAGS.entropy_cost, num_steps)
  # All the losses are time-major tensors, i.e., of shape [num_timesteps - 1,
  # batch_size]. Convert to batch-major and then choose which loss to apply to
  # each row.
  losses_dict = tf.nest.map_structure(utils.transpose_batch_time, losses_dict)
  loss = tf.reduce_mean(
      utils.gather_from_dict(losses_dict, current_batch_loss_type))
  # Total loss including regularizer losses.
  regularizers_loss = tf.add_n(agent.losses)
  total_loss = loss + regularizers_loss
  tf.summary.scalar('loss/task_loss', loss, step=num_steps)
  tf.summary.scalar('loss/regularizer_loss', regularizers_loss, step=num_steps)
  tf.summary.scalar('loss/total_loss', loss, step=num_steps)
  return total_loss
