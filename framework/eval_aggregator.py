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

"""Evaluation aggregator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import os
import pickle
import random

from absl import logging
import numpy as np
from seed_rl import grpc
from six.moves import queue as py_queue
import tensorflow.compat.v2 as tf
from valan.framework import common


StepSummaries = collections.namedtuple('StepSummaries',
                                       ('step', 'count', 'metrics_sum'))
NUM_EVAL_SAMPLES = 20


def update_summary(step_summaries, episode_metrics_dict):
  """Update the step summaries."""
  metrics_sum = step_summaries.metrics_sum
  for key, value in episode_metrics_dict.items():
    if key.endswith(common.VISUALIZATION_IMAGES):
      metrics_sum[key].extend(value)
      random.shuffle(metrics_sum[key])
      if len(metrics_sum[key]) > NUM_EVAL_SAMPLES:
        metrics_sum[key] = metrics_sum[key][:NUM_EVAL_SAMPLES]
    elif key.endswith(common.AUC):
      metrics_sum[key].extend(value)
    else:
      metrics_sum[key] += value
  return StepSummaries(
      step=step_summaries.step,
      count=step_summaries.count + 1,
      metrics_sum=metrics_sum)


def run_with_address(listen_address, hparams):
  """Run aggregator on the given server address.

  Args:
    listen_address: The network address on which to listen for eval actors.
    hparams: A dict containing hyperparameter settings.
  """
  eval_queue = py_queue.Queue(maxsize=8192)

  logging.info('Creating gRPC server on address %s', listen_address)
  server = grpc.Server([listen_address])

  @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
  def eval_enqueue(x):
    tf.py_function(lambda x: eval_queue.put(pickle.loads(x.numpy())), [x], [])
    return []

  server.bind(eval_enqueue)

  logging.info('Starting gRPC server')
  server.start()

  summary_writer = tf.summary.create_file_writer(
      hparams['logdir'], flush_millis=20000, max_queue=1000)
  with summary_writer.as_default():
    # Get first element
    episode_metrics_dict = eval_queue.get()
    current_step_sum = StepSummaries(
        step=episode_metrics_dict.pop(common.STEP),
        count=1,
        metrics_sum=copy.copy(episode_metrics_dict))
    next_step_sum = None
    num_samples = 0
    while num_samples < hparams['num_samples'] or hparams['num_samples'] == -1:
      episode_metrics_dict = eval_queue.get()
      num_samples += 1
      step = episode_metrics_dict.pop(common.STEP)
      if step < current_step_sum.step:
        # Skip and warning
        logging.warning(
            'Skipping old summary for step: %d while current step is %d.', step,
            current_step_sum.step)
        continue
      if step == current_step_sum.step:
        current_step_sum = update_summary(current_step_sum,
                                          episode_metrics_dict)
      elif step > current_step_sum.step:
        if not next_step_sum:
          next_step_sum = StepSummaries(
              step=step, count=1, metrics_sum=copy.copy(episode_metrics_dict))
        elif step == next_step_sum.step:
          next_step_sum = update_summary(next_step_sum, episode_metrics_dict)
        else:
          logging.info('Printing summaries for step %d, summaries count is %d',
                       current_step_sum.step, current_step_sum.count)
          # find prefix (if applicable) from metric names.
          one_key = next(iter(current_step_sum.metrics_sum.keys()))
          idx = one_key.rfind('/')
          metrics_prefix = one_key[:idx] if idx >= 0 else ''
          current_step = int(current_step_sum.step)
          tf.summary.scalar(
              '{}/summary_count'.format(metrics_prefix),
              float(current_step_sum.count),
              step=current_step)
          for key, value in current_step_sum.metrics_sum.items():
            if key.endswith(common.VISUALIZATION_IMAGES):
              logging.info('Visualizing images....')
              images = np.stack(value)
              tf.summary.image(
                  key,
                  images,
                  max_outputs=NUM_EVAL_SAMPLES,
                  step=current_step)
            elif key.endswith(common.AUC):
              predictions = [v[0][0] for v in value]
              labels = [v[1] for v in value]
              auc = tf.keras.metrics.AUC()
              auc.update_state(labels, predictions)
              tf.summary.scalar(
                  key,
                  float(auc.result().numpy()),
                  step=current_step)
              tf.summary.scalar(
                  key.replace(common.AUC, 'average_label'),
                  np.mean(labels),
                  step=current_step)
              # Histograms.
              tf.summary.histogram(
                  'Predictions/hist_all',
                  predictions,
                  step=current_step,
                  buckets=50)
              if np.sum(np.array(labels) == 1) > 0:
                tf.summary.histogram(
                    'Predictions/hist_GT',
                    [p for p, l in zip(predictions, labels) if l == 1],
                    step=current_step,
                    buckets=50)
              if np.sum(np.array(labels) == 0) > 0:
                tf.summary.histogram(
                    'Predictions/hist_synthetic',
                    [p for p, l in zip(predictions, labels) if l == 0],
                    step=current_step,
                    buckets=50)
            else:
              metric_avg = value / float(current_step_sum.count)
              logging.info('Key %s, value %f', key, metric_avg)
              tf.summary.scalar(
                  key, metric_avg, step=int(current_step_sum.step))
          # Swap
          current_step_sum = copy.copy(next_step_sum)
          next_step_sum = StepSummaries(
              step=step, count=1, metrics_sum=copy.copy(episode_metrics_dict))


def run(
    aggregator_prefix: str = 'default',
    logdir: str = '/tmp/agent/',
    server_address: str = 'unix:/tmp/foo',
):
  """Run the eval_aggregator."""
  tf.enable_v2_behavior()
  hparams = {}
  if aggregator_prefix:
    logdir = os.path.join(logdir, aggregator_prefix)
  hparams['logdir'] = logdir
  hparams['num_samples'] = -1
  run_with_address(server_address, hparams)
