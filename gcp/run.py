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

"""Script to launch jobs using GCP's AI Platform Training tool.

To learn more about AI Platform Training, see:
https://cloud.google.com/ai-platform/training/docs/overview

The type of jobs to launch (e.g., learner, train actor, eval actor, or eval
aggregator) depends on the task type defined in the global variable
`CLUSTER_SPEC` set by the AI Platform when the job is submitted.
See schema here:
https://cloud.google.com/ai-platform/training/docs/distributed-training-containers#about-cluster-spec

For instance, the `CLUSTER_SPEC` for an eval worker looks like this:
{
    'cluster': {
        'master': ['localhost:123', 'localhost:456'],
        'worker': ['localhost:1234', 'localhost:3456'],
        'evaluator': ['localhost:12345', 'localhost:23456'],
    },
    'task': {'type': 'evaluator', 'index': 0}
}

Each full training job consists of a learner (the main training node), a group
of train actors that process and enqueue data for the learner concurrently, as
well as several sets of eval actors for the evaluation job, all of which run
asynchronously with the learner.  The evaluation job can have multiple data
sources, for instance, "train", "val_seen", and "val_unseen" for the training
set, the "val_seen" set, and "val_unseen" set respectively. Each of them has its
own group of actors plus one eval aggregator that dequeues and aggregates
evaluated examples from all eval actors within this group.

The learner always runs on the "master" node with an accelerator. The train
actors run on the "worker" nodes and use multi-threading to concurrently load
and process the input data. All eval actors and eval aggregators run on the
"evaluator" nodes. The learner runs on accelerator and all actors run on
CPUs.
"""

import concurrent.futures
import json
import os
import subprocess
import sys
from typing import List

from absl import app
from absl import flags
from absl import logging
from valan.framework import hyperparam_flags  
from valan.r2r import custom_flags  


flags.DEFINE_integer('num_train_workers', 1,
                     'Number of workers for the train actor.')
flags.DEFINE_integer('actors_per_train_worker', 1,
                     'Number of actors to run on a single train worker.')
flags.DEFINE_spaceseplist(
    'eval_data_source', '',
    'A space-separated list of sources to read the data from. This is usually '
    'name(s) of the eval splits from which the actor reads the data, e.g., '
    ' "val_seen val_unseen". NOTE: If each set of eval source contains '
    'multiple files, they can be separated by commas, e.g., '
    ' "val_seen_1,val_seen2 val_unseen_1,val_unseen2" ')
flags.DEFINE_spaceseplist(
    'num_eval_workers', '1',
    'Space-separated number of workers for each eval_data_source.')
flags.DEFINE_integer('actors_per_eval_worker', 2,
                     'Number of actors to run on a single eval worker.')

FLAGS = flags.FLAGS


# Task specific dir where main functions (e.g., learner_main, actor_main) reside
TASK_DIR = {
    'R2R': 'r2r',
    'NDH': 'r2r',
    'R2R+NDH': 'r2r',
    'RxR': 'rxr',
    'touchdown': 'touchdown',
}


def _get_task_main_func(
    task_type: str,
    base_dir: str = '/valan',
) -> str:
  """Gets the main function path for learner, actor, or aggregator."""
  if FLAGS.problem not in TASK_DIR:
    raise ValueError(f'Task must be one of {TASK_DIR.keys()}. '
                     f'Not supported: {FLAGS.problem}.')
  func_path = os.path.join(base_dir, TASK_DIR[FLAGS.problem],
                           f'{task_type}_main.py')
  if not os.path.isfile(func_path):
    raise FileNotFoundError(f'Main function not found: {func_path}.')
  return func_path


def run_learner(
    executor: concurrent.futures.Executor,
    learner_address: str,
) -> concurrent.futures.Future:
  """Runs learner job using executor."""
  _, master_port = learner_address.split(':', 1)
  args = [
      'python',
      _get_task_main_func('learner'),
      f'--problem={FLAGS.problem}',
      f'--server_address=[::]:{master_port}',
      f'--agent_type={FLAGS.agent_type}',
      f'--scan_base_dir={FLAGS.scan_base_dir}',
      f'--data_base_dir={FLAGS.data_base_dir}',
      f'--vocab_dir={FLAGS.vocab_dir}',
      f'--vocab_file={FLAGS.vocab_file}',
      f'--image_features_dir={FLAGS.image_features_dir}',
      f'--logdir={FLAGS.logdir}',
  ]
  # Add additional input args if provided.
  if '--' in sys.argv:
    args.extend(sys.argv[sys.argv.index('--') + 1:])
  # Remove `--job-dir` passed from GCP.
  args = [arg for arg in args if '--job-dir' not in arg]
  logging.info('Submitting learner for problem %s with server port: %s',
               FLAGS.problem, master_port)
  logging.info('Learner args: %s', args)
  return executor.submit(subprocess.check_call, args)


def run_actor(
    executor: concurrent.futures.Executor,
    server_addr: str,
    worker_index: int,
    actor_id: int,
    data_source: str,
    mode: str,
    num_eval_workers: int = None,
) -> concurrent.futures.Future:
  """Runs train or eval actor job using executor."""
  if mode == 'train':
    num_tasks = FLAGS.num_train_workers * FLAGS.actors_per_train_worker
    actors_per_worker = FLAGS.actors_per_train_worker
  elif mode == 'eval':
    if num_eval_workers is None:
      raise ValueError('num_eval_workers must be provided. Got None.')
    actors_per_worker = FLAGS.actors_per_eval_worker
    num_tasks = num_eval_workers * FLAGS.actors_per_eval_worker
  else:
    raise ValueError(f'Mode not supported: {mode}')
  task_id = worker_index * actors_per_worker + actor_id

  args = [
      'python',
      _get_task_main_func('actor'),
      f'--mode={mode}',
      f'--problem={FLAGS.problem}',
      f'--agent_type={FLAGS.agent_type}',
      f'--data_source={data_source}',
      f'--server_address={server_addr}',
      f'--num_tasks={num_tasks}',
      f'--task={task_id}',
      f'--scan_base_dir={FLAGS.scan_base_dir}',
      f'--data_base_dir={FLAGS.data_base_dir}',
      f'--vocab_dir={FLAGS.vocab_dir}',
      f'--vocab_file={FLAGS.vocab_file}',
      f'--image_features_dir={FLAGS.image_features_dir}',
      f'--logdir={FLAGS.logdir}',
  ]
  # Add additional input args if exist.
  if '--' in sys.argv:
    args.extend(sys.argv[sys.argv.index('--') + 1:])
  # Remove `--job-dir` passed from GCP.
  args = [arg for arg in args if '--job-dir' not in arg]
  logging.info('Submitting actor for mode %s with client address: %s', mode,
               server_addr)
  logging.info('Actor args: %s', args)
  return executor.submit(subprocess.check_call, args)


def run_aggregator(
    executor: concurrent.futures.Executor,
    server_addr: str,
    aggregator_prefix: str,
) -> concurrent.futures.Future:
  """Runs the eval_aggregator using executor."""
  _, aggregator_port = server_addr.split(':', 1)
  args = [
      'python',
      _get_task_main_func('eval_aggregator'),
      f'--logdir={FLAGS.logdir}',
      f'--aggregator_prefix={aggregator_prefix}',
      f'--server_address=[::]:{aggregator_port}',
  ]
  logging.info('Submitting aggregator with server port: %s', aggregator_port)
  logging.info('agg args: %s', args)
  return executor.submit(subprocess.check_call, args)


def main(argv: List[str]) -> None:
  """Launches the learner, train actor, eval work, or eval aggregator."""
  if len(FLAGS.eval_data_source) != len(FLAGS.num_eval_workers):
    raise ValueError(
        'Length of eval_data_source must equal length of num_eval_workers.')
  num_eval_workers = [int(x) for x in FLAGS.num_eval_workers]
  logging.info('eval_data_source, num_eval_workers: %s, %s',
               FLAGS.eval_data_source, num_eval_workers)

  # Get environment var CLUSTER_SPEC set by the AI Platform.
  cluster_spec = os.environ.get('CLUSTER_SPEC', None)
  logging.info('CLUSTER_SPEC: %s', cluster_spec)

  cluster_config = json.loads(cluster_spec)
  task_type = cluster_config.get('task', {}).get('type')
  # Global worker index.
  worker_index = cluster_config.get('task', {}).get('index')
  logging.info('CLUSTER_SPEC: task_type: %s', task_type)
  logging.info('CLUSTER_SPEC: worker_index: %s', worker_index)

  # Get max_threads per worker for different tasks.
  if task_type == 'master':
    max_threads = 1
  elif task_type == 'worker':
    max_threads = FLAGS.actors_per_train_worker
  elif task_type == 'evaluator':
    max_threads = FLAGS.actors_per_eval_worker

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=max_threads) as executor:
    futures = []
    if task_type == 'master':
      learner_address = cluster_config.get('cluster').get('master')[0]
      futures.append(run_learner(executor, learner_address))

    elif task_type == 'worker':
      # Train actors.
      learner_address = cluster_config.get('cluster').get('master')[0]
      for actor_id in range(max_threads):
        # Runs multiple threads per worker concurrently.
        futures.append(
            run_actor(
                executor=executor,
                server_addr=learner_address,
                worker_index=worker_index,
                actor_id=actor_id,
                data_source=FLAGS.data_source,
                mode='train'))

    elif task_type == 'evaluator':
      # Each eval data source with index i in FLAGS.eval_data_source (e.g. a
      # list like ['val_seen', 'val_unseen']) has num_eval_workers[i]
      # workers and one corresponding aggregator.
      eval_source_ids = []  # indices for eval data sources
      internal_worker_ids_for_source = []  # internal id for each data source.
      aggregator_addresses = []
      all_addresses = cluster_config.get('cluster').get('evaluator')
      logging.info('num_eval_workers: %s', num_eval_workers)

      # Note that num_eval_workers is a list of integers.
      for source_id, num_workers in enumerate(num_eval_workers):
        eval_source_ids += [source_id] * (num_workers + 1)  # 1 for aggregator.
        # Internal worker index for each data source. `-1` means aggregator.
        internal_worker_ids_for_source += [-1] + list(range(num_workers))

      # Get aggregator addresses.
      aggregator_indices = [
          i for i, x in enumerate(internal_worker_ids_for_source) if x == -1
      ]
      aggregator_addrs = [all_addresses[i] for i in aggregator_indices]
      logging.info('aggregator_addrs: %s', aggregator_addrs)
      logging.info('worker_index: %s', worker_index)
      logging.info('eval_source_ids: %s', eval_source_ids)
      logging.info('internal_worker_ids_for_source: %s',
                   internal_worker_ids_for_source)

      current_aggregator_addr = aggregator_addrs[eval_source_ids[worker_index]]
      current_internal_worker_id = internal_worker_ids_for_source[worker_index]
      eval_data_source = FLAGS.eval_data_source[eval_source_ids[worker_index]]
      logging.info('Eval Data Source: %s', eval_data_source)

      if current_internal_worker_id == -1:
        # Aggregator
        futures.append(
            run_aggregator(
                executor=executor,
                server_addr=current_aggregator_addr,
                aggregator_prefix=eval_data_source))
      else:
        # Eval actor.
        current_num_workers = num_eval_workers[eval_source_ids[worker_index]]

        # Runs multiple threads per worker.
        for actor_id in range(max_threads):
          futures.append(
              run_actor(
                  executor=executor,
                  server_addr=current_aggregator_addr,
                  worker_index=current_internal_worker_id,
                  actor_id=actor_id,
                  data_source=eval_data_source,
                  mode='eval',
                  num_eval_workers=current_num_workers))

    # Now execute all Future tasks.
    for f in futures:
      f.result()


if __name__ == '__main__':
  app.run(main)
