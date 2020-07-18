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

"""SDR model training with TensorFlow eager execution."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os



from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v2 as tf
import numpy as np
from valan.touchdown.sdr import lingunet


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "train_input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "dev_input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "test_input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string("model_dir", "/tmp/tensorflow/generalization/checkpoints/",
                    "Directory to write TensorBoard summaries")

flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")

flags.DEFINE_integer(
    "batch_size", 64, "Batch size for training and evaluation. When using "
    "multiple gpus, this is the global batch size for "
    "all devices. For example, if the batch size is 32 "
    "and there are 4 GPUs, each GPU will get 8 examples on "
    "each step.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_boolean("use_tpu", True, "Name of the TPU to use.")
flags.DEFINE_string("tpu", None, "Name of the TPU to use.")
# copybara:strip_begin
flags.DEFINE_integer("vm_config", 2, "Running in 2VM or 1VM mode.")
# copybara:strip_end

NUM_PERSPECTIVE_IMAGES = 8
DOWNSCALED_IMAGE_HEIGHT = 100
DOWNSCALED_IMAGE_WIDTH = 58
DOWNSCALED_PANO_HEIGHT = DOWNSCALED_IMAGE_HEIGHT
DOWNSCALED_PANO_WIDTH = DOWNSCALED_IMAGE_WIDTH * NUM_PERSPECTIVE_IMAGES
NUM_CHANNELS = 128

NUM_EXAMPLES_TRAIN = 17000
NUM_EXAMPLES_EVAL = 3835
NUM_EPOCHS = 15

RESOLUTION_MULTIPLIER = 8


# Increase num_cpu_threads if the perf is input bound.
def get_batched_dataset(pattern,
                        max_seq_length,
                        batch_size,
                        is_training=True,
                        num_cpu_threads=64):
  """tf.data.Dataset object for MNIST training data."""
  input_files = tf.io.gfile.glob(pattern)
  logging.info("*** Input Files ***")
  for input_file in input_files:
    logging.info("  %s", input_file)

  # For training, we want a lot of parallel reading and shuffling.
  # For eval, we want no shuffling and parallel reading doesn"t matter.
  if is_training:
    d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
    d = d.repeat()
    d = d.shuffle(buffer_size=len(input_files))

    # `cycle_length` is the number of parallel files that get read.
    cycle_length = min(num_cpu_threads, len(input_files))

    # `sloppy` mode means that the interleaving is not exact. This adds
    # even more randomness to the training pipeline.
    d = d.apply(
        tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset,
            sloppy=is_training,
            cycle_length=cycle_length))
    d = d.shuffle(buffer_size=128)
  else:
    d = tf.data.TFRecordDataset(input_files)
    # Since we evaluate for a fixed number of steps we don"t want to encounter
    # out-of-range exceptions.
    d = d.repeat()

  # Create a description of the features.
  feature_description = {
      "input_ids":
          tf.io.FixedLenFeature([max_seq_length], tf.int64),
      "input_ids_length":
          tf.io.FixedLenFeature([], tf.int64),
      "pano_features":
          tf.io.FixedLenFeature(
              [DOWNSCALED_PANO_HEIGHT * DOWNSCALED_PANO_WIDTH * NUM_CHANNELS],
              tf.float32),
      "target_features":
          tf.io.FixedLenFeature(
              [DOWNSCALED_PANO_HEIGHT * DOWNSCALED_PANO_WIDTH], tf.float32),
  }

  def parse_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    features = tf.io.parse_single_example(example_proto, feature_description)
    features["pano_features"] = tf.reshape(
        features["pano_features"],
        [DOWNSCALED_PANO_HEIGHT, DOWNSCALED_PANO_WIDTH, NUM_CHANNELS])
    return features

  # We must `drop_remainder` on training because the TPU requires fixed
  # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
  # and we *don"t* want to drop the remainder, otherwise we wont cover
  # every sample.
  d = d.apply(
      tf.data.experimental.map_and_batch(
          parse_function,
          batch_size=batch_size,
          num_parallel_batches=num_cpu_threads,
          drop_remainder=True))
  # Prefetch 1 batch.
  d = d.prefetch(1)
  return d


def distance_metric(preds, targets):
  """Calculate distances between model predictions and targets within a batch."""
  batch_size = preds.shape[0]
  preds = tf.reshape(
      preds, [batch_size, DOWNSCALED_PANO_HEIGHT, DOWNSCALED_PANO_WIDTH])
  targets = tf.reshape(
      targets, [batch_size, DOWNSCALED_PANO_HEIGHT, DOWNSCALED_PANO_WIDTH])
  distances = []
  for pred, target in zip(preds, targets):
    pred_coord = np.unravel_index(np.argmax(pred), pred.shape)
    target_coord = np.unravel_index(np.argmax(target), target.shape)
    dist = np.sqrt((target_coord[0] - pred_coord[0])**2 +
                   (target_coord[1] - pred_coord[1])**2)
    dist = dist * RESOLUTION_MULTIPLIER
    distances.append(dist)
  return distances


def accuracy(distances, margin=10):
  """Calculating accuracy at 80 pixel by default."""
  num_correct = 0
  for distance in distances:
    num_correct = num_correct + 1 if distance < margin else num_correct
  return num_correct / len(distances)


def get_features(features):
  input_ids = features["input_ids"]
  input_ids_length = features["input_ids_length"]
  pano_features = features["pano_features"]
  target_features = features["target_features"]
  return input_ids, input_ids_length, pano_features, target_features


def run_eager():
  """Run MNIST training and eval loop in eager mode."""
  tf.enable_v2_behavior()

  if FLAGS.use_tpu:
    job_name = "worker"
    primary_cpu_task = "/job:%s" % job_name
    logging.info("Use TPU at %s",
                 FLAGS.tpu if FLAGS.tpu is not None else "local")
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
  else:
    primary_cpu_task = "/task:0"
    strategy = tf.distribute.MirroredStrategy()
  logging.info("Distribution strategy: %s", strategy)

  model_dir = FLAGS.model_dir
  logging.info("Saving checkpoints at %s", model_dir)

  steps_per_epoch = int(NUM_EXAMPLES_TRAIN // FLAGS.batch_size)
  steps_per_eval = int(NUM_EXAMPLES_EVAL // FLAGS.batch_size)

  with tf.device(primary_cpu_task):

    def get_dataset_fn(input_file):
      def dataset_fn(input_context):
        batch_size = input_context.get_per_replica_batch_size(FLAGS.batch_size)
        d = get_batched_dataset(input_file, FLAGS.max_seq_length, batch_size)
        return d.shard(input_context.num_input_pipelines,
                       input_context.input_pipeline_id)

      return dataset_fn

    train_dataset = strategy.experimental_distribute_datasets_from_function(
        get_dataset_fn(FLAGS.train_input_file))
    test_dataset = strategy.experimental_distribute_datasets_from_function(
        get_dataset_fn(FLAGS.test_input_file))
    dev_dataset = strategy.experimental_distribute_datasets_from_function(
        get_dataset_fn(FLAGS.dev_input_file))

    with strategy.scope():
      # Create the model and optimizer
      model = lingunet.LingUNet(num_channels=NUM_CHANNELS)

      for var in model.trainable_variables:
        tf.logging.info("  name = %s, shape = %s", var.name, var.shape)

      optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
      training_loss = tf.keras.metrics.Mean("training_loss", dtype=tf.float32)
      test_loss = tf.keras.metrics.Mean("test_loss", dtype=tf.float32)
      logging.info("Finished building Keras LingUNet model")
      checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
      latest_checkpoint = tf.train.latest_checkpoint(model_dir)
      initial_epoch = 0

      if latest_checkpoint:
        # checkpoint.restore must be within a strategy.scope() so that optimizer
        # slot variables are mirrored.
        checkpoint.restore(latest_checkpoint)
        logging.info("Loaded checkpoint %s", latest_checkpoint)
        initial_epoch = optimizer.iterations.numpy() // steps_per_epoch

    # Create summary writers
    train_summary_writer = tf.summary.create_file_writer(
        os.path.join(model_dir, "summaries/train"))
    test_summary_writer = tf.summary.create_file_writer(
        os.path.join(model_dir, "summaries/test"))
    dev_summary_writer = tf.summary.create_file_writer(
        os.path.join(model_dir, "summaries/dev"))

    @tf.function
    def train_step(iterator):
      """Training StepFn."""

      def step_fn(inputs):
        """Per-Replica StepFn."""
        input_ids, input_ids_length, pano_features, target_features = get_features(
            inputs)
        with tf.GradientTape() as tape:
          predicted_targets = model(pano_features, input_ids, input_ids_length,
                                    True)
          loss = tf.keras.losses.kullback_leibler_divergence(
              target_features, predicted_targets)
          loss = tf.reduce_mean(loss) / strategy.num_replicas_in_sync
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        training_loss.update_state(loss)
        return predicted_targets, target_features

      predicted_targets, target_features = strategy.run(
          step_fn, args=(next(iterator),))
      predicted_targets = tf.concat(
          strategy.experimental_local_results(predicted_targets), axis=0)
      target_features = tf.concat(
          strategy.experimental_local_results(target_features), axis=0)
      return predicted_targets, target_features

    @tf.function
    def test_step(iterator):
      """Evaluation StepFn."""

      def step_fn(inputs):
        input_ids, input_ids_length, pano_features, target_features = get_features(
            inputs)
        predicted_targets = model(pano_features, input_ids, input_ids_length,
                                  False)
        loss = tf.keras.losses.kullback_leibler_divergence(
            target_features, predicted_targets)
        loss = tf.reduce_mean(loss) / strategy.num_replicas_in_sync
        test_loss.update_state(loss)
        return predicted_targets, target_features

      predicted_targets, target_features = strategy.run(
          step_fn, args=(next(iterator),))
      predicted_targets = tf.concat(
          strategy.experimental_local_results(predicted_targets), axis=0)
      target_features = tf.concat(
          strategy.experimental_local_results(target_features), axis=0)
      return predicted_targets, target_features

    train_iterator = iter(train_dataset)
    for epoch in range(initial_epoch, NUM_EPOCHS):
      logging.info("Starting to run epoch: %s", epoch)
      with train_summary_writer.as_default():
        for step in range(steps_per_epoch):
          if step % 5 == 0:
            logging.info("Running step %s in epoch %s", step, epoch)
          predicted_targets, target_features = train_step(train_iterator)
          distances = distance_metric(predicted_targets, target_features)
          mean_dist = np.mean(distances)
          tf.summary.scalar(
              "mean_dist", mean_dist, step=optimizer.iterations)
          tf.summary.scalar(
              "loss", training_loss.result(), step=optimizer.iterations)
          tf.summary.scalar(
              "accuracy@40",
              accuracy(distances, margin=40),
              step=optimizer.iterations)
          tf.summary.scalar(
              "accuracy@80",
              accuracy(distances, margin=80),
              step=optimizer.iterations)
          tf.summary.scalar(
              "accuracy@120",
              accuracy(distances, margin=120),
              step=optimizer.iterations)
          training_loss.reset_states()

      test_iterator = iter(test_dataset)
      dev_iterator = iter(dev_dataset)

      for iterator, summary_writer in zip(
          [test_iterator, dev_iterator],
          [test_summary_writer, dev_summary_writer]):
        with summary_writer.as_default():
          eval_distances = []
          for step in range(steps_per_eval):
            if step % 5 == 0:
              logging.info("Starting to run eval step %s of epoch: %s", step,
                           epoch)
            predicted_targets, target_features = test_step(iterator)
            distances = distance_metric(predicted_targets, target_features)
            eval_distances.extend(distances)
          tf.summary.scalar(
              "mean_dist", np.mean(eval_distances), step=optimizer.iterations)
          tf.summary.scalar(
              "loss", test_loss.result(), step=optimizer.iterations)
          tf.summary.scalar(
              "accuracy@40",
              accuracy(eval_distances, margin=40),
              step=optimizer.iterations)
          tf.summary.scalar(
              "accuracy@80",
              accuracy(eval_distances, margin=80),
              step=optimizer.iterations)
          tf.summary.scalar(
              "accuracy@120",
              accuracy(eval_distances, margin=120),
              step=optimizer.iterations)
          test_loss.reset_states()

      checkpoint_name = checkpoint.save(os.path.join(model_dir, "checkpoint"))
      logging.info("Saved checkpoint to %s", checkpoint_name)


def main(_):
  run_eager()


if __name__ == "__main__":
  app.run(main)
