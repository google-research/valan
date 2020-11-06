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

"""Common utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import time

from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow.io import gfile


def levenshtein(seq1, seq2):
  """Computes Levenshtein edit distance between two sequences.

  Adapted from online tutorial by Frank Hofmann.

  Args:
    seq1: The first sequence of any datatype that implements equality operator.
    seq2: The second sequence of any datatype that implements equality operator.

  Returns:
    Levenshtein edit distance (integer) between the two sequences.
  """
  size_x = len(seq1) + 1
  size_y = len(seq2) + 1
  mat = np.zeros((size_x, size_y))
  for x in range(size_x):
    mat[x, 0] = x
  for y in range(size_y):
    mat[0, y] = y

  for x in range(1, size_x):
    for y in range(1, size_y):
      if seq1[x-1] == seq2[y-1]:
        mat[x, y] = min(
            mat[x-1, y] + 1,
            mat[x-1, y-1],
            mat[x, y-1] + 1
        )
      else:
        mat[x, y] = min(
            mat[x-1, y] + 1,
            mat[x-1, y-1] + 1,
            mat[x, y-1] + 1
            )

  return mat[size_x - 1, size_y - 1]


def stack_nested_tensors(list_of_nests):
  """Stack a list of nested tensors.

  Args:
    list_of_nests: A list of nested tensors (or numpy arrays) of the same
      shape/structure.

  Returns:
    A nested array containing batched items, where each batched item is obtained
    by stacking corresponding items from the list of nested_arrays.
  """


  def stack_tensor(*tensors):
    result = [tf.convert_to_tensor(t) for t in tensors]
    return tf.stack(result)

  return tf.nest.map_structure(stack_tensor, *list_of_nests)


def add_time_batch_dim(*nested_tensors):
  if len(nested_tensors) == 1:
    return tf.nest.map_structure(
        lambda t: tf.expand_dims(tf.expand_dims(t, 0), 0), nested_tensors[0])
  return [
      tf.nest.map_structure(lambda t: tf.expand_dims(tf.expand_dims(t, 0), 0),
                            tensor) for tensor in nested_tensors
  ]


def add_batch_dim(nest):
  return tf.nest.map_structure(lambda t: tf.expand_dims(t, 0), nest)


def remove_time_batch_dim(*nested_tensors):
  if len(nested_tensors) == 1:
    return tf.nest.map_structure(lambda t: tf.squeeze(t, [0, 1]),
                                 nested_tensors[0])
  return [
      tf.nest.map_structure(lambda t: tf.squeeze(t, [0, 1]), tensor)
      for tensor in nested_tensors
  ]


def get_row_nested_tensor(nest, row_idx):
  return tf.nest.map_structure(lambda t: t[row_idx], nest)


def batch_apply(fn, inputs):
  """Folds time into the batch dimension, runs `fn()` and unfolds the result.

  Args:
    fn: A callable
    inputs: A tensor or nested structure with individual tensors that have first
      two dimension equal to [num_timesteps, batch_size].

  Returns:
    Output of `fn(inputs)` which is a tensor or nested structure with
      individual tensors that have first two dimensions same as `inputs`.
  """
  fold_time_batch = lambda t: tf.reshape(t, [-1] + t.shape[2:].as_list())
  folded_input = tf.nest.map_structure(fold_time_batch, inputs)
  folded_output = fn(folded_input)
  prefix = [int(tf.nest.flatten(inputs)[0].shape[0]), -1]
  unfold_time_batch = lambda t: tf.reshape(t, prefix + t.shape[1:].as_list())
  return tf.nest.map_structure(unfold_time_batch, folded_output)


def gather_from_dict(tensor_dict, choice):
  """Chooses tensor values along first dimension using given choice.

  If `tensor_dict` = {
    0: zeros(shape=(6)),
    1: ones(shape=(6)),
    2: twos(shape=(6)),
    3: threes(shape=(6))
  }
  and choice = [0, 0, 2, 2, 1, 0]
  then returned tensor is [0., 0., 2., 2., 1., 0.]

  Args:
    tensor_dict: A dict with int keys and tensor values. All tensor values must
      be of same type and shape.
    choice: A 1-d int tensor with number of elements equal to first dimension of
      tensors in `tensor_dict`. The values in the tensor must be valid keys in
      `tensor_dict`.

  Returns:
    A tensor of same type and shape as tensors in `tensor_dict`.
  """
  one_tensor = next(iter(tensor_dict.values()))

  # Check number of elements in `choice`.
  tf.debugging.assert_rank(choice, rank=1)
  tf.debugging.assert_equal(tf.size(choice), tf.shape(one_tensor)[0])

  zeros_tensor = tf.zeros_like(one_tensor)
  final_tensor = zeros_tensor
  for c, t in tensor_dict.items():
    # Check shapes and type
    tf.debugging.assert_equal(tf.shape(t), tf.shape(one_tensor))
    tf.debugging.assert_type(t, tf_type=one_tensor.dtype)
    final_tensor += tf.compat.v1.where(tf.equal(choice, c), t, zeros_tensor)
  return final_tensor


def get_first_true_column(x):
  """Transforms `x` into a tensor which has all elements set to False except the first True in the column.

  If x is [[True, False, False],
           [True, False, False],
           [False, True, False],
           [False, True, True]]
  the output should be
          [[True, False, False],
           [False, False, False],
           [False, True, False],
           [False, False, True]
          ]

  Args:
    x: A bool tensor with shape [num_steps, batch_size]

  Returns:
    A bool tensor with the same shape.
  """
  x = tf.transpose(x, perm=[1, 0])
  # Get indices
  y = tf.where(tf.equal(x, True))
  # Find first column in every row which is True
  first_true_cols = tf.cast(
      tf.math.segment_min(data=y[:, 1], segment_ids=y[:, 0]), tf.int32)
  # Convert back to indices
  first_true_indices = tf.stack(
      [tf.range(tf.size(first_true_cols)), first_true_cols], axis=1)
  # Now create the mask
  first_true_mask_sparse = tf.SparseTensor(
      indices=tf.cast(first_true_indices, tf.int64),
      values=tf.ones([tf.size(first_true_cols)], dtype=tf.bool),
      dense_shape=x.shape)
  first_true_mask = tf.sparse.to_dense(
      first_true_mask_sparse, default_value=False)
  return tf.transpose(first_true_mask, perm=[1, 0])


def get_last_true_column(x):
  """Gets the last True value in each col as True and all others False.

  Similar to `get_first_true_column` except it is reversed.

  Args:
    x: A bool tensor with shape [num_steps, batch_size]

  Returns:
    A bool tensor of the same shape with the last True element in each col set
      to True and every one else set to False.
  """
  # Reverse the row order.
  x = tf.reverse(x, axis=[0])
  x_first_true = get_first_true_column(x)
  # Reverse the rows back.
  return tf.reverse(x_first_true, axis=[0])


def write_specs(logdir, specs):
  """Writes given specs to given location."""
  specs_path = os.path.join(logdir, 'specs')
  if tf.io.gfile.exists(specs_path):
    logging.warning('Specs file already exists, not overwriting %s', specs_path)
  else:
    tf.io.gfile.makedirs(logdir)
    # Write atomically.
    temp_path = os.path.join(logdir, 'temp_specs_temp')
    with tf.io.gfile.GFile(temp_path, 'wb') as f:
      pickle.dump(specs, f)
    tf.io.gfile.rename(temp_path, specs_path)
    logging.info('Done writing tensor specs file at %s.', specs_path)


def read_specs(logdir):
  """Reads specs from given location."""
  specs_path = os.path.join(logdir, 'specs')
  while not tf.io.gfile.exists(specs_path):
    logging.info('Waiting for tensor specs.')
    time.sleep(5)
  with tf.io.gfile.GFile(specs_path, 'rb') as f:
    logging.info('Reading from specs file at %s.', specs_path)
    return pickle.load(f)



def circular_pad(input_tensor, axis, padding):
  """Pads tensor circularly.

  More specifically, pad on the right with the tensor values from the left of
  the tensor, as if you had concatenated the tensor on the right and vice versa.

  Args:
    input_tensor: typically a batch of input "images"
    axis: on which to perform the circluar padding
    padding: See tf.nn.conv2d arg: padding

  Returns:
    Tensor of shape BxHxWxC2
  """
  assert 0 <= axis < len(input_tensor.shape), 'Axis out of bounds'
  multiples = [1] * len(input_tensor.shape)
  multiples[axis] = 3
  tiled_input = tf.tile(input_tensor, multiples)
  left = input_tensor.shape[axis] - padding[0]
  right = 2 * input_tensor.shape[axis] + padding[1]

  begin = [0] * len(input_tensor.shape)
  end = list(input_tensor.shape)
  begin[axis] = left
  end[axis] = right
  size = [a - b for a, b in zip(end, begin)]

  output_tensor = tf.slice(tiled_input, begin, size)
  # Do a shape assert
  return output_tensor


def parallel_conv2d(inputs, filters, strides, padding):
  """Applies each filter in the batch of filters to each input.

  tf.nn.conv2d only supports applying the same filter on a batch of inputs.
  This function provides a similar interface, but allowing a batch of filters,
  a different one for each input.

  In the below definitions, B is the batch size, H and W are spatial input or
  output dimensions (overloaded between input and output), C1 is the input
  number of channels, C2 is output number of channels, KHxKW is the
  convolutional kernel spatial size.

  Args:
    inputs: BxHxWxC1 tensor - batch of input "images"
    filters: BxKHxKWxC1xC2 tensor - batch of convolutional kernels
    strides: See tf.nn.conv2d arg: strides
    padding: See tf.nn.conv2d arg: padding

  Returns:
    Tensor of shape BxHxWxC2
  """
  batch_size = inputs.shape[0]

  output_slices = [tf.nn.conv2d(inputs[i:i+1], filters[i], strides, padding)
                   for i in range(batch_size)]
  output = tf.stack(output_slices, axis=0)
  # Each output slice has a batch dimension of size 1. Get rid of it.
  assert output.shape[1] == 1, 'Each slice should have batch size of 1'
  output = output[:, 0, :, :, :]
  # Output should have same batch size and spatial dimensions as input, but
  # the number of channels is determined by the convolution filter
  assert_shape((batch_size, inputs.shape[1], inputs.shape[2], filters.shape[4]),
               output.shape)
  return output


def assert_shape(expected_shape, actual_shape):
  """Asserts that the shapes are equal, printing a useful message otherwise."""
  assert expected_shape == actual_shape, (
      'Shape mismatch. Expected: {}. Found: {}'.format(
          expected_shape, actual_shape))


def transpose_batch_time(x):
  """Transposes the batch and time dimensions of a Tensor.

  If the input tensor has rank < 2 it returns the original tensor. Retains as
  much of the static shape information as possible.

  Args:
    x: A Tensor.

  Returns:
    x transposed along the first two dimensions.
  """
  x_static_shape = x.shape
  rank = x_static_shape.rank
  if rank is not None and rank < 2:
    return x

  x_t = tf.transpose(a=x, perm=tf.concat(([1, 0], tf.range(2, rank)), axis=0))
  x_t.set_shape(
      tf.TensorShape(
          [x_static_shape.dims[1].value,
           x_static_shape.dims[0].value]).concatenate(x_static_shape[2:]))
  return x_t


class WallTimer(object):
  """Collect the duration of an operation using Python's 'with' statement."""

  def __enter__(self):
    self.start = tf.timestamp()
    return self

  def __exit__(self, *args):
    self.stop = tf.timestamp()
    self.duration = self.stop - self.start


class NavigationScorePickleAggregator(object):
  """VALAN navigation score aggregator.

  VALAN scores are computed and saved as pickles. This class contains
  methods for loading, aggregating and saving combined results.
  Use case: with async-worker aggregation, sometimes we have workers completing
  at different paces, which results in the need to use sleep timers or
  incomplete computation. This method circumvents this problem at the small
  cost of not being able to view results in TensorBoard (instead results are
  saved to a pickle).

  Example usage:
    score_dir = '/path/to/pickled_results/'
    aggregation_save_path = '/path/to/save/aggregation/results.p'
    aggregator = utils.NavigationScorePickleAggregator()
    score_dict, avg_scores = aggregator.get_aggregated_scores([score_dir])
    with gfile.Open(aggregation_save_path, 'wb') as fp:
      pickle.dump((score_dict, avg_scores), fp)
  """

  def _get_scores(self, pickle_path):
    """Retrieves SR/SPL/SDTW/NDTW from VALAN score pickles.

    The pickled VALAN scores have the following format
    {
      'valan_score_<TASK_ID>_<STEP>_<PATH_ID>': {
        'eval/success_rate': ..,
        'eval/spl': ..,
        'eval/sdtw': ..,
        'eval/norm_dtw': ..,
        ..
      },
      ..
    }
    and the files are named with <DATA_ID>_<WORKER_ID>.p

    Args:
      pickle_path: (string) location of pickle.

    Returns:
      (list) of VALAN scores for the dataset with <DATA_ID> produced with
      the worker with <WORKER_ID>.
    """
    with gfile.GFile(pickle_path, 'rb') as fp:
      content = pickle.load(fp)
    scores = []
    for value in content.values():
      scores.append(
          (float(value['eval/success_rate']),
           float(value['eval/spl']),
           float(value['eval/sdtw']),
           float(value['eval/norm_dtw']))
      )
    return scores

  def get_aggregated_scores(self, pickle_dirs):
    """Processes all the VALAN score pickle files in a directory.

    Args:
      pickle_dirs: (list) of directories of VALAN score pickles.

    Returns:
      score_dict: (dict) for lists of SR/SPL/SDTW/NDTW scores. E.g.
        {'sr': [0.35, 0.34, 0.27], 'spl', [0.46, 0.34, 0.45], ...}
      avg_scores: (dict) the average of SR/SPL/SDTW/NDTW scores. E.g.
        {'sr': 0.30, 'spl': 0.40, ...}
    """
    score_dict = {
        'sr': [],
        'spl': [],
        'sdtw': [],
        'ndtw': []
    }
    pickle_paths = []
    for pickle_dir in pickle_dirs:
      pickle_paths += [os.path.join(pickle_dir, filename)
                       for filename in gfile.listdir(pickle_dir)]
    for pickle_path in pickle_paths:
      scores = self._get_scores(pickle_path)
      for score in scores:
        sr, spl, sdtw, ndtw = score
        score_dict['sr'].append(sr)
        score_dict['spl'].append(spl)
        score_dict['sdtw'].append(sdtw)
        score_dict['ndtw'].append(ndtw)

    avg_scores = {
        'sr': np.mean(score_dict['sr']),
        'spl': np.mean(score_dict['spl']),
        'sdtw': np.mean(score_dict['sdtw']),
        'ndtw': np.mean(score_dict['ndtw'])
    }

    return score_dict, avg_scores


