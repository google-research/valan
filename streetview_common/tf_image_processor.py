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

"""Utility for generating image feature vectors from a pre-trained model.

Expected Usage:
---------------
  tf_hub_module_spec = hub.load_module_spec(FLAGS.module)
  tf_image_processor = TFImageProcessor(tf_hub_module_spec=tf_hub_module_spec)
  encoded_jpeg_image = read_image_bytes(...)
  feature_vector = tf_image_processor.process(encoded_jpeg_image)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tensorflow_hub as hub


class TFImageProcessor(object):
  """Utility to access a TF session that can output image feature vectors."""

  def __init__(self,
               tf_hub_module_spec=None,
               tf_hub_module_path=None,
               use_jpeg_input=True):
    """Creates an instance to extract image features from a pre-trained model.

    The model to use may be specified as a TF-hub module (either by ModuleSpec
    or path) or as an Inception V4 model checkpoint.

    If a TF-hub module is given, it is assumed to conform to the interface
    described in [1]. Its default signature should take an input 'images' Tensor
    with shape [batch_size, height, width, num_channels=3] and return a
    [batch_size, feature_dim] Tensor of features. Pass
    `tf_hub_module_spec=make_module_spec_for_testing()` to stub out the model
    for tests.

    [1]
    https://www.tensorflow.org/hub/common_signatures/images#image_feature_vector

    Args:
      tf_hub_module_spec: `hub.ModuleSpec` or None, the TF-hub module to load.
      tf_hub_module_path: str or None, the location of the TF-hub module to load
        in a format understood by `load_module_spec()` (URL,
        '@internal/module/name', '/on/disk/path', etc.)
      use_jpeg_input: Set to True to pass jpeg-encoded Image bytes data,
        otherwise a uint8 numpy or tf tensor will be expected.

    Raises:
      ValueError: if not exactly one kwarg specifying the model is given.
    """
    self.spec_str = None  # String describing the model/module being used.

    # Input and output tensors for the image to representation computation.
    # The output tensor will depend on the model options.
    self._input = None
    self._output = None
    self._session = None
    self._use_jpeg_input = use_jpeg_input

    num_kwargs = sum(
        int(kwarg is not None) for kwarg in
        [tf_hub_module_spec, tf_hub_module_path])
    if num_kwargs != 1:
      raise ValueError(
          'Must provide exactly one of "tf_hub_module_spec", '
          '"tf_hub_module_path".')

    if tf_hub_module_spec:
      self.spec_str = 'user_provided_module'
      self._initialize_from_hub_module(tf_hub_module_spec)
    elif tf_hub_module_path:
      self.spec_str = tf_hub_module_path
      self._initialize_from_hub_module(hub.load_module_spec(tf_hub_module_path))

  def _preprocess_ops(self, desired_size):
    """Image preprocessing graph prior to feeding into TF-Hub module.

    Reads the image encoding into a tensor and resize to appropriate dimensions
    expected by the image model (TF-hub or pre-trained image checkpoint).

    Args:
      desired_size: (tuple) (height, width) expected by the model.

    Returns:
      Tensor <float>[batch_size=1, height, width, channels=3] representing the
        preprocessed image tensor.
    """
    if self._use_jpeg_input:
      self._input = tf.placeholder(tf.string)
      image = tf.image.decode_jpeg(self._input, channels=3)
    else:  # Allow for uncompressed image inputs.
      self._input = tf.placeholder(tf.uint8)
      image = self._input
    image = tf.image.convert_image_dtype(image, tf.float32)
    resized_image = tf.image.resize_image_with_pad(
        image, desired_size[0], desired_size[1])
    return tf.expand_dims(resized_image, 0)

  def _initialize_from_hub_module(self, module_spec):
    """Initialize image processing pipeline to use TF-hub module."""
    self._graph = tf.Graph()
    desired_size = hub.image_util.get_expected_image_size(module_spec)
    with self._graph.as_default():
      # <float>[batch_size, num_features]
      module = hub.Module(module_spec)
      signature = None  # default
      output_name = 'default'
      self.output_shape = module.get_output_info_dict(
          signature)[output_name].get_shape()
      self._output = module(self._preprocess_ops(desired_size), signature=None)
      self._session = tf.Session(graph=self._graph)
      self._session.run(tf.global_variables_initializer())

  def process(self, encoded_image):
    """Computes feature vector for the given image bytes.

    Args:
      encoded_image: Image bytes data.

    Returns:
      Feature vector with shape <float>[num_features].
    """
    # The returned shape is <float>[batch_size, num_features] so we need to
    # index the first dimension before converting to list.
    fv = self._session.run(self._output, {self._input: encoded_image})
    return fv[0].tolist()


def make_module_spec_for_testing(input_image_height=289,
                                 input_image_width=289,
                                 output_feature_dim=64):
  """Makes a stub image feature module for use in `TFImageProcessor` tests.

  The resulting module has the signature expected by `TFImageProcessor`, but it
  has no trainable variables and its initialization loads nothing from disk.

  Args:
    input_image_height: int, height of the module's input images.
    input_image_width: int, width of module's input images.
    output_feature_dim: int, dimension of the output feature vectors.

  Returns:
    `hub.ModuleSpec`
  """

  def module_fn():
    """Builds the graph and signature for the stub TF-hub module."""
    image_data = tf.placeholder(
        shape=[1, input_image_height, input_image_width, 3], dtype=tf.float32)
    # Linearly project image_data to shape [1, output_feature_dim] features.
    projection_matrix = tf.ones([tf.size(image_data), output_feature_dim],
                                dtype=tf.float32)
    encoder_output = tf.matmul(
        tf.reshape(image_data, [1, -1]), projection_matrix)
    # NB: the input feature must be named 'images' to satisfy
    # hub.image_util.get_expected_image_size().
    hub.add_signature(
        'default', inputs={'images': image_data}, outputs=encoder_output)

  return hub.create_module_spec(module_fn)
