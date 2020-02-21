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

r"""Saves images for navigation tasks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io

from absl import logging

import nfov
import numpy as np
import PIL
import tensorflow.compat.v2 as tf

from valan.streetview_common import tf_image_processor

tf.compat.v1.enable_v2_behavior()


IMSIZE_PER_FEATURE_EXTRCTOR = {
    "resnet_50": 224,
    "starburst_v4": 289,
    "bottleneck_7": 225
}
TFHUB_MODULES = {
    "resnet_50": "@slim/imagenet/resnet_v2_50/feature_vector/4",
    "starburst_v4": "@starburst/starburst_v4/2",
    "bottleneck_7": "@ica/hornet/bottleneck/7",
}


def get_feature_filename(
    feature_extractor_name,
    h_stride,
    v_stride,
    feature_height,
    fov):
  """Returns a filename given feature map parameters."""
  return "features_{}_h-{}_v-{}_height-{}_fov-{}.sstable@100".format(
      feature_extractor_name, h_stride, v_stride, feature_height, fov)


def compute_equirectangular_pano_size(imsize, fov):
  """Computes pano dims s.t. perspective crop with given fov is imsize large."""
  pano_height = int(imsize * 180 / fov)
  pano_width = pano_height * 2
  return (pano_width, pano_height)


def init_image_processor(feature_extractor_name):
  """Creates an image processor for this feature extractor."""
  module_path = TFHUB_MODULES.get(feature_extractor_name)
  if module_path is None:
    return None
  return tf_image_processor.TFImageProcessor(tf_hub_module_path=module_path)


def init_nfov_projector(imsize, fov):
  """Creates an equirectangular to perspective projector.

  Args:
    imsize: The width and height of the square perspective image.
    fov: Field of view of the square perspective image.
  Returns:
    nfov.NFOV projector.
  """
  horizontal_fov_frac = float(fov) / 360
  vertical_fov_frac = 2 * horizontal_fov_frac
  nfov_projector = nfov.NFOV(
      imsize,
      imsize,
      fov=(horizontal_fov_frac, vertical_fov_frac))
  return nfov_projector


def extract_pano_raw_features(pano_np, height):
  """Extracts fake features in form of a smaller raw RGB image."""
  # Resize the pano to the desired height.
  big_height = int(180 * height / 60)
  big_width = int(pano_np.shape[1] * big_height / pano_np.shape[0])
  pano_s_np = np.array(
      PIL.Image.fromarray(pano_np).resize(
          (big_width, big_height), PIL.Image.BILINEAR))

  # Crop a horizontal slice through the middle of the panorama
  #  basically throw away sky and ground pixels
  v_start = int((pano_s_np.shape[0] - height) / 2)
  pano_np_crop_out = pano_s_np[v_start:v_start + height, :, :]

  return pano_np_crop_out


def extract_pano_neural_features(
    nfov_projector,
    im_processor,
    pano_np,
    heading_angles,
    pitch_angles):
  """Extracts a feature vector for each heading and pitch angle.

  For each heading and pitch pair in heading_angles and pitch_angles, uses
  nfov_projector to obtain a perspective image from the pano_np panorama.

  Then processes this perspective image with the im_processor, before assembling
  all feature vectors into a single coherent 2D feature map of shape HxWxC.

  Args:
    nfov_projector: nfov.NFOV configured to give perspective images of
      the correct size.
    im_processor: tf_image_processor.TFImageProcessor that applies the desired
      feature extractor / neural network.
    pano_np: A 2D equirectangular panorama image.
    heading_angles: A list of yaw angles in degrees.
    pitch_angles: A list of pitch angles in degrees.

  Returns:
    np.ndarray of shapre HxWxC, where H = len(pitch_angles),
      W = len(heading_angles) and C is the feature vector size as returned by
      im_processor.

  """
  pano_image_features = []
  for pitch in pitch_angles:
    pitch_features = []

    for heading in heading_angles:
      # Normalize heading and pitch to [0, 1] interval.
      center_point = np.array([heading/360.0, 0.5 - pitch/180.0])

      # Compute perspective image
      perspective_image = nfov_projector.to_nfov(
          pano_np, center_point)

      # JPEG encode because that's what tf_image_processor expects?

      image_buffer = io.BytesIO()
      pil_image = PIL.Image.fromarray(perspective_image)
      pil_image.save(image_buffer, format="jpeg")

      # Apply the neural net to extract image features
      view_features = im_processor.process(image_buffer.getvalue())
      # view_features is a list of floats representing the feature vector

      # Collect feature vectors from all headings for this pitch
      pitch_features.append(view_features)

    # Collect lists of feature vectors for all pitches in a 2D feature map
    pano_image_features.append(pitch_features)

  pano_image_features = np.asarray(pano_image_features)
  return pano_image_features


def extract_image_features(
    get_pano_image_fn,
    save_pano_features_fn,
    panoids,
    feature_extractor_name,
    h_stride,
    v_stride,
    feature_height,
    fov):
  """Saves images or image features for each pano in proto format.

  Loops through the provided panoids, and for each one uses the function
  get_pano_image_fn to retrieve the panorama. The panorama is then processed
  with the desired feature extractor to produce a HxWxC feature map as a numpy
  ndarray. This feature map is then saved by calling save_pano_features_fn.

  Appropriate implementations of get_pano_image_fn and save_pano_features_fn
  would allow applying this pipeline with different input and output formats.

  Args:
    get_pano_image_fn: A function that takes a panoID and returns an np image.
    save_pano_features_fn: A function that takes a panoID and feature map
      in form of numpy ndarray as input and saves the feature map.
    panoids: An iterable of panoIDs for which to compute features.
    feature_extractor_name: One of resnet50, starburst_v4, bottleneck7, raw
    h_stride: Angle (degrees) between horizontally adjacent feature vectors
    v_stride: Angle (degrees) between vertically adjacent feature vectors
    feature_height: Height of the output feature map
    fov: Field of view (degrees) of each perspective image
  """
  # Settings for "extracting" raw image features
  if feature_extractor_name == "raw":
    imsize = feature_height

  # Settings for extracting neural network features
  else:
    nn_heading_angles = list(range(0, 360, h_stride))
    nn_min_pitch = -int(feature_height/2) * v_stride
    nn_max_pitch = nn_min_pitch + feature_height * v_stride
    nn_pitch_angles = list(range(nn_min_pitch, nn_max_pitch, v_stride))
    imsize = IMSIZE_PER_FEATURE_EXTRCTOR[feature_extractor_name]
    nfov_projector = init_nfov_projector(imsize, fov)
    im_processor = init_image_processor(feature_extractor_name)

  # Common settings for both types of features
  pano_size_pil = compute_equirectangular_pano_size(imsize, fov)

  for idx, pano_id in enumerate(sorted(panoids)):
    logging.info(
        "Processing pano %s (%d/%d).", pano_id, idx, len(panoids))

    # Grab the pano and resize it as small as permissible
    pano_image_np = get_pano_image_fn(pano_id)
    pano_image_s_np = np.array(
        PIL.Image.fromarray(pano_image_np).resize(
            pano_size_pil, PIL.Image.BILINEAR))

    # Extract 2D feature maps from this pano
    if feature_extractor_name == "raw":
      features = extract_pano_raw_features(
          pano_image_s_np,
          imsize
      )
    else:
      features = extract_pano_neural_features(
          nfov_projector,
          im_processor,
          pano_image_s_np,
          nn_heading_angles,
          nn_pitch_angles)

    save_pano_features_fn(pano_id, features)
  logging.info("Finished processing panos")
