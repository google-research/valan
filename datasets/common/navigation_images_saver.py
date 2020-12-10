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

r"""Saves images for navigation tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

from absl import logging
import nfov
import numpy as np
from PIL import Image
import tensorflow as tf
from valan.datasets.common import utils
from valan.framework import image_features_pb2


class NavigationImagesSaver(object):
  """Base class for saving images for navigation tasks."""

  def __init__(self,
               graph,
               pano_filenames,
               scan_to_pano_ids=None,
               num_tasks=1,
               task_id=0):
    """Initialize the generator.

    Args:
      graph: earthsea.lookfar.datasets.common.Graph object.
      pano_filenames: Dictionary mapping pano ids to the filenames containing
        the equirectangular projection of the panorama.
      scan_to_pano_ids: Dictionary that maps scan_id to a list of pano_ids.
      num_tasks: Number of tasks for distributed processing.
      task_id: Id of current task for distributed processing.
    """
    self.graph = graph
    self.pano_filenames = pano_filenames
    self.scan_to_pano_ids = scan_to_pano_ids
    self.num_tasks = num_tasks
    self.task_id = task_id

  def get_image_signature(self, image, tf_image_processor, heading, pitch,
                          radians):
    """Returns image signature composed by angle signature and image features.

    Args:
      image: Numpy array with dtype np.uint8 containing the pixel values of the
        image.
      tf_image_processor: earthsea.tools.tf_image_processor.TFImageProcessor for
        extracting image features.
      heading: Heading angle.
      pitch: Pitch angle.
      radians: Whether angles are in radians or not.

    Returns:
      A numpy array with the image signature. This is computed by concatenating
        the angle signature [heading, pitch]
        with the image features extracted by the tf_image_processor.
    """

    image_features = tf_image_processor.process(image)
    if not radians:
      heading = math.radians(heading)
      pitch = math.radians(pitch)
    return np.array([heading, pitch] + image_features)

  def save_image_features(self,
                          output_dir,
                          heading_angles,
                          pitch_angles,
                          image_size,
                          output_format='proto',
                          tf_image_processor=None,
                          horizontal_fov=0.25,
                          save_stop_node_features=False):
    """Saves images or image features for each pano.

    Currently supports 3 formats: raw, proto and npy.
    In raw format, viewpoint images are stored under
    <output_dir>/raw/fov=<horizontal_fov>/<pano_id>/<heading>_<pitch>.jpeg, and
    connection images are stored under
    <output_dir>/raw/fov=<horizontal_fov>/<pano_id>/<neighbor_id>.jpeg

    In proto and npy format, there are two files stored per pano,
    base_dir/<pano_id>_viewpoints_<output_format> and
    base_dir/<pano_id>_connections_<output_format>,
    where base dir is <output_dir>/<output_format>/fov=<horizontal_fov>/ +
        image_module=<image_module_name>.

    Args:
      output_dir: Directory on which to store the output files.
      heading_angles: List of heading angles in degrees to save. Heading is
        determined relative to the center of the input equirectangular images
        (increasing clockwise).
      pitch_angles: List of pitch (elevation) angles in degrees to save. Pitch 0
        means horizontal gaze.
      image_size: Tuple of (height, width) with the dimensions of the output
        images.
      output_format: Format of the output. One of {'raw', 'npy', 'proto'}.
      tf_image_processor: earthsea.tools.tf_image_processor.TFImageProcessor to
        use as feature extractor for images. Unused if output_format is 'raw'.
      horizontal_fov: Horizontal field of view (`float`) defined as a proportion
        of 360 degrees.
      save_stop_node_features: If set to True and output_format is not 'raw', we
        store extra viewpoints and connections files for STOP_NODE. This is used
        in research/lux/earthsea/navigation/environment/r2r_tf.py.
    """
    output_dir = os.path.join(output_dir, output_format,
                              'fov=%s' % horizontal_fov)

    assert output_format in ['raw', 'npy', 'proto'], 'Invalid output format.'
    if output_format != 'raw':
      assert tf_image_processor, ('tf_image_processor must not be none for'
                                  'output_format %s' % output_format)
      output_dir = os.path.join(
          output_dir, 'image_module=%s' %
          tf_image_processor.spec_str.replace('https://', ''))
    tf.io.gfile.makedirs(output_dir)

    height, width = image_size
    hfov_radians = horizontal_fov * math.pi * 2
    vfov_radians = 2.0 * math.atan(np.tan(hfov_radians / 2.0) * height / width)
    # Vertical field of view defined as a proportion of 180 degrees.
    vertical_fov = vfov_radians / math.pi

    nfov_projector = nfov.NFOV(
        height, width, fov=(horizontal_fov, vertical_fov))

    def _get_viewpoints_and_connections(pano_id, viewpoint_features,
                                        connection_features_dict):
      # Retrieve pano image.
      pano_filename = self.pano_filenames[pano_id]
      pano_image = np.array(Image.open(tf.io.gfile.GFile(pano_filename, 'rb')))
      if output_format == 'raw':
        tf.io.gfile.makedirs(os.path.join(output_dir, pano_id))
      # Get viewpoint features.
      for heading in heading_angles:
        for pitch in pitch_angles:
          # Normalize heading and pitch to [0, 1] interval. Zero heading and
          # elevation is directly in the center of the equirectangular image.
          center_point = np.array([0.5 + heading / 360.0, 0.5 - pitch / 180.0])
          perspective_image = nfov_projector.to_nfov(pano_image, center_point)
          if output_format == 'raw':
            output_filename = os.path.join(output_dir, pano_id,
                                           '%d_%d.jpeg' % (heading, pitch))
            utils.save_np_image(
                perspective_image, output_filename, output_format='jpeg')
          else:
            viewpoint_features.append(
                self.get_image_signature(perspective_image, tf_image_processor,
                                         heading, pitch, False))
      # Get connection features.
      for neighbor_pano_id in self.graph.get_neighbors(pano_id):
        connection = self.graph.get_connection(pano_id, neighbor_pano_id)
        # Find center coordinates for generating perspective image.
        center_x = 0.5 + connection.heading / (2 * math.pi)
        center_y = 0.5 - connection.pitch / math.pi  # 0 is up, 1 is down.
        center_point = np.array([center_x % 1.0, center_y])
        perspective_image = nfov_projector.to_nfov(pano_image, center_point)
        if output_format == 'raw':
          output_filename = os.path.join(output_dir, pano_id,
                                         '%s.jpeg' % neighbor_pano_id)
          utils.save_np_image(
              perspective_image, output_filename, output_format='jpeg')
        else:
          image_signature = self.get_image_signature(perspective_image,
                                                     tf_image_processor,
                                                     connection.heading,
                                                     connection.pitch, True)
          connection_features_dict[neighbor_pano_id] = image_signature
      return viewpoint_features, connection_features_dict

    # Per-scan mode.
    if self.scan_to_pano_ids is not None:
      all_scans = sorted(list(self.scan_to_pano_ids.keys()))
      scans_to_process = all_scans[self.task_id::self.num_tasks]
      for idx, scan_id in enumerate(scans_to_process):
        viewpoint_features = []
        connection_features_dict = {}
        logging.info('Processing scan %s (%d/%d).', scan_id, idx,
                     len(scans_to_process))
        for pano_id in self.scan_to_pano_ids[scan_id]:
          (viewpoint_features,
           connection_features_dict) = _get_viewpoints_and_connections(
               pano_id, viewpoint_features, connection_features_dict)
        self._save_image_features(output_dir, scan_id, output_format,
                                  np.array(viewpoint_features),
                                  connection_features_dict)
    # Per-pano mode.
    else:
      all_panos = sorted(self.graph.nodes.keys())
      panos_to_process = all_panos[self.task_id::self.num_tasks]
      for idx, pano_id in enumerate(panos_to_process):
        viewpoint_features = []
        connection_features_dict = {}
        logging.info('Processing pano %s (%d/%d).', pano_id, idx,
                     len(panos_to_process))
        (viewpoint_features,
         connection_features_dict) = _get_viewpoints_and_connections(
             pano_id, viewpoint_features, connection_features_dict)
        self._save_image_features(output_dir, pano_id, output_format,
                                  np.array(viewpoint_features),
                                  connection_features_dict)

    if save_stop_node_features and output_format != 'raw':
      # We add 2 due to the angle signature (heading, pitch).
      feature_dim = int(tf_image_processor.output_shape[1]) + 2
      num_viewpoints = len(heading_angles) * len(pitch_angles)
      viewpoint_features = np.zeros([num_viewpoints, feature_dim])
      self._save_image_features(output_dir, 'STOP_NODE', output_format,
                                viewpoint_features, {})

  def _save_image_features(
      self,
      output_dir,
      file_id,  # Can be pano_id or scan_id.
      output_format,
      viewpoint_features,
      connection_features_dict):
    """Internal method for saving image features."""
    base_filename = os.path.join(output_dir, str(file_id))
    viewpoints_filename = '%s_viewpoints_%s' % (base_filename, output_format)
    connections_filename = '%s_connections_%s' % (base_filename, output_format)
    if output_format == 'npy':
      viewpoint_features = viewpoint_features.flatten()
      with tf.io.gfile.GFile(viewpoints_filename, 'wb') as f:
        np.save(f, viewpoint_features, allow_pickle=True)
      with tf.io.gfile.GFile(connections_filename, 'wb') as f:
        np.save(f, [connection_features_dict], allow_pickle=True)
    elif output_format == 'proto':
      viewpoints_proto = image_features_pb2.ImageFeatures()
      viewpoints_proto.heading.extend(viewpoint_features[:, 0])
      viewpoints_proto.pitch.extend(viewpoint_features[:, 1])
      viewpoint_features = viewpoint_features[:, 2:]  # Drop angles.
      viewpoints_proto.shape.extend(viewpoint_features.shape)
      viewpoints_proto.value.extend(list(viewpoint_features.flatten()))
      connections_proto = image_features_pb2.ImageFeatures()
      for neighbor_id, connection_features in connection_features_dict.items():
        connections_proto.heading.append(connection_features[0])
        connections_proto.pitch.append(connection_features[1])
        connection_features = connection_features[2:]  # Drop angles.
        connections_proto.value.extend(list(connection_features))
        connections_proto.pano_id.append(neighbor_id)
      num_viewpoints = len(connections_proto.pano_id)
      if num_viewpoints > 0:
        connections_proto.shape.extend(
            [num_viewpoints,
             len(connections_proto.value) // num_viewpoints])
      with tf.io.TFRecordWriter(viewpoints_filename) as writer:
        writer.write(viewpoints_proto.SerializeToString())
      with tf.io.TFRecordWriter(connections_filename) as writer:
        writer.write(connections_proto.SerializeToString())
