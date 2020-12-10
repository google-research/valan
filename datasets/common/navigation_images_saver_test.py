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

"""Tests for datasets.common.navigation_images_saver."""

import collections
import os

from absl import flags
import numpy as np
import tensorflow as tf
from valan.datasets.common import navigation_images_saver
from valan.datasets.common import utils
from valan.framework import image_features_pb2
from valan.r2r import house_utils
from valan.streetview_common import tf_image_processor

FLAGS = flags.FLAGS


class NavigationImagesSaverTest(tf.test.TestCase):
  """Tests for navigation_images_saver.py."""

  def setUp(self):
    super(NavigationImagesSaverTest, self).setUp()
    self.test_tmp_dir = FLAGS.test_tmpdir
    self.output_dir = os.path.join(self.test_tmp_dir, 'test_output')
    self.fake_input_dir = os.path.join(self.test_tmp_dir, 'test_fake_input')
    for dirname in [self.output_dir, self.fake_input_dir]:
      tf.io.gfile.makedirs(dirname)

    connections = {  # Maps (source_node, target_node): connection_angle
        ('0', '1'): 10,
        ('1', '2'): 20,
        ('2', '3'): 30,
        ('3', '4'): 40,
        ('4', '5'): 50,
        ('5', '0'): 60
    }

    graph_dict = collections.defaultdict(house_utils.NodeInfo)
    for (source, target), heading in connections.items():
      connection_info = house_utils.ConnectionInfo(1, heading)
      graph_dict[source].connections[target] = connection_info
    self.graph = house_utils.Graph(graph_dict)

    # Create fake images for each point.
    scan_to_pano_ids = collections.defaultdict(list)
    pano_filenames = {}
    for mock_scan_id, pano in enumerate(self.graph.nodes):
      fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
      filename = os.path.join(self.fake_input_dir, pano + '.jpeg')
      utils.save_np_image(fake_image, filename)
      scan_to_pano_ids[mock_scan_id].append(pano)
      pano_filenames[pano] = filename

    self.image_saver = navigation_images_saver.NavigationImagesSaver(
        graph=self.graph,
        pano_filenames=pano_filenames,
        scan_to_pano_ids=scan_to_pano_ids)
    self.heading_angles = list(range(0, 360, 30))
    self.pitch_angles = [-30, 0, 30]
    self.image_size = [10, 10]

    self.image_features_size = 10
    self.image_module_name = 'image_module'
    # Use a real image processor instance backed by a mock TF-hub module.
    self.tf_image_processor = tf_image_processor.TFImageProcessor(
        tf_hub_module_spec=tf_image_processor.make_module_spec_for_testing(
            output_feature_dim=self.image_features_size),
        use_jpeg_input=False)
    self.tf_image_processor.spec_str = 'image_module'

  def test_save_image_features_raw(self):
    self.image_saver.save_image_features(
        output_dir=self.output_dir,
        heading_angles=self.heading_angles,
        pitch_angles=self.pitch_angles,
        image_size=self.image_size,
        output_format='raw')

    test_output_dir = os.path.join(self.output_dir, 'raw', 'fov=0.25')
    self.assertCountEqual(self.graph.nodes.keys(),
                          tf.io.gfile.listdir(test_output_dir))
    for pano in self.graph.nodes:
      pano_files = tf.io.gfile.listdir(os.path.join(test_output_dir, pano))
      self.assertLen(pano_files,
                     len(self.heading_angles) * len(self.pitch_angles) + 1)
      # Add viewpoints.
      for heading in self.heading_angles:
        for pitch in self.pitch_angles:
          self.assertIn('%d_%d.jpeg' % (heading, pitch), pano_files)
      # Add connections.
      for neighbor in self.graph.get_connections(pano).keys():
        self.assertIn('%s.jpeg' % neighbor, pano_files)

  def test_save_image_features_npy(self):
    self.image_saver.save_image_features(
        output_dir=self.output_dir,
        heading_angles=self.heading_angles,
        pitch_angles=self.pitch_angles,
        image_size=self.image_size,
        output_format='npy',
        tf_image_processor=self.tf_image_processor)

    test_output_dir = os.path.join(self.output_dir, 'npy', 'fov=0.25',
                                   'image_module=%s' % self.image_module_name)
    output_files = tf.io.gfile.listdir(test_output_dir)
    num_viewpoints = len(self.heading_angles) * len(self.pitch_angles)
    # Plus two since Numpy version encodes heading,pitch into the feature.
    expected_viewpoint_image_feature_size = (
        num_viewpoints * (2 + self.image_features_size))
    for pano in self.graph.nodes:
      viewpoints_filename = '%s_viewpoints_npy' % pano
      self.assertIn(viewpoints_filename, output_files)
      with tf.io.gfile.GFile(
          os.path.join(test_output_dir, viewpoints_filename), 'rb') as f:
        image_features = np.load(f, allow_pickle=True)
      self.assertListEqual(
          list(image_features.shape), [expected_viewpoint_image_feature_size])

      connections_filename = '%s_connections_npy' % pano
      self.assertIn(connections_filename, output_files)
      with tf.io.gfile.GFile(
          os.path.join(test_output_dir, connections_filename), 'rb') as f:
        image_features_dict = np.load(f, allow_pickle=True)[0]
      self.assertLen(image_features_dict, 1)
      self.assertListEqual(
          list(next(iter(image_features_dict.values())).shape),
          # Plus two since Numpy version encodes heading,pitch into the feature.
          [2 + self.image_features_size])

  def test_save_image_features_proto(self):
    self.image_saver.save_image_features(
        output_dir=self.output_dir,
        heading_angles=self.heading_angles,
        pitch_angles=self.pitch_angles,
        image_size=self.image_size,
        output_format='proto',
        tf_image_processor=self.tf_image_processor,
        save_stop_node_features=True)

    def get_protos(test_output_dir, filename):
      protos = []
      records = tf.data.TFRecordDataset(os.path.join(test_output_dir, filename))
      for record in records.take(1):
        proto = image_features_pb2.ImageFeatures()
        proto.ParseFromString(record.numpy())
        protos.append(proto)
      return protos

    test_output_dir = os.path.join(self.output_dir, 'proto', 'fov=0.25',
                                   'image_module=%s' % self.image_module_name)
    output_files = tf.io.gfile.listdir(test_output_dir)
    num_viewpoints = len(self.heading_angles) * len(self.pitch_angles)
    expected_viewpoint_image_feature_size = (
        num_viewpoints * (self.image_features_size))
    for pano in list(self.graph.nodes.keys()):
      # Viewpoint test.
      viewpoints_filename = '%s_viewpoints_proto' % pano
      self.assertIn(viewpoints_filename, output_files)
      protos = get_protos(test_output_dir, viewpoints_filename)
      self.assertLen(protos[0].value, expected_viewpoint_image_feature_size)
      self.assertLen(protos[0].heading, num_viewpoints)
      self.assertLen(protos[0].pitch, num_viewpoints)
      # Connection test.
      connections_filename = '%s_connections_proto' % pano
      self.assertIn(connections_filename, output_files)
      protos = get_protos(test_output_dir, connections_filename)
      if pano != 'STOP_NODE':
        self.assertLen(protos, 1)
        self.assertLen(protos[0].value, self.image_features_size)
        self.assertLen(protos[0].pano_id, 1)
        self.assertLen(protos[0].heading, 1)
        self.assertLen(protos[0].pitch, 1)
      else:
        self.assertLen(protos, 1)
        self.assertEmpty(protos[0].value)
        self.assertEmpty(protos[0].pano_id)
        self.assertEmpty(protos[0].heading)
        self.assertEmpty(protos[0].pitch)


if __name__ == '__main__':
  tf.test.main()
