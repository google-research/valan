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

"""Tests for house_parser."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from valan.r2r import house_parser
from valan.r2r import house_utils

FLAGS = flags.FLAGS


def distance_heading_pitch(from_xyz, to_xyz, apply_r2r_correction=True):
  displacement = np.array(to_xyz) - np.array(from_xyz)
  distance = np.linalg.norm(displacement)
  heading = np.arctan2(displacement[1], displacement[0])
  if apply_r2r_correction:
    heading = 0.5 * np.pi - heading
  pitch = np.arctan2(displacement[2], np.linalg.norm(displacement[:2]))
  return distance, heading, pitch


class R2RHouseParserTest(tf.test.TestCase):

  def setUp(self):
    super(R2RHouseParserTest, self).setUp()
    base_dir = FLAGS.test_srcdir + 'valan/r2r/testdata/'

    scan_id = 'p5wJjkQkbXX'
    house_file_path = os.path.join(base_dir, 'scans', scan_id,
                                   'house_segmentations', scan_id + '.house')
    self.connections_file_path = os.path.join(base_dir, 'connections',
                                              scan_id + '_connectivity.json')

    # The parser performs many assertion tests to ensure file is not malformed
    # or there are no colliding indices for annotations in the same category.
    self.house = house_parser.R2RHouseParser(
        house_file_path, category_map_dir=base_dir)
    self.addTypeEqualityFunc(float, self.assertAngleAlmostEqual)

  def assertAngleAlmostEqual(self, actual, expected, msg=None):
    # Account for periodicity in angle comparisons.
    self.assertAlmostEqual(np.cos(actual), np.cos(expected), msg)
    self.assertAlmostEqual(np.sin(actual), np.sin(expected), msg)

  def test_parser(self):
    self.assertEqual(37, self.house.num_regions)
    self.assertEqual(155, self.house.num_panos)
    self.assertEqual(1659, self.house.num_categories)
    self.assertEqual(728, self.house.num_objects)

    test_region = self.house.regions[36]
    self.assertEqual(36, test_region.index)
    self.assertEqual(1, test_region.level_index)
    self.assertEqual('c', test_region.label)
    self.assertEqual((11.995, -4.96902, 0.0194385), test_region.center)

    test_pano = self.house.panos[150]
    self.assertEqual(150, test_pano.index)
    self.assertEqual('f86f17660af34b46b5a55eca66f1dc6b', test_pano.name)
    self.assertEqual(26, test_pano.region_index)
    self.assertEqual((6.45101, -2.02881, 1.61766), test_pano.center)
    self.assertEqual(150, self.house.pano_name_map[test_pano.name])

    test_category = self.house.categories[1018]
    self.assertEqual(1018, test_category.index)
    self.assertEqual(1019, test_category.category_mapping_index)
    self.assertEqual('arcade game', test_category.category_mapping_name)
    self.assertEqual(40, test_category.mpcat40_index)
    self.assertEqual('misc', test_category.mpcat40_name)

    test_object = self.house.objects[727]
    self.assertEqual(727, test_object.index)
    self.assertEqual(36, test_object.region_index)
    self.assertEqual(0, test_object.category_index)
    self.assertEqual((12.0952, -4.27746, 1.55877), test_object.center)

    test_image = self.house.images[0]
    self.assertEqual(0, test_image.index)
    self.assertEqual(0, test_image.pano_index)
    self.assertEqual('0024bf6c261d4759be19234c9af1e819', test_image.name)

  def test_get_pano_objects(self):
    """Verifies objects we can obtain from a specific (pano, region).

    Here are some objects from region 36.
    O  154 36 1480  11.3198 -4.86241 1.07245
    O  720 36 22  12.0639 -5.18869 1.40459
    O  721 36 39  12.0604 -5.14268 1.40856
    O  722 36 3  12.0977 -5.11946 0.0729712
    O  723 36 2  12.0949 -5.18761 2.8041
    O  724 36 7  12.1216 -4.18795 1.41865
    O  725 36 1  12.6623 -4.5497 1.33353
    O  726 36 0  12.1493 -5.72757 1.42229
    O  727 36 0  12.0952 -4.27746 1.55877

    Region 36 is a closet with 1 pano.
    P  668ea79be07941669840dbf7fdaa03ee  60 36  0  12.246 -4.47023 1.62653
    R  36 1  0 0  c  11.995 -4.96902 0.0194385  11.345 -6.25896 0.0194385

    The objects we should expect in entirety are:
    [(1.14833, (11.3198, -4.86241, 1.07245), 'remove#/#behind', 'void'),
     (0.77369, (12.0639, -5.18869, 1.40459), 'shelf', 'shelving'),
     (0.73085, (12.0604, -5.14268, 1.40856), 'unknown', 'misc'),
     (1.69028, (12.0977, -5.11946, 0.0729712), 'floor', 'floor'),
     (1.38713, (12.0949, -5.18761, 2.8041), 'ceiling', 'ceiling'),
     (0.37198, (12.1216, -4.18795, 1.41865), 'doorframe', 'door'),
     (0.51524, (12.6623, -4.5497, 1.33353), 'door', 'door'),
     (1.27749, (12.1493, -5.72757, 1.42229), 'wall', 'wall'),
     (0.25395, (12.0952, -4.27746, 1.55877), 'wall', 'wall')]

    Because of deduping (taking only closest of a particular mp40 category)
    and removing miscellaneous mp40cat objects, we should be left with the
    following items.
    [(0.77369, (12.0639, -5.18869, 1.40459), 'shelf', 'shelving'),
     (1.69028, (12.0977, -5.11946, 0.0729712), 'floor', 'floor'),
     (1.38713, (12.0949, -5.18761, 2.8041), 'ceiling', 'ceiling'),
     (0.37198, (12.1216, -4.18795, 1.41865), 'doorframe', 'door'),
     (0.51524, (12.6623, -4.5497, 1.33353), 'door', 'door'),
     (1.27749, (12.1493, -5.72757, 1.42229), 'wall', 'wall'),
     (0.25395, (12.0952, -4.27746, 1.55877), 'wall', 'wall')]
    """
    objects = self.house.get_pano_objects('668ea79be07941669840dbf7fdaa03ee')
    shelf_shelving_center = (12.0639, -5.18869, 1.40459)
    floor_floor_center = (12.0977, -5.11946, 0.0729712)
    ceiling_ceiling_center = (12.0949, -5.18761, 2.8041)
    doorframe_door_center = (12.1216, -4.18795, 1.41865)
    door_door_center = (12.6623, -4.5497, 1.33353)
    wall_wall_center_1 = (12.1493, -5.72757, 1.42229)
    wall_wall_center_2 = (12.0952, -4.27746, 1.55877)

    shelf_shelving_bbox_axis0 = (0, 0, 1)
    floor_floor_bbox_axis0 = (0, 1, 0)
    ceiling_ceiling_bbox_axis0 = (0, 1, 0)
    doorframe_door_bbox_axis0 = (0, 0, 1)
    door_door_bbox_axis0 = (0, 0, 1)
    wall_wall_bbox_axis0_1 = (0, 0, 1)
    wall_wall_bbox_axis0_2 = (0, 0, 1)

    shelf_shelving_bbox_axis1 = (0, 1, -0)
    floor_floor_bbox_axis1 = (1, 0, 0)
    ceiling_ceiling_bbox_axis1 = (1, 0, 0)
    doorframe_door_bbox_axis1 = (0.92388, 0.382683, -0)
    door_door_bbox_axis1 = (2.22045e-16, 1, -0)
    wall_wall_bbox_axis1_1 = (-1, 2.22045e-16, 0)
    wall_wall_bbox_axis1_2 = (1, 0, 0)

    shelf_shelving_bbox_radii = (1.4205, 1.06979, 0.846345)
    floor_floor_bbox_radii = (1.10849, 0.813632, 0.0888786)
    ceiling_ceiling_bbox_radii = (1.05966, 0.835095, 0.0394197)
    doorframe_door_bbox_radii = (1.4126, 0.600197, 0.231877)
    door_door_bbox_radii = (1.33114, 0.428531, 0.102481)
    wall_wall_bbox_radii_1 = (1.41631, 0.786427, 0.530909)
    wall_wall_bbox_radii_2 = (1.27426, 0.731555, 0.161997)

    self.assertLen(objects, 7)
    self.assertCountEqual([
        shelf_shelving_center, floor_floor_center, ceiling_ceiling_center,
        doorframe_door_center, door_door_center, wall_wall_center_1,
        wall_wall_center_2
    ], objects.keys())

    self.assertAlmostEqual(
        0.25395, objects[wall_wall_center_2].distance, places=5)
    self.assertAlmostEqual(objects[wall_wall_center_2].oriented_bbox.axis0,
                           wall_wall_bbox_axis0_2)
    self.assertAlmostEqual(objects[wall_wall_center_2].oriented_bbox.axis1,
                           wall_wall_bbox_axis1_2)
    self.assertAlmostEqual(objects[wall_wall_center_2].oriented_bbox.radii,
                           wall_wall_bbox_radii_2)
    self.assertEqual('wall', objects[wall_wall_center_2].name)
    self.assertEqual('wall', objects[wall_wall_center_2].category)

    self.assertAlmostEqual(
        1.27749, objects[wall_wall_center_1].distance, places=5)
    self.assertAlmostEqual(objects[wall_wall_center_1].oriented_bbox.axis0,
                           wall_wall_bbox_axis0_1)
    self.assertAlmostEqual(objects[wall_wall_center_1].oriented_bbox.axis1,
                           wall_wall_bbox_axis1_1)
    self.assertAlmostEqual(objects[wall_wall_center_1].oriented_bbox.radii,
                           wall_wall_bbox_radii_1)
    self.assertEqual('wall', objects[wall_wall_center_1].name)
    self.assertEqual('wall', objects[wall_wall_center_1].category)

    self.assertAlmostEqual(
        1.69028, objects[floor_floor_center].distance, places=5)
    self.assertAlmostEqual(objects[floor_floor_center].oriented_bbox.axis0,
                           floor_floor_bbox_axis0)
    self.assertAlmostEqual(objects[floor_floor_center].oriented_bbox.axis1,
                           floor_floor_bbox_axis1)
    self.assertAlmostEqual(objects[floor_floor_center].oriented_bbox.radii,
                           floor_floor_bbox_radii)
    self.assertEqual('floor', objects[floor_floor_center].name)
    self.assertEqual('floor', objects[floor_floor_center].category)

    self.assertAlmostEqual(
        0.37198, objects[doorframe_door_center].distance, places=5)
    self.assertAlmostEqual(objects[doorframe_door_center].oriented_bbox.axis0,
                           doorframe_door_bbox_axis0)
    self.assertAlmostEqual(objects[doorframe_door_center].oriented_bbox.axis1,
                           doorframe_door_bbox_axis1)
    self.assertAlmostEqual(objects[doorframe_door_center].oriented_bbox.radii,
                           doorframe_door_bbox_radii)
    self.assertEqual('doorframe', objects[doorframe_door_center].name)
    self.assertEqual('door', objects[doorframe_door_center].category)

    self.assertAlmostEqual(
        0.51524, objects[door_door_center].distance, places=5)
    self.assertAlmostEqual(objects[door_door_center].oriented_bbox.axis0,
                           door_door_bbox_axis0)
    self.assertAlmostEqual(objects[door_door_center].oriented_bbox.axis1,
                           door_door_bbox_axis1)
    self.assertAlmostEqual(objects[door_door_center].oriented_bbox.radii,
                           door_door_bbox_radii)
    self.assertEqual('door', objects[door_door_center].name)
    self.assertEqual('door', objects[door_door_center].category)

    self.assertAlmostEqual(
        1.38713, objects[ceiling_ceiling_center].distance, places=5)
    self.assertAlmostEqual(objects[ceiling_ceiling_center].oriented_bbox.axis0,
                           ceiling_ceiling_bbox_axis0)
    self.assertAlmostEqual(objects[ceiling_ceiling_center].oriented_bbox.axis1,
                           ceiling_ceiling_bbox_axis1)
    self.assertAlmostEqual(objects[ceiling_ceiling_center].oriented_bbox.radii,
                           ceiling_ceiling_bbox_radii)
    self.assertEqual('ceiling', objects[ceiling_ceiling_center].name)
    self.assertEqual('ceiling', objects[ceiling_ceiling_center].category)

    self.assertAlmostEqual(
        0.77369, objects[shelf_shelving_center].distance, places=5)
    self.assertAlmostEqual(objects[shelf_shelving_center].oriented_bbox.axis0,
                           shelf_shelving_bbox_axis0)
    self.assertAlmostEqual(objects[shelf_shelving_center].oriented_bbox.axis1,
                           shelf_shelving_bbox_axis1)
    self.assertAlmostEqual(objects[shelf_shelving_center].oriented_bbox.radii,
                           shelf_shelving_bbox_radii)
    self.assertEqual('shelf', objects[shelf_shelving_center].name)
    self.assertEqual('shelving', objects[shelf_shelving_center].category)

  def test_get_panos_graph(self):
    graph = self.house.get_panos_graph(self.connections_file_path)

    # Graph should be symmetric.
    for node in graph.nodes:
      for neighbor in graph.get_connections(node):
        self.assertIn(node, graph.get_connections(neighbor))

    test_node = '7dd8fe295b224962b6933cbd67d77eeb'
    test_node_xyz = (2.77008, 5.14591, 1.51923)

    expected_neighbors = {  # Maps pano_ids to xyz coordinates of their center
        '0469b4fc6cbd47339d41631e3ec093a4': (-0.0844072, 8.84346, 1.49613),
        'd3f5d9ad84994c68afac6a292c39e2b3': (-0.0301065, 1.99833, 1.61085),
        '418fb1f91e9543aab0c4205a90ce217d': (0.394006, 6.79126, 1.61644),
        '858d6ec47f1045bbb9f2e5e0e1b380ae': (6.7937, 6.78876, 1.52271),
        '8b07092911a24d28aa3761a261c10eaf': (-1.43799, 3.36807, 1.51478),
        'ecd1018346dd4921a3f57e81ea74cf3b': (-0.227239, 5.19582, 1.5069),
        '5e8accac78704491a88a1aad7cb00d05': (4.90387, 6.68458, 1.52262),
        'd3c1b1ed115c4034be5946179563ff22': (2.74947, 6.71263, 1.51295)
    }

    self.assertCountEqual(expected_neighbors.keys(),
                          graph.get_connections(test_node).keys())

    for neighbor, xyz in expected_neighbors.items():
      expected_distance = house_utils.compute_distance(test_node_xyz, xyz)
      expected_heading = house_utils.compute_heading_angle(test_node_xyz, xyz)
      expected_pitch = house_utils.compute_pitch_angle(test_node_xyz, xyz)
      connection = graph.get_connection(test_node, neighbor)
      # Test that results are consistent with house_utils.
      self.assertAlmostEqual(connection.distance, expected_distance)
      self.assertAlmostEqual(connection.heading, expected_heading)
      self.assertAlmostEqual(connection.pitch, expected_pitch)
      # Test results against independent calculations.
      expected_distance, expected_heading, expected_pitch =\
          distance_heading_pitch(test_node_xyz, xyz)
      self.assertAngleAlmostEqual(connection.distance, expected_distance)
      self.assertAngleAlmostEqual(connection.heading, expected_heading)
      self.assertAngleAlmostEqual(connection.pitch, expected_pitch)

    expected_excluded_panos = [
        '49806ddec46b4c1b8a268f1fa24fa0c0', 'e89e910f7fd0456f8bfc5ac4821caafc',
        'fdad368d493446b5a174695b65a94f56'
    ]

    for excluded_pano in expected_excluded_panos:
      self.assertNotIn(excluded_pano, graph.nodes)


class CategoryTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(CategoryTest, self).setUp()
    self.base_dir = FLAGS.test_srcdir + (
        'valan/r2r/testdata/')
    self.banned_mp40_cat_index = {0, 41}
    self.cat_map = house_parser._load_cat_map(
        category_map_file='category_mapping.tsv', file_dir=self.base_dir)

  def test_load_cat_map(self):
    """Test loaded category mappings."""
    self.assertCountEqual(
        self.cat_map[1].values(),
        ['wall', 'wall', 7667, 1, 'wall'])
    self.assertCountEqual(
        self.cat_map[100].values(),
        ['skylight', 'skylight', 57, 9, 'window'])
    self.assertCountEqual(
        self.cat_map[500].values(),
        ['lights', 'light', 3, 28, 'lighting'])
    self.assertCountEqual(
        self.cat_map[1659].values(),
        ['washbasin top', 'washbasin', 1, 15, 'sink'])

  @parameterized.named_parameters(
      ('case1', 'C  116  117 drawer  13 chest_of_drawers  0 0 0 0 0', 'drawer'),
      ('case2', 'C  117  118 bathroom#countertop#object  39 objects  0 0 0 0 0',
       'object'),
      ('case3', 'C  118  119 washing#machine  37 appliances  0 0 0 0 0',
       'washing machine'),
      ('case4', 'C  119  120 shower#curtain  12 curtain  0 0 0 0 0',
       'shower curtain'),
      ('case5', 'C  146  147 decorative#object  39 objects  0 0 0 0 0',
       'decoration'))
  def test_category(self, line, expected_clean_cat_name):
    expected = line.split()
    category = house_parser.Category(line, self.cat_map)
    self.assertEqual(category.index, int(expected[1]))
    self.assertEqual(category.category_mapping_index, int(expected[2]))
    self.assertEqual(
        category.category_mapping_name, expected[3].replace('#', ' '))
    self.assertEqual(category.mpcat40_index, int(expected[4]))
    self.assertEqual(category.mpcat40_name, expected[5])
    self.assertEqual(category.clean_category_name, expected_clean_cat_name)


if __name__ == '__main__':
  tf.test.main()
