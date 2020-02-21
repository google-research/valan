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

"""Utility for parsing artifacts about a R2R house specification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import json
import math
import os
import re
import numpy as np

import tensorflow.compat.v2 as tf

from valan.r2r import house_utils

_BANNED_MP40_CAT_INDEX = {0, 40, 41}
# Note: `oriented_bbox` is another namedtuple of ('axis0', 'axis1', 'radii')
RoomObject = collections.namedtuple('RoomObject', [
    'center', 'distance', 'name', 'category', 'clean_category', 'oriented_bbox'
])


class Region(object):
  """Region data in the house segmentation file."""

  NUM_TOKENS = 20  # Number of tokens to expect in line.
  INDEX_LOC = 1    # Location of region index.
  LEVEL_LOC = 2    # Location of level index.
  LABEL_LOC = 5    # Location of label.
  PX_LOC = 6       # Location of px.
  PY_LOC = 7       # Location of py.
  PZ_LOC = 8       # Location of pz.

  def __init__(self, region_line):
    parts = region_line.strip().split()
    assert 'R' == parts[0]
    assert self.NUM_TOKENS == len(parts)
    self.index = int(parts[self.INDEX_LOC])
    self.level_index = int(parts[self.LEVEL_LOC])
    self.label = parts[self.LABEL_LOC]
    self.center = (float(parts[self.PX_LOC]), float(parts[self.PY_LOC]),
                   float(parts[self.PZ_LOC]))


class Pano(object):
  """Pano data in the house segmentation file."""

  NUM_TOKENS = 13  # Number of tokens to expect in line.
  NAME_LOC = 1     # Location of name.
  INDEX_LOC = 2    # Location of index.
  REGION_LOC = 3   # Location of region index.
  PX_LOC = 5       # Location of px.
  PY_LOC = 6       # Location of py.
  PZ_LOC = 7       # Location of pz.

  def __init__(self, pano_line):
    parts = pano_line.strip().split()
    assert 'P' == parts[0]
    assert self.NUM_TOKENS == len(parts)
    self.name = parts[self.NAME_LOC]
    self.index = int(parts[self.INDEX_LOC])
    self.region_index = int(parts[self.REGION_LOC])
    self.center = (float(parts[self.PX_LOC]), float(parts[self.PY_LOC]),
                   float(parts[self.PZ_LOC]))
    # List of images (`Image`) of the pano.
    self.images = []

  def get_available_headings(self):
    # Only need to consider the heading angles of one of the 3 cameras.
    return [image.heading for image in self.images if image.camera_index == 0]

  def get_images_at(self, heading, threshold=math.pi / 12):
    """Returns the images within threshold arc distance from given heading."""
    angle_distances = [
        house_utils.compute_arc_distance(heading, image.heading)
        for image in self.images
    ]
    matches = [
        self.images[idx]
        for idx, dist in enumerate(angle_distances)
        if dist <= threshold
    ]
    # Sort matches by pitch angle for consistency.
    sorted_images = sorted(matches, key=lambda image: image.pitch)
    return sorted_images


class Image(object):
  """Image data in the house segmentation file."""

  NUM_TOKENS = 41                  # Number of tokens to expect in line.
  INDEX_LOC = 1                    # Location of image index.
  PANO_INDEX_LOC = 2               # Location of panorama index.
  NAME_LOC = 3                     # Location of the name of the image.
  CAMERA_INDEX_LOC = 4             # Location of the camera index.
  HEADING_INDEX_LOC = 5            # Location of the heading (yaw) index.
  EXTRINSIC_MATRIX_START_LOC = 6   # Start location of extrinsic camera matrix.
  EXTRINSIC_MATRIX_END_LOC = 18    # End location of extrinsic camera matrix.

  def __init__(self, image_line, scan_id):
    self.scan_id = scan_id
    parts = image_line.strip().split()
    assert 'I' == parts[0]
    assert self.NUM_TOKENS == len(parts)
    self.index = int(parts[self.INDEX_LOC])
    self.pano_index = int(parts[self.PANO_INDEX_LOC])
    self.name = parts[self.NAME_LOC]
    self.camera_index = int(parts[self.CAMERA_INDEX_LOC])
    self.heading_index = int(parts[self.HEADING_INDEX_LOC])

    # Compute heading and pitch from extrinsing matrix.
    extrinsic_coordinates = [
        float(coord) for coord in
        parts[self.EXTRINSIC_MATRIX_START_LOC:self.EXTRINSIC_MATRIX_END_LOC]
    ]
    rotation_matrix = [
        extrinsic_coordinates[i:i + 3]
        for i in range(0, len(extrinsic_coordinates), 4)
    ]
    self.heading, self.pitch, _ = house_utils.get_euler_angles(rotation_matrix)


class Category(object):
  """Category data in the house segmentation file."""

  NUM_TOKENS = 11            # Number of tokens to expect in line.
  INDEX_LOC = 1              # Location of index.
  CAT_MAP_INDEX_LOC = 2      # Location of category mapping index.
  CAT_MAP_NAME_LOC = 3       # Location of category mapping name.
  MPCAT_INDEX_LOC = 4        # Location of mpcat index.
  MPCAT_NAME_LOC = 5         # Location of mpcat name.

  def __init__(self, category_line, category_map=None):
    """Extract object index and category name for a line in the .house file."""
    parts = category_line.strip().split()
    assert 'C' == parts[0]
    assert self.NUM_TOKENS == len(parts)
    self.index = int(parts[self.INDEX_LOC])
    self.category_mapping_index = int(parts[self.CAT_MAP_INDEX_LOC])
    # Raw category name
    self.category_mapping_name = ' '.join(
        parts[self.CAT_MAP_NAME_LOC].split('#'))
    self.mpcat40_index = int(parts[self.MPCAT_INDEX_LOC])
    self.mpcat40_name = parts[self.MPCAT_NAME_LOC]
    # Cleaned category name
    if category_map:
      self.clean_category_name = self._get_clean_cat_name(
          self.category_mapping_index, category_map)

  def _get_clean_cat_name(self,
                          category_index,
                          category_map,
                          count_cutoff_threshold=5):
    """Map category index to a clean category name instead of raw categeory.

    The clean categories are from the R2R `category_mapping.tsv` file. It
    corrects typos and standardizes the raw categories, which is much more fine
    grained than the mpcat40 categories (which only has 40 categories).
    For more information see:
    https://github.com/niessner/Matterport/blob/master/metadata/category_mapping.tsv

    Args:
      category_index: int; the category mapping index extracted from the
        category line from the .house file.
      category_map: a dict returned by `_load_cat_map()`, containing mappings
        from category index to category names, mpcat40 name, and count.
      count_cutoff_threshold: categories with counts below the threshold are
        replaced with their corresponding mpcat40 names. This is used to
        truncate the long tail of rarely used category names.

    Returns:
      A unicode string for the clean category name.
    """
    cat_map = category_map[category_index]
    if cat_map['count'] >= count_cutoff_threshold:
      clean_name = cat_map['clean_category']
    else:
      clean_name = cat_map['mpcat40']
    return clean_name


class Object(object):
  """Object data in the house segmentation file."""

  NUM_TOKENS = 24  # Number of tokens to expect in line.
  INDEX_LOC = 1    # Location of index.
  REGION_LOC = 2   # Location of index.
  CAT_LOC = 3      # Location of index.
  PX_LOC = 4       # Location of px.
  PY_LOC = 5       # Location of py.
  PZ_LOC = 6       # Location of pz.
  AXIS0_X_LOC = 7    # Location of X axis min of the oriented bbox
  AXIS0_Y_LOC = 8    # Location of Y axis min of the oriented bbox
  AXIS0_Z_LOC = 9    # Location of Z axis min of the oriented bbox
  AXIS1_X_LOC = 10   # Location of X axis max of the oriented bbox
  AXIS1_Y_LOC = 11   # Location of Y axis max of the oriented bbox
  AXIS1_Z_LOC = 12   # Location of Z axis max of the oriented bbox
  RADIUS_X_LOC = 13  # Location of X raidus of the oriented bbox
  RADIUS_Y_LOC = 14  # Location of Y raidus of the oriented bbox
  RADIUS_Z_LOC = 15  # Location of Z raidus of the oriented bbox

  def __init__(self, object_line):
    parts = object_line.strip().split()
    assert 'O' == parts[0]
    assert self.NUM_TOKENS == len(parts)
    self.index = int(parts[self.INDEX_LOC])
    self.region_index = int(parts[self.REGION_LOC])
    self.category_index = int(parts[self.CAT_LOC])
    self.center = (float(parts[self.PX_LOC]), float(parts[self.PY_LOC]),
                   float(parts[self.PZ_LOC]))
    # Oriented bounding box (obbox)
    oriented_bbox = collections.namedtuple('OrientedBbox',
                                           ['axis0', 'axis1', 'radii'])
    axis0 = (float(parts[self.AXIS0_X_LOC]), float(parts[self.AXIS0_Y_LOC]),
             float(parts[self.AXIS0_Z_LOC]))
    axis1 = (float(parts[self.AXIS1_X_LOC]), float(parts[self.AXIS1_Y_LOC]),
             float(parts[self.AXIS1_Z_LOC]))
    radii = (float(parts[self.RADIUS_X_LOC]), float(parts[self.RADIUS_Y_LOC]),
             float(parts[self.RADIUS_Z_LOC]))
    self.obbox = oriented_bbox(axis0, axis1, radii)

  def is_well_formed(self):
    return (self.index >= 0 and self.region_index >= 0 and
            self.category_index >= 0)


class R2RHouseParser(object):
  """Parser to extract various annotations in a house to assist perception."""

  def __init__(self,
               house_file_path,
               category_map_dir=None,
               category_map_file='category_mapping.tsv',
               banned_mp40_cat_index=None):
    """Parses regions, panos, categories and objects from house spec file.

    For more information see:
    https://github.com/niessner/Matterport/blob/master/data_organization.md

    Args:
      house_file_path: Path to scan id house specification file.
      category_map_dir: Dir of category mapping file 'category_mapping.tsv'.
        If not provided, then the `clean_category` will be omitted.
      category_map_file: str; file name for the category mapping table. Use
        default unless the mapping table is different.
      banned_mp40_cat_index: A set of mpcat40 category indices, e.g., {0, 41}
        for (void, unlabled). If provided, then these categories will be ignored
        when extracting objects of each pano.
    """
    # Load category map and banned mp40 categories.
    if category_map_dir:
      assert tf.io.gfile.isdir(
          category_map_dir), '{} must be an existing dir.'.format(
              category_map_dir)
      category_map = _load_cat_map(
          category_map_file, category_map_dir, delimiter='\t')
    else:
      # Default to None and omit `clean_category_name` if dir is not given.
      category_map = None

    if not banned_mp40_cat_index:
      self.banned_mp40_cat_index = _BANNED_MP40_CAT_INDEX
    else:
      self.banned_mp40_cat_index = banned_mp40_cat_index

    self.scan_id = os.path.splitext(os.path.basename(house_file_path))[0]
    with tf.io.gfile.GFile(house_file_path, 'r') as input_file:
      # Skip but check header line.
      assert re.match('^ASCII .*', input_file.readline().strip()) is not None
      house_info = input_file.readline().strip().split()
      assert 29 == len(house_info)
      self.num_images = int(house_info[3])
      self.num_panos = int(house_info[4])
      self.num_objects = int(house_info[8])
      self.num_categories = int(house_info[9])
      self.num_regions = int(house_info[10])
      self.regions = {}
      self.panos = {}
      self.categories = {}
      self.objects = {}
      self.images = {}
      for line in input_file:
        if line[0] == 'R':
          r = Region(line)
          assert r.index not in self.regions
          self.regions[r.index] = r
        elif line[0] == 'P':
          p = Pano(line)
          assert p.index not in self.panos
          self.panos[p.index] = p
        elif line[0] == 'C':
          c = Category(line, category_map)
          assert c.index not in self.categories
          self.categories[c.index] = c
        elif line[0] == 'O':
          o = Object(line)
          assert o.index not in self.objects
          self.objects[o.index] = o
        elif line[0] == 'I':
          i = Image(line, self.scan_id)
          assert i.index not in self.images
          self.images[i.index] = i

      assert self.num_regions == len(self.regions)
      assert self.num_panos == len(self.panos)
      assert self.num_categories == len(self.categories)
      assert self.num_objects == len(self.objects)
      assert self.num_images == len(self.images)

    # Organize objects into regions for easy retrieval later.
    self.pano_name_map = {}
    for p in self.panos.values():
      self.pano_name_map[p.name] = p.index
    self.region_object_map = collections.defaultdict(list)
    for o in self.objects.values():
      self.region_object_map[o.region_index] += [o.index]

    # Add images to the associated panos.
    for image in self.images.values():
      pano = self.get_pano_by_name(image.name)
      pano.images.append(image)

  def __repr__(self):
    return 'Regions: {}, Panos: {}, Cats: {}, Objs: {}'.format(
        self.num_regions, self.num_panos, self.num_categories, self.num_objects)

  def get_pano_objects(self, pano_name):
    """Extract the set of objects given a pano.

    Only returns the closest object of the same mp40 category and skips any
    objects with mp40 category in `self.banned_mp40_cat_index` (e.g. misc,
    void, unlabeled categories).

    Args:
      pano_name: panoromic hash id.

    Returns:
      Dictionary where key is the center of the `RoomObject` and the
      value is the `RoomObject` named tuple. In particular,
      `RoomObject.oriented_bbox` is another named tuple (axis0, axis1, radii)
      containing the axis orientation of the bounding box and the radii of the
      object along each axis.
    """
    pano_id = self.pano_name_map.get(pano_name, None)
    if pano_id is None:  # Note that `pano_id` is int and thus can be 0.

      return {}

    room_objects = {}
    region_index = self.panos[pano_id].region_index
    pano_center = self.panos[pano_id].center
    for object_index in self.region_object_map[region_index]:
      try:
        category = self.categories[self.objects[object_index].category_index]
        # NOTE: 'unknown' objects are sometimes labeled as mpcat40=40 (misc)
        # instead of mpcat40=41 (unlabeled). So we specifically exclude it here.
        if ('unknown' not in category.category_mapping_name.lower()) and (
            category.mpcat40_index not in self.banned_mp40_cat_index):
          object_center = self.objects[object_index].center
          assert object_center not in room_objects, self.objects[object_index]
          room_objects[object_center] = RoomObject(
              object_center,
              np.linalg.norm(np.array(object_center) - np.array(pano_center)),
              category.category_mapping_name,
              category.mpcat40_name,
              (category.clean_category_name if hasattr(
                  category, 'clean_category_name') else None),
              self.objects[object_index].obbox)
      except KeyError:
        # Note that this happens because some objects have been marked with -1
        # categories. We can safely ignore these objects.
        assert self.objects[object_index].category_index == -1
    return room_objects

  def get_pano_by_name(self, pano_name):
    return self.panos[self.pano_name_map[pano_name]]

  def get_panos_graph(self,
                      connections_file,
                      cache_best_paths_and_distances=False):
    """Returns a house_utils.Graph object with the panos as nodes.

    The connectivity file should be a json with a single line, containing one
    dictionary for every pano in the house scan. The fields of this dictionary
    used in this method are 'image_id', a unique string that identifies the
    pano, and 'unobstructed', a list of booleans with length equal to the number
    of panos in the scan, representing whether there is an unobstructed direct
    path from the current pano to the pano with that index.

    A more detailed description of the format of the file can be found at
    https://github.com/peteanderson80/Matterport3DSimulator/tree/master/connectivity.

    Args:
      connections_file: Path to the file containing the connections.
      cache_best_paths_and_distances: Whether or not to cache the best path and
        distance for every pair of nodes in the graph.
    """
    assert tf.io.gfile.exists(connections_file), ('Missing required file: %s' %
                                                  connections_file)

    with tf.io.gfile.GFile(connections_file, 'r') as f:
      connections_info = json.loads(f.readline())

    pano_idx_map = {}  # Dictionary mapping idx in connection file to pano name.
    excluded_pano_names = set()  # Set of names of excluded panos.
    for idx, pano_info in enumerate(connections_info):
      pano_name = pano_info['image_id']
      pano_idx_map[idx] = pano_name
      if not pano_info['included']:
        excluded_pano_names.add(pano_name)

    # We build a dictionary indexed by the id of the panos, with values
    # being a dictionary of connected panos to a house_utils.ConnectionInfo
    # storing the distance and heading_angle between them.
    nodes_dict = collections.defaultdict(house_utils.NodeInfo)
    for pano_info in connections_info:
      current_pano = self.get_pano_by_name(pano_info['image_id'])
      if current_pano.name in excluded_pano_names:
        continue  # Don't add excluded panos to the graph.

      for index, is_unobstructed in enumerate(pano_info['unobstructed']):
        connected_pano = self.get_pano_by_name(pano_idx_map[index])
        # Path should be unobstructed and target point should be included in the
        # simulator.
        if is_unobstructed and connected_pano.name not in excluded_pano_names:
          distance = house_utils.compute_distance(current_pano.center,
                                                  connected_pano.center)
          heading_angle = house_utils.compute_heading_angle(
              current_pano.center,
              connected_pano.center,
              radians=True,
              apply_r2r_correction=True)
          pitch_angle = house_utils.compute_pitch_angle(
              current_pano.center, connected_pano.center, radians=True)

          connection_info = house_utils.ConnectionInfo(distance, heading_angle,
                                                       pitch_angle)
          nodes_dict[current_pano.name].connections[connected_pano.name] = (
              connection_info)

    graph = house_utils.Graph(nodes_dict, cache_best_paths_and_distances)
    for node in graph.nodes.keys():
      graph.nodes[node].coords = self.get_pano_by_name(node).center

    return graph


def _load_cat_map(category_map_file='category_mapping.tsv',
                  file_dir='',
                  delimiter='\t'):
  """Load category mapping table from file.

  The mapping table is available at:
  https://github.com/niessner/Matterport/blob/master/metadata/category_mapping.tsv

  Args:
    category_map_file: str; category mapping table file.
    file_dir: str; the dir to `category_map_file`.
    delimiter: str; optional delimiter for the input table.

  Returns:
    A dict that maps category index to category names and counts.
  """
  filepath = os.path.join(file_dir, category_map_file)
  assert tf.io.gfile.exists(filepath), (
      'Missing category mapping file: {}'.format(filepath))
  data = {}
  with tf.io.gfile.GFile(filepath, 'r') as f:
    reader = csv.reader(f, delimiter=delimiter)
    for i, row in enumerate(reader):
      assert len(row) == 18, 'Num columns must be 18.'
      if i == 0:
        header = row
        assert header[0:4] == ['index', 'raw_category', 'category', 'count']
        assert header[-2:] == ['mpcat40index', 'mpcat40']
      else:
        entries = [r.lower() for r in row]
        # entries[0] is the index of each line.
        data[int(entries[0])] = {
            'raw_category': entries[1],
            'clean_category':  # Only take the first part if has '\' in name.
                entries[2].split('/')[0].strip(),
            'count': int(entries[3]),
            'mpcat40index': int(entries[-2]),
            'mpcat40': entries[-1],
        }
  return data
