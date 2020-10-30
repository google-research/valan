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

"""Graph utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random
import networkx as nx
import numpy as np


class ConnectionInfo(object):

  def __init__(self, distance=1, heading=0, pitch=0):
    self.distance = distance
    self.heading = heading
    self.pitch = pitch


class NodeInfo(object):
  """Contains information of a node in a graph."""

  def __init__(self, connections=None, coords=None):
    """Initializes a NodeInfo object.

    Args:
      connections: Dictionary mapping connected node_ids to ConnectionInfo
        objects.
      coords: (x, y, z) coordinates of the node.
    """
    self.connections = connections if connections is not None else {}
    # Dictionary mapping target node_ids to their distance in the graph.
    self.distances = {}
    # Dictionary mapping target node_ids to a sequence of node_ids (the
    # shortest path to the target).
    self.best_paths = {}
    self.coords = coords  # xyz coordinates.


class Graph(object):
  """Graph utils."""

  def __init__(self, nodes_dict, cache_best_paths_and_distances=False):
    """Initializes the graph.

    Args:
      nodes_dict: Dictionary mapping node ids to NodeInfos.
      cache_best_paths_and_distances: Whether or not to cache the best path and
        distance for every pair of nodes in the graph.
    """
    self.nodes = nodes_dict
    self.best_paths_and_distances_cached = False
    if cache_best_paths_and_distances:
      self.cache_best_paths_and_distances()

  def _get_one_random_path_from_source(self, source, path_len):
    """Generate one paths from source with length path_len.

    Args:
      source: Source node.
      path_len: Maximum path length.

    Returns:
      One list with node ids.
    """
    now_id = source
    path_ids = []
    for _ in range(path_len):
      path_ids.append(now_id)
      # The graph is bidirectional. There must be a valid next step.
      assert self.get_connections(now_id)
      now_id = random.choice(list(self.get_connections(now_id).keys()))
    return path_ids

  def get_connections(self, source):
    return self.nodes[source].connections

  def get_connection(self, source, target):
    return self.nodes[source].connections[target]

  def get_distance(self, source, target):
    return self.nodes[source].distances[target]

  def get_neighbors(self, source):
    # Sort for consistency.
    return sorted(self.get_connections(source).keys())

  def get_random_paths_from_source(self, source, path_len, num_paths):
    """Sample num_paths paths which has path_len lengths and start from source.

    Args:
      source: Source node.
      path_len: Maximum path length.
      num_paths: Number of sampled paths.

    Returns:
      A list of paths. Every path is a list of node ids.
    """
    result = []
    for _ in range(num_paths):
      result.append(self._get_one_random_path_from_source(source, path_len))
    return result

  def merge(self, graph):
    """Merge nodes from other graph to this one."""
    self.nodes.update(graph.nodes)

  def cache_best_paths_and_distances(self):
    graph = nx.Graph()
    for source in self.nodes:
      for target in self.get_neighbors(source):
        connection_info = self.get_connection(source, target)
        graph.add_edge(source, target, weight=connection_info.distance)
    for source, (distances, best_paths) in nx.all_pairs_dijkstra(graph):
      self.nodes[source].distances.update(distances)
      self.nodes[source].best_paths.update(best_paths)
    self.best_paths_and_distances_cached = True

  def get_closest_neighbor_to_target(self, source, target):
    """Finds the connected node closest to a given target.

    This method relies on pre-computed distances from each node to others.

    Args:
      source: node_id of the source node.
      target: node_id of the target node.

    Returns:
      best_neighbor: node_id of the best connected node.
    """
    if source == target:
      # If the pano is the target, we return itself as a convention.
      return source
    return self.nodes[source].best_paths[target][1]

  def get_shortest_path_to_target(self, source, target):
    """Finds the shortest path from source to a given target.

    This method relies on pre-computed distances from each node to others.

    Args:
      source: node_id of the source node.
      target: node_id of the target node.

    Returns:
      shortest_path: a list of node ids on the shortest path.
    """
    if source == target:
      # If the pano is the target, we return itself as a convention.
      return [source]
    return self.nodes[source].best_paths[target]

  def get_edit_distance(self, path1, path2):
    """Returns the minimum cost of edition from one path to another.

    We only take into account edge operations, weighting them by their distance
    attribute.

    Args:
      path1: A list of node ids in the graph representing the first path.
      path2: A list of node ids in the graph representing the second path.

    Returns:
      The edit distance, analogous to Levenshtein distance for strings, weighted
        by the distances of each edge in the graph.
    """

    def get_nx_subgraph(path):
      """Creates a networkx graph from a list of nodes."""
      graph = nx.Graph()
      for idx in range(len(path) - 1):
        cur_node = path[idx]
        next_node = path[idx + 1]
        connection_info = self.get_connection(cur_node, next_node)
        graph.add_edge(
            cur_node,
            next_node,
            weight=connection_info.distance,
            src=cur_node,
            tgt=next_node)

      return graph

    graph1 = get_nx_subgraph(path1)
    graph2 = get_nx_subgraph(path2)

    def edge_subst_cost(edge1, edge2):
      """Substituition cost for edges."""
      if edge1['src'] == edge2['src'] and edge1['tgt'] == edge2['tgt']:
        return 0
      return abs(edge1['weight']) + abs(edge2['weight'])

    edge_del_or_ins_cost = lambda e: abs(e['weight'])
    node_op_cost = lambda *args, **kwargs: 0  # Only measure edge similarity.
    return nx.graph_edit_distance(
        graph1,
        graph2,
        node_subst_cost=node_op_cost,
        node_del_cost=node_op_cost,
        node_ins_cost=node_op_cost,
        edge_subst_cost=edge_subst_cost,
        edge_del_cost=edge_del_or_ins_cost,
        edge_ins_cost=edge_del_or_ins_cost)

  def get_closest_neighbor_to_path(self, source, path, strategy='greedy'):
    """Returns the closes neighbor to a given path.

    This method relies on pre-computed distances and best_neighbors from each
    node to others.

    Args:
      source: Id of the source node.
      path: List of node ids composed by the target nodes of a path.
      strategy: One of {'greedy', 'path_first'}. Greedy strategy looks for the
        neighbor closest to last node in the path. Path first looks for the
        neighbor closest to any node in the path.

    Returns:
      Id of the closest neighbor to the path.
    """
    assert self.best_paths_and_distances_cached, (
        'Best paths and distances must be cached. Please call '
        'cache_best_paths_and_distances method before this or '
        'instantiate the class with cache_best_paths_and_distances=True')

    # If the source is the target, we return itself as a convention.
    if source == path[-1]:
      return source
    # Otherwise, if the node is part of the path, we return the next node on
    # the path, no matter the strategy.
    if source in path:
      source_index = path.index(source)
      best_neighbor = path[source_index + 1]
    elif strategy == 'greedy':
      # Get neighbor that leads us closest to the last node in the path.
      best_neighbor = self.get_closest_neighbor_to_target(source, path[-1])
    elif strategy == 'path_first':
      # Get neighbor that leads us closes to any node in the path.
      best_neighbor = None
      min_distance = float('inf')
      for target in path:
        current_distance = self.get_distance(source, target)
        if current_distance < min_distance:
          best_neighbor = self.get_closest_neighbor_to_target(source, target)
          min_distance = current_distance
    else:
      raise ValueError(
          'Unknown strategy. Supported values are {greedy, path_first}.')

    return best_neighbor

  def to_nx(self):
    """Returns networkx graph."""
    nx_graph = nx.Graph()
    for source in self.nodes:
      nx_graph.add_node(source, coords=self.nodes[source].coords[:2])
      for target in self.get_neighbors(source):
        connection_info = self.get_connection(source, target)
        nx_graph.add_edge(source, target, weight=connection_info.distance)
    return nx_graph


def compute_arc_distance(angle_1, angle_2, radians=True):
  """Returns the distance between two angles in a circle."""
  whole_circle = 2 * math.pi if radians else 360
  # First we compute the absolute difference between the angles.
  abs_diff = abs(angle_1 - angle_2) % whole_circle
  # The distance is either that or 2*pi - that (the other way around).
  return min(abs_diff, whole_circle - abs_diff)


def get_euler_angles(rotation_matrix, radians=True):
  """Returns the euler angles from a rotation matrix.

  Euler angles are, in order, yaw (heading), pitch (elevation) and roll.
  Angles are returned in degrees unless specified otherwise by setting the
  parameter radians to True.

  Args:
    rotation_matrix: 3x3 rotation matrix.
    radians: Whether to output values in radians or in degrees.

  Returns:
    Triplet of angles yaw (heading), pitch (elevation) and roll.
  """
  pi = math.pi
  yaw = -math.atan2(rotation_matrix[0][1], rotation_matrix[0][0]) % (2 * pi)
  pitch = -math.atan2(rotation_matrix[1][2], rotation_matrix[2][2]) % (2 * pi)
  roll = math.asin(rotation_matrix[0][2]) % (2 * pi)
  if not radians:
    yaw, pitch, roll = [math.degrees(angle) for angle in [yaw, pitch, roll]]
  return yaw, pitch, roll


def compute_heading_angle(source_coordinates,
                          target_coordinates,
                          radians=True,
                          apply_r2r_correction=True):
  """Computes heading angle in the xy plane given two xyz coordinates.

  Args:
    source_coordinates: Source triplet with xyz coordinates.
    target_coordinates: Target triplet with xyz coordinates.
    radians: Whether to output in radians or degrees.
    apply_r2r_correction: Whether to use the R2R dataset heading definition.
        If true, heading is computed with respect to the y-axis, increasing
        clockwise. See
        https://github.com/peteanderson80/Matterport3DSimulator/blob/master/README.md#simulator-api.

  Returns:
    Heading angle in the xy plane. Angle is computed with respect to the y-axis,
    increasing clockwise, unless apply_r2r_correction is false.
  """
  source_x, source_y, _ = source_coordinates
  target_x, target_y, _ = target_coordinates
  angle = math.atan2(target_y - source_y, target_x - source_x)

  if apply_r2r_correction:
    angle = 0.5 * math.pi - angle

  if radians:
    return angle % (2 * math.pi)
  return math.degrees(angle) % 360


def compute_pitch_angle(source_coordinates, target_coordinates, radians=True):
  """Computes pitch (elevation) angle given two xyz coordinates.

  Args:
    source_coordinates: Source triplet with xyz coordinates.
    target_coordinates: Target triplet with xyz coordinates.
    radians: Whether to output in radians or degrees.

  Returns:
    Pitch angle. 0 means neutral elevation, angles increase towards positive
    z coordinates. In radians, pi means looking at the ceiling and -pi means
    looking at the floor.
  """
  source_x, source_y, source_z = source_coordinates
  target_x, target_y, target_z = target_coordinates

  delta_x = target_x - source_x
  delta_y = target_y - source_y
  delta_z = target_z - source_z
  delta_xy = math.sqrt(delta_x**2 + delta_y**2)

  pitch = math.atan2(delta_z, delta_xy)

  if radians:
    return pitch
  return math.degrees(pitch)


def compute_distance(source_coordinates, target_coordinates):
  """Computes distance between two xyz coordinates.

  Args:
    source_coordinates: Source triplet with xyz coordinates.
    target_coordinates: Target triplet with xyz coordinates.

  Returns:
    Distance between the points.
  """
  return np.linalg.norm(
      np.array(source_coordinates) - np.array(target_coordinates))
