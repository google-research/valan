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

"""Touchdown environment graph specific utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf


GraphState = collections.namedtuple(
    'GraphState', ('pano_id', 'heading', 'latitude', 'longitude'))
Entry = collections.namedtuple(
    'Entry',
    ('route', 'start_heading', 'end_heading', 'navigation_text', 'route_id'))


class Node(object):

  def __init__(self, pano_id, pano_yaw_angle, lat, lng):
    self.pano_id = pano_id
    self.pano_yaw_angle = pano_yaw_angle
    self.neighbors = {}
    self.latitude = lat
    self.longitude = lng


class Graph(object):
  """Graph."""

  def __init__(self, panoramic_actions=False, pano_heading_window=None):
    """Constructs the navigation graph.

    Args:
      panoramic_actions: Whether to use panoramic action space (boolean)
      pano_heading_window: Max transition heading error (float). Must be set
        if panoramic_actions is True.
    """
    self.nodes = {}
    self.shortest_path_cache = {}
    self._panoramic_actions = panoramic_actions
    self._pano_heading_window = pano_heading_window
    assert ((not self._panoramic_actions) or
            (self._panoramic_actions and self._pano_heading_window)), (
                'pano_heading_window must be set if using pano action space!')

  def add_node(self, pano_id, pano_yaw_angle, lat, lng):
    self.nodes[pano_id] = Node(pano_id, pano_yaw_angle, lat, lng)

  def add_edge(self, start_pano_id, end_pano_id, heading):
    start_node = self.nodes[start_pano_id]
    end_node = self.nodes[end_pano_id]
    start_node.neighbors[heading] = end_node

  def shortest_path(
      self, start_pano_id, end_pano_id, return_none_no_path=False):
    """Returns the shortest path between start and end panos.

    Args:
      start_pano_id: Start panoID from which to start searching for a path.
      end_pano_id: End panoID
      return_none_no_path: What to do if no path from start to end is found.
        If False, raise ValueError. If True, return None.

    Returns:
      Path from start to end as a list of panoIDs. Note that start is excluded,
      but end is included in the returned path.
    """
    cache_key = start_pano_id + '_' + end_pano_id
    if cache_key in self.shortest_path_cache:
      return self.shortest_path_cache[cache_key]

    logging.info('Cache size: %i', len(self.shortest_path_cache))

    q = collections.deque([start_pano_id])

    shortest_paths = {}
    shortest_paths[start_pano_id] = []
    while q:
      node_id = q.popleft()
      # Loop over neighbors.
      for _, v in self.nodes[node_id].neighbors.items():
        # If we have distance for the neighbor, we visited it before so skip.
        if v.pano_id not in shortest_paths:
          shortest_paths[v.pano_id] = shortest_paths[node_id] + [v.pano_id]
          q.append(v.pano_id)

      # If we added the distance for end pano, use it and return.
      if end_pano_id in shortest_paths:
        self.shortest_path_cache[cache_key] = shortest_paths[end_pano_id]
        return shortest_paths[end_pano_id]

    if return_none_no_path:
      return None
    else:
      raise ValueError('No path from start pano to end pano in the nav graph')

  def shortest_path_length(self, start_pano_id, end_pano_id):
    """Returns the length of the shortest path between start and end panos."""
    return len(self.shortest_path(start_pano_id, end_pano_id))

  def get_next_graph_state(self, curr_state, action):
    if self._panoramic_actions:
      return self.get_next_graph_state_panoramic(curr_state, action)
    else:
      return self.get_next_graph_state_discrete(curr_state, action)

  def get_next_graph_state_discrete(self, curr_state, go_towards):
    """Get next state without changing the current state."""
    if go_towards == 'forward':
      neighbors = self.nodes[curr_state.pano_id].neighbors
      if curr_state.heading in neighbors:
        # use current heading to point to the next node
        next_node = neighbors[curr_state.heading]
      else:
        # weird node, stay put
        next_node = self.nodes[curr_state.pano_id]
    elif go_towards == 'left' or go_towards == 'right':
      # if turn left or right, stay at the same node
      next_node = self.nodes[curr_state.pano_id]
    else:
      raise ValueError('Invalid action.')

    next_panoid = next_node.pano_id
    next_heading = self.get_nearest_heading(curr_state, next_node, go_towards)
    next_latitude = next_node.latitude
    next_longitude = next_node.longitude
    return GraphState(next_panoid, next_heading, next_latitude, next_longitude)

  def get_next_graph_state_panoramic(self,
                                     curr_state,
                                     ego_action_heading_deg):
    """Get next state after executing action with the given heading.

    If there is an adjacent pano within +/- 40 degrees in the direction of
    ego_action_heading_deg, then we switch to that pano. Otherwise, remain at
    the current pano_id.

    If we switched panos, the new heading angle is the allocentric heading from
    the old pano to the new pano that we jumped to.
    If no switch took place, then the new heading angle is the allocentric
    conversion of ego_action_heading_deg

    Args:
      curr_state: Current GraphState.
      ego_action_heading_deg: The egocentric heading yaw angle in degrees
        towards which to jump.

    Returns:
      next GraphState
    """
    neighbors = self.nodes[curr_state.pano_id].neighbors
    neighbor_headings, neighbor_nodes = zip(*neighbors.items())
    # Compute allocentric action angle
    allo_action_heading_deg = curr_state.heading + ego_action_heading_deg

    # Differences between action angle and heading to each pano
    neighbor_offsets_abs_deg = [self._angle_abs_diff(n, allo_action_heading_deg)
                                for n in neighbor_headings]
    min_idx = np.argmin(neighbor_offsets_abs_deg)
    min_offset = neighbor_offsets_abs_deg[min_idx]
    min_offset_neighbor_heading = neighbor_headings[min_idx]

    # If there is a transition in the selected direction, then do the jump
    if min_offset < self._pano_heading_window:
      next_node = neighbor_nodes[min_idx]
      next_pano_id = next_node.pano_id
      next_heading = min_offset_neighbor_heading
    # If there is no such transition, then stay put. (Should we rotate?)
    else:
      next_pano_id = curr_state.pano_id
      next_node = self.nodes[next_pano_id]
      next_heading = allo_action_heading_deg

    next_latitude = next_node.latitude
    next_longitude = next_node.longitude
    return GraphState(next_pano_id, next_heading, next_latitude, next_longitude)

  def get_nearest_heading(self, curr_state, next_node, go_towards):
    """Get nearest heading at the next node."""
    curr_heading = curr_state.heading
    next_heading = None

    def forward_diff(next_heading, curr_heading):
      return 180 - abs(abs(next_heading - curr_heading) - 180)

    def left_diff(next_heading, curr_heading):
      return (curr_heading - next_heading) % 360

    def right_diff(next_heading, curr_heading):
      return (next_heading - curr_heading) % 360

    diff = float('inf')
    if go_towards == 'forward':
      diff_func = forward_diff
    elif go_towards == 'left':
      diff_func = left_diff
    elif go_towards == 'right':
      diff_func = right_diff
    else:
      return curr_heading

    for heading in next_node.neighbors.keys():
      if heading == curr_heading and go_towards != 'forward':
        # don't match to the current heading when turning
        continue
      diff_ = diff_func(int(heading), int(curr_heading))
      if diff_ < diff:
        diff = diff_
        next_heading = heading

    if next_heading is None:
      next_heading = curr_heading
    return next_heading

  def get_available_next_moves(self, graph_state):
    """Given current node, get available next actions and states."""
    next_actions = ['forward', 'left', 'right']
    next_graph_states = [
        self.get_next_graph_state(graph_state, 'forward'),
        self.get_next_graph_state(graph_state, 'left'),
        self.get_next_graph_state(graph_state, 'right')
    ]
    return next_actions, next_graph_states

  def show_state_info(self, graph_state):
    """Given a graph state, show current state information and available next moves."""
    logging.info('==============================')
    logging.info('Current graph state: %r', graph_state)
    available_actions, next_graph_states = self.get_available_next_moves(
        graph_state)

    logging.info('Available next actions and graph states:')
    for action, next_graph_state in zip(available_actions, next_graph_states):
      logging.info('Action: %s, to graph state: %r, ', action, next_graph_state)
    logging.info('==============================')

  def _clip_angle(self, angle):
    """Restricts an angle (degrees) to a range of [-180, 180)."""
    return -180 + (angle + 180) % 360

  def _angle_abs_diff(self, a, b):
    """Returns absolute difference between two angles."""
    diff = self._clip_angle(self._clip_angle(a) - self._clip_angle(b))
    abs_diff = min(abs(diff), abs(360-diff))
    return abs_diff

  def _f(self, curr_state):
    return self.get_next_graph_state(curr_state, 'forward')

  def _lf(self, curr_state):
    curr_state = self.get_next_graph_state(curr_state, 'left')
    return self.get_next_graph_state(curr_state, 'forward')

  def _rf(self, curr_state):
    curr_state = self.get_next_graph_state(curr_state, 'right')
    return self.get_next_graph_state(curr_state, 'forward')

  def _llf(self, curr_state):
    curr_state = self.get_next_graph_state(curr_state, 'left')
    curr_state = self.get_next_graph_state(curr_state, 'left')
    return self.get_next_graph_state(curr_state, 'forward')

  def _rrf(self, curr_state):
    curr_state = self.get_next_graph_state(curr_state, 'right')
    curr_state = self.get_next_graph_state(curr_state, 'right')
    return self.get_next_graph_state(curr_state, 'forward')

  def _rrrf(self, curr_state):
    curr_state = self.get_next_graph_state(curr_state, 'right')
    curr_state = self.get_next_graph_state(curr_state, 'right')
    curr_state = self.get_next_graph_state(curr_state, 'right')
    return self.get_next_graph_state(curr_state, 'forward')

  def get_golden_actions(self, entry):
    if self._panoramic_actions:
      return self.get_golden_actions_panoramic(entry)
    else:
      return self.get_golden_actions_discrete(entry)

  def get_golden_actions_panoramic(self, entry):
    """Returns a list of golden actions for a path with panoramic act. space.

    The actions here are egocentric heading angles that need to be sequentially
    passed to the method get_next_graph_state_panoramic such as to visit all
    panoids in entry.route.

    Args:
      entry: The annotation entry, where entry.route is the ground truth.

    Returns:
      A sequence of egocentric jump heading angles and a final action 'stop'
    """
    route = entry.route
    start_heading = entry.start_heading

    curr_state = GraphState(route[0], start_heading, 0, 0)
    actions = []
    path = [route[0]]
    for idx, this_panoid in enumerate(route):
      if idx == len(route) - 1:
        continue

      next_panoid = route[idx + 1]
      this_node = self.nodes[this_panoid]

      # Find heading to the neighbor with pano_id = next_panoid
      allo_action_heading = None
      for heading, neighbor in this_node.neighbors.items():
        if neighbor.pano_id == next_panoid:
          allo_action_heading = heading
          break

      assert allo_action_heading is not None, (
          'Misaligned graph and annotations! No neighbor matches route!'
          'Node: {}, Neighbor: {}'.format(this_panoid, next_panoid))

      ego_action_heading = allo_action_heading - curr_state.heading
      ego_action_heading = self._clip_angle(ego_action_heading)

      next_state = self.get_next_graph_state_panoramic(
          curr_state, ego_action_heading)

      assert next_state.pano_id == next_panoid, (
          'Invalid pano_id after attempted transition to next on-route pano.')
      assert abs(next_state.heading % 360 - allo_action_heading % 360) < 1, (
          'Invalid heading after transition. Expected: {}, Found: {}'.format(
              allo_action_heading, next_state.heading))

      # Action - heading to next node, not stopping
      actions.append(ego_action_heading)

      curr_state = next_state
      path.append(next_panoid)

    # Add stop action at the end
    actions.append('stop')
    path += [curr_state.pano_id]
    return actions, path

  def get_golden_actions_discrete(self, entry):
    """Returns a list of golden actions for a path."""
    route = entry.route
    start_heading = entry.start_heading

    curr_state = GraphState(route[0], start_heading, 0, 0)
    actions = []
    path = [route[0]]
    for idx, _ in enumerate(route):
      if idx == len(route)-1:
        continue

      next_panoid = route[idx+1]

      # Order matters, prefer shortest paths.
      f = self._f(curr_state)
      if f.pano_id == next_panoid:
        actions += ['forward']
        path += [next_panoid]
        curr_state = f
        continue

      lf = self._lf(curr_state)
      if lf.pano_id == next_panoid:
        actions += ['left', 'forward']
        path += [curr_state.pano_id, next_panoid]
        curr_state = lf
        continue

      rf = self._rf(curr_state)
      if rf.pano_id == next_panoid:
        actions += ['right', 'forward']
        path += [curr_state.pano_id, next_panoid]
        curr_state = rf
        continue

      llf = self._llf(curr_state)
      if llf.pano_id == next_panoid:
        actions += ['left', 'left', 'forward']
        path += [curr_state.pano_id, curr_state.pano_id, next_panoid]
        curr_state = llf
        continue

      rrf = self._rrf(curr_state)
      if rrf.pano_id == next_panoid:
        actions += ['right', 'right', 'forward']
        path += [curr_state.pano_id, curr_state.pano_id, next_panoid]
        curr_state = rrf
        continue

      # Should not happen.
      # raise ValueError('Found invalid set of action sequence in the dataset.')
      # @valts: Actually might happen in some CrowdDriving areas. So just stop
      # instead of throwing a ValueError.
      break

    actions += ['stop']
    path += [curr_state.pano_id]
    return actions, path


class GraphLoader(object):
  """GraphLoader."""

  def __init__(self, node_file, links_file,
               panoramic_actions=False,
               pano_heading_window=None):
    self.graph = Graph(panoramic_actions=panoramic_actions,
                       pano_heading_window=pano_heading_window)
    self._node_file = node_file
    self._link_file = links_file

  def construct_graph(self):
    """construct_graph."""
    with tf.io.gfile.GFile(self._node_file) as f:
      for line in f:
        line = str(line)
        pano_id, pano_yaw_angle, lat, lng = line.strip().split(',')
        self.graph.add_node(pano_id, int(pano_yaw_angle), float(lat),
                            float(lng))

    with tf.io.gfile.GFile(self._link_file) as f:
      for line in f:
        line = str(line)
        start_pano_id, heading, end_pano_id = line.strip().split(',')
        self.graph.add_edge(start_pano_id, end_pano_id, int(heading))

    num_edges = 0
    for pano_id in self.graph.nodes:
      num_edges += len(self.graph.nodes[pano_id].neighbors)

    logging.info('===== Graph loaded =====')
    logging.info('Number of nodes: %s', len(self.graph.nodes))
    logging.info('Number of edges: %s', num_edges)
    logging.info('========================')
    return self.graph
