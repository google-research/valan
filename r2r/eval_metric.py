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

"""Utilities function for evaluations in R2R."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections

from absl import flags

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  
import networkx
import numpy as np

import tensorflow.compat.v2 as tf
from valan.framework import eval_metric as base_eval_metric

from valan.r2r import constants

FLAGS = flags.FLAGS

SUCCESS_THRESHOLD = 3
DEFAULT_PARAM = {
    'arrowsize': 15,
    'width': 1.3,
    'node_size': 20,
}


def _get_path_length(path, scan, environment):
  return sum([
      environment.get_distance(node1, node2, scan)
      for node1, node2 in zip(path[:-1], path[1:])])


def _get_current_scan(env_output_list):
  return env_output_list[0].observation[constants.SCAN_ID]


def _get_ground_truth_path(action_list, env_output_list):
  del action_list
  return env_output_list[0].observation[constants.GOLDEN_PATH]


def _get_predicted_path(action_list, env_output_list):
  return [e.observation[constants.PANO_ID] for e in env_output_list[:-1]
         ] + [action_list[-1]]


def get_path_length(action_list, env_output_list, environment):
  current_scan = _get_current_scan(env_output_list)
  all_panos = _get_predicted_path(action_list, env_output_list)
  return _get_path_length(all_panos, current_scan, environment)


def get_oracle_success(action_list, env_output_list, environment):
  """Get oracle success."""
  all_panos = _get_predicted_path(action_list, env_output_list)
  goal_pano_id = env_output_list[0].observation[constants.GOAL_PANO_ID]
  current_scan = _get_current_scan(env_output_list)
  dist_to_goal = []
  for p in all_panos:
    dist_to_goal.append(environment.get_distance(p, goal_pano_id, current_scan))
  if min(dist_to_goal) < SUCCESS_THRESHOLD:
    return 1.
  return 0.


def get_success_rate(action_list, env_output_list, environment):
  """Get success rate.

  Args:
    action_list: List of actions.
    env_output_list: List of observations in the trajectories.
    environment: Testing environment.

  Returns:
    1. if the endpoint is within SUCCESS_THRESHOLD with the oracle destination.
    0. if not.
  """
  if get_navigation_error(action_list, env_output_list,
                          environment) < SUCCESS_THRESHOLD:
    return 1.
  return 0.


def get_spl(action_list, env_output_list, environment):
  """Returns success rate weighted by predicted path length.

  Args:
    action_list: List of actions.
    env_output_list: List of observations in the trajectories.
    environment: Testing environment.

  Returns:
    0. if unsuccessful,
    shortest_path_len / max(shortest_path_len, predicted_path_len) otherwise.
  """
  if get_success_rate(action_list, env_output_list, environment) == 0.:
    return 0.
  start_pano_id = env_output_list[0].observation[constants.PANO_ID]
  goal_pano_id = env_output_list[0].observation[constants.GOAL_PANO_ID]
  current_scan = _get_current_scan(env_output_list)
  shortest_path_len = environment.get_distance(start_pano_id, goal_pano_id,
                                               current_scan)
  predicted_path_len = get_path_length(action_list, env_output_list,
                                       environment)
  if shortest_path_len > 0.:
    return shortest_path_len / max(shortest_path_len, predicted_path_len)
  else:
    return 0.


def get_navigation_error(action_list, env_output_list, environment):
  """Get navigation error, which is the distance of last pano to goal.

  Args:
    action_list: List of actions.
    env_output_list: List of observations in the trajectories.
    environment: Testing environment.

  Returns:
    1. if the endpoint is within 5m with the oracle destination.
    0. if not.
  """
  all_panos = _get_predicted_path(action_list, env_output_list)
  goal_pano_id = env_output_list[0].observation[constants.GOAL_PANO_ID]
  current_scan = _get_current_scan(env_output_list)
  if all_panos[-1] == constants.STOP_NODE_ID:
    assert all_panos[-2] is not constants.STOP_NODE_ID
    last_node = all_panos[-2]
  else:
    last_node = all_panos[-1]
  return environment.get_distance(last_node, goal_pano_id, current_scan)


def get_num_steps_before_stop(action_list, env_output_list, environment):
  """Get number of steps taken by the agent in this episode.

  NOTE that STOP action is not counted as a step.

  Args:
    action_list: List of actions.
    env_output_list: List of observations in the trajectories.
    environment: Testing environment.

  Returns:
    The number of steps taken by the agent before predicting STOP action.
  """
  del environment
  all_panos = _get_predicted_path(action_list, env_output_list)
  if all_panos[-1] == constants.STOP_NODE_ID:
    assert all_panos[-2] is not constants.STOP_NODE_ID
    panos_before_stop = all_panos[:-1]
  else:
    panos_before_stop = all_panos
  return len(panos_before_stop) - 1


def get_undisc_episode_reward(action_list, env_output_list, environment):
  del action_list, environment
  return sum([env_output.reward for env_output in env_output_list])


def get_dtw(action_list, env_output_list, environment):
  """Dynamic Time Warping (DTW).

  Muller, Meinard. "Dynamic time warping."
  Information retrieval for music and motion (2007): 69-84.

  Dynamic Programming implementation, O(NM) time and memory complexity.

  Args:
    action_list: List of actions.
    env_output_list: List of observations in the trajectories.
    environment: Testing environment.

  Returns:
    The DTW score.
  """
  invalid_panos = [constants.STOP_NODE_ID, constants.INVALID_NODE_ID]
  obs_panos = _get_predicted_path(action_list, env_output_list)
  obs_panos = [pano for pano in obs_panos if pano not in invalid_panos]

  golden_panos = env_output_list[0].observation[constants.GOLDEN_PATH]
  golden_panos = [pano for pano in golden_panos if pano not in invalid_panos]

  scan_id = env_output_list[0].observation[constants.SCAN_ID]
  dtw_matrix = base_eval_metric.get_dtw_matrix(
      obs_panos,
      golden_panos,
      lambda pano1, pano2: environment.get_distance(pano1, pano2, scan_id))

  golden_path_length = _get_path_length(golden_panos, scan_id, environment)
  # Note: We normalize DTW (which is sum of distances in the graph) by
  # golden_path_length.
  return dtw_matrix[len(obs_panos)][len(golden_panos)] / golden_path_length


def get_norm_dtw(action_list, env_output_list, environment):
  """Returns normalized DTW.

  "Effective and General Evaluation for Instruction Conditioned Navigation using
  Dynamic Time Warping" 2019 Magalhaes et al. https://arxiv.org/abs/1907.05446

  Args:
    action_list: List of actions.
    env_output_list: List of observations in the trajectories.
    environment: Testing environment.

  Returns:
    Value of normalized DTW.
  """
  invalid_panos = [constants.STOP_NODE_ID, constants.INVALID_NODE_ID]
  golden_panos = env_output_list[0].observation[constants.GOLDEN_PATH]
  golden_panos = [pano for pano in golden_panos if pano not in invalid_panos]

  obs_panos = [e.observation[constants.PANO_ID] for e in env_output_list[:-1]]
  obs_panos.append(action_list[-1])
  obs_panos = [pano for pano in obs_panos if pano not in invalid_panos]
  scan_id = env_output_list[0].observation[constants.SCAN_ID]
  dtw_matrix = base_eval_metric.get_dtw_matrix(
      obs_panos,
      golden_panos,
      lambda pano1, pano2: environment.get_distance(pano1, pano2, scan_id))

  dtw = dtw_matrix[len(obs_panos)][len(golden_panos)]
  return np.exp(-1. * dtw / (SUCCESS_THRESHOLD * len(golden_panos)))


def get_sdtw(action_list, env_output_list, environment):
  """Returns success rate normalized by DTW.

  "Effective and General Evaluation for Instruction Conditioned Navigation using
  Dynamic Time Warping" 2019 Magalhaes et al. https://arxiv.org/abs/1907.05446

  Args:
    action_list: List of actions.
    env_output_list: List of observations in the trajectories.
    environment: Testing environment.

  Returns:
    Value of success rate normalized by DTW.
  """
  success_rate = get_success_rate(action_list, env_output_list, environment)
  if not success_rate:
    return 0.

  return get_norm_dtw(action_list, env_output_list, environment)


def get_cls(action_list, env_output_list, environment):
  """Coverage weighted by Lengh Score (CLS).

  "Stay on the Path: Instruction Fidelity in Vision and Language Navigation."
  Jain et al., ACL 2019. https://arxiv.org/abs/1905.12255

  Args:
    action_list: List of actions.
    env_output_list: List of observations in the trajectories.
    environment: Testing environment.

  Returns:
    The CLS score.
  """
  invalid_panos = [constants.STOP_NODE_ID, constants.INVALID_NODE_ID]
  obs_panos = _get_predicted_path(action_list, env_output_list)
  obs_panos = [pano for pano in obs_panos if pano not in invalid_panos]

  golden_panos = env_output_list[0].observation[constants.GOLDEN_PATH]
  golden_panos = [pano for pano in golden_panos if pano not in invalid_panos]

  scan_id = env_output_list[0].observation[constants.SCAN_ID]

  def _distance_to_path(query_node, path):
    return np.min(
        [environment.get_distance(query_node, node, scan_id) for node in path])
  coverage = np.mean([
      np.exp(-1 * _distance_to_path(node, obs_panos) / SUCCESS_THRESHOLD)
      for node in golden_panos])

  obs_path_length = _get_path_length(obs_panos, scan_id, environment)
  golden_path_length = _get_path_length(golden_panos, scan_id, environment)

  expected_length = coverage * golden_path_length
  if expected_length == 0 and obs_path_length == 0:  # include the edge case
    return 1
  length_score = expected_length / (
      expected_length + np.abs(expected_length - obs_path_length))
  return coverage * length_score


def _draw_path(path, pos, color='black'):
  """Draw one path in one networkx graph."""
  path_graph = networkx.DiGraph()
  path_graph.add_node(path[0])
  edges = collections.defaultdict(list)
  for idx in range(len(path) - 1):
    if path[idx +
            1] not in [constants.STOP_NODE_NAME, constants.INVALID_NODE_NAME]:
      path_graph.add_edge(path[idx], path[idx + 1])
      start = min(path[idx], path[idx + 1])
      end = max(path[idx], path[idx + 1])
      edges[(start, end)].append((path[idx], path[idx + 1], color))
  networkx.draw_networkx_nodes(
      path_graph, pos, edge_color=color, node_color=color, **DEFAULT_PARAM)
  return path_graph, edges


def get_visualization_image(action_list, env_output_list, environment):
  """Get visualization images.

  Args:
    action_list: List of actions.
    env_output_list: List of observations in the trajectories.
    environment: Testing environment.

  Returns:
    A list which only have one numpy array. The numpy array is the image matrix
    with shape [width, height, 3]
  """

  def is_valid_node(node):
    if node == constants.STOP_NODE_NAME or node == constants.INVALID_NODE_NAME:
      return False
    return True

  current_scan = _get_current_scan(env_output_list)
  ground_truth_path = _get_ground_truth_path(action_list, env_output_list)
  predicted_path = _get_predicted_path(action_list, env_output_list)

  # Map from pano_id to pano_name.
  ground_truth_path = [
      environment.pano_id_to_name(pano_id, current_scan)
      for pano_id in ground_truth_path
  ]
  predicted_path = [
      environment.pano_id_to_name(pano_id, current_scan)
      for pano_id in predicted_path
  ]

  fig = plt.figure(figsize=(2, 2))
  ax = fig.add_subplot(1, 1, 1)
  # Get a graph_utils.Graph instance.
  base_graph = environment.get_scan_graph(current_scan)
  # Convert to networkx Graph.
  base_graph = base_graph.to_nx()
  node_pos_dict = networkx.get_node_attributes(base_graph, 'coords')

  all_nodes = ground_truth_path + predicted_path
  all_nodes = list(filter(is_valid_node, all_nodes))

  # Zoom-in to the predicted subgraph.
  cut = 1.3
  xs = [node_pos_dict[node][0] for node in all_nodes]
  ys = [node_pos_dict[node][1] for node in all_nodes]
  min_x, max_x = min(xs), max(xs)
  min_y, max_y = min(ys), max(ys)
  x_range, y_range = cut * (max_x - min_x), cut * (max_y - min_y)
  max_range = max(x_range, y_range)
  center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
  ax.set_xlim(center_x - max_range / 2, center_x + max_range / 2)
  ax.set_ylim(center_y - max_range / 2, center_y + max_range / 2)
  ax.set_ymargin(0.01)

  networkx.draw(
      base_graph,
      node_pos_dict,
      edge_color='lightgrey',
      node_color='lightgrey',
      node_size=20,
      width=0.3)

  ground_truth_graph, edges = _draw_path(
      ground_truth_path,
      node_pos_dict,
      color='cornflowerblue')
  predicted_graph, pr_edges = _draw_path(
      predicted_path, node_pos_dict, color='orange')

  for key, value in pr_edges.items():
    edges[key].extend(value)

  combined_graph = networkx.compose(ground_truth_graph, predicted_graph)
  overlap_offset = 0.015

  # Draw the edges with a small offset to avoid overlap.
  for (start, end), edge_list in edges.items():
    edge_dir = np.array(node_pos_dict[end]) - np.array(node_pos_dict[start])
    offset_dir = np.array([edge_dir[1], -edge_dir[0]])
    offset_dir /= np.linalg.norm(offset_dir) / max_range
    edge_list = edge_list[:3]
    num_edges = len(edge_list)
    for edge_idx, (s, e, color) in enumerate(edge_list):
      offset = (edge_idx - (num_edges - 1.) / 2) * overlap_offset
      shifted_pos = {
          key:
          (val[0] + offset * offset_dir[0], val[1] + offset * offset_dir[1])
          for key, val in node_pos_dict.items()
      }
      networkx.draw_networkx_edges(
          combined_graph,
          shifted_pos,
          edgelist=[(s, e)],
          edge_color=color,
          **DEFAULT_PARAM)

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)
  return [data]


def get_goal_progress(action_list, env_output_list, environment):
  """Get goal progress, which is the reduced distance to the goal room.

  For Navigation from Dialogue History task.

  Args:
    action_list: List of actions.
    env_output_list: List of observations in the trajectories.
    environment: Testing environment.

  Returns:
    the reduced distance to the goal room.
  """
  all_panos = _get_predicted_path(action_list, env_output_list)
  current_scan = _get_current_scan(env_output_list)
  start_node = all_panos[0]
  if all_panos[-1] == constants.STOP_NODE_ID:
    assert all_panos[-2] is not constants.STOP_NODE_ID
    last_node = all_panos[-2]
  else:
    last_node = all_panos[-1]
  padded_goal_room_panos = env_output_list[0].observation[
      constants.GOAL_ROOM_PANOS]
  goal_room_panos = [
      _ for _ in padded_goal_room_panos if _ != constants.INVALID_NODE_ID
  ]
  dist_to_goal_start = None
  dist_to_goal_end = None
  for goal_pano in goal_room_panos:
    d = environment.get_distance(start_node, goal_pano, current_scan)
    if dist_to_goal_start is None or d < dist_to_goal_start:
      dist_to_goal_start = d
    d = environment.get_distance(last_node, goal_pano, current_scan)
    if dist_to_goal_end is None or d < dist_to_goal_end:
      dist_to_goal_end = d
  return dist_to_goal_start - dist_to_goal_end


def get_score_label(action_list, env_output_list, action_output, environment):
  del action_list, environment
  # sigmoid
  return [(tf.sigmoid(action_output[0][0]),
           env_output_list[0].observation[constants.LABEL])]


class DiscriminatorMetric(object):
  """Eval metric for discriminator."""

  def __init__(self, mode):
    self._mode = mode

  def get_score_label(self, action_list, env_output_list, action_output,
                      environment):
    """Gets the sigmoid probability and GT labels for DiscriminatorAgent."""
    score_label = get_score_label(action_list, env_output_list, action_output,
                                  environment)
    if self._mode == 'predict':
      instruction_ids = self._get_instruction_ids(env_output_list)
      return [(score_label[0][0], score_label[0][1], instruction_ids)]
    else:
      return score_label

  def get_score_label_v2(self, action_list, env_output_list, agent_output,
                         environment):
    """Gets the probability score and GT labels for DiscriminatorAgentV2."""
    del action_list, environment
    # Remove the unused timestep dimension.
    labels = tf.squeeze(agent_output.policy_logits['labels'], axis=0)
    logits = tf.squeeze(agent_output.baseline, axis=0)
    if self._mode == 'predict':
      instruction_ids = self._get_instruction_ids(env_output_list)
      return [(tf.sigmoid(logits), labels, instruction_ids)]
    else:
      return [(tf.sigmoid(logits), labels)]

  def _get_instruction_ids(self, env_output_list):
    """Get instruction token ids for identification."""
    instruction_ids = [
        int(x)
        for x in env_output_list[0].observation[constants.INS_TOKEN_IDS]
        if x != 0
    ]
    return instruction_ids
