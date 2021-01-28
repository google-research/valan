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

"""Tests for valan.r2r.env."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import flags
import networkx as nx
import numpy as np
import tensorflow.compat.v2 as tf
from valan.framework import common
from valan.framework import hparam
from valan.framework import image_features_pb2

from valan.r2r import constants
from valan.r2r import env
from valan.r2r import env_config

FLAGS = flags.FLAGS


class EnvTest(tf.test.TestCase):

  def setUp(self):
    super(EnvTest, self).setUp()
    self.data_dir = FLAGS.test_srcdir + (
        'valan/r2r/testdata')

    self._env_config = hparam.HParams(
        problem='R2R',
        scan_base_dir=self.data_dir,
        data_base_dir=self.data_dir,
        vocab_dir=self.data_dir,
        vocab_file='vocab.txt',
        images_per_pano=36,
        max_conns=14,
        image_encoding_dim=64,
        direction_encoding_dim=256,
        image_features_dir=os.path.join(self.data_dir, 'image_features'),
        instruction_len=50,
        max_agent_actions=6,
        reward_fn=env_config.RewardFunction.get_reward_fn('distance_to_goal'))
    self._runtime_config = common.RuntimeConfig(task_id=0, num_tasks=1)

    self._env = env.R2REnv(
        data_sources=['R2R_small_split'],
        runtime_config=self._runtime_config,
        env_config=self._env_config)

    # For deterministic behavior in test
    np.random.seed(0)

  def _get_pano_id(self, pano_name, scan_id):
    return self._env._scan_info[scan_id].pano_name_to_id[pano_name]

  def testDenseDtw(self):
    scan_id = 0  # testdata only has single scan 'gZ6f7yhEvPG'
    scan_info = self._env._scan_info[scan_id]
    path_names = [
        'ba27da20782d4e1a825f0a133ad84da9', '47d8a8282c1c4a7fb3eeeacc45e9d959',
        '0ee20663dfa34b438d48750ddcd7366c'
    ]
    path = [self._get_pano_id(name, scan_id) for name in path_names]
    distance_0_1 = self._env.get_distance(path[0], path[1], scan_id)
    distance_0_2 = self._env.get_distance(path[0], path[2], scan_id)
    distance_1_2 = self._env.get_distance(path[1], path[2], scan_id)
    expected_dtw_rewards = [((distance_0_1 + distance_0_2) - distance_1_2),
                            (distance_1_2 - 0.)]
    # Step through the trajectory and verify the rewards.
    for i in range(len(expected_dtw_rewards)):
      path_history = path[:i + 1]
      next_pano = path[i + 1]
      dtw = env_config.dense_dtw(path_history, next_pano, path_names, False,
                                 scan_info)
      self.assertAlmostEqual(dtw, expected_dtw_rewards[i])

  def testReset(self):
    scan_id = 0  # testdata only has single scan 'gZ6f7yhEvPG'
    verify_env_output(
        self,
        self._env.reset(),
        expected_reward=0,
        expected_done=False,
        expected_info='',
        expected_time_step=0,
        expected_path_id=1304,
        expected_pano_name='80929af5cf234ae38ac3a2a4e60e4342',
        expected_heading=6.101,
        expected_pitch=0.,
        expected_scan_id=scan_id,
        expected_oracle_action=self._get_pano_id(
            'ba27da20782d4e1a825f0a133ad84da9', scan_id),
        expected_prev_action=np.zeros(320, dtype=np.float32))

  def testStep(self):
    scan_id = 0  # testdata only has single scan 'gZ6f7yhEvPG'
    obs = self._env.reset()
    golden_path = [
        '80929af5cf234ae38ac3a2a4e60e4342', 'ba27da20782d4e1a825f0a133ad84da9',
        '47d8a8282c1c4a7fb3eeeacc45e9d959', '46cecea0b30e4786b673f5e951bf82d4'
    ]
    nav_graph_filepath = os.path.join(
        self.data_dir, 'connections/gZ6f7yhEvPG_connectivity.json')
    graph = load_nav_graph(nav_graph_filepath)

    # Step through the trajectory and verify the env_output.

    for i, action in enumerate(
        [self._get_pano_id(p, scan_id) for p in golden_path[1:]]):
      expected_time_step = i + 1
      expected_heading, expected_pitch, _ = get_heading_pitch_distance(
          graph, golden_path[i], golden_path[i + 1])
      if i + 2 < len(golden_path):
        expected_oracle_action = self._get_pano_id(golden_path[i + 2], scan_id)
      else:
        expected_oracle_action = constants.STOP_NODE_ID
      action_idx = np.argwhere(
          obs.observation[constants.CONN_IDS] == action).item()
      expected_prev_action = obs.observation[constants.CONN_ENC][action_idx]
      obs = self._env.step(action)
      verify_env_output(
          self,
          obs,
          expected_reward=1,  #  Moving towards goal.
          expected_done=False,
          expected_info='',
          expected_time_step=expected_time_step,
          expected_path_id=1304,
          expected_pano_name=golden_path[i + 1],
          expected_heading=expected_heading,
          expected_pitch=expected_pitch,
          expected_scan_id=scan_id,
          expected_oracle_action=expected_oracle_action,
          expected_prev_action=expected_prev_action)

    # Stop at goal pano. Terminating the episode results in resetting the
    # observation to next episode.
    verify_env_output(
        self,
        self._env.step(constants.STOP_NODE_ID),
        expected_reward=4,  # reached goal and stopped
        expected_done=True,  # end of episode
        expected_info='',
        # observation for next episode.
        expected_time_step=0,
        expected_path_id=1304,
        expected_pano_name='80929af5cf234ae38ac3a2a4e60e4342',
        expected_heading=6.101,
        expected_pitch=0.,
        expected_scan_id=scan_id,
        expected_oracle_action=self._get_pano_id(
            'ba27da20782d4e1a825f0a133ad84da9', scan_id),
        expected_prev_action=np.zeros(320, dtype=np.float32))

  def testSetState(self):
    scan_id = 0  # testdata only has single scan 'gZ6f7yhEvPG'
    _ = self._env.reset()
    golden_path = [
        '80929af5cf234ae38ac3a2a4e60e4342', 'ba27da20782d4e1a825f0a133ad84da9',
        '47d8a8282c1c4a7fb3eeeacc45e9d959', '46cecea0b30e4786b673f5e951bf82d4'
    ]
    nav_graph_filepath = os.path.join(
        self.data_dir, 'connections/gZ6f7yhEvPG_connectivity.json')
    graph = load_nav_graph(nav_graph_filepath)
    states = [self._env.get_state()]
    expected_prev_action = [np.zeros(320, dtype=np.float32)]

    # Step through the trajectory and save states.
    for i, action in enumerate(
        [self._get_pano_id(p, scan_id) for p in golden_path[1:]]):
      obs = self._env.get_current_env_output()
      action_idx = np.argwhere(
          obs.observation[constants.CONN_IDS] == action).item()
      expected_prev_action.append(
          obs.observation[constants.CONN_ENC][action_idx])
      self._env.step(action)
      states.append(self._env.get_state())
    self._env.step(constants.STOP_NODE_ID)
    states.append(self._env.get_state())

    # Restore states and verify output
    self._env.set_state(states[0])
    for i in range(len(golden_path)):
      expected_time_step = i
      if i == 0:
        expected_heading = 6.10099983215332
        expected_pitch = 0
      else:
        expected_heading, expected_pitch, _ = get_heading_pitch_distance(
            graph, golden_path[i - 1], golden_path[i])
      if i + 1 == len(golden_path):
        expected_oracle_action = constants.STOP_NODE_ID
      else:
        expected_oracle_action = self._get_pano_id(golden_path[i + 1], scan_id)
      verify_env_output(
          self,
          self._env.get_current_env_output(),
          expected_reward=0 if i == 0 else 1,  #  Moving towards goal.
          expected_done=False,
          expected_info='',
          expected_time_step=expected_time_step,
          expected_path_id=1304,
          expected_pano_name=golden_path[i],
          expected_heading=expected_heading,
          expected_pitch=expected_pitch,
          expected_scan_id=scan_id,
          expected_oracle_action=expected_oracle_action,
          expected_prev_action=expected_prev_action[i])
      if i + 1 < len(golden_path):
        if i % 2 == 0:  # Alternatively set state or choose action to test both.
          self._env.set_state(states[i+1])
        else:
          action = self._get_pano_id(golden_path[i + 1], scan_id)
          self._env.step(action)

  def testGetDistance(self):
    scan_id = 0  # testdata only has single scan 'gZ6f7yhEvPG'
    self.assertAlmostEqual(
        self._env.get_distance(2, 5, scan_id), 2.9398794637365659)

  def testRandom(self):
    scan_id = 0  # testdata only has single scan 'gZ6f7yhEvPG'
    scan_info = self._env._scan_info[scan_id]
    path_names = [
        'ba27da20782d4e1a825f0a133ad84da9', '47d8a8282c1c4a7fb3eeeacc45e9d959',
        '0ee20663dfa34b438d48750ddcd7366c'
    ]
    path = [self._get_pano_id(name, scan_id) for name in path_names]
    path_history = path[0]
    next_pano = path[1]
    np.random.seed(0)
    expected_random_rwd = np.random.uniform(-1, 1)
    np.random.seed(0)
    random_rwd = env_config.random_reward(path_history, next_pano, path_names,
                                          False, scan_info)
    self.assertLessEqual(random_rwd, 1)
    self.assertEqual(random_rwd, expected_random_rwd)

  def testRandomGoal(self):
    scan_id = 0  # testdata only has single scan 'gZ6f7yhEvPG'
    scan_info = self._env._scan_info[scan_id]
    path_names = [
        'ba27da20782d4e1a825f0a133ad84da9', '47d8a8282c1c4a7fb3eeeacc45e9d959',
        '0ee20663dfa34b438d48750ddcd7366c'
    ]
    path = [self._get_pano_id(name, scan_id) for name in path_names]
    path_history = path[:1]
    next_pano = path[1]
    goal_random_rwd = env_config.goal_plus_random_reward(
        path_history, next_pano, path_names, False, scan_info)
    self.assertLessEqual(goal_random_rwd, 4)


def _get_encoding(filename, expected_size):
  protos = []
  # Pretend to read more than 1 values and then verify there is exactly 1.
  for record in tf.data.TFRecordDataset([filename]).take(2):
    protos.append(record.numpy())
  assert len(protos) == 1
  parsed_record = image_features_pb2.ImageFeatures()
  parsed_record.ParseFromString(protos[0])
  enc = np.zeros(expected_size)
  enc[:parsed_record.shape[0]] = np.array(
      parsed_record.value).reshape(parsed_record.shape)
  return enc


def load_nav_graph(filename):
  def distance(pose1, pose2):
    # Euclidean distance between two graph poses.
    return ((pose1['pose'][3]-pose2['pose'][3])**2\
      + (pose1['pose'][7]-pose2['pose'][7])**2\
      + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

  graph = nx.Graph()
  positions = {}
  data = json.load(tf.io.gfile.GFile(filename))
  for i, item in enumerate(data):
    if item['included']:
      for j, conn in enumerate(item['unobstructed']):
        if conn and data[j]['included']:
          positions[item['image_id']] = np.array(
              [item['pose'][3], item['pose'][7], item['pose'][11]])
          assert data[j]['unobstructed'][i], 'Graph should be undirected'
          graph.add_edge(item['image_id'], data[j]['image_id'],
                         weight=distance(item, data[j]))
  nx.set_node_attributes(graph, values=positions, name='position')
  return graph


def get_heading_pitch_distance(graph, from_pano_name, to_pano_name):
  # Calculate heading, pitch, distance directly from the connection graph.
  positions = graph.nodes(data='position')
  heading_vector = positions[to_pano_name] - positions[from_pano_name]
  # R2R dataset heading is defined clockwise from y-axis.
  heading = np.pi / 2.0 - np.arctan2(heading_vector[1], heading_vector[0])
  pitch = np.arctan2(heading_vector[2],
                     (heading_vector[0]**2 + heading_vector[1]**2)**0.5)
  distance = np.linalg.norm(heading_vector)
  return heading, pitch, distance


def verify_env_output(test, env_output, expected_reward, expected_done,
                      expected_info, expected_time_step, expected_path_id,
                      expected_pano_name, expected_heading, expected_pitch,
                      expected_scan_id, expected_oracle_action,
                      expected_prev_action=None):
  test.assertEqual(expected_reward, env_output.reward)
  test.assertEqual(expected_done, env_output.done)
  test.assertEqual(expected_info, env_output.info)
  obs = env_output.observation
  test.assertEqual(expected_time_step, obs[constants.TIME_STEP])
  test.assertEqual(expected_path_id, obs[constants.PATH_ID])
  test.assertEqual(
      test._env._scan_info[expected_scan_id]
      .pano_name_to_id[expected_pano_name], obs[constants.PANO_ID])
  # Allow small differences since env.py uses house files and the test calcs are
  # based on the connection graphs, which are in turn based on the matterport
  # camera poses files. Camera positions differ by a few centimeters.
  test.assertAlmostEqual(expected_heading, obs[constants.HEADING].item(),
                         places=1)
  test.assertAlmostEqual(expected_pitch, obs[constants.PITCH].item(),
                         places=1)
  test.assertEqual(expected_scan_id, obs[constants.SCAN_ID])
  test.assertEqual(expected_oracle_action, obs[constants.ORACLE_NEXT_ACTION])
  if expected_prev_action is not None:
    np.testing.assert_array_almost_equal(expected_prev_action,
                                         obs[constants.PREV_ACTION_ENC])
  instr_len = (obs[constants.INS_TOKEN_IDS] > 0).sum()
  test.assertEqual(instr_len, obs[constants.INS_LEN])
  valid_conn_mask = obs[constants.CONN_IDS] >= 0
  np.testing.assert_array_equal(valid_conn_mask, obs[constants.VALID_CONN_MASK])
  raw_pano_enc = _get_encoding(
      os.path.join(test._env_config.image_features_dir,
                   '{}_viewpoints_proto'.format(expected_pano_name)),
      [36, 64])
  np.testing.assert_array_almost_equal(
      raw_pano_enc, obs[constants.PANO_ENC][:, :64])
  raw_conn_enc = _get_encoding(
      os.path.join(test._env_config.image_features_dir,
                   '{}_connections_proto'.format(expected_pano_name)),
      [14, 64])
  np.testing.assert_array_almost_equal(
      raw_conn_enc, obs[constants.CONN_ENC][:, :64])
  # Tests next_golden_action_enc.
  if constants.NEXT_GOLDEN_ACTION_ENC in obs:
    current_pano_id = obs[constants.PANO_ID]
    current_pano_index = obs[constants.GOLDEN_PATH].index(current_pano_id)
    next_pano_id = obs[constants.GOLDEN_PATH][current_pano_index + 1]
    if next_pano_id == -1:
      # Replace invalid node to stop node.
      next_pano_id = 0
    conn_idx = np.argwhere(test._env._scan_info[expected_scan_id]
                           .conn_ids[current_pano_id] == next_pano_id).item()
    np.testing.assert_array_almost_equal(
        raw_conn_enc[conn_idx], obs[constants.NEXT_GOLDEN_ACTION_ENC][:64])


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
