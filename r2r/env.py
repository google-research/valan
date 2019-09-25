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

"""Environment class for R2R problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import os
import re
import string

from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf
from valan.framework import base_env
from valan.framework import common

from valan.r2r import constants
from valan.r2r import env_config as default_env_config
from valan.r2r import house_parser
from valan.r2r import image_features_pb2

INF = 1e9


class R2REnv(base_env.BaseEnv):
  """Environment class for R2R problem.

  An object of this class holds three main types of data:
  1. The graph of the underlying environment(s) - If running in distributed
     mode, every instance loads only selected scans and the connectivity graph
     of those scans.
  2. The paths from the selected environment(s). These are read from
     corresponding JSON files and filtered to contain only the selected
     environments.
  3. The image features of all the panos from the selected environment(s). The
     image features are expected to be in image_features_pb2.ImageFeatures
     format serialized using TFRecordWriter.
  """

  def __init__(self, data_sources, runtime_config, env_config=None):
    """Initializes an instance of R2REnv.

    Args:
      data_sources: A list of strings. The paths from 'R2R_{}.json'.format(
        source) are cached for each of the source in data_sources.
      runtime_config: An instance of `common.RuntimeConfig`.
      env_config: Optional. If None, defaults to config specified in
        lookfar/r2r/env_config.py.
    """
    super(R2REnv, self).__init__()
    assert isinstance(runtime_config, common.RuntimeConfig)

    env_config = env_config if env_config else _get_default_env_config()

    all_paths = _get_all_paths(
        data_sources=data_sources,
        base_path=env_config.base_path,
        vocab_file=env_config.vocab_file,
        fixed_instruction_len=env_config.instruction_len)

    self._compute_reward = env_config.reward_fn

    self._setup(data_sources, runtime_config, env_config, all_paths)

  def _setup(self, data_sources, runtime_config, env_config, all_paths):
    all_scans = sorted(list(set([path['scan'] for path in all_paths])))
    all_scan_ids = [i for i in range(len(all_scans))]
    start_scan_idx = runtime_config.task_id % len(all_scans)
    my_scan_names = all_scans[start_scan_idx::runtime_config.num_tasks]
    my_scan_ids = all_scan_ids[start_scan_idx::runtime_config.num_tasks]
    my_scans = {name: idx for name, idx in zip(my_scan_names, my_scan_ids)}

    assert my_scans, ('No scan for this poor actor :(, my task: {}, total '
                      'scans: {}').format(runtime_config.task_id,
                                          len(all_scans))

    self._paths = []
    for path in all_paths:
      if path['scan'] in my_scans:
        # Replace `scan` with `scan_id`.
        one_path = copy.copy(path)
        one_path['scan_id'] = my_scans[one_path['scan']]
        del one_path['scan']
        self._paths.append(one_path)
    assert self._paths, 'No paths found for scans {} in data_sources {}'.format(
        my_scans, data_sources)
    logging.info(
        'Found %d paths from %d scans with data_sources %s for runtime '
        'settings %s', len(self._paths), len(my_scans), data_sources,
        runtime_config)
    # Randomly shuffle the order of paths.
    np.random.shuffle(self._paths)
    self._current_idx = -1
    self._scan_info = {}
    for scan_name, scan_idx in my_scans.items():
      self._scan_info[scan_idx] = _get_scan_info(
          env_config=env_config,
          scan_name=scan_name,
          stop_node_id=constants.STOP_NODE_ID,
          default_conn_id=constants.INVALID_NODE_ID)
    self._max_actions_per_episode = env_config.max_agent_actions

  @property
  def num_paths(self):
    return len(self._paths)

  def _get_next_idx(self, current_idx):
    """Get the next data idx in the environment."""
    return (current_idx + 1) % self.num_paths

  def _reset(self):
    """Reset the environment with new data.

    Returns:
      A instance of common.EnvOutput, which is the initial Observation.
    """
    self._current_idx = self._get_next_idx(self._current_idx)
    current_scan_id = self._paths[self._current_idx]['scan_id']
    current_pano_id = self._scan_info[current_scan_id].pano_name_to_id[
        self._paths[self._current_idx]['path'][0]]
    self._path_history = [current_pano_id]
    return common.EnvOutput(
        reward=np.float32(0.0),
        done=False,
        observation=self._get_current_observation(current_pano_id,
                                                  current_scan_id, 0),
        info='')

  def _step(self, action):
    """Updates the state using the provided action.

    Sets `done=True` if either this action corresponds to stop node or the
    budget for the current episode is exhausted.

    Args:
      action: An integer specifying the next pano id.

    Returns:
      A tuple `EnvOutput`.
    """
    # First check this is a valid action.
    assert action >= 0
    current_observations = self.get_current_env_output().observation
    current_pano_id = current_observations[constants.PANO_ID]
    current_scan_id = current_observations[constants.SCAN_ID]
    current_time_step = current_observations[constants.TIME_STEP]
    assert action in self._scan_info[current_scan_id].conn_ids[current_pano_id]
    next_pano_id = action
    self._path_history.append(next_pano_id)
    done = False
    if (next_pano_id == constants.STOP_NODE_ID or
        current_time_step == self._max_actions_per_episode):
      done = True
    return common.EnvOutput(
        reward=np.float32(
            self._compute_reward(
                path_history=self._path_history[:-1],
                next_pano=next_pano_id,
                golden_path=self._paths[self._current_idx]['path'],
                end_of_episode=done,
                scan_info=self._scan_info[current_scan_id])),
        done=done,
        observation=self._get_current_observation(next_pano_id, current_scan_id,
                                                  current_time_step + 1),
        info='')

  def get_distance(self, source_pano_id, target_pano_id, scan_id):
    """Get distance between two nodes.

    Args:
      source_pano_id: An integer id of the source pano.
      target_pano_id: An integer id of the target pano.
      scan_id: Scan id.

    Returns:
      Distance from source_pano_id to target_pano_id.
    """
    if source_pano_id == target_pano_id:
      return 0.0
    elif source_pano_id == constants.STOP_NODE_ID:
      return INF
    elif target_pano_id == constants.STOP_NODE_ID:
      return 0.0
    scan_info = self._scan_info[scan_id]
    source = scan_info.pano_id_to_name[source_pano_id]
    target = scan_info.pano_id_to_name[target_pano_id]
    return scan_info.graph.get_distance(source, target)

  def pano_id_to_name(self, pano_id, scan_id):
    if pano_id == constants.INVALID_NODE_ID:
      return constants.INVALID_NODE_NAME
    return self._scan_info[scan_id].pano_id_to_name[pano_id]

  def get_scan_graph(self, scan_id):
    return self._scan_info[scan_id].graph

  def _get_current_observation(self, pano_id, scan_id, time_step):
    heading, pitch = self._get_heading_pitch(pano_id, scan_id, time_step)
    golden_path = [self._scan_info[scan_id].pano_name_to_id[pano_name]
                   for pano_name in self._paths[self._current_idx]['path']]
    goal_pano_id = golden_path[-1]

    def _pad_with_invalid(path):
      padding = max(0, self._max_actions_per_episode + 1 - len(path))
      return path + [constants.INVALID_NODE_ID] * padding

    padded_golden_path = _pad_with_invalid(golden_path)
    padded_observed_path = _pad_with_invalid(self._path_history)

    # to make the code more elegant, cut off the paths longer than max_actions,
    # otherwise, there will be a mismatch bug if max_actions is set smaller than
    # the longest path.
    padded_golden_path = padded_golden_path[:self._max_actions_per_episode]

    obs = {
        constants.IS_START:
            pano_id == padded_golden_path[0],
        constants.DISC_MASK:  # We will modify it in the postprocessing.
            True,
        constants.GOLDEN_PATH:
            padded_golden_path,
        constants.GOAL_PANO_ID:
            goal_pano_id,
        constants.TIME_STEP:
            time_step,
        constants.PATH_ID:
            self._paths[self._current_idx]['path_id'],
        constants.OBSERVED_PATH:
            padded_observed_path,
        constants.PANO_ID:
            pano_id,
        constants.HEADING:
            heading,
        constants.PITCH:
            pitch,
        constants.SCAN_ID:
            scan_id,
        constants.INS_TOKEN_IDS:
            self._paths[self._current_idx]['instruction_token_ids'],
        constants.INS_LEN:
            self._paths[self._current_idx]['instruction_len'],
        constants.CONN_IDS:
            self._scan_info[scan_id].conn_ids[pano_id],
        constants.VALID_CONN_MASK:
            self._scan_info[scan_id].conn_ids[pano_id] >= 0,
        constants.PANO_ENC:
            get_rel_enc(self._scan_info[scan_id].pano_enc[pano_id], heading,
                        pitch),
        constants.CONN_ENC:
            get_rel_enc(self._scan_info[scan_id].conn_enc[pano_id], heading,
                        pitch),
        constants.ORACLE_NEXT_ACTION:
            self._get_oracle_action(pano_id, scan_id),
    }

    # Label does not always exists, we will check whether we can load the label.
    if 'label' in self._paths[self._current_idx]:
      obs[constants.LABEL] = self._paths[self._current_idx]['label']

    return obs

  def _get_oracle_action(self, current_pano_id, current_scan_id):
    """Returns oracle action.

    The current implementation always chooses oracle action to be the neighbor
    pano that will take the agent closest to goal pano.

    Args:
      current_pano_id: The id of the current location.
      current_scan_id: The id of the current scan

    Returns:
      The oracle action.
    """
    current_pano_name = self._scan_info[current_scan_id].pano_id_to_name[
        current_pano_id]
    target_pano_name = self._paths[self._current_idx]['path'][-1]
    if (current_pano_name == target_pano_name or
        current_pano_name == constants.STOP_NODE_NAME):
      # If the agent is at goal pano, it must choose to STOP.
      # Also return STOP if the agent is already at STOP node.
      return constants.STOP_NODE_ID
    oracle_pano_name = self._scan_info[
        current_scan_id].graph.get_closest_neighbor_to_target(
            current_pano_name, target_pano_name)
    return self._scan_info[current_scan_id].pano_name_to_id[oracle_pano_name]

  def _get_heading_pitch(self, pano_id, scan_id, time_step):
    if time_step == 0:
      # Initial heading and pitch. Initial pitch is always 0.
      return np.array([self._paths[self._current_idx]['heading']
                      ]).astype(np.float32), np.array([0.]).astype(np.float32)
    else:
      prev_pano_id = self.get_current_env_output().observation[
          constants.PANO_ID]
      conn_idx = np.asscalar(
          np.argwhere(
              self._scan_info[scan_id].conn_ids[prev_pano_id] == pano_id))
      conn_enc = self._scan_info[scan_id].conn_enc[prev_pano_id][conn_idx]
      return _get_heading(conn_enc), _get_pitch(conn_enc)


def get_rel_enc(enc, heading, pitch):
  """Replaces absolute angles with relative angles in `enc`.

  Args:
    enc: A 2-d numpy array of shape [N, ?]; where N is the number of images and
      second dim is the encoding of each of the images.
    heading: A scalar, the current heading in radians relative to which the new
      heading is computed and replaced in the `enc`.
    pitch: A scalar, the current pitch in radians relative to which the new
      pitch is computed and replaced in the `enc`.

  Returns:
    A 2-d numpy array of same shape as `enc`.
  """
  rel_heading = _get_heading(enc) - heading
  rel_pitch = _get_pitch(enc) - pitch
  rel_signature = np.concatenate([
      np.sin(rel_heading),
      np.cos(rel_heading),
      np.sin(rel_pitch),
      np.cos(rel_pitch)
  ],
                                 axis=1)
  return np.concatenate([rel_signature, enc[:, 4:]], axis=1).astype(np.float32)


def _get_heading(enc):
  """Returns heading angle for each of the images in `enc`.

  The last dim in `enc` is assumed to have first two entries equal to sine
  and cosine of heading.

  Args:
    enc: A n-d numpy array, where n >= 1. The last dim is expected to be the
      encoding.

  Returns:
    A n-d numpy array of shape [..., 1], where leading dimensions are same as
    `enc`.
  """
  sin_h, cos_h, _ = np.split(enc, [1, 2], axis=-1)
  return np.arctan2(sin_h, cos_h).astype(np.float32)


def _get_pitch(enc):
  """Returns pitch angle for each of the images in `enc`.

  The last dim in `enc` is assumed to have third and fourth entries equal to
  sine and cosine of pitch.

  Args:
    enc: A n-d numpy array, where n >= 1. The last dim is expected to be the
      encoding.

  Returns:
    A n-d numpy array of shape [..., 1], where leading dimensions are same as
    `enc`.
  """
  _, sin_p, cos_p, _ = np.split(enc, [2, 3, 4], axis=-1)
  return np.arctan2(sin_p, cos_p).astype(np.float32)


def _get_default_env_config():
  return default_env_config.get_default_env_config()


def _get_scan_info(env_config, scan_name, stop_node_id, default_conn_id):
  """Returns `ScanInfo` for the given scan.

  Args:
    env_config: A config object.
    scan_name: The name of the scan whose information is looked up.
    stop_node_id: An int specifying the id used for stop node.
    default_conn_id: An int specifying the default connections ids to use for
      panos that have less than `env_config.max_conns`.

  Returns:
    An object of `ScanInfo`.
  """
  house_file = os.path.join(env_config.base_path, 'scans', scan_name,
                            'house_segmentations', '{}.house'.format(scan_name))
  house_info = house_parser.R2RHouseParser(house_file)
  connections_file = os.path.join(env_config.base_path, 'connections',
                                  '{}_connectivity.json'.format(scan_name))
  house_graph = house_info.get_panos_graph(connections_file, True)

  pano_names = sorted(house_graph.nodes)
  # First entry always corresponds to STOP_NODE.
  pano_names.insert(0, 'STOP_NODE')

  p_enc_list = []
  c_enc_list = []
  conn_ids = np.full(
      shape=[len(pano_names), env_config.max_conns],
      fill_value=default_conn_id,
      dtype=np.int32)

  pano_name_to_id = {p: i for i, p in enumerate(pano_names)}
  logging.info('Scan: %s, START loading pano and connection encodings.',
               scan_name)
  expected_pano_enc_shape = (env_config.image_encoding_dim *
                             env_config.images_per_pano,)
  for i, p in enumerate(pano_names):
    pano_enc_file = os.path.join(env_config.image_features_dir,
                                 '{}_viewpoints_proto'.format(p))
    image_f = _get_image_features(pano_enc_file)
    p_enc = np.array(image_f.value)
    assert np.shape(p_enc) == expected_pano_enc_shape, (
        'Pano {}: expected pano encoding shape: {} but '
        'found: {}').format(p, expected_pano_enc_shape, np.shape(p_enc))
    p_enc_list.append(
        np.reshape(
            p_enc,
            (1, env_config.images_per_pano, env_config.image_encoding_dim)))

    pano_conn_file = os.path.join(env_config.image_features_dir,
                                  '{}_connections_proto'.format(p))
    image_f = _get_image_features(pano_conn_file)
    conn_name = image_f.pano_id
    assert len(conn_name) < env_config.max_conns
    c_enc = np.array(image_f.value)
    if conn_name:
      expected_conn_enc_shape = (env_config.image_encoding_dim *
                                 len(conn_name),)
      assert np.shape(c_enc) == expected_conn_enc_shape, (
          'Pano {}: expected connection encoding '
          'shape: {}, but found: {}').format(p, expected_conn_enc_shape,
                                             np.shape(c_enc))
    c_enc_list.append(
        np.reshape(
            np.pad(
                c_enc,
                (0, env_config.max_conns * env_config.image_encoding_dim -
                 np.size(c_enc)),
                mode='constant'),
            (1, env_config.max_conns, env_config.image_encoding_dim)))
    conn_id = [pano_name_to_id[c] for c in conn_name]
    # We always add a connection to STOP_NODE at the end
    conn_id.append(stop_node_id)
    conn_ids[i][0:len(conn_id)] = conn_id
    # Logging
    if (i + 1) % 10 == 0:
      logging.info('Done: %d out of %d', i + 1, len(pano_names))
  logging.info('Scan: %s, END loading pano and connection encodings.',
               scan_name)

  return constants.ScanInfo(
      pano_name_to_id=pano_name_to_id,
      pano_id_to_name={v: k for k, v in pano_name_to_id.items()},
      pano_enc=np.concatenate(p_enc_list, axis=0),
      conn_ids=conn_ids,
      conn_enc=np.concatenate(c_enc_list, axis=0),
      graph=house_graph)


def _get_image_features(filename):
  protos = []
  # Pretend to read more than 1 values and then verify there is exactly 1.
  for record in tf.data.TFRecordDataset([filename]).take(2):
    protos.append(record.numpy())
  assert len(protos) == 1
  parsed_record = image_features_pb2.ImageFeatures()
  parsed_record.ParseFromString(protos[0])
  return parsed_record


def _get_all_paths(data_sources, base_path, vocab_file,
                   fixed_instruction_len):
  """Returns list of all paths from the given `data_sources`."""
  vocab_filepath = os.path.join(base_path, vocab_file)
  vocab = load_vocab(vocab_filepath)
  filenames = [
      os.path.join(base_path, 'R2R_{}.json'.format(source))
      for source in data_sources
  ]
  processed_paths = []
  for filename in filenames:
    for entry in json.load(tf.io.gfile.GFile(filename)):
      instructions = entry.pop('instructions')
      for ins in instructions:
        e = copy.copy(entry)
        e['instruction_token_ids'], e['instruction_len'] = get_token_ids(
            ins, fixed_instruction_len, vocab)
        processed_paths.append(e)
  return processed_paths


def get_token_ids(sentence, fixed_length, vocab, is_tokenized=False):
  """Splits the sentence into tokens and returns their ids."""
  tokens = Tokenizer.split_sentence(sentence) if not is_tokenized else sentence
  num_tokens = len(tokens)
  token_ids = []
  oov_token_id = vocab[constants.OOV_TOKEN]
  pad_token_id = vocab[constants.PAD_TOKEN]
  for i in range(fixed_length):
    if i < num_tokens:
      token_ids.append(vocab.get(tokens[i], oov_token_id))
    else:
      token_ids.append(pad_token_id)
  return np.array(token_ids), num_tokens


def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  with tf.io.gfile.GFile(vocab_file) as f:
    tokens = f.readlines()
  tokens = [token.strip() for token in tokens]
  vocab = {k: idx for idx, k in enumerate(tokens)}
  logging.info('Read a vocab with %d tokens.', len(vocab))
  return vocab


class Tokenizer(object):
  """A class to tokenize and encode sentences."""

  # Split on any non-alphanumeric character.
  # https://github.com/ronghanghu/speaker_follower
  SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')

  @staticmethod
  def split_sentence(sentence):
    """Break sentence into a list of words and punctuation."""
    toks = []
    for s in Tokenizer.SENTENCE_SPLIT_REGEX.split(sentence.strip()):
      if s.strip():
        word = s.strip().lower()
        if (all(c in string.punctuation for c in word) and
            not all(c in '.' for c in word)):
          toks += list(word)
        else:
          toks.append(word)
    return toks
