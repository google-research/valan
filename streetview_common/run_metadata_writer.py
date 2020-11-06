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

"""This class exports run metadata (e.g. executed trajectories) to log dir.

TensorBoard is great, but it only allows viewing certain types of information.
This metadata writer allows writing different types of output, such as
trajectory latitudes and longitues, for opening and inspection in a Colab
notebook.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import json
import os

from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf


class RunMetadataWriter(object):
  """This class exports run metadata (e.g. executed trajectories) to logdir."""

  def __init__(self, logdir, write_every_n=20):
    logging.info("Starting RunMetadataWriter for: %s", logdir)
    self._logdir = logdir
    self._data = None
    self._tensor_data = None
    self._seq_idx = 0
    self._write_every_n = write_every_n
    self.reset()

  def reset(self, step_idx=None):
    """Resets logger for a new execution."""
    logging.info("RunMetadataWriter %s: reset", self._logdir)
    if step_idx is not None:
      self._seq_idx = step_idx
    else:
      self._seq_idx += 1
    self._data = {
        "run_data": None,
        "state_trace": [],
        "action_trace": [],
        "observation_trace": [],
    }
    self._tensor_data = {}

  def log_state(self, graph_state):
    logging.info("RunMetadataWriter %s: log_state", self._logdir)
    graph_state_dict = {
        "latitude": graph_state.latitude,
        "longitude": graph_state.longitude,
        "pano_id": graph_state.pano_id,
        "heading": graph_state.heading
    }
    self._data["state_trace"].append(graph_state_dict)

  def log_action(self, agent_action, go_towards):
    logging.info("RunMetadataWriter %s: log_action", self._logdir)
    action_dict = {
        "action_idx": int(agent_action),
        "go_towards": str(go_towards)
    }
    self._data["action_trace"].append(action_dict)

  def log_observation(self, observation):
    del observation
    # Nothing of consequence to be logged here, but subclasses may override.
    logging.info("RunMetadataWriter %s: log_observation", self._logdir)

  def log_run_data(self, data):
    self._data["run_data"] = data

  def log_named_tensor(self, name, tensor_data):
    if name not in self._tensor_data:
      self._tensor_data[name] = [tensor_data]
    else:
      self._tensor_data[name].append(tensor_data)

  def write(self):
    """Writes so-far collected data to files."""
    logging.info("RunMetadataWriter %s: write", self._logdir)
    if self._seq_idx % self._write_every_n == 0:
      logging.info("Writing data with sequence ID %s to %s",
                   str(self._seq_idx), str(self._logdir))
      filename = "{}.json".format(self._seq_idx)
      datadir = os.path.join(self._logdir, str(self._seq_idx))
      tf.io.gfile.makedirs(datadir)
      filepath = os.path.join(datadir, "env.json")
      with tf.io.gfile.GFile(filepath, "w") as fp:
        json.dump(self._data, fp, indent=4)

      for name, tensors in self._tensor_data.items():
        logging.info("Logging tensor data with name: %s and length: %d",
                     name, len(tensors))
        stacked_tensors = np.stack(tensors) if len(tensors) > 1 else tensors
        filename = "{}_{}.npy".format(self._seq_idx, name)
        filepath = os.path.join(datadir, filename)
        with tf.io.gfile.GFile(filepath, "wb") as fp:
          np.save(fp, stacked_tensors)
