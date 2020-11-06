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


"""Extracts detailed checkpoint restoration information."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import os

from absl import logging
import tensorflow as tf


def log_status(ckpt_status, output_dir):
  """Saves matched and unmatched object information to file for examination.

  Args:
    ckpt_status: a status object created by `assert_nontrivial_match()` or
    `assert_consumed()` methods of a checkpoint object, e.g.,
        ckpt = tf.Checkpoint(model=model)
        status = ckpt.restore(path_to_ckpt).assert_nontrivial_match()
    output_dir: str; the output dir to save detailed restoration log.
  """
  try:
    
    ckpt = ckpt_status._checkpoint
    pretty_printer = _ObjectGraphProtoPrettyPrinter(
        ckpt.object_graph_proto)

    matched_objects = []
    unresolved_objects = []
    for node_id in range(len(ckpt.object_graph_proto.nodes)):
      text = '{}\n'.format(pretty_printer.node_names[node_id])
      if node_id in ckpt.matched_proto_ids:
        matched_objects.append(text)
      else:
        unresolved_objects.append(text)

    unused_attributes = []
    for node_id, attribute_name in ckpt.unused_attributes.items():
      unused_attributes.append('Unused attribute in object {}: {}'.format(
          pretty_printer.node_names[node_id], attribute_name))
  except AttributeError:
    logging.error('Checkpoint status object must have attribute `_checkpoint`.')

  # Save information to file.
  if not tf.io.gfile.isdir(output_dir):
    logging.warning('Dir not found. Skip saving restoration information to: %s',
                    output_dir)
    return

  output_file = os.path.join(output_dir, 'ckpt_restoration_log')
  with tf.io.gfile.GFile(output_file, 'w') as f:
    f.write('Restored checkpoint: {}\n\n'.format(ckpt.save_path_string))

    for fn in [f.write, logging.info]:
      fn('Unmatched objects: \n -------------------------\n')
      for text in unresolved_objects:
        fn(text)

      fn('\n\n\n Unmatched attributes: \n -------------------------\n')
      for text in unused_attributes:
        fn(text)

      fn('\n\n\n Matched objects: \n -------------------------\n')
      for text in matched_objects:
        fn(text)
  logging.info('Saved checkpoint restoration details to: %s', output_file)


class _ObjectGraphProtoPrettyPrinter(object):
  """Lazily traverses an object graph proto to pretty print names.

  If no calls to `node_names` are made this object has no performance
  overhead. On the other hand, it will only traverse the object graph once, so
  repeated naming is cheap after the first.
  """

  def __init__(self, object_graph_proto):
    self._object_graph_proto = object_graph_proto
    self._node_name_cache = None

  @property
  def node_names(self):
    """Lazily creates a mapping from node id to ("path", "to", "root")."""
    if self._node_name_cache is not None:
      return self._node_name_cache
    path_to_root = {}
    path_to_root[0] = ('(root)',)
    to_visit = collections.deque([0])
    while to_visit:
      node_id = to_visit.popleft()
      obj = self._object_graph_proto.nodes[node_id]
      for child in obj.children:
        if child.node_id not in path_to_root:
          path_to_root[child.node_id] = (
              path_to_root[node_id] + (child.local_name,))
          to_visit.append(child.node_id)

    node_names = {}
    for node_id, path_to_root in path_to_root.items():
      node_names[node_id] = '.'.join(path_to_root)

    for node_id, node in enumerate(self._object_graph_proto.nodes):
      for slot_reference in node.slot_variables:
        node_names[slot_reference.slot_variable_node_id] = (
            '{}\'s state "{}" for {}'.format(
                node_names[node_id], slot_reference.slot_name,
                node_names[slot_reference.original_variable_node_id]))
    self._node_name_cache = node_names
    return node_names
