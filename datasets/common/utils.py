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

r"""Common utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import io
import math
import re
import string
from PIL import Image

import tensorflow.compat.v1 as tf


NavigationJsonKeys = collections.namedtuple(
    'NavigationJsonKeys', ('path_key', 'path_id_key', 'instructions_key',
                           'start_heading_key', 'start_pitch_key', 'scan_key'))


def get_float32_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def get_bytes_feature(value):
  if not isinstance(value, list):
    value = [value]
  value_bytes = [tf.compat.as_bytes(element) for element in value]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value_bytes))


def to_radians(angle, zero_centered=True):
  """Transforms an angle in degrees to radians.

  Args:
    angle: angle in degrees.
    zero_centered: If set to True, output values from -pi to pi. Otherwise,
      output values from 0 to 2 * pi.
  Returns:
    The angle in radians.
  """
  radians = math.radians(angle % 360)
  if zero_centered and radians > math.pi:
    radians -= 2 * math.pi
  return radians


def to_degrees(angle, round_to_int=False):
  """Transforms an angle in radians to degrees.

  Args:
    angle: The angle in radians.
    round_to_int: If set to True, we return the closest integer to the floating
      point value of the angle in degrees.
  Returns:
    The angle in radians, ranging from 0 to 360.
  """
  degrees = math.degrees(angle) % 360
  if round_to_int:
    return int(round(degrees))
  return degrees


def distance(angle_1, angle_2, radians=True):
  """Returns the distance between two angles in a circle."""
  whole_circle = 2 * math.pi if radians else 360
  # First we compute the absolute difference between the angles.
  abs_diff = abs(angle_1 - angle_2) % whole_circle
  # The distance is either that or 2*pi - that (the other way around).
  return min(abs_diff, whole_circle - abs_diff)


def save_np_image(image_array, output_filename, output_format='jpeg'):
  image_buffer = io.BytesIO()
  pil_image = Image.fromarray(image_array)
  pil_image.save(image_buffer, format=output_format)
  with tf.gfile.Open(output_filename, 'wb') as f:
    f.write(image_buffer.getvalue())


def get_angle_signature(heading, pitch=0, radians=True):
  if not radians:
    heading = math.radians(heading)
    pitch = math.radians(pitch)

  return [math.sin(heading),
          math.cos(heading),
          math.sin(pitch),
          math.cos(pitch)]


def normalize_instruction(instruction):
  instruction = instruction.lower()
  regex = re.compile('[%s]' % re.escape(string.punctuation))
  instruction = regex.sub(lambda x: ' ' + x.group(0) + ' ', instruction)
  instruction = ' '.join(instruction.split())
  return instruction
