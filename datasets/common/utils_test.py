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

"""Tests for datasets.common.utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random

import tensorflow as tf

from valan.datasets.common import utils


class UtilsTest(tf.test.TestCase):

  def test_to_radians(self):
    pi = math.pi
    angle_pairs = {
        0: 0,
        45: pi/4,
        90: pi/2,
        135: 3*pi/4,
        180: pi,
        270: -pi/2,
        360: 0,
        -90: -pi/2,
        -180: pi,
        -270: pi/2,
        450: pi/2
    }

    for degrees, radians in angle_pairs.items():
      self.assertAlmostEqual(utils.to_radians(degrees), radians)

  def test_to_degrees(self):
    pi = math.pi
    angle_pairs = {
        0: 0,
        pi/4: 45,
        pi/2: 90,
        3*pi/4: 135,
        0.5: 28.64788975654116,
        -pi/2: 270,
        3*pi: 180
    }

    for radians, degrees in angle_pairs.items():
      self.assertAlmostEqual(utils.to_degrees(radians), degrees)

  def test_distance(self):
    # Tests with fixed values.
    test_set = {  # Maps (angle_1, angle_2) to expected distance.
        (0, 0): 0,
        (0, 90): 90,
        (0, -90): 90,
        (-90, 0): 90,
        (90, 0): 90,
        (-90, 90): 180,
        (1, 359): 2,
        (45, 360 + 45 + 45): 45,
        (45, -360 + 45 - 45): 45,
    }
    for (angle_1, angle_2), expected_distance in test_set.items():
      distance = utils.distance(angle_1, angle_2, radians=False)
      self.assertAlmostEqual(distance, expected_distance)

    # Tests with random values.
    for _ in range(50):
      angle_1 = random.uniform(0, 2 * math.pi)
      diff = random.uniform(0, math.pi)
      self.assertAlmostEqual(
          utils.distance(angle_1, angle_1 + diff) % (2 * math.pi), diff)
      self.assertAlmostEqual(
          utils.distance(angle_1, angle_1 - diff) % (2 * math.pi), diff)

  def test_normalize_instruction(self):
    # Test with fixed values.
    test_set = {
        'hello, world!': 'hello , world !',
        'Foo Bar Baz': 'foo bar baz',
        'hot-potato': 'hot - potato',
        'door.turn right': 'door . turn right',
    }
    for instruction, expected_output in test_set.items():
      output = utils.normalize_instruction(instruction)
      self.assertEqual(output, expected_output)

if __name__ == '__main__':
  tf.test.main()
