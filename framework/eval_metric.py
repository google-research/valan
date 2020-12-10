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

import numpy as np


def get_dtw_matrix(observed_panos, golden_panos, distance_fn):
  """Dynamic Time Warping (DTW).

  Muller, Meinard. "Dynamic time warping."
  Information retrieval for music and motion (2007): 69-84.

  Dynamic Programming implementation, O(NM) time and memory complexity.

  Args:
    observed_panos: List of observed pano ids or names.
    golden_panos: List of golden pano ids or names.
    distance_fn: Method for getting the distance between two panos.

  Returns:
    A 2-D matrix with DTW scores.
  """
  num_obs_panos = len(observed_panos)
  num_golden_panos = len(golden_panos)

  dtw_matrix = np.inf * np.ones((num_obs_panos + 1, num_golden_panos + 1))
  dtw_matrix[0][0] = 0
  for i in range(num_obs_panos):
    for j in range(num_golden_panos):
      best_prev_cost = min(
          dtw_matrix[i][j],    # Move both
          dtw_matrix[i+1][j],  # Move query
          dtw_matrix[i][j+1]   # Move reference
      )
      cost = distance_fn(observed_panos[i], golden_panos[j])
      dtw_matrix[i+1][j+1] = cost + best_prev_cost

  return dtw_matrix
