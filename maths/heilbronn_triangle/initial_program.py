# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""AlphaEvolve initial program for the Heilbronn triangle problem.

Each point is represented by barycentric coordinates (u, v), where u, v, and
1-u-v are in [0, 1]. If the outer triangle has vertices (0, 0), (1, 0), and
(0.5, sqrt(3)/2), then Cartesian coordinates of point (u, v) is given by
(0.5 * u + v, sqrt(3)/2 * u).
"""

# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np
# EVOLVE-BLOCK-END

NUM_POINTS = 11

# EVOLVE-BLOCK-START


def search_for_best_point_configuration(
    init_point_set: np.ndarray,
) -> np.ndarray:
  """Searches for a point set with largest minimum-triangle-area ratio.

  Args:
    init_point_set: The initial point set to start the search from.

  Returns:
    A point set with largest minimum-triangle-area ratio and same shape as
    `init_point_set`.
  """
  point_set = init_point_set
  n = point_set.shape[0]
  best_point_set = init_point_set.copy()
  best_score = min_area_triangle_ratio(best_point_set)
  rng = np.random.default_rng()

  start_time = time.time()
  while time.time() - start_time < rng.uniform(100, 1000):
    # Generate random point (u, v) with the property that
    # u, v, and 1 - u - v are uniform in [0, 1].
    new_point = np.diff(np.concatenate(([0], np.sort(rng.random(2)), [1])))[
        :2
    ].reshape(1, 2)
    point_set[rng.integers(0, n)] = new_point
    score = min_area_triangle_ratio(point_set)

    if score > best_score:
      best_score = score
      best_point_set = point_set.copy()
    elif rng.random() < 0.5:
      # Reset to the best point set.
      point_set = best_point_set.copy()
  return best_point_set


# EVOLVE-BLOCK-END


def min_area_triangle_ratio(coordinates: np.ndarray) -> float:
  """Returns the minimum area of triangles formed by points / outer triangle.

  Args:
    coordinates: np.array of shape (n, 2) representing barycentric coordinates
      of the points.

  Returns:
    For every triple of points, a triangle is formed. The area of this triangle
    is computed and normalized by the area of the outer triangle. This function
    returns the minimum ratio found among all possible triangles. Note that this
    is independent of the shape of the outer triangle.
  """
  # 0. Check that the input is valid.
  if (
      not isinstance(coordinates, np.ndarray)
      or coordinates.shape != (NUM_POINTS, 2)
      or np.isnan(coordinates).any()
      or np.isinf(coordinates).any()
  ):
    # invalid input
    return 0
  coordinates = np.asarray(coordinates, np.float64)

  # 1. Check that all points are inside or on the triangle.
  # u >= 0, v >= 0, and (1 - u - v) >= 0
  u = coordinates[:, 0]
  v = coordinates[:, 1]
  w = 1.0 - u - v
  if np.any(u < 0) or np.any(v < 0) or np.any(w < 0):
    return 0

  # 2. Compute the minimum area ratio.
  return min(
      abs(np.linalg.det([p1 - p3, p2 - p3]))
      for p1, p2, p3 in itertools.combinations(coordinates, 3)
  )


def evaluate() -> dict[str, any]:
  """Evaluates the `search_for_best_point_configuration` function."""

  # Generate NUM_POINTS random points (u, v) with the property that
  # u, v, and 1-u-v are uniform in [0, 1].
  random_point_set = np.diff(
      np.insert(
          np.sort(np.random.default_rng().random((NUM_POINTS, 2)), axis=1),
          [0, 2],
          [0, 1],
          axis=1,
      )
  )[:, :2]

  old_point_set = globals().get('PARENT_OUTPUT', random_point_set)
  old_point_set = np.asarray(old_point_set, np.float64)
  new_point_set = search_for_best_point_configuration(old_point_set)
  new_point_set = np.asarray(new_point_set, np.float64)
  score = min_area_triangle_ratio(new_point_set)
  if score > 1:
    # This is theoretically impossible, so we return a very small score
    # to indicate that something went wrong.
    score = -float('inf')

  return {
      'scores_to_maximize': {'score': score},
      'output_artifacts': new_point_set.tolist(),
  }
