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

"""AlphaEvolve initial program for the Heilbronn problem for convex regions.

The problem is to find a set of 2d points in a convex region with unit area such
that the smallest area triangle formed by any three of the points is maximized.

The number of points can be set by changing `NUM_POINTS` below.
"""

# EVOLVE-BLOCK-START
import itertools
import time

import numpy as np
import scipy

# EVOLVE-BLOCK-END
NUM_POINTS = 14
# EVOLVE-BLOCK-START


def search_for_best_point_configuration(
    init_point_set: np.ndarray,
) -> np.ndarray:
  """Returns a point set with maximum smallest area triangle / convex hull area.

  Args:
    init_point_set: The initial point set to start the search from.

  Returns:
    A point set with maximum smallest area triangle normalized by convex hull
    area and same shape as `init_point_set`.
  """
  point_set = init_point_set
  best_point_set = init_point_set.copy()
  best_score = smallest_area_triangle_normalized_by_convex_hull_area(
      best_point_set
  )
  rng = np.random.default_rng()

  start_time = time.time()
  while time.time() - start_time < rng.uniform(100, 1000):
    # Update a random point from the point set.
    point_set[rng.integers(0, NUM_POINTS)] = rng.uniform(0, 1, size=2)
    score = smallest_area_triangle_normalized_by_convex_hull_area(point_set)

    if score > best_score:
      best_score = score
      best_point_set = point_set.copy()
    elif rng.random() < 0.2:
      # Reset to the best point set with some probability.
      point_set = best_point_set.copy()

  return best_point_set


# EVOLVE-BLOCK-END
def smallest_area_triangle_normalized_by_convex_hull_area(
    point_set: np.ndarray,
) -> float:
  """Finds the smallest area triangle normalized by convex hull area."""
  # Reject invalid input types and shapes.
  if (
      not isinstance(point_set, np.ndarray)
      or not np.isfinite(point_set).all()
      or point_set.shape != (NUM_POINTS, 2)
  ):
    return 0

  # Convert to float64 and clip values to avoid numerical instability.
  point_set = np.asarray(point_set, np.float64).clip(0, 1)

  # Generate all combinations of 3 indices.
  idx = np.array(list(itertools.combinations(range(NUM_POINTS), 3)))

  # Extract the points for each combination.
  p1 = point_set[idx[:, 0]]
  p2 = point_set[idx[:, 1]]
  p3 = point_set[idx[:, 2]]

  # Compute all triangle areas using the cross product formula
  # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|.
  areas = 0.5 * np.abs(
      p1[:, 0] * (p2[:, 1] - p3[:, 1])
      + p2[:, 0] * (p3[:, 1] - p1[:, 1])
      + p3[:, 0] * (p1[:, 1] - p2[:, 1])
  )

  convex_hull_area = scipy.spatial.ConvexHull(point_set).volume

  # Reject degenerate point sets, which have a small convex hull area.
  if convex_hull_area < 0.01:
    return 0
  return np.min(areas) / convex_hull_area


def evaluate() -> dict[str, any]:
  """Evaluates the `search_for_best_point_configuration` function."""
  random_point_set = np.random.default_rng().uniform(0, 1, size=(NUM_POINTS, 2))
  old_point_set = globals().get('PARENT_OUTPUT', random_point_set)
  old_point_set = np.asarray(old_point_set, np.float64)
  new_point_set = search_for_best_point_configuration(old_point_set)
  new_point_set = np.asarray(new_point_set, np.float64).clip(0, 1)
  score = smallest_area_triangle_normalized_by_convex_hull_area(new_point_set)
  if score > 1:
    # This is theoretically impossible, so we return a very small score
    # to indicate that something went wrong.
    score = -float('inf')

  return {
      'scores_to_maximize': {
          'score': score
      },
      'output_artifacts': new_point_set.tolist(),
  }
