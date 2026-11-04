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

"""AlphaEvolve initial program for max/min distance ratio problem.

The goal is to find a point set that minimizes the
max_pairwise_distance / min_pairwise_distance ratio.

Change the values of constants DIMENSION and NUM_POINTS as desired.
"""

# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy

# EVOLVE-BLOCK-END
DIMENSION = 2
NUM_POINTS = 16
# EVOLVE-BLOCK-START


def search_for_best_point_configuration(
    init_point_set: np.ndarray,
) -> np.ndarray:
  """Searches for a point set with minimum max_distance/min_distance ratio.

  Args:
    init_point_set: The initial point set to start the search from.

  Returns:
    A point set with minimum max_distance/min_distance ratio and same shape as
    `init_point_set`.
  """
  point_set = init_point_set
  best_point_set = init_point_set.copy()
  best_score = compute_negative_squared_max_min_ratio(best_point_set)
  rng = np.random.default_rng()

  start_time = time.time()
  while time.time() - start_time < rng.uniform(100, 1000):
    # Update a random point from the point set.
    point_set[rng.integers(0, NUM_POINTS)] = rng.uniform(-1, 1, size=DIMENSION)
    score = compute_negative_squared_max_min_ratio(point_set)

    if score > best_score:
      best_score = score
      best_point_set = point_set.copy()
    elif rng.random() < 0.2:
      # Reset to the best point set with some probability.
      point_set = best_point_set.copy()

  return best_point_set


# EVOLVE-BLOCK-END


def compute_negative_squared_max_min_ratio(point_set: np.ndarray) -> float:
  """Returns negative squared ratio of max distance over min distance."""
  # Reject invalid input types and shapes.
  if (
      not isinstance(point_set, np.ndarray)
      or not np.isfinite(point_set).all()
      or point_set.shape != (NUM_POINTS, DIMENSION)
  ):
    return -np.inf

  # Convert to float64 and clip values to avoid numerical instability.
  point_set = np.asarray(point_set, np.float64).clip(-1000, 1000)

  # Calculate squared Euclidean distances.
  squared_distances = scipy.spatial.distance.pdist(
      point_set, metric='sqeuclidean'
  )

  # Reject very small minimum distance, as it can cause numerical instability.
  min_squared_distance = np.min(squared_distances)
  if min_squared_distance < np.finfo(np.float64).eps:
    return -np.inf

  return -np.max(squared_distances) / min_squared_distance


def evaluate() -> dict[str, any]:
  """Evaluates the `search_for_best_point_configuration` function."""
  random_point_set = np.random.default_rng().uniform(
      -1, 1, size=(NUM_POINTS, DIMENSION)
  )
  old_point_set = globals().get('PARENT_OUTPUT', random_point_set)
  old_point_set = np.asarray(old_point_set, np.float64)
  new_point_set = search_for_best_point_configuration(old_point_set)
  new_point_set = np.asarray(new_point_set, np.float64).clip(-1000, 1000)

  score = compute_negative_squared_max_min_ratio(new_point_set)
  if score > -1:
    # This is theoretically impossible, so we return a very small score
    # to indicate that something went wrong.
    score = -np.inf

  return {
      'scores_to_maximize': {'score': score},
      'output_artifacts': new_point_set.tolist(),
  }
