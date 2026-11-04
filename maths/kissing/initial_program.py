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

"""Initial program for the kissing number packing problem.

The program searches for the best sphere packing for the kissing number problem,
which means a set of points of norm 1 such that the distance between any two is
at least 1.
"""

# EVOLVE-BLOCK-START

import time
from typing import Any
import numpy as np


def search_for_best_sphere_packing(
    num_spheres: int, dimension: int, init_packing: list[list[int]] | None
) -> np.ndarray:
  """Searches for the best sphere packing.

  Args:
    num_spheres: The number of spheres to pack.
    dimension: The number of dimensions for the space.
    init_packing: The initial packing to start the search from.

  Returns:
    A numpy array of shape (num_spheres, dimension) containing the centers of
    the best sphere packing found. The best sphere packing is such that all
    centers have norm 1 and the distance between any two centers is at least 1.
  """
  centers = None
  if init_packing is not None:
    centers = np.array(init_packing, dtype=np.float64)

  rng = np.random.default_rng()
  if centers is None or centers.shape != (num_spheres, dimension):
    # Initialize the centers randomly.
    centers = rng.standard_normal((num_spheres, dimension))

  best_score = check_and_calculate_packing_score(
      centers, num_spheres, dimension
  )
  best_centers = centers.copy()

  start_time = time.time()
  while time.time() - start_time < rng.uniform(100, 1000):
    # Move a single point randomly.
    index_to_mutate = rng.integers(0, num_spheres)
    centers[index_to_mutate] += rng.uniform(-1, 1, dimension)
    score = check_and_calculate_packing_score(centers, num_spheres, dimension)
    if score > best_score:
      best_score = score
      best_centers = centers.copy()
    if rng.uniform() < 0.1:
      centers = best_centers.copy()

  return best_centers


# EVOLVE-BLOCK-END


def check_and_calculate_packing_score(
    centers: np.ndarray, n: int, d: int
) -> float:
  """Checks and computes the score of a sphere packing.

  Checks if the input is valid; otherwise, returns -n*n. Normalizes the centers
  to have norm 1. For any two centers x and y, penalty = max(0, 1 - ||x - y||),
  and the total penalty is the sum of the penalties over all pairs of centers.
  The final score is the negative of the total penalty. Converts the array to
  float64 for maximum precision and uses NumPy operations for efficiency.

  Args:
    centers: The spheres' centers.
    n: The expected number of points (rows) in centers.
    d: The expected dimensionality of each point (columns) in centers.

  Returns:
    The negative of the total penalty, or -n*n if any validation, numerical
    issue, or a zero vector point is found.
  """
  # Check that the input is valid.
  if (
      not isinstance(centers, np.ndarray)
      or centers.shape != (n, d)
      or not np.issubdtype(centers.dtype, np.floating)
      or np.isnan(centers).any()
      or np.isinf(centers).any()
  ):
    return -n * n

  # Normalize the centers to have norm 1.
  centers = centers.astype(np.float64)
  norms = np.linalg.norm(centers, axis=1)
  epsilon = np.finfo(np.float64).eps
  # Too small or too large norms can cause numerical issues and are declined.
  if np.min(norms) < epsilon or np.max(norms) * epsilon > 1:
    return -n * n
  centers /= norms[:, np.newaxis]

  # Re-check for NaN or Inf values after normalization.
  if np.isnan(centers).any() or np.isinf(centers).any():
    return -n * n

  # Compute pairwise distances and penalties.
  expanded_a = np.expand_dims(centers, axis=1)
  expanded_b = np.expand_dims(centers, axis=0)
  distances = np.linalg.norm(expanded_a - expanded_b, axis=2)
  penalties = np.maximum(0.0, 1.0 - distances)
  return -np.sum(penalties[np.triu_indices(n, k=1)])


def evaluate() -> dict[str, Any]:
  """Evaluates the `search_for_best_sphere_packing` function."""
  num_circles, dimension = 593, 11

  best_packing_found_before = globals().get('PARENT_OUTPUT', None)

  best_packing_found_after = search_for_best_sphere_packing(
      num_circles, dimension, best_packing_found_before
  )
  score = check_and_calculate_packing_score(
      best_packing_found_after, num_circles, dimension
  )

  if score > 0:
    # This is theoretically impossible, so we return a very small score
    # to indicate that something went wrong.
    score = -float('inf')

  return {
      'scores_to_maximize': {'packing_score': score},
      'output_artifacts': best_packing_found_after.tolist(),
  }
