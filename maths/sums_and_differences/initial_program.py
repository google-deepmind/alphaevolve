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

"""AlphaEvolve initial program for the Sums and Differences problem.

Instead of directly maximizing the score, we ask AlphaEvolve to maximize
auxiliary_score = score - 0.01 / solution_size.

This is a tiebreaker term to encourage AlphaEvolve to explore larger solution
sets and avoid getting stuck in local optima that are small. It turned out to be
crucial for finding the best solutions.
"""

# EVOLVE-BLOCK-START
import time
import numpy as np


def search_for_best_set(initial_list: list[int]) -> np.ndarray:
  """Searches for the best integer set for the sums and differences problem.

  Args:
    initial_list: The initial set of integers to start the search from.

  Returns:
    The best set of integers found.
  """
  curr_set = np.unique(np.abs(np.asarray(initial_list, dtype=np.int64)))
  if 0 not in curr_set:
    curr_set = np.insert(curr_set, 0, 0)

  best_set = curr_set.copy()
  best_score = compute_score(best_set)

  rng = np.random.default_rng()
  start_time = time.time()

  while time.time() - start_time < 1000:
    random_index = rng.integers(1, len(best_set))
    old_value = best_set[random_index]
    best_set[random_index] += rng.integers(-3, 4)
    curr_score = compute_score(best_set)

    if curr_score > best_score:
      best_score = curr_score
    else:
      best_set[random_index] = old_value
  return best_set


# EVOLVE-BLOCK-END


def compute_score(u: np.ndarray) -> float:
  """Returns the score of the set u."""
  try:
    u = np.asanyarray(u)
    if u.ndim != 1 or not np.issubdtype(u.dtype, np.integer) or u.size < 2:
      return 0
    # Convert to int64, take absolute values, remove duplicates, and add 0.
    u = np.unique(np.abs(u.astype(np.int64)))
    if u[0] != 0:
      u = np.insert(u, 0, 0)
  except (TypeError, ValueError):
    return 0

  max_u = u[-1]
  u_minus_u = np.zeros(
      2 * max_u + 1, dtype=bool
  )  # Store the set u - u as an array of booleans.
  u_plus_u = np.zeros(
      2 * max_u + 1, dtype=bool
  )  # Store the set u + u as an array of booleans.

  for i in u:
    u_minus_u[i - u + max_u] = True
    u_plus_u[i + u] = True

  u_minus_u_size = np.sum(u_minus_u)
  u_plus_u_size = np.sum(u_plus_u)

  if u_minus_u_size > 2 * max_u + 1:
    # The constraint |U - U| <= 2 max (U) + 1 is not satisfied.
    return 0

  return np.log(u_minus_u_size / u_plus_u_size) / np.log(2 * max_u + 1) + 1.0


def evaluate() -> dict[str, any]:
  """Evaluates the `search_for_best_set` function."""
  parent_solution = globals().get('PARENT_OUTPUT', [0, 2, 3, 7, 15, 18, 22])
  new_solution = search_for_best_set(parent_solution)
  new_solution = np.unique(np.abs(np.asarray(new_solution, dtype=np.int64)))
  score = compute_score(new_solution)
  if score > 2:
    # This is theoretically impossible, so we return a very small score
    # to indicate that something went wrong.
    score = -float('inf')

  return {
      'scores_to_maximize': {
          'auxiliary_score': score - 0.01 / new_solution.size,
      },
      'output_artifacts': new_solution.tolist(),
  }
