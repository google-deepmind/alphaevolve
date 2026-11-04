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

"""AlphaEvolve initial program for the cap set problem."""

# EVOLVE-BLOCK-START
import copy
import random
import time
from typing import Any
import numpy as np
# EVOLVE-BLOCK-END

_DIMENSION = 9


# EVOLVE-BLOCK-START
def search_for_best_cap_set(
    initial_solution: set[tuple[int, ...]],
    dimension: int,
) -> set[tuple[int, ...]]:
  """Searches for a large cap set in the given dimension.

  Args:
    initial_solution: An initial solution to start the search from.
    dimension: The dimension of the vectors in the cap set.

  Returns:
    A set of tuples, where each tuple represents a vector in the cap set. Each
    vector must have elements in the set {0, 1, 2} and be of length `dimension`.
    For it to be a valid cap set, no three elements in the set sum to the zero
    vector (mod 3).
  """
  # With probability 0.9, start with the provided initial solution. Otherwise,
  # start with element (0, ..., 0).
  if random.random() < 0.9:
    best_cap_set = initial_solution
  else:
    best_cap_set = {(0,) * dimension}
  best_score, _ = calculate_score_cap_set(best_cap_set, dimension)

  # For the allowed time (20 minutes), search for a larger cap set.
  start_time = time.time()
  while time.time() - start_time < 20 * 60:
    cap_set = copy.copy(best_cap_set)
    # Heuristic: randomly remove one element of the cap set, and try to add (at
    # most) 10 random new elements.
    element_to_remove = random.choice(list(cap_set))
    cap_set.remove(element_to_remove)
    for _ in range(10):
      new_element = tuple(random.randint(0, 2) for _ in range(dimension))
      cap_set.add(new_element)
    # Evaluate the solution.
    score, _ = calculate_score_cap_set(cap_set, dimension)
    if score > best_score:
      best_cap_set = copy.copy(cap_set)
      best_score = score
  return best_cap_set

# EVOLVE-BLOCK-END


def calculate_score_cap_set(
    cap_set: set[tuple[int, ...]],
    dimension: int,
) -> tuple[float, set[tuple[int, ...]]]:
  """Calculates the score of a candidate cap set.

  Checking the cap set property naively takes O(num_elements^3 dimension) time,
  where num_elements is the size of the given cap set. This function implements
  a faster check that runs in O(num_elements^2 dimension).

  Args:
    cap_set: A set of vectors of length `dimension` with elements in {0, 1, 2}.
    dimension: The dimension of the vectors in the cap set.

  Returns:
    A tuple of (score, final_cap_set), where `score` is the score of the cap set
    (i.e., its length after removing invalid elements), and `final_cap_set` is
    a valid cap set obtained from `cap_set` after removing invalid elements.
  """
  # Remove invalid elements from `cap_set`.
  cleared_cap_set = set()
  for element in cap_set:
    if (
        not isinstance(element, tuple)  # Wrong type.
        or len(element) != dimension  # Wrong length.
        or not all(isinstance(x, int) for x in element)  # Wrong type.
        or any(x < 0 or x >= 3 for x in element)  # Wrong values.
    ):
      continue
    # Add valid elements to `cleared_cap_set`.
    cleared_cap_set.add(element)

  num_elements = len(cleared_cap_set)
  if num_elements <= 1:
    return float(num_elements), cleared_cap_set

  # Convert the cap set to an array of integers.
  cleared_cap_set = np.array(
      list(cleared_cap_set), dtype=np.int32
  )  # Shape (num_elements, dimension).
  powers = 3 ** np.arange(dimension - 1, -1, -1)  # Shape (power,).
  cleared_cap_set_as_int = np.einsum(
      'np,p->n', cleared_cap_set, powers
  )  # Shape (num_elements,).

  # Starting from the empty set, we iterate through the vectors in
  # `cleared_cap_set` one by one and check that the vector can be inserted into
  # the set without violating the defining property of cap set. To make this
  # check fast we maintain an array `invalid` indicating, for each element of
  # Z_3^n, whether that element can be inserted into the growing set without
  # violating the cap set property.
  final_cap_set = set()
  invalid = np.zeros(shape=(3 ** dimension,), dtype=np.bool_)
  for idx_i, (new_vector, new_vector_as_int) in enumerate(
      zip(cleared_cap_set, cleared_cap_set_as_int, strict=True)
  ):
    if invalid[new_vector_as_int]:
      continue
    if idx_i >= 1:
      invalid_idx = np.einsum(
          'np,p->n',
          (
              - np.array(list(final_cap_set), dtype=np.int32)
              - new_vector[None]
          ) % 3,
          powers,
      )
      invalid[invalid_idx] = True
    invalid[new_vector_as_int] = True
    final_cap_set.add(tuple(map(int, new_vector)))

  return float(len(final_cap_set)), final_cap_set


def evaluate() -> dict[str, Any]:
  """Evaluates a solution."""
  initial_solution = {
      tuple(x)
      for x in globals().get('PARENT_OUTPUT', [(0,) * _DIMENSION])
  }
  cap_set = search_for_best_cap_set(initial_solution, _DIMENSION)
  score, cleared_cap_set = calculate_score_cap_set(cap_set, _DIMENSION)
  return {
      'scores_to_maximize': {'score': score},
      'output_artifacts': list(list(x) for x in cleared_cap_set),
  }
