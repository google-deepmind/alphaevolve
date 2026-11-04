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

"""AlphaEvolve initial program for the Shannon capacity of cycle graphs."""

# EVOLVE-BLOCK-START
import copy
import itertools
import random
import time
from typing import Any
import numpy as np
# EVOLVE-BLOCK-END

_NUM_NODES = 5
_POWER = 2


# EVOLVE-BLOCK-START
def search_for_best_independent_set(
    initial_solution: set[tuple[int, ...]],
    num_nodes: int,
    power: int,
) -> set[tuple[int, ...]]:
  """Searches for a large independent set in the power of a cycle graph.

  Args:
    initial_solution: An initial solution to start the search from.
    num_nodes: The number of nodes of the base cycle graph.
    power: The power to raise the cycle graph to.

  Returns:
    A set of tuples, where each tuple represents a node in the graph obtained by
    taking the `power`-th power of a cycle graph of `num_nodes` nodes. Each
    tuple must thus have `power` elements, each containing an integer in the
    range `[0, num_nodes - 1]`; e.g., if `num_nodes = 5` and `power = 2`, a
    valid return value would be: `{(0, 0), (1, 2), (2, 4), (3, 1), (4, 3)}`.
    The returned set must be an independent set in the power graph, i.e., no
    two of the returned nodes are connected. This means that the difference
    between any two of the returned tuples must contain at least one element
    distinct from `{0, 1, num_nodes - 1}`; e.g., adding element `(4, 1)` to the
    set above would violate this requirement, since `(4, 1)` is connected to
    both `(0, 0)` and `(3, 1)`---the differences are `(4, 1)` and `(1, 0)`,
    respectively.
  """
  # With probability 0.9, start with the provided initial solution. Otherwise,
  # start with element (0, ..., 0).
  if random.random() < 0.9:
    best_independent_set = initial_solution
  else:
    best_independent_set = {(0,) * power}
  best_score = calculate_score_independent_set(
      best_independent_set, num_nodes, power
  )

  # For the allowed time (20 minutes), search for a larger independent set.
  start_time = time.time()
  while time.time() - start_time < 20 * 60:
    independent_set = copy.copy(best_independent_set)
    # Heuristic: randomly remove one element of the independent set, and try to
    # add (at most) 10 random new elements.
    element_to_remove = random.choice(list(independent_set))
    independent_set.remove(element_to_remove)
    for _ in range(10):
      new_element = tuple(
          random.randint(0, num_nodes - 1) for _ in range(power)
      )
      independent_set.add(new_element)
    # Evaluate the solution.
    score = calculate_score_independent_set(independent_set, num_nodes, power)
    if score > best_score:
      best_independent_set = copy.copy(independent_set)
      best_score = score
  return best_independent_set

# EVOLVE-BLOCK-END

# Best known independent set sizes for powers of cycle graphs (as of 16 June
# 2026).
_SOTA_INDEPENDENT_SET_SIZE = {
    (5, 2): 5,
    (7, 2): 10,
    (7, 3): 33,
    (7, 4): 108,
    (7, 5): 367,
    (7, 6): 1101,
    (7, 7): 3670,
    (7, 8): 12111,
    (9, 2): 18,
    (9, 3): 81,
    (9, 4): 324,
    (9, 5): 1458,
    (9, 6): 6561,
    (9, 7): 26244,
    (11, 2): 27,
    (11, 3): 148,
    (11, 4): 754,
    (11, 5): 4009,
    (13, 2): 39,
    (13, 3): 247,
    (13, 4): 1534,
    (13, 5): 9633,
    (15, 3): 381,
    (15, 4): 2720,
    (15, 5): 19946,
}


def calculate_score_independent_set(
    independent_set: set[tuple[int, ...]],
    num_nodes: int,
    power: int,
    return_cleared_independent_set: bool = False,
) -> float | tuple[float, set[tuple[int, ...]]]:
  """Calculates the score of an independent set."""
  best_known_independent_set_size = _SOTA_INDEPENDENT_SET_SIZE.get(
      (num_nodes, power), 0
  )

  # Remove invalid elements from `independent_set`.
  cleared_independent_set = set()
  for element in independent_set:
    if (
        not isinstance(element, tuple)  # Wrong type.
        or len(element) != power  # Wrong length.
        or not all(isinstance(x, int) for x in element)  # Wrong type.
        or any(x < 0 or x >= num_nodes for x in element)  # Wrong values.
    ):
      continue
    # Add valid elements to `cleared_independent_set`.
    cleared_independent_set.add(element)

  num_elements = len(cleared_independent_set)
  if num_elements <= 1:
    return float(num_elements - best_known_independent_set_size)

  # Convert the independent set to an array of integers.
  cleared_independent_set_np = np.array(
      list(cleared_independent_set), dtype=np.int32
  )  # Shape (num_elements, power).
  powers = np.array(
      [num_nodes ** i for i in range(power - 1, -1, -1)], dtype=np.int32
  )  # Shape (power,).
  cleared_independent_set_as_int = np.einsum(
      'np,p->n', cleared_independent_set_np, powers
  )  # Shape (num_elements,).

  # Iteratively find which elements do not satisfy the independence condition,
  # and mark them as "dependent".
  forbidden_diffs = np.array(
      list(itertools.product([-1, 0, 1], repeat=power)), dtype=np.int32
  ) % num_nodes  # Shape (num_forbidden, power).
  dependent = np.zeros(  # Initially assume no elements are dependent.
      num_elements, dtype=np.bool_
  )  # Shape (num_elements,).
  for idx_i, element in enumerate(cleared_independent_set_np):
    if dependent[idx_i]:
      continue
    # Find all "forbidden" elements, i.e., those that would create a dependent
    # set if found in `cleared_independent_set_np[idx_i + 1 :]`.
    all_forbidden_elements = np.expand_dims(
        np.einsum(
            'np,p->n', (forbidden_diffs + element[None]) % num_nodes, powers
        ),  # Shape (num_forbidden,).
        axis=-1,
    )  # Shape (num_forbidden, 1).
    # Find which elements in `cleared_independent_set_np[idx_i + 1 :]` are
    # indeed forbidden.
    other_elements = np.expand_dims(
        cleared_independent_set_as_int[idx_i + 1 :],
        axis=0,
    )  # Shape (1, num_elements_other).
    idx_blocking = np.any(
        # Shape (num_forbidden, num_elements_other).
        other_elements == all_forbidden_elements,
        axis=0,
    )  # Shape (num_elements_other,).
    # Update `dependent` to reflect the newly found dependent elements.
    idx_blocking = np.concatenate(
        [np.zeros(idx_i + 1, dtype=np.bool_), idx_blocking]
    )  # Shape (num_elements,).
    dependent[idx_blocking] = True

  num_dependent = np.sum(dependent)
  score = float(num_elements - num_dependent - best_known_independent_set_size)
  return (
      (score, cleared_independent_set)
      if return_cleared_independent_set
      else score
  )


def evaluate() -> dict[str, Any]:
  """Evaluates a solution."""
  initial_solution = {
      tuple(x)
      for x in globals().get('PARENT_OUTPUT', [(0,) * _POWER])
  }
  independent_set = search_for_best_independent_set(
      initial_solution, num_nodes=_NUM_NODES, power=_POWER
  )
  score, cleared_independent_set = calculate_score_independent_set(
      independent_set,
      num_nodes=_NUM_NODES,
      power=_POWER,
      return_cleared_independent_set=True,
  )
  return {
      'scores_to_maximize': {'score': score},
      'output_artifacts': list(list(x) for x in cleared_independent_set),
  }
