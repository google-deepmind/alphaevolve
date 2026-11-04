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

"""Searches for a step function to improve the first autocorrelation inequality.

A step function defined on [-0.25, 0.25] is represented by a list of numbers.
The step function has equally spaced intervals, and the numbers specify the
value of the function at each interval.
"""

# EVOLVE-BLOCK-START
import time

import numpy as np
import scipy


def search_for_best_sequence(init_sequence: list[float] | None) -> np.ndarray:
  """Main search function: searches for a step function maximizing the score.

  Args:
    init_sequence: A given step function that can be the starting point for the
      search. If None, a random step function is used.

  Returns:
    The best step function found, represented as a sequence of numbers.
  """
  rng = np.random.default_rng()
  if init_sequence is None:
    best_sequence = rng.random(600)
  else:
    best_sequence = np.array(init_sequence, dtype=np.float64)

  curr_sequence = best_sequence.copy()
  best_score = -np.inf
  start_time = time.time()
  while time.time() - start_time < rng.uniform(100, 1000):
    good_direction = good_direction_to_move_into(curr_sequence)
    if good_direction is None:
      # Add random noise.
      curr_sequence = curr_sequence + np.random.normal(
          0, 0.05, size=curr_sequence.shape
      )
    else:
      curr_sequence = good_direction

    # Evaluate the current sequence and update the best score if necessary.
    curr_score = sequence_score(curr_sequence)
    if curr_score > best_score:
      best_score = curr_score
      best_sequence = curr_sequence.copy()

  return best_sequence


def normalize(sequence: np.ndarray) -> np.ndarray:
  """Returns a normalized version of the sequence."""
  return sequence * (np.sqrt(2 * len(sequence)) / sequence.sum())


def good_direction_to_move_into(sequence: np.ndarray) -> np.ndarray | None:
  """Returns a better step function based on solving an LP.

  This function is based on the algorithm described in page 445 of
  https://doi.org/10.1016/j.jmaa.2010.07.030.

  Args:
    sequence: The current step function.

  Returns:
    A better step function, or None if the linear program could not be solved.
  """
  normalized_sequence = normalize(sequence)

  # Formulate and solve the linear program.
  rhs = np.max(np.convolve(normalized_sequence, normalized_sequence))
  n = len(normalized_sequence)

  # 1. Objective: Maximize sum(g) -> Minimize -sum(g)
  c = -np.ones(n)

  # 2. Constraints
  a_ub = scipy.linalg.convolution_matrix(normalized_sequence, n, mode='full')
  b_ub = np.full(2 * n - 1, rhs)

  # 3. Solve
  result = scipy.optimize.linprog(
      c, A_ub=a_ub, b_ub=b_ub, bounds=[(0, None)] * n, method='highs'
  )

  if not result.success:
    return None

  g_fun = np.asanyarray(result.x, dtype=np.float64)
  return 0.99 * sequence + 0.01 * normalize(g_fun)


# EVOLVE-BLOCK-END


def sequence_score(sequence: np.ndarray) -> float:
  """Evaluates a step function defined by `sequence` and returns its score."""
  n = len(sequence)
  # Reject short sequences or those containing non-numeric values.
  if n < 500 or not np.isfinite(sequence).all():
    return -np.inf

  sequence = np.clip(sequence, 0, 10)
  convolution = np.convolve(sequence, sequence)
  sequence_sum_squared = sequence.sum() ** 2

  # Reject near-zero denominators.
  if sequence_sum_squared < 0.01:
    return -np.inf

  return -2 * n * np.max(convolution) / sequence_sum_squared


def evaluate() -> dict[str, float]:
  """Returns the score and output of `search_for_best_sequence` function."""
  init_sequence = globals().get('PARENT_OUTPUT', None)
  best_sequence = np.clip(search_for_best_sequence(init_sequence), 0, 10)
  score = sequence_score(best_sequence)
  if score > 0:
    # This is theoretically impossible, so we return a very small score
    # to indicate that something went wrong.
    score = -float('inf')

  return {
      'scores_to_maximize': {'score': score},
      'output_artifacts': best_sequence.tolist(),
  }
