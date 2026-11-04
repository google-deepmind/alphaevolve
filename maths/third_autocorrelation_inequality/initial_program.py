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

"""Searches for a step function to improve the third autocorrelation inequality.

A step function is represented by a list of numbers.
The step function has equally spaced intervals, and the numbers specify the
value of the function at each interval.
"""

# EVOLVE-BLOCK-START
import time
from typing import Any
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
    best_sequence = rng.uniform(low=-1, high=1, size=150)
  else:
    best_sequence = np.array(init_sequence, dtype=np.float64)

  best_score = sequence_score(best_sequence)
  curr_sequence = best_sequence.copy()

  start_time = time.time()
  while time.time() - start_time < rng.uniform(100, 1000):
    new_sequence = lp_based_improvement(curr_sequence)
    if new_sequence is None:
      # Add random noise.
      curr_sequence = curr_sequence + rng.normal(
          0, 0.05, size=curr_sequence.shape
      )
    else:
      curr_sequence = new_sequence

    # Evaluate the current sequence and update the best score if necessary.
    curr_score = sequence_score(curr_sequence)
    if curr_score > best_score:
      best_score = curr_score
      best_sequence = curr_sequence.copy()

  return best_sequence


def lp_based_improvement(sequence: np.ndarray) -> np.ndarray:
  """Returns a better step function based on solving an LP.

  This function is based on the algorithm described in page 445 of
  https://doi.org/10.1016/j.jmaa.2010.07.030.

  Args:
    sequence: The current step function.

  Returns:
    A better step function, or None if the linear program could not be solved.
  """
  # 1. Setup and Normalization
  n = len(sequence)
  norm_seq = sequence * np.sqrt(2 * n) / np.sum(sequence)

  # 2. Setup Linear Program Constraints
  # We need max(conv(f, g)) <= max(conv(f, f)).
  # rhs is the peak of the convolution of the normalized sequence with itself.
  rhs = np.max(np.convolve(norm_seq, norm_seq, mode='full'))

  # Construct the convolution matrix A where A @ g = conv(f, g).
  # This uses a Toeplitz matrix structure.
  col0 = np.pad(norm_seq, (0, n - 1))  # First column: f padded with zeros
  row0 = np.zeros(n)
  row0[0] = norm_seq[0]  # First row: f[0] followed by zeros
  a_ub = scipy.linalg.toeplitz(col0, row0)
  b_ub = np.full(2 * n - 1, rhs)  # Upper bound vector
  c = -np.ones(n)  # Objective: Minimize -sum(g) -> Maximize sum(g)

  # 3. Solve the LP
  # bounds=(0, None) enforces non-negativity (g >= 0).
  result = scipy.optimize.linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=(0, None))

  if not result.success:
    return None

  # 4. Post-processing
  g_fun = result.x

  # Normalize the resulting g function
  norm_g = g_fun * np.sqrt(2 * n) / np.sum(g_fun)

  return 0.9 * sequence + 0.1 * norm_g


# EVOLVE-BLOCK-END


def sequence_score(sequence: np.ndarray) -> float:
  """Evaluates a step function defined by `sequence` and returns its score."""
  sequence = np.clip(np.array(sequence, dtype=np.float64), -1, 1)
  convolution = np.convolve(sequence, sequence)
  denominator = np.sum(sequence) ** 2
  if denominator < 0.01:
    return -np.inf
  return -2 * len(sequence) * np.max(abs(convolution)) / denominator


def evaluate() -> dict[str, Any]:
  """Returns the score and output of `search_for_best_sequence` function."""
  init_sequence = globals().get('PARENT_OUTPUT', None)
  best_sequence = np.clip(search_for_best_sequence(init_sequence), -1, 1)
  score = sequence_score(best_sequence)
  if score > 0:
    # This is theoretically impossible, so we return a very small score
    # to indicate that something went wrong.
    score = -float('inf')

  return {
      'scores_to_maximize': {'score': score},
      'output_artifacts': best_sequence.tolist(),
  }
