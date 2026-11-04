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

"""Searches for a step function to improve second autocorrelation inequality.

A step function is represented by a list of numbers.
The step function has equally spaced intervals, and the numbers specify the
value of the function at each interval.
"""

# EVOLVE-BLOCK-START

import time
from typing import Any
import numpy as np


def search_for_best_sequence(init_sequence: list[float] | None) -> np.ndarray:
  """Main search function: searches for a step function maximizing the score.

  Args:
    init_sequence: A step function that can be the starting point for the
      search. If None, a random step function is used.

  Returns:
    The best step function found, represented as a sequence of numbers.
  """
  rng = np.random.default_rng()
  if init_sequence is None:
    best_sequence = rng.uniform(0, 10, size=50000)
  else:
    best_sequence = np.array(init_sequence, dtype=np.float64)

  curr_sequence = best_sequence.copy()
  best_score = -np.inf
  start_time = time.time()
  while time.time() - start_time < rng.uniform(100, 1000):
    curr_sequence = curr_sequence + np.random.normal(
        0, 1, size=curr_sequence.shape
    )

    # Evaluate the current sequence and update the best score if necessary.
    curr_sequence = np.clip(curr_sequence, 0, 100)
    curr_score = sequence_score(curr_sequence)
    if curr_score > best_score:
      best_score = curr_score
      best_sequence = curr_sequence.copy()

  return best_sequence


# EVOLVE-BLOCK-END


def sequence_score(sequence: np.ndarray) -> float:
  """Returns the bound `sequence` gives for 2nd autocorrelation inequality."""
  # Reject short sequences as they won't be able to improve SOTA.
  if len(sequence) < 50000:
    return -np.inf

  # Compute linear convolution of the sequence with itself (autocorrelation).
  conv = np.convolve(sequence, sequence)

  # Define the step size based on the convolution length for trapezoidal-style
  # integration.
  h = 1.0 / (len(conv) + 1)

  # Pad the convolution with a zero at the start and end to ensure the piecewise
  # function starts and ends at 0.
  y = np.pad(conv, (1, 1), mode='constant')

  # Create two overlapping views of the data to represent the start (y1)
  # and end (y2) of each interval.
  y1 = y[:-1]
  y2 = y[1:]

  # Calculate the squared L2 norm using Simpson's/Trapezoidal logic for
  # piecewise linear segments:
  # Integral of (ax + b)^2 over an interval is (h/3) * (y1^2 + y1*y2 + y2^2).
  l2_norm_squared = (h / 3) * np.sum(y1**2 + y1 * y2 + y2**2)

  # Calculate L1 norm (integral of the absolute values) scaled by the step size.
  norm_1 = np.sum(np.abs(conv)) * h

  # Find the L-infinity norm (the maximum absolute value in the convolution).
  norm_inf = np.max(np.abs(conv))

  # Remove the possibility of numerical erros.
  if norm_inf * norm_1 < 0.01:
    return -np.inf

  # Return ratio of squared L2 norm to product of L1 and L-infinity norms.
  return l2_norm_squared / (norm_1 * norm_inf)


def evaluate() -> dict[str, Any]:
  """Returns the score and output of `search_for_best_sequence` function."""
  init_sequence = globals().get('PARENT_OUTPUT', None)
  best_sequence = np.clip(search_for_best_sequence(init_sequence), 0, 100)

  score = sequence_score(best_sequence)
  if score > 1:
    # This is theoretically impossible, so we return a very small score
    # to indicate that something went wrong.
    score = -float('inf')

  return {
      'scores_to_maximize': {'score': score},
      'output_artifacts': best_sequence.tolist(),
  }
