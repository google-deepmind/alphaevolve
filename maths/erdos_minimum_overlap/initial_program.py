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

"""AlphaEvolve initial program for the Erdos Minimum Overlap problem.

We represent each step function as a sequence of numbers, as follows:
Starting from an arbitrary sequence of numbers, we first augment it to
simultaneously satisfy three requirements:
  (a) a zeroed prefix (first 12.5% of elements),
  (b) a strict range of [0, 1], and
  (c) a mirrored sum equal to exactly half the length of the resulting symmetric
    sequence.
Then we create a mirrored sequence, which is the perfectly symmetrical version
of this sequence. This mirrored sequence defines the heights of the step
function, which has all intervals of equal length.
"""

# EVOLVE-BLOCK-START
import time
from typing import Any

import numpy as np


def search_for_best_sequence(init_sequence: np.ndarray) -> np.ndarray:
  """Searches for a sequence with high score for Erdos minimum overlap problem.

  Args:
    init_sequence: the sequence to start the search from.

  Returns:
    The best sequence found during the search.
  """
  n = init_sequence.shape[0]
  rng = np.random.default_rng()

  curr_sequence = init_sequence
  best_sequence = init_sequence.copy()
  best_score = sequence_score(curr_sequence, n)

  start_time = time.time()
  while time.time() - start_time < rng.uniform(100, 1000):
    curr_sequence = curr_sequence + rng.standard_normal(n) * 0.01
    curr_score = sequence_score(curr_sequence, n)
    if curr_score > best_score:
      best_sequence = curr_sequence.copy()
      best_score = curr_score
    if rng.uniform() < 0.1:
      curr_sequence = best_sequence.copy()
  return best_sequence


# EVOLVE-BLOCK-END


def sequence_score(sequence: np.ndarray, sequence_length: int) -> float:
  """Scores a sequence."""
  sequence = np.array(sequence, dtype=np.float64)
  if len(sequence.shape) != 1 or sequence.shape[0] != sequence_length:
    return -np.inf

  # Augments the sequence to simultaneously satisfy three requirements:
  # (a) a zeroed prefix (first 12.5% of elements),
  # (b) a strict range of [0, 1], and
  # (c) a mirrored sum equal to exactly half the length of the resulting
  # symmetric sequence.
  # By repeatedly scaling and clipping the values, we bounce the sequence
  # between these constraints until it converges on a state where all three
  # conditions are satisfied within floating-point tolerance.
  success = False
  n = sequence.shape[0]
  target_mirrored_sum = (2 * n - 1) / 2.0
  for _ in range(100):
    # 1. Force Hard Zero Constraint.
    sequence[: n // 8] = 0.0

    # 2. Scale to satisfy the Sum Constraint.
    current_sum = 2 * np.sum(sequence) - sequence[-1]
    if current_sum != 0:
      sequence *= target_mirrored_sum / current_sum
    else:
      sequence += 0.1  # Nudge to avoid division by zero.

    # 3. Clamp to satisfy Range Constraint [0, 1].
    sequence = np.clip(sequence, 0, 1)

    # 4. Final Verification.
    actual_mirrored_sum = 2 * np.sum(sequence) - sequence[-1]

    condition_a = np.all(sequence[: n // 8] == 0)
    condition_b = np.all((sequence >= 0) & (sequence <= 1))
    condition_c = np.isclose(
        actual_mirrored_sum, target_mirrored_sum, atol=1e-12
    )

    if condition_a and condition_b and condition_c:
      success = True
      break

  if not success:
    return -np.inf

  # Now, create a mirrored sequence, which defines the step function.
  mirrored_sequence = np.concatenate((sequence[:-1], sequence[::-1]))
  convolution_values = np.correlate(
      np.array(mirrored_sequence), 1 - np.array(mirrored_sequence), mode='full'
  )
  return -np.max(convolution_values) / len(mirrored_sequence) * 2


def evaluate() -> dict[str, Any]:
  """Evaluates the `search_for_best_sequence` function."""
  sequence_length = 48
  init_sequence = globals().get('PARENT_OUTPUT', [0.5] * sequence_length)
  best_sequence = search_for_best_sequence(
      np.array(init_sequence, dtype=np.float64)
  )
  score = sequence_score(best_sequence, sequence_length)
  if score > 0:
    # This is theoretically impossible, so we return a very small score
    # to indicate that something went wrong.
    score = -float('inf')

  return {
      'scores_to_maximize': {'score': score},
      'output_artifacts': best_sequence.tolist(),
  }
