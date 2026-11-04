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

"""AlphaEvolve initial program for the circle packing problem.

The program searches for the circle centers. The largest possible sum of radii
is computed optimally by solving a linear program.

You can specify the container shape (the unit square or a rectangle with
perimeter 4) and the number of circles to pack by setting the `CONTAINER_SHAPE`
and `NUM_CIRCLES` constants below.
"""

# EVOLVE-BLOCK-START

import enum
import time
from typing import List
import numpy as np
import scipy.optimize


# EVOLVE-BLOCK-END
class ContainerShape(enum.Enum):
  """Defines the container shape."""
  SQUARE = enum.auto()  # Pack circles in a square.
  RECTANGLE = enum.auto()  # Pack circles in a rectangle.


# Change this to ContainerShape.SQUARE to pack the circles in a square.
CONTAINER_SHAPE = ContainerShape.RECTANGLE

# Change this to the desired number of circles to pack.
NUM_CIRCLES = 21

# EVOLVE-BLOCK-START


def search_for_best_circle_packing(
    init_centers: np.ndarray
) -> np.ndarray:
  """Searches for the best packing of circles within a rectangle.

  Args:
    init_centers: The initial centers of circles (shape (n, 2)) to start the
      search from.

  Returns:
    The centers of circles (shape (n, 2)) with largest sum of radii subject to
    the constraints:
    (a) all circles are disjoint, and
    (b) all circles lie within the container, which is a unit square or a
        rectangle with perimeter 4. The bottom-left coordinate is (0, 0).
  """
  rng = np.random.default_rng()
  points = init_centers.copy()
  num_circles = points.shape[0]

  radii, _ = maximize_radii_sum(points, num_circles)
  best_score = np.sum(radii)
  best_points = points.copy()

  start_time = time.time()
  while time.time() - start_time < rng.uniform(100, 1000):
    index_to_mutate = rng.integers(0, num_circles)
    # Mutate the center of a random circle by sampling a new (x, y) coordinate.
    points[index_to_mutate] = rng.random((1, 2))

    radii, _ = maximize_radii_sum(points, num_circles)
    new_score = np.sum(radii)

    if new_score > best_score:
      best_score = new_score
      best_points = points.copy()

    # With probability 0.5, revert to the best points.
    if rng.random() < 0.5:
      points = best_points.copy()

  return best_points


# EVOLVE-BLOCK-END


def maximize_radii_sum(
    centers: np.ndarray,
    num_circles: int,
) -> tuple[np.ndarray | None, tuple[float, float] | None]:
  """Maximizes the sum of radii for disjoint circles centered at `centers`.

  Solves a linear program to maximize the sum of radii subject to constraints:
    (a) all circles are disjoint, and
    (b) all circles lie within the container, which is a unit square or a
        rectangle with perimeter 4. The bottom-left coordinate is (0, 0).
  In the linear program, we enforce all constraints to be satisfied with a
  slack of 1e-7.

  Args:
    centers: (n, 2) coordinates of circle centers. Must be positive.
    num_circles: Expected number of centers.

  Returns:
    A tuple of (radii, dims), where
    - radii: Maximum radii for each point respecting the constraints.
    - dims: (width, height) of the container.
  """
  n = num_circles
  if (
      not isinstance(centers, np.ndarray)
      or centers.shape != (n, 2)
      or (not np.all(centers >= 0))
      or (not np.all(np.isfinite(centers)))
  ):
    return None, None

  centers = np.ascontiguousarray(centers, dtype=np.float64)
  slack = 1e-7

  # LP variables: x = [r0, ..., r_{n-1}, w, h]
  c: np.ndarray = np.zeros(n + 2)
  # LP objective: Maximize sum(r_i)
  c[:n] = -1.0

  # LP constraint matrices: a_ub * x <= b_ub and a_eq * x == b_eq.
  a_ub: List[List[float]] = []
  b_ub: List[float] = []
  a_eq: List[List[float]] = []
  b_eq: List[float] = []

  # 1. Perimeter Constraint: w + h <= 2
  row_p = [0.0] * (n + 2)
  row_p[n], row_p[n + 1] = 1.0, 1.0
  a_ub.append(row_p)
  b_ub.append(2.0)

  # 2. Shape Constraint (if square)
  if CONTAINER_SHAPE == ContainerShape.SQUARE:
    # w = 1 (h == 1 will follow from constraint 1).
    row_s = [0.0] * (n + 2)
    row_s[n] = 1.0
    a_eq.append(row_s)
    b_eq.append(1.0)

  for i in range(n):
    xi, yi = centers[i]

    # 3. Top and right boundary constraints
    # r_i + x_i + slack <= w  =>  r_i - w <= -x_i - slack
    row_w = [0.0] * (n + 2)
    row_w[i] = 1.0
    row_w[n] = -1.0
    a_ub.append(row_w)
    b_ub.append(-xi - slack)

    # r_i + y_i + slack <= h  =>  r_i - h <= -y_i - slack
    row_h = [0.0] * (n + 2)
    row_h[i] = 1.0
    row_h[n + 1] = -1.0
    a_ub.append(row_h)
    b_ub.append(-yi - slack)

    # 4. Pairwise Non-overlapping: r_i + r_j <= dist - slack
    for j in range(i + 1, n):
      distance = float(np.linalg.norm(centers[i] - centers[j]))
      row_d = [0.0] * (n + 2)
      row_d[i] = 1.0
      row_d[j] = 1.0
      a_ub.append(row_d)
      b_ub.append(distance - slack)

  # 5. Bottom and left boundary constraints
  # r_i <= x_i - slack and r_i <= y_i - slack
  lp_bounds = [
      (0.0, min(centers[i, 0], centers[i, 1]) - slack) for i in range(n)
  ] + [(0.0, None), (0.0, None)]

  # Solve the linear program.
  options = {
      'primal_feasibility_tolerance': 1e-12,
      'dual_feasibility_tolerance': 1e-12,
  }
  result = scipy.optimize.linprog(
      c,
      A_ub=a_ub,
      b_ub=b_ub,
      A_eq=a_eq if a_eq else None,
      b_eq=b_eq if b_eq else None,
      bounds=lp_bounds,
      method='highs-ds',
      options=options,
  )

  if result.success:
    return result.x[:n], (result.x[n], result.x[n + 1])
  return None, None


def evaluate() -> dict[str, float]:
  """Evaluates the `search_for_best_circle_packing` function."""
  rng = np.random.default_rng()

  old_centers = np.ascontiguousarray(
      globals().get('PARENT_OUTPUT', rng.random((NUM_CIRCLES, 2))),
      dtype=np.float64,
  )
  new_centers = search_for_best_circle_packing(old_centers)

  radii, _ = maximize_radii_sum(new_centers, NUM_CIRCLES)
  if radii is None or np.sum(radii) > np.sqrt(NUM_CIRCLES / np.pi):
    # This is theoretically impossible, so we return a very small score
    # to indicate that something went wrong.
    result = {'sum_of_radii': -float('inf')}
  else:
    result = {'sum_of_radii': np.sum(radii)}

  return {
      'scores_to_maximize': result,
      'output_artifacts': new_centers.tolist(),
  }
