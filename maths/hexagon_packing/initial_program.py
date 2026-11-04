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

"""AlphaEvolve initial program for hexagon packing.

The number of hexagons can be changed by changing NUM_POLYGONS_TO_PACK below.
"""

# pylint: disable=g-import-not-at-top
# EVOLVE-BLOCK-START
import itertools
import math
import time
from typing import Any, TypeAlias

import numpy as np
import shapely

# A transformed hexagon is a tuple of (x, y, angle) in that order.
TransformedHexagon: TypeAlias = tuple[float, float, float]
Point: TypeAlias = shapely.geometry.Point
Polygon: TypeAlias = shapely.geometry.Polygon

# EVOLVE-BLOCK-END
NUM_POLYGONS_TO_PACK = 11
# EVOLVE-BLOCK-START


def search_for_best_packings(
    initial_solution: list[TransformedHexagon],
) -> list[TransformedHexagon]:
  """Searches for the best packing of unit hexagons in an outer hexagon.

  Args:
    initial_solution: The initial solution to start the search from.

  Returns:
    The best packing found.
  """
  current_placements = initial_solution
  best_placements = initial_solution.copy()
  best_score = calculate_packing_score_polygons(best_placements)

  rng = np.random.default_rng()
  start_time = time.time()
  while time.time() - start_time < 1000:
    idx = rng.integers(0, len(current_placements))
    old_val = current_placements[idx]

    # 80% chance to 'jitter' (fine-tune), 20% to 'leap' (explore).
    if rng.random() < 0.8 and best_score != -float('inf'):
      x, y, a = old_val
      mutation = (
          x + rng.uniform(-2, 2),
          y + rng.uniform(-2, 2),
          a + rng.uniform(0, 360),
      )
    else:
      mutation = (rng.uniform(0, 50), rng.uniform(0, 50), rng.uniform(0, 360))

    current_placements[idx] = mutation
    score = calculate_packing_score_polygons(current_placements)

    if score > best_score:
      best_score = score
      best_placements = list(current_placements)
    else:
      # Revert to last known good state.
      current_placements[idx] = old_val

  return best_placements


# EVOLVE-BLOCK-END

# Pre-calculate unit hexagon vertices to avoid re-computation.
UNIT_HEX_POLY = Polygon([
    (1.0, 0.0),
    (0.5, math.sqrt(3) / 2),
    (-0.5, math.sqrt(3) / 2),
    (-1.0, 0.0),
    (-0.5, -math.sqrt(3) / 2),
    (0.5, -math.sqrt(3) / 2),
])


def transformed_unit_hexagon(
    x: float, y: float, angle: float, scale: float = 1.0
) -> Polygon:
  """Returns a transformed version of the pre-defined unit hexagon."""
  poly = UNIT_HEX_POLY
  if scale != 1.0:
    poly = shapely.affinity.scale(
        poly, xfact=scale, yfact=scale, origin='centroid'
    )
  if angle != 0.0:
    poly = shapely.affinity.rotate(poly, angle, origin='centroid')
  return shapely.affinity.translate(poly, xoff=x, yoff=y)


def find_smallest_scale_outer_polygon(
    placements: list[TransformedHexagon], precision: float = 1e-4
) -> float:
  """Finds the minimum scale for the container using geometric bounds."""
  inner_polys = [transformed_unit_hexagon(*p) for p in placements[:-1]]
  out_x, out_y, out_ang = placements[-1]
  container_center = Point(out_x, out_y)

  # Find max reach: distance from center to furthest point of any inner poly.
  max_dist = 0.0
  for poly in inner_polys:
    for point in poly.exterior.coords:
      dist = container_center.distance(Point(point))
      if dist > max_dist:
        max_dist = dist

  # Bounds: Theoretical floor (area) vs Geometric Reach.
  area_floor = math.sqrt(len(inner_polys))
  low = max(area_floor, max_dist)
  high = 2 * max_dist / math.sqrt(3)

  while (high - low) > precision:
    mid = (low + high) / 2
    container = transformed_unit_hexagon(out_x, out_y, out_ang, scale=mid)
    if all(container.contains(p) for p in inner_polys):
      high = mid
    else:
      low = mid

  return high


def calculate_packing_score_polygons(
    placements: list[TransformedHexagon],
) -> float:
  """Validates geometry and returns -scale (higher score is tighter packing)."""
  # Validation
  if (
      not isinstance(placements, list)
      or len(placements) != NUM_POLYGONS_TO_PACK + 1
  ):
    return -float('inf')

  # Check bounds and data types.
  clamped: list[TransformedHexagon] = []
  for p in placements:
    if not all(math.isfinite(v) for v in p):
      return -float('inf')
    clamped.append((
        float(np.clip(p[0], -1000, 1000)),
        float(np.clip(p[1], -1000, 1000)),
        float(p[2]),
    ))

  inner_hexagons = [transformed_unit_hexagon(*p) for p in clamped[:-1]]
  for hexagon1, hexagon2 in itertools.combinations(inner_hexagons, 2):
    if hexagon1.intersects(hexagon2):
      return -float('inf')

  return -find_smallest_scale_outer_polygon(clamped)


def evaluate() -> dict[str, Any]:
  """Evaluates the `search_for_best_packings` function."""
  rng = np.random.default_rng()

  # Load from global state if available (for iterative evolution).
  parent_solution = globals().get('PARENT_OUTPUT')
  if not parent_solution:
    parent_solution = [
        (rng.uniform(0, 50), rng.uniform(0, 50), rng.uniform(0, 360))
        for _ in range(NUM_POLYGONS_TO_PACK + 1)
    ]

  final_solution = search_for_best_packings(parent_solution)
  final_score = calculate_packing_score_polygons(final_solution)

  if final_score > - np.sqrt(NUM_POLYGONS_TO_PACK):
    # This is theoretically impossible, so we return a very small score
    # to indicate that something went wrong.
    final_score = -float('inf')

  return {
      'scores_to_maximize': {'score': final_score},
      'output_artifacts': final_solution,
  }
