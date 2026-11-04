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

"""AlphaEvolve initial program for the extremal girth 5 problem."""

# EVOLVE-BLOCK-START
import math
import random
import time

import numba
import numpy as np


njit = numba.njit


def search_for_best_graph(
    num_nodes: int, init_edges: list[tuple[int, int]]
) -> list[tuple[int, int]]:
  """Searches for a graph with lots of edges but no triangles or four-cycles.

  Uses a stochastic hill-climbing algorithm with edge additions and swaps.

  Args:
    num_nodes: The number of nodes in the graph.
    init_edges: The initial edges of the graph. It's a valid graph with no
      triangles or four-cycles and is used as the base for the search.

  Returns:
    The best graph found.
  """
  start_time = time.time()
  time_limit = 60

  # 1. Start with a valid base.
  # Check if the initial graph is already valid; if not, start empty.
  initial_score = calculate_graph_score(num_nodes, init_edges)
  current_edges = set()

  if initial_score > 0:
    for u, v in init_edges:
      current_edges.add((min(u, v), max(u, v)))

  best_edges = set(current_edges)

  # Pre-generate all possible edge pairs to avoid repeated calculations.
  all_possible_pairs = []
  for u in range(num_nodes):
    for v in range(u + 1, num_nodes):
      all_possible_pairs.append((u, v))

  while time.time() - start_time < time_limit:
    # Shuffle potential edges to try adding them in a random order.
    random.shuffle(all_possible_pairs)

    # --- PHASE 1: GREEDY ADDITION ---
    # Try to shove in as many edges as possible without breaking the girth.
    for edge in all_possible_pairs:
      if edge not in current_edges:
        current_edges.add(edge)
        if calculate_graph_score(num_nodes, list(current_edges)) > 0:
          if len(current_edges) > len(best_edges):
            best_edges = set(current_edges)
        else:
          current_edges.remove(edge)  # Undo if it created a cycle

    # --- PHASE 2: STOCHASTIC PERTURBATION (SHAKE) ---
    # If we can't add more edges, try removing 2 random edges,
    # and see if that opens up a "hole" to add 3+ new edges.
    if len(current_edges) > 2:
      for edge in random.sample(list(current_edges), 2):
        current_edges.remove(edge)

  return list(best_edges)


# EVOLVE-BLOCK-END
uint64 = numba.uint64


def calculate_graph_score(n: int, edges: list[tuple[int, int]]) -> int:
  """Checks a graph for 3-cycles and 4-cycles and returns a score.

  The function filters the input edges for validity (no self-loops,
  indices must be within [0, n-1], and must be integers). It then
  checks if the graph contains any 3-cycles (triangles) or 4-cycles.

  Args:
    n: The number of nodes in the graph.
    edges: A list of tuples/lists representing edges.

  Returns:
    0 if a 3-cycle or 4-cycle is detected, otherwise the number of edges.
  """
  # 1. Pre-process and Filter
  # We use a set of sorted tuples to handle duplicate edges.
  unique_edges = set()
  for e in edges:
    try:
      u, v = int(e[0]), int(e[1])
      if u == v or not (0 <= u < n and 0 <= v < n):
        continue
      # Canonical ordering (min, max) ensures (0,1) == (1,0).
      if u < v:
        unique_edges.add((u, v))
      else:
        unique_edges.add((v, u))
    except (ValueError, TypeError, IndexError):
      continue

  if not unique_edges:
    return 0

  # Convert to NumPy for Numba processing.
  edge_array = np.array(list(unique_edges), dtype=np.int32)

  # Calculate words needed for bit-packing (64 nodes per uint64 word).
  num_words = (n + 63) // 64

  # 2. Run JIT-optimized cycle detection.
  if _has_cycles_bitpacked(n, edge_array, num_words):
    return 0

  # 3. Return score (number of edges)
  return len(edge_array)


@njit(fastmath=True)
def _popcount(x: int) -> int:
  """Returns the number of set bits (1s) in a 64-bit integer."""
  c = uint64(0)
  while x > 0:
    x &= x - uint64(1)
    c += uint64(1)
  return c


@njit(fastmath=True)
def _has_cycles_bitpacked(
    n: int, edge_array: np.ndarray, num_words: int
) -> bool:
  """Detects 3-cycles and 4-cycles using bitwise adjacency intersections.

  Leverages bit-parallelism to check 64 potential neighbors simultaneously.

  Args:
    n: The number of nodes in the graph.
    edge_array: The edge array of the graph.
    num_words: The number of words needed for bit-packing.

  Returns:
    True if a 3-cycle or 4-cycle is found, False otherwise.
  """
  # Initialize a bit-packed adjacency matrix (n x ceil(n/64)).
  bit_adj = np.zeros((n, num_words), dtype=uint64)
  # Build the bit-packed matrix.
  for u, v in edge_array:
    bit_adj[u, v >> 6] |= uint64(1) << uint64(v & 63)
    bit_adj[v, u >> 6] |= uint64(1) << uint64(u & 63)
  # Scan pairs of nodes to find common neighbors.
  for u in range(n):
    for v in range(u + 1, n):
      common_count = 0
      for k in range(num_words):
        intersect = bit_adj[u, k] & bit_adj[v, k]
        if intersect:
          common_count += _popcount(intersect)
      if common_count == 0:
        continue
      # Logic:
      # 1. If nodes u and v are connected and share a neighbor, then
      #   a 3-cycle exists.
      # 2. If nodes u and v are connected and share two or more neighbors,
      #   then a 4-cycle exists.
      is_connected = (bit_adj[u, v >> 6] >> uint64(v & 63)) & uint64(1)
      if is_connected or common_count >= 2:
        return True
  return False


def evaluate() -> tuple[dict[str, float], dict[str, str]]:
  """Evaluates the `search_for_best_graph` function."""
  init_graphs = globals().get('PARENT_OUTPUT', dict())
  best_graphs = dict()

  aggregated_score = 0
  n_values = [53, 73, 85, 101, 126, 145, 163, 181, 197, 203]
  for n in n_values:
    init_graph = init_graphs.get(n, [])
    init_score = calculate_graph_score(n, init_graph)
    new_graph = search_for_best_graph(n, init_graph)
    new_score = calculate_graph_score(n, new_graph)
    if new_score > init_score:
      best_graphs[n] = new_graph
      best_score = new_score
    else:
      best_graphs[n] = init_graph
      best_score = init_score
    aggregated_score += best_score / n / math.sqrt(n) / len(n_values)

  return {
      'scores_to_maximize': {'score': aggregated_score},
      'output_artifacts': best_graphs,
  }
