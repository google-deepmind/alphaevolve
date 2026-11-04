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

"""Initial program, uncertainty principle inequality, Laguerre polynomials."""

# EVOLVE-BLOCK-START

import time
import numpy as np
import sympy


def search_for_best_roots(initial_roots: list[float]) -> list[float]:
  """Searches for roots for optimal Laguerre combination by adaptive mutation.

  Args:
    initial_roots: The roots to start the search from.

  Returns:
    The best-scoring roots found by the search.
  """
  best_roots = sorted(initial_roots)
  best_score = get_score(best_roots)
  curr_roots = best_roots.copy()

  rng = np.random.default_rng()
  start_time = time.time()

  # 1000-second search budget.
  while time.time() - start_time < 1000:
    temp_roots = curr_roots.copy()
    idx = rng.integers(0, len(temp_roots))

    # Decide mutation type: 70% small additive, 30% relative scaling.
    if rng.uniform() < 0.7:
      # Additive mutation (fine-tuning).
      temp_roots[idx] += rng.uniform(-1.0, 1.0)
    else:
      # Multiplicative mutation (structural shift).
      temp_roots[idx] *= rng.uniform(0.95, 1.05)

    # Ensure root remains valid (positive and not excessively large).
    temp_roots[idx] = max(0.1, min(temp_roots[idx], 499.0))
    temp_roots.sort()

    curr_score = get_score(temp_roots)

    # Only accept the move if it's mathematically valid.
    if curr_score > -float('inf'):
      # Greedily accept improvements
      if curr_score > best_score:
        best_score = curr_score
        best_roots = temp_roots.copy()
      curr_roots = temp_roots

    # 20% chance to reset to best known position.
    if rng.uniform() < 0.2:
      curr_roots = best_roots.copy()

  return best_roots


# EVOLVE-BLOCK-END


def construct_laguerre_combination(
    double_root_points: list[float]
) -> sympy.Expr:
  """Constructs linear combination of Laguerre polynomials with given roots.

  Constraints:
  - Sets g(0) = 0 and g'(0) = 1.
  - Sets g(z) = 0 and g'(z) = 0 for each z in root_points.

  Args:
    double_root_points: the list of double roots.

  Returns:
    A sympy expression g(x), as a linear combination of Laguerre polynomials,
    satisfying the above constraints.
  """
  # 1. Setup Parameters
  m = len(double_root_points)

  # The shape parameter (alpha) determines the specific family of
  # Laguerre polynomials. For the uncertainty principle in 1D,
  # we have alpha = -1/2.
  alpha = sympy.Rational(1, 2) - 1
  x = sympy.symbols('x')

  # 2. Define the polynomial basis: We need 2m + 2 polynomials
  # to satisfy 2m + 2 constraints.
  degrees = np.arange(0, 4 * m + 4, 2)
  basis_polys = [
      sympy.polys.orthopolys.laguerre_poly(
          n=int(d), x=x, alpha=alpha, polys=False
      )
      for d in degrees
  ]
  num_basis = len(basis_polys)
  num_constraints = 2 * m + 2
  # Pre-calculate derivatives of the basis for the matrix construction.
  basis_derivs = [p.diff(x) for p in basis_polys]

  # 3. Initialize the linear system for finding the coefficients.
  matrix = sympy.Matrix.zeros(num_constraints, num_basis)
  targets = sympy.Matrix.zeros(num_constraints, 1)

  # 4. Apply the boundary conditions at x=0: g(0) = 0 and g'(0) = 1.
  targets[1] = 1
  for j in range(num_basis):
    matrix[0, j] = basis_polys[j].subs(x, 0)
    matrix[1, j] = basis_derivs[j].subs(x, 0)

  # 5. Apply the double root conditions at each root z: g(z) = 0, g'(z) = 0.
  for i, val in enumerate(double_root_points):
    z_i = sympy.Rational(val)
    row_val = 2 * i + 2
    row_der = 2 * i + 3
    for j in range(num_basis):
      matrix[row_val, j] = basis_polys[j].subs(x, z_i)
      matrix[row_der, j] = basis_derivs[j].subs(x, z_i)

  # 6. Solve the system using LUsolve and assemble the final function g(x).
  coeffs = matrix.LUsolve(targets)

  return sum(coeffs[j] * basis_polys[j] for j in range(num_basis))


def verify_laguerre_construction(
    g_fn: sympy.Expr, expected_double_roots: list[float]
) -> sympy.Expr | None:
  """Verifies that the constructed function g(x) satisfies the constraints.

  Constraints:
  1. g(0) == 0.
  2. g'(0) == 1
  3. g(z) == 0 for all z in expected_double_roots.
  4. g'(z) == 0 for all z in expected_double_roots.
  5. gq_fn (g after factoring out known roots) has real roots (sign changes).

  Args:
    g_fn: The constructed function g(x).
    expected_double_roots: The list of expected double roots.

  Returns:
    gq_fn (g after factoring out known roots) if the construction is valid,
    None otherwise.
  """
  x = sympy.symbols('x')
  dg_fn = sympy.diff(g_fn, x)

  # 1. Check conditions at x = 0
  # Verify g(0) = 0.
  if not g_fn.subs(x, 0).is_zero:
    # Constraint failed: g(0) is not 0.
    return None

  # Verify g'(0) = 1.
  if not (dg_fn.subs(x, 0) - 1).is_zero:
    # Constraint failed: g'(0) is not 1.
    return None

  # 2. Check double roots at all points of expected_double_roots.
  for z in expected_double_roots:
    z_rational = sympy.Rational(z)
    # Verify g(z) = 0.
    if not g_fn.subs(x, z_rational).is_zero:
      # Constraint failed: g(z) is not 0.
      return None
    # Verify g'(z) = 0.
    if not dg_fn.subs(x, z_rational).is_zero:
      # Constraint failed: g'(z) is not 0.
      return None

  # 3. Factor out the known roots to check for the 'uncertainty' sign change,
  # div = x * product of (x - z_i)^2.
  div = x * sympy.prod(
      [(x - sympy.Rational(z)) ** 2 for z in expected_double_roots]
  )

  try:
    # Compute exact quotient.
    gq_fn = sympy.exquo(g_fn, div)
  except sympy.polys.polyerrors.BasePolynomialError:
    # Polynomial division failed.
    return None

  # 4. Check for real roots in the quotient (finding the upper bound).
  if not sympy.real_roots(gq_fn, x):
    # Verification failed: No sign changes (real roots) found in the quotient.
    return None

  # All constraints verified.
  return gq_fn


def compute_upper_bound(gq_fn: sympy.Expr) -> float:
  """Computes the uncertainty principle upper bound from the quotient function.

  Args:
    gq_fn: The quotient function gq_fn (g after factoring out known roots).

  Returns:
    The upper bound for the uncertainty inequality.
  """
  x = sympy.symbols('x')
  largest_sign_change = sympy.Integer(0)

  # Define a truly tiny epsilon for sign checking.
  epsilon = sympy.Rational(1, 10**198)

  for root in sympy.real_roots(gq_fn, x):
    # High precision evaluation
    approx_root = root.eval_rational(n=200)

    # Test points
    val_p = gq_fn.subs(x, approx_root + epsilon)
    val_m = gq_fn.subs(x, approx_root - epsilon)

    # Check for sign change (crossing)
    if (val_p * val_m) < 0:
      if approx_root > largest_sign_change:
        largest_sign_change = approx_root

  return float(largest_sign_change) / (2 * np.pi)


def get_score(roots: list[float]) -> float:
  """Returns the score of a Laguerre combination given the roots.

  Args:
    roots: The target list of double roots.

  Returns:
    The score of the given Laguerre combination, or `-inf` if invalid. The score
    equals the negative of the upper bound for the uncertainty inequality.
  """
  if not (
      isinstance(roots, list)
      and len(roots) == 12
      and all(isinstance(x, (int, float)) for x in roots)
      and np.max(np.abs(roots)) < 500
      and np.min(roots) >= 0
  ):
    return -float('inf')

  roots = sorted(roots)

  g_fn = construct_laguerre_combination(roots)
  if g_fn is None:
    return -float('inf')
  gq_fn = verify_laguerre_construction(g_fn, roots)
  if gq_fn is None:
    return -float('inf')
  return -compute_upper_bound(gq_fn)


def evaluate():
  """Evaluates the `search_for_best_roots` function."""
  initial_roots = [
      3.5,
      6.0,
      30.0,
      40.0,
      45.0,
      50.0,
      60.0,
      80.0,
      100.0,
      115.0,
      125.0,
      150.0,
  ]
  parent_roots = globals().get('PARENT_OUTPUT', initial_roots)
  # If the parent roots are invalid, start from scratch.
  if get_score(parent_roots) == -float('inf'):
    parent_roots = initial_roots

  new_roots = search_for_best_roots(parent_roots)

  score = get_score(new_roots)
  if score > 0:
    # This is theoretically impossible, so we return a very small score
    # to indicate that something went wrong.
    score = -float('inf')

  return {
      'scores_to_maximize': {'score': score},
      'output_artifacts': new_roots,
  }
