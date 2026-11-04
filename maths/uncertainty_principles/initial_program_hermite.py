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

"""Initial program, uncertainty principle inequality with Hermite polynomials.

Note that this is a minimization problem, but AlphaEvolve can only maximize, so
we maximize the negative of the objective function.
"""

# EVOLVE-BLOCK-START
import time
import numpy as np
import sympy


def search_for_best_coefficients(init_coeffs: list[float]) -> list[float]:
  """Searches for coefficients for optimal Hermite polynomial combination."""
  best_coeffs = init_coeffs
  best_score = get_score(best_coeffs)
  coeffs = best_coeffs.copy()
  rng = np.random.default_rng()

  start_time = time.time()
  while time.time() - start_time < rng.integers(100, 1000):
    # Random mutation.
    coeffs[rng.integers(0, len(coeffs))] += rng.standard_normal()
    new_score = get_score(coeffs)
    if new_score > best_score:
      best_score = new_score
      best_coeffs = coeffs.copy()

    # Reset to best with probability 0.5.
    if rng.random() < 0.5:
      coeffs = best_coeffs.copy()

  return best_coeffs


# EVOLVE-BLOCK-END


def compute_hermite_combination(coeffs: list[float]) -> sympy.Expr:
  """Constructs a linear combination of Hermite polynomials given coefficients.

  The resulting polynomial:
  1. Uses degrees 0, 4, 8, ..., 4m.
  2. Has a root at x=0 by calculating a specific final coefficient.
  3. Is normalized to be positive as x approaches infinity.

  Args:
    coeffs: A list of real coefficients c_i.

  Returns:
    A sympy expression for the linear combination of the Hermite polynomials.
  """
  m = len(coeffs)
  x = sympy.symbols('x')

  # Map degrees to 0, 4, 8, 12, etc.
  degrees = np.arange(0, 4 * m + 4, 4)
  rational_coeffs = [sympy.Rational(c) for c in coeffs]

  # Generate Hermite polynomials H_0, H_4, H_8, etc.
  hermite_polys = [
      sympy.polys.orthopolys.hermite_poly(n=d, x=x, polys=False)
      for d in degrees
  ]

  # Sum initial terms: partial_result = c0*H0 + c4*H4 + etc.
  partial_result = sympy.Add(
      *(c * h for c, h in zip(rational_coeffs, hermite_polys))
  )

  # Solve for the final coefficient so that final_result(0) = 0.
  # Equation: partial_result(0) + last_coeff * last_hermite(0) = 0.
  h_last_at_zero = hermite_polys[-1].subs(x, 0)
  partial_at_zero = partial_result.subs(x, 0)

  last_coeff = sympy.Rational(-partial_at_zero / h_last_at_zero)
  rational_coeffs.append(last_coeff)

  # Reconstruct the full polynomial.
  final_poly = sum(c * h for c, h in zip(rational_coeffs, hermite_polys))

  # Ensure leading behavior is positive (positive at infinity).
  if sympy.limit(final_poly, x, sympy.oo) < 0:
    final_poly = -final_poly

  return final_poly


# Precision constants for sign-change detection
PRECISION = 200
EPSILON = sympy.Rational(1, 10**198)  # Equivalent to 1e-198


def get_score(coeffs: list[float]) -> float:
  """Evaluates score of a coefficient set based on roots of the Hermite poly.

  The score identifies the largest real x-value where the polynomial crosses
  the x-axis (sign change). The score is returned as -x^2 / 2π.

  Args:
    coeffs: A list of real coefficients c_i. Must contain 3 floats. The maximum
      absolute value of coefficients must be between 1e-14 and 1000.

  Returns:
    The final score of the coefficients.
  """
  if (
      not isinstance(coeffs, list)
      or len(coeffs) != 3
      or not all(isinstance(x, (int, float, np.floating)) for x in coeffs)
      or max([abs(x) for x in coeffs]) > 1000
      or max([abs(x) for x in coeffs]) < 1e-14
  ):
    return float('-inf')

  x = sympy.symbols('x')
  poly_expr = compute_hermite_combination(coeffs)

  # Divide by x^2 to remove the known trivial root at the origin.
  reduced_poly = sympy.exquo(poly_expr, x**2)

  # Find all real roots of the remaining polynomial.
  real_roots = sympy.real_roots(reduced_poly, x)

  largest_crossing_root = 0

  for root in real_roots:
    # High-precision rational approximation.
    approx_root = root.eval_rational(n=PRECISION)

    # Check for sign change around the root (verifies it's not just a tangent).
    val_plus = reduced_poly.subs(x, approx_root + EPSILON)
    val_minus = reduced_poly.subs(x, approx_root - EPSILON)

    crosses_axis = (val_plus * val_minus) < 0

    if crosses_axis:
      largest_crossing_root = max(largest_crossing_root, approx_root)

  # Return the normalized score.
  return -float(largest_crossing_root**2) / (2 * np.pi)


def evaluate() -> tuple[dict[str, float], dict[str, str]]:
  """Evaluates the `search_for_best_coefficients` function."""

  old_coeffs = globals().get('PARENT_OUTPUT', [0.5, 0, 0])
  new_coeffs = search_for_best_coefficients(old_coeffs)

  score = get_score(new_coeffs)
  if score > 0:
    # This is theoretically impossible, so we return a very small score
    # to indicate that something went wrong.
    score = -float('inf')

  return {
      'scores_to_maximize': {'score': score},
      'output_artifacts': new_coeffs,
  }
