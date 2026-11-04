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

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name
"""Code for verifying a solution for combination loss analysis for omega.

This verification file uses fast, floating-point calculations and is prone to
small numerical errors; it's used to help AlphaEvolve hill-climb quickly. In
contrast, the verification Colab runs much slower but uses rational arithmetic
to ensure rigorous upper bounds on omega.
"""

import dataclasses
import math
from typing import Callable, Sequence

import numpy as np
from scipy import linalg
from scipy import special


class ParamManager:
  """Manages registration and value management of parameters."""

  def __init__(self):
    self.num_group = 0
    self.num_input = 0
    self.group_startpos: list[int | None] = [None]  # 1-based indexing
    self.group_size: list[int | None] = [None]
    self._lb_parts: list[np.ndarray] = []
    self._ub_parts: list[np.ndarray] = []
    self._finalized_box_constraints = False
    self.lb = np.array([], dtype=float)
    self.ub = np.array([], dtype=float)
    self.cur_x = np.array([], dtype=float)
    self._cur_x_valid = False
    self.initializer: list[Callable[[int], np.ndarray] | None] = [None]

  # Registration and value management
  def register(
      self, param_len: int, lb=-math.inf, ub=math.inf, initer=None
  ) -> int:
    if np.isscalar(lb):
      lb_vec = np.full(param_len, lb, dtype=float)
    else:
      lb_vec = np.array(lb, dtype=float)
    if np.isscalar(ub):
      ub_vec = np.full(param_len, ub, dtype=float)
    else:
      ub_vec = np.array(ub, dtype=float)

    self.num_group += 1
    group_id = self.num_group
    start = self.num_input
    self.group_startpos.append(start)
    self.group_size.append(param_len)
    self.num_input += param_len

    # Create list of box constraints (lower and upper bounds) for each part.
    # This will later be concatenated into a single vector, after which
    # self._finalized_box_constraints will be set to True.
    self._lb_parts.append(lb_vec)
    self._ub_parts.append(ub_vec)
    self._finalized_box_constraints = False
    self._cur_x_valid = False

    # Default initializer: uniform
    if initer is None:
      initer = (0.0, 0.01)
    if callable(initer):
      self.initializer.append(initer)  # pyrefly: ignore[bad-argument-type]
    else:
      lo, hi = initer

      def _init(sz: int, lo=lo, hi=hi):
        return np.random.uniform(lo, hi, size=sz)

      self.initializer.append(_init)

    return group_id

  def _finalize_box_constraints(self):
    if self._finalized_box_constraints:
      return
    self.lb = (
        np.concatenate(self._lb_parts)
        if self._lb_parts
        else np.array([], dtype=float)
    )
    self.ub = (
        np.concatenate(self._ub_parts)
        if self._ub_parts
        else np.array([], dtype=float)
    )
    self._finalized_box_constraints = True

  def _ensure_cur_x(self):
    # Ensure that x is "valid" in the sense that it has the correct length. This
    # can be achieved by padding if necessary.
    if self._cur_x_valid:
      return
    if self.cur_x.size < self.num_input:
      self.cur_x = np.pad(
          self.cur_x, (0, self.num_input - self.cur_x.size), mode="constant"
      )
    self._cur_x_valid = True

  def set_value(self, x: Sequence[float]):
    # Ensure all bounds are concatenated and self.lb and ub are set.
    self._finalize_box_constraints()
    x_vec = np.array(x, dtype=float)
    if x_vec.size != self.num_input:
      raise ValueError(
          f"Expected vector of length {self.num_input}, got {x_vec.size}"
      )
    self.cur_x = np.minimum(np.maximum(x_vec, self.lb), self.ub)
    self._cur_x_valid = True

  def set_single_param(self, group_id: int, value: Sequence[float] | float):
    self._finalize_box_constraints()
    self._ensure_cur_x()
    start = self.group_startpos[group_id]
    size = self.group_size[group_id]
    val = np.array(value, dtype=float)
    self.cur_x[start : start + size] = np.minimum(  # pyrefly: ignore[unsupported-operation]
        np.maximum(val, self.lb[start : start + size]),  # pyrefly: ignore[unsupported-operation]
        self.ub[start : start + size],  # pyrefly: ignore[unsupported-operation]
    )

  def random_init(self):
    self._finalize_box_constraints()
    x = np.zeros(self.num_input, dtype=float)
    for gid in range(1, self.num_group + 1):
      init_fn = self.initializer[gid]
      size = self.group_size[gid]
      x[self.group_startpos[gid] : self.group_startpos[gid] + size] = init_fn(  # pyrefly: ignore[not-callable, unsupported-operation]
          size  # pyrefly: ignore[bad-argument-type]
      )
    self.cur_x = np.minimum(np.maximum(x, self.lb), self.ub)
    self._cur_x_valid = True
    return x

  def get_param(self, group_id: int) -> np.ndarray:
    self._ensure_cur_x()
    start = self.group_startpos[group_id]
    size = self.group_size[group_id]
    return self.cur_x[start : start + size]  # pyrefly: ignore[unsupported-operation]

  # Constraint packing
  def pack_results(self, values_list) -> np.ndarray:
    if not values_list:
      return np.zeros(0, dtype=float)
    values = [np.asarray(v).reshape(-1) for v in values_list]
    return np.concatenate(values)


@dataclasses.dataclass
class EvalContext:
  q: float
  max_level: int
  K: float
  param_manager: ParamManager
  terms: list[list[object]]
  # maps (shape, identifier) (i.e. a term "fingerprint")
  # to an index in terms array
  term_index: list[dict[tuple[tuple[int, ...], tuple[int, ...]], int]]
  use_term_sharing: bool
  use_newton: bool


def entropy(arr: np.ndarray | float) -> float:
  return float(np.sum(special.entr(np.asarray(arr, dtype=float))))


def normalize_distribution(vec: np.ndarray) -> np.ndarray:
  assert np.all(vec >= 0)
  assert vec.dtype == np.float64
  vec = np.maximum(vec, 1e-18)
  vec = vec / np.sum(vec)
  assert np.allclose(np.sum(vec), 1.0, atol=1e-12)
  return vec


def project_distribution_to_linear_space(
    A: np.ndarray, dist_x: np.ndarray, dist_y: np.ndarray
) -> np.ndarray:
  """Find y' close to y so that x @ A == y' @ A and y' is a distribution."""
  delta_T = np.linalg.lstsq(A.T, ((dist_x - dist_y) @ A).T, rcond=None)[0]
  projected_y = dist_y + delta_T.T
  # At this point, y might have negative entries.
  # Shift back to the positive space by adding a multiple of x - y.
  # Calculate the minimum t so that y + t * (x - y) >= 1e-24

  # Returns the smallest floating-point number larger than x.
  def next_larger(x):
    return np.nextafter(x, np.inf)

  eps = 1e-24
  diff = dist_x - projected_y
  mask = (projected_y < eps) & (diff > eps)
  # We are shifting y to a slightly larger floating-point value to ensure
  # t is a valid upper bound.
  if np.any(mask):
    t = np.max(next_larger(next_larger(eps - projected_y[mask]) / diff[mask]))
  else:
    t = 0.0
  shifted_y = next_larger(
      projected_y + next_larger(np.clip(t, 0.0, 1.0) * diff)
  )
  assert np.all(shifted_y > 0)
  assert np.allclose(shifted_y @ A, dist_x @ A, atol=1e-12)
  assert np.allclose(np.sum(shifted_y), 1.0, atol=1e-12)
  return shifted_y


def _check_distribution(dist: np.ndarray, name: str):
  """Checks dist is float64, 1D, non-negative, and sums to 1 within 1e-12."""
  if dist.dtype != np.float64:
    raise ValueError(
        f"Expected `{name}` to have dtype float64, got {dist.dtype}."
    )
  if dist.ndim != 1:
    raise ValueError(f"Expected `{name}` to be 1D, got shape {dist.shape}.")
  if np.any(dist < 0):
    raise ValueError(
        f"Expected `{name}` to be non-negative, got {np.min(dist)}."
    )
  if np.sum(dist) - 1.0 > 1e-12:
    raise ValueError(f"Expected `{name}` to sum to 1, got {np.sum(dist)}.")


def _safe_log(arr: np.ndarray) -> np.ndarray:
  eps = np.finfo(float).eps
  return np.log(np.maximum(arr, eps))


def solve_max_entropy_newton(
    joint_to_margin_matrix: np.ndarray,
    dist_orig: np.ndarray,
    dist_max_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Finds a max-entropy distribution with the same marginals as `dist_orig`.

  This function solves the maximum entropy problem:
      maximize -sum(p * log(p))
      such that p @ joint_to_margin_matrix == dist_orig @ joint_to_margin_matrix
                sum(p) == 1
                p >= 0

  It uses a custom Newton method with line search to find the optimal
  distribution. It also computes the Lagrange multipliers associated with the
  constraints.

  Args:
    joint_to_margin_matrix: A 2D numpy array of shape `(num_joint_states,
      num_marginals)` that maps a joint distribution to its marginals.
    dist_orig: A 1D float64 numpy array of shape `(num_joint_states,)`
      representing the original distribution whose marginals we want to
      preserve. It should sum to 1.
    dist_max_init: An optional 1D float64 numpy array of shape
      `(num_joint_states,)` representing the initial guess for the maximum
      entropy distribution. It should sum to 1. If None, optimisation starts
      from a uniform distribution.

  Returns:
    A tuple `(dist_max, lam_margin, lam_sum)` where:
      - dist_max: A 1D float64 numpy array of shape `(num_joint_states,)`
        containing the maximum entropy distribution. It is guaranteed to have
        marginal_err and sum_err less than 1e-12.
      - lam_margin: A 1D float64 numpy array of shape `(num_marginals,)`
        containing the Lagrange multipliers for the marginal constraints.
      - lam_sum: A scalar float64 numpy array containing the Lagrange multiplier
        for the sum-to-one constraint.
    The Lagrange multipliers should satisfy:
      joint_to_margin_matrix @ lam_margin + lam_sum = np.log(dist_max) + 1.0.
  """
  _check_distribution(dist_orig, "dist_orig")
  if dist_max_init is not None:
    _check_distribution(dist_max_init, "dist_max_init")

  if joint_to_margin_matrix.ndim != 2:
    raise ValueError(
        "Expected `joint_to_margin_matrix` to be 2D, got shape "
        f"{joint_to_margin_matrix.shape}."
    )
  if joint_to_margin_matrix.shape[0] != dist_orig.shape[0]:
    raise ValueError(
        "Expected `joint_to_margin_matrix.shape[0]` to match "
        f"`dist_orig.shape[0]`, got {joint_to_margin_matrix.shape[0]} and "
        f"{dist_orig.shape[0]}."
    )

  num_shape = dist_orig.shape[0]
  target_marginals = dist_orig @ joint_to_margin_matrix

  # Compute a basis for the image of `joint_to_margin_matrix` using SVD.
  _, singular_values, vh = linalg.svd(
      joint_to_margin_matrix, full_matrices=False
  )
  rank = int(np.sum(singular_values > 1e-10))
  basis = vh[:rank].T
  target_basis = basis.T @ target_marginals

  # Define the objective function and gradient in terms of the coefficients `u`
  # of the basis expansion of the distribution.
  def objective(u: np.ndarray) -> np.float64:
    logits = joint_to_margin_matrix @ (basis @ u)
    return np.float64(special.logsumexp(logits) - np.dot(target_basis, u))

  def gradient(u: np.ndarray) -> np.ndarray:
    logits = joint_to_margin_matrix @ (basis @ u)
    probs = np.exp(logits - special.logsumexp(logits))
    return basis.T @ (joint_to_margin_matrix.T @ probs - target_marginals)

  # Warm-start from registered dist_max if available.
  if dist_max_init is not None:
    logits_init = _safe_log(dist_max_init)
    u = np.linalg.lstsq(
        joint_to_margin_matrix @ basis, logits_init, rcond=None
    )[0]
  else:
    u = np.zeros(rank, dtype=np.float64)

  # Use a custom Newton method with line search to refine the solution.
  best_u = u.copy()
  best_grad = np.max(np.abs(gradient(u)))
  for _ in range(200):
    logits = joint_to_margin_matrix @ (basis @ u)
    log_z = special.logsumexp(logits)
    probs = np.exp(logits - log_z)
    grad = basis.T @ (joint_to_margin_matrix.T @ probs - target_marginals)
    grad_norm = np.max(np.abs(grad))
    if grad_norm < best_grad:
      best_grad = grad_norm
      best_u = u.copy()
    if grad_norm < 1e-14:
      break
    dp = np.diag(probs) - np.outer(probs, probs)
    hess = (
        basis.T @ joint_to_margin_matrix.T @ dp @ joint_to_margin_matrix @ basis
    )
    eigvals, eigvecs = np.linalg.eigh(hess)
    reg_eigvals = np.maximum(eigvals, 1e-15)
    step = -eigvecs @ (eigvecs.T @ grad / reg_eigvals)
    alpha = 1.0
    obj_curr = objective(u)
    for _ in range(30):
      trial_u = u + alpha * step
      if objective(trial_u) <= obj_curr + 1e-4 * alpha * np.dot(grad, step):
        break
      alpha *= 0.5
    u = u + alpha * step

  # Compute the final distribution dist_max from best_u.
  logits = joint_to_margin_matrix @ (basis @ best_u)
  log_z = special.logsumexp(logits)
  dist_max = np.exp(logits - log_z)

  # Check the marginal and sum-to-one constraint violations and project to
  # linear space if necessary.
  marginal_err = np.max(
      np.abs(dist_max @ joint_to_margin_matrix - target_marginals)
  )
  sum_err = abs(np.sum(dist_max) - 1.0)
  if marginal_err > 1e-12 or sum_err > 1e-12:
    dist_max = project_distribution_to_linear_space(
        joint_to_margin_matrix, dist_orig, dist_max
    )
    marginal_err = np.max(
        np.abs(dist_max @ joint_to_margin_matrix - target_marginals)
    )
    sum_err = abs(np.sum(dist_max) - 1.0)

  assert marginal_err < 1e-12, f"Marginals violation: {marginal_err}"
  assert sum_err < 1e-12, f"Sum-to-one violation: {sum_err}"
  assert np.all(dist_max >= 0), f"Negative values in distribution: {dist_max}"

  # Compute the Lagrange multipliers for the marginal and sum-to-one constraints
  # by solving a linear system with least squares.
  log_p = _safe_log(dist_max)
  system_rhs = log_p + 1.0
  system_mat = np.hstack([joint_to_margin_matrix, np.ones((num_shape, 1))])
  x_ls = np.linalg.lstsq(system_mat, system_rhs, rcond=None)[0]
  lam_margin = np.asarray(x_ls[:-1], dtype=np.float64)
  lam_sum = np.asarray(x_ls[-1], dtype=np.float64)

  return dist_max, lam_margin, lam_sum


def concat_matrix_by_columns(mats: list[np.ndarray]) -> np.ndarray:
  return np.concatenate(mats, axis=1)


def normalized_entropy(arr: np.ndarray, p: float | np.ndarray) -> float:
  p_val = float(np.asarray(p).item()) if isinstance(p, np.ndarray) else float(p)
  if p_val <= 0:
    return 0.0
  val = np.clip(np.asarray(arr, dtype=float), 0.0, None)
  normalized = val / p_val
  return float(np.sum(-val * _safe_log(normalized)))


_DIMS_TABLE = (
    (1, 2, 3),
    (1, 3, 2),
    (2, 1, 3),
    (2, 3, 1),
    (3, 1, 2),
    (3, 2, 1),
)


def dims(region: int) -> tuple[int, int, int]:
  return _DIMS_TABLE[region - 1]


def prepare_shapes(level: int) -> np.ndarray:
  sum_col = 2**level
  shapes = []
  for i in range(sum_col + 1):
    for j in range(sum_col - i + 1):
      k = sum_col - i - j
      shapes.append((i, j, k))
  return np.array(shapes, dtype=int)


def prepare_splits(shape: Sequence[int] | np.ndarray) -> np.ndarray:
  shape = np.asarray(shape, dtype=int)
  sum_col = int(np.sum(shape))
  sum_half = sum_col // 2
  splits = []
  for i in range(min(sum_half, shape[0]) + 1):
    for j in range(min(sum_half - i, shape[1]) + 1):
      k = sum_half - i - j
      if k > shape[2]:
        continue
      splits.append((i, j, k, shape[0] - i, shape[1] - j, shape[2] - k))
  return np.array(splits, dtype=int)


def encode_csd(arr: Sequence[int] | np.ndarray) -> int:
  arr = list(arr)
  val = 0
  for a in arr:
    val = val * 3 + int(a)
  return val + 1


def decode_csd(idx: int, power: int) -> np.ndarray:
  arr = [0] * power
  val = idx - 1
  for i in range(power - 1, -1, -1):
    arr[i] = val % 3
    val //= 3
  return np.array(arr, dtype=int)


def concat_csd(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
  return np.kron(lhs, rhs)


def joint_to_margin(column: np.ndarray, sum_col: int) -> list[np.ndarray]:
  n_col = column.shape[0]
  matrices = []
  for dim in range(3):
    mat = np.zeros((n_col, sum_col + 1), dtype=float)
    for i in range(n_col):
      mat[i, column[i, dim]] = 1.0
    matrices.append(mat)
  return matrices


def marginal_dist(
    alph: np.ndarray, joint_to_margin_mats: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  return (
      alph @ joint_to_margin_mats[0],
      alph @ joint_to_margin_mats[1],
      alph @ joint_to_margin_mats[2],
  )


def rot3(vec: np.ndarray, n: int = 1) -> np.ndarray:
  vec = np.asarray(vec)
  assert vec.shape[0] == 3
  return np.roll(vec, shift=-n, axis=0)


def rot3c(vec: list[np.ndarray], n: int = 1) -> list[np.ndarray]:
  n = n % 3
  if n == 0:
    return vec
  if n == 1:
    return [vec[1], vec[2], vec[0]]
  return [vec[2], vec[0], vec[1]]


def create_context(
    q: float,
    max_level: int,
    K: float,
    use_term_sharing: bool = False,
    use_newton: bool = False,
) -> EvalContext:
  pm = ParamManager()
  terms: list[list[object]] = [[] for _ in range(max_level + 1)]
  term_index: list[dict[tuple[tuple[int, ...], tuple[int, ...]], int]] = [
      {} for _ in range(max_level + 1)
  ]
  return EvalContext(
      q=q,
      max_level=max_level,
      K=K,
      param_manager=pm,
      terms=terms,
      term_index=term_index,
      use_term_sharing=use_term_sharing,
      use_newton=use_newton,
  )


def _term_key(
    shape: Sequence[int] | np.ndarray,
    identifier: Sequence[int] | np.ndarray,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
  return (tuple(int(s) for s in shape), tuple(int(i) for i in identifier))


def find_term_by_identifier(
    context: EvalContext,
    level: int,
    shape: Sequence[int],
    identifier: Sequence[int],
):
  return context.term_index[level].get(_term_key(shape, identifier), -1)


class TermInfoBase:
  """Base class for all term info types."""

  def __init__(self, context: EvalContext):
    self.ctx = context
    self.level = 0
    self.power = 0
    self.sum_col = 0
    self.sum_half = 0
    self.term_id = 0
    self.shape = np.array([], dtype=int)
    self.identifier = np.array([], dtype=int)
    self.mat_size_contribution = None
    self.num_block_contribution = None
    self.term_frac = None
    self.hash_penalty_term = None
    self.complete_split = None
    self.p_compY = None
    self.p_compZ = None

  def _build_common(
      self,
      level: int,
      term_id: int,
      shape: Sequence[int],
      identifier: Sequence[int],
  ):
    self.level = level
    self.power = 2 ** (level - 1)
    self.sum_col = 2**level
    self.sum_half = self.sum_col // 2
    self.term_id = term_id
    self.shape = np.array(shape, dtype=int)
    self.identifier = np.array(identifier, dtype=int)


class TermInfoZero(TermInfoBase):
  """Handles level>=3 shapes containing zeros."""

  def __init__(self, context: EvalContext):
    super().__init__(context)
    self.zero_dim = None
    self.nonzero_dim_1 = None
    self.nonzero_dim_2 = None
    self.base_mat_size = None
    self.complete_split_id = None

  def build(
      self,
      level: int,
      term_id: int,
      shape: Sequence[int],
      identifier: Sequence[int],
  ):
    self._build_common(level, term_id, shape, identifier)
    pm = self.ctx.param_manager

    if shape[0] == 0:
      self.zero_dim = 1
      self.nonzero_dim_1, self.nonzero_dim_2 = 2, 3
      self.base_mat_size = np.array([0, 0, 1], dtype=float)
    elif shape[1] == 0:
      self.zero_dim = 2
      self.nonzero_dim_1, self.nonzero_dim_2 = 1, 3
      self.base_mat_size = np.array([1, 0, 0], dtype=float)
    else:
      self.zero_dim = 3
      self.nonzero_dim_1, self.nonzero_dim_2 = 1, 2
      self.base_mat_size = np.array([0, 1, 0], dtype=float)

    self.complete_split_id = [
        np.ones(3**self.power, dtype=int) * -1 for _ in range(3)
    ]
    self.complete_split_id[self.zero_dim - 1][0] = -2  # always one
    for idx in range(1, 3**self.power + 1):
      arr = decode_csd(idx, self.power)
      if np.sum(arr) != self.shape[self.nonzero_dim_1 - 1]:
        continue
      gid = pm.register(1, 0, 1, (0, 0.01))
      self.complete_split_id[self.nonzero_dim_1 - 1][idx - 1] = gid
      arr_opposite = np.ones(self.power, dtype=int) * 2 - arr
      opp_idx = encode_csd(arr_opposite)
      self.complete_split_id[self.nonzero_dim_2 - 1][opp_idx - 1] = gid

    zeros_vec = np.zeros(3, dtype=float)
    self.num_block_contribution = [zeros_vec for _ in range(6)]
    self.hash_penalty_term = [0.0] * 6
    self.p_compY = [0.0] * 6
    self.p_compZ = [0.0] * 6

  def evaluate_init(self):
    pm = self.ctx.param_manager
    self.term_frac = 0.0
    self.complete_split = []
    for t in range(3):
      vec = np.zeros(3**self.power, dtype=float)
      mask = np.zeros(3**self.power, dtype=bool)
      for idx in range(3**self.power):
        gid = self.complete_split_id[t][idx]  # pyrefly: ignore[unsupported-operation]
        if gid == -1:
          val = 0.0
        elif gid == -2:
          val = 1.0
          mask[idx] = True
        else:
          val = pm.get_param(gid).item()
          mask[idx] = True
        vec[idx] = val
      assert np.any(mask)
      vec[mask] = normalize_distribution(vec[mask])
      self.complete_split.append(vec)

  def evaluate_pre(self):
    pass  # no-op

  def evaluate_post(self):
    q = self.ctx.q
    # entropy of vector
    inner_prod_size = entropy(self.complete_split[self.nonzero_dim_1 - 1])  # pyrefly: ignore[unsupported-operation]
    for idx in range(1, 3**self.power + 1):
      arr = decode_csd(idx, self.power)
      inner_prod_size += (
          self.complete_split[self.nonzero_dim_1 - 1][idx - 1]  # pyrefly: ignore[unsupported-operation]
          * np.sum(arr == 1)
          * math.log(q)
      )
    self.mat_size_contribution = (
        self.term_frac * inner_prod_size * self.base_mat_size  # pyrefly: ignore[unsupported-operation]
    )


class TermInfoLv2(TermInfoBase):
  """Closed-form logic for level 2 terms."""

  def __init__(self, context: EvalContext):
    super().__init__(context)
    self.level = 2
    self.power = 2
    self.sum_col = 4
    self.sum_half = 2
    self.shape_type = None
    self.rotate_num = None
    self.split_0_id = None
    self.split_0 = None
    self.hash_penalty_term = 0.0
    self.p_compY = 0.0
    self.p_compZ = 0.0

  def build(
      self,
      level: int,
      term_id: int,
      shape: Sequence[int],
      identifier: Sequence[int],
  ):
    if level != 2:
      raise ValueError("TermInfoLv2 only supports level 2")
    self._build_common(level, term_id, shape, identifier)
    pm = self.ctx.param_manager

    if max(shape) == 2 and min(shape) == 0:
      self.shape_type = "022"
      standard = [0, 2, 2]
    elif max(shape) == 2:
      self.shape_type = "112"
      standard = [1, 1, 2]
    elif max(shape) == 3:
      if (
          np.array_equal(shape, [0, 1, 3])
          or np.array_equal(shape, [1, 3, 0])
          or np.array_equal(shape, [3, 0, 1])
      ):
        self.shape_type = "013"
        standard = [0, 1, 3]
      else:
        self.shape_type = "031"
        standard = [0, 3, 1]
    else:
      self.shape_type = "004"
      standard = [0, 0, 4]
    self.rotate_num = 0
    tmp_shape = list(shape)
    while tmp_shape != standard:
      tmp_shape = [tmp_shape[2], tmp_shape[0], tmp_shape[1]]
      self.rotate_num += 1

    if self.shape_type in ("112", "022"):
      self.split_0_id = pm.register(1, 0, 0.5, (0, 0.01))

  def evaluate_init(self):
    pm = self.ctx.param_manager
    if self.shape_type in ("112", "022"):
      self.split_0 = pm.get_param(self.split_0_id)  # pyrefly: ignore[bad-argument-type]
    self.term_frac = 0.0

  def evaluate_pre(self):
    pass

  def evaluate_post(self):
    q = self.ctx.q
    s0 = self.split_0.item() if self.split_0 is not None else 0.0
    s1 = 1 - 2 * s0
    if self.shape_type == "022":
      # num_block_contribution
      self.num_block_contribution = np.zeros(3, dtype=float)
      inner_prod_size = entropy(np.array([s0, s0, s1])) + 2 * math.log(q) * s1
      self.mat_size_contribution = np.array([0.0, 0.0, inner_prod_size])
      complete_split = [
          np.array([1, 0, 0, 0, 0, 0, 0, 0, 0.0]),
          np.array([0, 0, s0, 0, s1, 0, s0, 0, 0.0]),
          np.array([0, 0, s0, 0, s1, 0, s0, 0, 0.0]),
      ]
    elif self.shape_type == "112":
      nb = np.array(
          [
              math.log(2),
              math.log(2),
              entropy(np.array([s0, s0, s1])),
          ],
          dtype=float,
      )
      self.num_block_contribution = nb
      ms = 2 * s0 * np.array([0.0, math.log(q), 0.0]) + s1 * np.array(
          [math.log(q), 0.0, math.log(q)]
      )
      self.mat_size_contribution = ms
      complete_split = [
          np.array([0, 0.5, 0, 0.5, 0, 0, 0, 0, 0.0]),
          np.array([0, 0.5, 0, 0.5, 0, 0, 0, 0, 0.0]),
          np.array([0, 0, s0, 0, s1, 0, s0, 0, 0.0]),
      ]
    elif self.shape_type in ("013", "031"):
      self.num_block_contribution = np.zeros(3, dtype=float)
      inner_prod_size = math.log(2) + math.log(q)
      self.mat_size_contribution = np.array([0.0, 0.0, inner_prod_size])
      if self.shape_type == "013":
        complete_split = [
            np.array([1, 0, 0, 0, 0, 0, 0, 0, 0.0]),
            np.array([0, 0.5, 0, 0.5, 0, 0, 0, 0, 0.0]),
            np.array([0, 0, 0, 0, 0, 0.5, 0, 0.5, 0.0]),
        ]
      else:
        complete_split = [
            np.array([1, 0, 0, 0, 0, 0, 0, 0, 0.0]),
            np.array([0, 0, 0, 0, 0, 0.5, 0, 0.5, 0.0]),
            np.array([0, 0.5, 0, 0.5, 0, 0, 0, 0, 0.0]),
        ]
    else:  # 004
      self.num_block_contribution = np.zeros(3, dtype=float)
      self.mat_size_contribution = np.zeros(3, dtype=float)
      complete_split = [
          np.array([1, 0, 0, 0, 0, 0, 0, 0, 0.0]),
          np.array([1, 0, 0, 0, 0, 0, 0, 0, 0.0]),
          np.array([0, 0, 0, 0, 0, 0, 0, 0, 1.0]),
      ]

    self.num_block_contribution = (
        rot3(self.num_block_contribution, self.rotate_num) * self.term_frac  # pyrefly: ignore[bad-argument-type]
    )
    self.mat_size_contribution = (
        rot3(self.mat_size_contribution, self.rotate_num) * self.term_frac  # pyrefly: ignore[bad-argument-type]
    )
    rotated = rot3c(complete_split, self.rotate_num)  # pyrefly: ignore[bad-argument-type]
    self.complete_split = [arr for arr in rotated]


class TermInfo(TermInfoBase):
  """Main term class for level >=3 without zero entries."""

  def __init__(self, context: EvalContext):
    super().__init__(context)
    self.splits = None
    self.num_split = None
    self.joint_to_margin = None
    self.left_idx = None
    self.right_idx = None
    self.left_ptr = None
    self.right_ptr = None
    self.region_prop_id = None
    self.region_prop = None
    self.split_dist_id = None
    self.split_dist = None
    self.split_dist_max_id = None
    self.split_dist_max = None
    self.lam_margin_low = None
    self.lam_margin_high = None
    self.lam_margin_id = None
    self.lam_sum_id = None
    self.lam_margin = None
    self.lam_sum = None
    self.derivative_upper_bound_id = None
    self.derivative_upper_bound = None
    self.complete_split_region = None

  def find_or_create_term(
      self,
      context: EvalContext,
      level: int,
      shape: Sequence[int],
      identifier: Sequence[int],
  ):
    key = _term_key(shape, identifier)
    idx = context.term_index[level].get(key, -1)
    if idx != -1:
      return idx, context.terms[level][idx], False
    term_cls = self.create_instance(level, shape)
    term = term_cls(context)
    idx = len(context.terms[level])
    context.terms[level].append(term)
    context.term_index[level][key] = idx
    return idx, term, True

  def build(
      self,
      level: int,
      term_id: int,
      shape: Sequence[int],
      identifier: Sequence[int],
  ):
    self._build_common(level, term_id, shape, identifier)
    ctx = self.ctx
    self.splits = prepare_splits(self.shape)
    self.num_split = self.splits.shape[0]
    self.joint_to_margin = joint_to_margin(self.splits[:, :3], self.sum_half)

    # Lagrange multiplier bounds
    self.lam_margin_low = np.ones(3, dtype=float) * np.inf
    self.lam_margin_high = np.ones(3, dtype=float) * -np.inf
    for row in self.splits:
      for t in range(3):
        self.lam_margin_low[t] = min(self.lam_margin_low[t], row[t])
        self.lam_margin_high[t] = max(self.lam_margin_high[t], row[t])
    self.lam_margin_low = self.lam_margin_low.astype(int)
    self.lam_margin_high = self.lam_margin_high.astype(int)

    self._register_variables_and_constraints()

    # Recursively build lower-level terms
    self.left_idx = [[0] * self.num_split for _ in range(6)]
    self.right_idx = [[0] * self.num_split for _ in range(6)]
    self.left_ptr = [[None] * self.num_split for _ in range(6)]
    self.right_ptr = [[None] * self.num_split for _ in range(6)]
    for r in range(6):
      for i in range(self.num_split):
        if self.ctx.use_term_sharing:
          ident = np.append(self.shape, r + 1)
        else:
          ident = np.array([self.term_id, r + 1])
        left_id, left_ptr, is_new = self.find_or_create_term(
            ctx, level - 1, self.splits[i, :3], ident  # pyrefly: ignore[bad-argument-type]
        )
        if is_new:
          assert hasattr(left_ptr, "build")  # For pytype.
          left_ptr.build(level - 1, left_id + 1, self.splits[i, :3], ident)
        right_id, right_ptr, is_new = self.find_or_create_term(
            ctx, level - 1, self.splits[i, 3:], ident  # pyrefly: ignore[bad-argument-type]
        )
        if is_new:
          assert hasattr(right_ptr, "build")  # For pytype.
          right_ptr.build(level - 1, right_id + 1, self.splits[i, 3:], ident)
        self.left_idx[r][i] = left_id + 1
        self.right_idx[r][i] = right_id + 1
        self.left_ptr[r][i] = left_ptr
        self.right_ptr[r][i] = right_ptr

  def _register_variables_and_constraints(self):
    pm = self.ctx.param_manager
    assert self.num_split is not None
    assert self.lam_margin_high is not None
    assert self.lam_margin_low is not None
    assert self.joint_to_margin is not None
    assert self.sum_half is not None
    self.split_dist_id = [[] for _ in range(6)]
    self.split_dist_max_id = [[] for _ in range(6)]
    for r in range(6):
      self.split_dist_id[r] = pm.register(
          self.num_split, 0, 1, (0, 1 / max(self.num_split, 1))
      )
      self.split_dist_max_id[r] = pm.register(
          self.num_split, 0, 1, (0, 1 / max(self.num_split, 1))
      )
    self.region_prop_id = pm.register(6, 0, 1, (0, 1))
    self.lam_margin_id = [[None] * 3 for _ in range(6)]
    self.lam_sum_id = [None] * 6
    self.derivative_upper_bound_id = [None] * 6
    for r in range(6):
      for t in range(3):
        size = int(self.lam_margin_high[t] - self.lam_margin_low[t] + 1)
        self.lam_margin_id[r][t] = pm.register(
            size, -math.inf, math.inf, (-0.01, 0.01)
        )
      self.lam_sum_id[r] = pm.register(1, -math.inf, math.inf, (-0.01, 0.01))
      self.derivative_upper_bound_id[r] = pm.register(
          1, 0, math.inf, (0.01, 0.02)
      )

  def evaluate_init(self):
    pm = self.ctx.param_manager
    self.region_prop = normalize_distribution(pm.get_param(self.region_prop_id))  # pyrefly: ignore[bad-argument-type]
    self.split_dist = [
        normalize_distribution(pm.get_param(self.split_dist_id[r]))  # pyrefly: ignore[unsupported-operation]
        for r in range(6)
    ]
    split_dist_max_init = [
        normalize_distribution(pm.get_param(self.split_dist_max_id[r]))  # pyrefly: ignore[unsupported-operation]
        for r in range(6)
    ]
    if self.ctx.use_newton:
      self.split_dist_max = []
      self.lam_sum = [None] * 6
      self.lam_margin = [[None] * 3 for _ in range(6)]
      margin_len = self.joint_to_margin[0].shape[1]  # pyrefly: ignore[unsupported-operation]
      for r in range(6):
        # Compute dist_max, lam_margin and lam_sum using Newton's method.
        dist_max_r, lam_margin_r, lam_sum_r = solve_max_entropy_newton(
            concat_matrix_by_columns(self.joint_to_margin),  # pyrefly: ignore[bad-argument-type]
            self.split_dist[r],
            dist_max_init=split_dist_max_init[r],
        )
        self.split_dist_max.append(dist_max_r)
        self.lam_sum[r] = np.array([lam_sum_r])

        # lam_margin_r is shape (margin_len * 3,). For each of the chunks
        # t=0,1,2, select the indices of self.lam_margin from lam_margin_low[t]
        # to lam_margin_high[t].
        for t in range(3):
          full_lam_t = lam_margin_r[t * margin_len : (t + 1) * margin_len]
          low = int(self.lam_margin_low[t])  # pyrefly: ignore[unsupported-operation]
          high = int(self.lam_margin_high[t])  # pyrefly: ignore[unsupported-operation]
          self.lam_margin[r][t] = full_lam_t[low : high + 1]

    else:
      # Compute dist_max by projecting to the linear space.
      self.split_dist_max = [
          project_distribution_to_linear_space(
              concat_matrix_by_columns(self.joint_to_margin),  # pyrefly: ignore[bad-argument-type]
              self.split_dist[r],
              split_dist_max_init[r],
          )
          for r in range(6)
      ]

      # Set lam_sum and lam_margin from the registered variables.
      self.lam_sum = [pm.get_param(self.lam_sum_id[r]) for r in range(6)]  # pyrefly: ignore[unsupported-operation]
      self.lam_margin = [[None] * 3 for _ in range(6)]
      for r in range(6):
        for t in range(3):
          self.lam_margin[r][t] = pm.get_param(self.lam_margin_id[r][t])  # pyrefly: ignore[unsupported-operation]

    self.derivative_upper_bound = [
        self._get_max_derivative(r) for r in range(6)
    ]
    self.term_frac = 0.0

  def evaluate_pre(self):
    assert self.num_split is not None
    assert self.left_ptr is not None
    assert self.right_ptr is not None
    for i in range(self.num_split):
      for r in range(6):
        # split_dist is vector, region_prop is vector.
        # split_dist[r].value[i] -> split_dist[r][i]
        frac = self.term_frac * self.split_dist[r][i] * self.region_prop[r]  # pyrefly: ignore[unsupported-operation]
        left_ptr = self.left_ptr[r][i]
        right_ptr = self.right_ptr[r][i]
        if left_ptr is not None:
          assert hasattr(left_ptr, "term_frac")
          left_ptr.term_frac = left_ptr.term_frac + frac
        if right_ptr is not None:
          assert hasattr(right_ptr, "term_frac")
          right_ptr.term_frac = right_ptr.term_frac + frac

  def evaluate_post(self):
    self.hash_penalty_term = []
    for r in range(6):
      self.hash_penalty_term.append(
          (
              entropy(self.split_dist_max[r])  # pyrefly: ignore[unsupported-operation]
              - entropy(self.split_dist[r])  # pyrefly: ignore[unsupported-operation]
              + 2 * self.derivative_upper_bound[r]  # pyrefly: ignore[unsupported-operation]
          )
          * self.term_frac
          * self.region_prop[r]  # pyrefly: ignore[unsupported-operation]
      )

    self.num_block_contribution = []
    for r in range(6):
      dist_x, dist_y, dist_z = marginal_dist(
          self.split_dist[r], self.joint_to_margin  # pyrefly: ignore[bad-argument-type, unsupported-operation]
      )
      vec = np.array(
          [entropy(dist_x), entropy(dist_y), entropy(dist_z)], dtype=float
      )
      self.num_block_contribution.append(
          self.term_frac * self.region_prop[r] * vec  # pyrefly: ignore[unsupported-operation]
      )

    self.mat_size_contribution = np.zeros(3)

    self.complete_split_region = [[None] * 3 for _ in range(6)]
    for r in range(6):
      for t in range(3):
        g = np.zeros(3**self.power)
        for i in range(self.num_split):  # pyrefly: ignore[bad-argument-type]
          left_ptr = self.left_ptr[r][i]  # pyrefly: ignore[unsupported-operation]
          right_ptr = self.right_ptr[r][i]  # pyrefly: ignore[unsupported-operation]
          if left_ptr is not None and right_ptr is not None:
            assert hasattr(left_ptr, "complete_split")
            assert hasattr(right_ptr, "complete_split")
            g = g + self.split_dist[r][i] * concat_csd(  # pyrefly: ignore[unsupported-operation]
                left_ptr.complete_split[t],
                right_ptr.complete_split[t],
            )
        self.complete_split_region[r][t] = g
    self.complete_split = []
    for t in range(3):
      combined = (
          self.complete_split_region[0][t] * self.region_prop[0]  # pyrefly: ignore[unsupported-operation]
          + self.complete_split_region[1][t] * self.region_prop[1]  # pyrefly: ignore[unsupported-operation]
          + self.complete_split_region[2][t] * self.region_prop[2]  # pyrefly: ignore[unsupported-operation]
          + self.complete_split_region[3][t] * self.region_prop[3]  # pyrefly: ignore[unsupported-operation]
          + self.complete_split_region[4][t] * self.region_prop[4]  # pyrefly: ignore[unsupported-operation]
          + self.complete_split_region[5][t] * self.region_prop[5]  # pyrefly: ignore[unsupported-operation]
      )
      self.complete_split.append(combined)

    self.p_compY = []
    self.p_compZ = []
    for r in range(6):
      self.p_compY.append(
          (self._get_pcomp_y_numerator(r) - self._get_pcomp_y_denominator(r))
          * self.term_frac
          * self.region_prop[r]  # pyrefly: ignore[unsupported-operation]
      )
    for r in range(6):
      self.p_compZ.append(
          (self._get_pcomp_z_numerator(r) - self._get_pcomp_z_denominator(r))
          * self.term_frac
          * self.region_prop[r]  # pyrefly: ignore[unsupported-operation]
      )

  def _get_pcomp_y_numerator(self, region: int):
    assert self.power is not None
    assert self.sum_half is not None
    assert self.num_split is not None
    assert self.left_ptr is not None
    assert self.right_ptr is not None
    assert self.split_dist is not None
    _, dim_y, dim_z = dims(region + 1)
    res = 0.0
    weighted_sum = [
        np.zeros(3 ** (self.power // 2)) for _ in range(self.sum_half + 1)
    ]
    prob_sum = [0.0 for _ in range(self.sum_half + 1)]
    used = [False] * (self.sum_half + 1)

    for i in range(self.num_split):
      for ptr in (self.left_ptr[region][i], self.right_ptr[region][i]):
        if ptr is None:
          continue
        assert hasattr(ptr, "shape")
        assert hasattr(ptr, "complete_split")
        if ptr.shape[dim_y - 1] == 0:
          continue
        if ptr.shape[dim_z - 1] == 0:
          res = res + self.split_dist[region][i] * entropy(
              ptr.complete_split[dim_y - 1]
          )
        else:
          idx = ptr.shape[dim_y - 1]
          weighted_sum[idx] = (
              weighted_sum[idx]
              + self.split_dist[region][i] * ptr.complete_split[dim_y - 1]
          )
          prob_sum[idx] = prob_sum[idx] + self.split_dist[region][i]
          used[idx] = True

    for i in range(1, self.sum_half + 1):
      if not used[i]:
        continue
      res = res + normalized_entropy(weighted_sum[i], prob_sum[i])
    return res

  def _get_pcomp_y_denominator(self, region: int):
    assert self.complete_split_region is not None
    assert self.split_dist is not None
    assert self.joint_to_margin is not None
    _, dim_y, _ = dims(region + 1)
    res = entropy(self.complete_split_region[region][dim_y - 1])
    dist_x, dist_y, dist_z = marginal_dist(
        self.split_dist[region], self.joint_to_margin
    )
    dist = [dist_x, dist_y, dist_z][dim_y - 1]
    res = res - entropy(dist)
    return res

  def _get_pcomp_z_numerator(self, region: int):
    assert self.power is not None
    assert self.sum_half is not None
    assert self.num_split is not None
    assert self.left_ptr is not None
    assert self.right_ptr is not None
    assert self.split_dist is not None
    _, _, dim_z = dims(region + 1)
    res = 0.0
    weighted_sum = [
        np.zeros(3 ** (self.power // 2)) for _ in range(self.sum_half + 1)
    ]
    prob_sum = [0.0 for _ in range(self.sum_half + 1)]
    used = [False] * (self.sum_half + 1)

    for i in range(self.num_split):
      for ptr in (self.left_ptr[region][i], self.right_ptr[region][i]):
        if ptr is None:
          continue
        assert hasattr(ptr, "shape")
        assert hasattr(ptr, "complete_split")
        if ptr.shape[dim_z - 1] == 0:
          continue
        if np.min(ptr.shape) == 0:
          res = res + self.split_dist[region][i] * entropy(
              ptr.complete_split[dim_z - 1]
          )
        else:
          idx = ptr.shape[dim_z - 1]
          weighted_sum[idx] = (
              weighted_sum[idx]
              + self.split_dist[region][i] * ptr.complete_split[dim_z - 1]
          )
          prob_sum[idx] = prob_sum[idx] + self.split_dist[region][i]
          used[idx] = True

    for i in range(1, self.sum_half + 1):
      if not used[i]:
        continue
      res = res + normalized_entropy(weighted_sum[i], prob_sum[i])
    return res

  def _get_pcomp_z_denominator(self, region: int):
    assert self.complete_split_region is not None
    assert self.split_dist is not None
    assert self.joint_to_margin is not None
    _, _, dim_z = dims(region + 1)
    res = entropy(self.complete_split_region[region][dim_z - 1])
    dist_x, dist_y, dist_z = marginal_dist(
        self.split_dist[region], self.joint_to_margin
    )
    dist = [dist_x, dist_y, dist_z][dim_z - 1]
    res = res - entropy(dist)
    return res

  def _get_max_derivative(self, r):
    assert self.num_split is not None
    assert self.splits is not None
    assert self.lam_margin is not None
    assert self.lam_margin_low is not None
    assert self.lam_sum is not None
    assert self.split_dist_max is not None
    max_derivative = 0.0
    for t in range(self.num_split):
      left_shape = self.splits[t, :3]
      derivative_term = (
          self.lam_margin[r][0][left_shape[0] - self.lam_margin_low[0]].item()
          + self.lam_margin[r][1][left_shape[1] - self.lam_margin_low[1]].item()
          + self.lam_margin[r][2][left_shape[2] - self.lam_margin_low[2]].item()
          + self.lam_sum[r][0].item()
          - 1
          - math.log(self.split_dist_max[r][t])
      )
      max_derivative = max(max_derivative, abs(derivative_term))
    return max_derivative

  @staticmethod
  def create_instance(level: int, shape: Sequence[int]):
    if level == 2:
      return TermInfoLv2
    if min(shape) == 0:
      return TermInfoZero
    return TermInfo


class GlobalStage:

  def __init__(self, context: EvalContext):
    self.ctx = context
    self.level = None
    self.power = None
    self.sum_col = None
    self.shapes = None
    self.num_shape = None
    self.joint_to_margin = None
    self.term_id = None
    self.term_ptr = None
    self.region_prop_id = None
    self.region_prop = None
    self.dist_id = None
    self.dist = None
    self.dist_max_id = None
    self.dist_max = None
    self.lam_margin_id = None
    self.lam_sum_id = None
    self.lam_margin = None
    self.lam_sum = None
    self.derivative_upper_bound_id = None
    self.derivative_upper_bound = None
    self.mat_size = None
    self.num_block = None
    self.hash_penalty_term = None
    self.p_compY = None
    self.p_compZ = None
    self.complete_split = None

  def build(self):
    ctx = self.ctx
    self.level = ctx.max_level
    self.power = 2 ** (self.level - 1)
    self.sum_col = 2**self.level
    self.shapes = prepare_shapes(self.level)
    self.num_shape = self.shapes.shape[0]
    self.joint_to_margin = joint_to_margin(self.shapes, self.sum_col)
    self._register_variables()

    self.term_id = [[] for _ in range(6)]
    self.term_ptr = [[] for _ in range(6)]
    for r in range(6):
      ids = []
      ptrs = []
      for i in range(self.num_shape):
        cur_id = len(ctx.terms[self.level])
        term_cls = TermInfo.create_instance(self.level, self.shapes[i])
        term = term_cls(ctx)
        ctx.terms[self.level].append(term)
        ctx.term_index[self.level][
            _term_key(self.shapes[i], np.array([0, r + 1]))
        ] = cur_id
        term.build(self.level, cur_id + 1, self.shapes[i], np.array([0, r + 1]))
        ids.append(cur_id + 1)
        ptrs.append(term)
      self.term_id[r] = ids
      self.term_ptr[r] = ptrs

  def _register_variables(self):
    assert self.num_shape is not None
    assert self.sum_col is not None
    assert self.joint_to_margin is not None
    pm = self.ctx.param_manager
    self.region_prop_id = pm.register(6, 0, 1, (0, 1))
    self.dist_id = [None] * 6
    self.dist_max_id = [None] * 6
    self.lam_margin_id = [[None] * 3 for _ in range(6)]
    self.lam_sum_id = [None] * 6
    self.derivative_upper_bound_id = [None] * 6
    for r in range(6):
      self.dist_id[r] = pm.register(
          self.num_shape, 0, 1, (0, 1 / max(self.num_shape, 1))
      )
      self.dist_max_id[r] = pm.register(
          self.num_shape, 0, 1, (0, 1 / max(self.num_shape, 1))
      )
    for r in range(6):
      for t in range(3):
        self.lam_margin_id[r][t] = pm.register(
            self.sum_col + 1, -math.inf, math.inf, (-0.01, 0.01)
        )
      self.lam_sum_id[r] = pm.register(1, -math.inf, math.inf, (-0.01, 0.01))
      self.derivative_upper_bound_id[r] = pm.register(1, 0, math.inf, (0, 1))

  def evaluate_init(self):
    assert self.region_prop_id is not None
    assert self.dist_id is not None
    assert self.dist_max_id is not None
    assert self.lam_sum_id is not None
    assert self.lam_margin_id is not None
    pm = self.ctx.param_manager
    self.region_prop = normalize_distribution(pm.get_param(self.region_prop_id))
    self.dist = [
        normalize_distribution(pm.get_param(gid)) for gid in self.dist_id
    ]
    dist_max_init = [
        normalize_distribution(pm.get_param(gid)) for gid in self.dist_max_id
    ]
    if self.ctx.use_newton:
      self.dist_max = []
      self.lam_sum = [None] * 6
      self.lam_margin = [[None] * 3 for _ in range(6)]
      margin_len = self.joint_to_margin[0].shape[1]  # pyrefly: ignore[unsupported-operation]
      for r in range(6):
        # Compute dist_max, lam_margin, and lam_sum using Newton's method.
        dist_max_r, lam_margin_r, lam_sum_r = solve_max_entropy_newton(
            concat_matrix_by_columns(self.joint_to_margin),  # pyrefly: ignore[bad-argument-type]
            self.dist[r],
            dist_max_init=dist_max_init[r],
        )
        self.dist_max.append(dist_max_r)
        self.lam_sum[r] = np.array([lam_sum_r])

        # lam_margin_r is shape (3 * margin_len,). Split it into three vectors
        # of length margin_len.
        self.lam_margin[r][0] = lam_margin_r[:margin_len]
        self.lam_margin[r][1] = lam_margin_r[margin_len : 2 * margin_len]
        self.lam_margin[r][2] = lam_margin_r[2 * margin_len :]

    else:
      # Compute dist_max by projecting dist onto the linear space.
      self.dist_max = [
          project_distribution_to_linear_space(
              concat_matrix_by_columns(self.joint_to_margin),  # pyrefly: ignore[bad-argument-type]
              self.dist[r],
              dist_max_init[r],
          )
          for r in range(6)
      ]
      # Set lam_sum and lam_margin to the values of the registered variables.
      self.lam_sum = [pm.get_param(gid) for gid in self.lam_sum_id]
      self.lam_margin = [[None] * 3 for _ in range(6)]
      for r in range(6):
        for t in range(3):
          self.lam_margin[r][t] = pm.get_param(self.lam_margin_id[r][t])

    self.derivative_upper_bound = [
        self._get_max_derivative(r) for r in range(6)
    ]

  def evaluate_pre(self):
    for r in range(6):
      for i in range(self.num_shape):  # pyrefly: ignore[bad-argument-type]
        self.term_ptr[r][i].term_frac = self.dist[r][i] * self.region_prop[r]  # pyrefly: ignore[unsupported-operation]

  def evaluate_post(self):
    self.hash_penalty_term = []
    for r in range(6):
      self.hash_penalty_term.append(
          (
              entropy(self.dist_max[r])  # pyrefly: ignore[unsupported-operation]
              - entropy(self.dist[r])  # pyrefly: ignore[unsupported-operation]
              + 2 * self.derivative_upper_bound[r]  # pyrefly: ignore[unsupported-operation]
          )
          * self.region_prop[r]  # pyrefly: ignore[unsupported-operation]
      )
    self.num_block = []
    for r in range(6):
      dist_x, dist_y, dist_z = marginal_dist(self.dist[r], self.joint_to_margin)  # pyrefly: ignore[bad-argument-type, unsupported-operation]
      self.num_block.append(
          np.array([
              entropy(dist_x),
              entropy(dist_y),
              entropy(dist_z),
          ])
          * self.region_prop[r]  # pyrefly: ignore[unsupported-operation]
      )
    self.mat_size = np.zeros(3)
    for l in range(2, self.level + 1):  # pyrefly: ignore[unsupported-operation]
      for term in self.ctx.terms[l]:
        assert hasattr(term, "mat_size_contribution")  # For pytype.
        self.mat_size = self.mat_size + term.mat_size_contribution

    self.complete_split = [[None] * 3 for _ in range(6)]
    for r in range(6):
      for t in range(3):
        csd = np.zeros(3**self.power)  # pyrefly: ignore[unsupported-operation]
        for i in range(self.num_shape):  # pyrefly: ignore[bad-argument-type]
          ptr = self.term_ptr[r][i]  # pyrefly: ignore[unsupported-operation]
          csd = csd + self.dist[r][i] * ptr.complete_split[t]  # pyrefly: ignore[unsupported-operation]
        self.complete_split[r][t] = csd

    self.p_compY = []
    self.p_compZ = []
    for r in range(6):
      self.p_compY.append(
          (self._pcomp_y_numerator(r) - self._pcomp_y_denominator(r))
          * self.region_prop[r]  # pyrefly: ignore[unsupported-operation]
      )
      self.p_compZ.append(
          (self._pcomp_z_numerator(r) - self._pcomp_z_denominator(r))
          * self.region_prop[r]  # pyrefly: ignore[unsupported-operation]
      )

  def _pcomp_z_numerator(self, region: int):
    assert self.power is not None
    assert self.sum_col is not None
    assert self.num_shape is not None
    assert self.term_ptr is not None
    assert self.dist is not None
    _, _, dim_z = dims(region + 1)
    res = 0.0
    weighted = [np.zeros(3**self.power) for _ in range(self.sum_col + 1)]
    prob_sum = [0.0 for _ in range(self.sum_col + 1)]
    used = [False] * (self.sum_col + 1)

    for i in range(self.num_shape):
      ptr = self.term_ptr[region][i]
      if ptr.shape[dim_z - 1] == 0:
        continue
      if np.min(ptr.shape) == 0:
        res = res + self.dist[region][i] * entropy(
            ptr.complete_split[dim_z - 1]
        )
      else:
        idx = ptr.shape[dim_z - 1]
        weighted[idx] = (
            weighted[idx] + self.dist[region][i] * ptr.complete_split[dim_z - 1]
        )
        prob_sum[idx] = prob_sum[idx] + self.dist[region][i]
        used[idx] = True
    for i in range(1, self.sum_col + 1):
      if not used[i]:
        continue
      res = res + normalized_entropy(weighted[i], prob_sum[i])
    return res

  def _pcomp_z_denominator(self, region: int):
    assert self.dist is not None
    assert self.joint_to_margin is not None
    _, _, dim_z = dims(region + 1)
    res = entropy(self.complete_split[region][dim_z - 1])  # pyrefly: ignore[unsupported-operation]
    dist_x, dist_y, dist_z = marginal_dist(
        self.dist[region], self.joint_to_margin
    )
    res = res - entropy([dist_x, dist_y, dist_z][dim_z - 1])
    return res

  def _pcomp_y_numerator(self, region: int):
    assert self.power is not None
    assert self.sum_col is not None
    assert self.num_shape is not None
    assert self.term_ptr is not None
    assert self.dist is not None
    _, dim_y, dim_z = dims(region + 1)
    res = 0.0
    weighted = [np.zeros(3**self.power) for _ in range(self.sum_col + 1)]
    prob_sum = [0.0 for _ in range(self.sum_col + 1)]
    used = [False] * (self.sum_col + 1)
    for i in range(self.num_shape):
      ptr = self.term_ptr[region][i]
      if ptr.shape[dim_y - 1] == 0:
        continue
      if ptr.shape[dim_z - 1] == 0:
        res = res + self.dist[region][i] * entropy(
            ptr.complete_split[dim_y - 1]
        )
      else:
        idx = ptr.shape[dim_y - 1]
        weighted[idx] = (
            weighted[idx] + self.dist[region][i] * ptr.complete_split[dim_y - 1]
        )
        prob_sum[idx] = prob_sum[idx] + self.dist[region][i]
        used[idx] = True
    for i in range(1, self.sum_col + 1):
      if not used[i]:
        continue
      res = res + normalized_entropy(weighted[i], prob_sum[i])
    return res

  def _pcomp_y_denominator(self, region: int):
    assert self.dist is not None
    assert self.joint_to_margin is not None
    _, dim_y, _ = dims(region + 1)
    res = entropy(self.complete_split[region][dim_y - 1])  # pyrefly: ignore[unsupported-operation]
    dist_x, dist_y, dist_z = marginal_dist(
        self.dist[region], self.joint_to_margin
    )
    res = res - entropy([dist_x, dist_y, dist_z][dim_y - 1])
    return res

  def _get_max_derivative(self, r):
    assert self.num_shape is not None
    assert self.shapes is not None
    assert self.lam_margin is not None
    assert self.lam_sum is not None
    assert self.dist_max is not None
    max_derivative = 0.0
    for t in range(self.num_shape):
      cur_shape = self.shapes[t]
      derivative_term = (
          self.lam_margin[r][0][cur_shape[0]].item()
          + self.lam_margin[r][1][cur_shape[1]].item()
          + self.lam_margin[r][2][cur_shape[2]].item()
          + self.lam_sum[r][0].item()
          - 1
          - math.log(self.dist_max[r][t])
      )
      max_derivative = max(max_derivative, abs(derivative_term))
    return max_derivative


class Workspace:

  def __init__(self, context: EvalContext):
    self.ctx = context
    self.num_retain_comp_id = None
    self.num_retain_comp = None
    self.num_retain_comp_lasteval = None
    self.num_retain_glob_id = None
    self.num_retain_glob = None
    self.num_retain_glob_lasteval = None
    self.single_mat_size_id = None
    self.single_mat_size = None
    self.single_mat_size_lasteval = None
    self.value = None
    self.omega_id = None
    self.omega = None
    self.K_id = None
    self.K = None
    self.num_constr_c = 0
    self.num_constr_ceq = 0
    self.globstage = None

  def build(self, max_level: int):
    ctx = self.ctx
    pm = ctx.param_manager
    ctx.max_level = max_level
    self.num_retain_comp_id = [[None] * (max_level + 1) for _ in range(6)]
    self.num_retain_glob_id = [None] * 6
    for r in range(6):
      for lv in range(2, max_level + 1):
        self.num_retain_comp_id[r][lv] = pm.register(1, 0, math.inf, (0, 0.01))
    for r in range(6):
      self.num_retain_glob_id[r] = pm.register(1, 0, math.inf, (0, 0.01))
    self.single_mat_size_id = pm.register(1, 0, math.inf, (0, 0.01))

    self.omega_id = pm.register(1, 0, math.inf)
    self.K_id = pm.register(1, ctx.K, ctx.K)

    ctx.terms = [[] for _ in range(max_level + 1)]
    ctx.term_index = [{} for _ in range(max_level + 1)]
    self.globstage = GlobalStage(ctx)
    self.globstage.build()

  def evaluate(self):
    ctx = self.ctx
    pm = ctx.param_manager
    max_level = ctx.max_level
    self.num_retain_comp = [[None] * (max_level + 1) for _ in range(6)]
    self.num_retain_comp_lasteval = [[None] * (max_level + 1) for _ in range(6)]
    self.num_retain_glob = [None] * 6
    self.num_retain_glob_lasteval = [None] * 6
    assert self.num_retain_comp_id is not None
    assert self.num_retain_glob_id is not None
    for r in range(6):
      for l in range(2, max_level + 1):
        self.num_retain_comp[r][l] = pm.get_param(self.num_retain_comp_id[r][l])
    for r in range(6):
      self.num_retain_glob[r] = pm.get_param(self.num_retain_glob_id[r])
    assert self.single_mat_size_id is not None
    assert self.omega_id is not None
    assert self.K_id is not None
    self.single_mat_size = pm.get_param(self.single_mat_size_id)
    self.omega = pm.get_param(self.omega_id)
    self.K = pm.get_param(self.K_id)

    assert self.globstage is not None  # For pytype.
    self.globstage.evaluate_init()
    for l in range(1, max_level + 1):
      for term in ctx.terms[l]:
        assert hasattr(term, "evaluate_init")  # For pytype.
        term.evaluate_init()
    self.globstage.evaluate_pre()
    for l in range(max_level, 0, -1):
      for term in ctx.terms[l]:
        assert hasattr(term, "evaluate_pre")  # For pytype.
        term.evaluate_pre()
    for l in range(1, max_level + 1):
      for term in ctx.terms[l]:
        assert hasattr(term, "evaluate_post")  # For pytype.
        term.evaluate_post()
    self.globstage.evaluate_post()

    for l in range(3, max_level + 1):
      for r in range(6):
        dim_x, dim_y, _ = dims(r + 1)
        num_block = np.zeros(3)
        hash_penalty = 0.0
        p_compY = 0.0
        p_compZ = 0.0
        for term in ctx.terms[l]:
          assert hasattr(term, "num_block_contribution")  # For pytype.
          assert hasattr(term, "hash_penalty_term")  # For pytype.
          assert hasattr(term, "p_compY")  # For pytype.
          assert hasattr(term, "p_compZ")  # For pytype.
          num_block = num_block + term.num_block_contribution[r]
          hash_penalty = hash_penalty + term.hash_penalty_term[r]
          p_compY = p_compY + term.p_compY[r]
          p_compZ = p_compZ + term.p_compZ[r]
        min_val = float("inf")
        for t in range(3):
          if t + 1 == dim_x:
            foo_val = num_block[t] - hash_penalty
          elif t + 1 == dim_y:
            foo_val = num_block[t] - p_compY
          else:
            foo_val = num_block[t] - p_compZ
          min_val = min(min_val, foo_val)
        self.num_retain_comp[r][l] = min_val

    num_block = np.zeros(3)
    for term in ctx.terms[2]:
      assert hasattr(term, "num_block_contribution")  # For pytype.
      num_block = num_block + term.num_block_contribution
    self.num_retain_comp[0][2] = np.min(num_block)
    for r in range(1, 6):
      self.num_retain_comp[r][2] = 0.0

    for r in range(6):
      dim_x, dim_y, _ = dims(r + 1)
      num_block = self.globstage.num_block[r]
      hash_penalty = self.globstage.hash_penalty_term[r]
      p_compY = self.globstage.p_compY[r]
      p_compZ = self.globstage.p_compZ[r]
      min_val = float("inf")
      for t in range(3):
        if t + 1 == dim_x:
          foo_val = num_block[t] - hash_penalty
        elif t + 1 == dim_y:
          foo_val = num_block[t] - p_compY
        else:
          foo_val = num_block[t] - p_compZ
        min_val = min(min_val, foo_val)
      self.num_retain_glob[r] = min_val

    mat_size = self.globstage.mat_size
    self.single_mat_size = min(
        mat_size[0],
        mat_size[1],
        mat_size[2] / self.K,
    )

    self.num_mat = 0.0
    for r in range(6):
      self.num_mat = self.num_mat + self.num_retain_glob[r]
    for l in range(2, max_level + 1):
      for r in range(6):
        self.num_mat = self.num_mat + self.num_retain_comp[r][l]
    # We should have
    # self.num_mat + self.single_mat_size * omega >=
    # math.log(ctx.q + 2) * (2 ** (max_level - 1)).
    self.omega = (
        math.log(ctx.q + 2) * (2 ** (max_level - 1)) - self.num_mat
    ) / self.single_mat_size

  def get_omega_pos(self):
    return self.ctx.param_manager.group_startpos[self.omega_id]  # pyrefly: ignore[bad-index]

  def get_k_pos(self):
    return self.ctx.param_manager.group_startpos[self.K_id]  # pyrefly: ignore[bad-index]


def get_feasibility(workspace: Workspace) -> float:
  workspace.evaluate()
  omega_verified = workspace.omega
  assert omega_verified is not None
  omega_verified = omega_verified.item()

  return omega_verified


def verify_omega(
    q: float,
    K: float,
    max_level: int,
    x_vector: np.ndarray,
    use_term_sharing: bool = False,
    use_newton: bool = False,
) -> float:
  """Verifies a candidate solution for the matrix multiplication exponent omega.

  Given alphabet size `q`, parameter `K`, recursion depth `max_level`, and a
  flat parameter vector `x_vector`, this function constructs the full
  verification workspace (a hierarchy of `GlobalStage` and `TermInfo` objects
  up to `max_level`), loads the parameters, evaluates all stages bottom-up, and
  computes the verified omega exponent.

  The parameter vector encodes all decision variables: distributions over
  shapes, split distributions, region proportions, Lagrange multipliers,
  and auxiliary bounds.  If the vector is invalid (wrong size, contains
  NaN/Inf, or cannot be converted to a float array), the function returns
  ``float('inf')`` to indicate infeasibility.

  Args:
    q: The alphabet size (number of symbols per dimension).
    K: The asymmetry parameter for the third matrix dimension.
    max_level: The recursion depth of the construction.  Level 2 uses
      closed-form terms; higher levels recurse into sub-terms.
    x_vector: A 1-D numpy array (or array-like) of decision variables. Its
      expected length is determined by the workspace structure built from ``(q,
      K, max_level, use_term_sharing)``.
    use_term_sharing: If True, sub-terms at the same level with the same shape
      share parameters. This reduces the total number of decision variables.
    use_newton: If True, compute the maximum-entropy distribution and its
      Lagrange multipliers via ``solve_max_entropy_newton`` instead of reading
      them from the parameter vector and projecting with
      ``project_distribution_to_linear_space``.

  Returns:
    The verified omega exponent. A finite value indicates a feasible
    construction proving ``omega(q, K) <= returned_value``.  Returns
    ``float('inf')`` if the input is invalid.
  """
  # 1. Validate input x_vector type and content (nan/inf)
  try:
    x_vec = np.asarray(x_vector, dtype=float).reshape(-1)
    if not np.all(np.isfinite(x_vec)):
      return float("inf")
  except (TypeError, ValueError):
    # Catch errors during array conversion.
    return float("inf")

  ctx = create_context(
      q=q,
      max_level=max_level,
      K=K,
      use_term_sharing=use_term_sharing,
      use_newton=use_newton,
  )
  workspace = Workspace(ctx)
  workspace.build(max_level=max_level)
  pm = ctx.param_manager

  # 2. Validate dimension
  if x_vec.size != pm.num_input:
    return float("inf")

  pm.set_value(x_vec)  # pyrefly: ignore[bad-argument-type]

  omega_val = get_feasibility(workspace)

  return omega_val
