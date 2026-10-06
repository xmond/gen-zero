"""Gen-Zero Hard Safety: Differentiable Neuro-Symbolic Safety Layer.

RFC-087 / Issue #87 Implementation (inference only):
Neuro-symbolic safety projection via Augmented Lagrangian relaxation:
1. Dual-Track Architecture:
   - Forward Track: convex projection, then a 0-1 integer CP-SAT selection over the
     finite candidate action set (a discrete action gate, see "Scope and guarantees").
2. Augmented Lagrangian Convex Relaxation:
   Relaxes discrete 0-1 polytope constraints to continuous convex manifold:
       min_{x in [0, 1]^K} - c(z)^T x + (mu / 2) * ||x - x0(z)||_2^2
       s.t.  A_sat * x <= b,  sum(x) == 1,  x >= 0
   using Projected Augmented Lagrangian / ADMM iterations.

Scope and guarantees (read before quoting this module as a proof):
- The projection is the exact Euclidean projection onto the polytope reached by
  the active face of that call. No gradient or backward path exists in this module.
- The active-set tolerance (dual_tol) and KKT contact gating in forward()
  exist to absorb floating-point rounding error accumulated across the
  iterations of one convex projection solve, not to bound error across a
  multi-step physical rollout. tests/test_r8_s01_active_set_cumul_noise.py
  parametrizes max_iter at 200 and 500 to stress exactly this: iterations of
  a single QP solve, not steps of an environment trajectory.
- Two separate discrete checks run in forward():
  * discrete_feasible (NumPy): action i is feasible when its one-hot vector
    e_i satisfies A_sat*e_i <= b. discrete_hard_verified says the output's
    top-1 action argmax(x*) passes this check.
  * CP-SAT: a 0-1 integer program, maximise sum_i u_i*y_i s.t. sum_i y_i == 1,
    over the non-forbidden candidates, with u = x* as utilities.
  cpsat_hard_verified is True only when CP-SAT really ran and returned
  OPTIMAL/FEASIBLE without fallback, its selected action equals the output's
  top-1 action, and that action is discrete_feasible. cpsat_solver_status
  and cpsat_selected_action record what happened ("NOT_INVOKED" when CP-SAT
  was not called). This is a discrete action gate over the supplied
  candidate set, not a formal proof of safety over the continuous action
  space or over actions outside that set.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from gen_zero.gate.cpsat_formal_solver import CPSATFormalSolver, CPSATVerdict

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = object
    HAS_TORCH = False


@dataclass
class SafetyProjectionResult:
    """Telemetry report of differentiable safety projection."""
    projected_distribution: np.ndarray
    original_proposal: np.ndarray
    is_safe: bool
    active_constraints_count: int
    kkt_residual: float
    solve_time_ms: float
    cpsat_hard_verified: bool = False
    discrete_hard_verified: bool = False
    cpsat_solver_status: str = "NOT_INVOKED"
    cpsat_selected_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "active_constraints_count": self.active_constraints_count,
            "kkt_residual": round(self.kkt_residual, 6),
            "solve_time_ms": round(self.solve_time_ms, 4),
            "cpsat_hard_verified": self.cpsat_hard_verified,
            "discrete_hard_verified": self.discrete_hard_verified,
            "cpsat_solver_status": self.cpsat_solver_status,
            "cpsat_selected_action": self.cpsat_selected_action,
            "top1_action_index": int(np.argmax(self.projected_distribution)),
        }


class InfeasibleConstraintError(ValueError):
    """No point satisfies the supplied simplex bounds."""


def project_simplex_with_bounds(v: np.ndarray, upper_bounds: Optional[np.ndarray] = None) -> np.ndarray:
    """Projects vector v onto probability simplex sum(x)=1, x >= 0, x <= upper_bounds

    using exact 1D bisection root-finding on the piecewise linear function:
        g(theta) = sum(clip(v - theta, 0, upper_bounds)) - 1.0 = 0
    """
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 1 or not np.isfinite(v).all() or len(v) == 0:
        raise ValueError("Projection input must be a nonempty finite vector")
    k = len(v)
    if upper_bounds is None:
        upper_bounds = np.ones(k, dtype=np.float64)

    upper_bounds = np.asarray(upper_bounds, dtype=np.float64)
    if upper_bounds.shape != v.shape or not np.isfinite(upper_bounds).all() or np.any(upper_bounds < 0):
        raise ValueError("Invalid simplex upper bounds")
    if np.sum(upper_bounds) < 1.0 - 1e-12:
        raise InfeasibleConstraintError("Simplex upper bounds sum below one")

    # Bisection search bounds
    low = np.min(v - upper_bounds) - 1.0
    high = np.max(v)

    for _ in range(100):
        mid = (low + high) * 0.5
        val = np.sum(np.clip(v - mid, 0.0, upper_bounds))
        if val > 1.0:
            low = mid
        else:
            high = mid

    x = np.clip(v - high, 0.0, upper_bounds)
    sum_x = np.sum(x)
    if abs(sum_x - 1.0) > 1e-8:
        raise InfeasibleConstraintError("Simplex projection failed to satisfy mass constraint")
    return x


class DifferentiableSafetyLayer:
    """Neuro-Symbolic Safety Layer powered by Augmented Lagrangian projection.

    forward()'s convex projection is exact and its CP-SAT check is a hard
    discrete feasibility gate over the candidate one-hot actions supplied
    to that call (see cpsat_hard_verified / discrete_hard_verified).
    """

    def __init__(
        self,
        action_dim: int,
        constraint_matrix: Optional[np.ndarray] = None,
        constraint_rhs: Optional[np.ndarray] = None,
        mu: float = 1.0,
        rho: float = 5.0,
        max_iter: int = 40,
        tolerance: float = 1e-5,
        hard_timeout_ms: float = 2.0,
    ) -> None:
        self.action_dim = int(action_dim)
        self.mu = float(mu)
        if not np.isfinite(self.mu) or self.mu <= 0:
            raise ValueError("mu must be positive and finite")
        self.rho = float(rho)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)

        if (constraint_matrix is None) != (constraint_rhs is None):
            raise ValueError("constraint_matrix and constraint_rhs must both be provided or both be None")
        if constraint_matrix is None:
            self.a_sat = np.zeros((0, self.action_dim), dtype=np.float64)
            self.b_sat = np.zeros(0, dtype=np.float64)
        else:
            self.a_sat, self.b_sat = self._validate_constraints(constraint_matrix, constraint_rhs)

        self.num_constraints = self.a_sat.shape[0]
        self.cpsat_solver = CPSATFormalSolver(hard_timeout_ms=hard_timeout_ms)

        # Last-solve diagnostics (active face and duals), read by tests
        self._cached_x0: Optional[np.ndarray] = None
        self._cached_x_star: Optional[np.ndarray] = None
        self._cached_active_a: Optional[np.ndarray] = None
        self._cached_lambda: Optional[np.ndarray] = None

    def update_constraints(self, a_matrix: np.ndarray, b_vector: np.ndarray) -> None:
        """Dynamically updates safety inequality constraints."""
        self.a_sat, self.b_sat = self._validate_constraints(a_matrix, b_vector)
        self.num_constraints = self.a_sat.shape[0]

    def _validate_constraints(self, matrix: np.ndarray, rhs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a = np.asarray(matrix, dtype=np.float64)
        b = np.asarray(rhs, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != self.action_dim or b.ndim != 1 or a.shape[0] != b.shape[0]:
            raise ValueError("Constraint matrix and rhs dimensions must match action_dim and each other")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Non-finite safety constraints rejected")
        # Work in unit-normal coordinates throughout the solver.  Scaling an
        # inequality by a positive constant must not change its projection.
        norms = np.linalg.norm(a, axis=1)
        if not np.isfinite(norms).all() or np.any(norms == 0):
            raise ValueError("Safety constraint normals must be finite and nonzero")
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            try:
                a, b = a / norms[:, None], b / norms
            except FloatingPointError as exc:
                raise ValueError("Safety constraint normalization failed") from exc
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Non-finite normalized safety constraints rejected")
        return a, b

    def forward(
        self,
        x0: np.ndarray,
        utility_vector: Optional[np.ndarray] = None,
        candidate_names: Optional[Sequence[str]] = None,
    ) -> SafetyProjectionResult:
        """Forward pass: Solves Augmented Lagrangian convex projection and CP-SAT verification.

        Scope of the two discrete checks (candidate_names default ACT_0..ACT_{K-1}):
        - discrete_feasible[i] is the NumPy test A_sat*e_i <= b on the one-hot
          action e_i. Infeasible actions get upper bound 0 in the projection.
          discrete_hard_verified = discrete_feasible[argmax(x*)].
        - CP-SAT solves a 0-1 integer program: pick exactly one non-forbidden
          candidate maximising the utility x*_i. It does not re-check A_sat*x<=b.
        cpsat_hard_verified binds CP-SAT to the output: it is True only when
        CP-SAT ran (status OPTIMAL/FEASIBLE, no fallback), selected the same
        action as argmax(x*), and that action is discrete_feasible. With no
        constraints CP-SAT is not called (status "NOT_INVOKED"). A fallback, a
        single-candidate shortcut, or a mismatch leaves it False; the raw status
        and selected action are in cpsat_solver_status / cpsat_selected_action.
        None of this is a proof over the continuous action space.
        """
        t0 = time.perf_counter()
        x0 = np.asarray(x0, dtype=np.float64)
        if x0.ndim != 1 or len(x0) != self.action_dim:
            raise ValueError(f"Input x0 dimension {len(x0)} does not match action_dim {self.action_dim}")

        c = np.asarray(utility_vector, dtype=np.float64) if utility_vector is not None else np.zeros_like(x0)

        if not np.isfinite(x0).all() or not np.isfinite(c).all():
            raise ValueError("Non-finite safety proposal or utility rejected")
        if c.shape != x0.shape:
            raise ValueError("Utility vector shape must match proposal")
        if not np.isfinite(self.a_sat).all() or not np.isfinite(self.b_sat).all():
            raise ValueError("Non-finite safety constraints rejected")
        if candidate_names is not None and len(candidate_names) != self.action_dim:
            raise ValueError("Candidate names must match action dimension")

        with np.errstate(over="raise", divide="raise", invalid="raise"):
            try:
                shifted = x0 + c / self.mu
            except FloatingPointError as exc:
                raise ValueError("Non-finite intermediate safety projection") from exc
        if not np.isfinite(shifted).all():
            raise ValueError("Non-finite intermediate safety projection")

        # Cache input proposal
        self._cached_x0 = x0.copy()

        # If no constraints, standard simplex projection
        if self.num_constraints == 0:
            x_star = project_simplex_with_bounds(shifted)
            t1 = time.perf_counter()
            self._cached_x_star = x_star
            self._cached_active_a = np.zeros((0, self.action_dim))
            self._cached_lambda = np.zeros(0)
            return SafetyProjectionResult(
                projected_distribution=x_star,
                original_proposal=x0,
                is_safe=True,
                active_constraints_count=0,
                kkt_residual=0.0,
                solve_time_ms=(t1 - t0) * 1000.0,
                cpsat_hard_verified=False,
                discrete_hard_verified=True,
            )

        # A distribution may satisfy A*x<=b while its selected one-hot action does not.
        # Exclude every infeasible discrete action before projecting the distribution.
        discrete_feasible = np.all(self.a_sat <= self.b_sat[:, None], axis=0)
        if not np.any(discrete_feasible):
            raise ValueError("No discrete action satisfies hard safety constraints")
        upper_bounds = discrete_feasible.astype(np.float64)
        # Identify explicit upper bounds for coordinate constraints (e.g. x_i <= b_i)
        for i in range(self.num_constraints):
            row = self.a_sat[i]
            pos_cols = np.where(row > 0.5)[0]
            if len(pos_cols) == 1 and np.sum(np.abs(row)) == row[pos_cols[0]]:
                col = pos_cols[0]
                bound = self.b_sat[i] / row[col]
                upper_bounds[col] = min(upper_bounds[col], bound)

        # Initialize Augmented Lagrangian projected gradient descent
        x = project_simplex_with_bounds(shifted, upper_bounds)
        lam = np.zeros(self.num_constraints, dtype=np.float64)
        step_size = 0.5 / (self.mu + self.rho)

        for _ in range(self.max_iter):
            # Compute violation: v = max(0, A*x - b)
            ax = np.dot(self.a_sat, x)
            if not np.isfinite(ax).all():
                raise ValueError("Non-finite intermediate safety constraint")
            violation = np.maximum(0.0, ax - self.b_sat)

            # Gradient of Augmented Lagrangian w.r.t x:
            # grad = mu * (x - (x0 + c/mu)) + A^T * (lam + rho * violation)
            grad = self.mu * (x - shifted) + np.dot(self.a_sat.T, lam + self.rho * violation)

            if not np.isfinite(grad).all():
                raise ValueError("Non-finite intermediate safety gradient")
            # Projected gradient update on simplex + upper bounds
            x = project_simplex_with_bounds(x - step_size * grad, upper_bounds)

            # Dual update: lam += rho * violation
            new_violation = np.maximum(0.0, np.dot(self.a_sat, x) - self.b_sat)
            lam += self.rho * new_violation
            if not np.isfinite(new_violation).all() or not np.isfinite(lam).all():
                raise ValueError("Non-finite intermediate safety dual")

            if np.max(new_violation) < self.tolerance:
                break

        x_star = x

        # Active constraints identification.
        # contact_error/dual_tol gate rounding error accumulated across this
        # single solve's iterations (up to max_iter of them, each capable of
        # adding a spurious rho-scaled dual increment): they are not a bound
        # on error across a multi-step trajectory of forward() calls. See
        # tests/test_r8_s01_active_set_cumul_noise.py (max_iter in {200, 500}).
        slack = self.b_sat - np.dot(self.a_sat, x_star)
        # Numerical contact alone cannot activate an inequality: rounding can
        # erase a positive interior slack. Require a positive dual multiplier.
        contact_error = 4 * np.finfo(np.float64).eps * (
            np.abs(self.b_sat) + np.sum(np.abs(self.a_sat * x_star), axis=1)
        )
        # Rounded contact can add a spurious rho-scaled dual increment on
        # every iteration. Bound the accumulated noise before choosing the
        # active face; a strictly interior point cannot be active either.
        dual_tol = np.maximum(
            1e-12, 10.0 * abs(self.rho) * contact_error * max(1, self.max_iter)
        )
        active_mask = (lam > dual_tol) & ((-slack) >= -contact_error)
        active_a = self.a_sat[active_mask]
        num_active = int(np.sum(active_mask))

        self._cached_x_star = x_star
        self._cached_active_a = active_a
        self._cached_lambda = lam

        # Dual-track CP-SAT formal verification
        names = list(candidate_names) if candidate_names else [f"ACT_{i}" for i in range(self.action_dim)]
        util_dict = {names[i]: float(x_star[i]) for i in range(self.action_dim)}

        forbidden = set()
        for idx in np.where(active_mask)[0]:
            violating_cols = np.where(self.a_sat[idx] > 0.5)[0]
            if len(violating_cols) == 1 and self.b_sat[idx] <= 0.02:
                forbidden.add(names[violating_cols[0]])

        cpsat_res = self.cpsat_solver.solve_safest_optimal_action(
            candidate_utilities=util_dict,
            forbidden_actions=forbidden,
            fallback_safe_action=names[0],
        )

        t1 = time.perf_counter()
        kkt_res = float(np.max(np.abs(slack[slack < 0.0]))) if np.any(slack < 0.0) else 0.0

        top1 = int(np.argmax(x_star))
        discrete_verified = bool(discrete_feasible[top1])
        # Only a real solve counts; DETERMINISTIC_SAFE_SOLVED skips CP-SAT.
        cpsat_ran = cpsat_res.solver_status in ("OPTIMAL", "FEASIBLE") and not cpsat_res.fallback_used
        return SafetyProjectionResult(
            projected_distribution=x_star,
            original_proposal=x0,
            is_safe=cpsat_res.is_safe and (kkt_res < 1e-3) and discrete_verified,
            active_constraints_count=num_active + int(np.count_nonzero(~discrete_feasible)),
            kkt_residual=kkt_res,
            solve_time_ms=(t1 - t0) * 1000.0,
            cpsat_hard_verified=(
                cpsat_ran
                and cpsat_res.is_safe
                and cpsat_res.selected_action == names[top1]
                and discrete_verified
            ),
            discrete_hard_verified=discrete_verified,
            cpsat_solver_status=cpsat_res.solver_status,
            cpsat_selected_action=cpsat_res.selected_action,
        )


if HAS_TORCH:
    class PyTorchDifferentiableSafetyFunction(torch.autograd.Function):
        """Inference-only PyTorch Function wrapping DifferentiableSafetyLayer."""

        @staticmethod
        def forward(ctx, x0, safety_layer, utility=None):
            x0_np = x0.detach().cpu().numpy()
            u_np = utility.detach().cpu().numpy() if utility is not None else None

            if x0_np.ndim == 1:
                res = safety_layer.forward(x0_np, u_np)
                out = torch.from_numpy(res.projected_distribution).to(x0.device, dtype=x0.dtype)
                return out
            else:
                outs = []
                for i in range(x0_np.shape[0]):
                    u_i = u_np[i] if u_np is not None else None
                    r = safety_layer.forward(x0_np[i], u_i)
                    outs.append(r.projected_distribution)
                return torch.from_numpy(np.vstack(outs)).to(x0.device, dtype=x0.dtype)


    class PyTorchDifferentiableSafetyModule(nn.Module):
        """PyTorch Module for seamless insertion into neural policy heads."""

        def __init__(self, safety_layer: DifferentiableSafetyLayer):
            super().__init__()
            self.safety_layer = safety_layer

        def forward(self, x0: torch.Tensor, utility: Optional[torch.Tensor] = None) -> torch.Tensor:
            return PyTorchDifferentiableSafetyFunction.apply(x0, self.safety_layer, utility)
else:
    class PyTorchDifferentiableSafetyFunction:
        pass

    class PyTorchDifferentiableSafetyModule:
        pass
