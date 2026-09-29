"""Bifurcated fractal engine: macro multi-hypothesis race, micro affine convergence.

Design: docs/research/04-bifurcation-control-multi-hypothesis.md.

Three stages, all pure NumPy, all built from inner products, norms and linear
combinations of the actual state vectors (orthogonal-equivariant, no coordinate
axis is ever distinguished):

  [A] bifurcate   K in {2,3,4} orthogonal momentum perturbations, K chosen from
                  the normalized entropy of the counterfactual repulsion scores.
  [B] race        up to `max_macro_steps` damped leapfrog steps at dt_macro on
                  the whole (K, dim) pool at once; after every step at least
                  floor(n_alive/2) branches are dropped, so K=4 -> 2 -> 1.
                  Non-finite cost or repulsion above threshold kills a branch.
                  If the hard repulsion gate would kill every branch in a step,
                  the pool is never allowed to go extinct: the branch with the
                  least *finite* cost is rescued instead (cost already prices
                  repulsion in softly, see `_cost`), and `RaceTrace.rescued`
                  records that it happened. Only when not one branch even has
                  a finite cost is the race a genuine PRUNED result (no state
                  to rescue from, not a random survivor).
  [C] micro       the single survivor is refined by the LoRA affine recurrence
                  h <- A h + B c, either by the closed-form Woodbury fixed point
                  (default: O(dim*rank^2), zero iterations) or by <= 20 explicit
                  steps. The iterative path is only meaningful once A becomes
                  nonlinear; with a linear A it is strictly slower than closed form.

Honesty notes:
  * The adapter's fixed point (I-A)^{-1} B c equals the well minimum q=c only
    if the adapter has been trained to that contract. An untrained adapter
    converges fast to a point that has nothing to do with c. `think` reports
    `micro.fixed_point_gap` = ||h* - c|| so a caller can see that gap.
  * `working_set_bytes` is a byte ledger, not a hardware cache measurement.
  * Every step is a NumPy call costing ~10-30 us of interpreter time; the
    absolute latency floor is set by that, not by FLOPs (see design doc 0.4).
"""
from __future__ import annotations

import dataclasses
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .fractal_multiscale_engine import (
    _positive_float,
    _positive_int,
    _repulsor_matrix,
    _vector,
)
from .parallel_rnn_lora import ParallelRNNLoRAAdapter

__all__ = [
    "BranchPool",
    "BifurcationDecision",
    "RaceTrace",
    "MicroResult",
    "BifurcatedThinkResult",
    "gram_schmidt",
    "normalized_entropy",
    "pick_k",
    "spectral_radius",
    "repulsion_scores_batch",
    "woodbury_fixed_point",
    "working_set_bytes",
    "BifurcatedFractalEngine",
]

STATUS_CONVERGED = "CONVERGED"
STATUS_PRUNED = "PRUNED"
STATUS_UNCONVERGED = "UNCONVERGED"


# --------------------------------------------------------------------------
# Data structures (immutable; every macro step yields a new BranchPool)
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class BranchPool:
    q: np.ndarray          # (K, dim)
    p: np.ndarray          # (K, dim)
    cost: np.ndarray       # (K,)
    alive: np.ndarray      # (K,) bool
    seed_dirs: np.ndarray  # (K-1, dim) orthonormal


@dataclasses.dataclass(frozen=True)
class BifurcationDecision:
    K: int
    uncertainty: float
    repulsion_scores: Tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class RaceTrace:
    steps: int
    alive_history: Tuple[int, ...]
    cost_history: Tuple[Tuple[float, ...], ...]
    pruned_all: bool
    rescued: bool = False


@dataclasses.dataclass(frozen=True)
class MicroResult:
    path: str
    steps: int
    converged: bool
    residual: float
    fixed_point_gap: float
    h: np.ndarray


@dataclasses.dataclass(frozen=True)
class BifurcatedThinkResult:
    status: str
    survivor: Optional[int]
    bifurcation: BifurcationDecision
    race: RaceTrace
    micro: Optional[MicroResult]
    q: Optional[np.ndarray]
    p: Optional[np.ndarray]
    residual: float
    working_set_bytes: int


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------

def gram_schmidt(candidates: Sequence[np.ndarray], *, max_dirs: int,
                 tol: float = 1e-10) -> np.ndarray:
    """Orthonormalize `candidates` in order, dropping near-dependent vectors.

    Returns an (n, dim) array with n <= max_dirs. Built only from inner
    products and norms, so it commutes with any orthogonal Q applied to every
    candidate. Never pads with coordinate axes: if the candidates span fewer
    than `max_dirs` directions, fewer are returned.
    """
    out: List[np.ndarray] = []
    for v in candidates:
        if len(out) >= max_dirs:
            break
        w = np.array(v, dtype=np.float64)
        for u in out:
            w = w - float(u @ w) * u
        n = float(np.linalg.norm(w))
        if n > tol * max(1.0, float(np.linalg.norm(v))):
            out.append(w / n)
    if not out:
        return np.zeros((0, len(candidates[0]) if candidates else 0))
    return np.stack(out)


def normalized_entropy(weights: np.ndarray) -> float:
    """Entropy of a softmax over `weights`, divided by log(m); in [0, 1]. m<2 -> 0."""
    w = np.asarray(weights, dtype=np.float64)
    if w.size < 2:
        return 0.0
    z = w - np.max(w)
    prob = np.exp(z) / np.sum(np.exp(z))
    ent = -float(np.sum(prob * np.log(prob + 1e-300)))
    return max(0.0, min(1.0, ent / math.log(w.size)))


def pick_k(uncertainty: float, *, u_lo: float, u_hi: float) -> int:
    if not (0.0 <= uncertainty <= 1.0):
        raise ValueError("uncertainty must be in [0, 1]")
    if uncertainty < u_lo:
        return 2
    if uncertainty < u_hi:
        return 3
    return 4


def spectral_radius(matrix: np.ndarray) -> float:
    """max |eigenvalue|; the true long-run contraction rate of h <- A h + u."""
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(matrix, dtype=np.float64)))))


def repulsion_scores_batch(directions: np.ndarray, repulsors: Optional[np.ndarray]) -> np.ndarray:
    """(K, dim) x (m, dim) -> (K,): row-wise `counterfactual_repulsion_score`.

    Same definition, max_j max(0, cos)^2, with zero-norm rows contributing 0,
    computed with one matrix product instead of K*m Python calls.
    """
    d = np.asarray(directions, dtype=np.float64)
    if repulsors is None or repulsors.shape[0] == 0:
        return np.zeros(d.shape[0])
    return np.max(np.maximum(_cosine_rows(d, repulsors), 0.0) ** 2, axis=1)


def _cosine_rows(d: np.ndarray, r: np.ndarray) -> np.ndarray:
    """(K, dim) x (m, dim) -> (K, m) cosines; a zero-norm row on either side gives 0."""
    nd = np.linalg.norm(d, axis=1)
    nr = np.linalg.norm(r, axis=1)
    denom = nd[:, None] * nr[None, :]
    safe = np.where(denom > 0.0, denom, 1.0)
    return np.where(denom > 0.0, (d @ r.T) / safe, 0.0)


def woodbury_fixed_point(adapter: ParallelRNNLoRAAdapter, x: np.ndarray, *,
                         verify_tol: float = 1e-9) -> Tuple[np.ndarray, float]:
    """Closed-form fixed point h* = (I - A)^{-1} B x without forming A.

    A = s (D + U V^T) with D diagonal, so I - A = M - s U V^T, M = I - s D.
    Woodbury: (M - sUV^T)^{-1} = M^{-1} + s M^{-1} U (I_r - s V^T M^{-1} U)^{-1} V^T M^{-1}.
    Cost O(dim*rank^2). Fails closed: raises if M is singular or the r x r
    core is singular, and returns the verified residual ||h* - A h* - B x||.
    """
    x = _vector(x, "x", adapter.input_dim)
    s = adapter._scale
    d = adapter._dvec()
    m_diag = 1.0 - s * d
    if np.any(np.abs(m_diag) < 1e-12):
        raise ValueError("woodbury_fixed_point: I - s*D is singular")
    u = adapter.U_A
    v = adapter.V_A
    b = adapter.U_B @ (adapter.V_B.T @ x)
    minv_b = b / m_diag
    minv_u = u / m_diag[:, None]
    core = np.eye(adapter.rank) - s * (v.T @ minv_u)
    if abs(float(np.linalg.det(core))) < 1e-14:
        raise ValueError("woodbury_fixed_point: rank-r core is singular")
    h = minv_b + s * minv_u @ np.linalg.solve(core, v.T @ minv_b)
    resid = float(np.linalg.norm(h - adapter.step(x, h)))
    if not np.isfinite(resid) or resid > verify_tol * max(1.0, float(np.linalg.norm(h))):
        raise ValueError(f"woodbury_fixed_point: residual {resid} exceeds tolerance")
    return h, resid


def working_set_bytes(dim: int, k: int, rank: int, n_repulsors: int, *,
                      dtype_bytes: int = 8) -> int:
    """Byte ledger from design doc section 2.5. Not a cache-miss measurement."""
    pool = k * dim * 4 * dtype_bytes
    lora = (dim * rank * 4 + dim) * dtype_bytes
    rep = n_repulsors * dim * dtype_bytes
    scalars = k * (8 + 8 + 1)
    return pool + lora + rep + scalars


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

class BifurcatedFractalEngine:
    def __init__(self, dim: int, adapter: ParallelRNNLoRAAdapter, *, tau_macro: int = 8,
                 dt_micro: float = 0.2, omega: float = 1.0, gamma_macro: float = 2.0,
                 max_macro_steps: int = 4, repulsion_threshold: float = 0.5,
                 lambda_rep: float = 1.0, u_lo: float = 0.33, u_hi: float = 0.66,
                 entropy_temperature: float = 1.0, rho_spec: float = 0.55,
                 micro_path: str = "closed_form", micro_max_steps: int = 20,
                 micro_tol: float = 1e-5, l2_bytes: Optional[int] = None,
                 n_repulsors: int = 0) -> None:
        self.dim = _positive_int(dim, "dim")
        if not isinstance(adapter, ParallelRNNLoRAAdapter):
            raise TypeError("adapter must be a ParallelRNNLoRAAdapter")
        if adapter.dim != self.dim or adapter.input_dim != self.dim:
            raise ValueError("adapter dim and input_dim must both equal dim")
        rho = spectral_radius(adapter.transition_matrix())
        if not (0.0 < float(rho_spec) < 1.0):
            raise ValueError("rho_spec must be in (0, 1)")
        if rho > rho_spec:
            raise ValueError(
                f"adapter spectral radius {rho:.4f} exceeds rho_spec={rho_spec}: "
                f"{micro_max_steps} steps cannot guarantee convergence")
        self.adapter = adapter
        self.adapter_spectral_radius = rho
        self.tau_macro = _positive_int(tau_macro, "tau_macro")
        if self.tau_macro < 2 or (self.tau_macro & (self.tau_macro - 1)) != 0:
            raise ValueError("tau_macro must be a power of two >= 2")
        self.dt_micro = _positive_float(dt_micro, "dt_micro")
        self.omega = _positive_float(omega, "omega")
        self.gamma_macro = _positive_float(gamma_macro, "gamma_macro")
        self.dt_macro = self.dt_micro * self.tau_macro
        if self.dt_macro * self.omega >= 2.0:
            raise ValueError("dt_micro * tau_macro * omega must stay below 2.0")
        self.max_macro_steps = _positive_int(max_macro_steps, "max_macro_steps")
        if not (0.0 < float(repulsion_threshold) <= 1.0):
            raise ValueError("repulsion_threshold must be in (0, 1]")
        self.repulsion_threshold = float(repulsion_threshold)
        self.lambda_rep = float(lambda_rep)
        if not (0.0 <= self.lambda_rep):
            raise ValueError("lambda_rep must be >= 0")
        if not (0.0 < u_lo < u_hi < 1.0):
            raise ValueError("need 0 < u_lo < u_hi < 1")
        self.u_lo, self.u_hi = float(u_lo), float(u_hi)
        self.entropy_temperature = _positive_float(entropy_temperature, "entropy_temperature")
        if micro_path not in ("closed_form", "iterative"):
            raise ValueError("micro_path must be 'closed_form' or 'iterative'")
        self.micro_path = micro_path
        self.micro_max_steps = _positive_int(micro_max_steps, "micro_max_steps")
        self.micro_tol = _positive_float(micro_tol, "micro_tol")
        self.rank = adapter.rank
        if isinstance(n_repulsors, bool) or not isinstance(n_repulsors, (int, np.integer)) or n_repulsors < 0:
            raise ValueError("n_repulsors must be a non-negative integer")
        self.n_repulsors = int(n_repulsors)
        if l2_bytes is None:
            # No explicit budget given: size the L2 ledger to what this dim actually
            # needs (a fixed 64 KiB default silently rejects any manifold above
            # ~192D, see TestLedger.test_ledger_matches_design_doc_table) instead of
            # leaving high-dimensional manifolds permanently unusable in this mode.
            needed = working_set_bytes(self.dim, 4, self.rank, self.n_repulsors)
            self.l2_bytes = max(65536, int(needed * 1.5))
        else:
            # An explicit budget is a caller decision (e.g. a real cache-size
            # constraint) and is honored as-is, including failing closed if it is
            # too small -- see TestFailClosed.test_rejects_working_set_over_budget.
            self.l2_bytes = _positive_int(l2_bytes, "l2_bytes")

    # -- [A] bifurcate ------------------------------------------------------

    def bifurcate(self, q0, p0, c, repulsors=None) -> Tuple[BranchPool, BifurcationDecision]:
        q0 = _vector(q0, "q0", self.dim)
        p0 = _vector(p0, "p0", self.dim)
        c = _vector(c, "c", self.dim)
        r = None if repulsors is None else _repulsor_matrix(repulsors, "repulsors", self.dim)
        grad = (self.omega ** 2) * (q0 - c)

        if r is None:
            scores: Tuple[float, ...] = ()
            u_val = 0.0
        else:
            # One score per repulsor: max(0, cos(p0, r_j))^2, one matrix product.
            scores = tuple(float(x) for x in np.max(
                np.maximum(_cosine_rows(p0[None, :], r), 0.0) ** 2, axis=0))
            u_val = normalized_entropy(np.array(scores) / self.entropy_temperature)
        k = pick_k(u_val, u_lo=self.u_lo, u_hi=self.u_hi)

        candidates: List[np.ndarray] = []
        if r is not None:
            candidates.extend(list(r))
        candidates.extend([p0, grad])
        dirs = gram_schmidt(candidates, max_dirs=k - 1)
        k = dirs.shape[0] + 1  # fewer directions -> fewer branches, never padded

        scale = float(np.linalg.norm(p0))
        if scale == 0.0:
            scale = float(np.linalg.norm(grad)) * self.dt_macro
        p_rows = [p0]
        for d in dirs:
            sign = 1.0
            if r is not None:
                # Move away from the repulsor most aligned with this direction.
                # Gotcha: a direction Gram-Schmidt made orthogonal to every
                # repulsor has align ~ +-1e-16; its sign is rounding noise and
                # would make the pool depend on the coordinate frame. Below a
                # relative tolerance there is no repulsor to move away from.
                align = r @ d
                j = int(np.argmax(np.abs(align)))
                if align[j] > 1e-9 * float(np.linalg.norm(r[j])):
                    sign = -1.0
            p_rows.append(p0 + scale * sign * d)
        pool = BranchPool(
            q=np.tile(q0, (k, 1)), p=np.stack(p_rows),
            cost=np.full(k, np.inf), alive=np.ones(k, dtype=bool), seed_dirs=dirs,
        )
        return pool, BifurcationDecision(k, u_val, scores)

    # -- [B] race & prune ---------------------------------------------------

    def _cost(self, pool: BranchPool, c: np.ndarray, r: Optional[np.ndarray]):
        """Vectorized cost over the pool; element-wise equal to the scalar helpers."""
        g = (self.omega ** 2) * (pool.q - c)
        res = np.sqrt(np.einsum("ki,ki->k", g, g) + np.einsum("ki,ki->k", pool.p, pool.p))
        rep = repulsion_scores_batch(pool.p, r)
        return res * (1.0 + self.lambda_rep * rep), rep

    def race(self, pool: BranchPool, c, repulsors=None) -> Tuple[BranchPool, RaceTrace]:
        c = _vector(c, "c", self.dim)
        r = None if repulsors is None else _repulsor_matrix(repulsors, "repulsors", self.dim)
        alive_hist: List[int] = [int(pool.alive.sum())]
        cost_hist: List[Tuple[float, ...]] = []
        damp = math.exp(-self.gamma_macro * self.dt_macro * 0.5)
        half = 0.5 * self.dt_macro
        w2 = self.omega ** 2
        steps = 0
        rescued = False
        for steps in range(1, self.max_macro_steps + 1):
            q, p = pool.q, pool.p
            # Vectorized damped leapfrog over the whole pool (same map as leapfrog_step).
            p = damp * p
            p = p - half * w2 * (q - c)
            q = q + self.dt_macro * p
            p = p - half * w2 * (q - c)
            p = damp * p
            pool = BranchPool(q, p, pool.cost, pool.alive, pool.seed_dirs)
            cost, rep = self._cost(pool, c, r)
            alive = pool.alive & np.isfinite(cost) & (rep <= self.repulsion_threshold)
            n = int(alive.sum())
            if n == 0:
                # The hard repulsion gate killed every branch this step. `cost`
                # already prices repulsion in softly (see `_cost`), so rescue the
                # single least-cost branch among the ones that were alive going
                # into this step, instead of reporting zero survivors. Only if
                # not one of those branches even has a finite cost is there truly
                # nothing to rescue.
                finite = pool.alive & np.isfinite(cost)
                if bool(finite.any()):
                    best = int(min(np.flatnonzero(finite), key=lambda i: (float(cost[i]), int(i))))
                    alive = np.zeros_like(pool.alive)
                    alive[best] = True
                    n = 1
                    rescued = True
                # else: not one previously-alive branch has a finite cost, so
                # there is nothing real to rescue from -- stays a genuine PRUNED
                # result (n == 0), never a fabricated survivor.
            elif n >= 2:
                order = sorted(np.flatnonzero(alive), key=lambda i: (float(cost[i]), int(i)))
                keep = order[: n // 2]
                alive = np.zeros_like(alive)
                alive[keep] = True
                n = len(keep)
            pool = BranchPool(q, p, cost, alive, pool.seed_dirs)
            alive_hist.append(n)
            cost_hist.append(tuple(float(x) for x in cost))
            if n <= 1:
                break
        return pool, RaceTrace(steps, tuple(alive_hist), tuple(cost_hist),
                               int(pool.alive.sum()) == 0, rescued)

    # -- [C] micro convergence ---------------------------------------------

    def micro(self, h0: np.ndarray, c: np.ndarray) -> MicroResult:
        c = _vector(c, "c", self.dim)
        if self.micro_path == "closed_form":
            h, resid = woodbury_fixed_point(self.adapter, c)
            return MicroResult("closed_form", 0, True, resid, float(np.linalg.norm(h - c)), h)
        h = _vector(h0, "h0", self.dim)
        converged = False
        steps = 0
        resid = float("inf")
        for steps in range(1, self.micro_max_steps + 1):
            h_next = self.adapter.step(c, h)
            resid = float(np.linalg.norm(h_next - h))
            h = h_next
            if resid < self.micro_tol * (1.0 + float(np.linalg.norm(h))):
                converged = True
                break
        return MicroResult("iterative", steps, converged, resid, float(np.linalg.norm(h - c)), h)

    # -- full pipeline -------------------------------------------------------

    def think(self, q0, p0, c, repulsors=None) -> BifurcatedThinkResult:
        c_vec = _vector(c, "c", self.dim)
        pool, decision = self.bifurcate(q0, p0, c_vec, repulsors)
        n_rep = 0 if repulsors is None else int(np.asarray(repulsors).shape[0])
        wsb = working_set_bytes(self.dim, decision.K, self.adapter.rank, n_rep)
        if wsb > self.l2_bytes:
            raise ValueError(f"working set {wsb} B exceeds l2_bytes={self.l2_bytes}")
        pool, trace = self.race(pool, c_vec, repulsors)
        if trace.pruned_all:
            return BifurcatedThinkResult(STATUS_PRUNED, None, decision, trace, None,
                                         None, None, float("nan"), wsb)
        survivors = np.flatnonzero(pool.alive)
        survivor = int(min(survivors, key=lambda i: (float(pool.cost[i]), int(i))))
        micro = self.micro(pool.q[survivor], c_vec)
        status = STATUS_CONVERGED if micro.converged else STATUS_UNCONVERGED
        return BifurcatedThinkResult(status, survivor, decision, trace, micro,
                                     micro.h, pool.p[survivor], micro.residual, wsb)
