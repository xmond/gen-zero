"""Cache-oblivious fractal multiscale dynamical thinking engine.

Implements module 2 of docs/research/lora_parallel_rnn_and_fractal_cpu_zero_token.md
(section 3): a dyadic (binary self-similar) time-scale hierarchy over a phase
state x = (q, p), used as a two-phase 0-token continuous thinking step
z_{k+1} = Phi(z_k; c) around a frozen problem context c.

Self-similarity (section 3.1). Every scale s uses the *same* damped leapfrog
map `leapfrog_step`, only the step size dt = dt_micro * tau_s changes:

    h^{(s)}_{m+1} = Phi_{tau_s}(h^{(s)}_m; c)

Phase 1 -- macro pruning (tau_macro, section 4.2). At most two coarse steps
run a counterfactual-repulsion check: if the momentum direction lines up with
a supplied forbidden ("counterfactual") direction beyond `repulsion_threshold`,
the branch is an irreversible logical contradiction and gets a damped
backtrack (a reflection of the offending momentum component) and is pruned,
never reaching phase 2. `counterfactual_repulsion_score` is built only from
inner products and vector norms of the *actual* state vectors -- never a
coordinate-wise statistic such as `x - x.mean()` -- so it, and every step of
the engine, is exactly equivariant under any orthogonal Q (Q^T Q = I): reflect
q, p and every repulsor by Q and the whole trajectory reflects by Q too.

Phase 2 -- micro relaxation (tau=1, section 4.2). Damped leapfrog is
conformally symplectic: dH/dt = -gamma*||p||^2 <= 0, so H is a Lyapunov
function and the phase point contracts onto the harmonic well's minimum
q=c, p=0. Iterates until the algebraic conservation residual
||R(q,c)|| = sqrt(||grad V(q;c)||^2 + ||p||^2) drops below `residual_tol`.

Cache-oblivious blocking (section 3.2). `verify_cache_bound` reports the
*algorithmic* working-set byte count of a batch (state, momentum, force and
one scratch vector per item) against an L1 budget (32KB by default);
`recursive_cache_oblivious_partition` bisects a batch until every block fits.
This verifies the byte-accounting bound the algorithm is designed to respect;
it is not a hardware cache-miss measurement (no perf counters are read here),
consistent with the anti-hype rule in section 5.4 of the design doc.

Honesty notes (section 5.4 "reject metric hallucination"):
  * V(q;c) is a pure harmonic well: "thinking" here is *linear* relaxation
    around a frozen context c, not a learned or nonlinear potential. The
    Lyapunov/self-similarity machinery is real and testable precisely because
    the dynamics are linear; a nonlinear V would need the same two-phase
    structure but a different (non-closed-form) stability argument.
  * `think_batch` walks the cache-oblivious block schedule item by item; it
    honestly implements the *algorithmic* working-set bound `verify_cache_bound`
    checks, not a measured wall-clock or hardware cache-hit speedup.
  * The design doc's own illustrative "5-8 micro steps" (section 4.2) assumes a
    much tighter macro localization than a 2-step linear leapfrog achieves
    against a residual_tol=1e-5; this implementation instead delivers a
    reproducible ~30-40% cut in total integration steps versus the single-scale
    baseline (see test_fractal_multiscale_engine.py), not that specific figure.
"""
from __future__ import annotations

import dataclasses
import math
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "L1_CACHE_BYTES_DEFAULT",
    "STATE_VECTORS_PER_ITEM",
    "CacheBoundReport",
    "MacroPruneResult",
    "MicroRelaxResult",
    "ThinkResult",
    "working_set_bytes",
    "verify_cache_bound",
    "recursive_cache_oblivious_partition",
    "leapfrog_step",
    "hamiltonian",
    "residual_norm",
    "counterfactual_repulsion_score",
    "FractalMultiscaleEngine",
]

L1_CACHE_BYTES_DEFAULT = 32768
STATE_VECTORS_PER_ITEM = 4  # q, p, grad(V), leapfrog scratch, per batch item


# --------------------------------------------------------------------------
# Validation helpers (fail-closed: reject, never coerce or silently repair)
# --------------------------------------------------------------------------

def _vector(value, name: str, dim: Optional[int] = None) -> np.ndarray:
    v = np.asarray(value, dtype=np.float64)
    if v.ndim != 1 or v.size == 0:
        raise ValueError(f"{name} must be a nonempty 1-D vector")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must contain only finite values")
    if dim is not None and v.shape[0] != dim:
        raise ValueError(f"{name} must have dimension {dim}, got {v.shape[0]}")
    return v.copy()


def _repulsor_matrix(value, name: str, dim: int) -> np.ndarray:
    r = np.asarray(value, dtype=np.float64)
    if r.ndim != 2 or r.shape[0] == 0 or r.shape[1] != dim:
        raise ValueError(f"{name} must be a nonempty (m, {dim}) array")
    if not np.all(np.isfinite(r)):
        raise ValueError(f"{name} must contain only finite values")
    return r.copy()


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_float(value, name: str) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return v


# --------------------------------------------------------------------------
# Cache-oblivious L1 working-set bound
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CacheBoundReport:
    """Byte-accounting report for `verify_cache_bound`. Not a hardware measurement."""
    dim: int
    batch_size: int
    dtype_bytes: int
    working_set_bytes: int
    l1_bytes: int
    within_bound: bool


def working_set_bytes(dim: int, batch_size: int, *, dtype_bytes: int = 8,
                      state_vectors: int = STATE_VECTORS_PER_ITEM) -> int:
    return dim * batch_size * dtype_bytes * state_vectors


def verify_cache_bound(dim: int, batch_size: int, *, dtype_bytes: int = 8,
                       l1_bytes: int = L1_CACHE_BYTES_DEFAULT,
                       state_vectors: int = STATE_VECTORS_PER_ITEM) -> CacheBoundReport:
    """Report whether a (dim, batch_size) block's working set fits in `l1_bytes`."""
    dim = _positive_int(dim, "dim")
    batch_size = _positive_int(batch_size, "batch_size")
    dtype_bytes = _positive_int(dtype_bytes, "dtype_bytes")
    l1_bytes = _positive_int(l1_bytes, "l1_bytes")
    state_vectors = _positive_int(state_vectors, "state_vectors")
    wsb = working_set_bytes(dim, batch_size, dtype_bytes=dtype_bytes, state_vectors=state_vectors)
    return CacheBoundReport(dim, batch_size, dtype_bytes, wsb, l1_bytes, wsb <= l1_bytes)


def recursive_cache_oblivious_partition(dim: int, batch_size: int, *, dtype_bytes: int = 8,
                                        l1_bytes: int = L1_CACHE_BYTES_DEFAULT,
                                        state_vectors: int = STATE_VECTORS_PER_ITEM,
                                        ) -> Tuple[int, ...]:
    """Binary-recursive cache-oblivious partition of a batch into L1-bounded blocks.

    Bisects the batch until every block's `verify_cache_bound` working set fits
    in `l1_bytes`. Fails closed: if a single item (batch_size == 1) still
    exceeds the bound, `dim` itself is too large for this cache tier and this
    raises rather than silently returning an oversized block.
    """
    dim = _positive_int(dim, "dim")
    batch_size = _positive_int(batch_size, "batch_size")

    def split(n: int) -> Tuple[int, ...]:
        report = verify_cache_bound(dim, n, dtype_bytes=dtype_bytes, l1_bytes=l1_bytes,
                                    state_vectors=state_vectors)
        if report.within_bound:
            return (n,)
        if n == 1:
            raise ValueError(
                f"dim={dim} alone needs {report.working_set_bytes} bytes per item, "
                f"exceeding l1_bytes={l1_bytes} even at batch_size=1"
            )
        left = n // 2
        return split(left) + split(n - left)

    return split(batch_size)


# --------------------------------------------------------------------------
# Self-similar harmonic phase dynamics: shared by every dyadic scale
# --------------------------------------------------------------------------

def _grad_v(q: np.ndarray, c: np.ndarray, omega: float) -> np.ndarray:
    """grad V(q; c) for the coercive well V(q;c) = 0.5*omega^2*||q - c||^2."""
    return (omega ** 2) * (q - c)


def hamiltonian(q: np.ndarray, p: np.ndarray, c: np.ndarray, omega: float) -> float:
    """H(q, p; c) = 0.5*||p||^2 + 0.5*omega^2*||q - c||^2."""
    return 0.5 * float(p @ p) + 0.5 * (omega ** 2) * float((q - c) @ (q - c))


def residual_norm(q: np.ndarray, p: np.ndarray, c: np.ndarray, omega: float) -> float:
    """||R(q, c)|| = sqrt(||grad V(q;c)||^2 + ||p||^2): zero exactly at (q=c, p=0)."""
    g = _grad_v(q, c, omega)
    return float(math.sqrt(float(g @ g) + float(p @ p)))


def leapfrog_step(q: np.ndarray, p: np.ndarray, c: np.ndarray, *, omega: float,
                  dt: float, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
    """One damped leapfrog (Stormer-Verlet) step: kick(dt/2), drift(dt), kick(dt/2).

    This single functional form Phi is reused, unchanged, at every dyadic
    scale s -- only dt = dt_micro * tau_s and gamma differ between macro and
    micro calls. Every operation is a scalar multiply or an inner product of
    q, p and c, so `leapfrog_step` is exactly equivariant under any orthogonal
    Q applied consistently to q, p and c.
    """
    damp = math.exp(-gamma * dt * 0.5)
    half = 0.5 * dt
    p = damp * p
    p = p - half * _grad_v(q, c, omega)
    q_next = q + dt * p
    p = p - half * _grad_v(q_next, c, omega)
    p = damp * p
    return q_next, p


def counterfactual_repulsion_score(direction, repulsors) -> float:
    """Orthogonal-equivariant counterfactual repulsion score, in [0, 1].

    score = max_j max(0, cos(direction, repulsor_j))^2

    Built only from the inner product and norm of `direction` against each row
    of `repulsors`: no coordinate-wise statistic (e.g. `x - x.mean()`) is used,
    since that assumes a distinguished direction (the all-ones axis) that an
    arbitrary orthogonal Q does not preserve. For any Q with Q^T Q = I, applying
    Q to both `direction` and every row of `repulsors` leaves the score exactly
    unchanged, because Q preserves inner products and norms.
    """
    d = _vector(direction, "direction")
    r = _repulsor_matrix(repulsors, "repulsors", d.shape[0])
    nd = np.linalg.norm(d)
    if nd == 0.0:
        return 0.0
    best = 0.0
    for row in r:
        nr = np.linalg.norm(row)
        if nr == 0.0:
            continue
        cos = float(np.dot(d, row) / (nd * nr))
        best = max(best, max(0.0, cos) ** 2)
    return best


def _repel(p: np.ndarray, repulsor: np.ndarray, strength: float) -> np.ndarray:
    """Damped backtrack: reflect+shrink the component of p aligned with `repulsor`.

    Only the component of p pointing *into* the forbidden direction is
    touched, and only via the inner product and the repulsor vector itself, so
    this is equivariant under any orthogonal Q applied to p and repulsor.
    """
    nr = np.linalg.norm(repulsor)
    if nr == 0.0:
        return p
    unit = repulsor / nr
    aligned = float(np.dot(p, unit))
    if aligned <= 0.0:
        return p
    return p - (1.0 + strength) * aligned * unit


# --------------------------------------------------------------------------
# Two-phase engine
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class MacroPruneResult:
    pruned: bool
    steps: int
    repulsion_score: float
    q: np.ndarray
    p: np.ndarray
    energy_trace: Tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class MicroRelaxResult:
    steps: int
    converged: bool
    residual: float
    q: np.ndarray
    p: np.ndarray
    energy_trace: Tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class ThinkResult:
    pruned: bool
    macro: MacroPruneResult
    micro: Optional[MicroRelaxResult]
    total_steps: int
    q: np.ndarray
    p: np.ndarray
    residual: float


class FractalMultiscaleEngine:
    """Two-phase dyadic (macro/micro) thinking engine over a harmonic phase well.

    Phase 1 (macro, up to `max_macro_steps` <= 2 steps of size
    dt_micro*tau_macro): counterfactual-repulsion pruning against `repulsors`,
    with a damped backtrack the instant a contradiction is found.
    Phase 2 (micro, tau=1 steps of size dt_micro): Lyapunov-dissipative
    convergence to the algebraic residual tolerance, seeded from the surviving
    macro state (the cross-scale state hand-off of section 3.1).
    """

    def __init__(self, dim: int, *, tau_macro: int = 8, dt_micro: float = 0.125,
                omega: float = 1.0, gamma_macro: float = 2.0, gamma_micro: float = 2.0,
                repulsion_threshold: float = 0.5, repulsion_backtrack_strength: float = 1.0,
                residual_tol: float = 1e-5, max_macro_steps: int = 2,
                max_micro_steps: int = 300, l1_bytes: int = L1_CACHE_BYTES_DEFAULT) -> None:
        self.dim = _positive_int(dim, "dim")
        self.tau_macro = _positive_int(tau_macro, "tau_macro")
        if self.tau_macro < 2 or (self.tau_macro & (self.tau_macro - 1)) != 0:
            raise ValueError(
                f"tau_macro must be a dyadic power of two (2^s, s>=1) per the "
                f"design's binary self-similar tree, got {self.tau_macro}"
            )
        self.dt_micro = _positive_float(dt_micro, "dt_micro")
        self.omega = _positive_float(omega, "omega")
        self.gamma_macro = _positive_float(gamma_macro, "gamma_macro")
        self.gamma_micro = _positive_float(gamma_micro, "gamma_micro")
        if not (0.0 < float(repulsion_threshold) <= 1.0):
            raise ValueError("repulsion_threshold must be in (0, 1]")
        self.repulsion_threshold = float(repulsion_threshold)
        self.repulsion_backtrack_strength = _positive_float(
            repulsion_backtrack_strength, "repulsion_backtrack_strength")
        self.residual_tol = _positive_float(residual_tol, "residual_tol")
        self.max_macro_steps = _positive_int(max_macro_steps, "max_macro_steps")
        if self.max_macro_steps > 2:
            raise ValueError(
                "max_macro_steps must be <= 2: contradictions must backtrack within "
                "a fast damped window, per the design's macro-pruning contract"
            )
        self.max_micro_steps = _positive_int(max_micro_steps, "max_micro_steps")
        self.l1_bytes = _positive_int(l1_bytes, "l1_bytes")
        dt_macro = self.dt_micro * self.tau_macro
        if dt_macro * self.omega >= 2.0:
            raise ValueError(
                "dt_micro * tau_macro * omega must stay below the leapfrog "
                f"stability limit of 2.0 (got {dt_macro * self.omega})"
            )
        self.dt_macro = dt_macro

    # -- phase 1: macro counterfactual pruning ---------------------------

    def macro_prune(self, q0, p0, c, repulsors=None) -> MacroPruneResult:
        q = _vector(q0, "q0", self.dim)
        p = _vector(p0, "p0", self.dim)
        c = _vector(c, "c", self.dim)
        r = None if repulsors is None else _repulsor_matrix(repulsors, "repulsors", self.dim)
        trace = [hamiltonian(q, p, c, self.omega)]

        for step in range(1, self.max_macro_steps + 1):
            q, p = leapfrog_step(q, p, c, omega=self.omega, dt=self.dt_macro,
                                 gamma=self.gamma_macro)
            trace.append(hamiltonian(q, p, c, self.omega))
            if r is None:
                continue
            score = counterfactual_repulsion_score(p, r)
            if score > self.repulsion_threshold:
                for row in r:
                    p = _repel(p, row, self.repulsion_backtrack_strength)
                return MacroPruneResult(True, step, score, q, p, tuple(trace))
        score = 0.0 if r is None else counterfactual_repulsion_score(p, r)
        return MacroPruneResult(False, self.max_macro_steps, score, q, p, tuple(trace))

    # -- phase 2: micro Lyapunov relaxation -------------------------------

    def micro_relax(self, q0, p0, c, *, max_steps: Optional[int] = None) -> MicroRelaxResult:
        q = _vector(q0, "q0", self.dim)
        p = _vector(p0, "p0", self.dim)
        c = _vector(c, "c", self.dim)
        budget = self.max_micro_steps if max_steps is None else _positive_int(max_steps, "max_steps")
        trace = [hamiltonian(q, p, c, self.omega)]
        residual = residual_norm(q, p, c, self.omega)
        converged = residual < self.residual_tol
        steps = 0
        if not converged:
            for steps in range(1, budget + 1):
                q, p = leapfrog_step(q, p, c, omega=self.omega, dt=self.dt_micro,
                                     gamma=self.gamma_micro)
                trace.append(hamiltonian(q, p, c, self.omega))
                residual = residual_norm(q, p, c, self.omega)
                if residual < self.residual_tol:
                    converged = True
                    break
        return MicroRelaxResult(steps, converged, residual, q, p, tuple(trace))

    def single_scale_baseline(self, q0, p0, c, *, max_steps: int = 4096) -> MicroRelaxResult:
        """Reference tau=1-only convergence with no macro pre-localization.

        Used to measure the step-count saving multiscale thinking buys over
        running the fine-grained integrator alone from the raw initial state.
        Does not mutate engine state: the larger step budget is passed straight
        through to `micro_relax`.
        """
        return self.micro_relax(q0, p0, c, max_steps=max_steps)

    # -- combined two-phase step -------------------------------------------

    def think(self, q0, p0, c, repulsors=None) -> ThinkResult:
        macro = self.macro_prune(q0, p0, c, repulsors)
        if macro.pruned:
            residual = residual_norm(macro.q, macro.p, _vector(c, "c", self.dim), self.omega)
            return ThinkResult(True, macro, None, macro.steps, macro.q, macro.p, residual)
        micro = self.micro_relax(macro.q, macro.p, c)
        return ThinkResult(False, macro, micro, macro.steps + micro.steps,
                           micro.q, micro.p, micro.residual)

    def think_batch(self, q0_batch, p0_batch, c, repulsors=None) -> List[ThinkResult]:
        """Process a batch through cache-oblivious, L1-bounded blocks.

        Repartitions the batch via `recursive_cache_oblivious_partition` so
        that no block's working set (per `verify_cache_bound`) exceeds
        `l1_bytes`, then runs `think` block by block, bounding how much state
        is live at any one time to the cache budget.
        """
        q_batch = np.asarray(q0_batch, dtype=np.float64)
        p_batch = np.asarray(p0_batch, dtype=np.float64)
        if (q_batch.ndim != 2 or q_batch.shape != p_batch.shape
                or q_batch.shape[1] != self.dim or q_batch.shape[0] == 0):
            raise ValueError("q0_batch and p0_batch must both be nonempty (batch, dim) arrays")
        n = q_batch.shape[0]
        blocks = recursive_cache_oblivious_partition(self.dim, n, l1_bytes=self.l1_bytes)
        results: List[ThinkResult] = []
        offset = 0
        for size in blocks:
            for i in range(offset, offset + size):
                results.append(self.think(q_batch[i], p_batch[i], c, repulsors))
            offset += size
        return results
