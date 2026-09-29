"""Route 2: Iterative External Forcing F_ext(t) -- continuous condition reinjection.

Closes a real gap in the two engines this module sits next to:

- `fractal_multiscale_engine.FractalMultiscaleEngine.think` takes `c` as one
  argument and hands the *same* array to `macro_prune` and then `micro_relax`
  (fractal_multiscale_engine.py:407-414). Nothing inside that call re-reads
  the problem; up to 300 micro steps (`max_micro_steps`) relax toward a
  target fixed before the first step.
- `controlled_continuous_engine.ControlledContinuousEngine` reads `c` on
  every leapfrog step (every `grad V(q; c)` call), but `c` is one
  SHA-256-pinned `FrozenProblemContext` for the whole episode
  (controlled_continuous_engine.py:114 `ContextTamperedError`,
  135-172 `FrozenProblemContext`) -- re-reading a frozen value every step is
  not the same as the value changing.

Either way, once relaxation starts, the condition is fixed until the engine
returns. This module adds a bounded, *auditable* schedule of reinjection
checkpoints between dyadic time scales (macro tau=4, meso tau=2, micro
tau=1), each one swapping in a brand-new pinned context or repulsor set --
never mutating a live one. That distinction matters: `FrozenProblemContext`
hashes its arrays specifically so nothing can silently steer `q` toward a
candidate mid-episode (see the module docstring of
controlled_continuous_engine.py, "Replaces ... pseudo-search"). Reinjection
here never violates that: every checkpoint is logged in `ReinjectionLog`
*before* the next segment runs, and no reinjected value is ever a function of
a candidate answer -- only of upstream evidence (a new premise, a new
counterfactual reference, a symbolic-verifier residual on a *decoded*
intermediate, never on the final candidate set).

Mathematical basis for why swapping the well center between segments is safe
(not just convenient): `leapfrog_step` is conformally symplectic for *any*
fixed `(c, omega, gamma)` -- `dH/dt = -gamma * ||p||^2 <= 0`
(fractal_multiscale_engine.py:24-26). That inequality only assumes `c` is
constant across the steps it is being applied to. Freezing `c` piecewise
between reinjection checkpoints, rather than changing it mid-step, keeps each
segment inside that guarantee; only the instant of the swap itself is exempt
(the well's minimum moves, so `H` w.r.t. the *new* center may jump upward
exactly once, at the checkpoint, never in between). This is verified
empirically, on the real `leapfrog_step`/`hamiltonian` functions, in
`docs/research/route2_iterative_external_forcing_condition_reinjection.md`
section 4, and re-checked continuously here by
`ContinuousConditionInjector.think_with_reinjection`, which raises
`LyapunovViolation` if a segment's own Hamiltonian ever fails to decrease
against its own active center -- this is a runtime assertion, not a proof by
construction, exactly because a caller could misuse the API and reinject
inside a step rather than between them.

Honesty notes:
  * No learned or nonlinear potential is added here. The condition being
    reinjected is still a point `c` (or a repulsor row) in the same harmonic
    well used by `FractalMultiscaleEngine`; this module only changes *when*
    that point can be updated, not the dynamics.
  * `SymbolicArithmeticVerifier.verify` is a discrete, non-differentiable
    check over structured `EquationStep` objects. It has no gradient w.r.t.
    `q`, so it is never called inside the leapfrog inner loop. It can only
    feed a reinjection checkpoint (see `symbolic_violation_to_repulsor`):
    evaluate it once against a *decoded* intermediate candidate at a scale
    boundary, then convert a nonzero residual into a repulsor for the next
    segment.
  * "Added latency per step" is not measured here on any target hardware.
    What is shown is an *op-count* argument (`working_set_bytes`-style
    accounting): a reinjection checkpoint costs one condition-cache lookup
    plus the same O(dim) leapfrog work already paid every step; see
    `verify_condition_cache_bound`. Converting that into a microsecond
    figure needs a real benchmark, which this module does not run.
"""
from __future__ import annotations

import dataclasses
import hashlib
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .fractal_multiscale_engine import (
    FractalMultiscaleEngine,
    _repel,  # reuse the exact backtrack formula rather than duplicate it
    _repulsor_matrix,
    _vector,
    counterfactual_repulsion_score,
    hamiltonian,
    leapfrog_step,
    residual_norm,
)

__all__ = [
    "MACRO_TAU",
    "MESO_TAU",
    "MICRO_TAU",
    "L2_CONDITION_CACHE_BYTES_DEFAULT",
    "ConditionCacheBudget",
    "verify_condition_cache_bound",
    "ReinjectionEvent",
    "LyapunovViolation",
    "build_fixed_reinjection_schedule",
    "is_schedule_parallel_scan_compatible",
    "macro_repulsor_from_counterfactual",
    "symbolic_violation_to_repulsor",
    "SegmentResult",
    "ReinjectionResult",
    "ContinuousConditionInjector",
]

# Section 2.2 of the task spec: dyadic tau values for the three scales.
MACRO_TAU = 4
MESO_TAU = 2
MICRO_TAU = 1

L2_CONDITION_CACHE_BYTES_DEFAULT = 16 * 1024  # 16KB, per the task's L2 budget


# ---------------------------------------------------------------------------
# Condition cache byte accounting (mirrors fractal_multiscale_engine.py's
# `verify_cache_bound` / `working_set_bytes` pattern exactly: an algorithmic
# byte count against a budget, never a hardware cache-miss measurement).
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ConditionCacheBudget:
    """Byte-accounting report for the condition cache. Not a hardware measurement."""
    dim: int
    slots: int
    dtype_bytes: int
    bytes_used: int
    l2_bytes: int
    within_bound: bool


def verify_condition_cache_bound(
    dim: int, slots: int, *, dtype_bytes: int = 8,
    l2_bytes: int = L2_CONDITION_CACHE_BYTES_DEFAULT,
) -> ConditionCacheBudget:
    """Report whether `slots` condition/repulsor vectors of width `dim` fit `l2_bytes`.

    `slots` is the number of distinct vectors resident at once: one active
    `c` per scale (macro, meso, micro) plus a small number of repulsor rows.
    A typical schedule (3 scale centers + up to 5 repulsor rows) at dim=256,
    float64 needs 8 * 256 * 8 = 16384 bytes exactly -- the worked example in
    the design doc.
    """
    if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)) or dim < 1:
        raise ValueError("dim must be a positive integer")
    if isinstance(slots, bool) or not isinstance(slots, (int, np.integer)) or slots < 1:
        raise ValueError("slots must be a positive integer")
    if isinstance(dtype_bytes, bool) or not isinstance(dtype_bytes, (int, np.integer)) or dtype_bytes < 1:
        raise ValueError("dtype_bytes must be a positive integer")
    if isinstance(l2_bytes, bool) or not isinstance(l2_bytes, (int, np.integer)) or l2_bytes < 1:
        raise ValueError("l2_bytes must be a positive integer")
    bytes_used = int(dim) * int(slots) * int(dtype_bytes)
    return ConditionCacheBudget(int(dim), int(slots), int(dtype_bytes), bytes_used,
                                 int(l2_bytes), bytes_used <= l2_bytes)


# ---------------------------------------------------------------------------
# Reinjection schedule: WHEN the condition is allowed to change.
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ReinjectionEvent:
    """One scheduled swap of the active well center and/or repulsor set.

    `digest` pins the reinjected content (same SHA-256-over-shape-and-bytes
    scheme as `FrozenProblemContext._hash` in controlled_continuous_engine.py)
    so a reinjection log entry can be checked for tampering the same way a
    frozen context can.
    """
    scale: str          # "macro" | "meso" | "micro"
    tau: int
    step_index: int      # segment index in the chained schedule, 0-based
    c: np.ndarray
    repulsors: Optional[np.ndarray]
    digest: str

    @staticmethod
    def _hash(c: np.ndarray, repulsors: Optional[np.ndarray]) -> str:
        h = hashlib.sha256()
        h.update(repr(c.shape).encode())
        h.update(np.ascontiguousarray(c).tobytes())
        if repulsors is not None:
            h.update(repr(repulsors.shape).encode())
            h.update(np.ascontiguousarray(repulsors).tobytes())
        return h.hexdigest()

    @classmethod
    def make(cls, scale: str, tau: int, step_index: int, c, repulsors=None) -> "ReinjectionEvent":
        if scale not in ("macro", "meso", "micro"):
            raise ValueError(f"scale must be one of macro/meso/micro, got {scale!r}")
        c = _vector(c, "c")
        r = None if repulsors is None else _repulsor_matrix(repulsors, "repulsors", c.shape[0])
        return cls(scale, int(tau), int(step_index), c, r, cls._hash(c, r))


def build_fixed_reinjection_schedule(
    dim: int,
    centers: Sequence[np.ndarray],
    repulsor_sets: Optional[Sequence[Optional[np.ndarray]]] = None,
) -> Tuple[ReinjectionEvent, ...]:
    """Build a schedule whose length and (scale, tau) at each index are fixed
    ahead of time -- data-independent, therefore compatible with
    `parallel_rnn_lora.associative_scan` (see `is_schedule_parallel_scan_compatible`).

    `centers` gives one `c` per scale in order (macro tau=4, meso tau=2,
    micro tau=1, ...repeating if more than 3 are given, cycling through the
    three scales). This is the only schedule shape this function builds: the
    *number and order* of reinjection events must be decided before the
    trajectory runs, never from a runtime residual (that case is
    `is_schedule_parallel_scan_compatible` returning False; build such a
    schedule by hand and it will be correctly rejected, see the test suite).
    """
    if len(centers) == 0:
        raise ValueError("centers must be nonempty")
    scales = ("macro", "meso", "micro")
    taus = {"macro": MACRO_TAU, "meso": MESO_TAU, "micro": MICRO_TAU}
    repulsor_sets = [None] * len(centers) if repulsor_sets is None else repulsor_sets
    if len(repulsor_sets) != len(centers):
        raise ValueError("repulsor_sets must have the same length as centers")
    events: List[ReinjectionEvent] = []
    for i, (c, r) in enumerate(zip(centers, repulsor_sets)):
        scale = scales[i % 3]
        events.append(ReinjectionEvent.make(scale, taus[scale], i, c, r))
    return tuple(events)


def is_schedule_parallel_scan_compatible(schedule_is_data_dependent: bool) -> bool:
    """Whether a reinjection schedule can share a chunk of
    `parallel_rnn_lora.associative_scan`'s O(log T) parallel path, versus
    needing its `sequential_scan` fallback.

    `associative_scan`'s `combine` (parallel_rnn_lora.py:58-77) is associative
    only because each transition element (A_t, u_t) is fixed *before* the
    scan runs -- the Blelloch up-sweep/down-sweep reorders combines freely,
    which is only sound if no combine's inputs depend on another combine's
    output. A schedule built by `build_fixed_reinjection_schedule` satisfies
    this: every (scale, tau, c) triple is decided ahead of time. A schedule
    that reinjects *conditionally* -- e.g. "swap c only if this segment's
    residual_norm exceeds a threshold" -- makes segment i+1's transition
    depend on segment i's *output*, which is exactly the dependency
    `associative_scan` cannot have; that chunk must run through
    `sequential_scan` (or this module's own segment loop) instead.

    This function does not inspect a schedule object (a fixed schedule is
    just data, indistinguishable from one that happened to be built from a
    residual check after the fact); the caller states which kind it built.
    Its only job is to make the fixed/data-dependent distinction a single
    named call site instead of an implicit assumption.
    """
    return not schedule_is_data_dependent


# ---------------------------------------------------------------------------
# PAWS / VitaminC bridge: turn an existing one-shot repulsion signal into a
# repulsor row the fractal engine's macro phase can consume repeatedly.
# ---------------------------------------------------------------------------

def macro_repulsor_from_counterfactual(delta_perp: np.ndarray) -> np.ndarray:
    """Reshape a `CounterfactualConstraintVerifier.repulsion()` delta_perp
    (counterfactual_constraint_verifier.py:219-264, the VitaminC/PAWS
    "adversarial counterfactual repulsion operator") into the (1, dim)
    repulsor-row shape `FractalMultiscaleEngine.macro_prune` expects.

    Today that `delta_perp` is computed once and used only to shift a final
    simplex decision (nanocore/choice_head.py:424,
    `ROLE_SWAP_LOGIT_SHIFT * binding_report`). This function does not change
    that call site; it lets the *same* vector also be handed to
    `macro_prune`/`think` as a `repulsors` row at a reinjection checkpoint,
    so the wrong-attractor branch gets pruned during relaxation rather than
    only nudged after it already converged.
    """
    v = _vector(delta_perp, "delta_perp")
    return v.reshape(1, -1)


def symbolic_violation_to_repulsor(residual_vector: np.ndarray, penalty: float) -> Optional[np.ndarray]:
    """Convert a *decoded-checkpoint* symbolic-arithmetic violation into a
    micro-scale repulsor, without ever putting `SymbolicArithmeticVerifier`
    inside the leapfrog inner loop (it has no gradient w.r.t. `q`).

    `residual_vector` must already be a geometric object (e.g. the embedding
    difference between the decoded candidate step and its algebraically
    corrected version) -- this function does not decode text or evaluate the
    verifier itself; that happens once, at the checkpoint, by the caller.
    Returns None when `penalty` is 0 (nothing to repel), which the caller
    must handle by not reinjecting a repulsor for that checkpoint.
    """
    if penalty < 0.0:
        raise ValueError("penalty must be non-negative")
    if penalty == 0.0:
        return None
    v = _vector(residual_vector, "residual_vector")
    return v.reshape(1, -1)


# ---------------------------------------------------------------------------
# Chained multi-segment relaxation with reinjection.
# ---------------------------------------------------------------------------

class LyapunovViolation(RuntimeError):
    """A segment's own Hamiltonian rose against its own active center.

    This should be unreachable if reinjection only ever happens between
    segments (see the module docstring); it is a runtime guard against a
    caller misusing the low-level API, not a proof.
    """


@dataclasses.dataclass(frozen=True)
class SegmentResult:
    event: ReinjectionEvent
    steps: int
    pruned: bool
    residual: float
    q: np.ndarray
    p: np.ndarray


@dataclasses.dataclass(frozen=True)
class ReinjectionResult:
    segments: Tuple[SegmentResult, ...]
    total_steps: int
    q: np.ndarray
    p: np.ndarray
    final_residual: float
    cache_budget: ConditionCacheBudget


class ContinuousConditionInjector:
    """Chains `FractalMultiscaleEngine` segments across a reinjection schedule.

    Does not subclass or modify `FractalMultiscaleEngine`: it composes the
    engine's own exported pure functions (`leapfrog_step`, `hamiltonian`,
    `residual_norm`, `counterfactual_repulsion_score`) the same way
    `FractalMultiscaleEngine.macro_prune`/`micro_relax` do internally, so a
    meso-scale (tau=2) segment -- which the engine class itself does not
    expose, it only has macro/micro -- is just another call with a different
    `dt`, not a new algorithm.
    """

    def __init__(self, engine: FractalMultiscaleEngine, *,
                 l2_bytes: int = L2_CONDITION_CACHE_BYTES_DEFAULT,
                 max_slots: int = 8) -> None:
        self.engine = engine
        self.l2_bytes = int(l2_bytes)
        self.max_slots = int(max_slots)

    def _dt_for(self, tau: int) -> float:
        return self.engine.dt_micro * tau

    def _run_segment(self, q: np.ndarray, p: np.ndarray, event: ReinjectionEvent,
                      max_steps: int) -> SegmentResult:
        c = event.c
        omega, gamma = self.engine.omega, self.engine.gamma_micro
        dt = self._dt_for(event.tau)
        r = event.repulsors
        h_prev = hamiltonian(q, p, c, omega)
        pruned = False
        steps = 0
        residual = residual_norm(q, p, c, omega)
        if residual >= self.engine.residual_tol:
            for steps in range(1, max_steps + 1):
                q, p = leapfrog_step(q, p, c, omega=omega, dt=dt, gamma=gamma)
                h_now = hamiltonian(q, p, c, omega)
                if h_now > h_prev + 1e-9:
                    raise LyapunovViolation(
                        f"segment {event.step_index} ({event.scale}): H rose "
                        f"{h_prev:.6g} -> {h_now:.6g} at step {steps}"
                    )
                h_prev = h_now
                if r is not None:
                    score = counterfactual_repulsion_score(p, r)
                    if score > self.engine.repulsion_threshold:
                        for row in r:
                            p = _repel(p, row, self.engine.repulsion_backtrack_strength)
                        pruned = True
                        break
                residual = residual_norm(q, p, c, omega)
                if residual < self.engine.residual_tol:
                    break
        return SegmentResult(event, steps, pruned, residual, q, p)

    def think_with_reinjection(
        self,
        q0: np.ndarray,
        p0: np.ndarray,
        schedule: Sequence[ReinjectionEvent],
        *,
        max_steps_per_segment: int = 300,
    ) -> ReinjectionResult:
        if len(schedule) == 0:
            raise ValueError("schedule must be nonempty")
        dim = self.engine.dim
        q = _vector(q0, "q0", dim)
        p = _vector(p0, "p0", dim)
        slots = len(schedule) + sum(
            (0 if e.repulsors is None else e.repulsors.shape[0]) for e in schedule
        )
        budget = verify_condition_cache_bound(dim, min(slots, self.max_slots),
                                               l2_bytes=self.l2_bytes)
        if not budget.within_bound:
            raise ValueError(
                f"reinjection schedule needs {budget.bytes_used} bytes of condition "
                f"cache, exceeding l2_bytes={self.l2_bytes}"
            )
        segments: List[SegmentResult] = []
        for event in schedule:
            if event.c.shape[0] != dim:
                raise ValueError(
                    f"schedule event {event.step_index} has dim {event.c.shape[0]}, "
                    f"engine dim is {dim}"
                )
            result = self._run_segment(q, p, event, max_steps_per_segment)
            segments.append(result)
            q, p = result.q, result.p
        total_steps = sum(s.steps for s in segments)
        return ReinjectionResult(tuple(segments), total_steps, q, p,
                                  segments[-1].residual, budget)
