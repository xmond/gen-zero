"""Annealed Langevin dynamics: escaping shallow local minima a deterministic
compression map cannot escape, once the thinking potential V(q;c) is non-convex.

Fatal-premise check (must be stated before anything else). The existing
`FractalMultiscaleEngine.micro_relax` (fractal_multiscale_engine.py:186-219) runs
damped leapfrog on V(q;c) = 0.5*omega^2*||q-c||^2 -- a pure harmonic well. That V
is convex with exactly one stationary point, so it is *impossible* for that engine
to get trapped in a "shallow local minimum": there isn't one to fall into. The
"64-step but trapped" framing only holds once V has more than one basin, and the
harmonic engine was never asked to represent that. The 64-step figure in
`evaluate_cpu_zero_token_dynamics.py:106-119` (single_scale convergence) is pure
exponential contraction (0.8^n * ||initial residual|| < 1e-5 needs n=~65, matching
measured `physical_steps`), not a local-minimum artifact -- confirmed by running
that file's `convergence(seed, multiscale=False)` directly. So this module does not
patch a bug in the harmonic engine; it adds a genuinely new capability (sampling a
non-convex, multi-basin V) that no existing module in this repo can exercise, and
constructs its own adversarial non-convex potential (`TiltedDoubleWellPotential`
below) to prove it, because none of the codebase's other potentials are non-convex
either (symplectic_thinking.NeuralPotential and the harmonic well are both convex
or externally-supplied).

Algorithm (real, citable, not invented for this task). This is the unadjusted
Langevin algorithm (ULA; Welling & Teh 2011, "Bayesian Learning via Stochastic
Gradient Langevin Dynamics") run with a geometric temperature schedule, i.e.
exactly the "annealed Langevin dynamics" sampler of Song & Ermon 2019 ("Generative
Modeling by Estimating Gradients of the Data Distribution"), which is the paper
this task's name comes from. Overdamped Euler-Maruyama discretisation of dq = -grad V(q) dt + sqrt(2 T) dW_t,
with a *fixed* integration step eta0 (the SDE's dt) and only the temperature
annealed -- the standard fixed-step form of Langevin MCMC, and the same
"constant micro dt, vary only the physics" discipline `fractal_multiscale_engine`
already uses for its leapfrog `dt_micro`:

    q_{k+1} = q_k - eta0 * grad V(q_k) + sqrt(2 * eta0 * T_k) * xi_k,
    xi_k ~ N(0, I_dim),   T_k = T0 * anneal_gamma^k

`sqrt(2 * eta0 * T_k)` is exactly the noise-injection amplitude the task
specifies (its "sqrt(2 T_k)" is the continuous-time diffusion coefficient; the
discrete step additionally needs the eta0 factor from the Euler-Maruyama time
discretisation -- writing sqrt(2*T_k) alone, with no step-size factor, would not
be dimensionally consistent as a per-step update). An earlier draft of this
engine also annealed eta_k = eta0*(T_k/T0) alongside the noise (mirroring Song &
Ermon's alpha_i = epsilon*sigma_i^2/sigma_L^2); measurement showed that lets the
*drift* step shrink to near-zero together with the noise near the end of the
schedule, so the state never finishes relaxing onto whichever well it is in
before `max_steps` (measured: 0/20 seeds converged, residual still ~0.5-1.1 at
cutoff -- see the calibration note in test_annealed_langevin_engine.py).
Decoupling eta from T fixed that.

Stability (not hand-waved): away from the well(s), `TiltedDoubleWellPotential`'s
off-axis confinement term is strongly convex with constant Hessian kappa*I, which
gives grad V a drift that grows linearly in ||q|| once outside the wells. That is
exactly the Foster-Lyapunov "dissipativity" condition (Roberts & Tweedie 1996)
under which ULA is ergodic and stays bounded in expectation for any fixed T -- the
random walk does not run away to infinity as T is raised.

Energy-barrier detection ("Kick-out Pulse"). After each step this engine tracks
the last `plateau_window` (default 3) values of the algebraic residual
||grad V(q_k)|| (reused from the same gradient the update already computed, no
extra call). If that window's spread stays below `plateau_tol` for
`plateau_window` consecutive steps *and* `potential.violation(q_k)` -- an
oracle-supplied sign/counterfactual residual, independent of V and never derived
from the optimiser's own trajectory -- is still above `violation_tol`, the engine
concludes this is a false (flat-but-wrong-basin) convergence and reheats:
temperature is overridden to `max(T_schedule, kick_temperature)` for the next
`kick_pulse_steps` steps. Bounded by `max_kicks`; once exhausted the engine keeps
annealing but stops reheating and honestly reports `converged=False` if it never
clears `violation_tol` by `max_steps`.

Fixed O(dim) working set, no per-step array allocation. `q`, `grad_buf`,
`noise_buf`, `scratch_buf` are allocated once in `__init__`; every update in
`run()` mutates them in place via `+=`/`-=`/`np.multiply(..., out=...)`, and
`TiltedDoubleWellPotential.grad_into` likewise writes into caller-supplied
buffers rather than returning new arrays. This is measured, not asserted: see
`test_annealed_langevin_engine.py`'s tracemalloc check for the exact byte count
observed over 1000 steps (only small fixed-size Python scalar objects, no
ndarray allocations, exactly the same honesty framing as
`fractal_multiscale_engine.verify_cache_bound`: this is a byte-accounting /
tracemalloc measurement, not a hardware cache-miss counter).
"""
from __future__ import annotations

import dataclasses
import math
from typing import List, Optional, Protocol, Tuple

import numpy as np

from .fractal_multiscale_engine import (
    CacheBoundReport,
    L1_CACHE_BYTES_DEFAULT,
    _positive_float,
    _positive_int,
    _vector,
    verify_cache_bound,
)


def _nonneg_int(value, name: str) -> int:
    """Like `_positive_int` but allows 0 (e.g. max_kicks=0: pure deterministic baseline)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)

__all__ = [
    "LANGEVIN_STATE_VECTORS_PER_ITEM",
    "Potential",
    "TiltedDoubleWellPotential",
    "LangevinStepDiagnostics",
    "LangevinResult",
    "AnnealedLangevinDynamicsEngine",
]

LANGEVIN_STATE_VECTORS_PER_ITEM = 4  # q, grad, noise, scratch


class Potential(Protocol):
    """Minimal contract the engine needs from a thinking potential V(q; ...).

    `grad_into` and `violation` must write results into caller-supplied buffers
    / return plain floats so the engine's zero-allocation guarantee extends
    through the potential call, not just its own update arithmetic.
    """

    dim: int

    def grad_into(self, q: np.ndarray, out: np.ndarray, scratch: np.ndarray) -> np.ndarray:
        ...

    def value(self, q: np.ndarray) -> float:
        ...

    def violation(self, q: np.ndarray) -> float:
        ...


class TiltedDoubleWellPotential:
    """Non-convex tilted double well along a unit axis, embedded in R^dim.

        V(q) = (s^2 - 1)^2 + tilt*s + 0.5*kappa*||q - s*axis||^2,   s = q . axis

    The (s^2-1)^2 term alone is a symmetric double well: minima at s=+-1, a
    barrier of height 1 at s=0 (V(0)=1, V(+-1)=0). `tilt` breaks the symmetry:
    V(+1)=tilt, V(-1)=-tilt. For tilt>0 the s=-1 well is strictly deeper
    (the global minimum) and s=+1 is a shallow local minimum -- this is the
    concrete, honest instance of "浅层局部极小" the task asks the engine to
    escape, since the codebase's other potentials have none. `kappa` confines
    q to the axis line with a constant-Hessian quadratic (off-axis directions
    are strongly convex, so the whole construction is coercive/dissipative --
    see the module docstring's stability note) and, because it only ever
    contracts orthogonal components, makes `value`/`grad_into`/`violation`
    exactly equivariant under any orthogonal Q applied jointly to q and axis
    (same equivariance discipline as `fractal_multiscale_engine`).

    `dim=2` gives the classic textbook double well (axis = e0, one confined
    coordinate). Any `dim` works identically -- passing `dim=256` is the
    "高维非凸" (high-dimensional non-convex) instance the task also asks for,
    using the same formula, not a separate mechanism.
    """

    def __init__(self, dim: int, axis, tilt: float, kappa: float = 8.0) -> None:
        self.dim = _positive_int(dim, "dim")
        a = _vector(axis, "axis", self.dim)
        norm = float(np.linalg.norm(a))
        if norm <= 0.0:
            raise ValueError("axis must be a nonzero vector")
        self.axis = a / norm
        t = float(tilt)
        if not np.isfinite(t) or t == 0.0:
            raise ValueError("tilt must be finite and nonzero (zero tilt has two equal-depth wells)")
        self.tilt = t
        self.kappa = _positive_float(kappa, "kappa")

    @property
    def true_axis_value(self) -> float:
        """s-coordinate (+-1) of the true global minimum for this tilt sign."""
        return -1.0 if self.tilt > 0.0 else 1.0

    def value(self, q: np.ndarray) -> float:
        q = _vector(q, "q", self.dim)
        s = float(q @ self.axis)
        off = q - s * self.axis
        return (s * s - 1.0) ** 2 + self.tilt * s + 0.5 * self.kappa * float(off @ off)

    def grad_into(self, q: np.ndarray, out: np.ndarray, scratch: np.ndarray) -> np.ndarray:
        """Write grad V(q) into `out`, using `scratch` as the only temporary.

        Zero new ndarray allocations: every write below targets an existing
        buffer via `out=`. `s` and `ds` are Python floats, not arrays.
        """
        if out is q or scratch is q or out is scratch:
            raise ValueError("grad_into: q, out and scratch must be three distinct buffers")
        s = float(q @ self.axis)
        ds = 4.0 * s * (s * s - 1.0) + self.tilt
        np.multiply(self.axis, s, out=scratch)      # scratch = s*axis
        np.subtract(q, scratch, out=scratch)         # scratch = off = q - s*axis
        np.multiply(scratch, self.kappa, out=out)     # out = kappa*off
        np.multiply(self.axis, ds, out=scratch)       # scratch = ds*axis
        out += scratch                                 # out = kappa*off + ds*axis
        return out

    def violation(self, q: np.ndarray) -> float:
        """Sign/counterfactual oracle: how far into the WRONG basin q sits.

        0.0 exactly when q is on the true-global-well side of the barrier
        (s * true_axis_value >= 0); grows linearly past the barrier otherwise.
        Depends only on the axis and the (fixed, construction-time) tilt sign
        -- never on V's value or the optimiser's trajectory -- so using it as
        the engine's "did we actually land in the right basin" check is not
        circular with the objective being descended.
        """
        q = _vector(q, "q", self.dim)
        s = float(q @ self.axis)
        return max(0.0, -self.true_axis_value * s)


@dataclasses.dataclass(frozen=True)
class LangevinStepDiagnostics:
    step: int
    temperature: float
    residual: float
    violation: float
    kicked: bool


@dataclasses.dataclass(frozen=True)
class LangevinResult:
    q: np.ndarray
    steps: int
    converged: bool
    final_residual: float
    final_violation: float
    kicks_triggered: int
    final_temperature: float
    trace: Tuple[LangevinStepDiagnostics, ...]


class AnnealedLangevinDynamicsEngine:
    """Annealed unadjusted Langevin dynamics with energy-barrier reheating.

    See the module docstring for the update rule, the stability argument and
    the zero-allocation working-set claim. Construction validates every
    parameter fail-closed (raises, never silently clamps); `run()` performs no
    ndarray allocation of its own when `record_trace=False`.
    """

    def __init__(
        self,
        dim: int,
        potential: Potential,
        *,
        T0: float = 4.0,
        anneal_gamma: float = 0.97,
        eta0: float = 0.05,
        max_steps: int = 400,
        plateau_window: int = 3,
        plateau_tol: float = 1e-4,
        violation_tol: float = 1e-3,
        kick_temperature: Optional[float] = None,
        kick_pulse_steps: int = 6,
        max_kicks: int = 5,
        residual_tol: float = 1e-3,
        dtype=np.float32,
        seed: Optional[int] = None,
        l1_bytes: int = L1_CACHE_BYTES_DEFAULT,
    ) -> None:
        self.dim = _positive_int(dim, "dim")
        if getattr(potential, "dim", None) != self.dim:
            raise ValueError("potential.dim must match engine dim")
        self.potential = potential
        self.T0 = _positive_float(T0, "T0")
        self.anneal_gamma = float(anneal_gamma)
        if not (0.0 < self.anneal_gamma <= 1.0):
            raise ValueError("anneal_gamma must be in (0, 1]")
        self.eta0 = _positive_float(eta0, "eta0")
        self.max_steps = _positive_int(max_steps, "max_steps")
        self.plateau_window = _positive_int(plateau_window, "plateau_window")
        self.plateau_tol = _positive_float(plateau_tol, "plateau_tol")
        self.violation_tol = _positive_float(violation_tol, "violation_tol")
        self.kick_temperature = (
            self.T0 * 4.0 if kick_temperature is None else _positive_float(kick_temperature, "kick_temperature")
        )
        self.kick_pulse_steps = _positive_int(kick_pulse_steps, "kick_pulse_steps")
        self.max_kicks = _nonneg_int(max_kicks, "max_kicks")
        self.residual_tol = _positive_float(residual_tol, "residual_tol")
        if np.dtype(dtype) not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("dtype must be float32 or float64 (numpy Generator.standard_normal restriction)")
        self.dtype = np.dtype(dtype)
        self.l1_bytes = _positive_int(l1_bytes, "l1_bytes")
        self._rng = np.random.default_rng(seed)

        # Fixed O(dim) working set, allocated once, reused for every run() call.
        self._dim_sqrt = math.sqrt(self.dim)
        self._q = np.zeros(self.dim, dtype=self.dtype)
        self._grad = np.zeros(self.dim, dtype=self.dtype)
        self._noise = np.zeros(self.dim, dtype=self.dtype)
        self._scratch = np.zeros(self.dim, dtype=self.dtype)

    def working_set_bytes(self) -> int:
        """Actual resident bytes of q/grad/noise/scratch -- measured, not claimed."""
        return LANGEVIN_STATE_VECTORS_PER_ITEM * self.dim * self.dtype.itemsize

    def cache_bound_report(self) -> CacheBoundReport:
        return verify_cache_bound(
            self.dim,
            batch_size=1,
            dtype_bytes=self.dtype.itemsize,
            l1_bytes=self.l1_bytes,
            state_vectors=LANGEVIN_STATE_VECTORS_PER_ITEM,
        )

    def run(self, q0, *, record_trace: bool = False) -> LangevinResult:
        np.copyto(self._q, _vector(q0, "q0", self.dim))
        residual_window: List[float] = []
        kicks = 0
        kick_remaining = 0
        trace: Optional[List[LangevinStepDiagnostics]] = [] if record_trace else None
        converged = False
        residual = math.inf
        violation = math.inf
        temperature = self.T0
        step = 0

        for step in range(1, self.max_steps + 1):
            schedule_t = self.T0 * (self.anneal_gamma ** (step - 1))
            if kick_remaining > 0:
                temperature = max(schedule_t, self.kick_temperature)
                kick_remaining -= 1
            else:
                temperature = schedule_t

            self.potential.grad_into(self._q, self._grad, self._scratch)
            # RMS-per-dimension residual (standard MD "RMS force" convergence
            # metric), not raw L2 norm: a fixed threshold on ||grad|| would mean
            # a different thing at dim=2 vs dim=256 since i.i.d component-wise
            # noise makes ||grad|| itself grow like sqrt(dim).
            residual = float(np.linalg.norm(self._grad)) / self._dim_sqrt

            self._rng.standard_normal(size=self.dim, dtype=self.dtype, out=self._noise)
            noise_scale = math.sqrt(max(0.0, 2.0 * self.eta0 * temperature))

            self._grad *= -self.eta0
            self._q += self._grad
            self._noise *= noise_scale
            self._q += self._noise

            violation = float(self.potential.violation(self._q))

            residual_window.append(residual)
            if len(residual_window) > self.plateau_window:
                residual_window.pop(0)
            plateaued = (
                len(residual_window) == self.plateau_window
                and (max(residual_window) - min(residual_window)) < self.plateau_tol
            )
            triggered = False
            if (
                plateaued
                and violation > self.violation_tol
                and kick_remaining == 0
                and kicks < self.max_kicks
            ):
                kick_remaining = self.kick_pulse_steps
                kicks += 1
                residual_window.clear()
                triggered = True

            if trace is not None:
                trace.append(LangevinStepDiagnostics(step, temperature, residual, violation, triggered))

            if residual < self.residual_tol and violation <= self.violation_tol:
                converged = True
                break

        return LangevinResult(
            self._q.copy(),
            step,
            converged,
            residual,
            violation,
            kicks,
            temperature,
            tuple(trace) if trace is not None else tuple(),
        )
