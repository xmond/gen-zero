"""Symplectic Thinking: Leapfrog latent rollouts with Lyapunov early exit.

"Thinking" is modelled as Hamiltonian flow on a latent state (q, p), q and p both in R^2048:

    H(q, p) = 0.5 * ||p||^2 + V_theta(q)
    dq/dt =  p
    dp/dt = -grad V_theta(q) - gamma * p

1. Neural potential: V_theta(q) = 0.5 * omega^2 * ||q||^2 + w2 . softplus(W1 q + b1).
   The harmonic term makes V coercive (bounded below), so H is a valid Lyapunov candidate.
   grad V is analytic (no autograd), so a step costs two matvecs of size hidden x dim.
2. Leapfrog / Stormer-Verlet: kick (dt/2), drift (dt), kick (dt/2). It is symplectic and
   time-reversible. With gamma = 0 the energy error stays bounded at O(dt^2).
3. Friction (gamma > 0) is applied as an exact exponential half-step on p on each side of the
   leapfrog (conformal splitting). It keeps the map conformally symplectic and makes
   dH/dt = -gamma * ||p||^2 <= 0, so H acts as a Lyapunov function.
4. Lyapunov early exit: stop when the trajectory has settled instead of burning the full
   step budget. Exit reasons are listed in `ExitReason`.

Pure NumPy. No file-system paths, no network, no global state.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

LATENT_DIM = 2048


class ExitReason(str, enum.Enum):
    """Why a `think` rollout stopped."""
    Converged = "Converged"
    Stalled = "Stalled"
    LyapunovViolation = "LyapunovViolation"
    NonFinite = "NonFinite"
    MaxSteps = "MaxSteps"


class NeuralPotential:
    """Scalar potential V_theta(q) with an analytic gradient."""

    def __init__(
        self,
        dim: int = LATENT_DIM,
        hidden_dim: int = 256,
        omega: float = 1.0,
        scale: float = 0.1,
        seed: Optional[int] = 0,
    ) -> None:
        if dim <= 0 or hidden_dim <= 0:
            raise ValueError("dim and hidden_dim must be positive")
        if omega <= 0.0:
            raise ValueError("omega must be positive so that V stays bounded below")
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.omega = float(omega)

        rng = np.random.default_rng(seed)
        limit = math.sqrt(6.0 / (dim + hidden_dim))
        self.w1 = rng.uniform(-limit, limit, (hidden_dim, dim))
        self.b1 = np.zeros(hidden_dim)
        # Small output weights keep the harmonic well dominant: one clear basin to settle into.
        self.w2 = rng.uniform(-limit, limit, hidden_dim) * scale

    def energy_and_grad(self, q: np.ndarray) -> Tuple[float, np.ndarray]:
        """Return V(q) and dV/dq.

        dV/dq = omega^2 * q + W1^T (sigmoid(W1 q + b1) * w2)
        """
        h = self.w1 @ q + self.b1
        # Stable softplus: max(h, 0) + log1p(exp(-|h|)); its derivative is sigmoid(h).
        softplus = np.maximum(h, 0.0) + np.log1p(np.exp(-np.abs(h)))
        sigmoid = 0.5 * (1.0 + np.tanh(0.5 * h))
        energy = 0.5 * self.omega ** 2 * float(q @ q) + float(self.w2 @ softplus)
        grad = self.omega ** 2 * q + self.w1.T @ (sigmoid * self.w2)
        return energy, grad

    def energy(self, q: np.ndarray) -> float:
        return self.energy_and_grad(q)[0]

    def grad(self, q: np.ndarray) -> np.ndarray:
        return self.energy_and_grad(q)[1]


@dataclasses.dataclass
class ThinkingResult:
    """Outcome of one `SymplecticThinker.think` rollout."""
    q: np.ndarray
    p: np.ndarray
    steps: int
    max_steps: int
    exit_reason: ExitReason
    initial_energy: float
    final_energy: float
    energy_trace: List[float]

    @property
    def early_exit(self) -> bool:
        return self.exit_reason is not ExitReason.MaxSteps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "max_steps": self.max_steps,
            "exit_reason": self.exit_reason.value,
            "early_exit": self.early_exit,
            "initial_energy": self.initial_energy,
            "final_energy": self.final_energy,
        }


class SymplecticThinker:
    """Leapfrog integrator over (q, p) with Lyapunov-based early exit."""

    def __init__(
        self,
        potential: Optional[NeuralPotential] = None,
        dt: float = 0.05,
        gamma: float = 0.5,
        max_steps: int = 200,
        grad_tol: float = 1e-4,
        momentum_tol: float = 1e-4,
        stall_tol: float = 1e-9,
        patience: int = 5,
        lyapunov_slack: float = 1e-6,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if gamma < 0.0:
            raise ValueError("gamma must be >= 0")
        if max_steps < 1 or patience < 1:
            raise ValueError("max_steps and patience must be >= 1")
        self.potential = potential if potential is not None else NeuralPotential()
        self.dim = self.potential.dim
        self.dt = float(dt)
        self.gamma = float(gamma)
        self.max_steps = int(max_steps)
        self.grad_tol = float(grad_tol)
        self.momentum_tol = float(momentum_tol)
        self.stall_tol = float(stall_tol)
        self.patience = int(patience)
        self.lyapunov_slack = float(lyapunov_slack)

    # ---- state helpers -------------------------------------------------

    def _check(self, name: str, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.dim,):
            raise ValueError(f"{name} must have shape ({self.dim},), got {x.shape}")
        return x

    def hamiltonian(self, q: np.ndarray, p: np.ndarray) -> float:
        """H(q, p) = 0.5 ||p||^2 + V(q)."""
        q = self._check("q", q)
        p = self._check("p", p)
        return 0.5 * float(p @ p) + self.potential.energy(q)

    # ---- integrator ----------------------------------------------------

    def step(
        self,
        q: np.ndarray,
        p: np.ndarray,
        grad_q: Optional[np.ndarray] = None,
        gamma: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One damped leapfrog step. Returns (q_next, p_next, grad V at q_next).

        Passing `grad_q` (grad V at q) reuses the last force evaluation, so a rollout
        costs one gradient per step, not two.
        """
        q = self._check("q", q)
        p = self._check("p", p)
        g = self.potential.grad(q) if grad_q is None else grad_q
        damp = math.exp(-(self.gamma if gamma is None else gamma) * self.dt * 0.5)
        half = 0.5 * self.dt

        p = damp * p
        p = p - half * g
        q_next = q + self.dt * p
        g_next = self.potential.grad(q_next)
        p = p - half * g_next
        p = damp * p
        return q_next, p, g_next

    def think(self, q0: np.ndarray, p0: Optional[np.ndarray] = None) -> ThinkingResult:
        """Roll the latent state forward until a Lyapunov exit fires or the budget ends.

        Exit checks, in order, after each step:
          NonFinite          state or energy is NaN/inf.
          LyapunovViolation  H rose by more than `lyapunov_slack` (relative). Only checked when
                             gamma > 0, since undamped H is only conserved up to O(dt^2).
          Converged          ||grad V|| and ||p|| both below tolerance: an equilibrium.
          Stalled            |dH| <= stall_tol * max(1, |H|) for `patience` steps in a row.
        """
        q = self._check("q0", q0).copy()
        p = np.zeros(self.dim) if p0 is None else self._check("p0", p0).copy()

        grad = self.potential.grad(q)
        energy = 0.5 * float(p @ p) + self.potential.energy(q)
        initial_energy = energy
        trace = [energy]
        stalled = 0
        reason = ExitReason.MaxSteps
        steps = 0

        if not (np.isfinite(energy) and np.all(np.isfinite(q)) and np.all(np.isfinite(p))):
            return ThinkingResult(q, p, 0, self.max_steps, ExitReason.NonFinite,
                                  initial_energy, energy, trace)

        for steps in range(1, self.max_steps + 1):
            q, p, grad = self.step(q, p, grad_q=grad)
            new_energy = 0.5 * float(p @ p) + self.potential.energy(q)
            trace.append(new_energy)

            if not (np.isfinite(new_energy) and np.all(np.isfinite(q)) and np.all(np.isfinite(p))):
                reason, energy = ExitReason.NonFinite, new_energy
                break

            delta = new_energy - energy
            scale = max(1.0, abs(energy))
            energy = new_energy

            if self.gamma > 0.0 and delta > self.lyapunov_slack * scale:
                reason = ExitReason.LyapunovViolation
                break
            if math.sqrt(float(grad @ grad)) <= self.grad_tol and \
                    math.sqrt(float(p @ p)) <= self.momentum_tol:
                reason = ExitReason.Converged
                break
            stalled = stalled + 1 if abs(delta) <= self.stall_tol * scale else 0
            if stalled >= self.patience:
                reason = ExitReason.Stalled
                break

        return ThinkingResult(q, p, steps, self.max_steps, reason, initial_energy, energy, trace)
