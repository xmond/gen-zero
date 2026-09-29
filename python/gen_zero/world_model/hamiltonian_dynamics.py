"""Gen-Zero World Model: Hamiltonian Dynamics & Symplectic Integrator.

RFC-086 / Issue #86 Implementation:
Structure-Preserving Continuous Latent World Model via Symplectic Mechanics:
1. Canonical Symplectic Coordinate Decomposition:
   Splits latent state z in R^D (D even) into canonical conjugate pairs:
       z = [q^T, p^T]^T
   where q in R^(D/2) represents generalized semantic coordinates and p in R^(D/2)
   represents generalized latent momentum (rate of semantic change).
2. Scalar Hamiltonian Energy Formulation:
   Instead of black-box vector residuals, the network parameterizes the scalar total energy
   H_theta(q, p) = T(p) + V_theta(q). Continuous dynamics follow Hamilton's canonical equations:
       dq/dt =  dH/dp
       dp/dt = -dH/dq + F_ext(a_t)
   where F_ext(a_t) is the generalized control force exerted by action a_t.
3. Störmer-Verlet Symplectic Numerical Integration:
   Integrates the system using explicit symplectic leapfrog:
       p_{t + 1/2} = p_t - (dt/2) * dV/dq(q_t) + (dt/2) * F(a_t)
       q_{t + 1}   = q_t + dt * p_{t + 1/2}
       p_{t + 1}   = p_{t + 1/2} - (dt/2) * dV/dq(q_{t+1}) + (dt/2) * F(a_t)
   Guarantees exact phase-space volume preservation:
       det( d(q_{t+1}, p_{t+1}) / d(q_t, p_t) ) = 1.0
   and bounds energy drift to O(dt^2) over arbitrary long-horizon imagination rollouts (H >= 100),
   completely eliminating phase space divergence.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = object
    HAS_TORCH = False


@dataclass
class HamiltonianStepResult:
    """Output of a single symplectic integration step."""
    next_state: np.ndarray
    q_next: np.ndarray
    p_next: np.ndarray
    hamiltonian_energy: float
    kinetic_energy: float
    potential_energy: float
    energy_drift_ratio: float
    step_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hamiltonian_energy": round(self.hamiltonian_energy, 6),
            "kinetic_energy": round(self.kinetic_energy, 6),
            "potential_energy": round(self.potential_energy, 6),
            "energy_drift_ratio": round(self.energy_drift_ratio, 6),
            "step_latency_ms": round(self.step_latency_ms, 4),
            "state_norm": round(float(np.linalg.norm(self.next_state)), 6),
        }


@dataclass
class HamiltonianRolloutResult:
    """Long-horizon imagination rollout audit report."""
    horizon: int
    states: List[np.ndarray]
    energies: List[float]
    initial_energy: float
    final_energy: float
    max_energy_drift_ratio: float
    mean_energy_drift_ratio: float
    is_stable: bool
    state_norms: List[float]
    max_state_norm: float
    divergence_rate: float
    avg_step_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon": self.horizon,
            "initial_energy": round(self.initial_energy, 6),
            "final_energy": round(self.final_energy, 6),
            "max_energy_drift_ratio": round(self.max_energy_drift_ratio, 6),
            "mean_energy_drift_ratio": round(self.mean_energy_drift_ratio, 6),
            "is_stable": self.is_stable,
            "max_state_norm": round(self.max_state_norm, 6),
            "divergence_rate": round(self.divergence_rate, 4),
            "avg_step_latency_ms": round(self.avg_step_latency_ms, 4),
        }


class PotentialEnergyNetwork:
    """Fast vectorized scalar potential energy V_theta(q) with analytical gradients."""

    def __init__(
        self,
        dim_q: int,
        hidden_dim: int = 64,
        omega: float = 1.0,
        seed: Optional[int] = 42,
    ) -> None:
        self.dim_q = dim_q
        self.hidden_dim = hidden_dim
        self.omega = float(omega)

        rng = np.random.RandomState(seed)
        # Xavier/He initialization for potential weights
        limit = math.sqrt(2.0 / (dim_q + hidden_dim))
        self.w1 = rng.uniform(-limit, limit, (hidden_dim, dim_q)).astype(np.float64)
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.w2 = rng.uniform(-limit, limit, hidden_dim).astype(np.float64) * 0.1
        self.b2 = 0.0

    def forward(self, q: np.ndarray) -> Tuple[float, np.ndarray]:
        """Computes potential energy V(q) and analytical gradient dV/dq.

        V(q) = 0.5 * omega^2 * ||q||^2 + w2^T * softplus(W1 * q + b1)
        dV/dq = omega^2 * q + W1^T * (sigmoid(W1 * q + b1) * w2)
        """
        # 1. Harmonic base potential
        harmonic_energy = 0.5 * (self.omega ** 2) * float(np.dot(q, q))
        harmonic_grad = (self.omega ** 2) * q

        # 2. Non-linear perturbation via Softplus
        h = np.dot(self.w1, q) + self.b1
        # Numerically stable softplus: log(1 + exp(h))
        softplus_h = np.where(h > 20.0, h, np.log1p(np.exp(np.clip(h, -50.0, 20.0))))
        nonlin_energy = float(np.dot(self.w2, softplus_h)) + self.b2

        # 3. Analytical gradient: d(softplus)/dh = sigmoid(h)
        sigmoid_h = 1.0 / (1.0 + np.exp(-np.clip(h, -50.0, 50.0)))
        d_nonlin = np.dot(self.w1.T, sigmoid_h * self.w2)

        total_v = harmonic_energy + nonlin_energy
        total_grad = harmonic_grad + d_nonlin
        return total_v, total_grad


class HamiltonianWorldModel:
    """Structure-Preserving Latent World Model driven by Hamiltonian Mechanics."""

    def __init__(
        self,
        latent_dim: int = 128,
        action_dim: int = 32,
        dt: float = 0.05,
        mass: float = 1.0,
        potential_hidden_dim: int = 64,
        omega: float = 0.8,
        seed: Optional[int] = 42,
    ) -> None:
        """
        Args:
            latent_dim: Total latent dimension D (must be even, e.g. 128).
            action_dim: Action representation dimension for external control force.
            dt: Time step for Störmer-Verlet symplectic integrator.
            mass: Inertial mass matrix parameter (T(p) = 0.5 * ||p||^2 / mass).
            potential_hidden_dim: Hidden dimension of potential energy network.
            omega: Harmonic anchor frequency (ensures Lyapunov boundedness).
            seed: Random seed for deterministic initialization.
        """
        if latent_dim % 2 != 0:
            raise ValueError(f"latent_dim={latent_dim} must be an even integer for canonical coordinates.")

        self.latent_dim = latent_dim
        self.dim_q = latent_dim // 2
        self.action_dim = action_dim
        self.dt = float(dt)
        self.mass = float(mass)

        # Potential energy network V_theta(q)
        self.potential = PotentialEnergyNetwork(
            dim_q=self.dim_q,
            hidden_dim=potential_hidden_dim,
            omega=omega,
            seed=seed,
        )

        # Action control force encoder F_ext(a) in R^(D/2)
        rng = np.random.RandomState((seed + 1) if seed is not None else None)
        self.w_action = rng.randn(self.dim_q, action_dim).astype(np.float64) * 0.05

    def split_canonical_coordinates(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decomposes latent state z into position q and momentum p."""
        q = z[:self.dim_q]
        p = z[self.dim_q:]
        return q, p

    def join_canonical_coordinates(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Assembles position q and momentum p into latent state z."""
        return np.concatenate([q, p], axis=0)

    def compute_kinetic_energy(self, p: np.ndarray) -> Tuple[float, np.ndarray]:
        """T(p) = 0.5 * ||p||^2 / mass. Gradient: dT/dp = p / mass."""
        t_energy = 0.5 * float(np.dot(p, p)) / self.mass
        grad_p = p / self.mass
        return t_energy, grad_p

    def compute_hamiltonian(self, q: np.ndarray, p: np.ndarray) -> Tuple[float, float, float]:
        """Computes scalar total Hamiltonian H = T(p) + V(q).

        Returns:
            Tuple of (H_total, T_kinetic, V_potential).
        """
        t_energy, _ = self.compute_kinetic_energy(p)
        v_energy, _ = self.potential.forward(q)
        return t_energy + v_energy, t_energy, v_energy

    def compute_action_force(self, action_vec: Optional[np.ndarray]) -> np.ndarray:
        """Projects external action into generalized momentum force F_ext in R^(D/2)."""
        if action_vec is None:
            return np.zeros(self.dim_q, dtype=np.float64)
        if len(action_vec) != self.action_dim:
            # Pad or truncate to match action_dim
            padded = np.zeros(self.action_dim, dtype=np.float64)
            n = min(len(action_vec), self.action_dim)
            padded[:n] = action_vec[:n]
            action_vec = padded
        return np.dot(self.w_action, action_vec)

    def step(
        self,
        current_state: np.ndarray,
        action: Optional[Union[np.ndarray, str]] = None,
        dt_override: Optional[float] = None,
        reference_energy: Optional[float] = None,
    ) -> HamiltonianStepResult:
        """Performs one step of Störmer-Verlet Symplectic Integration.

        Symplectic Leapfrog update:
            p_{t+1/2} = p_t - (dt/2) * dV/dq(q_t) + (dt/2) * F(a)
            q_{t+1}   = q_t + dt * (p_{t+1/2} / mass)
            p_{t+1}   = p_{t+1/2} - (dt/2) * dV/dq(q_{t+1}) + (dt/2) * F(a)
        """
        t0 = time.perf_counter()
        dt = dt_override or self.dt
        q_t, p_t = self.split_canonical_coordinates(current_state)

        # Action force F_ext
        if isinstance(action, str):
            # Deterministic pseudo-action embedding
            h = abs(hash(action)) % (2**31)
            rng_act = np.random.RandomState(h)
            act_vec = rng_act.randn(self.action_dim) * 0.1
            f_ext = self.compute_action_force(act_vec)
        elif action is not None:
            f_ext = self.compute_action_force(np.asarray(action, dtype=np.float64))
        else:
            f_ext = np.zeros(self.dim_q, dtype=np.float64)

        # Initial Hamiltonian
        h_0, _, _ = self.compute_hamiltonian(q_t, p_t)
        base_h = reference_energy if reference_energy is not None else h_0

        # --- Störmer-Verlet Integration ---
        # 1. Half-step momentum kick
        _, grad_v_qt = self.potential.forward(q_t)
        p_half = p_t - 0.5 * dt * grad_v_qt + 0.5 * dt * f_ext

        # 2. Full-step position drift (using kinetic gradient: dT/dp = p_half / mass)
        q_next = q_t + dt * (p_half / self.mass)

        # 3. Half-step momentum kick
        _, grad_v_qnext = self.potential.forward(q_next)
        p_next = p_half - 0.5 * dt * grad_v_qnext + 0.5 * dt * f_ext

        # Combine coordinates
        z_next = self.join_canonical_coordinates(q_next, p_next)

        # Evaluate final energy
        h_next, t_next, v_next = self.compute_hamiltonian(q_next, p_next)
        drift_ratio = abs(h_next - base_h) / max(1e-6, abs(base_h))

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        return HamiltonianStepResult(
            next_state=z_next,
            q_next=q_next,
            p_next=p_next,
            hamiltonian_energy=h_next,
            kinetic_energy=t_next,
            potential_energy=v_next,
            energy_drift_ratio=drift_ratio,
            step_latency_ms=latency_ms,
        )

    def rollout(
        self,
        initial_state: np.ndarray,
        horizon: int = 100,
        actions: Optional[Sequence[Any]] = None,
        dt_override: Optional[float] = None,
        stability_norm_threshold: float = 100.0,
    ) -> HamiltonianRolloutResult:
        """Executes a long-horizon closed-loop simulation on the symplectic manifold."""
        dt = dt_override or self.dt
        q0, p0 = self.split_canonical_coordinates(initial_state)
        h0, _, _ = self.compute_hamiltonian(q0, p0)

        states = [initial_state.copy()]
        energies = [h0]
        state_norms = [float(np.linalg.norm(initial_state))]

        curr_state = initial_state.copy()
        step_latencies = []

        is_free_evolution = (actions is None)

        for step_idx in range(horizon):
            act = actions[step_idx] if (actions is not None and step_idx < len(actions)) else None
            ref_e = h0 if is_free_evolution else None

            res = self.step(
                current_state=curr_state,
                action=act,
                dt_override=dt,
                reference_energy=ref_e,
            )

            curr_state = res.next_state
            states.append(curr_state.copy())
            energies.append(res.hamiltonian_energy)
            norm_val = float(np.linalg.norm(curr_state))
            state_norms.append(norm_val)
            step_latencies.append(res.step_latency_ms)

        final_energy = energies[-1]
        energy_drift_ratios = [abs(e - h0) / max(1e-6, abs(h0)) for e in energies]
        max_drift = max(energy_drift_ratios)
        mean_drift = float(np.mean(energy_drift_ratios))

        max_norm = max(state_norms)
        # Divergence check: NaN, Inf, or exploded norm
        is_diverged = any(math.isnan(n) or math.isinf(n) or n > stability_norm_threshold for n in state_norms)
        divergence_rate = 1.0 if is_diverged else 0.0

        is_stable = (not is_diverged) and (max_drift <= 0.05 if is_free_evolution else True)
        avg_latency = float(np.mean(step_latencies)) if step_latencies else 0.0

        return HamiltonianRolloutResult(
            horizon=horizon,
            states=states,
            energies=energies,
            initial_energy=h0,
            final_energy=final_energy,
            max_energy_drift_ratio=max_drift,
            mean_energy_drift_ratio=mean_drift,
            is_stable=is_stable,
            state_norms=state_norms,
            max_state_norm=max_norm,
            divergence_rate=divergence_rate,
            avg_step_latency_ms=avg_latency,
        )


if HAS_TORCH:
    class PyTorchHamiltonianNeuralODE(nn.Module):
        """Differentiable Hamiltonian Neural ODE with autograd symplectic vector fields."""

        def __init__(
            self,
            latent_dim: int = 128,
            action_dim: int = 32,
            hidden_dim: int = 64,
        ) -> None:
            super().__init__()
            if latent_dim % 2 != 0:
                raise ValueError("latent_dim must be even.")

            self.latent_dim = latent_dim
            self.dim_q = latent_dim // 2
            self.action_dim = action_dim

            # Potential network V_theta(q)
            self.v_net = nn.Sequential(
                nn.Linear(self.dim_q, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1, bias=False),
            )

            # Mass parameter: learnable diagonal inverse mass matrix
            self.inv_mass = nn.Parameter(torch.ones(self.dim_q))
            # Action force encoder
            self.action_encoder = nn.Linear(action_dim, self.dim_q, bias=False)

        def hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            """Computes total scalar energy H = 0.5 * p^T * M^-1 * p + V(q)."""
            t_energy = 0.5 * torch.sum(p * (torch.abs(self.inv_mass) + 1e-4) * p, dim=-1, keepdim=True)
            v_energy = self.v_net(q)
            return t_energy + v_energy

        def forward_symplectic_step(
            self,
            z: torch.Tensor,
            action: Optional[torch.Tensor] = None,
            dt: float = 0.05,
        ) -> torch.Tensor:
            """Executes one step of differentiable Störmer-Verlet integration."""
            q = z[..., :self.dim_q]
            p = z[..., self.dim_q:]

            f_ext = self.action_encoder(action) if action is not None else 0.0

            # 1. dV/dq at q_t
            with torch.enable_grad():
                q_req = q.clone().detach().requires_grad_(True)
                v_q = self.v_net(q_req)
                grad_v_q = torch.autograd.grad(v_q.sum(), q_req, create_graph=True)[0]

            # 2. Half-step p
            p_half = p - 0.5 * dt * grad_v_q + 0.5 * dt * f_ext

            # 3. Full-step q (using dT/dp = inv_mass * p_half)
            eff_inv_m = torch.abs(self.inv_mass) + 1e-4
            q_next = q + dt * (eff_inv_m * p_half)

            # 4. dV/dq at q_{t+1}
            with torch.enable_grad():
                q_next_req = q_next.clone().detach().requires_grad_(True)
                v_qnext = self.v_net(q_next_req)
                grad_v_qnext = torch.autograd.grad(v_qnext.sum(), q_next_req, create_graph=True)[0]

            # 5. Half-step p
            p_next = p_half - 0.5 * dt * grad_v_qnext + 0.5 * dt * f_ext

            return torch.cat([q_next, p_next], dim=-1)
else:
    class PyTorchHamiltonianNeuralODE:
        pass
