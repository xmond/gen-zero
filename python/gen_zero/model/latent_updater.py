"""Gen-Zero Latent Updater: zero-token, multi-step latent-space reasoning.

A System 1 (sub-second, zero-token) reflex module. It never emits a token:
it spends a small, bounded number of steps refining a latent vector before
the reflex scorer runs. In the G/R shorthand used elsewhere in this repo,
G=0 (zero generated tokens) and R>0 (a positive "reflex depth", i.e. the
number of latent-update steps actually taken).

Recurrence:
    z_0 = W_in(x)
    for t in range(T):
        delta_t = UpdateBlock(z_t, x)
        z_{t+1} = LayerNorm(z_t + alpha * delta_t)

The loop stops early once the step residual ||z_{t+1} - z_t|| drops below a
configured epsilon, so the common case is far cheaper than the worst case.

`LatentUpdater` works with torch when available and falls back to a real
NumPy implementation when torch is absent. Both implementations share the
same public forward contract: `updater(x) -> (z_final, telemetry)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = object
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Plain dataclass telemetry (no torch dependency)
# ---------------------------------------------------------------------------

@dataclass
class LatentUpdaterTelemetry:
    """Telemetry from one LatentUpdater forward pass.

    Attributes:
        n_iterations: Number of latent-update steps actually taken (the "R"
            in G=0, R>0).
        final_residual: The step residual ||z_{t+1} - z_t|| at the last step
            taken.
        converged: True if the loop stopped early because the residual fell
            below epsilon, False if it exhausted the full step budget T.
        residual_history: The residual norm recorded after every step, in
            order. `len(residual_history) == n_iterations`.
        zero_token: Always True. Confirms the zero-token (G=0) contract:
            this module never returns text or token ids.
        reflex_depth: Alias for n_iterations, kept as an explicit field so
            callers can assert the R>0 contract without knowing the rest of
            the telemetry shape.
    """
    n_iterations: int
    final_residual: float
    converged: bool
    residual_history: List[float] = field(default_factory=list)
    zero_token: bool = True
    reflex_depth: int = 0


def _validate_config(
    input_dim: int,
    latent_dim: int,
    alpha: float,
    T: int,
    epsilon: float,
) -> None:
    """Fail-closed constructor validation shared by both backends."""
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}")
    if latent_dim <= 0:
        raise ValueError(f"latent_dim must be positive, got {latent_dim}")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")


# ---------------------------------------------------------------------------
# Real NumPy implementation. Defined unconditionally so it is directly
# importable and testable even in environments where torch is installed.
# ---------------------------------------------------------------------------

class NumpyLatentUpdater:
    """Pure-NumPy zero-token, multi-step latent-space reasoning module.

    Same recurrence and telemetry contract as the torch-backed
    `LatentUpdater`. Used as the fallback when torch is unavailable, and
    directly importable for unit-testing the NumPy code path on its own.

    Args:
        input_dim: Dimensionality of the conditioning input x.
        latent_dim: Dimensionality of the latent state z.
        alpha: Damping/contraction factor in (0, 1].
        T: Maximum number of reasoning steps.
        epsilon: Early-stop threshold on the step residual norm.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        alpha: float = 0.5,
        T: int = 4,
        epsilon: float = 1e-4,
    ) -> None:
        _validate_config(input_dim, latent_dim, alpha, T, epsilon)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.T = T
        self.epsilon = epsilon

        rng = np.random.default_rng()
        scale_in = 1.0 / np.sqrt(input_dim)
        scale_h = 1.0 / np.sqrt(2 * latent_dim)

        self.w_in = (rng.standard_normal((input_dim, latent_dim)) * scale_in)
        self.b_in = np.zeros(latent_dim)

        self.w_ctx = (rng.standard_normal((input_dim, latent_dim)) * scale_in)
        self.b_ctx = np.zeros(latent_dim)

        self.w_gate = (rng.standard_normal((2 * latent_dim, latent_dim)) * scale_h)
        self.b_gate = np.zeros(latent_dim)

        self.w_cand = (rng.standard_normal((2 * latent_dim, latent_dim)) * scale_h)
        self.b_cand = np.zeros(latent_dim)

        self.ln_weight = np.ones(latent_dim)
        self.ln_bias = np.zeros(latent_dim)
        self._ln_eps = 1e-5

    def _layer_norm(self, z: np.ndarray) -> np.ndarray:
        mean = z.mean(axis=-1, keepdims=True)
        var = z.var(axis=-1, keepdims=True)
        normed = (z - mean) / np.sqrt(var + self._ln_eps)
        return normed * self.ln_weight + self.ln_bias

    @staticmethod
    def _sigmoid(v: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-v))

    def _update_block(self, z: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Gated MLP producing the latent delta, conditioned on x."""
        ctx = x @ self.w_ctx + self.b_ctx
        h = np.concatenate([z, ctx], axis=-1)
        gate = self._sigmoid(h @ self.w_gate + self.b_gate)
        cand = np.tanh(h @ self.w_cand + self.b_cand)
        return gate * cand

    def __call__(self, x: Any) -> Tuple[np.ndarray, LatentUpdaterTelemetry]:
        return self.forward(x)

    def forward(self, x: Any) -> Tuple[np.ndarray, LatentUpdaterTelemetry]:
        """Run the recurrence and return (z_final, telemetry).

        Accepts a NumPy array (or anything np.asarray can convert) with
        shape (input_dim,) or (batch, input_dim). Batched input is
        preserved; a 1-D input yields a 1-D output.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        squeeze_output = x_arr.ndim == 1
        if squeeze_output:
            x_arr = x_arr[np.newaxis, :]
        if x_arr.ndim != 2 or x_arr.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected x with last dim {self.input_dim}, got shape {x_arr.shape}"
            )

        z = x_arr @ self.w_in + self.b_in
        residual_history: List[float] = []
        converged = False
        n_iterations = 0

        for _ in range(self.T):
            delta = self._update_block(z, x_arr)
            z_next = self._layer_norm(z + self.alpha * delta)
            # Per-sample L2 norm, reduced to the batch worst-case so the
            # early-stop decision is conservative under batching.
            residual = float(np.max(np.linalg.norm(z_next - z, axis=-1)))
            residual_history.append(residual)
            z = z_next
            n_iterations += 1
            if residual < self.epsilon:
                converged = True
                break

        telemetry = LatentUpdaterTelemetry(
            n_iterations=n_iterations,
            final_residual=residual_history[-1],
            converged=converged,
            residual_history=residual_history,
            zero_token=True,
            reflex_depth=n_iterations,
        )
        z_out = z[0] if squeeze_output else z
        return z_out, telemetry


# ---------------------------------------------------------------------------
# Torch-dependent implementation
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class _UpdateBlock(nn.Module):
        """Gated MLP producing the latent delta, conditioned on x."""

        def __init__(self, input_dim: int, latent_dim: int) -> None:
            super().__init__()
            self.context_proj = nn.Linear(input_dim, latent_dim)
            self.gate_proj = nn.Linear(2 * latent_dim, latent_dim)
            self.cand_proj = nn.Linear(2 * latent_dim, latent_dim)

        def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            ctx = self.context_proj(x)
            h = torch.cat([z, ctx], dim=-1)
            gate = torch.sigmoid(self.gate_proj(h))
            cand = torch.tanh(self.cand_proj(h))
            return gate * cand

    class LatentUpdater(nn.Module):
        """Zero-token (G=0), multi-step (R>0) latent-space reasoning module.

        Recurrence:
            z_0 = W_in(x)
            for t in range(T):
                delta_t = UpdateBlock(z_t, x)
                z_{t+1} = LayerNorm(z_t + alpha * delta_t)

        Stops early once the step residual ||z_{t+1} - z_t|| drops below
        epsilon. Intended to run in microseconds-to-low-milliseconds as a
        System 1 sub-stage: no attention, no extra abstraction layers.

        Args:
            input_dim: Dimensionality of the conditioning input x.
            latent_dim: Dimensionality of the latent state z.
            alpha: Damping/contraction factor in (0, 1].
            T: Maximum number of reasoning steps.
            epsilon: Early-stop threshold on the step residual norm.
        """

        def __init__(
            self,
            input_dim: int,
            latent_dim: int,
            alpha: float = 0.5,
            T: int = 4,
            epsilon: float = 1e-4,
        ) -> None:
            super().__init__()
            _validate_config(input_dim, latent_dim, alpha, T, epsilon)
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            self.alpha = alpha
            self.T = T
            self.epsilon = epsilon

            self.w_in = nn.Linear(input_dim, latent_dim)
            self.update_block = _UpdateBlock(input_dim, latent_dim)
            self.norm = nn.LayerNorm(latent_dim)

        def forward(self, x: Any) -> Tuple[torch.Tensor, LatentUpdaterTelemetry]:
            """Run the recurrence and return (z_final, telemetry).

            Accepts a torch.Tensor (or anything torch.as_tensor can
            convert) with shape (input_dim,) or (batch, input_dim). Batched
            input is preserved; a 1-D input yields a 1-D output.
            """
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x, dtype=self.w_in.weight.dtype)
            squeeze_output = x.dim() == 1
            x_in = x.unsqueeze(0) if squeeze_output else x
            if x_in.dim() != 2 or x_in.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Expected x with last dim {self.input_dim}, got shape {tuple(x.shape)}"
                )

            z = self.w_in(x_in)
            residual_history: List[float] = []
            converged = False
            n_iterations = 0

            for _ in range(self.T):
                delta = self.update_block(z, x_in)
                z_next = self.norm(z + self.alpha * delta)
                # Per-sample L2 norm, reduced to the batch worst-case so the
                # early-stop decision is conservative under batching.
                residual_t = (z_next - z).norm(dim=-1).max()
                residual_val = float(residual_t.detach().cpu())
                residual_history.append(residual_val)
                z = z_next
                n_iterations += 1
                if residual_val < self.epsilon:
                    converged = True
                    break

            telemetry = LatentUpdaterTelemetry(
                n_iterations=n_iterations,
                final_residual=residual_history[-1],
                converged=converged,
                residual_history=residual_history,
                zero_token=True,
                reflex_depth=n_iterations,
            )
            z_out = z.squeeze(0) if squeeze_output else z
            return z_out, telemetry

else:
    LatentUpdater = NumpyLatentUpdater  # type: ignore[misc, assignment]
