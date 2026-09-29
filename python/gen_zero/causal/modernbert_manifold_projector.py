"""ModernBERT hidden-state bridge into a stable 64-D causal manifold.

The numerical bridge is useful before training, but its random orthogonal
initialisation is *not* a semantic alignment.  Callers that need decisions must
load trained weights and set ``calibrated=True`` with provenance of that fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import Tensor, nn

__all__ = ["ModernBertManifoldProjector", "LyapunovAttractor", "ManifoldOutput"]


def _finite_float(name: str, value: Tensor, ndim: tuple[int, ...]) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim not in ndim or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating tensor with ndim in {ndim}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class ManifoldOutput:
    pooled: Tensor
    target: Tensor
    state: Tensor
    iterations: int
    converged: bool
    final_residual: float


class LyapunovAttractor(nn.Module):
    """Contractive dynamics with target ``u`` and V(z)=||z-u||^2.

    ``z[k+1] = contraction*z[k] + (1-contraction)*u`` gives
    ``V[k+1] = contraction**2 * V[k]`` (up to floating-point error).
    It has no trainable parameters and therefore cannot manufacture semantic
    reasoning; it only supplies a certified stable numerical interface.
    """

    def __init__(self, dimension: int = 64, contraction: float = 0.25) -> None:
        super().__init__()
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if not 0.0 < contraction < 1.0:
            raise ValueError("contraction must be strictly between zero and one")
        self.dimension = dimension
        self.contraction = float(contraction)

    def step(self, state: Tensor, target: Tensor) -> Tensor:
        _finite_float("state", state, (2,))
        _finite_float("target", target, (2,))
        if state.shape != target.shape or state.shape[-1] != self.dimension:
            raise ValueError("state and target must have shape [batch, dimension]")
        return torch.lerp(target, state, self.contraction)

    def forward(self, target: Tensor, *, initial_state: Optional[Tensor] = None,
                tolerance: float = 1e-5, max_steps: int = 16) -> tuple[Tensor, int, bool, float]:
        _finite_float("target", target, (2,))
        if target.shape[-1] != self.dimension:
            raise ValueError(f"target must have last dimension {self.dimension}")
        if tolerance <= 0 or max_steps <= 0:
            raise ValueError("tolerance and max_steps must be positive")
        state = torch.zeros_like(target) if initial_state is None else initial_state
        _finite_float("initial_state", state, (2,))
        if state.shape != target.shape:
            raise ValueError("initial_state must have the same shape as target")
        residual = float(torch.linalg.vector_norm(state - target, dim=-1).max().detach())
        steps = 0
        while residual > tolerance and steps < max_steps:
            state = self.step(state, target)
            steps += 1
            residual = float(torch.linalg.vector_norm(state - target, dim=-1).max().detach())
        return state, steps, residual <= tolerance, residual


class ModernBertManifoldProjector(nn.Module):
    """Pool 1024-D ModernBERT states and project them through a 64x1024 map."""

    def __init__(self, hidden_dim: int = 1024, manifold_dim: int = 64, *,
                 pooling: Literal["mean", "cls"] = "mean", contraction: float = 0.25,
                 calibrated: bool = False, provenance: Optional[str] = None) -> None:
        super().__init__()
        if hidden_dim <= 0 or manifold_dim <= 0:
            raise ValueError("hidden_dim and manifold_dim must be positive")
        if pooling not in ("mean", "cls"):
            raise ValueError("pooling must be 'mean' or 'cls'")
        if calibrated and not provenance:
            raise ValueError("calibrated weights require non-empty provenance")
        self.hidden_dim = hidden_dim
        self.manifold_dim = manifold_dim
        self.pooling = pooling
        self.calibrated = bool(calibrated)
        self.provenance = provenance
        self.normalizer = nn.LayerNorm(hidden_dim)
        self.projection = nn.Linear(hidden_dim, manifold_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        self.dynamics = LyapunovAttractor(manifold_dim, contraction)

    def pool(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        _finite_float("hidden_states", hidden_states, (2, 3))
        if hidden_states.shape[-1] != self.hidden_dim:
            raise ValueError(f"hidden_states last dimension must be {self.hidden_dim}")
        if hidden_states.ndim == 2:
            if attention_mask is not None:
                raise ValueError("attention_mask is only valid for sequence states")
            return hidden_states
        batch, length, _ = hidden_states.shape
        if attention_mask is None:
            if self.pooling == "mean":
                return hidden_states.mean(dim=1)
            return hidden_states[:, 0]
        if attention_mask.shape != (batch, length):
            raise ValueError(f"attention_mask must have shape {(batch, length)}")
        valid = attention_mask.to(device=hidden_states.device) != 0
        if not valid.any(dim=1).all():
            raise ValueError("every sample must contain an unmasked token")
        if self.pooling == "cls":
            first = valid.to(torch.int64).argmax(dim=1)
            return hidden_states[torch.arange(batch, device=hidden_states.device), first]
        weights = valid.unsqueeze(-1).to(hidden_states.dtype)
        return (hidden_states * weights).sum(dim=1) / weights.sum(dim=1)

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None, *,
                tolerance: float = 1e-5, max_steps: int = 16) -> ManifoldOutput:
        pooled = self.pool(hidden_states, attention_mask)
        target = self.projection(self.normalizer(pooled))
        state, iterations, converged, residual = self.dynamics(
            target, tolerance=tolerance, max_steps=max_steps)
        return ManifoldOutput(pooled, target, state, iterations, converged, residual)

    @property
    def projection_payload_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.parameters())
