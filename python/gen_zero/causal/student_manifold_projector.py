"""Trainable bridge from a small language model into a 64-D causal state.

This module deliberately does not pretend that a randomly initialized projection has
semantic meaning.  ``decide`` fails closed until the bridge is explicitly marked as
calibrated by the training/loading code that owns its provenance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

__all__ = [
    "StudentManifoldProjector",
    "CausalStateDynamics",
    "StudentCausalDecisionPipeline",
    "PipelineOutput",
]


def _check_float_tensor(name: str, value: Tensor, last_dim: int) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim not in (2, 3) or value.shape[-1] != last_dim:
        raise ValueError(f"{name} must have shape [batch, {last_dim}] or [batch, seq, {last_dim}]")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite floating-point values")


class StudentManifoldProjector(nn.Module):
    """Pool hidden states and map ``hidden_dim`` to a compact causal manifold.

    ``rank=None`` uses one affine map.  A positive rank uses a bias plus the
    factorisation ``down: hidden_dim -> rank`` and ``up: rank -> manifold_dim``.
    The layer is trainable and language-agnostic: it sees numeric states only.
    """

    def __init__(self, hidden_dim: int = 896, manifold_dim: int = 64,
                 rank: Optional[int] = None, *, normalize: bool = True) -> None:
        super().__init__()
        if hidden_dim <= 0 or manifold_dim <= 0:
            raise ValueError("hidden_dim and manifold_dim must be positive")
        if rank is not None and (rank <= 0 or rank > min(hidden_dim, manifold_dim)):
            raise ValueError("rank must be in [1, min(hidden_dim, manifold_dim)]")
        self.hidden_dim = hidden_dim
        self.manifold_dim = manifold_dim
        self.rank = rank
        self.normalizer = nn.LayerNorm(hidden_dim) if normalize else nn.Identity()
        if rank is None:
            self.projection = nn.Linear(hidden_dim, manifold_dim)
            self.down = self.up = None
        else:
            self.projection = None
            self.down = nn.Linear(hidden_dim, rank, bias=False)
            self.up = nn.Linear(rank, manifold_dim, bias=True)

    def pool(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        _check_float_tensor("hidden_states", hidden_states, self.hidden_dim)
        if hidden_states.ndim == 2:
            if attention_mask is not None:
                raise ValueError("attention_mask is only valid for sequence hidden states")
            return hidden_states
        batch, seq, _ = hidden_states.shape
        if attention_mask is None:
            return hidden_states[:, -1, :]
        if attention_mask.shape != (batch, seq):
            raise ValueError(f"attention_mask must have shape {(batch, seq)}")
        mask = attention_mask.to(device=hidden_states.device)
        if mask.dtype == torch.bool:
            valid = mask
        elif mask.is_floating_point() or mask.dtype in (torch.uint8, torch.int8, torch.int16,
                                                         torch.int32, torch.int64):
            valid = mask != 0
        else:
            raise ValueError("attention_mask must be numeric or boolean")
        lengths = valid.long().sum(dim=1)
        if (lengths == 0).any():
            raise ValueError("every sample must contain at least one unmasked token")
        # Works for left and right padding (and sparse masks): select the last true index.
        positions = torch.arange(seq, device=hidden_states.device).expand(batch, seq)
        indices = positions.masked_fill(~valid, -1).max(dim=1).values
        return hidden_states[torch.arange(batch, device=hidden_states.device), indices]

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        pooled = self.normalizer(self.pool(hidden_states, attention_mask))
        result = self.projection(pooled) if self.projection is not None else self.up(self.down(pooled))
        if not torch.isfinite(result).all():
            raise FloatingPointError("projection produced non-finite state")
        return result


class CausalStateDynamics(nn.Module):
    """Small residual dynamics operating entirely in the 64-D state space."""

    def __init__(self, dimension: int = 64, steps: int = 4) -> None:
        super().__init__()
        if dimension <= 0 or steps <= 0:
            raise ValueError("dimension and steps must be positive")
        self.dimension = dimension
        self.steps = steps
        self.drift = nn.Linear(dimension, dimension)
        self.gate = nn.Linear(dimension, dimension)

    def forward(self, initial_state: Tensor) -> Tensor:
        _check_float_tensor("initial_state", initial_state, self.dimension)
        if initial_state.ndim != 2:
            raise ValueError("initial_state must have shape [batch, dimension]")
        state = initial_state
        for _ in range(self.steps):
            state = state + torch.sigmoid(self.gate(state)) * torch.tanh(self.drift(state)) / self.steps
        if not torch.isfinite(state).all():
            raise FloatingPointError("dynamics produced non-finite state")
        return state


@dataclass
class PipelineOutput:
    initial_state: Tensor
    final_state: Tensor
    logits: Tensor
    decision: Optional[Tensor]


class StudentCausalDecisionPipeline(nn.Module):
    """Projection -> 64-D dynamics -> decision logits.

    The weights need supervised calibration outside this module.  Until then the
    argmax is exposed only when ``allow_uncalibrated=True`` and must not be
    reported as a meaningful task decision.
    """

    def __init__(self, projector: StudentManifoldProjector, num_decisions: int,
                 *, dynamics_steps: int = 4, calibrated: bool = False) -> None:
        super().__init__()
        if num_decisions < 2:
            raise ValueError("num_decisions must be at least two")
        self.projector = projector
        self.dynamics = CausalStateDynamics(projector.manifold_dim, dynamics_steps)
        self.decision_head = nn.Linear(projector.manifold_dim, num_decisions)
        self.calibrated = bool(calibrated)

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None,
                *, allow_uncalibrated: bool = False) -> PipelineOutput:
        initial = self.projector(hidden_states, attention_mask)
        final = self.dynamics(initial)
        logits = self.decision_head(final)
        decision = logits.argmax(dim=-1) if self.calibrated or allow_uncalibrated else None
        return PipelineOutput(initial, final, logits, decision)

    def decide(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        if not self.calibrated:
            raise RuntimeError("decision pipeline is not calibrated; semantic decisions are disabled")
        return self.forward(hidden_states, attention_mask).decision
