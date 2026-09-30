"""Deep projection adapter: nonlinear 896->256->64 branch atop the frozen ZCA linear skip.

Source: docs/zero/10-trunk-unfreeze-and-latent-recurrence-engineering-spec.md §3.3.
Why: the linear skip alone is ``ZeroManifold.project``, which already answered
"is the judging direction in the top-64 ZCA principal components" -- adding a
zero-initialized nonlinear branch on top, instead of replacing the skip, is what
lets an ablation attribute any post-training gain to information that lives in
the 896-D hidden state but outside the top-64 principal subspace (the doc's
"why it is ordered before LoRA" argument), rather than to a different starting
point that could itself explain a change in accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor, nn

from .zero_runtime import ZeroManifold

__all__ = ["DeepProjectionAdapter"]

DEFAULT_MANIFOLD_PATH = (
    Path(__file__).resolve().parents[3] / "benchmarks" / "artifacts" / "zero" / "zero_manifold_open_v1.npz"
)


def _load_manifold(manifold: Union[str, Path, ZeroManifold], encoder_id: Optional[str]) -> ZeroManifold:
    """Resolve the linear-skip source into a validated ``ZeroManifold``.

    ``ZeroManifold.load`` requires the caller to already know ``encoder_id`` (it
    is a pinning check against the artifact's own claim, see zero_runtime.py:209).
    When the caller has not pinned one, we read only the ``metadata.encoder_id``
    field the artifact declares about itself and feed it back into
    ``ZeroManifold.load`` -- every other check inside that classmethod (schema,
    shapes, finiteness, orthonormality) still runs unmodified.
    """
    if isinstance(manifold, ZeroManifold):
        return manifold
    path = Path(manifold)
    resolved_encoder_id = encoder_id
    if resolved_encoder_id is None:
        with np.load(path, allow_pickle=False) as data:
            if "metadata" not in data.files:
                raise ValueError(f"manifold file {path} is missing a metadata array")
            resolved_encoder_id = json.loads(str(data["metadata"])).get("encoder_id")
        if not resolved_encoder_id:
            raise ValueError(f"manifold file {path} declares no encoder_id; pass encoder_id explicitly")
    return ZeroManifold.load(path, encoder_id=resolved_encoder_id)


class DeepProjectionAdapter(nn.Module):
    """896 -> 256 -> 64 nonlinear residual branch summed with the linear ZCA skip.

    ``z = normalize(S @ (h - mu) + W2 @ GELU(W1 @ (h - mu) + b1) + b2)``

    ``S`` is initialized from the manifold as ``S = diag(scale) @ basis.T`` so
    that ``S @ (h - mu)`` reproduces ``ZeroManifold.project``'s pre-normalize
    output exactly (zero_runtime.py:191). ``W2``'s weight and bias are
    zero-initialized, so ``delta_z`` is identically zero at construction
    regardless of ``W1``: the adapter starts bit-identical (up to floating-point
    rounding) to the linear projector it extends, and any change after training
    is therefore attributable to learning, not to a different starting point
    (the doc's "any regression is immediately attributable" contract, doc 10 §3.3).
    """

    def __init__(
        self,
        manifold: Union[str, Path, ZeroManifold] = DEFAULT_MANIFOLD_PATH,
        *,
        encoder_id: Optional[str] = None,
        bottleneck_dim: int = 256,
    ) -> None:
        super().__init__()
        zm = _load_manifold(manifold, encoder_id)
        hidden_dim = zm.hidden
        manifold_dim = zm.dim
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")
        if hidden_dim <= 0 or manifold_dim <= 0:
            raise ValueError("manifold must have positive hidden and manifold dimensions")

        self.hidden_dim = hidden_dim
        self.manifold_dim = manifold_dim
        self.bottleneck_dim = bottleneck_dim

        mu = torch.from_numpy(np.ascontiguousarray(zm.mean, dtype=np.float32)).clone()
        self.register_buffer("mu", mu)

        # Linear skip S: (manifold_dim, hidden_dim), trainable (frozen explicitly
        # by the CPU fine-tune script, not by this module -- see doc 10 §3.3).
        self.skip = nn.Linear(hidden_dim, manifold_dim, bias=False)
        basis = np.asarray(zm.basis, dtype=np.float64)      # (hidden_dim, manifold_dim)
        scale = np.asarray(zm.scale, dtype=np.float64)      # (manifold_dim,)
        s_weight = (basis * scale[None, :]).T.astype(np.float32)  # (manifold_dim, hidden_dim)
        with torch.no_grad():
            self.skip.weight.copy_(torch.from_numpy(np.ascontiguousarray(s_weight)))

        # Nonlinear bottleneck branch. branch[0] = W1/b1, branch[2] = W2/b2.
        self.branch = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, manifold_dim),
        )
        with torch.no_grad():
            self.branch[2].weight.zero_()
            self.branch[2].bias.zero_()

    def forward(self, h: Tensor) -> Tensor:
        if not isinstance(h, Tensor):
            raise TypeError("h must be a torch.Tensor")
        if h.ndim < 1 or h.shape[-1] != self.hidden_dim:
            raise ValueError(f"h must have last dimension {self.hidden_dim}")
        if not h.is_floating_point():
            raise ValueError("h must be a floating-point tensor")
        if not torch.isfinite(h).all():
            raise ValueError("h must contain only finite values")

        centered = h.to(self.mu.dtype) - self.mu
        z_skip = self.skip(centered)
        delta_z = self.branch(centered)
        z = z_skip + delta_z

        norms = z.norm(dim=-1, keepdim=True)
        if torch.any(norms <= 1e-12):
            raise ValueError("a state collapsed to the manifold origin; cannot sphere it")
        z = z / norms
        if not torch.isfinite(z).all():
            raise FloatingPointError("deep projection adapter produced a non-finite state")
        return z
