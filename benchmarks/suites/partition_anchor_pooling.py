"""Spec 20 P1: partition anchor pooling (docs/zero/20-triad-deep-enhancement-multi-dimensional-analysis-spec.md
S2.3 "路径A：完整编码 + 分区anchor pooling", S7.1 P1).

Uniform mean pooling dilutes a discriminative signal that lives in m << T tokens down to
m*v/T (S2.3). This module implements the fixed-width three-region readout instead:
head (first N_H valid tokens, the instruction/opening), tail (last N_T valid tokens, the
latest context/question) and a query-anchored top-K softmax region (S2.3 eq. s_ij, S_j, p_j)
that resists that dilution because its softmax denominator is bounded by K, not T.

Anchor without an explicit query: S2.3 requires a registered training-fold mean/whitened
anchor for that case, which needs artifacts this standalone module does not own. Absent a
`query` argument, PartitionAnchorPooler falls back to the mean of the valid tokens as the
anchor target; this is a documented, label-free default for pipeline completeness, not the
S2.3 production protocol, and callers that need the real protocol must pass `query`.
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class PartitionAnchorConfig:
    n_head: int = 64
    n_tail: int = 128
    n_anchor: int = 64
    tau: float = 0.1
    mode: str = "convex"
    alpha: Tuple[float, float, float] = (0.2, 0.5, 0.3)

    def __post_init__(self) -> None:
        for name in ("n_head", "n_tail", "n_anchor"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ValueError(f"{name} must be a positive int, got {v!r}")
        if not isinstance(self.tau, (int, float)) or isinstance(self.tau, bool) or self.tau <= 0:
            raise ValueError(f"tau must be a positive float, got {self.tau!r}")
        if self.mode not in ("convex", "concat"):
            raise ValueError(f"mode must be 'convex' or 'concat', got {self.mode!r}")
        alpha = tuple(self.alpha)
        if len(alpha) != 3:
            raise ValueError(f"alpha must have exactly 3 entries, got {alpha!r}")
        for a in alpha:
            if not isinstance(a, (int, float)) or isinstance(a, bool) or a < 0:
                raise ValueError(f"alpha entries must be non-negative floats, got {alpha!r}")
        if abs(sum(alpha) - 1.0) > 1e-6:
            raise ValueError(f"alpha must sum to 1, got {alpha!r} (sum={sum(alpha)})")
        object.__setattr__(self, "alpha", alpha)


def _is_torch_tensor(x) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return isinstance(x, torch.Tensor)


def _to_numpy(x) -> np.ndarray:
    if _is_torch_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _softmax(s: np.ndarray) -> np.ndarray:
    s = s - np.max(s)
    e = np.exp(s)
    return e / np.sum(e)


def _anchor_pool(Hv: np.ndarray, anchor_vec: np.ndarray, n_anchor: int, tau: float) -> np.ndarray:
    V = Hv.shape[0]
    k = min(n_anchor, V)
    Hv_norm = np.linalg.norm(Hv, axis=1, keepdims=True)
    Hn = Hv / np.maximum(Hv_norm, _EPS)
    a_norm = float(np.linalg.norm(anchor_vec))
    an = np.asarray(anchor_vec / max(a_norm, _EPS), dtype=Hn.dtype)
    s = np.dot(Hn, an) / tau
    if k < V:
        top_idx = np.argpartition(-s, k - 1)[:k]
        top_idx = top_idx[np.argsort(-s[top_idx])]
    else:
        top_idx = np.argsort(-s)
    w = _softmax(s[top_idx])
    return np.dot(w, Hv[top_idx])


def _pool_core(H: np.ndarray, mask: Optional[np.ndarray], query: Optional[np.ndarray],
               cfg: PartitionAnchorConfig) -> np.ndarray:
    if H.ndim != 2:
        raise ValueError(f"H must be a 2D (T, D) array, got shape {H.shape}")
    T, D = H.shape
    if T == 0:
        raise ValueError("H has zero tokens")

    if mask is None:
        valid_idx = np.arange(T)
    else:
        mask = np.asarray(mask)
        if mask.shape != (T,):
            raise ValueError(f"attention_mask must have shape ({T},), got {mask.shape}")
        valid_idx = np.nonzero(mask.astype(bool))[0]
    if valid_idx.size == 0:
        raise ValueError("attention_mask excludes every token; no valid tokens to pool")

    Hv = H[valid_idx].astype(np.float64, copy=False)
    if not np.all(np.isfinite(Hv)):
        raise ValueError("H contains non-finite values at valid (unmasked) positions")
    V = Hv.shape[0]

    if V <= cfg.n_head + cfg.n_tail:
        p_head = Hv.mean(axis=0)
        p_tail = p_head
    else:
        p_head = Hv[: cfg.n_head].mean(axis=0)
        p_tail = Hv[-cfg.n_tail:].mean(axis=0)

    if query is None:
        anchor_vec = Hv.mean(axis=0)
    else:
        query = np.asarray(query, dtype=np.float64)
        if query.ndim == 1:
            if query.shape[0] != D:
                raise ValueError(f"query must have width {D}, got shape {query.shape}")
            anchor_vec = query
        elif query.ndim == 2:
            if query.shape[1] != D:
                raise ValueError(f"query must have width {D}, got shape {query.shape}")
            if query.shape[0] == 0:
                raise ValueError("query has zero tokens")
            if not np.all(np.isfinite(query)):
                raise ValueError("query contains non-finite values")
            anchor_vec = query.mean(axis=0)
        else:
            raise ValueError(f"query must be 1D or 2D, got shape {query.shape}")
        if not np.all(np.isfinite(anchor_vec)):
            raise ValueError("query pooled to a non-finite anchor")

    p_anchor = _anchor_pool(Hv, anchor_vec, cfg.n_anchor, cfg.tau)

    if cfg.mode == "convex":
        a_h, a_t, a_a = cfg.alpha
        out = a_h * p_head + a_t * p_tail + a_a * p_anchor
    else:
        out = np.concatenate([p_head, p_tail, p_anchor])

    if not np.all(np.isfinite(out)):
        raise ValueError("pooled output contains non-finite values")
    return out


class PartitionAnchorPooler:
    """Head/tail/anchor partition pooling, Spec 20 S2.3 path A."""

    def __init__(self, config: Optional[PartitionAnchorConfig] = None) -> None:
        self.config = config or PartitionAnchorConfig()

    def pool(self, H, attention_mask=None, query=None):
        is_torch = _is_torch_tensor(H)
        H_np = _to_numpy(H)
        mask_np = _to_numpy(attention_mask) if attention_mask is not None else None
        query_np = _to_numpy(query) if query is not None else None

        out_np = _pool_core(H_np, mask_np, query_np, self.config)

        if is_torch:
            import torch
            return torch.as_tensor(out_np, dtype=H.dtype, device=H.device)
        return out_np.astype(H_np.dtype if np.issubdtype(H_np.dtype, np.floating) else np.float64)
