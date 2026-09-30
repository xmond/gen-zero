"""Zero task head: a supervised bilinear scorer over the label-free 64-D manifold.

``score_k = (W @ z0) . zc_k``

``W`` is the only learned parameter. ``z0``/``zc`` are the same unit-sphere
manifold coordinates every other Zero component uses (`ZeroManifold.project`,
fit label-free). Training reads labels from ``benchmarks/data/calibration_clean_16.jsonl``
(the trainer lives in gen-zero-research), a split that is proven disjoint
from the frozen 930-record test set (`benchmarks/data/CALIBRATION_SPLIT.md`).

At inference this module reads nothing but two vectors: no task id, no text,
no label. ``decide()`` in ``zero_runtime.py`` does not tell it which of the
13 benchmarks a record belongs to.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = ["HEAD_VERSION", "ZeroTaskHead"]

HEAD_VERSION = 1


@dataclasses.dataclass(frozen=True)
class ZeroTaskHead:
    weight: np.ndarray  # (dim, dim), float32
    provenance: Dict

    @property
    def dim(self) -> int:
        return int(self.weight.shape[0])

    def score(self, prompt_state: np.ndarray, candidate_states: np.ndarray) -> np.ndarray:
        """(dim,), (K, dim) -> (K,) finite scores, higher is better. Order-invariant per-row."""
        z0 = np.asarray(prompt_state, dtype=np.float64)
        zc = np.asarray(candidate_states, dtype=np.float64)
        if z0.shape != (self.dim,):
            raise ValueError(f"prompt_state must have shape ({self.dim},)")
        if zc.ndim != 2 or zc.shape[1] != self.dim or zc.shape[0] < 1:
            raise ValueError(f"candidate_states must be (K, {self.dim})")
        if not (np.isfinite(z0).all() and np.isfinite(zc).all()):
            raise ValueError("states must be finite")
        projected = self.weight.astype(np.float64) @ z0
        scores = zc @ projected
        if not np.isfinite(scores).all():
            raise FloatingPointError("task head produced a non-finite score")
        return scores

    def save(self, path: Path) -> None:
        with open(path, "wb") as stream:
            np.savez(stream, metadata=json.dumps(self.provenance), weight=self.weight)

    @classmethod
    def load(cls, path: Path, *, encoder_id: str) -> "ZeroTaskHead":
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"metadata", "weight"}:
                raise ValueError("invalid task head schema")
            meta = json.loads(str(data["metadata"]))
            if meta.get("version") != HEAD_VERSION or meta.get("encoder_id") != encoder_id:
                raise ValueError("task head version or frozen encoder mismatch")
            if meta.get("split") not in ("train", "calibration"):
                raise ValueError("task head provenance split is not train/calibration")
            if not meta.get("manifold_sha256"):
                raise ValueError("task head provenance missing manifold_sha256")
            weight = data["weight"]
        if weight.dtype != np.float32 or weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
            raise ValueError("task head weight must be a square float32 matrix")
        if not np.isfinite(weight).all():
            raise ValueError("task head weight must be finite")
        return cls(weight=weight, provenance=meta)

    @classmethod
    def fit(cls, z0s: Sequence[np.ndarray], candidate_blocks: Sequence[np.ndarray],
            positive_indices: Sequence[int], *, manifold_sha256: str, source: str, split: str,
            encoder_id: str, weight_decay: float = 1e-2, epochs: int = 300, lr: float = 0.05,
            seed: int = 0) -> Tuple["ZeroTaskHead", List[float]]:
        """Per-record softmax cross-entropy fit. Candidate count K may vary record to record.

        Regularized toward ``W = I`` (plain cosine similarity in the manifold, the
        unsupervised baseline every other Zero score already uses), so a
        ``weight_decay`` that dominates the data term degrades gracefully back
        to that baseline rather than to an arbitrary matrix.
        """
        import torch
        if split not in ("train", "calibration"):
            raise ValueError("the task head may only be fitted on train/calibration splits")
        if not source or not encoder_id or not manifold_sha256:
            raise ValueError("source, encoder_id and manifold_sha256 are required")
        n = len(z0s)
        if n == 0 or len(candidate_blocks) != n or len(positive_indices) != n:
            raise ValueError("z0s, candidate_blocks and positive_indices must have the same nonzero length")
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be positive finite")
        dim = int(np.asarray(z0s[0]).shape[0])
        for z0, zc, pos in zip(z0s, candidate_blocks, positive_indices):
            z0a, zca = np.asarray(z0), np.asarray(zc)
            if z0a.shape != (dim,) or zca.ndim != 2 or zca.shape[1] != dim or zca.shape[0] < 2:
                raise ValueError("every record needs a (dim,) prompt state and a (K>=2, dim) candidate block")
            if not (0 <= int(pos) < zca.shape[0]):
                raise ValueError("positive_indices must index into that record's own candidate block")
            if not (np.isfinite(z0a).all() and np.isfinite(zca).all()):
                raise ValueError("features must be finite")

        from collections import defaultdict

        torch.manual_seed(seed)
        eye = torch.eye(dim, dtype=torch.float64)
        w = torch.eye(dim, dtype=torch.float64, requires_grad=True)

        by_k: dict[int, list[tuple[np.ndarray, np.ndarray, int]]] = defaultdict(list)
        for z0, zc, pos in zip(z0s, candidate_blocks, positive_indices):
            by_k[zc.shape[0]].append((z0, zc, pos))

        groups = []
        for k_val, items in by_k.items():
            z_grp = torch.tensor(np.stack([it[0] for it in items], axis=0), dtype=torch.float64)
            c_grp = torch.tensor(np.stack([it[1] for it in items], axis=0), dtype=torch.float64)
            p_grp = torch.tensor([it[2] for it in items], dtype=torch.long)
            groups.append((z_grp, c_grp, p_grp))

        optimizer = torch.optim.Adam([w], lr=lr)
        history: List[float] = []
        for _ in range(epochs):
            optimizer.zero_grad()
            data_loss = torch.zeros((), dtype=torch.float64)
            for z_grp, c_grp, p_grp in groups:
                wz = z_grp @ w.T
                logits = torch.bmm(c_grp, wz.unsqueeze(-1)).squeeze(-1)
                data_loss = data_loss + torch.nn.functional.cross_entropy(logits, p_grp, reduction="sum")
            loss = data_loss / n + weight_decay * (w - eye).square().sum()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite task head loss")
            loss.backward()
            if w.grad is None or not torch.isfinite(w.grad).all():
                raise FloatingPointError("nonfinite task head gradient")
            optimizer.step()
            history.append(float(loss.detach()))

        weight = w.detach().numpy().astype(np.float32)
        if not np.isfinite(weight).all():
            raise FloatingPointError("task head training produced non-finite weights")
        digest = hashlib.sha256()
        for z0, zc, pos in zip(z0s, candidate_blocks, positive_indices):
            digest.update(np.ascontiguousarray(z0, dtype=np.float32).tobytes())
            digest.update(np.ascontiguousarray(zc, dtype=np.float32).tobytes())
            digest.update(np.int64(pos).tobytes())
        provenance = dict(version=HEAD_VERSION, dim=dim, encoder_id=encoder_id,
                          manifold_sha256=manifold_sha256, source=source, split=split,
                          samples=n, weight_decay=weight_decay, epochs=epochs, lr=lr,
                          seed=seed, training_sha256=digest.hexdigest(),
                          initial_loss=history[0], final_loss=history[-1])
        return cls(weight=weight, provenance=provenance), history
