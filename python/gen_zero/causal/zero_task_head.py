"""Zero task head: a static bilinear scorer over the label-free 64-D manifold.

``score_k = (W @ z0) . zc_k``

``W`` is a fixed matrix loaded from an exported artifact; this package never
fits it. ``z0``/``zc`` are the same unit-sphere manifold coordinates every
other Zero component uses (`ZeroManifold.project`, fit label-free). The
artifact provenance must name a train/calibration split.

At inference this module reads nothing but two vectors: no task id, no text,
no label. ``decide()`` in ``zero_runtime.py`` does not tell it which of the
13 benchmarks a record belongs to.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Dict

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
