"""Manifold Anchor Distiller: offline conformal compression of GCCA features.

Fits a fixed, orthonormal ``output_dim``-dimensional linear projection
(``P``, shape ``(output_dim, input_dim)``) from a one-shot batch of
``input_dim``-dimensional GCCA features via thin SVD on the centered data
matrix. ``P`` satisfies ``P @ P.T == I`` by construction (it is a slice of
the right-singular-vector matrix ``Vt``), which is what preserves pairwise
angles (conformality) between the ambient space and the projected space
when the data's intrinsic rank is <= ``output_dim``.

This is an **offline, non-parametric** PCA-style projection: it is fit once
on a frozen batch of features and has no notion of drift, no online update,
and no adaptation. If the upstream feature extractor (or its distribution)
changes, callers must refit a new ``ManifoldAnchorDistiller`` from scratch --
this module will not detect staleness on its own.

The fitted basis intentionally does *not* whiten by the singular values:
whitening a basis (dividing rows of ``P`` by their singular value) breaks
``P @ P.T == I`` and, with it, angle preservation. The singular values are
still recorded as a separate "variance scaling factor" (metadata) for
callers that want it, but ``project`` never folds it into ``P``.

The exported ``.npz`` artifact (``P``, ``mu``, ``singular_values`` plus a
JSON metadata blob with a payload sha256) feeds the Rust ``LatentState<128>``
(``CompressedLatent``) type via ``nanocore_bridge.NanocoreAnchorBridge``,
which this module does not depend on -- see that module and
``crates/gen-zero-service/tests/test_nanocore_live.rs`` for the actual
consumption path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np

__all__ = ["ManifoldAnchorDistiller"]


def _finite_2d(value, name, *, dtype=np.float64) -> np.ndarray:
    a = np.asarray(value, dtype=dtype)
    if a.ndim != 2 or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: must be a nonempty finite 2D array")
    return a


class ManifoldAnchorDistiller:
    """Fits a fixed orthonormal ``(output_dim, input_dim)`` projection basis.

    Usage: ``fit(X_train)`` once on a frozen feature batch, then ``project``
    or ``evaluate_distortion`` on new batches, then ``save``/``load`` the
    fitted basis as a sha256-checked ``.npz`` artifact.
    """

    FORMAT_VERSION: int = 2

    def __init__(self, input_dim: int, output_dim: int = 128):
        if not isinstance(input_dim, (int, np.integer)) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if not isinstance(output_dim, (int, np.integer)) or output_dim <= 0:
            raise ValueError("output_dim must be a positive integer")
        if output_dim > input_dim:
            raise ValueError("output_dim must be less than or equal to input_dim")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        self.space = None
        self.gcca = None
        self._fitted = False
        self.mu: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.singular_values: Optional[np.ndarray] = None
        self._energy_retained_ratio: Optional[float] = None
        self._train_data_sha256: Optional[str] = None
        self._created_at: Optional[str] = None

    def attach_gcca(self, path):
        if self._fitted:
            raise ValueError("cannot replace GCCA after anchor fitting")
        with np.load(path, allow_pickle=False) as data:
            manifest_text = str(data["manifest"])
            artifact_digest = hashlib.sha256(manifest_text.encode())
            for key in sorted(set(data.files) - {"manifest", "artifact_sha256"}):
                value = np.ascontiguousarray(data[key])
                artifact_digest.update(key.encode())
                artifact_digest.update(str(value.dtype).encode())
                artifact_digest.update(str(value.shape).encode())
                artifact_digest.update(value.tobytes())
            if artifact_digest.hexdigest() != str(data["artifact_sha256"]):
                raise ValueError("GCCA artifact checksum mismatch")
            space = json.loads(manifest_text)
            payload = {k: np.asarray(data[k], dtype=np.float64) for k in ("W", "mean", "scale")}
        digest = hashlib.sha256()
        for key in sorted(payload):
            digest.update(np.ascontiguousarray(payload[key]).tobytes())
        if digest.hexdigest() != space.get("GCCA_map"):
            raise ValueError("GCCA map checksum mismatch")
        if not all(isinstance(space.get(k), str) and space[k].strip()
                   for k in ("source_model", "layer", "norm", "GCCA_map")):
            raise ValueError("incomplete source space manifest")
        w, mean, scale = payload["W"], payload["mean"], payload["scale"]
        if (w.ndim != 2 or w.shape[1] != self.input_dim or mean.shape != (w.shape[0],)
                or scale.shape != mean.shape or np.any(scale <= 0)
                or not all(np.isfinite(v).all() for v in payload.values())):
            raise ValueError("invalid GCCA transform dimensions or values")
        self.gcca, self.space = payload, space
        return self

    def transform_source(self, samples, space):
        """Project raw SOURCE-space samples through the attached GCCA transform.

        ``space`` identifies the RAW feature's provenance only -- exactly
        ``{source_model, layer, norm}`` (see ``feature_space.py``). It never
        carries ``GCCA_map``: that field identifies the GCCA transform itself,
        which is a property of this anchor's manifest, not of the raw
        upstream feature file (a feature store is written before any GCCA
        transform exists, so it cannot possibly know that transform's
        checksum). The GCCA_map identity is instead checked by
        ``NanocoreAnchorBridge`` when it compares this anchor's full space
        against an independently supplied core manifest.
        """
        self._require_fitted()
        expected = {k: self.space[k] for k in ("source_model", "layer", "norm")} if self.space else None
        if self.gcca is None or space != expected:
            raise ValueError("source space identity mismatch or missing GCCA transform")
        x = _finite_2d(samples, "source samples")
        if x.shape[1] != self.gcca["W"].shape[0]:
            raise ValueError("source width mismatch")
        return self.project(((x - self.gcca["mean"]) / self.gcca["scale"]) @ self.gcca["W"])

    def bind_core(self, core: str, domain_id: int):
        self._require_fitted()
        if self.space is None or not isinstance(core, str) or not core.strip():
            raise ValueError("core binding requires GCCA provenance and a core artifact digest")
        if len(core) != 64 or any(c not in "0123456789abcdef" for c in core):
            raise ValueError("core must be a SHA256 digest")
        if isinstance(domain_id, bool) or not isinstance(domain_id, int) or not 0 <= domain_id <= 0xffffffff:
            raise ValueError("core domain must be u32")
        basis = hashlib.sha256(self.P.tobytes() + self.mu.tobytes()).hexdigest()
        self.space = {**self.space, "anchor_basis": basis, "core": core, "domain_id": domain_id}
        return self

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("ManifoldAnchorDistiller: call fit() before this method")

    def fit(self, X_train) -> "ManifoldAnchorDistiller":
        if self._fitted:
            raise ValueError("anchor is already fitted; create a new artifact to refit")
        x = _finite_2d(X_train, "X_train")
        if x.shape[1] != self.input_dim:
            raise ValueError(f"X_train must have shape (*, {self.input_dim})")
        n = x.shape[0]
        if n < self.output_dim + 1:
            raise ValueError(
                f"X_train needs at least {self.output_dim + 1} rows "
                f"(output_dim + 1, centering removes one rank of freedom), got {n}"
            )

        mu = x.mean(axis=0)
        xc = x - mu

        # Thin SVD on the (N, D) centered matrix -- O(N^2 D), never build the
        # (D, D) covariance matrix (that would be O(D^3) time and ~512MB for
        # D=8192, which is exactly what this module avoids).
        _, s, vt = np.linalg.svd(xc, full_matrices=False)

        if not np.all(np.isfinite(s)):
            raise ValueError("numerical overflow in energy computation")

        if s[0] <= 0.0:
            raise ValueError("degenerate input: all singular values are zero")
        if s.shape[0] < self.output_dim or np.any(s[: self.output_dim] < 1e-10 * s[0]):
            raise ValueError(
                "degenerate/rank-deficient input: one or more of the top "
                f"{self.output_dim} singular values is smaller than 1e-10 * "
                "the leading singular value; fitting a basis on near-zero-"
                "signal directions is unsafe"
            )

        p = vt[: self.output_dim]
        if not np.allclose(p @ p.T, np.eye(self.output_dim), atol=1e-8):
            raise ValueError("fitted basis is not orthonormal (P @ P.T != I); internal SVD invariant violated")

        # This division can genuinely overflow/produce NaN on adversarial-scale
        # singular values; that is exactly what the isfinite check below is
        # for, so the warning is expected and suppressed at its source, not
        # globally -- errstate scopes only this one expression.
        with np.errstate(over="ignore", invalid="ignore"):
            energy_retained = float(np.sum(s[: self.output_dim] ** 2) / np.sum(s ** 2))
        if not np.isfinite(energy_retained):
            raise ValueError("numerical overflow in energy computation")

        self.mu = mu.astype(np.float64)
        self.P = p.astype(np.float64)
        self.singular_values = s[: self.output_dim].astype(np.float64)
        self._energy_retained_ratio = energy_retained
        self._train_data_sha256 = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
        self._created_at = datetime.now(timezone.utc).isoformat()
        self._fitted = True
        return self

    def project(self, X) -> np.ndarray:
        self._require_fitted()
        x = _finite_2d(X, "X")
        if x.shape[1] != self.input_dim:
            raise ValueError(f"X must have shape (*, {self.input_dim})")
        xc = x - self.mu
        if not np.all(np.isfinite(xc)):
            raise ValueError("project: centering overflowed to non-finite values")
        out = xc @ self.P.T
        if not np.all(np.isfinite(out)):
            raise ValueError("project: projection overflowed to non-finite values")
        return out

    def evaluate_distortion(self, X_val) -> Dict[str, float]:
        self._require_fitted()
        x = _finite_2d(X_val, "X_val")
        if x.shape[1] != self.input_dim:
            raise ValueError(f"X_val must have shape (*, {self.input_dim})")

        xc = x - self.mu
        z = self.project(x)

        m = xc.shape[0]
        if m < 3:
            raise ValueError("evaluate_distortion requires at least 3 samples")

        xc_norms = np.linalg.norm(xc, axis=1)
        z_norms = np.linalg.norm(z, axis=1)
        if np.any(xc_norms <= 1e-12) or np.any(z_norms <= 1e-12):
            raise ValueError("X_val has near-zero-norm row(s) after centering: cosine similarity is undefined")

        iu = np.triu_indices(m, k=1)

        xc_unit = xc / xc_norms[:, None]
        z_unit = z / z_norms[:, None]
        cos_ambient = (xc_unit @ xc_unit.T)[iu]
        cos_projected = (z_unit @ z_unit.T)[iu]

        with np.errstate(invalid="ignore", divide="ignore"):
            correlation = float(np.corrcoef(cos_ambient, cos_projected)[0, 1])
        if not np.isfinite(correlation):
            raise ValueError(
                "evaluate_distortion: pairwise cosine correlation is non-finite "
                "(degenerate/zero-variance similarity distribution); refusing to "
                "return a silent NaN"
            )

        return {
            "pairwise_cosine_correlation": correlation,
            "energy_retained_ratio": self._energy_retained_ratio,
        }

    def save(self, path) -> Dict[str, object]:
        self._require_fitted()
        payload = {
            "P": np.asarray(self.P, dtype=np.float64),
            "mu": np.asarray(self.mu, dtype=np.float64),
            "singular_values": np.asarray(self.singular_values, dtype=np.float64),
        }
        if self.gcca is not None:
            payload.update({"gcca_" + k: v for k, v in self.gcca.items()})
        digest = hashlib.sha256()
        for name in sorted(payload):
            digest.update(np.ascontiguousarray(payload[name]).tobytes())
        meta = dict(
            space=self.space,
            format_version=self.FORMAT_VERSION,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            train_data_sha256=self._train_data_sha256,
            created_at=self._created_at,
            energy_retained_ratio=self._energy_retained_ratio,
            payload_sha256=digest.hexdigest(),
        )
        meta["manifest_sha256"] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
        with open(path, "wb") as f:
            np.savez(f, metadata=json.dumps(meta), **payload)
        return meta

    @classmethod
    def load(cls, path) -> "ManifoldAnchorDistiller":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            p = np.asarray(data["P"], dtype=np.float64)
            mu = np.asarray(data["mu"], dtype=np.float64)
            singular_values = np.asarray(data["singular_values"], dtype=np.float64)
            gcca = {k: np.asarray(data["gcca_" + k], dtype=np.float64) for k in ("W", "mean", "scale")} if "gcca_W" in data else None

        stored_manifest_digest = meta.pop("manifest_sha256", None)
        format_version = meta.get("format_version")
        if (
            not isinstance(format_version, int)
            or isinstance(format_version, bool)
            or format_version != cls.FORMAT_VERSION
        ):
            raise ValueError(
                f"ManifoldAnchorDistiller artifact format_version mismatch: "
                f"expected {cls.FORMAT_VERSION}, got {format_version!r}"
            )

        payload = {"P": p, "mu": mu, "singular_values": singular_values}
        if gcca is not None:
            payload.update({"gcca_" + k: v for k, v in gcca.items()})
        digest = hashlib.sha256()
        for name in sorted(payload):
            digest.update(np.ascontiguousarray(payload[name]).tobytes())
        if digest.hexdigest() != meta["payload_sha256"]:
            raise ValueError("ManifoldAnchorDistiller artifact payload_sha256 mismatch: corrupted or tampered file")

        output_dim = int(meta["output_dim"])
        input_dim = int(meta["input_dim"])
        if p.shape != (output_dim, input_dim):
            raise ValueError(
                f"loaded P has shape {p.shape}, expected ({output_dim}, {input_dim}); "
                "artifact is corrupted"
            )
        if not np.allclose(p @ p.T, np.eye(output_dim), atol=1e-6):
            raise ValueError("loaded P is not orthonormal (P @ P.T != I); artifact is corrupted or tampered")

        if mu.shape != (input_dim,) or not np.all(np.isfinite(mu)):
            raise ValueError(
                f"loaded mu has shape {mu.shape} (expected ({input_dim},)) or "
                "contains non-finite values; artifact is corrupted"
            )
        if singular_values.shape != (output_dim,) or not np.all(np.isfinite(singular_values)):
            raise ValueError(
                f"loaded singular_values has shape {singular_values.shape} "
                f"(expected ({output_dim},)) or contains non-finite values; "
                "artifact is corrupted"
            )

        energy_retained_ratio = float(meta["energy_retained_ratio"])
        if not np.isfinite(energy_retained_ratio) or not (0.0 <= energy_retained_ratio <= 1.0):
            raise ValueError(
                f"loaded energy_retained_ratio {energy_retained_ratio!r} is not "
                "finite or not in [0, 1]; artifact is corrupted"
            )

        obj = cls(input_dim=input_dim, output_dim=output_dim)
        obj.space = meta.get("space")
        obj.gcca = gcca
        if obj.space is not None and "anchor_basis" in obj.space:
            if obj.space["anchor_basis"] != hashlib.sha256(p.tobytes() + mu.tobytes()).hexdigest():
                raise ValueError("anchor basis identity mismatch")
        obj.mu = mu
        obj.P = p
        obj.singular_values = singular_values
        obj._energy_retained_ratio = energy_retained_ratio
        obj._train_data_sha256 = meta["train_data_sha256"]
        obj._created_at = meta["created_at"]
        if stored_manifest_digest != hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest():
            raise ValueError("artifact manifest checksum mismatch")
        obj._fitted = True
        return obj


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: fit a ManifoldAnchorDistiller on a real feature block.

    Loads an ``.npz`` file, pulls the array at ``--block``, uses the
    fit/eval partition recorded by GCCA (never resplit or refit GCCA), fits the distiller, runs
    ``evaluate_distortion`` on the held-out split, saves the artifact, and
    prints a single JSON summary line to stdout.
    """
    parser = argparse.ArgumentParser(
        description="Fit a ManifoldAnchorDistiller on real GCCA features and save the artifact."
    )
    parser.add_argument("--gcca-transform")
    core_group = parser.add_mutually_exclusive_group(required=False)
    core_group.add_argument("--core-sha256", help="explicit 64-hex-char sha256 digest of the core artifact to bind")
    core_group.add_argument(
        "--core", choices=["default-etf-head"],
        help="bind to the digest of a fresh default ActionETFChoiceHead(hidden_dim=128, action_dim=128), "
             "constructed exactly as client.py's GenZero.__init__ does. NOTE: that head is UNTRAINED "
             "(its state_projection is seeded random, see choice_head.py) -- this binds an identity, "
             "not a claim of decision quality.",
    )
    parser.add_argument("--domain-id", type=int)
    parser.add_argument("--features", required=True, help="Path to an .npz file containing the feature block.")
    parser.add_argument("--block", default="train_full", help="npz key to read as the feature matrix.")
    parser.add_argument("--output-dim", type=int, default=128, help="Target projection dimension.")
    parser.add_argument("--out", required=True, help="Path to save the fitted artifact .npz.")
    args = parser.parse_args(argv)

    start = time.monotonic()

    with np.load(args.features, allow_pickle=False) as data:
        block = np.asarray(data[args.block], dtype=np.float64)

    features_sha256 = hashlib.sha256(np.ascontiguousarray(block).tobytes()).hexdigest()

    n_total = block.shape[0]
    input_dim = block.shape[1]

    if not args.gcca_transform or not (args.core_sha256 or args.core) or args.domain_id is None:
        raise ValueError("--gcca-transform, one of --core-sha256/--core, and --domain-id are required")
    if args.core == "default-etf-head":
        # Lazy import: avoids a causal -> nanocore module-load dependency for
        # every other caller of this module that never touches --core.
        from gen_zero.nanocore.choice_head import ActionETFChoiceHead
        core_sha256 = ActionETFChoiceHead(hidden_dim=128, action_dim=128).core_digest()
    else:
        core_sha256 = args.core_sha256
    with np.load(args.gcca_transform, allow_pickle=False) as data:
        input_dim = data["W"].shape[1]
        if str(data["source_data_sha256"]) != features_sha256:
            raise ValueError("GCCA split belongs to different source data")
        fit_idx, eval_idx = data["fit_indices"], data["eval_indices"]
        if (fit_idx.dtype.kind not in "iu" or eval_idx.dtype.kind not in "iu"
                or fit_idx.ndim != 1 or eval_idx.ndim != 1
                or len(np.unique(np.concatenate([fit_idx, eval_idx]))) != len(fit_idx) + len(eval_idx)
                or np.any(np.concatenate([fit_idx, eval_idx]) < 0)
                or np.any(np.concatenate([fit_idx, eval_idx]) >= n_total)):
            raise ValueError("invalid GCCA fit/eval partition")
        n_fit, n_eval = len(fit_idx), len(eval_idx)
    distiller = ManifoldAnchorDistiller(input_dim=input_dim, output_dim=args.output_dim)
    distiller.attach_gcca(args.gcca_transform)
    def transform(x):
        return ((x - distiller.gcca["mean"]) / distiller.gcca["scale"]) @ distiller.gcca["W"]
    distiller.fit(transform(block[fit_idx]))
    distortion = distiller.evaluate_distortion(transform(block[eval_idx]))
    distiller.bind_core(core_sha256, args.domain_id)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_meta = distiller.save(str(out_path))

    core_manifest_path = out_path.parent / (out_path.stem + ".core_manifest.json")
    core_manifest_path.write_text(json.dumps(save_meta["space"], sort_keys=True, indent=2) + "\n")

    elapsed_seconds = time.monotonic() - start

    summary = {
        "features": args.features,
        "features_sha256": features_sha256,
        "block": args.block,
        "n_total": n_total,
        "n_fit": n_fit,
        "n_eval": n_eval,
        "input_dim": input_dim,
        "output_dim": args.output_dim,
        "split_source": str(args.gcca_transform),
        "energy_retained_ratio": distortion["energy_retained_ratio"],
        "pairwise_cosine_correlation": distortion["pairwise_cosine_correlation"],
        "payload_sha256": save_meta["payload_sha256"],
        "space": save_meta["space"],
        "core_manifest": str(core_manifest_path),
        "elapsed_seconds": elapsed_seconds,
        "loadavg": list(os.getloadavg()),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
