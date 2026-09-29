"""Universal Causal Manifold Extractor: the four pillars of doc 07.

Implements, as plain numpy with no text/label inspection anywhere:

  * Pillar 4 -- `StreamingCovarianceAccumulator`: O(d^2) streaming second-moment
    accumulation of shifted outer products whose eigendecomposition
    is mathematically identical to a full-batch PCA/SVD (Kornblith et al. style
    covariance identity: sum-of-outer-products commutes with batching). Memory
    is bounded by the (d, d) buffer, independent of how many samples pass
    through it.
  * Pillar 1 -- `PhaseTransitionLayerExtractor`: linear Centered Kernel
    Alignment (CKA) between layer activation matrices, used to locate the
    mid-depth concept manifold and the deep causal-collapse manifold.
  * Pillar 2 -- `CounterfactualGridSampler`: differential displacement vectors
    between paired interventional samples (x, do(x')) plus local manifold
    geometry (cosine spread, displacement norm, nearest-neighbor covering gap).
  * Pillar 3 -- `LyapunovPhaseSpaceReconstructor`: least-squares fit of a
    continuous linear system dz/dt = A z + B u from an observed state
    trajectory, with a strict Hurwitz check. Unstable fits are rejected,
    never altered to manufacture a stability certificate.

`UniversalManifoldExtractionPipeline` wires the four pillars into one
end-to-end `extract_from_stream(...)` call and an `export_codebook(...)`
artifact writer. The artifact here is a sha256-checked ``.npz`` (arrays plus a
JSON metadata blob); it is not the byte-identical `causal_codebook.bin` format
produced by `knowledge_compiler.py` -- consumers that need that exact binary
layout should feed this module's `A`/`B`/`U_k` arrays into that compiler.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "StreamingCovarianceAccumulator",
    "PhaseTransitionLayerExtractor",
    "CounterfactualGridSampler",
    "LyapunovPhaseSpaceReconstructor",
    "UniversalManifoldExtractionPipeline",
]


def _finite_2d(value, name, *, dtype=np.float64) -> np.ndarray:
    a = np.asarray(value, dtype=dtype)
    if a.ndim != 2 or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: must be a nonempty finite 2D array")
    return a


# --------------------------------------------------------------------------
# Pillar 4: streaming covariance
# --------------------------------------------------------------------------


class StreamingCovarianceAccumulator:
    """Bounded-memory streaming second-moment accumulator.

    Stores shifted second moments, a shifted sum, and a fixed shift vector.
    The default shift is the first sample, removing large common offsets
    before multiplication. Memory is O(dim**2), independent of sample count.
    Accuracy still depends on the data spread relative to the chosen shift;
    a supplied shift should be near the observations.
    """

    def __init__(self, dim: int, *, dtype=np.float64, shift=None):
        if not isinstance(dim, (int, np.integer)) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.dim = int(dim)
        self.dtype = np.dtype(dtype)
        if not np.issubdtype(self.dtype, np.floating):
            raise ValueError("dtype must be a real floating-point dtype")
        self._shift = np.zeros(self.dim, dtype=self.dtype)
        if shift is not None:
            supplied_shift = np.asarray(shift, dtype=self.dtype)
            if supplied_shift.shape != (self.dim,) or not np.isfinite(supplied_shift).all():
                raise ValueError(f"shift must be a finite vector of shape ({self.dim},)")
            self._shift[:] = supplied_shift
        self._shift_initialized = shift is not None
        self._c = np.zeros((self.dim, self.dim), dtype=dtype)
        self._sum = np.zeros(self.dim, dtype=dtype)
        self._n = 0

    def update(self, x_batch) -> None:
        """Fold one batch into the running accumulator, then forget it."""
        x = _finite_2d(x_batch, "x_batch", dtype=self.dtype)
        if x.shape[1] != self.dim:
            raise ValueError(f"x_batch must have shape (*, {self.dim})")
        shift = self._shift if self._shift_initialized else x[0]
        # Fail explicitly on arithmetic overflow, without partially updating state.
        with np.errstate(over="raise", invalid="raise"):
            shifted = x - shift
            new_c = self._c + shifted.T @ shifted
            new_sum = self._sum + shifted.sum(axis=0)
        if not (np.isfinite(new_c).all() and np.isfinite(new_sum).all()):
            raise FloatingPointError("non-finite shifted covariance accumulation")
        self._shift[:] = shift
        self._shift_initialized = True
        self._c[:] = new_c
        self._sum[:] = new_sum
        self._n += x.shape[0]

    @property
    def n_samples(self) -> int:
        return self._n

    @property
    def mean(self) -> np.ndarray:
        if self._n == 0:
            raise ValueError("no data has been accumulated yet")
        return self._shift + self._sum / self._n

    def covariance(self) -> np.ndarray:
        """Population covariance from shifted moments (ddof=0)."""
        if self._n < 2:
            raise ValueError("need at least 2 samples to form a covariance")
        mean_shifted = self._sum / self._n
        return self._c / self._n - np.outer(mean_shifted, mean_shifted)

    def compute_principal_basis(self, k: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """Top-``k`` orthogonal eigenbasis ``U_k`` (dim, k) with eigenvalues.

        Mathematically equivalent to taking the top-k left singular vectors
        of the centered full data matrix, since ``eigh(X_c^T X_c / N)`` and
        ``svd(X_c / sqrt(N))`` share the same eigenvectors/eigenvalues for a
        symmetric PSD matrix.
        """
        if not isinstance(k, (int, np.integer)) or k <= 0 or k > self.dim:
            raise ValueError(f"k must be an integer in [1, {self.dim}]")
        cov = self.covariance()
        eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order, symmetric solver
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order][:k]
        eigvecs = eigvecs[:, order][:, :k]
        return eigvecs, eigvals

    def memory_bytes(self) -> int:
        """Persistent NumPy buffer bytes (excluding Python object overhead)."""
        return int(self._c.nbytes + self._sum.nbytes + self._shift.nbytes)


# --------------------------------------------------------------------------
# Pillar 1: phase-transition layer profiling
# --------------------------------------------------------------------------


class PhaseTransitionLayerExtractor:
    """Locates the concept-formation and causal-collapse depth bands via CKA."""

    @staticmethod
    def linear_cka(x, y) -> float:
        """Linear-kernel Centered Kernel Alignment between two activation sets.

        ``x``, ``y``: (n_samples, features_x/features_y), same n_samples.
        Uses the closed-form equivalent of Kornblith et al. (2019):
        ``CKA = ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F)``,
        which avoids forming an (n, n) Gram matrix.
        """
        x = _finite_2d(x, "x")
        y = _finite_2d(y, "y")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")
        if x.shape[0] < 2:
            raise ValueError("need at least 2 samples to compute CKA")
        xc = x - x.mean(axis=0, keepdims=True)
        yc = y - y.mean(axis=0, keepdims=True)
        cross = xc.T @ yc
        numerator = float(np.sum(cross * cross))
        xx = xc.T @ xc
        yy = yc.T @ yc
        denom = float(np.sqrt(np.sum(xx * xx)) * np.sqrt(np.sum(yy * yy)))
        if denom <= 1e-30:
            raise ValueError("degenerate input: zero-variance features")
        return numerator / denom

    def compute_cka_matrix(self, layer_activations: Sequence[np.ndarray]) -> np.ndarray:
        layers = list(layer_activations)
        if len(layers) < 2:
            raise ValueError("need at least 2 layers to compute a CKA matrix")
        n = len(layers)
        k = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            k[i, i] = 1.0
            for j in range(i + 1, n):
                v = self.linear_cka(layers[i], layers[j])
                k[i, j] = k[j, i] = v
        return k

    def detect_phase_transitions(
        self, layer_activations: Sequence[np.ndarray], *, period: Optional[int] = None
    ) -> Dict[str, object]:
        """Locate the break at edge i (layers i and i+1).

        With a period, rank edges by CKA minus their phase-group mean and
        report same-phase CKA at lag period. This returns a candidate edge,
        not a significance-tested phase transition. Every edge phase must have at
        least two observations; otherwise detrending is not identifiable.
        """
        layers = list(layer_activations)
        if len(layers) < 2:
            raise ValueError("need at least 2 layers to detect a phase transition")
        if period is not None:
            if isinstance(period, (bool, np.bool_)) or not isinstance(period, (int, np.integer)) or period <= 0:
                raise ValueError("period must be a positive integer")
            if len(layers) - 1 < 2 * period:
                raise ValueError("need at least two edges per phase for periodic detrending")
        consecutive = np.array(
            [self.linear_cka(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )
        report = {"consecutive_cka": consecutive, "transition_index": int(np.argmin(consecutive))}
        if period is not None:
            residuals = consecutive.copy()
            for phase in range(period):
                residuals[phase::period] -= consecutive[phase::period].mean()
            report.update(
                detrended_residuals=residuals,
                in_phase_cka=np.array([
                    self.linear_cka(layers[i], layers[i + period])
                    for i in range(len(layers) - period)
                ]),
                period=int(period),
                transition_index=int(np.argmin(residuals)),
            )
        return report

    def extract_concept_and_causal_manifolds(
        self, layer_activations: Sequence[np.ndarray], *, mid_fraction: float = 0.5
    ) -> Dict[str, object]:
        """Mid-depth concept manifold and final-depth causal-collapse manifold."""
        layers = list(layer_activations)
        n = len(layers)
        if n < 2:
            raise ValueError("need at least 2 layers")
        if not (0.0 < mid_fraction < 1.0):
            raise ValueError("mid_fraction must be in (0, 1)")
        concept_index = int(round(mid_fraction * (n - 1)))
        concept_index = min(max(concept_index, 0), n - 1)
        causal_index = n - 1
        return {
            "concept_layer_index": concept_index,
            "causal_layer_index": causal_index,
            "concept_manifold": layers[concept_index],
            "causal_manifold": layers[causal_index],
        }


# --------------------------------------------------------------------------
# Pillar 2: counterfactual epsilon-grid sampling
# --------------------------------------------------------------------------


class CounterfactualGridSampler:
    """Differential geometry of paired interventional samples (x, do(x'))."""

    @staticmethod
    def delta(z_x, z_x_prime) -> np.ndarray:
        zx = np.asarray(z_x, dtype=np.float64)
        zxp = np.asarray(z_x_prime, dtype=np.float64)
        if zx.shape != zxp.shape:
            raise ValueError("z_x and z_x_prime must have the same shape")
        if not (np.isfinite(zx).all() and np.isfinite(zxp).all()):
            raise ValueError("z_x and z_x_prime must be finite")
        return zx - zxp

    def pairwise_delta(self, z_x, z_x_prime) -> np.ndarray:
        zx = _finite_2d(z_x, "z_x")
        zxp = _finite_2d(z_x_prime, "z_x_prime")
        if zx.shape != zxp.shape:
            raise ValueError("z_x and z_x_prime must have matching shape (n, dim)")
        return zx - zxp

    def local_cosine_variance(self, deltas) -> float:
        """Spread of pairwise cosine similarity among displacement directions.

        Low variance means the counterfactual perturbations collapse onto a
        single direction (degenerate coverage); high variance means the
        epsilon-grid is probing genuinely distinct directions of the manifold.
        """
        d = _finite_2d(deltas, "deltas")
        norms = np.linalg.norm(d, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("degenerate zero-norm delta vector(s): perturbation collapsed")
        unit = d / norms
        sims = unit @ unit.T
        iu = np.triu_indices(len(d), k=1)
        if len(iu[0]) == 0:
            return 0.0
        return float(np.var(sims[iu]))

    def geodesic_displacement_norm(self, z_x, z_x_prime) -> float:
        """Euclidean displacement norm, used as the flat-embedding proxy for
        geodesic displacement between the factual and counterfactual states."""
        return float(np.linalg.norm(self.delta(z_x, z_x_prime)))

    def epsilon_coverage(self, samples, epsilon: float) -> Dict[str, object]:
        """Leave-one-out nearest-neighbor gap: an Hausdorff-style covering check.

        ``covered`` is True iff every sample has another sample within
        ``epsilon``, i.e. the set has no gap larger than epsilon (no
        degenerate/under-sampled region of the epsilon-grid).
        """
        s = _finite_2d(samples, "samples")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be positive finite")
        if len(s) < 2:
            raise ValueError("need at least 2 samples to compute a covering gap")
        diff = s[:, None, :] - s[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        nn = dists.min(axis=1)
        max_gap = float(nn.max())
        return {"max_nearest_neighbor_gap": max_gap, "covered": bool(max_gap <= epsilon)}


# --------------------------------------------------------------------------
# Pillar 3: Lyapunov phase-space reconstruction
# --------------------------------------------------------------------------


class LyapunovPhaseSpaceReconstructor:
    """Fits continuous or discrete dynamics and rejects unstable estimates."""

    def __init__(self, state_dim: int, control_dim: int = 0, *, system: str = "continuous", epsilon: float = 1e-8):
        if not isinstance(state_dim, (int, np.integer)) or state_dim <= 0:
            raise ValueError("state_dim must be a positive integer")
        if not isinstance(control_dim, (int, np.integer)) or control_dim < 0:
            raise ValueError("control_dim must be a non-negative integer")
        if system not in ("continuous", "discrete"):
            raise ValueError("system must be continuous or discrete")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be positive finite")
        self.system = system
        self.epsilon = float(epsilon)
        self.state_dim = int(state_dim)
        self.control_dim = int(control_dim)

    def fit(
        self,
        z_trajectory,
        u_trajectory: Optional[np.ndarray] = None,
        *,
        dt: float = 1.0,
        ridge: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return (matrix, control, stability statistic) without modifying the fit.

        Continuous statistic is max(real(eigenvalues)); discrete is radius.
        Finite differences estimate the continuous generator, not a matrix log.
        """
        z = _finite_2d(z_trajectory, "z_trajectory")
        if z.shape[1] != self.state_dim:
            raise ValueError(f"z_trajectory must have shape (T, {self.state_dim})")
        if z.shape[0] < 2:
            raise ValueError("need at least 2 time steps to fit a transition")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive finite")
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be non-negative finite")

        z_t = z[:-1]
        z_dot = (z[1:] - z[:-1]) / dt if self.system == "continuous" else z[1:]

        if self.control_dim > 0:
            if u_trajectory is None:
                raise ValueError("control_dim > 0 requires u_trajectory")
            u = _finite_2d(u_trajectory, "u_trajectory")
            if u.shape != (z.shape[0], self.control_dim):
                raise ValueError(f"u_trajectory must have shape ({z.shape[0]}, {self.control_dim})")
            u_t = u[:-1]
            design = np.concatenate([z_t, u_t], axis=1)
        else:
            if u_trajectory is not None:
                raise ValueError("control_dim == 0 but u_trajectory was provided")
            design = z_t

        # Ridge regression via row-augmented SVD least squares, not the normal
        # equations: forming design.T @ design squares the design matrix's
        # condition number, which is catastrophic for a freely-decaying
        # trajectory (empirically cond(design) ~1e8 but cond(design^T design)
        # ~1e16). np.linalg.lstsq solves the (possibly rank-deficient) system
        # directly via SVD and stays accurate at the original condition number.
        n_features = design.shape[1]
        design_aug = np.concatenate([design, np.sqrt(ridge) * np.eye(n_features)], axis=0)
        target_aug = np.concatenate([z_dot, np.zeros((n_features, self.state_dim))], axis=0)
        coef, *_ = np.linalg.lstsq(design_aug, target_aug, rcond=None)  # (features, state_dim)
        m = coef.T  # (state_dim, features)
        a = m[:, : self.state_dim]
        b = m[:, self.state_dim :] if self.control_dim else np.zeros((self.state_dim, 0))

        statistic = self.require_stable(a)
        return a, b, statistic

    def require_stable(self, matrix) -> float:
        a = _finite_2d(matrix, "dynamics matrix")
        if a.shape != (self.state_dim, self.state_dim):
            raise ValueError("dynamics matrix dimension mismatch")
        eig = np.linalg.eigvals(a)
        statistic = float(np.max(eig.real) if self.system == "continuous" else np.max(np.abs(eig)))
        stable = statistic < -self.epsilon if self.system == "continuous" else statistic < 1.0
        if not stable:
            raise ValueError(f"unstable {self.system} dynamics: statistic={statistic}, epsilon={self.epsilon}")
        return statistic

    def export(self, path, a, b, u_basis=None, *, extra_metadata=None) -> Dict[str, object]:
        """Write A/B (and optionally the U_k basis) as a sha256-checked npz."""
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        statistic = self.require_stable(a)
        payload = {"A": a, "B": b}
        if u_basis is not None:
            payload["U_basis"] = np.asarray(u_basis, dtype=np.float32)
        meta = dict(version=2, state_dim=self.state_dim, control_dim=self.control_dim,
                    system=self.system, epsilon=self.epsilon, stability_statistic=statistic)
        if extra_metadata:
            if set(extra_metadata) & set(meta):
                raise ValueError("extra_metadata cannot override dynamics contract")
            meta.update(extra_metadata)
        digest = hashlib.sha256()
        for name in sorted(payload):
            digest.update(np.ascontiguousarray(payload[name]).tobytes())
        meta["sha256"] = digest.hexdigest()
        with open(path, "wb") as f:
            np.savez(f, metadata=json.dumps(meta), **payload)
        return meta

    @staticmethod
    def load(path) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            arrays = {k: data[k] for k in data.files if k != "metadata"}
        digest = hashlib.sha256()
        for name in sorted(arrays):
            digest.update(np.ascontiguousarray(arrays[name]).tobytes())
        if digest.hexdigest() != meta["sha256"]:
            raise ValueError("export payload sha256 mismatch: corrupted or tampered file")
        if meta.get("version") != 2:
            raise ValueError("missing version 2 dynamics contract")
        cls = LyapunovPhaseSpaceReconstructor(meta["state_dim"], meta["control_dim"],
                                             system=meta["system"], epsilon=meta["epsilon"])
        cls.require_stable(arrays["A"])
        return meta, arrays


# --------------------------------------------------------------------------
# End-to-end pipeline
# --------------------------------------------------------------------------


class UniversalManifoldExtractionPipeline:
    """Wires the four pillars into one streaming extract -> export flow."""

    def __init__(self, dim: int, k: int = 64, *, covariance_dtype=np.float64):
        if not isinstance(dim, (int, np.integer)) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        if not isinstance(k, (int, np.integer)) or k <= 0 or k > dim:
            raise ValueError(f"k must be an integer in [1, {dim}]")
        self.dim = int(dim)
        self.k = int(k)
        self.covariance = StreamingCovarianceAccumulator(self.dim, dtype=covariance_dtype)
        self.layer_extractor = PhaseTransitionLayerExtractor()
        self.counterfactual_sampler = CounterfactualGridSampler()
        self._last_result: Optional[Dict[str, object]] = None

    def ingest_batch(self, x_batch) -> None:
        self.covariance.update(x_batch)

    def extract_from_stream(
        self,
        batch_iterator: Iterable[np.ndarray],
        *,
        layer_activations: Optional[Sequence[np.ndarray]] = None,
        period: Optional[int] = None,
        counterfactual_pairs: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        trajectory: Optional[np.ndarray] = None,
        control_trajectory: Optional[np.ndarray] = None,
        dt: float = 1.0,
    ) -> Dict[str, object]:
        if period is not None and layer_activations is None:
            raise ValueError("period requires layer_activations")
        for batch in batch_iterator:
            self.ingest_batch(batch)
        if self.covariance.n_samples < self.k:
            raise ValueError(
                f"insufficient samples ({self.covariance.n_samples}) for rank-{self.k} basis"
            )

        u_k, eigvals = self.covariance.compute_principal_basis(self.k)
        result: Dict[str, object] = {
            "U_k": u_k,
            "eigenvalues": eigvals,
            "n_samples": self.covariance.n_samples,
            "mean": self.covariance.mean,
        }

        if layer_activations is not None:
            manifolds = self.layer_extractor.extract_concept_and_causal_manifolds(layer_activations)
            transitions = self.layer_extractor.detect_phase_transitions(layer_activations, period=period)
            result["layer_profile"] = {**manifolds, **transitions}

        if counterfactual_pairs is not None:
            z_x, z_x_prime = counterfactual_pairs
            deltas = self.counterfactual_sampler.pairwise_delta(z_x, z_x_prime)
            result["counterfactual"] = {
                "deltas": deltas,
                "local_cosine_variance": self.counterfactual_sampler.local_cosine_variance(deltas),
            }

        if trajectory is not None:
            reconstructor = LyapunovPhaseSpaceReconstructor(
                state_dim=np.asarray(trajectory).shape[1],
                control_dim=(np.asarray(control_trajectory).shape[1] if control_trajectory is not None else 0),
            )
            a, b, statistic = reconstructor.fit(trajectory, control_trajectory, dt=dt)
            result["dynamics"] = {"A": a, "B": b, "system": "continuous", "stability_statistic": statistic, "reconstructor": reconstructor}

        self._last_result = result
        return result

    def export_codebook(self, path) -> Dict[str, object]:
        if self._last_result is None:
            raise ValueError("nothing to export: call extract_from_stream first")
        r = self._last_result
        payload = {
            "U_k": np.asarray(r["U_k"], dtype=np.float32),
            "mean": np.asarray(r["mean"], dtype=np.float32),
            "eigenvalues": np.asarray(r["eigenvalues"], dtype=np.float32),
        }
        if "dynamics" in r:
            payload["A"] = np.asarray(r["dynamics"]["A"], dtype=np.float32)
            r["dynamics"]["reconstructor"].require_stable(payload["A"])
            payload["B"] = np.asarray(r["dynamics"]["B"], dtype=np.float32)
        meta = dict(version=2, dim=self.dim, k=self.k, n_samples=int(r["n_samples"]))
        if "dynamics" in r:
            reconstructor = r["dynamics"]["reconstructor"]
            meta["dynamics"] = dict(system=reconstructor.system, epsilon=reconstructor.epsilon,
                                    state_dim=reconstructor.state_dim, control_dim=reconstructor.control_dim)
        digest = hashlib.sha256()
        for name in sorted(payload):
            digest.update(np.ascontiguousarray(payload[name]).tobytes())
        meta["sha256"] = digest.hexdigest()
        with open(path, "wb") as f:
            np.savez(f, metadata=json.dumps(meta), **payload)
        return meta

    @staticmethod
    def load_codebook(path) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            arrays = {k: data[k] for k in data.files if k != "metadata"}
        digest = hashlib.sha256()
        for name in sorted(arrays):
            digest.update(np.ascontiguousarray(arrays[name]).tobytes())
        if digest.hexdigest() != meta["sha256"]:
            raise ValueError("codebook payload sha256 mismatch: corrupted or tampered file")
        if "A" in arrays:
            if meta.get("version") != 2 or "dynamics" not in meta:
                raise ValueError("codebook lacks dynamics contract")
            LyapunovPhaseSpaceReconstructor(**meta["dynamics"]).require_stable(arrays["A"])
        return meta, arrays
