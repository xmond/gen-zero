"""Calibrated counterfactual-drift dynamics: A, B, W_c loaded from one npz.

State recurrence (all matrices come from a provenance-checked npz artifact):

    h_{t+1} = A h_t + B x + W_c c

  x   compact prompt feature (PCA-reduced frozen encoder state, label-free)
  c   compact counterfactual feature (PCA-reduced, label-free: a difference or
      hypothesis-minus-premise vector the same frozen encoder produced)
  A   contractive transition matrix, certified rho(A) < 1 at load time
  B   input map, W_c the counterfactual drift map

Because A is contractive the recurrence has one fixed point
h* = (I - A)^{-1} (B x + W_c c).  Calibration fits the fixed point to a fixed
centered simplex codebook (one vertex per class) by ridge regression on the
CALIBRATION split only; the inference path never receives a label.

Honesty notes (read before citing this module as "reasoning"):
  * The fixed point is a linear function of (x, c).  A only shapes the transient,
    not the answer, so with a handful of calibration samples A is *set* contractive
    and stored, not learned; the npz records that.  The value of W_c is therefore
    exactly measurable by the `W_c = 0` ablation the evaluation suite runs.
  * "Annealed Langevin" here is `AnnealedLangevinDynamicsEngine` run from h* under
    a smooth-min candidate-attractor potential; "hard affine pruning" is the
    existing `continuous_causal_reasoning_expert` (macro race + KKT projection)
    run with the codebook vertices as candidates.  Both are switchable so their
    contribution is measured, never assumed.
  * No text, task names, answer strings or labels are inspected anywhere here.
    The only label use is `fit()`, which refuses any split other than
    train/calibration.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Optional, Sequence, Tuple

import numpy as np

from ..manifold import MasterClosedFormSolver
from .annealed_langevin_engine import AnnealedLangevinDynamicsEngine, LangevinResult
from .continuous_causal_reasoning_expert import (
    ContinuousCausalReasoningResult,
    continuous_causal_reasoning_expert,
)

__all__ = [
    "ARTIFACT_VERSION",
    "CalibrationSplitError",
    "CandidateAttractorPotential",
    "CounterfactualDriftDynamics",
    "CalibratedInferenceResult",
    "fit_counterfactual_drift_dynamics",
    "simplex_codebook",
]

ARTIFACT_VERSION = 1
ARRAY_KEYS = ("A", "B", "W_c", "codebook", "x_mean", "x_basis", "c_mean", "c_basis")
ALLOWED_SPLITS = ("train", "calibration")
RHO_MAX = 0.95


class CalibrationSplitError(ValueError):
    """Raised when someone tries to fit on anything but train/calibration data."""


def _finite(value, name: str, *, dtype=np.float64) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: must be a nonempty array of finite values")
    return arr


def simplex_codebook(k: int, dim: int) -> np.ndarray:
    """Fixed centered simplex, identical to dynamics_calibrator's regression target."""
    if k < 2 or k > dim:
        raise ValueError("need 2 <= k <= dim classes")
    cb = np.zeros((k, dim), dtype=np.float64)
    cb[:, :k] = (np.eye(k) - np.ones((k, k)) / k) / np.sqrt(1.0 - 1.0 / k)
    return cb


def _pca_fit(x: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and orthonormal basis (p_raw, n) from calibration rows only."""
    mean = x.mean(axis=0)
    xc = x - mean
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    n = min(n_components, vt.shape[0])
    if n < 1:
        raise ValueError("PCA needs at least one component")
    return mean, vt[:n].T.copy()


def _spectral_radius(a: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(a))))


class CandidateAttractorPotential:
    """Smooth-min attractor potential over codebook vertices.

        V(q) = -beta^{-1} log sum_k exp(-beta * ||q - c_k||^2 / 2)

    beta -> inf recovers min_k ||q - c_k||^2 / 2 (Voronoi wells with barriers on
    the bisector hyperplanes).  `violation` is the distance by which q lies
    outside the inscribed ball of its nearest vertex's Voronoi cell (0 inside),
    which depends only on the frozen codebook, never on the trajectory.
    Buffers are allocated once here so the engine's own loop stays
    allocation-free, matching the `Potential` protocol.
    """

    def __init__(self, codebook: np.ndarray, beta: float = 4.0) -> None:
        cb = _finite(codebook, "codebook")
        if cb.ndim != 2 or cb.shape[0] < 2:
            raise ValueError("codebook must be (K>=2, dim)")
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be positive finite")
        self.codebook = cb
        self.beta = float(beta)
        self.dim = int(cb.shape[1])
        k = cb.shape[0]
        self._diff = np.zeros((k, self.dim), dtype=np.float64)
        self._d2 = np.zeros(k, dtype=np.float64)
        self._w = np.zeros(k, dtype=np.float64)
        pair = np.linalg.norm(cb[:, None, :] - cb[None, :, :], axis=2)
        pair[np.arange(k), np.arange(k)] = np.inf
        self.inscribed_radius = float(pair.min()) / 2.0

    def _weights(self, q: np.ndarray) -> None:
        np.subtract(q[None, :], self.codebook, out=self._diff)
        np.einsum("kd,kd->k", self._diff, self._diff, out=self._d2)
        np.multiply(self._d2, -0.5 * self.beta, out=self._w)
        self._w -= self._w.max()
        np.exp(self._w, out=self._w)
        self._w /= self._w.sum()

    def value(self, q: np.ndarray) -> float:
        d2 = np.sum((q[None, :] - self.codebook) ** 2, axis=1)
        m = (-0.5 * self.beta * d2).max()
        return float(-(m + np.log(np.sum(np.exp(-0.5 * self.beta * d2 - m)))) / self.beta)

    def grad_into(self, q: np.ndarray, out: np.ndarray, scratch: np.ndarray) -> np.ndarray:
        if out is q or scratch is q or out is scratch:
            raise ValueError("grad_into: q, out and scratch must be three distinct buffers")
        self._weights(q)
        # grad = sum_k w_k (q - c_k)
        np.einsum("k,kd->d", self._w, self._diff, out=out)
        return out

    def violation(self, q: np.ndarray) -> float:
        d = np.linalg.norm(q[None, :] - self.codebook, axis=1)
        return float(max(0.0, d.min() - self.inscribed_radius))

    def nearest(self, q: np.ndarray) -> int:
        return int(np.argmin(np.linalg.norm(q[None, :] - self.codebook, axis=1)))


@dataclasses.dataclass(frozen=True)
class CalibratedInferenceResult:
    scores: np.ndarray                # (K,) higher is better
    prediction: int                   # argmax of scores
    fixed_point: np.ndarray           # h* (dim,)
    relaxed_state: np.ndarray         # iterated recurrence state (dim,)
    relaxation_residual: float        # ||relaxed_state - fixed_point||
    relaxation_steps: int
    langevin: Optional[LangevinResult]
    expert: Optional[ContinuousCausalReasoningResult]


class CounterfactualDriftDynamics:
    """Inference owner.  Construct via `load()` or `fit_counterfactual_drift_dynamics()`."""

    def __init__(self, *, A, B, W_c, codebook, x_mean, x_basis, c_mean, c_basis,
                 provenance: dict) -> None:
        self.A = _finite(A, "A", dtype=np.float32).astype(np.float64)
        self.B = _finite(B, "B", dtype=np.float32).astype(np.float64)
        self.W_c = _finite(W_c, "W_c", dtype=np.float32).astype(np.float64)
        self.codebook = _finite(codebook, "codebook", dtype=np.float32).astype(np.float64)
        self.x_mean = _finite(x_mean, "x_mean", dtype=np.float32).astype(np.float64)
        self.x_basis = _finite(x_basis, "x_basis", dtype=np.float32).astype(np.float64)
        self.c_mean = _finite(c_mean, "c_mean", dtype=np.float32).astype(np.float64)
        self.c_basis = _finite(c_basis, "c_basis", dtype=np.float32).astype(np.float64)
        if not isinstance(provenance, dict) or not provenance:
            raise ValueError("provenance is required")
        self.provenance = dict(provenance)
        self.certify()

    # -- certificate -------------------------------------------------------
    @property
    def dim(self) -> int:
        return int(self.A.shape[0])

    def certify(self) -> dict:
        d = self.dim
        if self.A.shape != (d, d):
            raise ValueError("A must be square")
        p, pc = self.B.shape[1], self.W_c.shape[1]
        if self.B.shape != (d, p) or self.W_c.shape != (d, pc):
            raise ValueError("B / W_c row count must equal dim")
        if self.codebook.ndim != 2 or self.codebook.shape[1] != d or self.codebook.shape[0] < 2:
            raise ValueError("codebook must be (K>=2, dim)")
        if self.x_basis.shape[1] != p or self.x_mean.shape != (self.x_basis.shape[0],):
            raise ValueError("x projection shape mismatch")
        if self.c_basis.shape[1] != pc or self.c_mean.shape != (self.c_basis.shape[0],):
            raise ValueError("c projection shape mismatch")
        rho = _spectral_radius(self.A)
        if not rho < RHO_MAX:
            raise ValueError(f"A is not contractive: rho={rho:.4f} >= {RHO_MAX}")
        return {"rho": rho, "dim": d, "input_dim": p, "counterfactual_dim": pc,
                "classes": int(self.codebook.shape[0]),
                "working_set_bytes": self.working_set_bytes()}

    def working_set_bytes(self) -> int:
        """Bytes touched by one `step` in the compact space (A, B, W_c, h, x, c, scratch)."""
        d, p, pc = self.dim, self.B.shape[1], self.W_c.shape[1]
        return int(4 * (d * d + d * p + d * pc + 3 * d + p + pc))

    # -- projections (label-free) -----------------------------------------
    def project_x(self, x_raw: np.ndarray) -> np.ndarray:
        return (_finite(x_raw, "x_raw") - self.x_mean) @ self.x_basis

    def project_c(self, c_raw: np.ndarray) -> np.ndarray:
        return (_finite(c_raw, "c_raw") - self.c_mean) @ self.c_basis

    # -- dynamics ------------------------------------------------------------
    def step(self, h: np.ndarray, x: np.ndarray, c: Optional[np.ndarray]) -> np.ndarray:
        drive = self.B @ x
        if c is not None:
            drive = drive + self.W_c @ c
        return self.A @ h + drive

    def fixed_point(self, x: np.ndarray, c: Optional[np.ndarray]) -> np.ndarray:
        drive = self.B @ x
        if c is not None:
            drive = drive + self.W_c @ c
        return np.linalg.solve(np.eye(self.dim) - self.A, drive)

    def infer(self, x_raw: np.ndarray, c_raw: Optional[np.ndarray], *,
              use_counterfactual: bool = True, relax_steps: int = 64,
              langevin: bool = True, langevin_seed: int = 0, langevin_steps: int = 400,
              expert: bool = True, expert_seed: int = 0) -> CalibratedInferenceResult:
        """Label-free inference.  Every stage is switchable for ablation."""
        x = self.project_x(x_raw)
        c = None
        if use_counterfactual:
            if c_raw is None:
                raise ValueError("counterfactual feature required when use_counterfactual=True")
            c = self.project_c(c_raw)
        h_star = self.fixed_point(x, c)
        h = np.zeros(self.dim)
        for _ in range(relax_steps):
            h = self.step(h, x, c)
        relax_residual = float(np.linalg.norm(h - h_star))

        q = h
        lg: Optional[LangevinResult] = None
        if langevin:
            pot = CandidateAttractorPotential(self.codebook)
            eng = AnnealedLangevinDynamicsEngine(
                self.dim, pot, T0=0.05, anneal_gamma=0.95, eta0=0.1,
                max_steps=langevin_steps, residual_tol=1e-3, violation_tol=1e-3,
                dtype=np.float64, seed=langevin_seed)
            lg = eng.run(q)
            q = lg.q

        ex: Optional[ContinuousCausalReasoningResult] = None
        if expert:
            ex = continuous_causal_reasoning_expert(q, self.codebook, seed=expert_seed)
            scores = np.asarray(ex.scores, dtype=np.float64)
        else:
            scores = -np.linalg.norm(q[None, :] - self.codebook, axis=1)
        return CalibratedInferenceResult(
            scores=scores, prediction=int(np.argmax(scores)), fixed_point=h_star,
            relaxed_state=h, relaxation_residual=relax_residual, relaxation_steps=relax_steps,
            langevin=lg, expert=ex)

    # -- persistence ----------------------------------------------------------
    def save(self, path) -> None:
        self.certify()
        meta = {"version": ARTIFACT_VERSION, "provenance": self.provenance,
                "shapes": {k: list(getattr(self, k).shape) for k in ARRAY_KEYS}}
        with open(path, "wb") as stream:
            np.savez(stream, metadata=json.dumps(meta, sort_keys=True),
                     **{k: getattr(self, k).astype(np.float32) for k in ARRAY_KEYS})

    @classmethod
    def load(cls, path, *, encoder_id: str) -> "CounterfactualDriftDynamics":
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"metadata", *ARRAY_KEYS}:
                raise ValueError(f"invalid artifact schema: {sorted(data.files)}")
            meta = json.loads(str(data["metadata"]))
            if meta.get("version") != ARTIFACT_VERSION:
                raise ValueError("artifact version mismatch")
            prov = meta.get("provenance") or {}
            if prov.get("encoder_id") != encoder_id:
                raise ValueError("frozen encoder mismatch")
            if prov.get("split") not in ALLOWED_SPLITS:
                raise CalibrationSplitError("artifact was not fitted on train/calibration data")
            arrays = {}
            for k in ARRAY_KEYS:
                v = data[k]
                if v.dtype != np.float32 or list(v.shape) != meta["shapes"][k]:
                    raise ValueError(f"{k}: invalid dtype/shape")
                arrays[k] = v
        return cls(provenance=prov, **arrays)


def fit_counterfactual_drift_dynamics(
    x_raw: np.ndarray, c_raw: np.ndarray, labels: Sequence[int], *,
    sample_ids: Sequence[str], source: str, split: str, encoder_id: str,
    n_classes: int, dim: int = 16, n_components: int = 16, cf_components: int = 8,
    ridge_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0), a_decay: float = 0.5,
    use_counterfactual: bool = True,
) -> Tuple[CounterfactualDriftDynamics, dict]:
    """Closed-form calibration on the declared calibration split ONLY.

    Ridge strength is chosen by leave-one-out on the calibration rows (labels of
    the calibration split only).  `use_counterfactual=False` fits W_c = 0 so the
    ablation artifact has the same schema and the same PCA.
    """
    if split not in ALLOWED_SPLITS:
        raise CalibrationSplitError("only train/calibration splits may be fitted")
    if not source or not encoder_id:
        raise ValueError("source and encoder_id are required")
    x = _finite(x_raw, "x_raw")
    c = _finite(c_raw, "c_raw")
    y = np.asarray(labels)
    ids = np.asarray(sample_ids, dtype=str)
    n = x.shape[0]
    if x.ndim != 2 or c.ndim != 2 or c.shape[0] != n or y.shape != (n,) or ids.shape != (n,):
        raise ValueError("shape mismatch between features, counterfactuals, labels, ids")
    if y.dtype.kind not in "iu" or y.min() < 0 or y.max() >= n_classes:
        raise ValueError("labels must be integers in [0, n_classes)")
    if len(set(ids.tolist())) != n:
        raise ValueError("sample ids must be unique")
    if not (0.0 <= a_decay < RHO_MAX):
        raise ValueError("a_decay must be in [0, RHO_MAX)")
    if n_classes > dim:
        raise ValueError("dim must be >= n_classes")

    x_mean, x_basis = _pca_fit(x, min(n_components, n - 1))
    c_mean, c_basis = _pca_fit(c, min(cf_components, n - 1))
    zx = (x - x_mean) @ x_basis
    zc = (c - c_mean) @ c_basis
    z = np.concatenate([zx, zc], axis=1) if use_counterfactual else zx
    codebook = simplex_codebook(n_classes, dim)
    t = codebook[y]

    # Leave-one-out selection of the ridge strength on calibration rows only.
    best_lam, best_acc = None, -1.0
    loo_table = {}
    for lam in ridge_grid:
        hits = 0
        for i in range(n):
            keep = np.arange(n) != i
            m = MasterClosedFormSolver().fit(z[keep], t[keep], lambda_reg=lam, fit_intercept=False).T
            pred = int(np.argmin(np.linalg.norm(codebook - (m @ z[i])[None, :], axis=1)))
            hits += int(pred == y[i])
        acc = hits / n
        loo_table[str(lam)] = acc
        if acc > best_acc:
            best_lam, best_acc = float(lam), acc
    # (dim, p_total): h* = M z; Cholesky solve that raises on ill-conditioned systems.
    m = MasterClosedFormSolver().fit(z, t, lambda_reg=best_lam, fit_intercept=False).T
    a = a_decay * np.eye(dim)
    i_minus_a = np.eye(dim) - a
    px = zx.shape[1]
    b = i_minus_a @ m[:, :px]
    if use_counterfactual:
        w_c = i_minus_a @ m[:, px:]
    else:
        w_c = np.zeros((dim, zc.shape[1]))

    digest = hashlib.sha256()
    for arr in (x, c, y, ids):
        digest.update(np.ascontiguousarray(arr).tobytes())
    provenance = {
        "source": source, "split": split, "encoder_id": encoder_id,
        "training_sha256": digest.hexdigest(), "samples": int(n),
        "sample_ids": ids.tolist(), "classes": int(n_classes),
        "ridge_lambda": best_lam, "ridge_loo_accuracy": best_acc,
        "ridge_loo_table": loo_table, "a_decay": a_decay,
        "a_note": "A is set contractive (a_decay * I), not learned: the fixed point "
                  "h* = (I-A)^-1 (B x + W_c c) is invariant to A given M, so with this "
                  "few calibration rows A carries no learnable signal.",
        "use_counterfactual": bool(use_counterfactual),
        "objective": "ridge_fixed_point_to_simplex_codebook_v1",
    }
    dyn = CounterfactualDriftDynamics(A=a, B=b, W_c=w_c, codebook=codebook, x_mean=x_mean,
                                      x_basis=x_basis, c_mean=c_mean, c_basis=c_basis,
                                      provenance=provenance)
    return dyn, {"ridge_lambda": best_lam, "ridge_loo_accuracy": best_acc, "loo_table": loo_table}
