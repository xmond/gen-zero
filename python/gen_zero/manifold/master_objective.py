"""Master closed-form objective: weighted ridge + prior shrinkage + graph smoothing.

Objective (F = Z W + 1 b^T are the fitted outputs, b is the unpenalised intercept):

    min_W  || M^{1/2} (Y - F) ||_F^2  +  lambda || W - W0 ||_F^2  +  eta * tr(F^T L F)

L is a combinatorial graph Laplacian (L 1 = 0), so the intercept drops out of the
graph term and tr(F^T L F) = tr(W^T Z^T L Z W), which is the textbook form.
Normal equations on the augmented design Za = [Z, 1]:

    (Za^T M Za + Lambda + eta Za^T L Za) Wa = Za^T M Y + Lambda W0a

with Lambda = diag(lambda, ..., lambda, 0). The system matrix is solved with a
Cholesky factorisation; it is never inverted. Every numerical failure raises
``MasterObjectiveError``; nothing falls back to a weaker solve.

Adaptive spectral scaling (``adaptive_spectral_scaling=True``, the default):
``Add = Z^T M Z`` grows with N_eff (diag(Add) ~ N_eff for standardised Z), so a
fixed lambda_reg dilutes as N_eff grows. We hold ``||Lambda||_F / ||Add||_F``
constant instead: ``lambda_eff = lambda_reg * ||Add||_F / sqrt(d)`` -- no
``/ N_eff``, since that division is what a naive reading of the "obvious"
formula adds, and it exactly cancels the growth being compensated for (see
``tests/test_spectral_scaling.py`` for the derivation and the invariance it
buys). ``eta`` is left unscaled: ``eta * Za^T L Za`` is already a raw sum over
O(N_eff) edges, so it already tracks ``Add``'s growth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from .graph_laplacian import MultiModelGraphLaplacian

DEFAULT_MAX_CONDITION = 1e12

MatrixLike = Union[np.ndarray, sp.spmatrix]


class MasterObjectiveError(ArithmeticError):
    """Raised when the closed-form system is non-finite, singular or ill-conditioned."""


@dataclass(frozen=True)
class SolveDiagnostics:
    n_samples: int
    n_features: int
    n_outputs: int
    condition_number: float
    min_eigenvalue: float
    objective: float
    data_term: float
    prior_term: float
    graph_term: float
    adaptive_spectral_scaling: bool
    spectral_scale: float
    lambda_reg: float
    lambda_eff: float
    eta_eff: float


def _finite_array(value, name: str, ndim: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise MasterObjectiveError(f"{name} contains NaN or Inf")
    return arr


def _weighted_gram(Za: np.ndarray, Y: np.ndarray, M) -> tuple[np.ndarray, np.ndarray]:
    """Return (Za^T M Za, Za^T M Y) without materialising diag(M)."""
    if M is None:
        return Za.T @ Za, Za.T @ Y
    if isinstance(M, np.ndarray) and M.ndim == 1:
        MZ = Za * M[:, None]
        return MZ.T @ Za, MZ.T @ Y
    MZ = M @ Za
    return Za.T @ MZ, MZ.T @ Y


def _validate_weights(M, n: int):
    if M is None:
        return None
    if sp.issparse(M):
        M = sp.coo_matrix(M, dtype=np.float64)
        if M.shape != (n, n):
            raise ValueError(f"M must be ({n}, {n}), got {M.shape}")
        if np.any((M.row != M.col) & (M.data != 0)):
            raise ValueError("sparse M must be diagonal; pass a dense array for a full weight matrix")
        M = np.asarray(M.tocsr().diagonal())
    else:
        M = _finite_array(M, "M")
    if M.ndim == 1:
        if M.shape != (n,):
            raise ValueError(f"M weights must have shape ({n},), got {M.shape}")
        if np.any(M < 0):
            raise ValueError("M weights must be non-negative")
        if not np.any(M > 0):
            raise ValueError("M must give positive weight to at least one sample")
        return M
    if M.shape != (n, n):
        raise ValueError(f"M must be ({n}, {n}), got {M.shape}")
    if not np.allclose(M, M.T, rtol=0.0, atol=1e-10 * max(1.0, np.abs(M).max())):
        raise ValueError("M must be symmetric")
    if np.linalg.eigvalsh(M)[0] < -1e-10 * max(1.0, np.abs(M).max()):
        raise ValueError("M must be positive semi-definite")
    return M


class MasterClosedFormSolver:
    """Closed-form solver for the master objective; see module docstring."""

    def __init__(self, max_condition: float = DEFAULT_MAX_CONDITION):
        if not (np.isfinite(max_condition) and max_condition > 1.0):
            raise ValueError("max_condition must be finite and > 1")
        self.max_condition = float(max_condition)
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.fit_intercept: bool = True
        self.diagnostics_: Optional[SolveDiagnostics] = None
        self._single_output = False

    def fit(
        self,
        Z,
        Y,
        W0=None,
        M: Optional[MatrixLike] = None,
        L: Optional[MatrixLike] = None,
        lambda_reg: float = 100.0,
        eta: float = 0.0,
        fit_intercept: bool = True,
        adaptive_spectral_scaling: bool = True,
    ) -> np.ndarray:
        """Fit and return W* with shape (d, k) (intercept is stored in ``intercept_``)."""
        Z = _finite_array(Z, "Z", ndim=2)
        n, d = Z.shape
        Y = _finite_array(Y, "Y")
        single_output = Y.ndim == 1
        if single_output:
            Y = Y[:, None]
        if Y.ndim != 2 or Y.shape[0] != n:
            raise ValueError(f"Y must have {n} rows, got shape {Y.shape}")
        k = Y.shape[1]
        if not (np.isfinite(lambda_reg) and lambda_reg >= 0.0):
            raise ValueError("lambda_reg must be finite and >= 0")
        if not (np.isfinite(eta) and eta >= 0.0):
            raise ValueError("eta must be finite and >= 0")
        if eta > 0.0 and L is None:
            raise ValueError("eta > 0 requires a graph Laplacian L")

        if W0 is None:
            W0 = np.zeros((d, k))
        W0 = _finite_array(W0, "W0")
        if W0.ndim == 1 and single_output:
            W0 = W0[:, None]
        if W0.shape != (d, k):
            raise ValueError(f"W0 must have shape ({d}, {k}), got {W0.shape}")
        M = _validate_weights(M, n)

        Za = np.hstack([Z, np.ones((n, 1))]) if fit_intercept else Z
        p = Za.shape[1]
        A, B = _weighted_gram(Za, Y, M)

        Add = A[:d, :d]
        spectral_norm = float(np.linalg.norm(Add))
        if M is None:
            n_eff = float(n)
        elif M.ndim == 1:
            n_eff = float(M.sum())
        else:
            n_eff = float(np.trace(M))

        if adaptive_spectral_scaling:
            if n == 0:
                raise MasterObjectiveError("adaptive spectral scaling requires at least one sample (N=0)")
            if d == 0:
                raise MasterObjectiveError("adaptive spectral scaling requires at least one feature (d=0)")
            if not (np.isfinite(n_eff) and n_eff > 0.0):
                raise MasterObjectiveError(
                    f"adaptive spectral scaling: effective sample weight N_eff={n_eff!r} is non-positive"
                )
            if not np.isfinite(spectral_norm):
                raise MasterObjectiveError("adaptive spectral scaling: ||Z^T M Z||_F is non-finite")
            if spectral_norm == 0.0:
                raise MasterObjectiveError(
                    "adaptive spectral scaling: ||Z^T M Z||_F is zero (Z=0 or degenerate features)"
                )
            lambda_eff = float(lambda_reg) * spectral_norm / np.sqrt(d)
        else:
            lambda_eff = float(lambda_reg)

        spectral_scale = (
            spectral_norm / (n_eff * np.sqrt(d))
            if (n_eff > 0.0 and d > 0 and np.isfinite(spectral_norm))
            else float("nan")
        )

        reg = np.full(p, lambda_eff)
        if fit_intercept:
            reg[-1] = 0.0
        W0a = np.vstack([W0, np.zeros((1, k))]) if fit_intercept else W0
        A = A + np.diag(reg)
        B = B + reg[:, None] * W0a
        G = None
        if L is not None:
            MultiModelGraphLaplacian.verify_laplacian(L, n_nodes=n)
            if eta > 0.0:
                G = MultiModelGraphLaplacian.quadratic_operator(Za, L)
                A = A + eta * G

        A = 0.5 * (A + A.T)
        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
            raise MasterObjectiveError("normal equations overflowed to NaN/Inf")
        eig = np.linalg.eigvalsh(A)
        if eig[0] <= 0.0:
            raise MasterObjectiveError(
                f"system matrix is not positive definite (min eigenvalue {eig[0]:.3e}); "
                "raise lambda_reg or add samples"
            )
        cond = float(eig[-1] / eig[0])
        if cond > self.max_condition:
            raise MasterObjectiveError(
                f"system matrix is ill-conditioned (cond {cond:.3e} > {self.max_condition:.1e}); "
                "raise lambda_reg or rescale Z"
            )
        try:
            Wa = scipy.linalg.solve(A, B, assume_a="pos", check_finite=True)
        except (np.linalg.LinAlgError, ValueError) as exc:
            raise MasterObjectiveError(f"Cholesky solve failed: {exc}") from exc
        if not np.all(np.isfinite(Wa)):
            raise MasterObjectiveError("solution contains NaN or Inf")

        self.fit_intercept = fit_intercept
        self.coef_ = Wa[:d]
        self.intercept_ = Wa[d] if fit_intercept else np.zeros(k)
        self._single_output = single_output

        R = Y - Za @ Wa
        data = float(np.sum(R * R) if M is None else (
            np.sum(M[:, None] * R * R) if M.ndim == 1 else np.sum(R * (M @ R))))
        prior = float(lambda_eff * np.sum((self.coef_ - W0) ** 2))
        graph = float(eta * np.sum(Wa * (G @ Wa))) if G is not None else 0.0
        self.diagnostics_ = SolveDiagnostics(
            n_samples=n, n_features=d, n_outputs=k, condition_number=cond,
            min_eigenvalue=float(eig[0]), objective=data + prior + graph,
            data_term=data, prior_term=prior, graph_term=graph,
            adaptive_spectral_scaling=bool(adaptive_spectral_scaling),
            spectral_scale=spectral_scale, lambda_reg=float(lambda_reg),
            lambda_eff=lambda_eff, eta_eff=float(eta),
        )
        return self.coef_

    def _require_fitted(self) -> None:
        if self.coef_ is None:
            raise RuntimeError("MasterClosedFormSolver is not fitted; call fit() first")

    def predict(self, Z_test) -> np.ndarray:
        """Return logits Z_test W* + b (1-D when fit was given 1-D Y)."""
        self._require_fitted()
        Z_test = _finite_array(Z_test, "Z_test", ndim=2)
        if Z_test.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"Z_test has {Z_test.shape[1]} features, solver was fitted on {self.coef_.shape[0]}"
            )
        out = Z_test @ self.coef_ + self.intercept_
        return out[:, 0] if self._single_output else out

    def predict_proba(self, Z_test, temperature: float = 1.0) -> np.ndarray:
        """Row-wise softmax of logits / temperature."""
        if not (np.isfinite(temperature) and temperature > 0.0):
            raise ValueError("temperature must be finite and > 0")
        logits = self.predict(Z_test)
        if logits.ndim == 1:
            raise ValueError("predict_proba needs a multi-output fit (one column per class)")
        s = logits / temperature
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(axis=1, keepdims=True)

    def save(self, path, **extra_arrays) -> None:
        """Write the fitted head to .npz; ``extra_arrays`` (e.g. preprocessing) ride along."""
        self._require_fitted()
        reserved = {"coef", "intercept", "fit_intercept", "single_output"} & set(extra_arrays)
        if reserved:
            raise ValueError(f"extra array names collide with solver fields: {sorted(reserved)}")
        np.savez(path, coef=self.coef_, intercept=self.intercept_,
                 fit_intercept=np.array(self.fit_intercept),
                 single_output=np.array(self._single_output), **extra_arrays)

    @classmethod
    def load(cls, path) -> "MasterClosedFormSolver":
        with np.load(path, allow_pickle=False) as data:
            for key in ("coef", "intercept", "fit_intercept", "single_output"):
                if key not in data:
                    raise KeyError(f"{path}: missing array {key!r}")
            solver = cls()
            solver.coef_ = _finite_array(data["coef"], "coef", ndim=2)
            solver.intercept_ = _finite_array(data["intercept"], "intercept", ndim=1)
            solver.fit_intercept = bool(data["fit_intercept"])
            solver._single_output = bool(data["single_output"])
        if solver.intercept_.shape[0] != solver.coef_.shape[1]:
            raise ValueError(f"{path}: intercept/coef shape mismatch")
        return solver


def ordinal_soft_targets(y, k: int, tau: float = 1.0) -> np.ndarray:
    """Gaussian ordinal-distance soft targets: row i peaks at y_i, decays with rank distance.

    Y[i, j] = softmax_j( -(j - y_i)^2 / (2 tau^2) ), j = 0..k-1.
    """
    if not isinstance(k, (int, np.integer)) or k < 2:
        raise ValueError(f"k must be an int >= 2, got {k!r}")
    if not (np.isfinite(tau) and tau > 0.0):
        raise ValueError(f"tau must be finite and > 0, got {tau!r}")
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y_arr.shape}")
    if not np.issubdtype(y_arr.dtype, np.integer):
        raise ValueError(f"y must be an integer array, got dtype {y_arr.dtype}")
    if np.any((y_arr < 0) | (y_arr >= k)):
        raise ValueError(f"y contains values outside [0, {k})")
    ranks = np.arange(k, dtype=np.float64)
    logits = -(ranks[None, :] - y_arr[:, None].astype(np.float64)) ** 2 / (2.0 * tau * tau)
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def ordinal_expected_decode(proba: np.ndarray) -> np.ndarray:
    """Expected-rank decode: round(sum_j j * p_ij), clipped to [0, k-1]. Returns int array shape (n,)."""
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2:
        raise ValueError(f"proba must be 2-D, got shape {proba.shape}")
    if not np.all(np.isfinite(proba)):
        raise ValueError("proba contains NaN or Inf")
    if np.any(proba < 0):
        raise ValueError("proba must be non-negative")
    if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("proba rows must sum to 1")
    expected = proba @ np.arange(proba.shape[1])
    return np.clip(np.round(expected), 0, proba.shape[1] - 1).astype(int)
