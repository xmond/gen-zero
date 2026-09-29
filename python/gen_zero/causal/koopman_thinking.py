"""Gen-Zero Causal: Continuous-Time Koopman Latent Thinking.

Silent multi-step lookahead in latent space, with no token emission.

The latent state z in R^D is assumed to evolve under a linear flow in a
Koopman-lifted coordinate system:

    dz/dt = A z            (A is the infinitesimal generator)
    z(t)  = exp(t * A) z(0)

Because exp((s + t) A) = exp(s A) exp(t A), the model forms a one-parameter
semigroup. Lookahead over any horizon, on any time grid (uniform or not), is
therefore exact for the fitted generator. It never re-enters the token loop.

Pieces:
1. `expm`: matrix exponential (Pade approximant with scaling and squaring).
2. `logm_real`: principal real matrix logarithm (eigendecomposition).
3. `KoopmanGenerator`: holds A, evaluates exp(t * A), and can bound growth.
4. `fit_generator_from_snapshots` / `fit_generator_from_derivatives`:
   estimate A from latent data (EDMD-style least squares).
5. `KoopmanThinker`: multi-step latent lookahead returning latent trajectories.

Only NumPy is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]

# Pade order 6 with ||M / 2^s||_1 <= 0.5 gives truncation error near 1e-17.
_PADE_ORDER = 6
_PADE_NORM_LIMIT = 0.5


def _as_square(matrix: ArrayLike, name: str = "matrix") -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2-D array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def expm(matrix: ArrayLike) -> np.ndarray:
    """Matrix exponential exp(M) by Pade approximation, scaling and squaring."""
    m = _as_square(matrix)
    n = m.shape[0]
    if n == 0:
        return m.copy()

    norm = float(np.linalg.norm(m, 1))
    squarings = 0
    if norm > _PADE_NORM_LIMIT:
        squarings = max(0, int(np.ceil(np.log2(norm / _PADE_NORM_LIMIT))))
    scaled = m / (2.0**squarings)

    # Pade [q/q] coefficients: c_k = c_{k-1} * (q - k + 1) / (k * (2q - k + 1)).
    q = _PADE_ORDER
    coeff = 1.0
    ident = np.eye(n)
    power = ident
    numer = ident.copy()
    denom = ident.copy()
    for k in range(1, q + 1):
        coeff *= (q - k + 1) / (k * (2 * q - k + 1))
        power = power @ scaled
        numer = numer + coeff * power
        denom = denom + coeff * ((-1) ** k) * power

    result = np.linalg.solve(denom, numer)
    for _ in range(squarings):
        result = result @ result
    return result


def logm_real(matrix: ArrayLike, tol: float = 1e-8) -> np.ndarray:
    """Principal real logarithm of a real matrix, via eigendecomposition.

    Raises ValueError when no real logarithm exists in this form (an eigenvalue
    is zero or on the negative real axis) or when the matrix is too close to
    defective for an eigendecomposition to be trusted.
    """
    k = _as_square(matrix, "matrix")
    eigvals, eigvecs = np.linalg.eig(k)
    if np.any(np.abs(eigvals) < tol):
        raise ValueError("matrix is singular: no logarithm exists")
    on_negative_axis = (np.abs(eigvals.imag) < tol) & (eigvals.real < 0.0)
    if np.any(on_negative_axis):
        raise ValueError("eigenvalue on the negative real axis: no real logarithm")

    log_k = eigvecs @ np.diag(np.log(eigvals)) @ np.linalg.inv(eigvecs)
    scale = 1.0 + float(np.linalg.norm(log_k))
    if float(np.linalg.norm(log_k.imag)) > 1e-6 * scale:
        raise ValueError("logarithm is not real: eigendecomposition is unreliable")
    log_real = log_k.real
    recon_err = float(np.linalg.norm(expm(log_real) - k))
    if recon_err > 1e-6 * (1.0 + float(np.linalg.norm(k))):
        raise ValueError(f"logarithm failed reconstruction check (error {recon_err:.3e})")
    return log_real


@dataclass(frozen=True)
class KoopmanGenerator:
    """Infinitesimal generator A of a linear latent flow dz/dt = A z."""

    matrix: np.ndarray

    def __post_init__(self) -> None:
        arr = _as_square(self.matrix, "generator")
        arr.setflags(write=False)
        object.__setattr__(self, "matrix", arr)

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[0])

    def flow(self, t: float) -> np.ndarray:
        """Return exp(t * A). Negative t runs the flow backward in time."""
        return expm(float(t) * self.matrix)

    def spectral_abscissa(self) -> float:
        """Largest real part among eigenvalues. Positive means the flow can blow up."""
        if self.dim == 0:
            return 0.0
        return float(np.max(np.linalg.eigvals(self.matrix).real))

    def stabilized(self, max_growth: float = 0.0) -> "KoopmanGenerator":
        """Return a generator whose eigenvalue real parts are clipped to <= max_growth.

        Conjugate pairs stay conjugate, so the result is still real. Oscillation
        frequencies are kept. Use it to keep long lookahead bounded.
        """
        eigvals, eigvecs = np.linalg.eig(self.matrix)
        clipped = np.minimum(eigvals.real, max_growth) + 1j * eigvals.imag
        rebuilt = eigvecs @ np.diag(clipped) @ np.linalg.inv(eigvecs)
        return KoopmanGenerator(rebuilt.real)


def fit_generator_from_snapshots(
    z_now: np.ndarray,
    z_next: np.ndarray,
    dt: float,
    ridge: float = 0.0,
) -> KoopmanGenerator:
    """Estimate A from snapshot pairs z_next ~ exp(dt * A) z_now.

    Rows are samples: both arrays have shape (N, D). Solves the one-step operator
    K by least squares, then takes A = log(K) / dt.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    x = np.asarray(z_now, dtype=np.float64)
    y = np.asarray(z_next, dtype=np.float64)
    if x.ndim != 2 or x.shape != y.shape:
        raise ValueError("z_now and z_next must be 2-D arrays of the same shape")
    k = _solve_operator(x, y, ridge)
    return KoopmanGenerator(logm_real(k) / dt)


def fit_generator_from_derivatives(
    z: np.ndarray,
    dz_dt: np.ndarray,
    ridge: float = 0.0,
) -> KoopmanGenerator:
    """Estimate A directly from states and time derivatives: dz_dt ~ A z."""
    x = np.asarray(z, dtype=np.float64)
    y = np.asarray(dz_dt, dtype=np.float64)
    if x.ndim != 2 or x.shape != y.shape:
        raise ValueError("z and dz_dt must be 2-D arrays of the same shape")
    return KoopmanGenerator(_solve_operator(x, y, ridge))


def _solve_operator(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """Solve y_i ~ M x_i for M (rows are samples), with optional ridge penalty."""
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("data must contain only finite values")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    d = x.shape[1]
    gram = x.T @ x + ridge * np.eye(d)
    # M^T = (X^T X + r I)^+ X^T Y. pinv keeps rank-deficient data usable.
    return (np.linalg.pinv(gram) @ x.T @ y).T


@dataclass(frozen=True)
class LookaheadResult:
    """Latent-only rollout. `states[i]` is the latent at `times[i]`."""

    times: np.ndarray
    states: np.ndarray

    @property
    def final_state(self) -> np.ndarray:
        return self.states[-1]


class KoopmanThinker:
    """Multi-step latent lookahead under a continuous-time Koopman generator.

    Every method returns latent vectors. Nothing here decodes to, or emits, tokens.
    """

    def __init__(self, generator: Union[KoopmanGenerator, ArrayLike]) -> None:
        self.generator = (
            generator if isinstance(generator, KoopmanGenerator) else KoopmanGenerator(np.asarray(generator))
        )

    @property
    def dim(self) -> int:
        return self.generator.dim

    def _check_state(self, z: ArrayLike) -> np.ndarray:
        vec = np.asarray(z, dtype=np.float64)
        if vec.shape != (self.dim,):
            raise ValueError(f"state must have shape ({self.dim},), got {vec.shape}")
        if not np.all(np.isfinite(vec)):
            raise ValueError("state must contain only finite values")
        return vec

    def propagate(self, z: ArrayLike, t: float) -> np.ndarray:
        """Jump straight to z(t) = exp(t * A) z in one shot."""
        return self.generator.flow(t) @ self._check_state(z)

    def lookahead(self, z: ArrayLike, horizon: int, dt: float) -> LookaheadResult:
        """Roll `horizon` uniform steps of size dt. Returns horizon + 1 states, z included.

        Builds exp(dt * A) once and applies it repeatedly, using the semigroup
        property. Cost per step is one matrix-vector product.
        """
        if horizon < 0:
            raise ValueError("horizon must be non-negative")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        step = self.generator.flow(dt)
        states = np.empty((horizon + 1, self.dim))
        states[0] = self._check_state(z)
        for i in range(horizon):
            states[i + 1] = step @ states[i]
        return LookaheadResult(times=dt * np.arange(horizon + 1, dtype=np.float64), states=states)

    def lookahead_at(self, z: ArrayLike, times: ArrayLike) -> LookaheadResult:
        """Evaluate z(t) at arbitrary, possibly irregular, non-negative times."""
        grid = np.asarray(times, dtype=np.float64)
        if grid.ndim != 1:
            raise ValueError("times must be a 1-D array")
        if np.any(grid < 0.0) or not np.all(np.isfinite(grid)):
            raise ValueError("times must be finite and non-negative")
        z0 = self._check_state(z)
        states = np.stack([self.generator.flow(t) @ z0 for t in grid]) if grid.size else np.empty((0, self.dim))
        return LookaheadResult(times=grid, states=states)

    def think(
        self,
        z: ArrayLike,
        horizon: int,
        dt: float,
        max_growth: Optional[float] = 0.0,
    ) -> LookaheadResult:
        """Bounded lookahead. Clips unstable modes first unless max_growth is None."""
        if max_growth is None or self.generator.spectral_abscissa() <= max_growth:
            return self.lookahead(z, horizon, dt)
        return KoopmanThinker(self.generator.stabilized(max_growth)).lookahead(z, horizon, dt)
