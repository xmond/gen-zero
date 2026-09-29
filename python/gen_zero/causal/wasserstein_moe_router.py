"""Heterogeneous Causal MoE 2.0: continuous optimal-transport routing gateway
plus a four orthogonal-tangent-space expert pool (doc `05-heterogeneous-causal-
moe-and-future-evolution.md`, Sprint 1 + Sprint 2).

Plain numpy, no GPU, no text/label inspection anywhere (matches the "no task
names, no regex" contract already followed by `universal_manifold_extractor.py`
and `compiled_manifold_runtime.py`).

  * `WassersteinOptimalTransportRouter` -- replaces the 1.0 hard-threshold
    cascade (Expert 0 -> Expert 1 -> Expert 2 fall-through) with a Bures-
    Wasserstein geodesic-distance softmin over each expert's attractor
    manifold, modeled as a Gaussian (mean, covariance) in the shared phase
    space. The output is a convex combination lambda_i in [0, 1], sum = 1,
    that is C^1 (in fact C-infinity) in the input mean whenever the input's
    own covariance is held fixed, because then every distance term reduces to
    a smooth quadratic in the mean plus a mean-independent trace constant --
    there is no threshold anywhere in the computation.
  * `MultiTangentManifoldPool` -- replaces the single entangled 64-dim phase
    space with K=4 mutually orthogonal 64-dim tangent charts (syntax, logic,
    numeric, intent) carved out of one shared ambient space via a single QR
    orthogonal matrix: since the four charts are disjoint column-blocks of
    one orthogonal matrix Q, U_i^T U_j is the exact zero matrix for i != j
    and U_k^T U_k = I by construction, not by approximate optimization. Each
    chart owns its own contractive Lyapunov pair (A_k, b_k) with a certified
    spectral radius < 1.0, using radial spectral projection for its discrete update.
  * `HeterogeneousCausalMoEPipeline` -- wires both into one
    route -> fuse -> project -> evolve -> reconstruct call.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ExpertManifold",
    "WassersteinOptimalTransportRouter",
    "TangentSpace",
    "MultiTangentManifoldPool",
    "MoEStepResult",
    "HeterogeneousCausalMoEPipeline",
]


def _finite_vector(value, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: must be a nonempty finite 1D array")
    return a


def _finite_square(value, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: must be a nonempty finite square 2D array")
    return a


def _symmetric_psd_sqrt(cov: np.ndarray, *, tol: float = 1e-9) -> np.ndarray:
    """Matrix square root of a symmetric PSD matrix via eigendecomposition.

    ``eigh`` is exact for symmetric matrices (unlike a general Schur/Denman-
    Beavers iteration), and negative eigenvalues below ``tol`` are clipped to
    zero: they can only arise from float rounding on a matrix that is PSD in
    exact arithmetic (a real covariance), never from a genuinely indefinite
    input, because callers validate PSD-ness at construction time.
    """
    sym = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    if np.any(eigvals < -tol):
        raise ValueError(f"matrix is not positive semi-definite (min eigenvalue {eigvals.min()})")
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvecs @ (np.sqrt(eigvals)[:, None] * eigvecs.T)


def bures_wasserstein_sq_distance(
    mean1: np.ndarray, cov1: np.ndarray, mean2: np.ndarray, cov2: np.ndarray
) -> float:
    """Squared Bures-Wasserstein (2-Wasserstein, W2) distance between two
    Gaussians ``N(mean1, cov1)`` and ``N(mean2, cov2)``:

        W2^2 = ||mean1 - mean2||^2 + Tr(cov1 + cov2 - 2 (cov1^{1/2} cov2 cov1^{1/2})^{1/2})

    This is the closed-form geodesic distance on the Bures-Wasserstein
    manifold of covariance-aware Gaussian measures (Bhatia, Jain & Lim 2019).
    When ``cov1 == cov2 == 0`` it degenerates to the plain Euclidean squared
    distance, so a zero-covariance "point" expert (e.g. a static cache grid)
    is handled without a special case.
    """
    m1 = _finite_vector(mean1, "mean1")
    m2 = _finite_vector(mean2, "mean2")
    if m1.shape != m2.shape:
        raise ValueError("mean1 and mean2 must have the same shape")
    c1 = _finite_square(cov1, "cov1")
    c2 = _finite_square(cov2, "cov2")
    if c1.shape != c2.shape or c1.shape[0] != m1.shape[0]:
        raise ValueError("cov1/cov2 must be square with dim matching the means")

    mean_term = float(np.sum((m1 - m2) ** 2))
    sqrt_c1 = _symmetric_psd_sqrt(c1)
    inner = sqrt_c1 @ c2 @ sqrt_c1
    sqrt_inner = _symmetric_psd_sqrt(inner)
    cov_term = float(np.trace(c1) + np.trace(c2) - 2.0 * np.trace(sqrt_inner))
    # Theoretically >= 0 (W2^2 is a squared distance); clip a hair of float
    # rounding rather than ever returning a spuriously negative distance.
    cov_term = max(cov_term, 0.0)
    return mean_term + cov_term


@dataclasses.dataclass(frozen=True)
class ExpertManifold:
    """One heterogeneous expert's attractor basin, modeled as a Gaussian in
    the shared phase space: ``mean`` is the attractor centroid, ``covariance``
    is the local curvature of the basin (zero covariance models a hard-cached
    point attractor like Expert 0's static semantic grid)."""

    name: str
    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self):
        mean = _finite_vector(self.mean, f"{self.name}.mean")
        cov = _finite_square(self.covariance, f"{self.name}.covariance")
        if cov.shape[0] != mean.shape[0]:
            raise ValueError(f"{self.name}: covariance dim must match mean dim")
        eigvals = np.linalg.eigvalsh(0.5 * (cov + cov.T))
        if np.any(eigvals < -1e-9):
            raise ValueError(f"{self.name}: covariance must be positive semi-definite")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", cov)

    @property
    def dim(self) -> int:
        return self.mean.shape[0]


class WassersteinOptimalTransportRouter:
    """Continuous softmin router over a fixed set of heterogeneous experts.

    Replaces the 1.0 hard-threshold cascade with a Bures-Wasserstein softmin:
    ``lambda_i = softmax(-d_i / temperature)`` where ``d_i`` is the geodesic
    distance from the (possibly covariance-aware) input to expert ``i``'s
    attractor manifold. This is a strict generalization of nearest-attractor
    hard routing (recovered as ``temperature -> 0``) that stays smooth for
    any ``temperature > 0``, so boundary samples interpolate instead of
    flipping discretely.
    """

    def __init__(self, experts: Sequence[ExpertManifold], *, temperature: float = 1.0):
        experts = list(experts)
        if len(experts) < 2:
            raise ValueError("need at least 2 experts to route between")
        names = [e.name for e in experts]
        if len(set(names)) != len(names):
            raise ValueError(f"expert names must be unique, got {names}")
        dims = {e.dim for e in experts}
        if len(dims) != 1:
            raise ValueError(f"all experts must share one phase-space dim, got {dims}")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive finite")
        self.experts: Tuple[ExpertManifold, ...] = tuple(experts)
        self.names: Tuple[str, ...] = tuple(names)
        self.dim = experts[0].dim
        self.temperature = float(temperature)

    def distances(self, x_mean, x_covariance: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Bures-Wasserstein squared distance from the input to every expert."""
        x_mean = _finite_vector(x_mean, "x_mean")
        if x_mean.shape[0] != self.dim:
            raise ValueError(f"x_mean must have shape ({self.dim},)")
        if x_covariance is None:
            x_covariance = np.zeros((self.dim, self.dim), dtype=np.float64)
        else:
            x_covariance = _finite_square(x_covariance, "x_covariance")
            if x_covariance.shape[0] != self.dim:
                raise ValueError(f"x_covariance must have shape ({self.dim}, {self.dim})")
        return {
            e.name: bures_wasserstein_sq_distance(x_mean, x_covariance, e.mean, e.covariance)
            for e in self.experts
        }

    def route(self, x_mean, x_covariance: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Continuous convex-combination weights lambda_i in [0, 1], sum = 1."""
        dist = self.distances(x_mean, x_covariance)
        d = np.array([dist[name] for name in self.names], dtype=np.float64)
        logits = -d / self.temperature
        logits -= logits.max()  # softmax shift invariance, keeps exp() finite
        weights = np.exp(logits)
        weights /= weights.sum()
        return {name: float(w) for name, w in zip(self.names, weights)}

    def fuse(self, weights: Mapping[str, float], vectors: Mapping[str, np.ndarray]) -> np.ndarray:
        """Convex combination ``sum_i lambda_i * vectors[i]`` over the shared
        phase-space representation (e.g. Expert 1's codebook state and
        Expert 2's Zero deep-manifold state), keyed by expert name."""
        if set(weights) != set(self.names):
            raise ValueError(f"weights must cover exactly the router's experts {self.names}")
        if set(vectors) != set(self.names):
            raise ValueError(f"vectors must cover exactly the router's experts {self.names}")
        out = np.zeros(self.dim, dtype=np.float64)
        for name in self.names:
            v = _finite_vector(vectors[name], f"vectors[{name!r}]")
            if v.shape[0] != self.dim:
                raise ValueError(f"vectors[{name!r}] must have shape ({self.dim},)")
            out += weights[name] * v
        return out


def _project_to_contractive(a: np.ndarray, target_rho: float) -> Tuple[np.ndarray, float]:
    """Radial spectral projection: uniformly scale ``a`` so rho(a) < 1.0.

    This applies to the discrete chart update: scaling a matrix by a real
    scalar scales every eigenvalue by that factor. It is not a continuous
    generator stability criterion.
    """
    rho = float(np.max(np.abs(np.linalg.eigvals(a))))
    if rho >= 1.0:
        a = a * (target_rho / rho)
        rho = float(np.max(np.abs(np.linalg.eigvals(a))))
    return a, rho


@dataclasses.dataclass(frozen=True)
class TangentSpace:
    """One orthogonal tangent chart: an orthonormal column-block ``basis``
    (ambient_dim, tangent_dim) of a shared orthogonal matrix, plus its own
    contractive Lyapunov pair ``(A, b)``."""

    name: str
    basis: np.ndarray
    a: np.ndarray
    b: np.ndarray
    rho: float

    @property
    def tangent_dim(self) -> int:
        return self.a.shape[0]

    def project(self, x_ambient: np.ndarray) -> np.ndarray:
        """Local coordinates ``U^T x``. Exact left-inverse of `lift` because
        ``basis`` has orthonormal columns (``U^T U = I``)."""
        x = _finite_vector(x_ambient, "x_ambient")
        if x.shape[0] != self.basis.shape[0]:
            raise ValueError(f"x_ambient must have shape ({self.basis.shape[0]},)")
        return self.basis.T @ x

    def lift(self, h_local: np.ndarray) -> np.ndarray:
        """Embed local coordinates back into the ambient space via ``U h``."""
        h = _finite_vector(h_local, "h_local")
        if h.shape[0] != self.tangent_dim:
            raise ValueError(f"h_local must have shape ({self.tangent_dim},)")
        return self.basis @ h

    def fixed_point(self, x_ambient: np.ndarray) -> np.ndarray:
        """Closed-form fixed point of the local recurrence
        ``h_{t+1} = A h_t + b + U^T x`` as ``t -> infinity``:
        ``h* = (I - A)^-1 (b + U^T x)``, well-posed because ``rho(A) < 1.0``
        certifies ``(I - A)`` is invertible (same closed-form technique as
        `CompiledManifoldRuntime.infer`)."""
        drive = self.b + self.project(x_ambient)
        return np.linalg.solve(np.eye(self.tangent_dim) - self.a, drive)


class MultiTangentManifoldPool:
    """K=4 mutually orthogonal 64-dim tangent charts disentangling the single
    64-dim phase space that used to host syntax, logic, numeric and intent
    tasks all at once (doc section 2.2's "manifold entanglement" bottleneck).

    Orthogonality is exact by construction: all four bases are disjoint
    column-blocks of one ``QR``-orthogonalized matrix, so ``U_i^T U_j`` is
    the zero matrix for ``i != j`` up to float rounding, not an approximate
    regularization target.
    """

    CHART_NAMES: Tuple[str, str, str, str] = ("M_syntax", "M_logic", "M_numeric", "M_intent")

    def __init__(
        self,
        tangent_dim: int = 64,
        *,
        seed: int = 0,
        target_rho: float = 0.9,
        dynamics_scale: float = 0.5,
        bias_scale: float = 0.01,
    ):
        if not isinstance(tangent_dim, (int, np.integer)) or tangent_dim <= 0:
            raise ValueError("tangent_dim must be a positive integer")
        if not np.isfinite(target_rho) or not (0.0 < target_rho < 1.0):
            raise ValueError("target_rho must be in (0, 1)")
        self.tangent_dim = int(tangent_dim)
        self.num_charts = len(self.CHART_NAMES)
        self.ambient_dim = self.tangent_dim * self.num_charts

        rng = np.random.default_rng(seed)
        raw = rng.normal(size=(self.ambient_dim, self.ambient_dim))
        q, _ = np.linalg.qr(raw)  # columns of q are exactly orthonormal

        spaces: Dict[str, TangentSpace] = {}
        for i, name in enumerate(self.CHART_NAMES):
            basis = q[:, i * self.tangent_dim : (i + 1) * self.tangent_dim]
            a_raw = rng.normal(size=(self.tangent_dim, self.tangent_dim)) * dynamics_scale
            a, rho = _project_to_contractive(a_raw, target_rho)
            b = rng.normal(size=self.tangent_dim) * bias_scale
            spaces[name] = TangentSpace(name=name, basis=basis, a=a, b=b, rho=rho)
        self.spaces: Dict[str, TangentSpace] = spaces

    def chart(self, name: str) -> TangentSpace:
        try:
            return self.spaces[name]
        except KeyError:
            raise KeyError(f"unknown tangent chart {name!r}; have {self.CHART_NAMES}")

    def orthogonality_matrix(self) -> np.ndarray:
        """``(K, K)`` matrix of ``max(abs(U_i^T U_j))``: the diagonal is 1.0
        (orthonormal columns), every off-diagonal entry is ~0 (disjoint
        blocks of one orthogonal matrix)."""
        n = self.num_charts
        out = np.zeros((n, n), dtype=np.float64)
        names = self.CHART_NAMES
        for i in range(n):
            for j in range(n):
                cross = self.spaces[names[i]].basis.T @ self.spaces[names[j]].basis
                out[i, j] = float(np.max(np.abs(cross)))
        return out

    def spectral_radii(self) -> Dict[str, float]:
        return {name: space.rho for name, space in self.spaces.items()}

    def evolve_all(self, x_ambient: np.ndarray) -> Dict[str, np.ndarray]:
        """Fixed point of every chart's independent contractive recurrence,
        each driven by its own orthogonal projection of the same ambient
        input -- the "collaborative dynamics evolution" interface."""
        return {name: space.fixed_point(x_ambient) for name, space in self.spaces.items()}

    def reconstruct_ambient(self, local_states: Mapping[str, np.ndarray]) -> np.ndarray:
        """Recompose an ambient-space vector by lifting and summing each
        chart's local state. Because the four bases are mutually orthogonal,
        this sum has zero cross-talk: chart k's contribution occupies
        exactly its own orthogonal complement of the ambient space."""
        if set(local_states) != set(self.CHART_NAMES):
            raise ValueError(f"local_states must cover exactly {self.CHART_NAMES}")
        out = np.zeros(self.ambient_dim, dtype=np.float64)
        for name, h in local_states.items():
            out += self.spaces[name].lift(h)
        return out


@dataclasses.dataclass(frozen=True)
class MoEStepResult:
    weights: Dict[str, float]
    fused_expert_state: np.ndarray
    tangent_fixed_points: Dict[str, np.ndarray]
    reconstructed_ambient: np.ndarray


class HeterogeneousCausalMoEPipeline:
    """End-to-end scheduler wiring the OT router, the orthogonal tangent-
    space pool, and the fixed-point dynamics solve into one call."""

    def __init__(self, router: WassersteinOptimalTransportRouter, pool: MultiTangentManifoldPool):
        self.router = router
        self.pool = pool

    def step(
        self,
        router_features,
        expert_state_vectors: Mapping[str, np.ndarray],
        tangent_input,
        *,
        router_covariance: Optional[np.ndarray] = None,
    ) -> MoEStepResult:
        """One MoE decision:

        1. Route: continuous convex weights over the heterogeneous experts.
        2. Fuse: weighted combination of the experts' shared-space outputs.
        3. Project + evolve: drive every orthogonal tangent chart's own
           contractive dynamics to its closed-form fixed point.
        4. Reconstruct: recompose the ambient decision state from the four
           disentangled charts.
        """
        weights = self.router.route(router_features, router_covariance)
        fused_expert_state = self.router.fuse(weights, expert_state_vectors)
        tangent_fixed_points = self.pool.evolve_all(tangent_input)
        reconstructed_ambient = self.pool.reconstruct_ambient(tangent_fixed_points)
        return MoEStepResult(
            weights=weights,
            fused_expert_state=fused_expert_state,
            tangent_fixed_points=tangent_fixed_points,
            reconstructed_ambient=reconstructed_ambient,
        )
