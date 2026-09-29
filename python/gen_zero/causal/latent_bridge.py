"""Latent Manifold Bridge: 0-token thinking between Qwen features and the LodGraph anchors.

Pipeline (one call to `LatentThinkingEngine.run`, no token is ever emitted):

    Segment T  h (R^4096) --W16 (ZCA top-16)--> z (R^16) --retract--> q0 in H^4 x S^3 x R^8
               packed into a 228-byte `LatentEntryContractV1` (gen2 wire format).
    Segment L  damped leapfrog on `MixedManifoldPotential` (softmin over class anchors),
               with a Lyapunov exit check (dH/dt <= 0 under friction).
    Segment R  read the class responsibilities at the final q and write them back into
               the anchor log-weights (co-evolution).

Reference implementations in gen2:
  * entry contract: crates/gen3-operator-kernel/src/engines/algebraic/latent_flow.rs
  * product metric: crates/foundation/manifold/src/product/simd_metric.rs (`fallback_geodesic_sq`)

Pure NumPy. No file-system paths, no network, no tokenizer, no global state.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .symplectic_thinking import ExitReason

MANIFOLD_DIM = 16
CHART_DIM = MANIFOLD_DIM
H_SLICE = slice(0, 4)      # Poincare ball H^4
S_SLICE = slice(4, 8)      # unit 3-sphere in R^4
E_SLICE = slice(8, 16)     # flat R^8
FEATURE_DIM = 4096         # Qwen 3.5 9B hidden size
DEFAULT_ZCA_SIGMA_CEILING = 30.0
LATENT_LATENCY_BUDGET_MS = 35.0

ENTRY_BYTES = 228
ENTRY_DOMAIN = b"LATENT_ENTRY_V1:"
PROJ_DOMAIN = b"PROJ_V1:"
HYPERBOLIC_MARGIN = 1e-4   # keeps q_H strictly inside the unit ball


# --------------------------------------------------------------------------------------
# Entry contract (228 bytes, big-endian, must stay bit-exact with gen2)
# --------------------------------------------------------------------------------------

def _f32_be(x: np.ndarray) -> bytes:
    return np.asarray(x, dtype=">f4").tobytes()


def _bytes32(name: str, value: Union[bytes, bytearray, memoryview]) -> bytes:
    b = bytes(value)
    if len(b) != 32:
        raise ValueError(f"{name} must be exactly 32 bytes, got {len(b)}")
    return b


@dataclasses.dataclass(eq=False)
class LatentEntryContractV1:
    """Cross-repo entry contract (TLDE I2).

    Layout: q0 f32x16 | p0 f32x16 | h0_digest 32 | projection_program_digest 32 |
            zca_sigma_max f32 | tenant_workspace_key 32  = 228 bytes. Floats are big-endian
    IEEE-754 bits, so -0.0 and NaN payloads survive a round trip.
    """
    q0: np.ndarray
    p0: np.ndarray
    h0_digest: bytes
    projection_program_digest: bytes
    zca_sigma_max: float
    tenant_workspace_key: bytes

    def __post_init__(self) -> None:
        self.q0 = np.array(self.q0, dtype=np.float32).reshape(-1)
        self.p0 = np.array(self.p0, dtype=np.float32).reshape(-1)
        if self.q0.shape != (MANIFOLD_DIM,) or self.p0.shape != (MANIFOLD_DIM,):
            raise ValueError(f"q0 and p0 must each hold {MANIFOLD_DIM} floats")
        self.h0_digest = _bytes32("h0_digest", self.h0_digest)
        self.projection_program_digest = _bytes32(
            "projection_program_digest", self.projection_program_digest)
        self.tenant_workspace_key = _bytes32("tenant_workspace_key", self.tenant_workspace_key)
        self.zca_sigma_max = np.float32(self.zca_sigma_max)

    def canonical_bytes(self) -> bytes:
        out = (
            _f32_be(self.q0)
            + _f32_be(self.p0)
            + self.h0_digest
            + self.projection_program_digest
            + _f32_be(np.array([self.zca_sigma_max]))
            + self.tenant_workspace_key
        )
        assert len(out) == ENTRY_BYTES
        return out

    @classmethod
    def from_canonical_bytes(cls, data: bytes) -> "LatentEntryContractV1":
        data = bytes(data)
        if len(data) != ENTRY_BYTES:
            raise ValueError(f"wrong length: expected {ENTRY_BYTES} bytes, got {len(data)}")
        f32 = lambda off, n: np.frombuffer(data, dtype=">f4", count=n, offset=off).astype(np.float32)
        return cls(
            q0=f32(0, 16),
            p0=f32(64, 16),
            h0_digest=data[128:160],
            projection_program_digest=data[160:192],
            zca_sigma_max=f32(192, 1)[0],
            tenant_workspace_key=data[196:228],
        )

    def digest(self) -> bytes:
        """SHA256(b'LATENT_ENTRY_V1:' + canonical_bytes())."""
        return hashlib.sha256(ENTRY_DOMAIN + self.canonical_bytes()).digest()

    def validate(self, ceiling: float = DEFAULT_ZCA_SIGMA_CEILING) -> bool:
        """G-6 guard. False for NaN, Inf, negative, or above ceiling (capped by DEFAULT_ZCA_SIGMA_CEILING)."""
        ceil32 = np.float32(ceiling)
        if not np.isfinite(ceil32) or ceil32 < 0.0:
            return False
        effective_ceiling = min(ceil32, np.float32(DEFAULT_ZCA_SIGMA_CEILING))
        sigma = self.zca_sigma_max
        if not np.isfinite(sigma) or sigma < 0.0 or sigma > effective_ceiling:
            return False
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LatentEntryContractV1):
            return NotImplemented
        return self.canonical_bytes() == other.canonical_bytes()

    def __hash__(self) -> int:
        return hash(self.canonical_bytes())


# --------------------------------------------------------------------------------------
# Mixed manifold H^4 x S^3 x R^8 and its product metric
# --------------------------------------------------------------------------------------

@dataclasses.dataclass
class ManifoldMetricParams:
    """Mirror of gen2 `ManifoldMetricParams` (defaults included)."""
    c_hyperbolic: float = 1.0
    c_spherical: float = 1.0
    w_hyperbolic: float = 1.0
    w_spherical: float = 1.0
    w_euclidean: float = 1.0
    anti_collapse_eps: float = 1e-4

    @property
    def c_h(self) -> float:
        return max(self.c_hyperbolic, 1e-6)

    @property
    def c_s(self) -> float:
        return max(self.c_spherical, 1e-6)

    @property
    def eps(self) -> float:
        return min(max(self.anti_collapse_eps, 1e-5), 0.1)


class MixedManifoldCoord:
    """A point of H^4 x S^3 x R^8 stored as 16 float64 coordinates.

    coords[0:4]  Poincare ball, c_H = 1, norm < 1
    coords[4:8]  unit 3-sphere in R^4, norm == 1
    coords[8:16] flat Euclidean R^8
    """

    def __init__(self, coords: Sequence[float]) -> None:
        c = np.array(coords, dtype=np.float64).reshape(-1)
        if c.shape != (MANIFOLD_DIM,):
            raise ValueError(f"coords must have {MANIFOLD_DIM} entries, got {c.shape}")
        self.coords = c

    @property
    def hyperbolic(self) -> np.ndarray:
        return self.coords[H_SLICE]

    @property
    def spherical(self) -> np.ndarray:
        return self.coords[S_SLICE]

    @property
    def euclidean(self) -> np.ndarray:
        return self.coords[E_SLICE]

    def is_valid(self, c_h: float = 1.0, tol: float = 1e-6) -> bool:
        if not np.all(np.isfinite(self.coords)):
            return False
        in_ball = c_h * float(self.hyperbolic @ self.hyperbolic) < 1.0
        on_sphere = abs(float(np.linalg.norm(self.spherical)) - 1.0) <= tol
        return bool(in_ball and on_sphere)

    @staticmethod
    def retract(z: np.ndarray, c_h: float = 1.0) -> "MixedManifoldCoord":
        return MixedManifoldCoord(retract_to_manifold(z, c_h=c_h))

    @staticmethod
    def fallback_geodesic_sq(
        x: "CoordLike", y: "CoordLike", params: Optional[ManifoldMetricParams] = None
    ) -> float:
        return fallback_geodesic_sq(x, y, params)

    def __repr__(self) -> str:
        return f"MixedManifoldCoord({np.array2string(self.coords, precision=4)})"


CoordLike = Union[MixedManifoldCoord, np.ndarray, Sequence[float]]


def _as_coords(x: CoordLike) -> np.ndarray:
    if isinstance(x, MixedManifoldCoord):
        return x.coords
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.shape != (MANIFOLD_DIM,):
        raise ValueError(f"coords must have {MANIFOLD_DIM} entries, got {a.shape}")
    return a


def retract_to_manifold(z: np.ndarray, c_h: float = 1.0) -> np.ndarray:
    """Map a raw R^16 vector onto H^4 x S^3 x R^8 (the retraction used by the projector)."""
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if z.shape != (MANIFOLD_DIM,):
        raise ValueError(f"z must have {MANIFOLD_DIM} entries, got {z.shape}")
    q = np.empty(MANIFOLD_DIM)
    z_h = z[H_SLICE]
    r = float(np.linalg.norm(z_h))
    curv_h = max(4.0, float(c_h))
    limit = (1.0 - HYPERBOLIC_MARGIN) / math.sqrt(curv_h)
    scale_h = math.tanh(r * math.sqrt(curv_h)) * limit / max(r, 1e-12)
    q[H_SLICE] = z_h * scale_h
    z_s = z[S_SLICE]
    n_s = float(np.linalg.norm(z_s))
    if n_s < 1e-12:
        # A zero vector has no direction. Pick a fixed pole so the point stays on S^3.
        q[S_SLICE] = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        q[S_SLICE] = z_s / max(n_s, 1e-12)
    q[E_SLICE] = z[E_SLICE]
    return q


def fallback_geodesic_sq(
    x: CoordLike, y: CoordLike, params: Optional[ManifoldMetricParams] = None
) -> float:
    """Scalar reference for the product metric d^2, term for term as in gen2 simd_metric.rs.

    Cross-language parity note: gen2 `crates/foundation/manifold/src/product/simd_metric.rs`
    uses SIMD-accelerated rational/polynomial Chebyshev approximations for hyperbolic
    and spherical geodesics with a maximum tolerance |d_approx - d_exact| <= 3e-4 over
    the operational domain ||x||_H < 0.95 / sqrt(c_h) and theta in [0, pi - 1e-4].
    This reference provides the exact float64 analytic formulation.
    """
    p = params or ManifoldMetricParams()
    xc, yc = _as_coords(x), _as_coords(y)
    c_h, c_s, eps = p.c_h, p.c_s, p.eps

    diff_h_sq = norm_x_h = norm_y_h = 0.0
    for i in range(0, 4):
        d = xc[i] - yc[i]
        diff_h_sq += d * d
        norm_x_h += xc[i] * xc[i]
        norm_y_h += yc[i] * yc[i]
    denom_x = max(1.0 - c_h * norm_x_h, eps)
    denom_y = max(1.0 - c_h * norm_y_h, eps)
    delta = (2.0 * c_h * diff_h_sq) / (denom_x * denom_y)
    d_h = math.acosh(1.0 + delta) / math.sqrt(c_h)

    dot_s = 0.0
    for i in range(4, 8):
        dot_s += xc[i] * yc[i]
    d_s = math.acos(min(max(dot_s, -1.0), 1.0)) / math.sqrt(c_s)

    dist_sq_e = 0.0
    for i in range(8, 16):
        d = xc[i] - yc[i]
        dist_sq_e += d * d

    return (
        p.w_hyperbolic * d_h * d_h
        + p.w_spherical * d_s * d_s
        + p.w_euclidean * dist_sq_e
    )


def geodesic_sq_and_grad(
    q: np.ndarray, anchors: np.ndarray, params: ManifoldMetricParams
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised d^2(q, c_i) and its analytic gradient w.r.t. q, for all anchors at once.

    Returns (d2 of shape (n,), grad of shape (n, 16)). The gradient is the ambient (Euclidean)
    gradient of the same function `fallback_geodesic_sq` evaluates.
    """
    c_h, c_s, eps = params.c_h, params.c_s, params.eps
    n = anchors.shape[0]
    grad = np.empty((n, MANIFOLD_DIM))

    # --- H^4 ---
    xh, yh = q[H_SLICE], anchors[:, H_SLICE]
    diff = xh - yh
    diff_sq = np.einsum("ij,ij->i", diff, diff)
    a_raw = 1.0 - c_h * float(xh @ xh)
    b_raw = 1.0 - c_h * np.einsum("ij,ij->i", yh, yh)
    a = max(a_raw, eps)
    b = np.maximum(b_raw, eps)
    ab = a * b
    delta = 2.0 * c_h * diff_sq / ab
    root = np.sqrt(delta * (delta + 2.0))
    arcosh = np.log1p(delta + root)  # arcosh(1 + delta), stable for small delta
    d_h_sq = arcosh * arcosh / c_h
    # d(d_h^2)/d(delta) = 2 * arcosh / (c_h * sqrt(delta (delta + 2))); the ratio tends to 1.
    ratio = np.where(root > 1e-12, arcosh / np.maximum(root, 1e-300), 1.0)
    dd_ddelta = 2.0 * ratio / c_h
    ddelta_dx = 4.0 * c_h * diff / ab[:, None]
    if a_raw > eps:  # clamped denominators are constants
        ddelta_dx = ddelta_dx + (delta * 2.0 * c_h / a)[:, None] * xh
    grad[:, H_SLICE] = params.w_hyperbolic * dd_ddelta[:, None] * ddelta_dx

    # --- S^3 ---
    xs, ys = q[S_SLICE], anchors[:, S_SLICE]
    dot_raw = ys @ xs
    dot = np.clip(dot_raw, -1.0 + 1e-7, 1.0 - 1e-7)
    ang = np.arccos(dot)
    d_s_sq = ang * ang / c_s
    sin = np.sqrt(np.maximum(1.0 - dot * dot, 1e-14))
    # Near theta -> 0, theta/sin(theta) -> 1. Near theta -> pi, avoid spurious zero tangent
    sin_safe = np.maximum(sin, 1e-6)
    ratio_s = np.where(dot > 0.999999, 1.0, ang / sin_safe)
    in_bounds = (dot_raw >= -1.0 + 1e-7) & (dot_raw <= 1.0 - 1e-7)
    unclamped_grad = (params.w_spherical * (-2.0 * ratio_s / c_s))[:, None] * ys
    grad[:, S_SLICE] = np.where(in_bounds[:, None], unclamped_grad, 0.0)

    # --- R^8 ---
    diff_e = q[E_SLICE] - anchors[:, E_SLICE]
    d_e = np.einsum("ij,ij->i", diff_e, diff_e)
    grad[:, E_SLICE] = params.w_euclidean * 2.0 * diff_e

    d2 = params.w_hyperbolic * d_h_sq + params.w_spherical * d_s_sq + params.w_euclidean * d_e
    return d2, grad


def _logsumexp(a: np.ndarray) -> float:
    m = float(np.max(a))
    if not math.isfinite(m):
        return m
    return m + math.log(float(np.sum(np.exp(a - m))))


# --------------------------------------------------------------------------------------
# Chart projector: R^4096 -> H^4 x S^3 x R^8
# --------------------------------------------------------------------------------------

class MixedManifoldChartProjector:
    """Linear ZCA chart W16 (16 x feature_dim) followed by the block retraction.

    `fit` takes the top-16 principal directions of the centred features and whitens them,
    W16 = diag(1 / sqrt(lambda_k + reg)) U_k^T. The spectral norm ||W16||_2 is the declared
    `zca_sigma_max`. If whitening would push it past `ceiling`, W16 is scaled down so the
    contract still passes the G-6 guard. `rescaled` records that this happened.
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        ceiling: float = DEFAULT_ZCA_SIGMA_CEILING,
        reg: float = 1e-6,
    ) -> None:
        if feature_dim < MANIFOLD_DIM:
            raise ValueError(f"feature_dim must be >= {MANIFOLD_DIM}")
        self.feature_dim = int(feature_dim)
        self.ceiling = float(ceiling)
        self.reg = float(reg)
        self.mu: Optional[np.ndarray] = None
        self.w16: Optional[np.ndarray] = None
        self.zca_sigma_max: float = float("nan")
        self.projection_program_digest: bytes = b""
        self.rescaled = False

    @property
    def fitted(self) -> bool:
        return self.w16 is not None

    def fit(self, features: np.ndarray) -> "MixedManifoldChartProjector":
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape (N, {self.feature_dim}), got {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("features contain NaN or Inf")
        n = x.shape[0]
        if n < MANIFOLD_DIM + 1:
            raise ValueError(f"need at least {MANIFOLD_DIM + 1} samples to fit, got {n}")

        mu = x.mean(axis=0)
        _, s, vt = np.linalg.svd(x - mu, full_matrices=False)
        lam = (s[:MANIFOLD_DIM] ** 2) / (n - 1)
        u = vt[:MANIFOLD_DIM].copy()
        # SVD signs are arbitrary. Fix them so the same data gives the same digest.
        signs = np.sign(u[np.arange(MANIFOLD_DIM), np.argmax(np.abs(u), axis=1)])
        u *= np.where(signs == 0.0, 1.0, signs)[:, None]
        w = u / np.sqrt(lam + self.reg)[:, None]

        w32 = w.astype(np.float32)
        sigma = float(np.linalg.norm(w32.astype(np.float64), 2))
        self.rescaled = sigma > self.ceiling
        if self.rescaled:
            w32 = (w32 * np.float32(self.ceiling * (1.0 - 1e-4) / sigma)).astype(np.float32)
            sigma = float(np.linalg.norm(w32.astype(np.float64), 2))
        sigma32 = float(np.float32(sigma))
        if not sigma32 <= self.ceiling:
            raise ValueError(f"zca_sigma_max {sigma32} exceeds ceiling {self.ceiling}")

        self.mu = mu.astype(np.float32)
        self.w16 = w32
        self.zca_sigma_max = sigma32
        self.projection_program_digest = hashlib.sha256(
            PROJ_DOMAIN + _f32_be(self.mu) + _f32_be(self.w16)
        ).digest()
        return self

    def _require_fit(self) -> None:
        if not self.fitted:
            raise RuntimeError("projector is not fitted; call fit() first")

    def _check_h(self, h: np.ndarray) -> np.ndarray:
        h = np.asarray(h, dtype=np.float64).reshape(-1)
        if h.shape != (self.feature_dim,):
            raise ValueError(f"h must have shape ({self.feature_dim},), got {h.shape}")
        if not np.all(np.isfinite(h)):
            raise ValueError("h contains NaN or Inf")
        return h

    def project(self, h: np.ndarray) -> np.ndarray:
        """h -> q in H^4 x S^3 x R^8 (float64, shape (16,))."""
        self._require_fit()
        h = self._check_h(h)
        z = self.w16.astype(np.float64) @ (h - self.mu.astype(np.float64))
        return retract_to_manifold(z)

    def build_contract(
        self, h: np.ndarray, tenant_key: bytes, p0: Optional[np.ndarray] = None
    ) -> LatentEntryContractV1:
        """Pack the projected state into the 228-byte entry contract. p0 defaults to rest."""
        self._require_fit()
        h = self._check_h(h)
        tenant_key = _bytes32("tenant_key", tenant_key)
        q0 = self.project(h)
        p0 = np.zeros(MANIFOLD_DIM) if p0 is None else np.asarray(p0, dtype=np.float64)
        return LatentEntryContractV1(
            q0=q0.astype(np.float32),
            p0=p0.astype(np.float32),
            h0_digest=hashlib.sha256(_f32_be(h)).digest(),
            projection_program_digest=self.projection_program_digest,
            zca_sigma_max=self.zca_sigma_max,
            tenant_workspace_key=tenant_key,
        )


# --------------------------------------------------------------------------------------
# Potential over class anchors
# --------------------------------------------------------------------------------------

class MixedManifoldPotential:
    """Hopfield / softmin energy over class anchor nodes.

        V(q) = -1/beta * log sum_i w_i exp(-beta * d_M^2(q, c_i))

    `log_w` holds log w_i, normalised so sum w_i = 1. grad V = sum_i r_i grad d^2(q, c_i),
    where r_i is the softmax responsibility. The gradient is analytic (no autograd).
    """

    def __init__(
        self,
        anchors: np.ndarray,
        log_w: Optional[np.ndarray] = None,
        beta: float = 4.0,
        params: Optional[ManifoldMetricParams] = None,
    ) -> None:
        a = np.array(anchors, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != MANIFOLD_DIM or a.shape[0] < 1:
            raise ValueError(f"anchors must have shape (n, {MANIFOLD_DIM}), n >= 1")
        if not np.all(np.isfinite(a)):
            raise ValueError("anchors contain NaN or Inf")
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        self.anchors = a
        self.beta = float(beta)
        self.params = params or ManifoldMetricParams()
        lw = np.zeros(a.shape[0]) if log_w is None else np.array(log_w, dtype=np.float64)
        if lw.shape != (a.shape[0],) or not np.all(np.isfinite(lw)):
            raise ValueError("log_w must be finite with one entry per anchor")
        self.log_w = lw - _logsumexp(lw)

    @property
    def n_anchors(self) -> int:
        return self.anchors.shape[0]

    @property
    def weights(self) -> np.ndarray:
        return np.exp(self.log_w)

    @classmethod
    def from_class_features(
        cls,
        projector: MixedManifoldChartProjector,
        class_features: np.ndarray,
        **kwargs: Any,
    ) -> "MixedManifoldPotential":
        """Build anchors by projecting one feature vector per class through the chart."""
        rows = [projector.project(h) for h in np.asarray(class_features)]
        return cls(np.stack(rows), **kwargs)

    def energy_and_grad(self, q: np.ndarray) -> Tuple[float, np.ndarray]:
        q = np.asarray(q, dtype=np.float64)
        d2, gd2 = geodesic_sq_and_grad(q, self.anchors, self.params)
        a = self.log_w - self.beta * d2
        lse = _logsumexp(a)
        r = np.exp(a - lse)
        return -lse / self.beta, r @ gd2

    def energy(self, q: np.ndarray) -> float:
        return self.energy_and_grad(q)[0]

    def grad(self, q: np.ndarray) -> np.ndarray:
        return self.energy_and_grad(q)[1]

    def responsibilities(self, q: np.ndarray) -> np.ndarray:
        d2, _ = geodesic_sq_and_grad(np.asarray(q, dtype=np.float64), self.anchors, self.params)
        a = self.log_w - self.beta * d2
        return np.exp(a - _logsumexp(a))

    def coevolve(self, r: np.ndarray, eta: float) -> np.ndarray:
        """Write-back: log w_i += eta * r_i, then renormalise. Returns the new log-weights.

        Validates that eta is positive and finite, and that r is a valid probability simplex vector
        (all finite, non-negative, and summing to 1.0 within numerical tolerance). Invalid inputs
        leave the weights strictly unchanged.
        """
        if not math.isfinite(eta) or eta <= 0.0:
            return self.log_w
        r = np.asarray(r, dtype=np.float64)
        if r.shape != self.log_w.shape:
            raise ValueError("r must have one entry per anchor")
        if not np.all(np.isfinite(r)):
            return self.log_w
        if np.any(r < 0.0) or not np.isclose(float(np.sum(r)), 1.0, atol=1e-3):
            return self.log_w
        lw = self.log_w + eta * r
        lw = np.clip(lw, -20.0, 20.0)
        self.log_w = lw - _logsumexp(lw)
        return self.log_w


# --------------------------------------------------------------------------------------
# Thinking engine
# --------------------------------------------------------------------------------------

@dataclasses.dataclass
class LatentThinkingResult:
    """Outcome of one `LatentThinkingEngine.run`. `tokens_emitted` is 0 by construction."""
    q: np.ndarray
    p: np.ndarray
    steps: int
    max_steps: int
    exit_reason: ExitReason
    initial_energy: float
    final_energy: float
    energy_trace: List[float]
    responsibilities: np.ndarray
    predicted: int
    confidence: float
    segment_t_ms: float = 0.0
    segment_l_ms: float = 0.0
    segment_r_ms: float = 0.0
    contract_digest: bytes = b""
    tokens_emitted: int = 0

    @property
    def early_exit(self) -> bool:
        return self.exit_reason is not ExitReason.MaxSteps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "max_steps": self.max_steps,
            "exit_reason": self.exit_reason.value,
            "early_exit": self.early_exit,
            "initial_energy": self.initial_energy,
            "final_energy": self.final_energy,
            "predicted": self.predicted,
            "confidence": self.confidence,
            "segment_t_ms": self.segment_t_ms,
            "segment_l_ms": self.segment_l_ms,
            "segment_r_ms": self.segment_r_ms,
            "tokens_emitted": self.tokens_emitted,
        }


class LatentThinkingEngine:
    """Damped leapfrog rollout on the mixed-manifold potential.

    Extrinsic dissipative descent scheme: the ambient gradient drives (q, p) in R^16.
    After each drift, q is retracted onto the product manifold H^4 x S^3 x R^8 (ball radius
    clipped, S^3 block normalised), and the S^3 component of momentum and force is projected
    onto the tangent space T_q S^3. Conformal damping is applied via an exact exponential
    half-step on each side: p -> exp(-gamma * dt / 2) * p.

    Design note: This extrinsic retraction and tangential projection introduces non-zero
    projection dissipation. It is an energy descent engine (Lyapunov stable, dH/dt <= 0)
    designed for rapid convergence to local minima, not an unconstrained symplectic map;
    it does not preserve phase-space volume or time-reversal symmetry. (For strict Hamiltonian
    symplectomorphisms on Euclidean spaces, see `SymplecticThinker`).

    Cooperative latency deadline: evaluated against LATENT_LATENCY_BUDGET_MS (35.0 ms)
    at the start, before each step, and after gradient evaluations. If exceeded, halts
    further integration and returns ExitReason.Stalled without co-evolution write-back.
    """

    def __init__(
        self,
        potential: MixedManifoldPotential,
        projector: Optional[MixedManifoldChartProjector] = None,
        dt: float = 0.05,
        gamma: float = 1.0,
        max_steps: int = 32,
        grad_tol: float = 1e-4,
        momentum_tol: float = 1e-4,
        stall_tol: float = 1e-9,
        patience: int = 3,
        lyapunov_slack: float = 1e-3,
        eta: float = 0.1,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if gamma < 0.0:
            raise ValueError("gamma must be >= 0")
        if max_steps < 1 or patience < 1:
            raise ValueError("max_steps and patience must be >= 1")
        self.potential = potential
        self.projector = projector
        self.dt = float(dt)
        self.gamma = float(gamma)
        self.max_steps = int(max_steps)
        self.grad_tol = float(grad_tol)
        self.momentum_tol = float(momentum_tol)
        self.stall_tol = float(stall_tol)
        self.patience = int(patience)
        self.lyapunov_slack = float(lyapunov_slack)
        self.eta = float(eta)
        self.tokens_emitted = 0  # never incremented: this engine has no decoder

    # ---- manifold helpers ----------------------------------------------------------

    @staticmethod
    def _retract(q: np.ndarray, c_h: float = 1.0) -> np.ndarray:
        q = q.copy()
        r = float(np.linalg.norm(q[H_SLICE]))
        curv_h = max(1e-6, float(c_h))
        limit = (1.0 - HYPERBOLIC_MARGIN) / math.sqrt(curv_h)
        if r > limit:
            q[H_SLICE] *= limit / r
        n_s = float(np.linalg.norm(q[S_SLICE]))
        if n_s < 1e-12:
            q[S_SLICE] = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            q[S_SLICE] /= n_s
        return q

    @staticmethod
    def _tangent(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        v = v.copy()
        qs = q[S_SLICE]
        v[S_SLICE] -= float(v[S_SLICE] @ qs) * qs
        return v

    def hamiltonian(self, q: np.ndarray, p: np.ndarray) -> float:
        return 0.5 * float(p @ p) + self.potential.energy(q)

    # ---- rollout (Segment L) -------------------------------------------------------

    def rollout(
        self, q0: np.ndarray, p0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, int, ExitReason, List[float]]:
        q0 = np.asarray(q0, dtype=np.float64).reshape(-1)
        if q0.shape != (MANIFOLD_DIM,):
            raise ValueError(f"q0 must have shape ({MANIFOLD_DIM},)")
        p = np.zeros(MANIFOLD_DIM) if p0 is None else np.asarray(p0, dtype=np.float64).reshape(-1)
        if p.shape != (MANIFOLD_DIM,):
            raise ValueError(f"p0 must have shape ({MANIFOLD_DIM},)")
        if not (np.all(np.isfinite(q0)) and np.all(np.isfinite(p))):
            return q0.copy(), p.copy(), 0, ExitReason.NonFinite, []

        t_start = time.perf_counter_ns()
        curv_h = self.potential.params.c_h
        q = self._retract(q0, curv_h)
        p = self._tangent(q, p)
        energy_v, g = self.potential.energy_and_grad(q)
        g = self._tangent(q, g)
        energy = 0.5 * float(p @ p) + energy_v
        trace = [energy]
        if (time.perf_counter_ns() - t_start) / 1e6 >= LATENT_LATENCY_BUDGET_MS:
            return q, p, 0, ExitReason.Stalled, trace

        damp = math.exp(-self.gamma * self.dt * 0.5)
        half = 0.5 * self.dt
        stalled = 0
        reason = ExitReason.MaxSteps
        steps = 0

        last_accepted_q = q.copy()
        last_accepted_p = p.copy()

        for step_i in range(1, self.max_steps + 1):
            if (time.perf_counter_ns() - t_start) / 1e6 >= LATENT_LATENCY_BUDGET_MS:
                reason = ExitReason.Stalled
                break

            p_cand = damp * p - half * g
            q_cand = self._retract(q + self.dt * p_cand, curv_h)
            p_cand = self._tangent(q_cand, p_cand)
            v_new, g_new = self.potential.energy_and_grad(q_cand)
            g_cand = self._tangent(q_cand, g_new)
            p_cand = damp * (p_cand - half * g_cand)
            new_energy = 0.5 * float(p_cand @ p_cand) + v_new

            if (time.perf_counter_ns() - t_start) / 1e6 >= LATENT_LATENCY_BUDGET_MS:
                reason = ExitReason.Stalled
                q, p = last_accepted_q, last_accepted_p
                break

            if not (math.isfinite(new_energy) and np.all(np.isfinite(q_cand)) and np.all(np.isfinite(p_cand))):
                reason = ExitReason.NonFinite
                q, p = last_accepted_q, last_accepted_p
                break

            delta = new_energy - energy
            scale = max(1.0, abs(energy))

            # dH/dt <= 0 holds only with friction; undamped H is conserved up to O(dt^2).
            if self.gamma > 0.0 and delta > self.lyapunov_slack * scale:
                reason = ExitReason.LyapunovViolation
                # Roll back to last accepted state preserving invariant
                q, p = last_accepted_q, last_accepted_p
                break

            # Accept state
            steps = step_i
            q, p, g = q_cand, p_cand, g_cand
            energy = new_energy
            trace.append(energy)
            last_accepted_q = q.copy()
            last_accepted_p = p.copy()

            # Check convergence, guarding against unstable antipodal maximum on S^3
            dot_s = self.potential.anchors[:, S_SLICE] @ q[S_SLICE]
            at_antipodal = np.any(dot_s <= -0.999)
            if at_antipodal:
                reason = ExitReason.Stalled
                break

            if math.sqrt(float(g @ g)) <= self.grad_tol and \
                    math.sqrt(float(p @ p)) <= self.momentum_tol:
                reason = ExitReason.Converged
                break
            stalled = stalled + 1 if abs(delta) <= self.stall_tol * scale else 0
            if stalled >= self.patience:
                reason = ExitReason.Stalled
                break

        return q, p, steps, reason, trace

    # ---- full bridge ---------------------------------------------------------------

    def run(
        self,
        h: np.ndarray,
        tenant_key: bytes,
        coevolve: bool = True,
    ) -> LatentThinkingResult:
        """Segment T (project) -> Segment L (trajectory) -> Segment R (readout + write-back)."""
        if self.projector is None:
            raise RuntimeError("run() needs a fitted projector; use rollout() for raw states")

        t0 = time.perf_counter()
        contract = self.projector.build_contract(h, tenant_key)
        if not contract.validate(self.projector.ceiling):
            raise ValueError("entry contract refused: zca_sigma_max outside the ceiling")
        q0 = contract.q0.astype(np.float64)
        p0 = contract.p0.astype(np.float64)
        t1 = time.perf_counter()

        q, p, steps, reason, trace = self.rollout(q0, p0)
        t2 = time.perf_counter()

        r = self.potential.responsibilities(q)
        predicted = int(np.argmax(r))
        if coevolve and reason == ExitReason.Converged and np.all(np.isfinite(r)):
            self.potential.coevolve(r, self.eta)
        t3 = time.perf_counter()

        return LatentThinkingResult(
            q=q,
            p=p,
            steps=steps,
            max_steps=self.max_steps,
            exit_reason=reason,
            initial_energy=trace[0] if trace else float("nan"),
            final_energy=trace[-1] if trace else float("nan"),
            energy_trace=trace,
            responsibilities=r,
            predicted=predicted,
            confidence=float(r[predicted]),
            segment_t_ms=(t1 - t0) * 1e3,
            segment_l_ms=(t2 - t1) * 1e3,
            segment_r_ms=(t3 - t2) * 1e3,
            contract_digest=contract.digest(),
            tokens_emitted=0,
        )
