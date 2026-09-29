"""Conditional forcing inside the affine semigroup scan (route 2, F_ext).

Problem this addresses. `parallel_rnn_lora.py` runs h_t = A h_{t-1} + u_t
with sigma_max(A) < 1. If the problem is injected once (u_0 = B x_0, u_t = 0
for t > 0) the state is h_t = A^t B x_0, which decays to the condition-free
fixed point 0. Components along eigendirections of A with small |lambda_i|
(the "high-frequency" clues) die first, as |lambda_i|^t.

Fix. Re-read the problem conditions at every step through a state-dependent
forcing term that stays affine, so every scan element is still an (A_t, u_t)
pair and `associative_scan` stays exact. Two forms, for equality conditions
g(s) = G s - b = 0:

  soft  F_ext(s) = -eta * G^T Lam (G s - b)
        element   (A - eta G^T Lam G,  u + eta G^T Lam b)
        stable only while sigma_max(A - eta G^T Lam G) < 1 (checked, fail closed)

  hard  KKT multipliers solved exactly -> orthogonal projector onto {G s = b}:
        element   (P A,  P u + G^+ b)   with P = I - G^+ G
        ||P A|| <= ||A|| < 1 always, and the unique fixed point satisfies
        G s* = b exactly when b is in range(G) (checked, fail closed).

Non-affine g (relu / bilinear terms) are linearised per chunk at a frozen
point s_bar: G = J(s_bar), b = J(s_bar) s_bar - g(s_bar). The scan is exact
for the linearised system; the gap to the true nonlinear path is second
order in ||s - s_bar||. See docs/research/04-conditional-forcing-and-continuous-reread.md.

Anti-fraud boundary: the condition set is built only from problem-derived
(G, b), is SHA-256 pinned, and never sees candidates or labels. An empty
condition set gives P = I, i.e. the unforced scan: forcing cannot add
information that is not in C. Pure NumPy, no text, no `re`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from .parallel_rnn_lora import associative_scan, sequential_scan

__all__ = [
    "ConditionSet",
    "InfeasibleConditionError",
    "UnstableForcingError",
    "hard_forced_elements",
    "soft_forced_elements",
    "hard_forced_step",
    "fixed_point",
    "impulse_decay",
    "linearize",
    "chunked_linearized_scan",
]


class InfeasibleConditionError(ValueError):
    """b is not in range(G): no state can satisfy every condition."""


class UnstableForcingError(ValueError):
    """The soft-forced transition matrix is not a contraction."""


def _finite(value, name, ndim=None) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: non-finite values")
    if ndim is not None and a.ndim != ndim:
        raise ValueError(f"{name}: expected ndim={ndim}")
    return a


@dataclass(frozen=True)
class ConditionSet:
    """Problem-derived equality conditions G s = b, read-only and hash-pinned."""

    G: np.ndarray
    b: np.ndarray
    digest: str
    pinv: np.ndarray  # G^+ (d, m)

    @classmethod
    def from_arrays(cls, G, b, *, feasibility_tol: float = 1e-8) -> "ConditionSet":
        G = _finite(G, "G", 2)
        b = _finite(b, "b", 1)
        if G.shape[0] != b.shape[0]:
            raise ValueError("G and b disagree on the number of conditions")
        pinv = np.linalg.pinv(G) if G.shape[0] else np.zeros((G.shape[1], 0))
        # Feasibility: the least-squares solution must satisfy the conditions exactly.
        gap = float(np.linalg.norm(G @ (pinv @ b) - b)) if G.shape[0] else 0.0
        if gap > feasibility_tol * max(1.0, float(np.linalg.norm(b))):
            raise InfeasibleConditionError(f"conditions inconsistent: ||G G^+ b - b|| = {gap:.3e}")
        h = hashlib.sha256()
        h.update(G.tobytes()); h.update(b.tobytes()); h.update(str(G.shape).encode())
        for a in (G, b, pinv):
            a.setflags(write=False)
        return cls(G=G, b=b, digest=h.hexdigest(), pinv=pinv)

    @classmethod
    def empty(cls, dim: int) -> "ConditionSet":
        return cls.from_arrays(np.zeros((0, dim)), np.zeros(0))

    @property
    def dim(self) -> int:
        return int(self.G.shape[1])

    @property
    def count(self) -> int:
        return int(self.G.shape[0])

    def projector(self) -> np.ndarray:
        """P = I - G^+ G: orthogonal projector onto the null space of G."""
        return np.eye(self.dim) - self.pinv @ self.G

    def residual(self, s) -> np.ndarray:
        return self.G @ _finite(s, "s", 1) - self.b


def hard_forced_elements(A, u_seq, cond: ConditionSet) -> Tuple[np.ndarray, np.ndarray]:
    """Scan elements (P A, P u_t + G^+ b) for the KKT-exact (projection) forcing."""
    A = _finite(A, "A", 2)
    u_seq = _finite(u_seq, "u_seq", 2)
    if A.shape != (cond.dim, cond.dim) or u_seq.shape[1] != cond.dim:
        raise ValueError("A / u_seq / condition dimension mismatch")
    P = cond.projector()
    offset = cond.pinv @ cond.b
    A_f = np.broadcast_to(P @ A, (u_seq.shape[0],) + A.shape).copy()
    u_f = u_seq @ P.T + offset
    return A_f, u_f


def soft_forced_elements(A, u_seq, cond: ConditionSet, eta: float,
                         lam: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Scan elements (A - eta G^T Lam G, u_t + eta G^T Lam b). Fails closed if not contractive."""
    A = _finite(A, "A", 2)
    u_seq = _finite(u_seq, "u_seq", 2)
    if not np.isfinite(eta) or eta < 0:
        raise ValueError("eta must be finite and >= 0")
    lam = np.ones(cond.count) if lam is None else _finite(lam, "lam", 1)
    if lam.shape != (cond.count,) or np.any(lam < 0):
        raise ValueError("lam must be a non-negative vector with one entry per condition")
    M = cond.G.T @ (lam[:, None] * cond.G)
    A_s = A - eta * M
    sigma = float(np.linalg.svd(A_s, compute_uv=False)[0]) if A_s.size else 0.0
    if sigma >= 1.0:
        raise UnstableForcingError(f"sigma_max(A - eta G^T Lam G) = {sigma:.4f} >= 1")
    A_f = np.broadcast_to(A_s, (u_seq.shape[0],) + A.shape).copy()
    u_f = u_seq + eta * (cond.G.T @ (lam * cond.b))
    return A_f, u_f


def hard_forced_step(step_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                     x_t: np.ndarray, s_prev: np.ndarray, cond: ConditionSet) -> np.ndarray:
    """One projected step without forming any (d, d) matrix.

    s' = v - G^+ (G v - b), v = step_fn(x_t, s_prev).  Working set O(d * m) on
    top of `step_fn`'s O(d + r): the forcing is itself a rank-m side-car.
    """
    v = step_fn(x_t, s_prev)
    if cond.count == 0:
        return v
    return v - cond.pinv @ (cond.G @ v - cond.b)


def fixed_point(A_f: np.ndarray, u_f: np.ndarray) -> np.ndarray:
    """Unique fixed point of s = A_f s + u_f. Requires rho(A_f) < 1 (checked)."""
    A_f = _finite(A_f, "A_f", 2)
    u_f = _finite(u_f, "u_f", 1)
    rho = float(np.max(np.abs(np.linalg.eigvals(A_f))))
    if rho >= 1.0:
        raise UnstableForcingError(f"rho(A_f) = {rho:.4f} >= 1: no unique attracting fixed point")
    return np.linalg.solve(np.eye(A_f.shape[0]) - A_f, u_f)


def impulse_decay(A, B, x0, T: int) -> np.ndarray:
    """||A^t B x_0|| for t = 0..T-1: what survives when x_0 enters only at t = 0."""
    A = _finite(A, "A", 2); B = _finite(B, "B", 2); x0 = _finite(x0, "x0", 1)
    u = np.zeros((T, A.shape[0])); u[0] = B @ x0
    h = sequential_scan(np.broadcast_to(A, (T,) + A.shape).copy(), u)
    return np.linalg.norm(h, axis=1)


def linearize(g: Callable[[np.ndarray], np.ndarray], jac: Callable[[np.ndarray], np.ndarray],
              s_bar: np.ndarray) -> ConditionSet:
    """First-order model g(s) ~ J s - (J s_bar - g(s_bar)) as an affine ConditionSet."""
    s_bar = _finite(s_bar, "s_bar", 1)
    J = _finite(jac(s_bar), "J", 2)
    b = J @ s_bar - _finite(g(s_bar), "g", 1)
    return ConditionSet.from_arrays(J, b)


def chunked_linearized_scan(A, u_seq, g, jac, s0, chunk: int) -> np.ndarray:
    """Per-chunk re-linearised hard forcing, each chunk run through `associative_scan`.

    Within a chunk the linearisation point is the chunk's entry state, and the
    affine scan is exact for that linearised system. Re-linearising at every
    chunk boundary keeps the second-order gap bounded by the chunk length.
    """
    A = _finite(A, "A", 2); u_seq = _finite(u_seq, "u_seq", 2); s = _finite(s0, "s0", 1)
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    out = []
    for start in range(0, u_seq.shape[0], chunk):
        cond = linearize(g, jac, s)
        A_f, u_f = hard_forced_elements(A, u_seq[start:start + chunk], cond)
        u_f = u_f.copy(); u_f[0] += A_f[0] @ s   # carry the chunk entry state
        h = associative_scan(A_f, u_f)
        out.append(h); s = h[-1]
    return np.concatenate(out, axis=0)
