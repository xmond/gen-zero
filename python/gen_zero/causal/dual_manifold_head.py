"""Dual-manifold state-action head (Spec 23 section 1, Spec 24 sections 4A/5/8.6).

State and action live on separate learned manifolds (``z_s``, ``z_a``). A
bilinear form ``S(s,a) = z_s^T M z_a + b`` scores state-action pairs. Spec 23
section 1.3 gives the offline precompile theorem (cache ``W = A M^T`` once
per fixed candidate set, one online GEMV per decision) and its folded raw-input
variant. Spec 24 section 5.2 gives a *separable* (Kronecker-assumption)
variance estimate for a linear-Gaussian posterior; it is explicitly not the
full ``vec(W)`` posterior (see ``variance()`` docstring). Spec 24 section
8.6.4 (theorem 11) gives the three-state semiring axiom mask: accept inside
support, refuse on empty support, unresolved on multi-support without a
variance gate. Section 4A / the tier table gives the SLA contract: a decision
is timed end-to-end and compared, after the fact, against a fixed per-tier
ceiling -- exceeding it returns ``TIMEOUT``, never a silently truncated
answer.

No task id, no label, no language-specific logic anywhere in this module.
"""
from __future__ import annotations

import dataclasses
import enum
import time
from statistics import NormalDist
from typing import Callable, Dict, Optional

import numpy as np

__all__ = [
    "simplex_etf",
    "TierContract",
    "TIERS",
    "Status",
    "DecisionResult",
    "DualManifoldHead",
]


def simplex_etf(K: int, d: int, seed: int = 0) -> np.ndarray:
    """Return a (K, d) equiangular tight frame: unit rows, pairwise cosine -1/(K-1).

    Construction: build the K-simplex ETF in R^K,
    ``sqrt(K/(K-1)) * (I_K - 11^T/K)``, then embed its rows into R^d with a
    random orthonormal (d, K) basis (Q factor of a seeded Gaussian QR). Q has
    orthonormal columns, so it is an isometry: it preserves both row norms
    and pairwise inner products exactly (up to float error).
    """
    if not (2 <= K <= d):
        raise ValueError(f"simplex_etf requires 2 <= K <= d, got K={K}, d={d}")
    base = np.sqrt(K / (K - 1)) * (np.eye(K, dtype=np.float64) - np.ones((K, K), dtype=np.float64) / K)
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((d, K))
    basis, _ = np.linalg.qr(gaussian)  # (d, K), orthonormal columns
    return base @ basis.T


@dataclasses.dataclass(frozen=True)
class TierContract:
    p99_target_us: float
    ceiling_us: float
    strict: bool  # True: ceiling is a strict "<"; False: ceiling is "<="


TIERS: Dict[str, TierContract] = {
    "tier0": TierContract(p99_target_us=50.0, ceiling_us=50.0, strict=True),
    "tier1": TierContract(p99_target_us=500.0, ceiling_us=1000.0, strict=False),
    "tier2": TierContract(p99_target_us=5000.0, ceiling_us=20000.0, strict=False),
}


class Status(str, enum.Enum):
    ACCEPT = "ACCEPT"
    REFUSE_EMPTY_SUPPORT = "REFUSE_EMPTY_SUPPORT"
    REFUSE_UNCERTAIN = "REFUSE_UNCERTAIN"
    TIMEOUT = "TIMEOUT"


@dataclasses.dataclass(frozen=True)
class DecisionResult:
    status: Status
    action_index: Optional[int]
    scores: np.ndarray
    guarded_scores: np.ndarray
    sigma2: Optional[np.ndarray]
    elapsed_us: float
    tier: str
    ceiling_us: float
    note: str

    def __post_init__(self) -> None:
        if self.status != Status.ACCEPT and self.action_index is not None:
            raise ValueError("action_index must be None whenever status is not ACCEPT")


def _check_finite(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")


class DualManifoldHead:
    """Bilinear state-action scorer with separate state/action projections.

    ``z_s = P_s^T (x_s - mu_s)``, ``z_a = P_a^T (x_a - mu_a)``,
    ``S(s,a) = z_s^T M z_a + b`` (Spec 23 section 1.2). Projections may be
    rectangular. ``Sigma_s``/``Sigma_a`` are optional symmetric PSD priors
    used only by the separable variance estimate (Spec 24 section 5.2 item 2).
    """

    def __init__(
        self,
        P_s: np.ndarray,
        mu_s: np.ndarray,
        P_a: np.ndarray,
        mu_a: np.ndarray,
        M: np.ndarray,
        b: float,
        Sigma_s: Optional[np.ndarray] = None,
        Sigma_a: Optional[np.ndarray] = None,
    ) -> None:
        P_s = np.asarray(P_s, dtype=np.float64)
        mu_s = np.asarray(mu_s, dtype=np.float64)
        P_a = np.asarray(P_a, dtype=np.float64)
        mu_a = np.asarray(mu_a, dtype=np.float64)
        M = np.asarray(M, dtype=np.float64)
        b = float(b)

        if P_s.ndim != 2:
            raise ValueError("P_s must be 2D (D_s, d_s)")
        D_s, d_s = P_s.shape
        if mu_s.shape != (D_s,):
            raise ValueError(f"mu_s must have shape ({D_s},)")
        if P_a.ndim != 2:
            raise ValueError("P_a must be 2D (D_a, d_a)")
        D_a, d_a = P_a.shape
        if mu_a.shape != (D_a,):
            raise ValueError(f"mu_a must have shape ({D_a},)")
        if M.shape != (d_s, d_a):
            raise ValueError(f"M must have shape ({d_s}, {d_a})")

        for name, arr in (("P_s", P_s), ("mu_s", mu_s), ("P_a", P_a), ("mu_a", mu_a), ("M", M)):
            _check_finite(name, arr)
        if not np.isfinite(b):
            raise ValueError("b must be finite")

        if (Sigma_s is None) != (Sigma_a is None):
            raise ValueError("Sigma_s and Sigma_a must both be given or both be omitted")

        if Sigma_s is not None:
            Sigma_s = np.asarray(Sigma_s, dtype=np.float64)
            Sigma_a = np.asarray(Sigma_a, dtype=np.float64)
            if Sigma_s.shape != (d_s, d_s):
                raise ValueError(f"Sigma_s must have shape ({d_s}, {d_s})")
            if Sigma_a.shape != (d_a, d_a):
                raise ValueError(f"Sigma_a must have shape ({d_a}, {d_a})")
            for name, S in (("Sigma_s", Sigma_s), ("Sigma_a", Sigma_a)):
                _check_finite(name, S)
                if np.max(np.abs(S - S.T)) > 1e-12:
                    raise ValueError(f"{name} must be symmetric within 1e-12")
                if np.linalg.eigvalsh(S).min() < -1e-10:
                    raise ValueError(f"{name} must be positive semi-definite")

        self.P_s = P_s
        self.mu_s = mu_s
        self.P_a = P_a
        self.mu_a = mu_a
        self.M = M
        self.b = b
        self.Sigma_s = Sigma_s
        self.Sigma_a = Sigma_a
        self.D_s, self.d_s = D_s, d_s
        self.D_a, self.d_a = D_a, d_a

    @property
    def has_sigma(self) -> bool:
        return self.Sigma_s is not None

    def project_state(self, x_s: np.ndarray) -> np.ndarray:
        x_s = np.asarray(x_s, dtype=np.float64)
        if x_s.shape != (self.D_s,):
            raise ValueError(f"x_s must have shape ({self.D_s},)")
        _check_finite("x_s", x_s)
        return self.P_s.T @ (x_s - self.mu_s)

    def project_action(self, x_a: np.ndarray) -> np.ndarray:
        x_a = np.asarray(x_a, dtype=np.float64)
        if x_a.ndim == 1:
            if x_a.shape != (self.D_a,):
                raise ValueError(f"x_a must have shape ({self.D_a},)")
            _check_finite("x_a", x_a)
            return self.P_a.T @ (x_a - self.mu_a)
        if x_a.ndim == 2:
            if x_a.shape[1] != self.D_a:
                raise ValueError(f"x_a must have shape (K, {self.D_a})")
            _check_finite("x_a", x_a)
            return (x_a - self.mu_a) @ self.P_a
        raise ValueError("x_a must be 1D (D_a,) or 2D (K, D_a)")

    def score_bilinear(self, z_s: np.ndarray, Z_a: np.ndarray) -> np.ndarray:
        """(d_s,), (K, d_a) -> (K,). Two-GEMV path: M^T z_s once, then Z_a @ (.)."""
        z_s = np.asarray(z_s, dtype=np.float64)
        Z_a = np.asarray(Z_a, dtype=np.float64)
        return Z_a @ (self.M.T @ z_s) + self.b

    def precompile(self, Z_a: np.ndarray):
        """Spec 23 theorem 1: cache W = A M^T, c = b*1 once per fixed candidate set."""
        Z_a = np.asarray(Z_a, dtype=np.float64)
        W = Z_a @ self.M.T
        c = self.b * np.ones(Z_a.shape[0], dtype=np.float64)
        return W, c

    def precompile_folded(self, Z_a: np.ndarray):
        """Fold P_s/mu_s into W,c so scoring works directly on raw x_s (Spec 23 line 194)."""
        W, c = self.precompile(Z_a)
        W_fold = W @ self.P_s.T
        c_fold = c - W_fold @ self.mu_s
        return W_fold, c_fold

    def score_precompiled(self, z_s: np.ndarray, W: np.ndarray, c: np.ndarray) -> np.ndarray:
        """One GEMV using a cached W, c from precompile()/precompile_folded()."""
        z_s = np.asarray(z_s, dtype=np.float64)
        return W @ z_s + c

    def variance(self, z_s: np.ndarray, Z_a: np.ndarray) -> np.ndarray:
        """Separable (Kronecker-assumption) epistemic variance, Spec 24 section 5.2 item 2.

        ``sigma2_k = (z_ak^T Sigma_a z_ak) * (z_s^T Sigma_s z_s)``. This
        assumes the full ``vec(W)`` posterior factors as ``Sigma_s (x) Sigma_a``,
        which the spec calls an ASSUMPTION, not a derived property of a
        general linear-Gaussian posterior (which also has cross-action terms
        ``Sigma_ab``). A single shared Sigma with no per-action index is not
        this: every action must get its own quadratic form through Sigma_a.
        """
        if not self.has_sigma:
            raise ValueError("variance() requires Sigma_s and Sigma_a to be set")
        z_s = np.asarray(z_s, dtype=np.float64)
        Z_a = np.asarray(Z_a, dtype=np.float64)
        s_part = float(z_s @ self.Sigma_s @ z_s)
        a_part = np.einsum("kd,de,ke->k", Z_a, self.Sigma_a, Z_a)
        return a_part * s_part

    def predict_with_sla(
        self,
        z_s: np.ndarray,
        candidates_z_a: np.ndarray,
        tier: str = "tier1",
        axiom_mask: Optional[np.ndarray] = None,
        *,
        W: Optional[np.ndarray] = None,
        c: Optional[np.ndarray] = None,
        delta: float = 0.05,
        clock: Callable[[], int] = time.perf_counter_ns,
    ) -> DecisionResult:
        """Score, mask, gate and tier-check one decision. Never raises on
        REFUSE/TIMEOUT -- only on invalid input. The tier is caller-chosen
        and never auto-selected or re-tiered here.

        The SLA check (step 7) times the *whole* decision (mask + variance +
        gate + argmax) and compares it, after the fact, to the fixed tier
        ceiling from TIERS. This is a post-hoc measurement of raw compute
        time, not a preemptive deadline: nothing is cut short to make the
        budget, and on overrun the result is downgraded to TIMEOUT with
        action_index=None regardless of what was computed above.
        """
        t0 = clock()
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}")
        contract = TIERS[tier]

        z_s = np.asarray(z_s, dtype=np.float64)
        if z_s.shape != (self.d_s,):
            raise ValueError(f"z_s must have shape ({self.d_s},)")
        _check_finite("z_s", z_s)

        Z_a = np.asarray(candidates_z_a, dtype=np.float64)
        if Z_a.ndim != 2 or Z_a.shape[1] != self.d_a or Z_a.shape[0] < 1:
            raise ValueError(f"candidates_z_a must be (K, {self.d_a}) with K >= 1")
        _check_finite("candidates_z_a", Z_a)
        K = Z_a.shape[0]

        if (W is None) != (c is None):
            raise ValueError("W and c must be given together")
        if W is not None:
            W = np.asarray(W, dtype=np.float64)
            c = np.asarray(c, dtype=np.float64)
            scores = self.score_precompiled(z_s, W, c)
        else:
            scores = self.score_bilinear(z_s, Z_a)

        mask_bool: Optional[np.ndarray] = None
        if axiom_mask is not None:
            mask_arr = np.asarray(axiom_mask)
            if mask_arr.shape != (K,):
                raise ValueError(f"axiom_mask must have shape ({K},)")
            if mask_arr.dtype == np.bool_:
                mask_bool = mask_arr
            else:
                mask_f = mask_arr.astype(np.float64)
                if not np.all(np.isin(mask_f, [0.0, 1.0])):
                    raise ValueError("axiom_mask values must be exactly 0 or 1")
                mask_bool = mask_f.astype(bool)
            # Additive -inf guard, not log(mask): avoids a log(0) warning and
            # is exactly the S + log M construction of theorem 11 since
            # log(1) == 0 and the -inf case is handled directly.
            guard = np.where(mask_bool, 0.0, -np.inf)
            guarded = scores + guard
        else:
            guarded = scores.copy()

        status: Status
        action_index: Optional[int] = None
        sigma2: Optional[np.ndarray] = None
        note = ""

        if mask_bool is not None and not mask_bool.any():
            # Theorem 11(iii): S_g is identically -inf, no W changes that.
            status = Status.REFUSE_EMPTY_SUPPORT
            note = "empty support after axiom mask"
        else:
            supp = np.arange(K) if mask_bool is None else np.flatnonzero(mask_bool)
            if supp.size == 1:
                # Theorem 11(ii): single-support output is independent of S.
                idx = int(supp[0])
                status = Status.ACCEPT
                action_index = idx
                note = "single support, variance gate skipped (theorem 11 ii)"
                if self.has_sigma:
                    sigma2 = self.variance(z_s, Z_a)
            else:
                candidate = int(supp[np.argmax(guarded[supp])])
                if self.has_sigma:
                    sigma2 = self.variance(z_s, Z_a)
                    beta = NormalDist().inv_cdf(1.0 - delta / (2.0 * K))
                    spread = beta * np.sqrt(sigma2)
                    lower = guarded - spread
                    upper = guarded + spread
                    others = supp[supp != candidate]
                    max_upper_others = float(np.max(upper[others])) if others.size else -np.inf
                    if lower[candidate] > max_upper_others:
                        status = Status.ACCEPT
                        action_index = candidate
                        note = "variance gate passed"
                    else:
                        status = Status.REFUSE_UNCERTAIN
                        note = "variance gate failed: no unambiguous ranking"
                else:
                    status = Status.ACCEPT
                    action_index = candidate
                    note = "no uncertainty gate"

        elapsed_us = (clock() - t0) / 1000.0
        within_budget = elapsed_us < contract.ceiling_us if contract.strict else elapsed_us <= contract.ceiling_us
        if not within_budget:
            status = Status.TIMEOUT
            action_index = None
            note = f"exceeded {tier} ceiling of {contract.ceiling_us}us"

        return DecisionResult(
            status=status,
            action_index=action_index,
            scores=scores,
            guarded_scores=guarded,
            sigma2=sigma2,
            elapsed_us=elapsed_us,
            tier=tier,
            ceiling_us=contract.ceiling_us,
            note=note,
        )
