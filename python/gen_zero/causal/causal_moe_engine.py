"""Multi-model causal Mixture-of-Experts over frozen teacher hidden states: pure NumPy CPU.

Each expert is one independent RNNSetAdapterRuntime (Parallel RNN think loop
with sigma_max(A) <= rho_max <= 0.95, plus a permutation-equivariant set block),
built on one teacher (Qwen3.5-9B, Qwen 27B, Gemma 26B, ...). Experts are fully
decoupled: each reads only its own native-width features (q_e, C_e), with its
own in_dim read from its npz. The only coupling is that every expert scores the
same K candidates in the same order.

"Native width" holds at the input boundary only: inside, each expert projects
D_e -> d (256 for the trained 9B adapter). Nothing is concatenated or pooled
across teachers, so no expert's geometry is compressed to fit another's.

Routing (all on CPU):

    phi(x)  = [rms((q_e - mu_e) @ P_e)  for e in experts]       P_e: (D_e, r) PCA, unsupervised
    g(x)    = softmax(W phi + b)                                cheap: O(sum_e D_e * r)
    P_e     = softmax(s_e / T_e)                                s_e = expert e's candidate scores

    dense      P(c|x) = sum_e g_e P_e(c|x)          (fusion="prob")
               P(c|x) ~ exp(sum_e g_e log P_e(c|x)) (fusion="logit", log-linear pool)
    sparse     top-k experts by g, renormalised; unselected experts are never run
    consensus  D = H(sum g_e P_e) - sum g_e H(P_e)  (gate-weighted Jensen-Shannon)
               D_n = D / log(min(n_active, K)) in [0, 1]
               D_n <= agree_tau and all argmax agree -> "consensus":
                   P ~ prod_e P_e^(n g_e)  (agreeing experts count as independent evidence,
                   so confidence rises; n identical experts give P^n renormalised)
               D_n >= conflict_tau -> "conservative": follow the expert with the highest
                   max P_e; confidence = max P_b * (1 - D_n)
               otherwise -> "mixture": the dense prob mixture

The gate never sees task names or labels: only each expert's query vector.
Every output is permutation equivariant over the K candidates, because each
expert is and the gate does not read candidates.

Threads: `n_threads` runs the active experts in parallel (BLAS releases the
GIL). `blas_threads` (default 1) caps the BLAS pool inside each inference via
threadpoolctl. The per-record matmuls are tiny, so a 24-thread OpenBLAS pool on
a loaded box spends more time waking threads than computing: measured 111 ms
min vs 7 ms min for three full-width experts at loadavg ~45.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from threadpoolctl import ThreadpoolController
except ImportError:  # only required when blas_threads is set
    ThreadpoolController = None

from .dual_manifold_head import DualManifoldHead
from .rnn_set_adapter import RNNSetAdapterRuntime
from .wasserstein_moe_router import bures_wasserstein_sq_distance

RHO_LIMIT = 0.95
_EPS = 1e-12
Inputs = Mapping[str, Tuple[np.ndarray, np.ndarray]]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _entropy(p: np.ndarray) -> np.ndarray:
    return -np.sum(p * np.log(np.clip(p, _EPS, None)), axis=-1)


def _finite(a, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: must be a nonempty array of finite values")
    return a


class RNNSetExpert:
    """One teacher's expert. Refuses any transition with rho_max above RHO_LIMIT."""

    def __init__(self, name: str, runtime: RNNSetAdapterRuntime) -> None:
        if runtime.rho_max > RHO_LIMIT + 1e-6 or runtime.sigma_max_A > RHO_LIMIT * (1 + 1e-4):
            raise ValueError(f"{name}: sigma_max(A)={runtime.sigma_max_A:.6g} exceeds {RHO_LIMIT}")
        self.name = name
        self.runtime = runtime
        self.in_dim = int(runtime.cfg["in_dim"])

    @classmethod
    def from_npz(cls, name: str, path) -> "RNNSetExpert":
        return cls(name, RNNSetAdapterRuntime.from_npz(path))

    def score(self, q: np.ndarray, C: np.ndarray) -> np.ndarray:
        return self.runtime.score(q, C)


class CausalMoERouter:
    """Linear softmax gate over per-expert PCA projections of the query vectors."""

    def __init__(self, names: Sequence[str], mus: Sequence[np.ndarray], projs: Sequence[np.ndarray],
                 W: np.ndarray, b: np.ndarray, temperatures: Optional[np.ndarray] = None,
                 agree_tau: float = 0.15, conflict_tau: float = 0.45) -> None:
        E = len(names)
        if E < 1 or len(set(names)) != E or len(mus) != E or len(projs) != E:
            raise ValueError("router: need one unique name, mu and projection per expert")
        self.names = list(names)
        self.mus = [_finite(m, f"mu[{n}]") if np.size(m) else np.zeros(0, np.float32)
                    for n, m in zip(names, mus)]
        self.projs = [np.asarray(p, dtype=np.float32) for p in projs]
        ranks = {p.shape[1] for p in self.projs}
        if len(ranks) != 1 or any(p.ndim != 2 for p in self.projs):
            raise ValueError("router: every projection must be 2-D with the same rank")
        self.rank = ranks.pop()
        self.W = np.asarray(W, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        if self.W.shape != (E, E * self.rank) or self.b.shape != (E,):
            raise ValueError(f"router: W must be {(E, E * self.rank)} and b {(E,)}")
        self.temperatures = (np.ones(E) if temperatures is None
                             else np.asarray(temperatures, dtype=np.float64))
        if self.temperatures.shape != (E,) or np.any(self.temperatures <= 0):
            raise ValueError("router: temperatures must be positive, one per expert")
        if not 0.0 <= agree_tau <= conflict_tau <= 1.0:
            raise ValueError("router: need 0 <= agree_tau <= conflict_tau <= 1")
        self.agree_tau, self.conflict_tau = float(agree_tau), float(conflict_tau)

    @classmethod
    def uniform(cls, names: Sequence[str]) -> "CausalMoERouter":
        E = len(names)
        return cls(names, [np.zeros(0)] * E, [np.zeros((0, 0))] * E, np.zeros((E, 0)), np.zeros(E))

    @classmethod
    def fit_projections(cls, names: Sequence[str], q_train: Mapping[str, np.ndarray],
                        rank: int = 16) -> "CausalMoERouter":
        """PCA of each expert's own train queries. Unsupervised: labels are not read."""
        mus, projs = [], []
        for n in names:
            X = _finite(q_train[n], f"q_train[{n}]").astype(np.float64)
            mu = X.mean(axis=0)
            _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
            if Vt.shape[0] < rank:
                raise ValueError(f"{n}: need at least {rank} train queries for rank-{rank} PCA")
            mus.append(mu.astype(np.float32))
            projs.append(Vt[:rank].T.astype(np.float32))
        E = len(names)
        return cls(names, mus, projs, np.zeros((E, E * rank)), np.zeros(E))

    def features(self, qs: Mapping[str, np.ndarray]) -> np.ndarray:
        if self.rank == 0:
            return np.zeros(0)
        parts = []
        for n, mu, P in zip(self.names, self.mus, self.projs):
            q = _finite(qs[n], f"q[{n}]")
            if q.shape != (P.shape[0],):
                raise ValueError(f"q[{n}]: expected shape ({P.shape[0]},), got {q.shape}")
            z = ((q - mu) @ P).astype(np.float64)
            parts.append(z / np.sqrt(np.mean(z * z) + 1e-6))
        return np.concatenate(parts)

    def gates_from_features(self, phi: np.ndarray) -> np.ndarray:
        return _softmax(self.W @ phi + self.b)

    def gates(self, qs: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.gates_from_features(self.features(qs))

    def fit(self, phis: np.ndarray, expert_probs: np.ndarray, *, l2: float = 1e-3,
            steps: int = 400, lr: float = 0.05) -> List[float]:
        """Minimise mean -log sum_e g_e P_e(y) with Adam; returns the loss history.

        phis: (N, E*r). expert_probs: (N, E) = P_e(y_i | x_i), each expert's
        probability of the true candidate (already temperature-scaled). The
        gradient w.r.t. the gate logits is g - rho, rho_e = g_e P_e(y) / sum_j g_j P_j(y).
        """
        phis = np.asarray(phis, dtype=np.float64)
        P = np.clip(np.asarray(expert_probs, dtype=np.float64), _EPS, None)
        N, E = P.shape
        if phis.shape != (N, E * self.rank) or E != len(self.names):
            raise ValueError("fit: shape mismatch between features, probs and router")
        params = [self.W, self.b]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        hist = []
        for t in range(1, steps + 1):
            g = _softmax(phis @ self.W.T + self.b)
            mix = np.sum(g * P, axis=1)
            hist.append(float(-np.mean(np.log(mix)) + 0.5 * l2 * np.sum(self.W ** 2)))
            dz = (g - g * P / mix[:, None]) / N
            grads = [dz.T @ phis + l2 * self.W, dz.sum(axis=0)]
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                p -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
        return hist

    def save_npz(self, path) -> None:
        arrays = {"W": self.W, "b": self.b, "temperatures": self.temperatures,
                  "agree_tau": np.float64(self.agree_tau), "conflict_tau": np.float64(self.conflict_tau),
                  "names_json": np.array(json.dumps(self.names))}
        for i, (mu, P) in enumerate(zip(self.mus, self.projs)):
            arrays[f"mu_{i}"], arrays[f"proj_{i}"] = mu, P
        np.savez(path, **arrays)

    @classmethod
    def from_npz(cls, path) -> "CausalMoERouter":
        with np.load(path, allow_pickle=False) as z:
            names = json.loads(str(z["names_json"]))
            return cls(names, [z[f"mu_{i}"] for i in range(len(names))],
                       [z[f"proj_{i}"] for i in range(len(names))], z["W"], z["b"],
                       z["temperatures"], float(z["agree_tau"]), float(z["conflict_tau"]))


class DirichletWassersteinRouter:
    """Unsupervised Gaussian-attractor gate (Spec 12 section 2.2 + section 4.4).

    Each expert fits its own Gaussian attractor ``(m_e, C_e)`` on its OWN
    train queries, in its OWN rank-``r`` PCA space (the same PCA as
    ``CausalMoERouter.fit_projections``: ``mu_e`` (raw-space mean) and
    ``P_e`` (D_e, r)), or, if a `DualManifoldHead` state hook is given, in
    that head's state manifold. Routing never reads task names, labels or
    candidates: `gates(qs)` takes only ``{expert_name: q_vector}``.

    Distance: for a query the projected point is a "point mass" -- a
    Gaussian with zero covariance. Its squared Bures-Wasserstein distance to
    the attractor ``N(m_e, C_e)`` is the closed-form
    ``||z_e - m_e||^2 + tr(C_e)`` (the point-mass specialisation of the
    general formula, see `bures_wasserstein_sq_distance`'s docstring: it
    already degenerates to exactly this when one side's covariance is zero,
    so calling it directly with ``cov1 = 0`` needs no special case and no
    reimplementation of the metric). ``metric="mahalanobis"`` instead
    whitens by ``C_e`` and uses the plain squared Mahalanobis distance.

    Concentrations ``alpha_e = alpha0 * exp(-d2_e / tau_e)``, where
    ``tau_e`` defaults (at `fit`) to the median in-sample distance of
    expert e's own train queries to its own attractor -- a data-derived
    scale, not a hand-set one. Gates are the Dirichlet mean
    ``g = alpha / sum(alpha)``.
    """

    def __init__(self, names: Sequence[str], mus: Sequence[np.ndarray], projs: Sequence[np.ndarray],
                 means: Sequence[np.ndarray], covs: Sequence[np.ndarray], tau: Sequence[float], *,
                 alpha0: float = 1.0, metric: str = "bures", temperatures: Optional[np.ndarray] = None,
                 agree_tau: float = 0.15, conflict_tau: float = 0.45, cov_ridge: float = 1e-4) -> None:
        E = len(names)
        if E < 1 or len(set(names)) != E:
            raise ValueError("router: need one unique name per expert")
        if len(mus) != E or len(projs) != E or len(means) != E or len(covs) != E:
            raise ValueError("router: need one mu, proj, mean and cov per expert")
        if metric not in ("bures", "mahalanobis"):
            raise ValueError(f"router: unknown metric {metric!r}, want 'bures' or 'mahalanobis'")
        if not np.isfinite(alpha0) or alpha0 <= 0:
            raise ValueError("router: alpha0 must be positive and finite")
        self.names = list(names)
        self.mus = [_finite(m, f"mu[{n}]") for n, m in zip(names, mus)]
        self.projs = [np.asarray(p, dtype=np.float32) for p in projs]
        ranks = {p.shape[1] for p in self.projs}
        if len(ranks) != 1 or any(p.ndim != 2 for p in self.projs):
            raise ValueError("router: every projection must be 2-D with the same rank")
        self.rank = ranks.pop()
        self.means = [_finite(m, f"mean[{n}]") for n, m in zip(names, means)]
        self.covs = []
        for n, C in zip(names, covs):
            C = np.asarray(C, dtype=np.float64)
            if C.ndim != 2 or C.shape[0] != C.shape[1] or C.shape[0] != self.rank:
                raise ValueError(f"cov[{n}]: must be ({self.rank}, {self.rank})")
            if not np.all(np.isfinite(C)):
                raise ValueError(f"cov[{n}]: must be finite")
            self.covs.append(C)
        self.tau = np.asarray(tau, dtype=np.float64)
        if self.tau.shape != (E,) or not np.all(np.isfinite(self.tau)) or np.any(self.tau <= 0):
            raise ValueError("router: tau must be positive and finite, one per expert")
        self.alpha0 = float(alpha0)
        self.metric = metric
        self.temperatures = (np.ones(E) if temperatures is None
                             else np.asarray(temperatures, dtype=np.float64))
        if self.temperatures.shape != (E,) or np.any(self.temperatures <= 0):
            raise ValueError("router: temperatures must be positive, one per expert")
        if not 0.0 <= agree_tau <= conflict_tau <= 1.0:
            raise ValueError("router: need 0 <= agree_tau <= conflict_tau <= 1")
        self.agree_tau, self.conflict_tau = float(agree_tau), float(conflict_tau)
        self.cov_ridge = float(cov_ridge)

    @classmethod
    def fit(cls, names: Sequence[str], q_train: Mapping[str, np.ndarray], rank: int = 16,
            alpha0: float = 1.0, cov_ridge: float = 1e-4, *, metric: str = "bures",
            temperatures: Optional[np.ndarray] = None, agree_tau: float = 0.15, conflict_tau: float = 0.45,
            state_head: Optional[DualManifoldHead] = None) -> "DirichletWassersteinRouter":
        """Fit one unsupervised Gaussian attractor per expert. Labels are never read.

        If `state_head` is given, the attractor for every expert is fitted
        on `state_head.project_state(q)` instead of PCA (this requires
        `state_head.D_s == expert in_dim` for every expert, and the
        projection used at routing time is then `state_head.P_s`, so the
        MultiModelCausalMoE shape check against each expert's in_dim still
        applies unchanged).
        """
        if metric not in ("bures", "mahalanobis"):
            raise ValueError(f"router: unknown metric {metric!r}, want 'bures' or 'mahalanobis'")
        mus, projs, means, covs, taus = [], [], [], [], []
        for n in names:
            X = _finite(q_train[n], f"q_train[{n}]").astype(np.float64)
            if X.ndim != 2:
                raise ValueError(f"q_train[{n}]: must be 2-D (N, D)")
            N = X.shape[0]
            if state_head is not None:
                if state_head.D_s != X.shape[1]:
                    raise ValueError(f"{n}: state_head.D_s={state_head.D_s} != q width {X.shape[1]}")
                r = state_head.d_s
                mu_raw = state_head.mu_s.astype(np.float64)
                P = state_head.P_s.astype(np.float32)
            else:
                r = rank
                mu_raw = X.mean(axis=0)
                _, _, Vt = np.linalg.svd(X - mu_raw, full_matrices=False)
                if Vt.shape[0] < r:
                    raise ValueError(f"{n}: need at least {r} train queries for rank-{r} PCA")
                P = Vt[:r].T.astype(np.float32)
            if N < r + 2:
                raise ValueError(f"{n}: need at least {r + 2} train queries for a rank-{r} attractor")
            Z = (X - mu_raw) @ P.astype(np.float64)
            m = Z.mean(axis=0)
            Zc = Z - m
            C = (Zc.T @ Zc) / max(N - 1, 1) + cov_ridge * np.eye(r)
            eigvals = np.linalg.eigvalsh(0.5 * (C + C.T))
            if np.any(eigvals <= 0):
                raise ValueError(f"{n}: covariance is not positive-definite after ridge {cov_ridge}")
            zero_cov = np.zeros((r, r))
            if metric == "bures":
                d2_train = np.array([bures_wasserstein_sq_distance(z, zero_cov, m, C) for z in Z])
            else:
                d2_train = np.array([float((z - m) @ np.linalg.solve(C, z - m)) for z in Z])
            tau_e = max(float(np.median(d2_train)), 1e-9)
            mus.append(mu_raw.astype(np.float32))
            projs.append(P)
            means.append(m)
            covs.append(C)
            taus.append(tau_e)
        return cls(names, mus, projs, means, covs, taus, alpha0=alpha0, metric=metric,
                   temperatures=temperatures, agree_tau=agree_tau, conflict_tau=conflict_tau,
                   cov_ridge=cov_ridge)

    def features(self, qs: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(self._z_per_expert(qs))

    def _z_per_expert(self, qs: Mapping[str, np.ndarray]) -> List[np.ndarray]:
        zs = []
        for n, mu, P in zip(self.names, self.mus, self.projs):
            q = _finite(qs[n], f"q[{n}]")
            if q.shape != (P.shape[0],):
                raise ValueError(f"q[{n}]: expected shape ({P.shape[0]},), got {q.shape}")
            z = (q.astype(np.float64) - mu.astype(np.float64)) @ P.astype(np.float64)
            zs.append(z)
        return zs

    def _distances(self, qs: Mapping[str, np.ndarray]) -> np.ndarray:
        zs = self._z_per_expert(qs)
        zero_cov = np.zeros((self.rank, self.rank))
        d2 = np.empty(len(self.names))
        for i, (z, m, C) in enumerate(zip(zs, self.means, self.covs)):
            if self.metric == "bures":
                d2[i] = bures_wasserstein_sq_distance(z, zero_cov, m, C)
            else:
                diff = z - m
                d2[i] = float(diff @ np.linalg.solve(C, diff))
        return d2

    def concentration(self, qs: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, float]:
        """Dirichlet concentrations ``alpha_e`` and their sum."""
        d2 = self._distances(qs)
        alpha = self.alpha0 * np.exp(-d2 / self.tau)
        return alpha, float(alpha.sum())

    def gates(self, qs: Mapping[str, np.ndarray]) -> np.ndarray:
        alpha, total = self.concentration(qs)
        return alpha / total

    def routing_uncertainty(self, qs: Mapping[str, np.ndarray]) -> float:
        """``1 / (1 + sum(alpha))``: high total concentration (query close to

        at least one attractor) -> low uncertainty; a query far from every
        attractor drives every alpha_e -> 0, sum(alpha) -> 0, uncertainty -> 1.
        (Chosen over the Dirichlet's expected entropy for simplicity; both
        are monotonic decreasing in sum(alpha).)
        """
        _, total = self.concentration(qs)
        return 1.0 / (1.0 + total)

    def set_temperature_scale(self, s: float) -> None:
        """Multiply every expert's ``tau`` by ``s`` (widens/narrows every attractor's reach)."""
        if not np.isfinite(s) or s <= 0:
            raise ValueError("router: temperature scale must be positive and finite")
        self.tau = self.tau * s

    def save_npz(self, path) -> None:
        arrays = {
            "names_json": np.array(json.dumps(self.names)),
            "tau": self.tau,
            "alpha0": np.float64(self.alpha0),
            "metric_json": np.array(json.dumps(self.metric)),
            "temperatures": self.temperatures,
            "agree_tau": np.float64(self.agree_tau),
            "conflict_tau": np.float64(self.conflict_tau),
            "cov_ridge": np.float64(self.cov_ridge),
        }
        for i, (mu, P, m, C) in enumerate(zip(self.mus, self.projs, self.means, self.covs)):
            arrays[f"mu_{i}"], arrays[f"proj_{i}"] = mu, P
            arrays[f"mean_{i}"], arrays[f"cov_{i}"] = m, C
        np.savez(path, **arrays)

    @classmethod
    def from_npz(cls, path) -> "DirichletWassersteinRouter":
        with np.load(path, allow_pickle=False) as z:
            names = json.loads(str(z["names_json"]))
            E = len(names)
            return cls(names, [z[f"mu_{i}"] for i in range(E)], [z[f"proj_{i}"] for i in range(E)],
                       [z[f"mean_{i}"] for i in range(E)], [z[f"cov_{i}"] for i in range(E)], z["tau"],
                       alpha0=float(z["alpha0"]), metric=json.loads(str(z["metric_json"])),
                       temperatures=z["temperatures"], agree_tau=float(z["agree_tau"]),
                       conflict_tau=float(z["conflict_tau"]), cov_ridge=float(z["cov_ridge"]))


def calibrate_temperature(scores: Sequence[np.ndarray], labels: Sequence[int]) -> float:
    """1-D NLL minimisation of softmax(s / T) over a log grid, then a local refinement."""
    def nll(T: float) -> float:
        return -float(np.mean([np.log(max(_softmax(s / T)[y], _EPS)) for s, y in zip(scores, labels)]))
    grid = np.exp(np.linspace(np.log(0.02), np.log(50.0), 61))
    best = grid[int(np.argmin([nll(T) for T in grid]))]
    fine = best * np.exp(np.linspace(-0.12, 0.12, 25))
    return float(fine[int(np.argmin([nll(T) for T in fine]))])


def disagreement(probs: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    """Gate-weighted Jensen-Shannon divergence D and D / log(min(n, K)) in [0, 1]."""
    mix = weights @ probs
    D = float(max(_entropy(mix) - weights @ _entropy(probs), 0.0))
    denom = np.log(min(probs.shape[0], probs.shape[1]))
    return D, (min(D / denom, 1.0) if denom > 0 else 0.0)


@dataclass
class MoEResult:
    probs: np.ndarray                 # (K,) fused distribution
    choice: int
    confidence: float
    decision: str                     # single | mixture | logit_pool | consensus | conservative
    gates: np.ndarray                 # (E,) full gate distribution
    active: List[str]                 # experts actually run
    active_weights: np.ndarray        # gates renormalised over active experts
    expert_scores: Dict[str, np.ndarray] = field(default_factory=dict)
    expert_probs: Dict[str, np.ndarray] = field(default_factory=dict)
    jsd: float = 0.0
    jsd_norm: float = 0.0
    vote_agreement: float = 1.0       # gate mass on experts whose argmax equals `choice`
    routing_uncertainty: Optional[float] = None  # router.routing_uncertainty(qs) if the router exposes it


class MultiModelCausalMoE:
    def __init__(self, experts: Sequence[RNNSetExpert], router: Optional[CausalMoERouter] = None,
                 n_threads: int = 1, blas_threads: Optional[int] = 1) -> None:
        self.experts = {e.name: e for e in experts}
        if len(self.experts) != len(experts) or not experts:
            raise ValueError("need at least one expert, with unique names")
        self.names = [e.name for e in experts]
        self.router = router or CausalMoERouter.uniform(self.names)
        if self.router.names != self.names:
            raise ValueError(f"router names {self.router.names} != experts {self.names}")
        for n, P in zip(self.names, self.router.projs):
            if self.router.rank and P.shape[0] != self.experts[n].in_dim:
                raise ValueError(f"router projection for {n} expects {P.shape[0]}-D, expert is {self.experts[n].in_dim}-D")
        self.n_threads = int(n_threads)
        self._pool = ThreadPoolExecutor(self.n_threads) if self.n_threads > 1 else None
        self.blas_threads = blas_threads
        self._blas = None
        if blas_threads is not None:
            if ThreadpoolController is None:
                raise ImportError("blas_threads needs threadpoolctl; install it or pass blas_threads=None")
            self._blas = ThreadpoolController()

    @classmethod
    def from_npz(cls, expert_paths: Mapping[str, str], router_path=None,
                 n_threads: int = 1, blas_threads: Optional[int] = 1) -> "MultiModelCausalMoE":
        experts = [RNNSetExpert.from_npz(n, p) for n, p in expert_paths.items()]
        router = CausalMoERouter.from_npz(router_path) if router_path else None
        return cls(experts, router, n_threads, blas_threads)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def _check(self, inputs: Inputs, names: Sequence[str]) -> int:
        Ks = set()
        for n in names:
            if n not in inputs:
                raise ValueError(f"missing input for expert {n}")
            C = np.asarray(inputs[n][1])
            if C.ndim != 2:
                raise ValueError(f"C[{n}] must be 2-D (K, D)")
            Ks.add(C.shape[0])
        if len(Ks) != 1:
            raise ValueError(f"experts disagree on the number of candidates: {sorted(Ks)}")
        return Ks.pop()

    def expert_scores(self, inputs: Inputs, names: Sequence[str]) -> Dict[str, np.ndarray]:
        run = lambda n: self.experts[n].score(*inputs[n])
        if self._pool is None or len(names) == 1:
            return {n: run(n) for n in names}
        return dict(zip(names, self._pool.map(run, names)))

    def infer(self, inputs: Inputs, strategy: str = "dense", *, top_k: int = 2,
              fusion: str = "prob") -> MoEResult:
        if self._blas is None:
            return self._infer(inputs, strategy, top_k, fusion)
        with self._blas.limit(limits=self.blas_threads, user_api="blas"):
            return self._infer(inputs, strategy, top_k, fusion)

    def _infer(self, inputs: Inputs, strategy: str, top_k: int, fusion: str) -> MoEResult:
        if strategy not in ("dense", "sparse", "consensus") or fusion not in ("prob", "logit"):
            raise ValueError(f"unknown strategy/fusion {strategy}/{fusion}")
        E = len(self.names)
        K = self._check(inputs, self.names)
        qs = {n: inputs[n][0] for n in self.names}
        gates = self.router.gates(qs)
        ru_fn = getattr(self.router, "routing_uncertainty", None)
        routing_uncertainty = float(ru_fn(qs)) if ru_fn is not None else None
        if strategy == "sparse":
            k = max(1, min(int(top_k), E))
            idx = np.sort(np.argsort(-gates, kind="stable")[:k])
        else:
            idx = np.arange(E)
        active = [self.names[i] for i in idx]
        w = gates[idx] / gates[idx].sum()
        scores = self.expert_scores(inputs, active)
        T = self.router.temperatures[idx]
        P = np.stack([_softmax(scores[n].astype(np.float64) / t) for n, t in zip(active, T)])
        jsd, jsd_n = disagreement(P, w) if K > 1 else (0.0, 0.0)
        mix = w @ P
        tops = P.argmax(axis=1)

        decision = "mixture"
        if len(active) == 1:
            fused, decision = P[0], "single"
        elif strategy == "consensus" and jsd_n <= self.router.agree_tau and np.all(tops == tops[0]):
            fused, decision = _softmax((len(active) * w) @ np.log(np.clip(P, _EPS, None))), "consensus"
        elif strategy == "consensus" and jsd_n >= self.router.conflict_tau:
            b = int(np.argmax(P.max(axis=1)))
            fused, decision = P[b], "conservative"
        elif fusion == "logit":
            fused, decision = _softmax(w @ np.log(np.clip(P, _EPS, None))), "logit_pool"
        else:
            fused = mix
        choice = int(np.argmax(fused))
        conf = float(fused[choice]) * ((1.0 - jsd_n) if decision == "conservative" else 1.0)
        return MoEResult(probs=fused.astype(np.float32), choice=choice, confidence=conf,
                         decision=decision, gates=gates, active=active, active_weights=w,
                         expert_scores=scores,
                         expert_probs={n: p.astype(np.float32) for n, p in zip(active, P)},
                         jsd=jsd, jsd_norm=jsd_n, vote_agreement=float(w[tops == choice].sum()),
                         routing_uncertainty=routing_uncertainty)
