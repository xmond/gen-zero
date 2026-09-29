"""Causal MCTS over a Parallel RNN + Set-Transformer prior (RFC-072 section 3-4).

Pieces, all pure NumPy on the CPU:

    LieLatentDynamics   z' = exp(dt * Omega_a) z, Omega_a skew (so(d)). The
                        generators are FIT from observed (z, a, z') transitions:
                        Kabsch gives the best rotation R_a in SO(d), a real
                        Schur logarithm gives Omega_a, and expm(Omega_a) must
                        reproduce R_a (fail closed). Norm is preserved exactly.
    ContractionDamper   A = a_scale*(diag(sig(lam)) + U_A V_A^T), sigma_max(A)
                        certified <= rho_max (<= 0.95) by SVD, the same clamp as
                        RNNSetAdapterRuntime. Off-manifold drift e = z - P z is
                        replaced by A^k e every `damper_every` rollout steps.
    CausalMCTSRNN       Prior = Parallel RNN think loop (with ACT halting) +
                        Set-Transformer block of a trained RNNSetAdapterRuntime.
                        Fast path returns the prior. Escalation (high entropy,
                        a probe rollout that disagrees with the prior, or high
                        counterfactual sensitivity) runs PUCT over the learned
                        latent dynamics with the prior pruned to top-k.

Honest limits, stated here so no caller has to discover them:

* A rotation is invertible. It cannot map two states onto one, so it can only
  model an environment whose per-action transitions are permutations of the
  latent codes. Irreversibility has to come from terminal readouts (death ends
  the episode), not from the dynamics.
* The damper has no semantic content. It only shrinks the component of z that
  left the learned manifold span.
* ACT halts on the residual ||h_t - h_{t-1}||. With a fixed linear contraction
  that residual depends on the input norm and on A, not on task difficulty.
* Counterfactual sensitivity perturbs the input orthogonally to span(q, C). It
  measures how much the decision rests on directions outside that span; it is
  not a proof of causal structure.
"""
from __future__ import annotations

import math
import time
import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import expm, schur

from .dual_manifold_head import TIERS, DualManifoldHead
from .rnn_set_adapter import RNNSetAdapterRuntime, _softmax

__all__ = [
    "LieLatentDynamics",
    "ContractionDamper",
    "LatentReadout",
    "DualManifoldPrior",
    "CausalMCTSRNN",
    "Decision",
    "certified_contraction",
]


def _finite(value, name: str, dtype=np.float64) -> np.ndarray:
    a = np.asarray(value, dtype=dtype)
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: must be a nonempty array of finite values")
    return a


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


# --------------------------------------------------------------------------- dynamics

def _kabsch(Z: np.ndarray, Zn: np.ndarray) -> np.ndarray:
    """argmin_{R in SO(d)} ||Z R^T - Zn||_F."""
    U, _, Vt = np.linalg.svd(Zn.T @ Z)
    s = np.ones(U.shape[0])
    s[-1] = np.sign(np.linalg.det(U @ Vt)) or 1.0
    return (U * s) @ Vt


def _real_log_so(R: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Real skew logarithm of R in SO(d) through the real Schur form.

    R orthogonal is normal, so its real Schur form is block diagonal: 2x2
    rotation blocks, +1 and -1 on the diagonal. det(R)=+1 makes the -1 count
    even; each -1 pair becomes a rotation by pi in their plane.
    """
    T, Q = schur(R, output="real")
    d = R.shape[0]
    L = np.zeros_like(R)
    minus: List[int] = []
    i = 0
    while i < d:
        if i + 1 < d and abs(T[i + 1, i]) > tol:
            c, s = T[i, i], T[i + 1, i]
            theta = math.atan2(s, c)
            scale = theta / s
            L[i, i + 1] = scale * T[i, i + 1]
            L[i + 1, i] = scale * T[i + 1, i]
            i += 2
            continue
        if T[i, i] < 0.0:
            minus.append(i)
        i += 1
    if len(minus) % 2:
        raise ValueError("rotation has an odd number of -1 eigenvalues (det != +1)")
    for a, b in zip(minus[0::2], minus[1::2]):
        L[a, b], L[b, a] = -math.pi, math.pi
    omega = Q @ L @ Q.T
    return 0.5 * (omega - omega.T)


class LieLatentDynamics:
    """Per-action latent step z' = exp(dt*Omega_a) z with Omega_a in so(d)."""

    def __init__(self, generators: np.ndarray, dt: float = 1.0, *, fit_report: Optional[dict] = None) -> None:
        G = _finite(generators, "generators")
        if G.ndim != 3 or G.shape[1] != G.shape[2]:
            raise ValueError("generators must have shape (A, d, d)")
        if not np.allclose(G, -np.transpose(G, (0, 2, 1)), atol=1e-9):
            raise ValueError("generators must be skew-symmetric")
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError("dt must be positive")
        self.generators = G
        self.dt = float(dt)
        R64 = np.stack([expm(self.dt * g) for g in G])
        err = float(np.max(np.abs(np.einsum("aij,akj->aik", R64, R64) - np.eye(G.shape[1]))))
        if err > 1e-6:
            raise ValueError(f"exp(dt*Omega) is not orthogonal (max err {err:.3g})")
        self.R64 = R64
        self.R = R64.astype(np.float32)
        self.orthogonality_error = err
        self.fit_report = dict(fit_report or {})

    @property
    def n_actions(self) -> int:
        return self.R.shape[0]

    @property
    def dim(self) -> int:
        return self.R.shape[1]

    @classmethod
    def fit(cls, Z: np.ndarray, actions: np.ndarray, Z_next: np.ndarray, n_actions: int,
            dt: float = 1.0) -> "LieLatentDynamics":
        """Kabsch rotation per action, then its real Lie logarithm."""
        Z = _finite(Z, "Z")
        Zn = _finite(Z_next, "Z_next")
        acts = np.asarray(actions)
        if Z.ndim != 2 or Z.shape != Zn.shape or acts.shape != (Z.shape[0],):
            raise ValueError("Z, Z_next must be (n, d) and actions (n,)")
        if not np.issubdtype(acts.dtype, np.integer) or acts.min() < 0 or acts.max() >= n_actions:
            raise ValueError("actions must be integers in [0, n_actions)")
        gens, resid, counts = [], [], []
        for a in range(n_actions):
            m = acts == a
            if not m.any():
                raise ValueError(f"no transitions observed for action {a}")
            R = _kabsch(Z[m], Zn[m])
            omega = _real_log_so(R)
            back = expm(omega)
            if float(np.max(np.abs(back - R))) > 1e-6:
                raise ValueError(f"action {a}: expm(log R) does not reproduce R")
            gens.append(omega / dt)
            resid.append(float(np.sqrt(np.mean(np.sum((Z[m] @ R.T - Zn[m]) ** 2, axis=1)))))
            counts.append(int(m.sum()))
        report = {"transitions_per_action": counts, "fit_rmse_per_action": resid}
        return cls(np.stack(gens), dt, fit_report=report)

    def step(self, z: np.ndarray, action: int) -> np.ndarray:
        return self.R[action] @ z

    def step_all(self, z: np.ndarray) -> np.ndarray:
        """(d,) -> (A, d): every action from one state in one matmul."""
        return self.R @ z

    def save_npz(self, path) -> None:
        np.savez(path, generators=self.generators, dt=np.float64(self.dt))

    @classmethod
    def from_npz(cls, path) -> "LieLatentDynamics":
        with np.load(path, allow_pickle=False) as z:
            return cls(z["generators"], float(z["dt"]))


def certified_contraction(lam: np.ndarray, U_A: np.ndarray, V_A: np.ndarray,
                          rho_max: float) -> Tuple[float, float]:
    """(a_scale, sigma_max(A)) with the RNNSetAdapterRuntime clamp, via float64 SVD."""
    dvec = 1.0 / (1.0 + np.exp(-np.asarray(lam, np.float64)))
    raw = np.diag(dvec) + np.asarray(U_A, np.float64) @ np.asarray(V_A, np.float64).T
    sigma_raw = float(np.linalg.svd(raw, compute_uv=False)[0])
    a_scale = min(1.0, rho_max / sigma_raw) if sigma_raw > 0.0 else 1.0
    return a_scale, a_scale * sigma_raw


class ContractionDamper:
    """Shrink off-manifold drift with a certified Lyapunov contraction.

    basis (d, m) spans the learned latent manifold (orthonormal columns).
    apply(z, k): z <- P z + A^k (z - P z). sigma_max(A) <= rho_max < 1, so the
    off-manifold part decays at least as rho_max**k; the on-manifold part is
    untouched, so a pure rotation of a manifold point keeps its norm.
    """

    def __init__(self, lam: np.ndarray, U_A: np.ndarray, V_A: np.ndarray, basis: np.ndarray,
                 rho_max: float = 0.95) -> None:
        if not (0.0 < rho_max <= 0.95):
            raise ValueError("rho_max must be in (0, 0.95]")
        self.lam = _finite(lam, "lam")
        self.U_A = _finite(U_A, "U_A")
        self.V_A = _finite(V_A, "V_A")
        B = _finite(basis, "basis")
        d = self.lam.shape[0]
        if self.U_A.shape[0] != d or self.V_A.shape != self.U_A.shape or B.shape[0] != d:
            raise ValueError("damper factor shapes disagree with lam")
        if not np.allclose(B.T @ B, np.eye(B.shape[1]), atol=1e-6):
            raise ValueError("basis columns must be orthonormal")
        self.a_scale, self.sigma_max_A = certified_contraction(self.lam, self.U_A, self.V_A, rho_max)
        if not self.sigma_max_A <= rho_max + 1e-9:
            raise ValueError(f"damper not contractive: sigma_max(A) = {self.sigma_max_A:.6g}")
        self.rho_max = float(rho_max)
        self.dvec = (1.0 / (1.0 + np.exp(-self.lam))).astype(np.float32)
        self.Ua = self.U_A.astype(np.float32)
        self.Va = self.V_A.astype(np.float32)
        self.basis = B.astype(np.float32)

    @classmethod
    def random(cls, basis: np.ndarray, rank: int = 4, rho_max: float = 0.95,
               seed: int = 0) -> "ContractionDamper":
        """Untrained factors: the damper only has to contract, not to know anything."""
        d = np.asarray(basis).shape[0]
        rng = np.random.default_rng(seed)
        s = 1.0 / math.sqrt(d)
        return cls(rng.standard_normal(d), rng.standard_normal((d, rank)) * s,
                   rng.standard_normal((d, rank)) * s, basis, rho_max)

    def _A(self, e: np.ndarray) -> np.ndarray:
        return np.float32(self.a_scale) * (self.dvec * e + self.Ua @ (self.Va.T @ e))

    def apply(self, z: np.ndarray, k: int = 1) -> np.ndarray:
        on = self.basis @ (self.basis.T @ z)
        e = z - on
        for _ in range(k):
            e = self._A(e)
        return on + e

    def off_manifold_norm(self, z: np.ndarray) -> float:
        return float(np.linalg.norm(z - self.basis @ (self.basis.T @ z)))


# --------------------------------------------------------------------------- readouts

@dataclass
class LatentReadout:
    """Linear readouts on the latent state (all rows are functionals of z).

    features  (D, d): prior input, q = F z, candidate k = F R_k z
    fail      (d,)  : > 0.5 means the state is a failure terminal
    success   (d,)  : > 0.5 means the state is a success terminal
    potential (d,)  : shaping value in [-1, 0] at rollout ends
    """
    features: np.ndarray
    fail: np.ndarray
    success: np.ndarray
    potential: np.ndarray

    def __post_init__(self) -> None:
        self.features = _finite(self.features, "features", np.float32)
        self.fail = _finite(self.fail, "fail", np.float32)
        self.success = _finite(self.success, "success", np.float32)
        self.potential = _finite(self.potential, "potential", np.float32)
        d = self.features.shape[1]
        for n in ("fail", "success", "potential"):
            if getattr(self, n).shape != (d,):
                raise ValueError(f"{n} must have shape ({d},)")
        # One matmul evaluates all three scalar readouts for a batch of states.
        self._scalars = np.stack([self.fail, self.success, self.potential])


# --------------------------------------------------------------------------- dual-manifold prior

class DualManifoldPrior:
    """MCTS prior driven by a precompiled DualManifoldHead score (Spec 23 theorem 1).

    The candidate action set is fixed for the life of this object: `X_a` is
    projected and precompiled (W, c = Z_a @ M^T, b*1) exactly once in the
    constructor, so `probs(z)` is a single cached GEMV + softmax. This never
    calls `predict_with_sla` (no tier/axiom-mask/variance-gate machinery
    inside the search loop -- that machinery is for a top-level decision,
    not a per-node MCTS expansion).
    """

    def __init__(self, head: DualManifoldHead, X_a: np.ndarray, temperature: float = 1.0) -> None:
        if not isinstance(head, DualManifoldHead):
            raise TypeError("head must be a DualManifoldHead")
        if not (math.isfinite(temperature) and temperature > 0.0):
            raise ValueError("temperature must be positive")
        X_a = _finite(X_a, "X_a", np.float64)
        if X_a.ndim != 2:
            raise ValueError("X_a must be 2D (A, D_a)")
        self.head = head
        self.T = float(temperature)
        self.Z_a = head.project_action(X_a)          # (A, d_a)
        self.W, self.c = head.precompile(self.Z_a)   # cached once (theorem 1)

    @property
    def n_actions(self) -> int:
        return self.Z_a.shape[0]

    @property
    def D_s(self) -> int:
        return self.head.D_s

    def probs(self, z: np.ndarray) -> np.ndarray:
        """(d,) latent state -> (A,) softmax over the precompiled scores, float64."""
        z_s = self.head.project_state(np.asarray(z, dtype=np.float64))
        scores = self.head.score_precompiled(z_s, self.W, self.c).astype(np.float64)
        return _softmax(scores / self.T)


# --------------------------------------------------------------------------- planner

@dataclass
class Decision:
    action: int
    probs: np.ndarray
    prior: np.ndarray
    mode: str                      # "fast" | "mcts" | "rollout"
    triggers: Dict[str, bool]
    entropy_norm: float
    cf_sensitivity: float
    probe_return: float
    act_steps: int
    n_simulations: int
    tree_nodes: int
    latency_ms: float
    root_q: Optional[np.ndarray] = None
    raw_policy: Optional[np.ndarray] = None   # visit (or prior) policy before the cf adjustment
    extras: Dict[str, float] = field(default_factory=dict)
    truncated: bool = False
    truncation_reason: str = ""               # "node_budget" | "tier_ceiling" | "" (+ "_no_sims" suffix)
    budget: Dict[str, float] = field(default_factory=dict)
    root_cf_sensitivity: Optional[np.ndarray] = None   # S/max(N,1) per root edge, cf_backprop only


class _Node:
    __slots__ = ("z", "depth", "prior", "allowed", "N", "W", "S", "children", "reward", "terminal", "expanded")

    def __init__(self, z: np.ndarray, depth: int, reward: float, terminal: bool) -> None:
        self.z = z
        self.depth = depth
        self.reward = reward
        self.terminal = terminal
        self.expanded = False
        self.prior = None
        self.allowed = None
        self.N = None
        self.W = None
        self.S = None
        self.children: Dict[int, "_Node"] = {}


class CausalMCTSRNN:
    """Adaptive System-1 / System-2 decision engine on a learned latent world model."""

    def __init__(self, runtime: Optional[RNNSetAdapterRuntime], dynamics: Optional[LieLatentDynamics] = None,
                 damper: Optional[ContractionDamper] = None, *, prior_source: Optional[DualManifoldPrior] = None,
                 temperature: float = 1.0,
                 top_k: int = 2, sims_min: int = 8, sims_max: int = 16, max_depth: int = 16,
                 rollout_len: int = 12, probe_len: int = 10, damper_every: int = 2,
                 c_puct: float = 1.5, gamma: float = 0.97, shaping: float = 0.1,
                 act_eps: float = 1e-3, act_max_steps: int = 16,
                 tau_entropy: float = 0.75, tau_cf: float = 0.25,
                 cf_eps: float = 0.05, cf_samples: int = 4, seed: int = 0,
                 node_budget: int = 64, tier: str = "tier2", clock: Callable[[], int] = time.perf_counter_ns,
                 selection: str = "puct",
                 cf_backprop: bool = False, cf_penalty: float = 0.5,
                 manifold_basis: Optional[np.ndarray] = None) -> None:
        if (runtime is None) == (prior_source is None):
            raise ValueError("exactly one of runtime or prior_source must be given")
        if runtime is not None and not isinstance(runtime, RNNSetAdapterRuntime):
            raise TypeError("runtime must be an RNNSetAdapterRuntime")
        if prior_source is not None and not isinstance(prior_source, DualManifoldPrior):
            raise TypeError("prior_source must be a DualManifoldPrior")
        if dynamics is not None and damper is not None and damper.basis.shape[0] != dynamics.dim:
            raise ValueError("damper and dynamics latent dims disagree")
        if dynamics is not None and prior_source is not None and prior_source.D_s != dynamics.dim:
            raise ValueError("prior_source.head.D_s must equal the dynamics latent dim")
        if not (math.isfinite(temperature) and temperature > 0.0):
            raise ValueError("temperature must be positive")
        if top_k < 1 or not (1 <= sims_min <= sims_max) or max_depth < 1 or rollout_len < 0:
            raise ValueError("invalid search budget")
        if act_max_steps < 1 or act_eps < 0.0 or not (0.0 < gamma <= 1.0):
            raise ValueError("invalid ACT / discount settings")
        if node_budget < 1:
            raise ValueError("node_budget must be >= 1")
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}; must be one of {sorted(TIERS)}")
        if selection not in ("puct", "ucb1"):
            raise ValueError(f"unknown selection {selection!r}; must be 'puct' or 'ucb1'")
        self.rt = runtime
        self.prior_source = prior_source
        self.dyn = dynamics
        self.damper = damper
        self.T = float(temperature)
        self.top_k = int(top_k)
        self.sims_min, self.sims_max = int(sims_min), int(sims_max)
        self.max_depth = int(max_depth)
        self.rollout_len = int(rollout_len)
        self.probe_len = int(probe_len)
        self.damper_every = max(1, int(damper_every))
        self.c_puct = float(c_puct)
        self.gamma = float(gamma)
        self.shaping = float(shaping)
        self.act_eps = float(act_eps)
        self.act_max_steps = int(act_max_steps)
        self.tau_entropy = float(tau_entropy)
        self.tau_cf = float(tau_cf)
        self.cf_eps = float(cf_eps)
        self.cf_samples = int(cf_samples)
        self.seed = int(seed)
        self.node_budget = int(node_budget)
        self.tier = tier
        self.clock = clock
        self.selection = selection
        self.cf_backprop = bool(cf_backprop)
        self.cf_penalty = float(cf_penalty)

        basis_for_cf: Optional[np.ndarray] = None
        if manifold_basis is not None:
            B = _finite(manifold_basis, "manifold_basis")
            if B.ndim != 2:
                raise ValueError("manifold_basis must be 2D (d, m)")
            if not np.allclose(B.T @ B, np.eye(B.shape[1]), atol=1e-6):
                raise ValueError("manifold_basis columns must be orthonormal")
            basis_for_cf = B  # keep float64: _cf_delta upcasts damper.basis too, this avoids losing precision
        if damper is not None:
            basis_for_cf = damper.basis
        if self.cf_backprop and basis_for_cf is None:
            raise ValueError("cf_backprop requires damper.basis or manifold_basis "
                              "(a random direction is not a counterfactual)")
        self._cf_basis = basis_for_cf

        if runtime is not None:
            p = runtime.p
            self._UB, self._VB = p["U_B"], p["V_B"]
            self._UA, self._VA = p["U_A"], p["V_A"]
            self._dvec, self._a = runtime._dvec, runtime.a_scale
            self._Ws = p["W_s"]
            self._inv_sqrt_d = np.float32(1.0 / math.sqrt(runtime.cfg["d"]))

    # ------------------------------------------------------------------ System 1
    def act_think(self, zq: np.ndarray) -> Tuple[np.ndarray, int]:
        """Parallel RNN think loop with residual halting (ACT).

        Same step as RNNSetAdapterRuntime._think. act_eps=0 with
        act_max_steps=think_steps reproduces it exactly.
        """
        if self.rt is None:
            raise ValueError("act_think() needs a runtime (RNNSetAdapterRuntime); "
                              "this engine was built with prior_source instead")
        u = self._UB @ (self._VB.T @ zq)
        h = np.zeros_like(zq)
        thr = self.act_eps * (float(np.linalg.norm(u)) + 1e-12)
        steps = 0
        for steps in range(1, self.act_max_steps + 1):
            h_new = self._a * (self._dvec * h + self._UA @ (self._VA.T @ h)) + u
            delta = float(np.linalg.norm(h_new - h))
            h = h_new
            if delta <= thr:
                break
        return zq + h, steps

    def prior_logits(self, q: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, int]:
        if self.rt is None:
            raise ValueError("prior_logits() needs a runtime (RNNSetAdapterRuntime); "
                              "this engine was built with prior_source instead")
        q_out, steps = self.act_think(self.rt._encode(q))
        H = self.rt._set_block(self.rt._encode(C))
        return (H @ (q_out @ self._Ws)) * self._inv_sqrt_d, steps

    def prior(self, q: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        if self.rt is None:
            raise ValueError("prior() needs a runtime (RNNSetAdapterRuntime); "
                              "this engine was built with prior_source instead")
        logits, steps = self.prior_logits(q, C)
        return _softmax(logits / np.float32(self.T)).astype(np.float64), logits, steps

    def counterfactual_sensitivity(self, q: np.ndarray, C: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
        """Mean total variation and argmax flip rate under do(X -> X + delta), delta _|_ span(q, C).

        delta has norm cf_eps * ||x|| per row and lies in the orthogonal
        complement of the rows of [q; C]. Each row's noise is seeded from that
        row's own bytes, so the estimate is deterministic and permutation
        equivariant over the candidates (the span is order-free too).
        """
        X = np.vstack([q[None, :], C]).astype(np.float64)
        D = X.shape[1]
        if X.shape[0] >= D:
            return 0.0, 0.0  # no orthogonal complement to perturb into
        Qb, _ = np.linalg.qr(X.T)
        rows32 = np.vstack([q[None, :], C]).astype(np.float32)
        rngs = [np.random.default_rng([self.seed, zlib.crc32(r.tobytes())]) for r in rows32]
        top = int(np.argmax(probs))
        tv, flips = 0.0, 0
        for _ in range(self.cf_samples):
            N = np.stack([g.standard_normal(D) for g in rngs])
            N -= (N @ Qb) @ Qb.T
            N *= (self.cf_eps * np.linalg.norm(X, axis=1, keepdims=True)
                  / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12))
            Xp = (X + N).astype(np.float32)
            p2, _, _ = self.prior(Xp[0], Xp[1:])
            tv += 0.5 * float(np.abs(p2 - probs).sum())
            flips += int(np.argmax(p2) != top)
        return tv / self.cf_samples, flips / self.cf_samples

    def classify(self, q: np.ndarray, C: np.ndarray) -> Decision:
        """Single-step candidate choice with no transition model (e.g. MCQ).

        The escalation triggers are still computed and reported, but with no
        dynamics there is nothing to search, so the decision is the fast path.
        """
        if self.rt is None:
            raise ValueError("classify() needs a runtime (RNNSetAdapterRuntime); "
                              "a prior_source-only engine has no single-step Set-Transformer "
                              "prior, use decide() over the latent dynamics instead")
        t0 = time.perf_counter()
        q = _finite(q, "q", np.float32)
        C = _finite(C, "C", np.float32)
        probs, _, steps = self.prior(q, C)
        h = _entropy(probs) / math.log(len(probs)) if len(probs) > 1 else 0.0
        cf, flip = self.counterfactual_sensitivity(q, C, probs)
        adj = (1.0 - cf) * probs + cf / len(probs)
        trig = {"entropy": h >= self.tau_entropy, "counterfactual": cf >= self.tau_cf, "probe_conflict": False}
        return Decision(int(np.argmax(probs)), adj, probs, "fast", trig, h, cf, 0.0, steps, 0, 0,
                        (time.perf_counter() - t0) * 1e3, extras={"cf_flip_rate": flip})

    # ------------------------------------------------------------------ latent world model
    def _scalars(self, Z: np.ndarray, ro: LatentReadout) -> np.ndarray:
        return ro._scalars @ Z  # (3,) or (3, n): fail, success, potential

    def _reward_terminal(self, z: np.ndarray, ro: LatentReadout) -> Tuple[float, bool]:
        f, s, _ = self._scalars(z, ro)
        if f > 0.5:
            return -1.0, True
        if s > 0.5:
            return 1.0, True
        return 0.0, False

    def _rollout(self, z: np.ndarray, ro: LatentReadout, length: int, depth0: int) -> float:
        """Cheap default policy: argmax of the one-step readout over all actions.

        This is the fast rollout policy of AlphaGo-style search; the expensive
        Set-Transformer prior is used only to expand tree nodes.
        """
        ret, disc = 0.0, 1.0
        for t in range(length):
            nxt = self.dyn.step_all(z)                          # (A, d)
            f, s, pot = self._scalars(nxt.T, ro)
            score = s - f + self.shaping * pot
            a = int(np.argmax(score))
            z = nxt[a]
            if (depth0 + t + 1) % self.damper_every == 0 and self.damper is not None:
                z = self.damper.apply(z)
            disc *= self.gamma
            if f[a] > 0.5:
                return ret - disc
            if s[a] > 0.5:
                return ret + disc
        return ret + disc * self.shaping * float(ro.potential @ z)

    def _expand(self, node: _Node, ro: LatentReadout) -> None:
        if self.prior_source is not None:
            probs = self.prior_source.probs(node.z)
        else:
            C = (ro.features @ self.dyn.step_all(node.z).T).T  # (A, D)
            probs, _, _ = self.prior(ro.features @ node.z, C)
        k = min(self.top_k, len(probs))
        allowed = np.argsort(-probs, kind="stable")[:k]
        pr = np.zeros_like(probs)
        pr[allowed] = probs[allowed] / probs[allowed].sum()
        node.prior, node.allowed = pr, allowed
        node.N = np.zeros(len(probs))
        node.W = np.zeros(len(probs))
        node.S = np.zeros(len(probs))
        node.expanded = True

    def _select(self, node: _Node) -> int:
        """Pick an allowed action: PUCT (default) or UCB1; Q swaps in a cf penalty when cf_backprop is on."""
        idx = node.allowed
        N = node.N[idx]
        W = node.W[idx]
        if self.cf_backprop:
            Q = np.where(N > 0, W / np.maximum(N, 1) - self.cf_penalty * node.S[idx] / np.maximum(N, 1), 0.0)
        else:
            Q = np.where(N > 0, W / np.maximum(N, 1), 0.0)
        if self.selection == "puct":
            n_tot = node.N.sum()
            u = self.c_puct * node.prior[idx] * math.sqrt(n_tot + 1.0) / (1.0 + N)
            return int(idx[int(np.argmax(Q + u))])
        # ucb1: any unvisited allowed child first, in prior order (idx is already prior-sorted)
        unvisited = idx[N == 0]
        if unvisited.size > 0:
            return int(unvisited[0])
        n_tot = node.N.sum()
        u = self.c_puct * np.sqrt(np.log(n_tot) / N)
        return int(idx[int(np.argmax(Q + u))])

    def _cf_delta(self, z: np.ndarray) -> Optional[np.ndarray]:
        """A direction orthogonal to the manifold span, seeded from z, scaled to cf_eps*||z||.

        None means the span already covers the full space (nothing orthogonal
        to perturb into): the caller must treat that as v_cf = v, not as an
        error and not as a fabricated number.
        """
        z64 = np.asarray(z, dtype=np.float64)
        rng = np.random.default_rng([self.seed, zlib.crc32(np.asarray(z, dtype=np.float32).tobytes())])
        delta = rng.standard_normal(z64.shape[0])
        if self._cf_basis is not None:
            B = np.asarray(self._cf_basis, dtype=np.float64)
            delta = delta - B @ (B.T @ delta)
        nrm = float(np.linalg.norm(delta))
        if nrm < 1e-12:
            return None
        znorm = float(np.linalg.norm(z64))
        return (delta * (self.cf_eps * znorm / nrm)).astype(np.float32)

    def _child(self, node: _Node, a: int, ro: LatentReadout) -> _Node:
        ch = node.children.get(a)
        if ch is None:
            z = self.dyn.step(node.z, a)
            if self.damper is not None and (node.depth + 1) % self.damper_every == 0:
                z = self.damper.apply(z)
            r, term = self._reward_terminal(z, ro)
            ch = _Node(z, node.depth + 1, r, term)
            node.children[a] = ch
        return ch

    def _simulate(self, root: _Node, ro: LatentReadout) -> int:
        path: List[Tuple[_Node, int]] = []
        node = root
        new_nodes = 0
        cf_gap = 0.0
        while True:
            if node.terminal:
                value = 0.0
                break
            if not node.expanded:
                self._expand(node, ro)
                new_nodes += 1
                value = self._rollout(node.z, ro, self.rollout_len, node.depth)
                if self.cf_backprop:
                    delta = self._cf_delta(node.z)
                    v_cf = value if delta is None else self._rollout(node.z + delta, ro, self.rollout_len, node.depth)
                    cf_gap = abs(value - v_cf)
                break
            if node.depth >= self.max_depth:
                value = self.shaping * float(ro.potential @ node.z)
                break
            a = self._select(node)
            path.append((node, a))
            node = self._child(node, a, ro)
        for parent, a in reversed(path):
            ch = parent.children[a]
            value = ch.reward + self.gamma * value
            parent.N[a] += 1
            parent.W[a] += value
            if self.cf_backprop:
                parent.S[a] += cf_gap
        return new_nodes

    def probe(self, z: np.ndarray, action: int, ro: LatentReadout) -> float:
        """Return of taking `action` then the rollout policy for probe_len steps."""
        z1 = self.dyn.step(z, action)
        r, term = self._reward_terminal(z1, ro)
        if term:
            return r
        return r + self.gamma * self._rollout(z1, ro, self.probe_len, 1)

    def decide(self, z: np.ndarray, readout: LatentReadout, *, mode: str = "adaptive") -> Decision:
        """mode: "adaptive" (gated), "fast" (never search), "mcts" (always search),
        "adaptive_entropy_only" (gate on entropy alone, no probe or cf),
        "rollout_only" (ablation: one probe rollout per action, pick the best, no tree)."""
        if self.dyn is None:
            raise ValueError("decide() needs a latent dynamics model; use classify() for single-step tasks")
        if mode not in ("adaptive", "fast", "mcts", "adaptive_entropy_only", "rollout_only"):
            raise ValueError(f"unknown mode {mode!r}")
        t0 = time.perf_counter()
        t_clock0 = self.clock()  # Spec 24 times the decision end-to-end, not just the sim loop
        z = _finite(z, "z", np.float32)
        if z.shape != (self.dyn.dim,):
            raise ValueError(f"z must have shape ({self.dyn.dim},)")
        using_prior_source = self.prior_source is not None
        if using_prior_source:
            probs, steps = self.prior_source.probs(z), 0
            q = C = None
        else:
            C = (readout.features @ self.dyn.step_all(z).T).T
            q = readout.features @ z
            probs, _, steps = self.prior(q, C)
        K = len(probs)
        h = _entropy(probs) / math.log(K) if K > 1 else 0.0
        top = int(np.argmax(probs))
        cf, flip, probe_ret = 0.0, 0.0, 0.0
        trig = {"entropy": h >= self.tau_entropy, "counterfactual": False, "probe_conflict": False}
        cf_extras: Dict[str, float] = {}
        if mode in ("adaptive", "mcts"):
            if using_prior_source:
                cf, flip = 0.0, 0.0
                cf_extras["cf_input_sensitivity"] = "n/a_dual_manifold"
            else:
                cf, flip = self.counterfactual_sensitivity(q, C, probs)
            trig["counterfactual"] = cf >= self.tau_cf
            probe_ret = self.probe(z, top, readout)
            trig["probe_conflict"] = probe_ret < -0.5
        if mode == "rollout_only":
            rets = np.array([self.probe(z, a, readout) for a in range(K)])
            best = int(np.lexsort((probs, rets))[-1])  # best return, ties by prior
            return Decision(best, probs, probs, "rollout", trig, h, cf, float(rets[top]), steps, K, 0,
                            (time.perf_counter() - t0) * 1e3, root_q=rets)
        escalate = mode == "mcts" or (mode != "fast" and any(trig.values()))
        if not escalate:
            adj = (1.0 - cf) * probs + cf / K
            extras = {"cf_flip_rate": flip}
            extras.update(cf_extras)
            return Decision(top, adj, probs, "fast", trig, h, cf, probe_ret, steps, 0, 0,
                            (time.perf_counter() - t0) * 1e3, extras=extras)
        urgency = min(1.0, h + (1.0 if trig["probe_conflict"] else 0.0))
        n_sim = int(round(self.sims_min + (self.sims_max - self.sims_min) * urgency))
        contract = TIERS[self.tier]
        root = _Node(z, 0, 0.0, False)
        nodes = 0
        sims_completed = 0
        truncated = False
        truncation_reason = ""
        for _ in range(n_sim):
            if nodes >= self.node_budget:
                truncated, truncation_reason = True, "node_budget"
                break
            elapsed_us = (self.clock() - t_clock0) / 1000.0
            over_budget = elapsed_us >= contract.ceiling_us if contract.strict else elapsed_us > contract.ceiling_us
            if over_budget:
                truncated, truncation_reason = True, "tier_ceiling"
                break
            nodes += self._simulate(root, readout)
            sims_completed += 1
        elapsed_us_final = (self.clock() - t_clock0) / 1000.0
        if sims_completed == 0 and truncated:
            truncation_reason = f"{truncation_reason}_no_sims"

        if sims_completed == 0:
            best, pol, adj = top, probs, probs
            qv = np.full(K, -np.inf)
        else:
            visits = root.N.copy()
            qv = np.where(visits > 0, root.W / np.maximum(visits, 1), -np.inf)
            best = int(np.lexsort((qv, visits))[-1])  # most visits, ties by Q
            pol = visits / visits.sum() if visits.sum() > 0 else probs
            adj = (1.0 - cf) * pol + cf / K

        root_cf_sensitivity = None
        if self.cf_backprop and sims_completed > 0:
            cf_extras["cf_backprop_mean_sensitivity"] = float(root.S.sum()) / max(float(root.N.sum()), 1.0)
            root_cf_sensitivity = root.S / np.maximum(root.N, 1)

        budget = {
            "node_budget": float(self.node_budget),
            "tier": self.tier,
            "ceiling_us": float(contract.ceiling_us),
            "nodes_used": float(nodes),
            "sims_completed": float(sims_completed),
            "elapsed_us": float(elapsed_us_final),
        }
        extras = {"cf_flip_rate": flip}
        extras.update(cf_extras)
        return Decision(best, adj, probs, "mcts", trig, h, cf, probe_ret, steps, sims_completed, nodes,
                        (time.perf_counter() - t0) * 1e3, root_q=qv, raw_policy=pol,
                        extras=extras, truncated=truncated, truncation_reason=truncation_reason,
                        budget=budget, root_cf_sensitivity=root_cf_sensitivity)
