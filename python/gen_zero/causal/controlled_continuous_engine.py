"""Controlled Continuous Dynamical Thinking Engine (pure NumPy).

Replaces the "pull the latent toward candidate a" pseudo-search
(``moved = tanh(zc + 0.3 * (emb[a] - zc))``). That step raises the score of
whatever candidate it moves toward, so any answer, even a wrong one, can be
made to look confident. Here the candidates never enter the dynamics.

Model
-----
Frozen problem context ``c`` (read-only, SHA-256 pinned). It is a set of
numeric constraints produced by an upstream encoder E(x, C):

    premises        A q = b                      (relations the answer must satisfy)
    counterfactuals U q <= m                     (states the problem rules out)
    exclusions      not (s_e . q > 0 and t_e . q > 0)   (mutually exclusive propositions)

Phase space z = (q, p). Dissipative Hamiltonian dynamics:

    dq/dt = M^-1 p
    dp/dt = -grad_q V(q; c) - gamma p
    H(q, p; c) = 0.5 p^T M^-1 p + V(q; c),   dH/dt = -gamma p^T M^-1 p <= 0

    V(q; c) = 0.5 w ||A q - b||^2                      (premise potential, bilinear in c and q)
            + kappa * L_contra(q; c)                   (contradiction energy)
            + 0.5 omega ||q||^2                        (optional regulariser, default 0)

    L_contra = 0.5 ||relu(U q - m)||^2 + 0.5 sum_e (relu(s_e.q) relu(t_e.q))^2

Every step evaluates grad V, which reads A, b, U, m, S, T from ``c``. There
is no step that does not read ``c``.

Integrator: conformally symplectic leapfrog with an exact exp(-gamma dt / 2)
friction half-step on each side. Leapfrog only conserves a *shadow*
Hamiltonian, so the raw discrete H can tick up by O(dt^2). The energy guard
rejects any step with H_new > H_old, zeroes the momentum (which can only
lower H) and halves dt. With the guard on, the recorded H trace is
non-increasing by construction; ``rejected_steps`` reports how often the
guard had to act.

Reflection and backtracking
---------------------------
The residual R(q, c) = [A q - b, relu(U q - m), relu(s.q) relu(t.q)] is the
backward consistency check: the state must reconstruct the premises and
break no counterfactual or exclusion. Only ||R|| < eps_tol counts as
convergence. The exclusion term is non-convex, so the free dynamics can
settle in a false attractor with ||R|| > 0. Then the engine backtracks: it
returns to the episode start (the problem-derived checkpoint), fixes one
side of every exclusion pair to a half-space (a convex sub-problem) and
runs again. Branch assignments are tried in order of how little they
contradict the failed attractor. If no branch reaches ||R|| < eps_tol, the
result is UNCONVERGED and the decoder abstains.

Decoder
-------
Candidates are scored only in ``decode``: each candidate gets its own
residual R(e_j, c) (does it satisfy the problem?) and its distance to z_K in
the row space of A (the part of z_K the premises pin down). A candidate is
admissible only if its residual is below ``candidate_tol``. The decoder
abstains when the engine did not converge, when no candidate is admissible,
or when the best two admissible candidates are closer than ``ambiguity_margin``.

Limits (stated plainly)
-----------------------
* This module does not encode text. It needs an encoder E(x, C) that turns a
  problem into (A, b, U, m, S, T). None is trained here. Accuracy on real
  benchmarks is therefore unmeasured.
* When the constraints are exact, the decoder's residual term alone can
  already reject a wrong candidate. The dynamics add a candidate-free answer
  state z_K, detection of contradictory contexts, and the gate that refuses
  to decode an unconverged state.
* Branch enumeration is exhaustive for up to ``MAX_EXHAUSTIVE_PAIRS``
  exclusion pairs. Above that only single-flip neighbours are tried, so the
  search is incomplete.

Language independence: all operations are float64 linear algebra on
arrays. There is no vocabulary, token, or text format anywhere.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ContextTamperedError",
    "Decision",
    "EngineConfig",
    "EngineStatus",
    "EpisodeRecord",
    "FrozenProblemContext",
    "ControlledContinuousEngine",
    "NonFiniteInputError",
    "ThinkingTrace",
    "context_residual",
]

MAX_EXHAUSTIVE_PAIRS = 12


class NonFiniteInputError(ValueError):
    """An input array holds NaN or Inf."""


class ContextTamperedError(RuntimeError):
    """The frozen context no longer matches its SHA-256 digest."""


class EngineStatus(str, enum.Enum):
    CONVERGED = "CONVERGED"
    UNCONVERGED = "UNCONVERGED"
    NONFINITE = "NONFINITE"


def _finite_array(name: str, x, ndim: int) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise NonFiniteInputError(f"{name} contains NaN or Inf")
    return a


def _readonly(a: np.ndarray) -> np.ndarray:
    # A view over an immutable bytes object: the WRITEABLE flag cannot be turned back on.
    return np.frombuffer(np.ascontiguousarray(a).tobytes(), dtype=np.float64).reshape(a.shape)


@dataclasses.dataclass(frozen=True)
class FrozenProblemContext:
    """Read-only problem memory c = E(x, C). Build it with ``from_arrays``."""

    A: np.ndarray
    b: np.ndarray
    U: np.ndarray
    m: np.ndarray
    S: np.ndarray
    T: np.ndarray
    digest: str

    @classmethod
    def from_arrays(cls, A, b, U=None, m=None, S=None, T=None) -> "FrozenProblemContext":
        A = _finite_array("A", A, 2)
        n_prem, dim = A.shape
        if n_prem < 1 or dim < 1:
            # No premise means the dynamics would not read the problem at all.
            raise ValueError("context needs at least one premise row and dim >= 1")
        b = _finite_array("b", b, 1)
        if b.shape != (n_prem,):
            raise ValueError(f"b must have shape ({n_prem},), got {b.shape}")
        U = np.zeros((0, dim)) if U is None else _finite_array("U", U, 2)
        m = np.zeros(0) if m is None else _finite_array("m", m, 1)
        if U.shape[1] != dim or m.shape != (U.shape[0],):
            raise ValueError("U must be (h, dim) and m must be (h,)")
        S = np.zeros((0, dim)) if S is None else _finite_array("S", S, 2)
        T = np.zeros((0, dim)) if T is None else _finite_array("T", T, 2)
        if S.shape != T.shape or S.shape[1] != dim:
            raise ValueError("S and T must both be (e, dim)")
        arrays = [_readonly(x) for x in (A, b, U, m, S, T)]
        return cls(*arrays, digest=cls._hash(arrays))

    @staticmethod
    def _hash(arrays: Sequence[np.ndarray]) -> str:
        h = hashlib.sha256()
        for a in arrays:
            h.update(repr(a.shape).encode())
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    @property
    def dim(self) -> int:
        return self.A.shape[1]

    @property
    def n_exclusions(self) -> int:
        return self.S.shape[0]

    def verify(self) -> None:
        if self._hash((self.A, self.b, self.U, self.m, self.S, self.T)) != self.digest:
            raise ContextTamperedError("frozen context changed after construction")

    def as_vector(self) -> np.ndarray:
        """Flat encoding c in R^{d_c}, for callers that want one vector."""
        return np.concatenate([x.ravel() for x in (self.A, self.b, self.U, self.m, self.S, self.T)])


def context_residual(q: np.ndarray, ctx: FrozenProblemContext) -> Dict[str, np.ndarray]:
    """Backward consistency residual R(q, c), split by constraint type."""
    x, y = ctx.S @ q, ctx.T @ q
    return {
        "premise": ctx.A @ q - ctx.b,
        "counterfactual": np.maximum(ctx.U @ q - ctx.m, 0.0),
        "exclusion": np.maximum(x, 0.0) * np.maximum(y, 0.0),
    }


def _residual_norm(r: Dict[str, np.ndarray]) -> float:
    return float(math.sqrt(sum(float(v @ v) for v in r.values())))


@dataclasses.dataclass(frozen=True)
class EngineConfig:
    premise_weight: float = 1.0
    kappa: float = 10.0            # contradiction energy weight
    omega: float = 0.0             # optional ||q||^2 regulariser
    mass: float = 1.0              # isotropic, so the flow is rotation-equivariant
    dt_frac: float = 0.9           # dt = dt_frac / sqrt(L / mass)
    gamma_frac: float = 0.1        # gamma = gamma_frac * sqrt(L / mass)
    max_steps: int = 20000         # per episode
    grad_tol: float = 1e-8
    momentum_tol: float = 1e-8
    eps_tol: float = 1e-6          # ||R|| threshold for a valid convergence
    max_backtracks: int = 16
    energy_guard: bool = True
    dt_min_frac: float = 1e-8
    stall_window: int = 500        # iterations without float-resolvable energy loss

    def __post_init__(self) -> None:
        for name in ("premise_weight", "kappa", "mass", "dt_frac", "gamma_frac",
                     "grad_tol", "momentum_tol", "eps_tol", "dt_min_frac"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be finite and > 0")
        if not (math.isfinite(self.omega) and self.omega >= 0):
            raise ValueError("omega must be finite and >= 0")
        if self.max_steps < 1 or self.max_backtracks < 0 or self.stall_window < 1:
            raise ValueError("max_steps must be >= 1 and max_backtracks >= 0")


@dataclasses.dataclass
class EpisodeRecord:
    branch: Optional[Tuple[int, ...]]   # None = free non-convex episode
    steps: int
    rejected_steps: int
    exit_reason: str                    # settled | stalled | guard_floor | max_steps | nonfinite
    residual_norm: float
    energy: List[float]                 # H after every accepted step (index 0 = start)
    contradiction: List[float]          # L_contra of the original constraints, same indices
    states: List[np.ndarray]            # q at the same indices (only if record_states)

    @property
    def settled(self) -> bool:
        return self.exit_reason == "settled"


@dataclasses.dataclass
class ThinkingTrace:
    status: EngineStatus
    q: Optional[np.ndarray]
    p: Optional[np.ndarray]
    residual: Optional[Dict[str, np.ndarray]]
    residual_norm: float
    episodes: List[EpisodeRecord]
    context_digest: str

    @property
    def backtracks(self) -> int:
        return max(0, len(self.episodes) - 1)


@dataclasses.dataclass
class Decision:
    choice: Optional[int]
    abstained: bool
    reason: str
    candidate_residuals: Optional[np.ndarray]
    proximity: Optional[np.ndarray]
    admissible: Optional[np.ndarray]


class _Potential:
    """V(q; c) for one episode. ``branch`` fixes one side of each exclusion pair."""

    def __init__(self, ctx: FrozenProblemContext, cfg: EngineConfig,
                 branch: Optional[Tuple[int, ...]]) -> None:
        self.ctx, self.cfg, self.branch = ctx, cfg, branch
        if branch is not None:
            # Row e of B is s_e (branch 0: s_e.q <= 0) or t_e (branch 1: t_e.q <= 0).
            rows = [ctx.S[e] if side == 0 else ctx.T[e] for e, side in enumerate(branch)]
            self.B = np.array(rows).reshape(len(rows), ctx.dim)

    def energy_grad(self, q: np.ndarray) -> Tuple[float, np.ndarray]:
        c, cfg = self.ctx, self.cfg
        r = c.A @ q - c.b
        energy = 0.5 * cfg.premise_weight * float(r @ r)
        grad = cfg.premise_weight * (c.A.T @ r)
        cf = np.maximum(c.U @ q - c.m, 0.0)
        contra = 0.5 * float(cf @ cf)
        cgrad = c.U.T @ cf
        if self.branch is None:
            xp, yp = np.maximum(c.S @ q, 0.0), np.maximum(c.T @ q, 0.0)
            f = xp * yp
            contra += 0.5 * float(f @ f)
            cgrad = cgrad + c.S.T @ (f * yp) + c.T.T @ (f * xp)
        elif len(self.branch):
            h = np.maximum(self.B @ q, 0.0)
            contra += 0.5 * float(h @ h)
            cgrad = cgrad + self.B.T @ h
        energy += cfg.kappa * contra + 0.5 * cfg.omega * float(q @ q)
        grad = grad + cfg.kappa * cgrad + cfg.omega * q
        return energy, grad

    def lipschitz_bound(self, q_scale: float) -> float:
        c, cfg = self.ctx, self.cfg
        # np.square, not float ** 2: an overflow must give inf (fail-closed), not raise.
        norm2 = lambda M: float(np.square(np.linalg.norm(M, 2))) if M.size else 0.0
        excl = (norm2(c.S) + norm2(c.T)) * max(q_scale, 1.0) ** 2 if self.branch is None else (
            norm2(self.B) if len(self.branch) else 0.0)
        return cfg.premise_weight * norm2(c.A) + cfg.kappa * (norm2(c.U) + excl) + cfg.omega


def _contradiction(q: np.ndarray, ctx: FrozenProblemContext) -> float:
    r = context_residual(q, ctx)
    return 0.5 * float(r["counterfactual"] @ r["counterfactual"] + r["exclusion"] @ r["exclusion"])


class ControlledContinuousEngine:
    """Context-conditioned dissipative latent dynamics with residual reflection."""

    def __init__(self, config: Optional[EngineConfig] = None, record_states: bool = False) -> None:
        self.config = config or EngineConfig()
        self.record_states = record_states

    # -- dynamics ---------------------------------------------------------
    def hamiltonian(self, q: np.ndarray, p: np.ndarray, pot: _Potential) -> float:
        return 0.5 * float(p @ p) / self.config.mass + pot.energy_grad(q)[0]

    def step(self, q: np.ndarray, p: np.ndarray, ctx: FrozenProblemContext, dt: float,
             gamma: float, branch: Optional[Tuple[int, ...]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """One friction-leapfrog step. Both force evaluations read ``ctx``."""
        return self._leapfrog(q, p, _Potential(ctx, self.config, branch), dt, gamma)

    def _leapfrog(self, q, p, pot: _Potential, dt: float, gamma: float):
        decay = math.exp(-0.5 * gamma * dt)
        p = decay * p
        p = p - 0.5 * dt * pot.energy_grad(q)[1]
        q = q + dt * p / self.config.mass
        p = p - 0.5 * dt * pot.energy_grad(q)[1]
        return q, decay * p

    def _episode(self, q0: np.ndarray, p0: np.ndarray, ctx: FrozenProblemContext,
                 branch: Optional[Tuple[int, ...]]):
        cfg = self.config
        pot = _Potential(ctx, cfg, branch)
        # ||q0|| (not max |q0_i|) keeps the step size rotation-invariant.
        with np.errstate(over="ignore", invalid="ignore"):
            lip = max(pot.lipschitz_bound(float(np.linalg.norm(q0))), 1e-12)
        omega0 = math.sqrt(lip / cfg.mass)
        dt_max = cfg.dt_frac / omega0
        dt_min = cfg.dt_min_frac * dt_max
        gamma = cfg.gamma_frac * omega0
        dt = dt_max

        q, p = q0.copy(), p0.copy()
        energy, grad = pot.energy_grad(q)
        h = 0.5 * float(p @ p) / cfg.mass + energy
        rec = EpisodeRecord(branch, 0, 0, "max_steps", float("nan"), [h], [_contradiction(q, ctx)],
                            [q.copy()] if self.record_states else [])
        if not (math.isfinite(h) and math.isfinite(lip) and np.all(np.isfinite(grad))):
            rec.exit_reason = "nonfinite"
            return None, None, rec
        h_window = h
        for it in range(1, cfg.max_steps + 1):
            if (float(np.linalg.norm(grad)) < cfg.grad_tol
                    and float(np.linalg.norm(p)) < cfg.momentum_tol):
                rec.exit_reason = "settled"
                break
            if it % cfg.stall_window == 0:
                # Near a minimum with V > 0, energy steps fall below float rounding and
                # the guard starts rejecting valid steps. Stop; the residual check decides.
                if h_window - h <= 1e-12 * abs(h_window):
                    rec.exit_reason = "stalled"
                    break
                h_window = h
            q_new, p_new = self._leapfrog(q, p, pot, dt, gamma)
            energy_new, grad_new = pot.energy_grad(q_new)
            h_new = 0.5 * float(p_new @ p_new) / cfg.mass + energy_new
            if not (math.isfinite(h_new) and np.all(np.isfinite(q_new)) and np.all(np.isfinite(p_new))):
                rec.exit_reason = "nonfinite"
                return None, None, rec
            if cfg.energy_guard and h_new > h:
                # Reject: drop momentum (lowers H to V(q)) and retry with a smaller step.
                rec.rejected_steps += 1
                p = np.zeros_like(p)
                h = energy
                rec.energy.append(h)
                rec.contradiction.append(rec.contradiction[-1])
                if self.record_states:
                    rec.states.append(q.copy())
                dt *= 0.5
                if dt < dt_min:
                    rec.exit_reason = "guard_floor"
                    break
                continue
            q, p, energy, grad, h = q_new, p_new, energy_new, grad_new, h_new
            rec.steps += 1
            rec.energy.append(h)
            rec.contradiction.append(_contradiction(q, ctx))
            if self.record_states:
                rec.states.append(q.copy())
            dt = min(dt * 1.1, dt_max)
        rec.residual_norm = _residual_norm(context_residual(q, ctx))
        return q, p, rec

    def _branch_order(self, q_fail: np.ndarray, ctx: FrozenProblemContext) -> List[Tuple[int, ...]]:
        """Branch assignments, least contradicted by the failed attractor first."""
        n = ctx.n_exclusions
        if n == 0:
            return []  # the free problem was already convex: nothing to flip
        pos = np.stack([np.maximum(ctx.S @ q_fail, 0.0), np.maximum(ctx.T @ q_fail, 0.0)], axis=1)
        greedy = tuple(int(i) for i in np.argmin(pos, axis=1))
        if n <= MAX_EXHAUSTIVE_PAIRS:
            options = list(itertools.product((0, 1), repeat=n))
        else:
            options = [greedy] + [greedy[:e] + (1 - greedy[e],) + greedy[e + 1:] for e in range(n)]
        cost = lambda br: float(sum(pos[e, side] for e, side in enumerate(br)))
        return sorted(options, key=lambda br: (cost(br), br))

    def think(self, q0, ctx: FrozenProblemContext, p0=None) -> ThinkingTrace:
        """Run the dynamics from q0 under context c. Candidates are not an input."""
        if not isinstance(ctx, FrozenProblemContext):
            raise TypeError("ctx must be a FrozenProblemContext")
        ctx.verify()
        q0 = _finite_array("q0", q0, 1)
        if q0.shape != (ctx.dim,):
            raise ValueError(f"q0 must have shape ({ctx.dim},), got {q0.shape}")
        p0 = np.zeros_like(q0) if p0 is None else _finite_array("p0", p0, 1)
        if p0.shape != q0.shape:
            raise ValueError("p0 must match q0")

        # Overflow turns into inf/nan, which the episode loop reports as NONFINITE.
        with np.errstate(over="ignore", invalid="ignore"):
            return self._think(q0, p0, ctx)

    def _think(self, q0: np.ndarray, p0: np.ndarray, ctx: FrozenProblemContext) -> ThinkingTrace:
        episodes: List[EpisodeRecord] = []
        q, p, rec = self._episode(q0, p0, ctx, None)
        episodes.append(rec)
        if q is None:
            return ThinkingTrace(EngineStatus.NONFINITE, None, None, None, float("nan"),
                                 episodes, ctx.digest)
        best = (rec.residual_norm, q, p)
        if rec.residual_norm >= self.config.eps_tol:
            # False attractor: backtrack to the problem-derived start and try branches.
            for branch in self._branch_order(q, ctx)[: self.config.max_backtracks]:
                qb, pb, rec = self._episode(q0, np.zeros_like(q0), ctx, branch)
                episodes.append(rec)
                if qb is None:
                    return ThinkingTrace(EngineStatus.NONFINITE, None, None, None, float("nan"),
                                         episodes, ctx.digest)
                if rec.residual_norm < best[0]:
                    best = (rec.residual_norm, qb, pb)
                if rec.residual_norm < self.config.eps_tol:
                    break
        ctx.verify()
        norm, q, p = best
        status = EngineStatus.CONVERGED if norm < self.config.eps_tol else EngineStatus.UNCONVERGED
        return ThinkingTrace(status, q, p, context_residual(q, ctx), norm, episodes, ctx.digest)

    # -- decoding ---------------------------------------------------------
    def decode(self, trace: ThinkingTrace, ctx: FrozenProblemContext, candidates,
               candidate_tol: float = 1e-6, ambiguity_margin: float = 1e-3) -> Decision:
        """D(z_K, c, C): the only place where candidates are scored."""
        ctx.verify()
        if trace.context_digest != ctx.digest:
            raise ContextTamperedError("trace was produced under a different context")
        C = _finite_array("candidates", candidates, 2)
        if C.shape[0] < 1 or C.shape[1] != ctx.dim:
            raise ValueError(f"candidates must be (k >= 1, {ctx.dim})")
        if trace.status is not EngineStatus.CONVERGED:
            return Decision(None, True, f"engine {trace.status.value}", None, None, None)
        res = np.array([_residual_norm(context_residual(e, ctx)) for e in C])
        # Distance to z_K where the premises pin z_K down: the row space of A.
        coords = np.linalg.lstsq(ctx.A.T, (C - trace.q).T, rcond=None)[0]
        prox = np.linalg.norm(ctx.A.T @ coords, axis=0)
        admissible = res < candidate_tol
        if not admissible.any():
            return Decision(None, True, "no candidate is consistent with the context", res, prox, admissible)
        score = np.where(admissible, res ** 2 + prox ** 2, np.inf)
        order = np.argsort(score, kind="stable")
        if admissible.sum() > 1 and score[order[1]] - score[order[0]] < ambiguity_margin:
            return Decision(None, True, "ambiguous: several candidates fit equally", res, prox, admissible)
        return Decision(int(order[0]), False, "ok", res, prox, admissible)

    def solve(self, q0, ctx: FrozenProblemContext, candidates, **decode_kwargs):
        trace = self.think(q0, ctx)
        return trace, self.decode(trace, ctx, candidates, **decode_kwargs)
