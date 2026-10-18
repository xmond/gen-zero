"""0-token latent MCTS over ETF candidate embeddings (pure NumPy).

The planner searches in latent space only. A state ``z`` is projected onto the
candidate embeddings, the softmax of the scaled dot products gives the node
priors ``P(a)``, and PUCT picks which action to expand next. No token is ever
generated.

Conventions:
  * ``transition_fn(z, action_idx) -> z'`` receives the action *index*, not the
    embedding. The default transition integrates a spherical Lyapunov gradient flow
    toward the chosen candidate embedding.
  * With ``num_simulations == 1`` the result equals the ETF MAP argmax: the
    PUCT exploration term uses ``max(N_total, 1)`` so an empty root still ranks
    actions by prior instead of falling back to index 0.
  * Value is in [0, 1] and is not sign-flipped on backup (single agent).
  * Opt-in, all off by default: root Sequential Halving over a Gumbel-Top-k
    candidate set (Karnin et al. 2013; Danihelka et al., ICLR 2022), root
    count-dominance early stopping, and bounded transposition memoization of
    ``transition_fn``, ``legal_actions_fn`` and the prior projection.
"""

from __future__ import annotations

import time
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .koopman_thinking import expm, fit_generator_from_derivatives

__all__ = ["LatentMctsNode", "LatentMctsPlanner", "fit_action_dynamics",
           "ContractiveHamiltonianTransition"]

_EPS = 1e-12
SEARCH_STRATEGIES = ("puct", "sequential_halving")


def fit_action_dynamics(trajectories, dim: int, num_actions: int):
    """Fit per-action continuous generators from offline transition records.

    Each record is ``(z, action, next_z)`` (unit time) or
    ``(z, action, next_z, dt)``. Returns ``(num_actions, dim, dim)``;
    unobserved actions have zero generators. Finite-difference regression
    estimates derivatives, so accuracy depends on sampling interval. These
    are observational estimates, not proof of causal identification.
    """
    if not isinstance(dim, (int, np.integer)) or dim < 1:
        raise ValueError("dim must be a positive integer")
    if not isinstance(num_actions, (int, np.integer)) or num_actions < 1:
        raise ValueError("num_actions must be a positive integer")
    samples = [[] for _ in range(num_actions)]
    derivatives = [[] for _ in range(num_actions)]
    for record in trajectories:
        if len(record) not in (3, 4):
            raise ValueError("records must contain state, action, next_state and optional dt")
        z, action, next_z = record[:3]
        dt = float(record[3]) if len(record) == 4 else 1.0
        x, y = np.asarray(z, dtype=float), np.asarray(next_z, dtype=float)
        if (x.shape != (dim,) or y.shape != (dim,) or
                not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))):
            raise ValueError("transition states must be finite vectors of length dim")
        if not isinstance(action, (int, np.integer)) or not 0 <= action < num_actions:
            raise ValueError("action must be an index in range")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        samples[action].append(x)
        derivatives[action].append((y - x) / dt)
    fitted = np.zeros((num_actions, dim, dim))
    for action in range(num_actions):
        if samples[action]:
            fitted[action] = fit_generator_from_derivatives(
                np.asarray(samples[action]), np.asarray(derivatives[action])
            ).matrix
    return fitted


def _unit(z):
    norm = math.hypot(*z)
    if 0 < norm < float("inf"):
        return z / norm
    # Scaling avoids overflow or underflow for extreme observations.
    scale = float(np.max(np.abs(z)))
    if scale == 0:
        return np.zeros_like(z)
    scaled = z / scale
    return scaled / np.linalg.norm(scaled)


class ContractiveHamiltonianTransition:
    """Split flow on the unit sphere with V(z)=1-e_a dot z.

    dz/dt = B_a z + gamma*(e_a-(e_a dot z)*z), where
    B_a=P_a (A_a-A_a.T)/2 P_a and P_a=I-e_a e_a.T.
    The skew Hamiltonian term preserves both norm and V; the spherical
    gradient gives dV/dt=-gamma*(1-(e_a dot z)**2). Each subflow is exact.
    The total arc per step is at most max_angle, hence chord divergence is
    at most 2*sin(max_angle/2). This bounds geometric drift, not calibrated
    epistemic uncertainty. Unfitted Hamiltonian generators default to zero.

    Targets supplied by the caller should be Simplex ETF rows. Arbitrary
    nonzero targets are supported. The antipode is stationary: no smooth
    spherical flow can have a single globally attracting equilibrium.
    Zero observations initialize at the target, outside the sphere contract.
    """

    def __init__(self, embeddings, gamma=0.3, generators=None, max_angle=0.3):
        e = np.asarray(embeddings, dtype=float)
        if (e.ndim != 2 or min(e.shape) < 1 or not np.all(np.isfinite(e))
                or np.any(np.max(np.abs(e), axis=1) == 0)):
            raise ValueError("embeddings must be finite nonzero rows")
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and non-negative")
        if not np.isfinite(max_angle) or not 0 < max_angle <= np.pi:
            raise ValueError("max_angle must be in (0, pi]")
        self.targets = np.stack([_unit(row) for row in e])
        self.gamma, self.max_angle = float(gamma), float(max_angle)
        self.rotations = None
        if generators is not None:
            a = np.asarray(generators, dtype=float)
            if a.shape != (len(e), e.shape[1], e.shape[1]) or not np.all(np.isfinite(a)):
                raise ValueError("generators must be finite with shape (actions, dim, dim)")
            self.rotations = []
            for target, matrix in zip(self.targets, a):
                projection = np.eye(e.shape[1]) - np.outer(target, target)
                skew = projection @ (0.5 * matrix - 0.5 * matrix.T) @ projection
                speed = np.linalg.norm(skew, 2)
                self.rotations.append(expm(skew * min(1.0, max_angle / (2 * max(speed, _EPS)))))

    def __call__(self, z, action):
        z = np.asarray(z, dtype=float)
        if z.shape != self.targets.shape[1:] or not np.all(np.isfinite(z)):
            raise ValueError("state must be a finite vector matching the targets")
        if not isinstance(action, (int, np.integer)) or not 0 <= action < len(self.targets):
            raise ValueError("action must be an index in range")
        target = self.targets[action]
        u = _unit(z)
        if not np.any(u):
            return target.copy()
        if self.rotations is not None:
            u = _unit(self.rotations[action] @ u)
        cosine = max(-1.0, min(1.0, float(target @ u)))
        tangent = target - cosine * u
        sine = math.sqrt(float(np.dot(tangent, tangent)))
        if sine < _EPS:
            return u
        theta = math.atan2(sine, cosine)
        next_theta = 2 * math.atan2(math.exp(-self.gamma) * math.sin(theta / 2), math.cos(theta / 2))
        budget = self.max_angle if self.rotations is None else self.max_angle / 2
        angle = min(budget, max(0.0, theta - next_theta))
        return _unit(math.cos(angle) * u + math.sin(angle) * tangent / sine)


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    if logits.ndim == 1 and (axis == -1 or axis == 0):
        if logits.size == 0 or not np.isfinite(logits).all():
            raise ValueError("softmax logits must be finite")
        hi = float(logits.max())
        lo = float(logits.min())
        if not math.isfinite(hi - lo):
            raise ValueError("softmax produced invalid probabilities")
        shifted = logits - hi
        exp = np.exp(shifted)
        total = float(exp.sum())
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("softmax produced invalid probabilities")
        probs = exp / total
        if not np.isfinite(probs).all() or float(probs.min()) < 0.0 or abs(float(probs.sum()) - 1.0) > 1e-5:
            raise ValueError("softmax produced invalid probabilities")
        return probs
    if not np.isfinite(logits).all():
        raise ValueError("softmax logits must be finite")
    with np.errstate(all="ignore"):
        shifted = logits - np.max(logits, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=axis, keepdims=True)
    if (not np.isfinite(shifted).all() or not np.isfinite(probs).all()
            or np.any(probs < 0) or np.any(np.abs(probs.sum(axis=axis) - 1.0) > 1e-5)):
        raise ValueError("softmax produced invalid probabilities")
    return probs


def _leaf_value(priors: np.ndarray, stability: float = 1.0, relational: float = 1.0) -> float:
    """Confidence discounted by Lyapunov and relational evidence, not correctness."""
    k = priors.shape[0]
    if k == 1:
        return stability * relational
    top2 = np.partition(priors, k - 2)[-2:]
    margin = float(top2[1] - top2[0])
    entropy = float(-np.sum(priors * np.log(priors + _EPS)))
    certainty = 1.0 - entropy / math.log(k)
    return (0.5 * margin + 0.5 * max(0.0, min(1.0, certainty))) * stability * relational


class LatentMctsNode:
    """One latent state with per-action prior, legal mask, visit count and Q-value.

    ``legal`` masks the actions admissible in this state (None: all of them):
    illegal priors are zeroed and the rest renormalised, and ``select`` never
    returns an illegal action. A node with no legal action is a dead end with value 0.
    """

    __slots__ = ("z", "priors", "legal", "visits", "q", "total_visits", "children", "value", "evidence", "unit_z")

    def __init__(self, z: np.ndarray, priors: np.ndarray, legal: Optional[np.ndarray] = None,
                 evidence: float = 1.0, unit_z: Optional[np.ndarray] = None) -> None:
        self.z = z
        self.unit_z = _unit(z) if unit_z is None else unit_z
        self.legal = legal
        if legal is not None:
            priors = np.where(legal, priors, 0.0)
            total = float(priors.sum())
            if total > 0:
                priors = priors / total
            if (not np.isfinite(priors).all() or np.any(priors < 0)
                    or legal.any() and abs(float(priors.sum()) - 1.0) > 1e-5):
                raise ValueError("node priors are invalid")
        self.priors = priors
        self.visits = np.zeros(priors.shape[0], dtype=np.int64)
        self.q = np.zeros(priors.shape[0], dtype=np.float64)
        self.total_visits = 0
        self.children: Dict[int, "LatentMctsNode"] = {}
        self.evidence = evidence
        self.value = self.leaf_value(evidence)
        if not math.isfinite(self.value):
            raise ValueError("node value is non-finite")

    def has_legal(self) -> bool:
        return self.legal is None or bool(self.legal.any())

    def leaf_value(self, stability: float = 1.0) -> float:
        if self.legal is None:
            return _leaf_value(self.priors, stability)
        if not self.legal.any():
            return 0.0
        return _leaf_value(self.priors[self.legal], stability)

    def select(self, c_puct: float) -> int:
        """PUCT over legal actions: argmax_a Q + c * P * sqrt(max(N_total, 1)) / (1 + N)."""
        # Hot path (called once per tree level per simulation): in-place updates keep
        # the same left-to-right arithmetic as Q + c*P*sqrt(N)/(1+N) with two temporaries.
        score = c_puct * self.priors
        score *= math.sqrt(max(self.total_visits, 1))
        score /= 1.0 + self.visits
        score += self.q
        if self.legal is not None:
            score[~self.legal] = -np.inf
        return int(score.argmax())


class _BoundedMemo:
    """Bounded FIFO memo for one ``plan`` call; keys are exact state bytes."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.table: Dict[Any, Any] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> Any:
        value = self.table.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def put(self, key: Any, value: Any) -> None:
        self.table[key] = value
        if len(self.table) > self.capacity:
            del self.table[next(iter(self.table))]


class _RootSequentialHalving:
    """Sequential Halving over the legal root actions (Karnin, Koren and Somekh, 2013).

    The candidate set is the top ``m`` legal actions by ``g + log P`` (Gumbel-Top-k,
    Danihelka et al. 2022). Each of ceil(log2 m) rounds gives every survivor
    ``max(1, budget // (rounds * |alive|))`` simulations and keeps the better
    ceil(|alive| / 2) by ``g + log P + (c_visit + max N) * c_scale * Q`` with
    min-max normalised Q. The last survivor takes the rest of the budget.
    """

    def __init__(self, root: "LatentMctsNode", budget: int, gumbel: np.ndarray,
                 max_considered: int, c_visit: float, c_scale: float) -> None:
        self.root, self.budget = root, budget
        self.c_visit, self.c_scale = c_visit, c_scale
        legal = np.flatnonzero(root.legal) if root.legal is not None else np.arange(root.priors.shape[0])
        with np.errstate(divide="ignore"):
            self.prior_score = gumbel + np.log(root.priors)
        self.m = max(1, min(max_considered, len(legal), budget))
        self.rounds = max(1, math.ceil(math.log2(self.m))) if self.m > 1 else 1
        order = sorted(legal.tolist(), key=lambda a: self.prior_score[a], reverse=True)
        self.alive: List[int] = order[: self.m]
        self.queue: List[int] = []
        self._fill_round()

    def _scores(self, actions: List[int]) -> Dict[int, float]:
        visits, q = self.root.visits, self.root.q
        seen = [a for a in range(len(visits)) if visits[a] > 0]
        q_lo = min((q[a] for a in seen), default=0.0)
        q_hi = max((q[a] for a in seen), default=0.0)
        sigma = (self.c_visit + int(visits.max())) * self.c_scale
        out = {}
        for a in actions:
            norm = (q[a] - q_lo) / (q_hi - q_lo) if (visits[a] > 0 and q_hi > q_lo) else 0.0
            out[a] = float(self.prior_score[a] + sigma * norm)
        return out

    def _fill_round(self) -> None:
        per = self.budget if len(self.alive) == 1 else max(1, self.budget // (self.rounds * len(self.alive)))
        self.queue = [a for _ in range(per) for a in self.alive]
        self.queue.reverse()

    def next_action(self) -> int:
        while not self.queue:
            if len(self.alive) > 1:
                scores = self._scores(self.alive)
                ranked = sorted(self.alive, key=lambda a: scores[a], reverse=True)
                self.alive = ranked[: math.ceil(len(self.alive) / 2)]
            self._fill_round()
        return self.queue.pop()

    def resolved(self) -> Optional[str]:
        if len(self.alive) == 1 and self.root.visits[self.alive[0]] > 0:
            return "sequential_halving_resolved"
        return None

    def final_choice(self, fallback: int) -> int:
        visited = [a for a in self.alive if self.root.visits[a] > 0]
        if not visited:
            return fallback
        scores = self._scores(visited)
        return max(visited, key=lambda a: scores[a])


def _count_dominance(root: "LatentMctsNode", remaining: int) -> Optional[str]:
    """Stop reason when the root visit leader can no longer be caught.

    The final pick sorts by visits first, so a visit lead larger than the remaining
    simulations fixes the decision. The leader's Q must not be lower (conservative).
    Unvisited actions need no guard here: nothing in this planner falsifies a root
    action, so no later simulation can remove the leader.
    """
    legal = np.flatnonzero(root.legal) if root.legal is not None else np.arange(root.visits.shape[0])
    order = sorted(legal.tolist(), key=lambda a: (root.visits[a], root.q[a]), reverse=True)
    lead = order[0]
    second = order[1] if len(order) > 1 else None
    second_visits = root.visits[second] if second is not None else 0
    if root.visits[lead] - second_visits <= remaining:
        return None
    if second is not None and root.q[lead] < root.q[second]:
        return None
    return "count_dominance"


class LatentMctsPlanner:
    """Zero-token MCTS over ETF candidate embeddings.

    Opt-in options (defaults keep plain PUCT over the full budget):

    * ``search_strategy="sequential_halving"``: root budget by Sequential Halving;
      the returned action is the best survivor by ``g + log P + sigma(Q)``.
      ``gumbel_scale`` > 0 samples the Gumbel-Top-k set (seeded by ``seed``).
    * ``early_stop=True``: stop when the root visit leader leads the runner-up by
      more than the remaining simulations (PUCT), or once one action survives
      Sequential Halving. ``info["simulations_run"]`` reports the work done.
    * ``transposition_cache=True``: memoize ``transition_fn``, ``legal_actions_fn``
      and the prior projection per exact state bytes. Only for deterministic,
      side-effect-free functions.
    """

    def __init__(
        self,
        num_simulations: int = 100,
        c_puct: float = 1.414,
        temperature: float = 0.1,
        max_depth: int = 6,
        damping: float = 0.3,
        search_strategy: str = "puct",
        early_stop: bool = False,
        transposition_cache: bool = False,
        transposition_cache_size: int = 100_000,
        gumbel_scale: float = 0.0,
        sh_max_considered: int = 16,
        sh_c_visit: float = 50.0,
        sh_c_scale: float = 1.0,
        seed: int = 0,
    ) -> None:
        if num_simulations < 1:
            raise ValueError("num_simulations must be >= 1")
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and > 0")
        if not np.isfinite(c_puct) or c_puct < 0:
            raise ValueError("c_puct must be finite and non-negative")
        if c_puct > np.finfo(np.float64).max / (math.sqrt(num_simulations) + 1.0):
            raise ValueError("c_puct would overflow the PUCT score")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if not 0.0 < damping <= 1.0:
            raise ValueError("damping must be in (0, 1]")
        if search_strategy not in SEARCH_STRATEGIES:
            raise ValueError(f"search_strategy must be one of {SEARCH_STRATEGIES}, got {search_strategy!r}")
        if transposition_cache_size < 1:
            raise ValueError("transposition_cache_size must be >= 1")
        if sh_max_considered < 2:
            raise ValueError("sh_max_considered must be >= 2")
        for name, val in (("gumbel_scale", gumbel_scale), ("sh_c_visit", sh_c_visit), ("sh_c_scale", sh_c_scale)):
            if not np.isfinite(val) or val < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        self.search_strategy = search_strategy
        self.early_stop = bool(early_stop)
        self.transposition_cache = bool(transposition_cache)
        self.transposition_cache_size = int(transposition_cache_size)
        self.gumbel_scale = float(gumbel_scale)
        self.sh_max_considered = int(sh_max_considered)
        self.sh_c_visit = float(sh_c_visit)
        self.sh_c_scale = float(sh_c_scale)
        self.rng = np.random.default_rng(seed)
        self.num_simulations = int(num_simulations)
        self.c_puct = float(c_puct)
        self.temperature = float(temperature)
        self.max_depth = int(max_depth)
        self.damping = float(damping)
        self.last_root: Optional[LatentMctsNode] = None

    def _project(self, z: np.ndarray, embeddings: np.ndarray, _in_errstate: bool = False) -> np.ndarray:
        if _in_errstate:
            logits = embeddings @ z
            scaled = logits / self.temperature
        else:
            with np.errstate(all="ignore"):
                logits = embeddings @ z
                scaled = logits / self.temperature
        if not np.isfinite(logits).all() or not np.isfinite(scaled).all():
            raise ValueError("projection or temperature scaling produced non-finite values")
        return _softmax(scaled)

    def _default_transition(self, embeddings: np.ndarray) -> Callable[[np.ndarray, int], np.ndarray]:
        return ContractiveHamiltonianTransition(embeddings, gamma=self.damping)

    def plan(
        self,
        initial_state: np.ndarray,
        candidates: Sequence[str],
        candidate_embeddings: np.ndarray,
        transition_fn: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        relational_constraint: Optional[Callable[[np.ndarray, np.ndarray, int], float]] = None,
        legal_actions_fn: Optional[Callable[[np.ndarray], Sequence[int]]] = None,
    ) -> Tuple[int, np.ndarray, Dict[str, Any]]:
        """Return ``(best_action_idx, search_policy_probs, info)``.

        relational_constraint(root_state, next_state, action) supplies external
        constraint satisfaction in [0,1]. Without it, only geometric consistency
        with the observation is checked; planning does not establish correctness.
        Violations persist down the branch so later confidence cannot erase them.

        legal_actions_fn(z) returns the candidate indices admissible in latent state
        ``z``; it is called for every node, and selection and transitions only use
        those indices. Without it every candidate is admissible in every state, and
        ``info["action_mask_source"]`` says so. No legal action at the root raises.
        """
        t0 = time.perf_counter()
        z0 = np.asarray(initial_state, dtype=np.float64).reshape(-1)
        emb = np.asarray(candidate_embeddings, dtype=np.float64)
        k = len(candidates)
        if k == 0:
            raise ValueError("candidates must not be empty")
        if emb.ndim != 2 or emb.shape[0] != k or emb.shape[1] != z0.shape[0]:
            raise ValueError(
                f"candidate_embeddings must have shape ({k}, {z0.shape[0]}), got {emb.shape}"
            )

        if not np.all(np.isfinite(z0)) or not np.all(np.isfinite(emb)):
            raise ValueError("states and embeddings must be finite")
        if np.any(np.max(np.abs(emb), axis=1) == 0):
            raise ValueError("candidate embeddings must be nonzero")
        targets = np.stack([_unit(row) for row in emb])
        root_unit = _unit(z0)
        step = transition_fn or self._default_transition(emb)
        calls = {"transition": 0, "legal": 0}
        memo = None
        if self.transposition_cache:
            memo = {name: _BoundedMemo(self.transposition_cache_size) for name in ("transition", "legal", "projection")}

        def raw_legal_at(z: np.ndarray) -> Optional[np.ndarray]:
            calls["legal"] += 1
            mask = np.zeros(k, dtype=bool)
            for idx in legal_actions_fn(z.copy()):
                if not isinstance(idx, (int, np.integer)) or isinstance(idx, bool) or not 0 <= idx < k:
                    raise ValueError(f"legal_actions_fn returned an invalid candidate index {idx!r}")
                mask[idx] = True
            return mask

        def legal_at(z: np.ndarray) -> Optional[np.ndarray]:
            if legal_actions_fn is None:
                return None
            if memo is None:
                return raw_legal_at(z)
            key = z.tobytes()
            hit = memo["legal"].get(key)
            if hit is None:
                hit = raw_legal_at(z)
                memo["legal"].put(key, hit)
            return hit.copy()

        def project(z: np.ndarray) -> np.ndarray:
            if memo is None:
                return self._project(z, emb, True)
            key = z.tobytes()
            hit = memo["projection"].get(key)
            if hit is None:
                hit = self._project(z, emb, True)
                memo["projection"].put(key, hit)
            if not np.isfinite(hit).all() or np.any(hit < 0) or abs(float(hit.sum()) - 1.0) > 1e-5:
                raise ValueError("cached projection is invalid")
            return hit.copy()

        def transition(z: np.ndarray, a: int) -> np.ndarray:
            if memo is not None:
                hit = memo["transition"].get((z.tobytes(), a))
                if hit is not None:
                    if not np.isfinite(hit).all():
                        raise ValueError("cached transition is non-finite")
                    return hit.copy()
            calls["transition"] += 1
            z_next = np.asarray(step(z, a), dtype=np.float64).reshape(-1)
            if z_next.shape != z0.shape:
                raise ValueError(
                    f"transition_fn must return shape {z0.shape}, got {z_next.shape}"
                )
            if not np.all(np.isfinite(z_next)):
                raise ValueError("transition_fn must return finite values")
            if memo is not None:
                memo["transition"].put((z.tobytes(), a), z_next.copy())
            return z_next

        with np.errstate(all="ignore"):
            root = LatentMctsNode(z0, project(z0), legal_at(z0))
            if not root.has_legal():
                raise ValueError("no candidate is legal at the root state")
            max_depth_seen = 0
            nodes = 1
            halving = None
            if self.search_strategy == "sequential_halving":
                gumbel = (self.gumbel_scale * self.rng.gumbel(size=k) if self.gumbel_scale > 0.0
                          else np.zeros(k))
                halving = _RootSequentialHalving(root, self.num_simulations, gumbel, self.sh_max_considered,
                                                 self.sh_c_visit, self.sh_c_scale)
            simulations_run = 0
            early_stop_reason: Optional[str] = None

            for _ in range(self.num_simulations):
                # The visit lead can exceed the remaining budget only after half of it.
                if self.early_stop and (halving is not None or 2 * simulations_run > self.num_simulations):
                    early_stop_reason = (halving.resolved() if halving is not None
                                         else _count_dominance(root, self.num_simulations - simulations_run))
                    if early_stop_reason is not None:
                        break
                simulations_run += 1
                node = root
                path = []
                depth = 0
                value = node.value
                while depth < self.max_depth and node.has_legal():
                    a = halving.next_action() if (halving is not None and node is root) else node.select(self.c_puct)
                    path.append((node, a))
                    depth += 1
                    child = node.children.get(a)
                    if child is None:
                        z_next = transition(node.z, a)
                        child_unit = _unit(z_next)
                        before = float(targets[a] @ node.unit_z)
                        after = float(targets[a] @ child_unit)
                        if not math.isfinite(before) or not math.isfinite(after):
                            raise ValueError("transition geometry is non-finite")
                        stability = math.exp(-max(0.0, before - after))
                        diff = child_unit - root_unit
                        relational = math.exp(-0.5 * math.sqrt(float(np.dot(diff, diff))))
                        if not math.isfinite(stability) or not math.isfinite(relational):
                            raise ValueError("transition evidence is non-finite")
                        if relational_constraint is not None:
                            score = float(relational_constraint(z0.copy(), z_next.copy(), a))
                            if not np.isfinite(score) or not 0 <= score <= 1:
                                raise ValueError("relational_constraint must return a finite score in [0,1]")
                            relational *= score
                        if not math.isfinite(relational):
                            raise ValueError("relational evidence is non-finite")
                        child = LatentMctsNode(z_next, project(z_next), legal_at(z_next),
                                               evidence=min(node.evidence, stability * relational),
                                               unit_z=child_unit)
                        node.children[a] = child
                        nodes += 1
                        value = child.value
                        break
                    node = child
                    value = node.value
                max_depth_seen = max(max_depth_seen, depth)

                for parent, a in path:
                    parent.visits[a] += 1
                    parent.total_visits += 1
                    updated_q = parent.q[a] + (value - parent.q[a]) / parent.visits[a]
                    if not math.isfinite(updated_q):
                        raise ValueError("Q update is non-finite")
                    parent.q[a] = updated_q

        self.last_root = root
        visits = root.visits
        if visits.sum() <= 0:
            raise ValueError("root has no visits")
        policy = visits / float(visits.sum())
        if (not np.isfinite(policy).all() or np.any(policy < 0)
                or abs(float(policy.sum()) - 1.0) > 1e-5
                or not np.isfinite(root.q).all()):
            raise ValueError("final policy or Q values are invalid")
        # Ties (e.g. 1 simulation) break on Q, then prior.
        best = int(np.lexsort((root.priors, root.q, visits))[-1])
        if halving is not None:
            best = halving.final_choice(best)
        map_action = int(np.argmax(root.priors))
        info: Dict[str, Any] = {
            "best_candidate": candidates[best],
            "map_action": map_action,
            "agrees_with_map": best == map_action,
            "root_priors": root.priors.copy(),
            "root_q": root.q.copy(),
            "root_visits": visits.copy(),
            "root_legal": np.ones(k, dtype=bool) if root.legal is None else root.legal.copy(),
            "action_mask_source": "legal_actions_fn" if legal_actions_fn is not None else "all_candidates_every_state",
            "num_simulations": self.num_simulations,
            "search_strategy": self.search_strategy,
            "early_stopped": early_stop_reason is not None,
            "early_stop_reason": early_stop_reason,
            "simulations_run": simulations_run,
            "transposition_cache": memo is not None,
            "transition_calls": calls["transition"],
            "legal_calls": calls["legal"],
            "cache_hits": 0 if memo is None else sum(m.hits for m in memo.values()),
            "transition_cache_hits": 0 if memo is None else memo["transition"].hits,
            "legal_cache_hits": 0 if memo is None else memo["legal"].hits,
            "projection_cache_hits": 0 if memo is None else memo["projection"].hits,
            "cache_misses": 0 if memo is None else sum(m.misses for m in memo.values()),
            "nodes": nodes,
            "max_depth": max_depth_seen,
            "tokens_generated": 0,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }
        return best, policy, info
