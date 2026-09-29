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
"""

from __future__ import annotations

import time
import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .koopman_thinking import expm, fit_generator_from_derivatives

__all__ = ["LatentMctsNode", "LatentMctsPlanner", "fit_action_dynamics",
           "ContractiveHamiltonianTransition"]

_EPS = 1e-12


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
        sine = float(np.linalg.norm(tangent))
        if sine < _EPS:
            return u
        theta = math.atan2(sine, cosine)
        next_theta = 2 * math.atan2(math.exp(-self.gamma) * math.sin(theta / 2), math.cos(theta / 2))
        budget = self.max_angle if self.rotations is None else self.max_angle / 2
        angle = min(budget, max(0.0, theta - next_theta))
        return _unit(math.cos(angle) * u + math.sin(angle) * tangent / sine)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _leaf_value(priors: np.ndarray, stability: float = 1.0, relational: float = 1.0) -> float:
    """Confidence discounted by Lyapunov and relational evidence, not correctness."""
    k = priors.shape[0]
    if k == 1:
        return stability * relational
    top2 = np.partition(priors, k - 2)[-2:]
    margin = float(top2[1] - top2[0])
    entropy = float(-np.sum(priors * np.log(priors + _EPS)))
    certainty = 1.0 - entropy / float(np.log(k))
    return (0.5 * margin + 0.5 * max(0.0, min(1.0, certainty))) * stability * relational


class LatentMctsNode:
    """One latent state with per-action prior, legal mask, visit count and Q-value.

    ``legal`` masks the actions admissible in this state (None: all of them):
    illegal priors are zeroed and the rest renormalised, and ``select`` never
    returns an illegal action. A node with no legal action is a dead end with value 0.
    """

    __slots__ = ("z", "priors", "legal", "visits", "q", "total_visits", "children", "value", "evidence", "unit_z")

    def __init__(self, z: np.ndarray, priors: np.ndarray, legal: Optional[np.ndarray] = None) -> None:
        self.z = z
        self.unit_z = _unit(z)
        self.legal = legal
        if legal is not None:
            priors = np.where(legal, priors, 0.0)
            total = float(priors.sum())
            if total > 0:
                priors = priors / total
        self.priors = priors
        self.visits = np.zeros(priors.shape[0], dtype=np.int64)
        self.q = np.zeros(priors.shape[0], dtype=np.float64)
        self.total_visits = 0
        self.children: Dict[int, "LatentMctsNode"] = {}
        self.evidence = 1.0
        self.value = self.leaf_value()

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
        explore = c_puct * self.priors * np.sqrt(max(self.total_visits, 1)) / (1.0 + self.visits)
        score = self.q + explore
        if self.legal is not None:
            score = np.where(self.legal, score, -np.inf)
        return int(np.argmax(score))


class LatentMctsPlanner:
    """Zero-token MCTS over ETF candidate embeddings."""

    def __init__(
        self,
        num_simulations: int = 100,
        c_puct: float = 1.414,
        temperature: float = 0.1,
        max_depth: int = 6,
        damping: float = 0.3,
    ) -> None:
        if num_simulations < 1:
            raise ValueError("num_simulations must be >= 1")
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and > 0")
        if not np.isfinite(c_puct) or c_puct < 0:
            raise ValueError("c_puct must be finite and non-negative")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if not 0.0 < damping <= 1.0:
            raise ValueError("damping must be in (0, 1]")
        self.num_simulations = int(num_simulations)
        self.c_puct = float(c_puct)
        self.temperature = float(temperature)
        self.max_depth = int(max_depth)
        self.damping = float(damping)
        self.last_root: Optional[LatentMctsNode] = None

    def _project(self, z: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        return _softmax(embeddings @ z / self.temperature)

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

        def legal_at(z: np.ndarray) -> Optional[np.ndarray]:
            if legal_actions_fn is None:
                return None
            mask = np.zeros(k, dtype=bool)
            for idx in legal_actions_fn(z.copy()):
                if not isinstance(idx, (int, np.integer)) or isinstance(idx, bool) or not 0 <= idx < k:
                    raise ValueError(f"legal_actions_fn returned an invalid candidate index {idx!r}")
                mask[idx] = True
            return mask

        root = LatentMctsNode(z0, self._project(z0, emb), legal_at(z0))
        if not root.has_legal():
            raise ValueError("no candidate is legal at the root state")
        max_depth_seen = 0
        nodes = 1

        for _ in range(self.num_simulations):
            node = root
            path = []
            depth = 0
            value = node.value
            while depth < self.max_depth and node.has_legal():
                a = node.select(self.c_puct)
                path.append((node, a))
                depth += 1
                child = node.children.get(a)
                if child is None:
                    z_next = np.asarray(step(node.z, a), dtype=np.float64).reshape(-1)
                    if z_next.shape != z0.shape:
                        raise ValueError(
                            f"transition_fn must return shape {z0.shape}, got {z_next.shape}"
                        )
                    if not np.all(np.isfinite(z_next)):
                        raise ValueError("transition_fn must return finite values")
                    child = LatentMctsNode(z_next, self._project(z_next, emb), legal_at(z_next))
                    before = float(targets[a] @ node.unit_z)
                    after = float(targets[a] @ child.unit_z)
                    stability = float(np.exp(-max(0.0, before - after)))
                    relational = float(np.exp(-0.5 * np.linalg.norm(child.unit_z - root_unit)))
                    if relational_constraint is not None:
                        score = float(relational_constraint(z0.copy(), z_next.copy(), a))
                        if not np.isfinite(score) or not 0 <= score <= 1:
                            raise ValueError("relational_constraint must return a finite score in [0,1]")
                        relational *= score
                    child.evidence = min(node.evidence, stability * relational)
                    child.value = child.leaf_value(child.evidence)
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
                parent.q[a] += (value - parent.q[a]) / parent.visits[a]

        self.last_root = root
        visits = root.visits
        policy = visits / float(visits.sum())
        # Ties (e.g. 1 simulation) break on Q, then prior.
        best = int(np.lexsort((root.priors, root.q, visits))[-1])
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
            "nodes": nodes,
            "max_depth": max_depth_seen,
            "tokens_generated": 0,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }
        return best, policy, info
