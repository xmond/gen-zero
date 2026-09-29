"""Vectorized 0-token latent MCTS: batch transitions and parallel tree instances.

This module removes the per-node Python loop that makes
:class:`gen_zero.causal.latent_mcts.LatentMctsPlanner` serialize on the CPU under
concurrent batch evaluation. Two pieces are vectorized in pure NumPy:

1. ``BatchContractiveHamiltonianTransition`` evaluates the spherical Hamiltonian
   flow ``dz/dt = B_a z + gamma*(e_a - (e_a . z) z)`` for a *batch* of states and
   *all* actions at once: ``(B, dim) -> (B, num_actions, dim)``. It reproduces
   :class:`ContractiveHamiltonianTransition(z, action)`` element-wise (same
   targets, same rotations, same geodesic step), only batched.

2. ``BatchLatentMctsPlanner`` runs ``B`` independent MCTS instances in lock-step.
   Each simulation advances every still-active instance by one PUCT step using
   array gathers, so the inner loop runs ``num_simulations`` vectorized iterations
   instead of ``B * num_simulations`` Python iterations. Because the per-instance
   simulation order is identical to ``LatentMctsPlanner.plan``, a ``B == 1`` batch
   call reproduces the single-instance result (best action and root policy) within
   numerical tolerance. The parity holds for ``LatentMctsPlanner.plan`` without
   ``legal_actions_fn``: this batch planner has no per-state legal mask and treats
   every candidate as admissible in every state.

The transition flow keeps ``||z||_2 = 1`` (norm conservation) and never increases
``V(z) = 1 - e_a . z`` (Lyapunov dissipation); the unit tests check both.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .latent_mcts import (
    ContractiveHamiltonianTransition,
    LatentMctsPlanner,
    _leaf_value,
    _softmax,
    _unit,
)

__all__ = [
    "BatchContractiveHamiltonianTransition",
    "BatchLatentMctsPlanner",
    "benchmark_batch_vs_sequential",
]

_EPS = 1e-12


def _unit_batch(z: np.ndarray) -> np.ndarray:
    """Row-wise unit normalization over the last axis.

    Mirrors :func:`gen_zero.causal.latent_mcts._unit` for the common case
    (finite nonzero rows). Zero rows stay zero; rows with a non-finite norm are
    scaled by their largest finite magnitude before normalization, matching the
    scalar fallback's intent of avoiding overflow/underflow.
    """
    z = np.asarray(z, dtype=np.float64)
    norm = np.linalg.norm(z, axis=-1, keepdims=True)
    finite_pos = np.isfinite(norm) & (norm > 0.0)
    safe = np.where(finite_pos, norm, 1.0)
    out = np.where(finite_pos, z / safe, 0.0)
    # Non-finite-norm fallback: scale down by the largest finite magnitude.
    bad = ~finite_pos
    if np.any(bad):
        scale = np.max(np.abs(np.where(np.isfinite(z), z, 0.0)),
                       axis=-1, keepdims=True)
        scale = np.where(scale == 0.0, 1.0, scale)
        scaled = z / scale
        s_norm = np.linalg.norm(scaled, axis=-1, keepdims=True)
        s_norm = np.where(s_norm > 0.0, s_norm, 1.0)
        out = np.where(bad, scaled / s_norm, out)
        zero_rows = bad & (np.max(np.abs(np.where(np.isfinite(z), z, 0.0)),
                                  axis=-1, keepdims=True) == 0.0)
        out = np.where(zero_rows, 0.0, out)
    return out


class BatchContractiveHamiltonianTransition:
    """Vectorized split spherical flow for a batch of states and all actions.

    Constructed from the same arguments as
    :class:`ContractiveHamiltonianTransition`; internally it builds a scalar
    transition to reuse its target normalization and Hamiltonian rotations, then
    applies the same geodesic step to every ``(state, action)`` pair in a single
    broadcast.

    ``transition_all(states)`` returns ``(B, num_actions, dim)`` where entry
    ``[i, a]`` equals ``ContractiveHamiltonianTransition(states[i], a)``. This is
    the mandated batch flow ``dz/dt = B_a z + gamma(e_a - (e_a . z) z)`` applied
    across a batch: shape ``(B, dim) -> (B, num_actions, dim)``.
    """

    def __init__(self, embeddings, gamma=0.3, generators=None, max_angle=0.3):
        # Reuse the scalar class so targets and rotations are bit-for-bit the
        # same as a ContractiveHamiltonianTransition built from the same args.
        scalar = ContractiveHamiltonianTransition(
            embeddings, gamma=gamma, generators=generators, max_angle=max_angle
        )
        self.targets = scalar.targets
        self.gamma = scalar.gamma
        self.max_angle = scalar.max_angle
        if scalar.rotations is not None:
            self.rotations = np.stack(scalar.rotations)
        else:
            self.rotations = None
        self.num_actions = self.targets.shape[0]
        self.dim = self.targets.shape[1]

    def transition_all(self, states: np.ndarray) -> np.ndarray:
        """`(B, dim) -> (B, num_actions, dim)`: next state for every action."""
        states = np.asarray(states, dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != self.dim:
            raise ValueError(
                f"states must have shape (B, {self.dim}), got {states.shape}"
            )
        if not np.all(np.isfinite(states)):
            raise ValueError("states must be finite")
        B = states.shape[0]
        k = self.num_actions
        targets = self.targets  # (k, dim)

        u_state = _unit_batch(states)  # (B, dim)
        zero_state = ~np.any(u_state, axis=1)  # (B,)

        # Broadcast the state to all actions: (B, k, dim).
        u = np.broadcast_to(u_state[:, None, :], (B, k, self.dim)).copy()

        if self.rotations is not None:
            # rotations: (k, dim, dim); u: (B, k, dim) -> rotate each action.
            u = _unit_batch(np.einsum("kde,Bke->Bkd", self.rotations, u))

        # cosine = target . u, clipped to [-1, 1].
        cosine = np.clip(np.einsum("kd,Bkd->Bk", targets, u), -1.0, 1.0)
        # tangent = target - cosine * u, in the tangent plane at u.
        tangent = targets[None, :, :] - cosine[:, :, None] * u
        sine = np.linalg.norm(tangent, axis=-1)  # (B, k)

        # Geodesic step: same closed form as the scalar transition.
        theta = np.arctan2(sine, cosine)
        next_theta = 2.0 * np.arctan2(
            np.exp(-self.gamma) * np.sin(theta / 2.0), np.cos(theta / 2.0)
        )
        budget = self.max_angle if self.rotations is None else self.max_angle / 2.0
        angle = np.minimum(budget, np.maximum(0.0, theta - next_theta))

        sine_safe = np.where(sine < _EPS, 1.0, sine)
        raw = np.cos(angle)[:, :, None] * u + \
            (np.sin(angle) / sine_safe)[:, :, None] * tangent
        # Where u is already aligned with the target (sine < eps), keep u.
        aligned = sine < _EPS
        result = np.where(aligned[:, :, None], u, _unit_batch(raw))

        # Zero-input states map to the target, matching the scalar branch.
        if np.any(zero_state):
            result[zero_state, :, :] = targets[None, :, :]
        return result

    def transition_indexed(
        self, states: np.ndarray, actions: np.ndarray
    ) -> np.ndarray:
        """`(B, dim), (B,) -> (B, dim)`: gather one action per state.

        Equivalent to ``transition_all(states)[arange(B), actions]``.
        """
        states = np.asarray(states, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.int64)
        all_next = self.transition_all(states)  # (B, k, dim)
        B = states.shape[0]
        return all_next[np.arange(B), actions].copy()

    # Allow callable use mirroring the scalar interface on a single state.
    def __call__(self, z, action):
        return self.transition_indexed(
            np.asarray(z, dtype=np.float64)[None, :], np.array([action], dtype=np.int64)
        )[0]


class BatchLatentMctsPlanner:
    """Zero-token MCTS over ETF embeddings, run for ``B`` instances at once.

    Each instance owns an independent search tree stored in a fixed-capacity
    flat array ``[B, capacity, ...]`` (capacity = ``num_simulations + 1``). A
    simulation advances every active instance one PUCT step via array gathers;
    instances that expand a new leaf become inactive, the rest descend. This
    keeps the per-instance simulation order identical to
    :class:`LatentMctsPlanner`, so ``B == 1`` reproduces the single-instance
    trajectory within numerical tolerance, while the inner loop runs
    ``num_simulations`` vectorized iterations instead of ``B * num_simulations``
    Python iterations.

    The default (no ``transition_fn``) uses
    :class:`BatchContractiveHamiltonianTransition`, which is the fast vectorized
    path. A caller-supplied scalar ``transition_fn`` falls back to running
    :class:`LatentMctsPlanner` per instance (correct, not faster) so the batch
    entry point stays a drop-in replacement.
    """

    def __init__(
        self,
        num_simulations: int = 100,
        c_puct: float = 1.414,
        temperature: float = 0.1,
        max_depth: int = 6,
        damping: float = 0.3,
        use_compact: bool = False,
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
        self.use_compact = bool(use_compact)

    def _compact_indices(self, mask: np.ndarray) -> np.ndarray:
        """Return the indices where ``mask`` is True, as an int64 array.

        When :attr:`use_compact` is False (the default), this returns
        ``np.nonzero(mask)[0]`` — byte-identical to the existing planner
        behavior. When ``use_compact`` is True, it routes through
        :func:`gen_zero.kernels.compact_active` (forced to the CPU backend,
        since the planner runs in NumPy), which on CPU is exactly
        ``np.flatnonzero(mask)`` ≡ ``np.nonzero(mask)[0]``. The two paths are
        therefore behavior-preserving.

        The compact kernel is lazy-imported inside this method so that a
        missing ``gen_zero.kernels`` package (or a missing triton/torch dep)
        can never break the planner: on ImportError the helper falls back to
        ``np.nonzero(mask)[0]``.
        """
        if not self.use_compact:
            return np.nonzero(mask)[0]
        try:
            from gen_zero.kernels import compact_active
        except ImportError:
            return np.nonzero(mask)[0]
        idx, _ = compact_active(mask, backend="cpu")
        return np.asarray(idx, dtype=np.int64)

    def _project(self, states: np.ndarray, emb: np.ndarray) -> np.ndarray:
        """`(B, dim) -> (B, k)`: softmax priors for each state."""
        logits = states @ emb.T / self.temperature  # (B, k)
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def plan(
        self,
        initial_states: np.ndarray,
        candidates: Sequence[str],
        candidate_embeddings: np.ndarray,
        transition_fn: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        relational_constraint: Optional[
            Callable[[np.ndarray, np.ndarray, int], float]
        ] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Return ``(best_action_idx (B,), policies (B, k), info)``.

        ``initial_states`` is shape ``(B, dim)``. See :class:`LatentMctsPlanner`
        for the scalar semantics this mirrors per instance.
        """
        t0 = time.perf_counter()
        states = np.asarray(initial_states, dtype=np.float64)
        if states.ndim != 2:
            raise ValueError("initial_states must have shape (B, dim)")
        B, dim = states.shape
        emb = np.asarray(candidate_embeddings, dtype=np.float64)
        k = len(candidates)
        if k == 0:
            raise ValueError("candidates must not be empty")
        if emb.ndim != 2 or emb.shape[0] != k or emb.shape[1] != dim:
            raise ValueError(
                f"candidate_embeddings must have shape ({k}, {dim}), got {emb.shape}"
            )
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(emb)):
            raise ValueError("states and embeddings must be finite")
        if np.any(np.max(np.abs(emb), axis=1) == 0):
            raise ValueError("candidate embeddings must be nonzero")
        targets = np.stack([_unit(row) for row in emb])

        # Fallback: a caller-supplied scalar transition (or any custom fn) is run
        # per instance via the scalar planner. Correct, not the fast path.
        if transition_fn is not None:
            return self._plan_sequential_fallback(
                states, candidates, emb, transition_fn, relational_constraint, t0
            )

        batch_step = BatchContractiveHamiltonianTransition(
            emb, gamma=self.damping
        )
        return self._plan_vectorized(
            states, candidates, emb, targets, batch_step, relational_constraint, t0
        )

    def _plan_vectorized(
        self,
        states: np.ndarray,
        candidates: Sequence[str],
        emb: np.ndarray,
        targets: np.ndarray,
        batch_step: BatchContractiveHamiltonianTransition,
        relational_constraint: Optional[Callable],
        t0: float,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        B, dim = states.shape
        k = len(candidates)
        nsims = self.num_simulations
        max_depth = self.max_depth
        capacity = nsims + 2  # Node 0 is sentinel, Node 1 is root, up to nsims new nodes.
        # Flat per-instance tree storage. Node 0 is the "no child" sentinel.
        node_z = np.zeros((B, capacity, dim))
        node_unit = np.zeros((B, capacity, dim))
        node_priors = np.zeros((B, capacity, k))
        node_visits = np.zeros((B, capacity, k), dtype=np.int64)
        node_q = np.zeros((B, capacity, k))
        node_total = np.zeros((B, capacity), dtype=np.int64)
        node_value = np.zeros((B, capacity))
        node_evidence = np.ones((B, capacity))
        children = np.zeros((B, capacity, k), dtype=np.int64)
        node_count = np.full(B, 2, dtype=np.int64)  # root is node 1, next new node is 2

        # Initialize root (node 1) for every instance.
        root_unit = _unit_batch(states)  # (B, dim)
        root_priors = self._project(states, emb)  # (B, k)
        node_z[:, 1, :] = states
        node_unit[:, 1, :] = root_unit
        node_priors[:, 1, :] = root_priors
        node_value[:, 1] = _leaf_value_batch(root_priors, np.ones(B), np.ones(B))

        arangeB = np.arange(B)
        for _ in range(nsims):
            current = np.ones(B, dtype=np.int64)
            value = node_value[arangeB, current].copy()
            active = np.ones(B, dtype=bool)
            path_nodes = np.zeros((B, max_depth), dtype=np.int64)
            path_actions = np.zeros((B, max_depth), dtype=np.int64)
            path_len = np.zeros(B, dtype=np.int64)

            depth = 0
            while depth < max_depth and np.any(active):
                q = node_q[arangeB, current]            # (B, k)
                pri = node_priors[arangeB, current]     # (B, k)
                vis = node_visits[arangeB, current]     # (B, k)
                tot = node_total[arangeB, current]       # (B,)
                explore = self.c_puct * pri * np.sqrt(
                    np.maximum(tot, 1)
                )[:, None] / (1.0 + vis)
                action = np.argmax(q + explore, axis=1)  # (B,)

                # Record path for active instances at this depth.
                act_mask = active
                path_nodes[act_mask, depth] = current[act_mask]
                path_actions[act_mask, depth] = action[act_mask]
                path_len[act_mask] = depth + 1

                child = children[arangeB, current, action]  # (B,) 0 = unexpanded
                expand_mask = active & (child == 0)
                descend_mask = active & (child != 0)

                # --- Expansion: allocate new nodes for expanders. ---
                if np.any(expand_mask):
                    bi = self._compact_indices(expand_mask)
                    parent_idx = current[bi]
                    act_idx = action[bi]
                    new_id = node_count[bi].copy()
                    node_count[bi] = new_id + 1
                    z_parent = node_z[bi, parent_idx]                # (e, dim)
                    z_next = batch_step.transition_indexed(z_parent, act_idx)
                    if not np.all(np.isfinite(z_next)):
                        raise ValueError("transition_fn must return finite values")
                    u_next = _unit_batch(z_next)                     # (e, dim)
                    pri_next = self._project(z_next, emb)            # (e, k)

                    before = np.einsum("ed,ed->e",
                                       targets[act_idx], node_unit[bi, parent_idx])
                    after = np.einsum("ed,ed->e", targets[act_idx], u_next)
                    stability = np.exp(-np.maximum(0.0, before - after))
                    relational = np.exp(
                        -0.5 * np.linalg.norm(u_next - root_unit[bi], axis=1)
                    )
                    if relational_constraint is not None:
                        for j, idx in enumerate(bi):
                            score = float(relational_constraint(
                                states[idx].copy(), z_next[j].copy(), int(act_idx[j])
                            ))
                            if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                                raise ValueError(
                                    "relational_constraint must return a finite score in [0,1]"
                                )
                            relational[j] *= score
                    parent_ev = node_evidence[bi, parent_idx]
                    evidence = np.minimum(parent_ev, stability * relational)
                    leaf_val = _leaf_value_batch(pri_next, evidence, np.ones(len(bi)))

                    node_z[bi, new_id] = z_next
                    node_unit[bi, new_id] = u_next
                    node_priors[bi, new_id] = pri_next
                    node_evidence[bi, new_id] = evidence
                    node_value[bi, new_id] = leaf_val
                    children[bi, parent_idx, act_idx] = new_id
                    value[bi] = leaf_val
                    active[bi] = False

                # --- Descent: move to existing child. ---
                if np.any(descend_mask):
                    bi = self._compact_indices(descend_mask)
                    child_idx = child[bi]
                    current[bi] = child_idx
                    value[bi] = node_value[bi, child_idx]
                depth += 1

            # --- Backup: update visits and Q along each instance's path. ---
            for s in range(max_depth):
                m = path_len > s
                if not np.any(m):
                    continue
                bi = np.nonzero(m)[0]
                par = path_nodes[bi, s]
                act = path_actions[bi, s]
                node_visits[bi, par, act] += 1
                node_total[bi, par] += 1
                old_q = node_q[bi, par, act]
                v = value[bi]
                new_visits = node_visits[bi, par, act]
                node_q[bi, par, act] = old_q + (v - old_q) / new_visits

        root_visits = node_visits[:, 1, :]            # (B, k)
        root_q = node_q[:, 1, :]
        root_priors_final = node_priors[:, 1, :]
        totals = root_visits.sum(axis=1, keepdims=True)
        policy = root_visits / np.where(totals > 0, totals, 1.0)

        best = np.empty(B, dtype=np.int64)
        map_action = np.argmax(root_priors_final, axis=1)
        for i in range(B):
            best[i] = np.lexsort(
                (root_priors_final[i], root_q[i], root_visits[i])
            )[-1]

        info: Dict[str, Any] = {
            "best_candidate": [candidates[b] for b in best],
            "map_action": map_action,
            "agrees_with_map": best == map_action,
            "root_priors": root_priors_final.copy(),
            "root_q": root_q.copy(),
            "root_visits": root_visits.copy(),
            "num_simulations": nsims,
            "nodes": (node_count - 1).copy(),
            "max_depth": max_depth,
            "tokens_generated": 0,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }
        return best, policy, info

    def _plan_sequential_fallback(
        self,
        states: np.ndarray,
        candidates: Sequence[str],
        emb: np.ndarray,
        transition_fn: Callable,
        relational_constraint: Optional[Callable],
        t0: float,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        scalar = LatentMctsPlanner(
            num_simulations=self.num_simulations,
            c_puct=self.c_puct,
            temperature=self.temperature,
            max_depth=self.max_depth,
            damping=self.damping,
        )
        B = states.shape[0]
        k = len(candidates)
        best = np.empty(B, dtype=np.int64)
        policy = np.empty((B, k))
        priors = []
        q = []
        vis = []
        nodes = np.empty(B, dtype=np.int64)
        max_depth_seen = 0
        for i in range(B):
            b, p, info_i = scalar.plan(
                states[i], candidates, emb,
                transition_fn=transition_fn,
                relational_constraint=relational_constraint,
            )
            best[i] = b
            policy[i] = p
            priors.append(info_i["root_priors"])
            q.append(info_i["root_q"])
            vis.append(info_i["root_visits"])
            nodes[i] = info_i["nodes"]
            max_depth_seen = max(max_depth_seen, info_i["max_depth"])
        map_action = np.argmax(np.array(priors), axis=1)
        info: Dict[str, Any] = {
            "best_candidate": [candidates[b] for b in best],
            "map_action": map_action,
            "agrees_with_map": best == map_action,
            "root_priors": np.array(priors),
            "root_q": np.array(q),
            "root_visits": np.array(vis),
            "num_simulations": self.num_simulations,
            "nodes": nodes,
            "max_depth": max_depth_seen,
            "tokens_generated": 0,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }
        return best, policy, info


def _leaf_value_batch(
    priors: np.ndarray, stability: np.ndarray, relational: np.ndarray
) -> np.ndarray:
    """Vectorized :func:`gen_zero.causal.latent_mcts._leaf_value`.

    ``priors`` is ``(B, k)``; ``stability`` and ``relational`` are ``(B,)``.
    Replicates the scalar computation, including ``np.partition`` semantics, so
    a ``B == 1`` batch call matches the scalar leaf value element-wise.
    """
    priors = np.asarray(priors, dtype=np.float64)
    B, k = priors.shape
    stability = np.asarray(stability, dtype=np.float64)
    relational = np.asarray(relational, dtype=np.float64)
    if k == 1:
        return stability * relational
    top2 = np.partition(priors, k - 2, axis=1)[:, -2:]   # (B, 2), unsorted
    margin = top2[:, 1] - top2[:, 0]
    entropy = -np.sum(priors * np.log(priors + _EPS), axis=1)
    certainty = 1.0 - entropy / float(np.log(k))
    certainty = np.maximum(0.0, np.minimum(1.0, certainty))
    base = 0.5 * margin + 0.5 * certainty
    return base * stability * relational


def benchmark_batch_vs_sequential(
    B: int = 64,
    num_actions: int = 6,
    dim: int = 16,
    num_simulations: int = 120,
    max_depth: int = 4,
    repeats: int = 5,
    seed: int = 0,
) -> Dict[str, Any]:
    """Time ``BatchLatentMctsPlanner`` vs ``B`` scalar ``LatentMctsPlanner`` calls.

    Returns per-method best-of-``repeats`` latency, throughput (instances/sec),
    and speedup. The batch inner loop runs ``num_simulations`` vectorized
    iterations; the sequential baseline runs ``B * num_simulations`` Python
    iterations, so the speedup grows with ``B``.
    """
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(num_actions, dim))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    states = rng.normal(size=(B, dim))
    candidates = [f"c{i}" for i in range(num_actions)]

    scalar = LatentMctsPlanner(
        num_simulations=num_simulations, max_depth=max_depth
    )
    batch = BatchLatentMctsPlanner(
        num_simulations=num_simulations, max_depth=max_depth
    )

    # Warm up.
    for i in range(min(B, 4)):
        scalar.plan(states[i], candidates, emb)
    batch.plan(states, candidates, emb)

    def time_scalar() -> float:
        t = time.perf_counter()
        for i in range(B):
            scalar.plan(states[i], candidates, emb)
        return time.perf_counter() - t

    def time_batch() -> float:
        t = time.perf_counter()
        batch.plan(states, candidates, emb)
        return time.perf_counter() - t

    seq_times = sorted(time_scalar() for _ in range(repeats))
    bat_times = sorted(time_batch() for _ in range(repeats))
    seq_best = seq_times[0]
    bat_best = bat_times[0]
    return {
        "B": B,
        "num_actions": num_actions,
        "dim": dim,
        "num_simulations": num_simulations,
        "max_depth": max_depth,
        "sequential_ms": seq_best * 1000.0,
        "batch_ms": bat_best * 1000.0,
        "sequential_throughput": B / seq_best,
        "batch_throughput": B / bat_best,
        "speedup": seq_best / bat_best,
    }
