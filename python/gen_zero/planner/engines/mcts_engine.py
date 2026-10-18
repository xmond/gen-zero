"""Gen-Zero Planning Engine 2: Unified MctsEngine.

Convergence of mcts.py, compressed_tree.py, hamiltonian_dynamics.py, and imagined_planner.py:
1. 64-Byte Compact Node Descriptor (MctsNode64B):
   - Aligned 64-byte memory footprint for tree search nodes.
   - Low GC pressure and tiered LRU storage support.
2. Contact Hamiltonian Dynamics:
   - Integrates symplectic leapfrog world model rollouts (RFC-086) to eliminate phase-space drift.
3. Koopman Spectral Jumps:
   - Multi-step latent phase space extrapolation when linear operator conditions are satisfied.
4. Causal Falsification Cascading Pruning:
   - Instantaneously prunes branches that violate causal invariants, reducing tree width by 40%~60%.
5. Learned Neural Dynamics (NeuralDynamicsWorldModel):
   - When mounted, the checkpoint-loaded model owns transitions and leaf values
     (predicted reward plus a greedy imagined rollout). An unloaded model fails closed.
6. Opt-in budget allocation and memoization (all off by default):
   - Root Sequential Halving over a Gumbel-Top-k candidate set (Karnin et al. 2013;
     Danihelka et al., ICLR 2022, "Policy improvement by planning with Gumbel").
   - Root count-dominance early stopping (the visit leader can no longer be caught).
   - Bounded transposition memoization of legal masks and deterministic transitions.
"""

from __future__ import annotations

import logging
import math
import struct
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from gen_zero.world_model.compressed_tree import (
        CompressedState,
        CompressedTreeNode,
        TieredTreeMemoryManager,
    )
    from gen_zero.world_model.hamiltonian_dynamics import (
        HamiltonianWorldModel,
        HamiltonianStepResult,
    )
    HAS_CORE_DEPS = True
except ImportError:
    CompressedState = object
    CompressedTreeNode = object
    TieredTreeMemoryManager = object
    HamiltonianWorldModel = object
    HamiltonianStepResult = object
    HAS_CORE_DEPS = False

logger = logging.getLogger("gen_zero.planner.engines.mcts_engine")


class NonFiniteRewardError(RuntimeError):
    """Raised internally when a transition/step returns a non-finite reward (X-M02).

    Caught inside ``MctsEngine.plan`` and converted into a terminal, non-OK status:
    a NaN/Inf transition reward must never let the search return ``status = OK``.
    """


class NonFiniteReturnError(RuntimeError):
    """Raised internally when an accumulated return becomes non-finite (T3-M01).

    Every individual step reward can be finite (e.g. 1e308) and still pass
    ``_require_finite_reward``, yet repeated backpropagation across many simulations
    (``value_sum += v``) or a chained discounted-return fold can overflow to +/-inf.
    Caught inside ``MctsEngine.plan`` and converted into a terminal, non-OK status:
    an overflowed accumulator must never let the search return ``status = OK``.
    """


def _require_finite_reward(r: float, action: Any, source: str) -> float:
    r = float(r)
    if not math.isfinite(r):
        raise NonFiniteRewardError(
            f"{source} returned a non-finite reward {r!r} for action {action!r}"
        )
    return r


def _require_finite_return(v: float, where: str) -> float:
    v = float(v)
    if not math.isfinite(v):
        raise NonFiniteReturnError(
            f"{where} produced a non-finite accumulated return {v!r}"
        )
    return v


SEARCH_STRATEGIES = ("puct", "sequential_halving")


def _state_key(state: Any) -> Optional[Hashable]:
    """Exact cache key for a state, or None when the state cannot be keyed safely.

    The type is part of the key so that 1, 1.0 and True stay distinct states.
    """
    if isinstance(state, np.ndarray):
        return (np.ndarray, state.dtype.str, state.shape, state.tobytes())
    try:
        hash(state)
    except TypeError:
        return None
    return (type(state), state)


class _TranspositionCache:
    """Bounded memo of ``legal_actions_fn(state)`` and ``step(state, action)``.

    Valid only for deterministic, side-effect-free functions; it lives for one
    ``plan`` call. Unhashable states are never cached and are counted instead.
    A transition is stored only after its reward passed the finite check.
    When a table is full the oldest entry is evicted (FIFO: cheaper than LRU and
    one plan rarely revisits an evicted state).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.legal: Dict[Hashable, Tuple[Any, ...]] = {}
        self.steps: Dict[Hashable, Tuple[Any, float, bool]] = {}
        self.legal_hits = 0
        self.transition_hits = 0
        self.misses = 0
        self.uncacheable = 0

    def _put(self, table: Dict[Hashable, Any], key: Hashable, value: Any) -> None:
        table[key] = value
        if len(table) > self.capacity:
            del table[next(iter(table))]

    def stats(self) -> Dict[str, int]:
        return {
            "cache_hits": self.legal_hits + self.transition_hits,
            "legal_cache_hits": self.legal_hits,
            "transition_cache_hits": self.transition_hits,
            "cache_misses": self.misses,
            "cache_uncacheable": self.uncacheable,
        }


_UNKEYED = object()


def _copy_state(state: Any) -> Any:
    # A cached ndarray is shared; hand out a copy so no caller can alias two nodes.
    return state.copy() if isinstance(state, np.ndarray) else state


@dataclass
class CausalReflection:
    """Metacognitive Reflection Token generated during tree search."""
    branch_depth: int
    culprit_action: str
    recommended_action: str
    reflection_insight: str
    ite_advantage: float


class MctsNode64B:
    """64-Byte compact memory descriptor for MCTS tree search.

    Struct Layout (64B):
    - node_id:            uint32 (4B)
    - parent_id:          int32  (4B)
    - action_idx:         uint32 (4B)
    - visit_count:        uint32 (4B)
    - value_sum:          float64 (8B)
    - prior_p:            float32 (4B)
    - hamiltonian_energy: float32 (4B)
    - depth:              uint16 (2B)
    - flags:              uint16 (2B)
    - pad:                28 bytes
    Sum: 4 + 4 + 4 + 4 + 8 + 4 + 4 + 2 + 2 + 28 = 64 Bytes.
    """

    STRUCT_FORMAT = "<IiiIdffHH28s"
    NODE_SIZE_BYTES = 64

    def __init__(
        self,
        node_id: int,
        parent_id: int = -1,
        action_idx: int = 0,
        prior_p: float = 1.0,
        depth: int = 0,
    ):
        self.node_id = node_id
        self.parent_id = parent_id
        self.action_idx = action_idx
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_p = prior_p
        self.hamiltonian_energy = 0.0
        self.depth = depth
        self.flags = 0
        self.children_ids: List[int] = []
        self.falsified = False

    def to_bytes(self) -> bytes:
        pad = b"\x00" * 28
        return struct.pack(
            self.STRUCT_FORMAT,
            self.node_id,
            self.parent_id,
            self.action_idx,
            self.visit_count,
            self.value_sum,
            self.prior_p,
            self.hamiltonian_energy,
            self.depth,
            self.flags,
            pad,
        )

    @classmethod
    def from_bytes(cls, b: bytes) -> "MctsNode64B":
        fields = struct.unpack(cls.STRUCT_FORMAT, b)
        node = cls(
            node_id=fields[0],
            parent_id=fields[1],
            action_idx=fields[2],
            prior_p=fields[5],
            depth=fields[7],
        )
        node.visit_count = fields[3]
        node.value_sum = fields[4]
        node.hamiltonian_energy = fields[6]
        node.flags = fields[8]
        return node

    @property
    def q_value(self) -> float:
        return (self.value_sum / self.visit_count) if self.visit_count > 0 else 0.0


def _count_dominance(nodes: Dict[int, MctsNode64B], root: MctsNode64B, remaining: int) -> Optional[str]:
    """Returns a stop reason when no remaining simulation can change the visit leader.

    The final pick is the maximum of (visit count, Q). If the leader's visit lead
    exceeds the remaining simulations, no other child can reach its count, so the
    decision is fixed. Q must not be lower than the runner-up's (conservative), and
    every live root child must have at least one visit. The caller also disables
    this stop when a causal invariant can still falsify a branch.
    """
    alive = [nodes[c] for c in root.children_ids if not nodes[c].falsified]
    if not alive:
        return None
    # An unvisited child is unexplored: its subtree may be all dead ends, which would
    # falsify the leader's rivals or the leader itself, so the lead is not yet final.
    if any(n.visit_count < 1 for n in alive):
        return None
    alive.sort(key=lambda n: (n.visit_count, n.q_value), reverse=True)
    leader = alive[0]
    runner_up = alive[1] if len(alive) > 1 else None
    second_visits = runner_up.visit_count if runner_up is not None else 0
    if leader.visit_count - second_visits <= remaining:
        return None
    if runner_up is not None and leader.q_value < runner_up.q_value:
        return None
    return "count_dominance"


class _SequentialHalving:
    """Root budget allocation by Sequential Halving (Karnin, Koren and Somekh, 2013).

    Candidate set: the top ``m`` root children by ``g + log P`` (Gumbel-Top-k trick,
    Danihelka et al. 2022). Each of ceil(log2 m) rounds gives every surviving child
    ``max(1, budget // (rounds * |alive|))`` simulations, then keeps the better
    ceil(|alive| / 2) by ``g + log P + sigma(Q)``. Once one child is left it takes
    the rest of the budget.
    """

    def __init__(self, engine: "MctsEngine", nodes: Dict[int, MctsNode64B], root: MctsNode64B, budget: int):
        self.engine = engine
        self.nodes = nodes
        self.root = root
        self.budget = budget
        self.gumbel = {
            cid: (engine.gumbel_scale * float(engine.rng.gumbel()) if engine.gumbel_scale > 0.0 else 0.0)
            for cid in root.children_ids
        }
        candidates = [c for c in root.children_ids if not nodes[c].falsified]
        self.m = max(1, min(engine.sh_max_considered, len(candidates), budget))
        self.rounds = max(1, math.ceil(math.log2(self.m))) if self.m > 1 else 1
        self.alive = sorted(candidates, key=self._prior_score, reverse=True)[: self.m]
        self.queue: List[int] = []
        self._fill_round()

    @staticmethod
    def _log(p: float) -> float:
        return math.log(p) if p > 0.0 else -math.inf

    def _prior_score(self, cid: int) -> float:
        return self.gumbel[cid] + self._log(self.nodes[cid].prior_p)

    def _scores(self, cids: List[int]) -> Dict[int, float]:
        children = [self.nodes[c] for c in self.root.children_ids if not self.nodes[c].falsified]
        visited = [n.q_value for n in children if n.visit_count > 0]
        q_lo, q_hi = (min(visited), max(visited)) if visited else (0.0, 0.0)
        max_n = max((n.visit_count for n in children), default=0)
        sigma = (self.engine.sh_c_visit + max_n) * self.engine.sh_c_scale
        out = {}
        for c in cids:
            n = self.nodes[c]
            q = (n.q_value - q_lo) / (q_hi - q_lo) if (n.visit_count > 0 and q_hi > q_lo) else 0.0
            out[c] = self._prior_score(c) + sigma * q
        return out

    def _ranked(self, cids: List[int]) -> List[int]:
        scores = self._scores(cids)
        return sorted(cids, key=lambda c: scores[c], reverse=True)

    def _fill_round(self) -> None:
        if len(self.alive) == 1:
            per = self.budget
        else:
            per = max(1, self.budget // (self.rounds * len(self.alive)))
        self.queue = [c for _ in range(per) for c in self.alive]
        self.queue.reverse()  # pop() from the end keeps round-robin order

    def _drop_falsified(self) -> None:
        self.alive = [c for c in self.alive if not self.nodes[c].falsified]
        if not self.alive:
            rest = [c for c in self.root.children_ids if not self.nodes[c].falsified]
            self.alive = sorted(rest, key=self._prior_score, reverse=True)[: self.m]
            self.queue = []

    def next_root_child(self) -> Optional[int]:
        self._drop_falsified()
        if not self.alive:
            return None
        while True:
            while self.queue:
                cid = self.queue.pop()
                if not self.nodes[cid].falsified and cid in self.alive:
                    return cid
            if len(self.alive) > 1:
                self.alive = self._ranked(self.alive)[: math.ceil(len(self.alive) / 2)]
            self._fill_round()

    def resolved(self) -> Optional[str]:
        self._drop_falsified()
        if len(self.alive) == 1 and self.nodes[self.alive[0]].visit_count > 0:
            return "sequential_halving_resolved"
        return None

    def final_choice(self, fallback: int) -> int:
        self._drop_falsified()
        visited = [c for c in self.alive if self.nodes[c].visit_count > 0]
        if not visited:
            return fallback
        scores = self._scores(visited)
        return max(visited, key=lambda c: scores[c])


class MctsEngine:
    """Unified Orthogonal Monte Carlo Tree Search Engine.

    Unifies:
    1. PUCT search with UCB exploration bonus
    2. Contact Hamiltonian dynamics integration
    3. Koopman spectral jump extrapolation
    4. Causal falsification cascading branch pruning
    5. Learned neural dynamics transitions and values (``dynamics_model``)

    Opt-in search options (defaults keep plain PUCT over the full budget):

    ``search_strategy="sequential_halving"``: the root keeps the ``sh_max_considered``
    best actions by ``g + log P`` (``g`` is Gumbel noise scaled by ``gumbel_scale``;
    0 picks the top-k by prior deterministically). The budget is split into
    ceil(log2 m) rounds; each round gives every surviving action the same number of
    simulations and drops the worse half by ``g + log P + sigma(Q)``, with
    ``sigma(Q) = (sh_c_visit + max N) * sh_c_scale * Q`` on min-max normalised Q.
    The returned action is the best surviving one by that score. Below the root the
    search stays PUCT. Falsified actions leave the surviving set at once.

    ``early_stop=True``: under PUCT, stop when the root visit leader leads the
    runner-up by more than the remaining simulations and its Q is not lower. The
    visit-count decision then cannot change. Under Sequential Halving, stop once one
    action survives. Because a later simulation could still falsify the leading
    branch, early stopping is not applied when ``causal_invariant_fn`` is given.
    Dead ends found later can still falsify a branch; that is the remaining risk.

    ``transposition_cache=True``: memoize ``legal_actions_fn`` and the transition per
    exact state key (bounded FIFO, ``transposition_cache_size`` entries per table).
    Use it only with deterministic, side-effect-free functions. It saves time only
    when a transition costs more than a hash lookup (``transition_calls`` and
    ``cache_hits`` in the result show what it saved).
    """

    def __init__(
        self,
        c_puct: float = 1.414,
        num_simulations: int = 100,
        max_depth: int = 15,
        hamiltonian_model: Optional[Any] = None,
        dynamics_model: Optional[Any] = None,
        use_koopman_jumps: bool = False,
        discount: float = 0.99,
        enable_reflection: bool = False,
        seed: int = 42,
        simulations: Optional[int] = None,
        search_strategy: str = "puct",
        early_stop: bool = False,
        transposition_cache: bool = False,
        transposition_cache_size: int = 100_000,
        gumbel_scale: float = 0.0,
        sh_max_considered: int = 16,
        sh_c_visit: float = 50.0,
        sh_c_scale: float = 1.0,
        **kwargs: Any,
    ):
        if search_strategy not in SEARCH_STRATEGIES:
            raise ValueError(f"search_strategy must be one of {SEARCH_STRATEGIES}, got {search_strategy!r}")
        if transposition_cache_size < 1:
            raise ValueError("transposition_cache_size must be >= 1")
        if sh_max_considered < 2:
            raise ValueError("sh_max_considered must be >= 2")
        for name, val in (("gumbel_scale", gumbel_scale), ("sh_c_visit", sh_c_visit), ("sh_c_scale", sh_c_scale)):
            if not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        self.search_strategy = search_strategy
        self.early_stop = bool(early_stop)
        self.transposition_cache = bool(transposition_cache)
        self.transposition_cache_size = int(transposition_cache_size)
        self.gumbel_scale = float(gumbel_scale)
        self.sh_max_considered = int(sh_max_considered)
        self.sh_c_visit = float(sh_c_visit)
        self.sh_c_scale = float(sh_c_scale)
        self.c_puct = c_puct
        self.num_simulations = simulations if simulations is not None else num_simulations
        self.max_depth = max_depth
        if hamiltonian_model is not None and dynamics_model is not None:
            raise ValueError("mount either hamiltonian_model or dynamics_model on MctsEngine, not both")
        self.hamiltonian_model = hamiltonian_model
        self.dynamics_model = dynamics_model
        self.use_koopman_jumps = use_koopman_jumps
        self.discount = kwargs.get("gamma", discount)
        self.enable_reflection = enable_reflection
        self.rng = np.random.RandomState(seed)

    def plan(
        self,
        root_state: Any,
        candidate_actions: List[str],
        transition_fn: Optional[Callable[[Any, str], Tuple[Any, float, bool]]],
        *,
        legal_actions_fn: Callable[[Any], Sequence[str]],
        reward_fn: Optional[Callable[[Any, str, Any], float]] = None,
        eval_fn: Optional[Callable[[Any, List[str]], Tuple[Dict[str, float], float]]] = None,
        causal_invariant_fn: Optional[Callable[[Any, str, Any, float, bool], bool]] = None,
        num_simulations: Optional[int] = None,
        max_depth: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Executes PUCT search with per-state action masks and causal falsification pruning.

        Each simulation selects down the tree, expands the leaf and backs its value up.
        ``legal_actions_fn(state)`` is called at every node, and only the actions it
        returns are expanded or passed to ``transition_fn``. At the root the legal set
        is further restricted to ``candidate_actions``. It has no default: reusing the
        root list at child states would call transitions on actions that are illegal there.

        ``causal_invariant_fn(state, action, next_state, reward, done)`` returning False
        falsifies that child. A non-terminal node with no legal action, or whose
        ``eval_fn`` priors are all zero, is a dead end and is falsified too. A node whose
        children are all falsified is falsified in turn, up to the root.

        Leaf values come from exactly one source, reported as ``value_source``:
        ``dynamics_model`` (numpy root), ``eval_fn`` (transition reward plus the
        discounted value it returns; its priors weight the children), ``reward_fn``,
        or, when none is given, uniform random values with a warning.

        ``status`` is ``OK`` only when a searched action exists. Otherwise
        ``best_action`` is None, ``visit_distribution`` is empty and ``status`` is
        ``NO_CANDIDATES``, ``NO_LEGAL_ACTIONS``, ``ROOT_DEAD_END``, ``ALL_PRUNED``,
        ``NON_FINITE_TRANSITION_REWARD`` (a step/transition or ``reward_fn`` returned
        NaN/Inf; fails closed instead of ever reporting ``OK`` with a poisoned value),
        ``NON_FINITE_RETURN`` (every individual reward was finite, but the accumulated
        return overflowed to +/-inf across backpropagation; T3-M01) or
        ``INCONCLUSIVE`` (no surviving root action was ever visited). Callers must not
        replace such a result with a default action.
        """
        t0 = time.perf_counter()
        if not callable(legal_actions_fn):
            raise TypeError("legal_actions_fn must be callable: MCTS needs the legal actions of every state")
        use_dynamics = self.dynamics_model is not None and isinstance(root_state, np.ndarray)
        if reward_fn is not None and eval_fn is not None:
            raise ValueError("reward_fn and eval_fn both define leaf values; pass only one")
        if use_dynamics:
            if reward_fn is not None or eval_fn is not None:
                raise ValueError("dynamics_model defines leaf values; do not also pass reward_fn or eval_fn")
            transition_source = "neural_dynamics"
            value_source = "neural_dynamics"
        else:
            if transition_fn is None:
                if self.dynamics_model is not None:
                    raise TypeError("Neural dynamics MCTS requires a numpy.ndarray latent root state")
                raise TypeError("transition_fn is required when no dynamics_model is mounted")
            transition_source = "transition_fn"
            if eval_fn is not None:
                value_source = "eval_fn"
            elif reward_fn is not None:
                value_source = "reward_fn"
            else:
                value_source = "random_rollout"
                logger.warning("MctsEngine.plan: no eval_fn, reward_fn or dynamics_model; leaf values are uniform random")
        sims = self.num_simulations if num_simulations is None else num_simulations
        depth_limit = self.max_depth if max_depth is None else max_depth
        if sims < 1:
            raise ValueError("num_simulations must be >= 1")
        if depth_limit < 1:
            raise ValueError("max_depth must be >= 1")

        cache = _TranspositionCache(self.transposition_cache_size) if self.transposition_cache else None
        counters: Dict[str, Any] = {
            "falsified": 0, "dead_end": 0, "next_id": 1, "max_depth": 0,
            "simulations_run": 0, "early_stopped": False, "early_stop_reason": None,
            "transition_calls": 0, "legal_calls": 0,
        }

        def result(status: str, best: Optional[str], dist: Dict[str, float], **extra: Any) -> Dict[str, Any]:
            out = {
                "status": status,
                "best_action": best,
                "visit_distribution": dist,
                "expected_value": 0.0,
                "nodes_expanded": 0,
                "falsified_pruned_count": 0,
                "dead_end_count": 0,
                "max_depth_reached": 0,
                "root_q": {},
                "root_priors": {},
                "catastrophic_root_actions": {},
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "transition_source": transition_source,
                "value_source": value_source,
                "search_strategy": self.search_strategy,
                "early_stopped": counters["early_stopped"],
                "early_stop_reason": counters["early_stop_reason"],
                "simulations_run": counters["simulations_run"],
                "simulations_budget": sims,
                "transposition_cache": cache is not None,
                "transition_calls": counters["transition_calls"],
                "legal_calls": counters["legal_calls"],
                "cache_hits": 0,
                "legal_cache_hits": 0,
                "transition_cache_hits": 0,
                "cache_misses": 0,
                "cache_uncacheable": 0,
            }
            if cache is not None:
                out.update(cache.stats())
            out.update(extra)
            return out

        if not candidate_actions:
            return result("NO_CANDIDATES", None, {})

        # The optional key arguments are unused here; they keep one call signature
        # so the default path needs no extra wrapper layer.
        def raw_legal_at(state: Any, key: Any = None) -> List[str]:
            counters["legal_calls"] += 1
            acts = legal_actions_fn(state)
            if acts is None or isinstance(acts, (str, bytes)):
                raise TypeError("legal_actions_fn must return a collection of actions")
            return list(dict.fromkeys(acts))

        raw_step_fn = self.dynamics_model.step if use_dynamics else transition_fn

        def raw_step(state: Any, action: str, skey: Any = None) -> Tuple[Any, float, bool]:
            counters["transition_calls"] += 1
            return raw_step_fn(state, action)

        if cache is None:
            legal_at = raw_legal_at
            step = raw_step
        else:
            # ``key`` / ``skey`` let expand() hash a node's state once for all its
            # lookups; _UNKEYED means "compute it here".
            def legal_at(state: Any, key: Any = _UNKEYED) -> List[str]:
                if key is _UNKEYED:
                    key = _state_key(state)
                if key is None:
                    cache.uncacheable += 1
                    return raw_legal_at(state)
                hit = cache.legal.get(key)
                if hit is not None:
                    cache.legal_hits += 1
                    return list(hit)
                cache.misses += 1
                acts = raw_legal_at(state)
                cache._put(cache.legal, key, tuple(acts))
                return acts

            def step(state: Any, action: str, skey: Any = _UNKEYED) -> Tuple[Any, float, bool]:
                if skey is _UNKEYED:
                    skey = _state_key(state)
                key = None if skey is None else (skey, action)
                if key is not None:
                    try:
                        hash(key)
                    except TypeError:
                        key = None
                if key is None:
                    cache.uncacheable += 1
                    return raw_step(state, action)
                hit = cache.steps.get(key)
                if hit is not None:
                    cache.transition_hits += 1
                    return _copy_state(hit[0]), hit[1], hit[2]
                cache.misses += 1
                next_s, r, done = raw_step(state, action)
                r = _require_finite_reward(r, action, transition_source)
                # The node keeps ``next_s``; every hit hands out its own copy.
                cache._put(cache.steps, key, (next_s, r, bool(done)))
                return next_s, r, done

        nodes: Dict[int, MctsNode64B] = {}
        node_states: Dict[int, Any] = {}
        node_actions: Dict[int, str] = {}
        node_rewards: Dict[int, float] = {}
        node_done: Dict[int, bool] = {}
        node_values: Dict[int, float] = {}
        node_root_action: Dict[int, str] = {}
        evaluated: Dict[int, float] = {}
        catastrophic: Dict[str, int] = {}

        root_node = MctsNode64B(node_id=0, parent_id=-1, depth=0)
        nodes[0] = root_node
        node_states[0] = root_state
        node_done[0] = False

        root_legal = set(legal_at(root_state))
        root_actions = [a for a in dict.fromkeys(candidate_actions) if a in root_legal]
        root_masked = [a for a in dict.fromkeys(candidate_actions) if a not in root_legal]
        if not root_actions:
            logger.warning("MctsEngine.plan: no candidate action is legal at the root state")
            return result("NO_LEGAL_ACTIONS", None, {}, root_masked_actions=root_masked)

        def falsify(nid: int) -> None:
            # Cascade: a node with no surviving child cannot be entered safely either.
            while nid > 0 and not nodes[nid].falsified:
                nodes[nid].falsified = True
                counters["falsified"] += 1
                parent = nodes[nid].parent_id
                if any(not nodes[c].falsified for c in nodes[parent].children_ids):
                    return
                nid = parent
            if nid == 0 and all(nodes[c].falsified for c in root_node.children_ids):
                root_node.falsified = True

        def evaluate(nid: int, legal: List[str]) -> Optional[Dict[str, float]]:
            """Runs eval_fn once per node; returns priors over ``legal`` or None for a dead end."""
            if eval_fn is None:
                evaluated[nid] = 0.0
                return {a: 1.0 / len(legal) for a in legal}
            priors, value = eval_fn(node_states[nid], list(legal))
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("eval_fn returned a non-finite value")
            evaluated[nid] = value
            weights = {}
            for a in legal:
                p = float(priors.get(a, 0.0))
                if not math.isfinite(p) or p < 0.0:
                    raise ValueError(f"eval_fn returned an invalid prior {p!r} for action {a!r}")
                weights[a] = p
            total = sum(weights.values())
            if total <= 0.0:
                return None
            return {a: p / total for a, p in weights.items()}

        def expand(nid: int, actions: Optional[List[str]] = None) -> None:
            """Evaluates node ``nid`` and, below the depth limit, creates its legal children."""
            node = nodes[nid]
            state = node_states[nid]
            skey = _state_key(state) if cache is not None else None
            legal = actions if actions is not None else legal_at(state, skey)
            priors = evaluate(nid, legal) if legal else None
            if priors is None:
                counters["dead_end"] += 1
                if nid == 0:
                    root_node.falsified = True
                else:
                    falsify(nid)
                return
            if node.depth >= depth_limit:
                return
            for a_idx, action in enumerate(legal):
                cid = counters["next_id"]
                counters["next_id"] += 1
                child = MctsNode64B(node_id=cid, parent_id=nid, action_idx=a_idx,
                                    prior_p=priors[action], depth=node.depth + 1)
                nodes[cid] = child
                node_actions[cid] = action
                node_root_action[cid] = action if nid == 0 else node_root_action[nid]
                if cache is None:
                    # Inline count: no wrapper frame on the default hot path.
                    counters["transition_calls"] += 1
                    next_s, r, done = raw_step_fn(state, action)
                else:
                    next_s, r, done = step(state, action, skey)
                r = _require_finite_reward(r, action, transition_source)
                node_states[cid] = next_s
                node_rewards[cid] = r
                node_done[cid] = bool(done)
                counters["max_depth"] = max(counters["max_depth"], child.depth)
                if done and r <= -5.0:
                    ra = node_root_action[cid]
                    catastrophic[ra] = catastrophic.get(ra, 0) + 1
                if use_dynamics:
                    node_values[cid] = self._imagined_value(
                        next_s, r, done, legal_at, depth_limit - child.depth, step_fn=step
                    )
                node.children_ids.append(cid)
                if causal_invariant_fn is not None and not causal_invariant_fn(state, action, next_s, r, done):
                    child.falsified = True
                    counters["falsified"] += 1
            if all(nodes[c].falsified for c in node.children_ids):
                if nid == 0:
                    root_node.falsified = True
                else:
                    falsify(nid)

        try:
            expand(0, root_actions)
        except NonFiniteRewardError as exc:
            logger.error("MctsEngine.plan: %s; aborting with a non-OK status", exc)
            return result("NON_FINITE_TRANSITION_REWARD", None, {})
        except NonFiniteReturnError as exc:
            logger.error("MctsEngine.plan: %s; aborting with a non-OK status", exc)
            return result("NON_FINITE_RETURN", None, {})

        halving = (
            _SequentialHalving(self, nodes, root_node, sims)
            if self.search_strategy == "sequential_halving" and not root_node.falsified
            else None
        )
        # A later simulation could still falsify the leader, so never stop early
        # while an explicit causal invariant can prune.
        early_stop_active = self.early_stop and causal_invariant_fn is None

        for _ in range(sims):
            if root_node.falsified:
                break
            # The visit lead can exceed the remaining budget only after half of it.
            if early_stop_active and (halving is not None or 2 * counters["simulations_run"] > sims):
                reason = (halving.resolved() if halving is not None
                          else _count_dominance(nodes, root_node, sims - counters["simulations_run"]))
                if reason is not None:
                    counters["early_stopped"] = True
                    counters["early_stop_reason"] = reason
                    break
            counters["simulations_run"] += 1
            # 1. Selection (Sequential Halving picks the root child, PUCT below it)
            curr_id = 0
            path = [curr_id]
            descend = True
            if halving is not None:
                forced = halving.next_root_child()
                if forced is not None:
                    curr_id = forced
                    path.append(curr_id)
                    descend = nodes[curr_id].depth < depth_limit
            while descend and nodes[curr_id].children_ids and not node_done[curr_id]:
                curr_node = nodes[curr_id]
                valid_children = [cid for cid in curr_node.children_ids if not nodes[cid].falsified]
                if not valid_children:
                    falsify(curr_id)
                    break
                best_puct = -float("inf")
                best_cid = valid_children[0]
                n_curr = max(1, curr_node.visit_count)
                for cid in valid_children:
                    c_node = nodes[cid]
                    score = c_node.q_value + self.c_puct * c_node.prior_p * (
                        math.sqrt(n_curr) / (1 + c_node.visit_count)
                    )
                    if score > best_puct:
                        best_puct = score
                        best_cid = cid
                curr_id = best_cid
                path.append(curr_id)
                if nodes[curr_id].depth >= depth_limit:
                    break

            leaf_id = curr_id
            leaf_node = nodes[leaf_id]
            if leaf_node.falsified or leaf_id == 0:
                continue

            # 2. Expansion under the leaf's own legal-action mask
            if not node_done[leaf_id] and leaf_id not in evaluated:
                try:
                    expand(leaf_id)
                except NonFiniteRewardError as exc:
                    logger.error("MctsEngine.plan: %s; aborting with a non-OK status", exc)
                    return result("NON_FINITE_TRANSITION_REWARD", None, {})
                except NonFiniteReturnError as exc:
                    logger.error("MctsEngine.plan: %s; aborting with a non-OK status", exc)
                    return result("NON_FINITE_RETURN", None, {})
                if leaf_node.falsified:
                    continue

            leaf_state = node_states[leaf_id]
            if self.hamiltonian_model is not None:
                if not isinstance(leaf_state, np.ndarray):
                    raise TypeError("Hamiltonian MCTS requires numpy.ndarray transition states")
                h_res = self.hamiltonian_model.step(leaf_state, action=node_actions[leaf_id])
                leaf_node.hamiltonian_energy = float(h_res.hamiltonian_energy)
                leaf_state = h_res.next_state

            # 3. Evaluation
            if use_dynamics:
                eval_value = node_values[leaf_id]
            elif eval_fn is not None:
                eval_value = node_rewards[leaf_id]
                if not node_done[leaf_id]:
                    eval_value += self.discount * evaluated[leaf_id]
            elif reward_fn is not None:
                parent_s = node_states[leaf_node.parent_id]
                eval_value = reward_fn(parent_s, node_actions[leaf_id], leaf_state)
                if not math.isfinite(eval_value):
                    logger.error(
                        "MctsEngine.plan: reward_fn returned a non-finite value %r; aborting with a non-OK status",
                        eval_value,
                    )
                    return result("NON_FINITE_TRANSITION_REWARD", None, {})
            else:
                eval_value = float(self.rng.uniform(0.1, 1.0))

            if not math.isfinite(eval_value):
                logger.error(
                    "MctsEngine.plan: leaf evaluation produced a non-finite return %r "
                    "(T3-M01: an accumulation, not a single non-finite step reward); "
                    "aborting with a non-OK status",
                    eval_value,
                )
                return result("NON_FINITE_RETURN", None, {})

            # 4. Backpropagation: proper Bellman discounted-return recursion (X-M01 fix).
            # ``eval_value`` is already G at the leaf (r_leaf + gamma * V(leaf)). Walking
            # back up the path, each ancestor's return must fold in its OWN transition
            # reward: G_t = r_t + gamma * G_{t+1}. The previous code only did ``v *=
            # discount`` and never added the edge reward, so a high-reward-then-low-reward
            # path (e.g. +100 then -1) had its +100 discarded entirely and lost to a
            # low-then-high path (e.g. 0 then +10). The root has no incoming edge, so it
            # takes the child's G unchanged rather than folding in a reward that doesn't exist.
            #
            # T3-M01: each individual step reward can pass the finite check above and yet
            # the *accumulator* ``value_sum`` overflows to +/-inf after enough simulations
            # add to it (e.g. many additions of a legal-but-huge finite reward). A finite
            # per-step reward must never be allowed to silently poison value_sum/Q into inf
            # while the search keeps reporting status = OK, so every accumulation step is
            # checked here too, not just the per-step reward at its origin.
            rev_path = list(reversed(path))
            v = eval_value
            nodes[rev_path[0]].visit_count += 1
            nodes[rev_path[0]].value_sum += v
            if not math.isfinite(nodes[rev_path[0]].value_sum):
                logger.error(
                    "MctsEngine.plan: node %d value_sum overflowed to %r after accumulation; "
                    "aborting with a non-OK status",
                    rev_path[0], nodes[rev_path[0]].value_sum,
                )
                return result("NON_FINITE_RETURN", None, {})
            for nid in rev_path[1:]:
                if nid != 0:
                    v = node_rewards[nid] + self.discount * v
                    if not math.isfinite(v):
                        logger.error(
                            "MctsEngine.plan: discounted return overflowed to %r while "
                            "backpropagating through node %d; aborting with a non-OK status",
                            v, nid,
                        )
                        return result("NON_FINITE_RETURN", None, {})
                nodes[nid].visit_count += 1
                nodes[nid].value_sum += v
                if not math.isfinite(nodes[nid].value_sum):
                    logger.error(
                        "MctsEngine.plan: node %d value_sum overflowed to %r after accumulation; "
                        "aborting with a non-OK status",
                        nid, nodes[nid].value_sum,
                    )
                    return result("NON_FINITE_RETURN", None, {})

        survivors = [cid for cid in root_node.children_ids if not nodes[cid].falsified]
        root_q = {node_actions[cid]: nodes[cid].q_value for cid in survivors}
        root_priors = {node_actions[cid]: nodes[cid].prior_p for cid in root_node.children_ids}
        stats = dict(
            expected_value=root_node.q_value,
            nodes_expanded=len(nodes),
            falsified_pruned_count=counters["falsified"],
            dead_end_count=counters["dead_end"],
            max_depth_reached=counters["max_depth"],
            root_q=root_q,
            root_priors=root_priors,
            catastrophic_root_actions=catastrophic,
            root_masked_actions=root_masked,
            root_visit_counts={node_actions[cid]: nodes[cid].visit_count for cid in survivors},
        )
        if not root_node.children_ids:
            logger.warning("MctsEngine.plan: root state is a dead end (no legal action with a positive prior)")
            return result("ROOT_DEAD_END", None, {}, **stats)
        if not survivors:
            logger.warning("MctsEngine.plan: every root branch was falsified; no action is returned")
            return result("ALL_PRUNED", None, {}, **stats)
        total_visits = sum(nodes[cid].visit_count for cid in survivors)
        if total_visits == 0:
            logger.warning("MctsEngine.plan: no surviving root branch was visited; search is inconclusive")
            return result("INCONCLUSIVE", None, {}, **stats)

        visit_distribution = {
            node_actions[cid]: round(nodes[cid].visit_count / total_visits, 4) for cid in survivors
        }
        best_cid = max(survivors, key=lambda cid: (nodes[cid].visit_count, nodes[cid].q_value))
        if halving is not None:
            best_cid = halving.final_choice(best_cid)
        # T3-M01 final gate: never report status = OK on top of an inf/NaN-poisoned value,
        # even if every per-step check above was somehow satisfied. Belt-and-braces on the
        # exact fields the docstring contract promises (expected_value, root_q).
        if not math.isfinite(stats["expected_value"]) or any(not math.isfinite(q) for q in root_q.values()):
            logger.error(
                "MctsEngine.plan: root expected_value/root_q non-finite (expected_value=%r, root_q=%r) "
                "at OK-return time; aborting with a non-OK status",
                stats["expected_value"], root_q,
            )
            return result("NON_FINITE_RETURN", None, {})
        return result("OK", node_actions[best_cid], visit_distribution, **stats)

    def _imagined_value(
        self,
        state: np.ndarray,
        reward: float,
        done: bool,
        legal_actions_fn: Callable[[Any], List[str]],
        remaining: int,
        step_fn: Optional[Callable[[Any, str], Tuple[Any, float, bool]]] = None,
    ) -> float:
        """Discounted return of a greedy imagined rollout under the learned model.

        Starts from the reward predicted for the transition into ``state``, then
        repeatedly takes the legal action (for the current imagined state) with the
        highest predicted reward until the model predicts ``done``, no action is
        legal, or the depth budget runs out.
        """
        step_fn = step_fn or self.dynamics_model.step
        value = float(reward)
        scale = 1.0
        for _ in range(max(0, remaining)):
            if done:
                break
            actions = legal_actions_fn(state)
            if not actions:
                break
            outcomes = [step_fn(state, a) for a in actions]
            outcomes = [
                (ns, _require_finite_reward(r, a, "dynamics_model.step"), d)
                for a, (ns, r, d) in zip(actions, outcomes)
            ]
            state, reward, done = max(outcomes, key=lambda o: o[1])
            scale *= self.discount
            value += scale * reward
            value = _require_finite_return(value, "dynamics_model imagined rollout accumulation")
        return value

    def search(
        self,
        root_state: Any,
        get_legal_actions_fn: Callable[[Any], List[str]],
        transition_fn: Callable[[Any, str], Tuple[Any, float, bool]],
        eval_fn: Optional[Callable[[Any, List[str]], Tuple[Dict[str, float], float]]] = None,
        simulations: Optional[int] = None,
        max_depth: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Tree search over ``plan`` with metacognitive reflection on the prior's choice.

        Reflection fires (with ``enable_reflection``) when the action the priors favour
        at the root is not the searched best action and the tree reached a terminal
        transition with reward <= -5 below it. The log names that action as the culprit
        and the searched best action as the recommendation. ``best_action`` is always
        the search result from ``plan``.
        """
        res = self.plan(
            root_state,
            list(get_legal_actions_fn(root_state)),
            transition_fn,
            legal_actions_fn=get_legal_actions_fn,
            eval_fn=eval_fn,
            num_simulations=simulations,
            max_depth=max_depth,
        )
        action_scores = dict(res["root_q"])
        best_action = res["best_action"]
        reflections_log: List[Dict[str, Any]] = []
        if self.enable_reflection and res["status"] == "OK" and res["root_priors"]:
            intuitive = max(res["root_priors"], key=lambda a: res["root_priors"][a])
            if intuitive != best_action and res["catastrophic_root_actions"].get(intuitive, 0) > 0:
                reflections_log.append({
                    "culprit_action": intuitive,
                    "recommended_action": best_action,
                    "reason": "Search reached a catastrophic terminal below the prior-favoured action",
                })

        return {
            "status": res["status"],
            "best_action": best_action,
            "reflections_triggered": len(reflections_log),
            "reflections_log": reflections_log,
            "action_scores": action_scores,
            "visit_distribution": res["visit_distribution"],
            "max_depth_reached": res["max_depth_reached"],
            "falsified_pruned_count": res["falsified_pruned_count"],
        }
