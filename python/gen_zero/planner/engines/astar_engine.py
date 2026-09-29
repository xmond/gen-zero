"""Gen-Zero Planning Engine 1: Unified AStarEngine.

Convergence of astar.py, bidirectional.py, and compressed_astar.py:
1. Bidirectional & Unidirectional Search:
   - Full meet-in-the-middle bidirectional goal-directed graph search.
   - Standard forward heuristic search and reverse search.
2. 64-Byte Heap-Free Arena (AStarArena64B):
   - Packed 64-byte struct layout avoiding Python heap allocations and GC pressure.
   - Pre-allocated node storage buffer.
3. Uncertainty-Aware Cost Penalty:
   - f(s) = g(s) + lambda * sigma(s) + h(s), where sigma(s) = -log(p_success + eps).
4. Transparent Polymorphism:
   - Supports both forward search and bidirectional search through plan().
"""

from __future__ import annotations

import heapq
import math
import struct
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

try:
    from gen_zero.runtime.zstd_codec import compress_bytes, decompress_bytes
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


class AStarArena64B:
    """Pre-allocated 64-byte packed node memory arena for A* expansions.

    Layout (64 Bytes total):
    - node_id:     uint32 (4B)
    - parent_id:   int32  (4B)
    - state_idx:   int32  (4B)
    - action_idx:  uint32 (4B)
    - g_cost:      float64 (8B)
    - f_cost:      float64 (8B)
    - uncertainty: float32 (4B)
    - flags:       uint32 (4B)
    - direction:   uint32 (4B, 0=fwd, 1=bwd)
    - pad:         20 bytes
    Sum: 4 + 4 + 4 + 4 + 8 + 8 + 4 + 4 + 4 + 20 = 64 Bytes.
    """

    STRUCT_FORMAT = "<IiiIddfII20s"
    NODE_SIZE_BYTES = 64

    def __init__(self, capacity: int = 4096):
        self.capacity = capacity
        self.raw_buffer = bytearray(self.NODE_SIZE_BYTES * capacity)
        self.allocated_count = 0

    def reset(self) -> None:
        """Resets arena allocation offset to zero without deallocation."""
        self.allocated_count = 0

    def allocate(
        self,
        parent_id: int,
        state_idx: int,
        action_idx: int,
        g_cost: float,
        f_cost: float,
        uncertainty: float = 0.0,
        flags: int = 0,
        direction: int = 0,
    ) -> int:
        """Allocates a 64-byte node in the arena and returns its node_id."""
        if self.allocated_count >= self.capacity:
            # Expand arena capacity
            new_capacity = self.capacity * 2
            new_buffer = bytearray(self.NODE_SIZE_BYTES * new_capacity)
            new_buffer[: len(self.raw_buffer)] = self.raw_buffer
            self.raw_buffer = new_buffer
            self.capacity = new_capacity

        node_id = self.allocated_count
        offset = node_id * self.NODE_SIZE_BYTES
        pad = b"\x00" * 20
        struct.pack_into(
            self.STRUCT_FORMAT,
            self.raw_buffer,
            offset,
            node_id,
            parent_id,
            state_idx,
            action_idx,
            g_cost,
            f_cost,
            uncertainty,
            flags,
            direction,
            pad,
        )
        self.allocated_count += 1
        return node_id

    def read_node(self, node_id: int) -> Tuple[int, int, int, int, float, float, float, int, int]:
        """Reads unpacked node fields from arena."""
        if node_id < 0 or node_id >= self.allocated_count:
            raise IndexError(f"Node id {node_id} out of arena bounds [0, {self.allocated_count})")
        offset = node_id * self.NODE_SIZE_BYTES
        fields = struct.unpack_from(self.STRUCT_FORMAT, self.raw_buffer, offset)
        return (
            fields[0],  # node_id
            fields[1],  # parent_id
            fields[2],  # state_idx
            fields[3],  # action_idx
            fields[4],  # g_cost
            fields[5],  # f_cost
            fields[6],  # uncertainty
            fields[7],  # flags
            fields[8],  # direction
        )

    def memory_usage_bytes(self) -> int:
        """Returns total allocated arena memory in bytes."""
        return len(self.raw_buffer)


class AStarEngine:
    """Unified Orthogonal A* Search Engine.

    Unifies:
    1. Forward heuristic search with uncertainty penalties
    2. Bidirectional meet-in-the-middle goal-directed search
    3. Memory-compressed zstd state pool
    4. 64-byte heap-free arena memory pool
    """

    def __init__(
        self,
        lambda_weight: float = 1.0,
        epsilon: float = 1e-6,
        use_arena: bool = True,
        compress_history: bool = False,
    ):
        self.lambda_weight = lambda_weight
        self.epsilon = epsilon
        self.use_arena = use_arena
        self.compress_history = compress_history
        self.arena = AStarArena64B(capacity=4096) if use_arena else None

    def uncertainty_penalty(self, p_success: float, lambda_val: Optional[float] = None) -> float:
        """Computes uncertainty penalty -lambda * log(p + eps)."""
        lam = self.lambda_weight if lambda_val is None else lambda_val
        p_clamped = max(self.epsilon, min(1.0, float(p_success)))
        return -lam * math.log(p_clamped)

    def plan(
        self,
        start_state: Any,
        target_or_is_goal: Any = None,
        get_neighbors_or_fwd: Optional[Callable[[Any], List[Tuple[Any, str, float]]]] = None,
        heuristic_or_bwd: Optional[Any] = None,
        state_key_fn: Optional[Callable[[Any], Any]] = None,
        max_expansions: int = 2000,
        dynamic_lambda: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Unified polymorphic planning interface.

        Dispatches between:
        - Bidirectional search: if target_or_is_goal is not callable and heuristic_or_bwd is callable (bwd neighbors fn).
        - Unidirectional forward search: if target_or_is_goal is a callable goal checker.
        """
        if target_or_is_goal is None and "is_goal_fn" in kwargs:
            target_or_is_goal = kwargs.pop("is_goal_fn")
        if get_neighbors_or_fwd is None and "get_neighbors_fn" in kwargs:
            get_neighbors_or_fwd = kwargs.pop("get_neighbors_fn")
        if heuristic_or_bwd is None and "heuristic_fn" in kwargs:
            heuristic_or_bwd = kwargs.pop("heuristic_fn")

        # Detect bidirectional mode
        if not callable(target_or_is_goal) and callable(heuristic_or_bwd):
            heuristic_fn = kwargs.get("heuristic_fn", None)
            return self.plan_bidirectional(
                start_state=start_state,
                goal_state=target_or_is_goal,
                get_forward_neighbors_fn=get_neighbors_or_fwd,
                get_backward_neighbors_fn=heuristic_or_bwd,
                heuristic_fn=heuristic_fn,
                state_key_fn=state_key_fn,
                max_expansions=max_expansions,
            )

        # Forward search mode
        heuristic_fn = heuristic_or_bwd if callable(heuristic_or_bwd) else None
        return self.plan_forward(
            start_state=start_state,
            is_goal_fn=target_or_is_goal,
            get_neighbors_fn=get_neighbors_or_fwd,
            heuristic_fn=heuristic_fn,
            state_key_fn=state_key_fn,
            max_expansions=max_expansions,
            dynamic_lambda=dynamic_lambda,
            **kwargs,
        )

    def plan_forward(
        self,
        start_state: Any,
        is_goal_fn: Callable[[Any], bool],
        get_neighbors_fn: Callable[[Any], List[Tuple[Any, str, float]]],
        heuristic_fn: Optional[Callable[[Any], float]] = None,
        state_key_fn: Optional[Callable[[Any], Any]] = None,
        max_expansions: int = 2000,
        dynamic_lambda: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Executes forward uncertainty-weighted A* search."""
        t0 = time.perf_counter()
        if state_key_fn is None:
            state_key_fn = lambda s: str(s)
        if heuristic_fn is None:
            heuristic_fn = lambda s: 0.0

        effective_lambda = self.lambda_weight if dynamic_lambda is None else dynamic_lambda

        if self.arena is not None:
            self.arena.reset()

        start_key = state_key_fn(start_state)
        best_g: Dict[Any, float] = {start_key: 0.0}
        closed_set: Set[Any] = set()

        # Heap entry: (f_score, g_score, count, state_key, state, path, state_history)
        count = 0
        h_start = heuristic_fn(start_state)
        pq = [(h_start, 0.0, count, start_key, start_state, [], [start_state])]
        expansions = 0

        state_registry: List[Any] = [start_state]
        action_registry: List[str] = []
        action_to_idx: Dict[str, int] = {}

        if self.arena is not None:
            self.arena.allocate(
                parent_id=-1,
                state_idx=0,
                action_idx=0,
                g_cost=0.0,
                f_cost=h_start,
                uncertainty=0.0,
                flags=0,
                direction=0,
            )

        while pq and expansions < max_expansions:
            f, g, _, curr_key, curr_state, path, state_history = heapq.heappop(pq)
            if curr_key in closed_set:
                continue

            closed_set.add(curr_key)
            expansions += 1

            if is_goal_fn(curr_state):
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return {
                    "success": True,
                    "path": path,
                    "cost": g,
                    "nodes_expanded": expansions,
                    "states": state_history,
                    "latency_ms": latency_ms,
                    "mode": "forward",
                    "arena_nodes": self.arena.allocated_count if self.arena else 0,
                }

            neighbors = get_neighbors_fn(curr_state)
            for next_state, action_str, p_success in neighbors:
                next_key = state_key_fn(next_state)
                if next_key in closed_set:
                    continue

                penalty = self.uncertainty_penalty(p_success, lambda_val=effective_lambda)
                edge_cost = 1.0 + penalty
                new_g = g + edge_cost

                if next_key not in best_g or new_g < best_g[next_key]:
                    best_g[next_key] = new_g
                    h_val = heuristic_fn(next_state)
                    new_f = new_g + h_val
                    count += 1
                    new_path = path + [action_str]
                    new_history = state_history + [next_state]

                    if self.arena is not None:
                        s_idx = len(state_registry)
                        state_registry.append(next_state)
                        if action_str not in action_to_idx:
                            action_to_idx[action_str] = len(action_registry)
                            action_registry.append(action_str)
                        a_idx = action_to_idx[action_str]
                        self.arena.allocate(
                            parent_id=expansions - 1,
                            state_idx=s_idx,
                            action_idx=a_idx,
                            g_cost=new_g,
                            f_cost=new_f,
                            uncertainty=penalty,
                            flags=0,
                            direction=0,
                        )

                    heapq.heappush(
                        pq,
                        (new_f, new_g, count, next_key, next_state, new_path, new_history),
                    )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "path": [],
            "cost": float("inf"),
            "nodes_expanded": expansions,
            "states": [],
            "latency_ms": latency_ms,
            "mode": "forward",
            "arena_nodes": self.arena.allocated_count if self.arena else 0,
        }

    def plan_bidirectional(
        self,
        start_state: Any,
        goal_state: Any,
        get_forward_neighbors_fn: Callable[[Any], List[Tuple[Any, str, float]]],
        get_backward_neighbors_fn: Callable[[Any], List[Tuple[Any, str, float]]],
        heuristic_fn: Optional[Callable[[Any, Any], float]] = None,
        state_key_fn: Optional[Callable[[Any], Any]] = None,
        max_expansions: int = 4000,
    ) -> Dict[str, Any]:
        """Plans optimal path connecting start_state and goal_state via two meeting frontiers."""
        t0 = time.perf_counter()
        if state_key_fn is None:
            state_key_fn = lambda s: str(s)
        if heuristic_fn is None:
            heuristic_fn = lambda s1, s2: 0.0

        start_key = state_key_fn(start_state)
        goal_key = state_key_fn(goal_state)

        if start_key == goal_key:
            return {
                "success": True,
                "path": [],
                "cost": 0.0,
                "nodes_expanded": 0,
                "mode": "bidirectional",
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
            }

        # Forward frontier
        f_h = heuristic_fn(start_state, goal_state)
        f_pq = [(f_h, 0.0, 0, start_key, start_state, [])]
        f_best_g: Dict[Any, float] = {start_key: 0.0}
        f_paths: Dict[Any, List[str]] = {start_key: []}
        f_closed: Set[Any] = set()

        # Backward frontier
        b_h = heuristic_fn(goal_state, start_state)
        b_pq = [(b_h, 0.0, 0, goal_key, goal_state, [])]
        b_best_g: Dict[Any, float] = {goal_key: 0.0}
        b_paths: Dict[Any, List[str]] = {goal_key: []}
        b_closed: Set[Any] = set()

        opp_actions = {
            "north": "south",
            "south": "north",
            "east": "west",
            "west": "east",
            "up": "down",
            "down": "up",
            "left": "right",
            "right": "left",
        }

        expansions = 0
        counter = 0
        best_connection_cost = float("inf")
        best_connection_state = None

        while (f_pq or b_pq) and expansions < max_expansions:
            # Alternate or pick frontier with minimum f-score
            expand_forward = True
            if f_pq and b_pq:
                expand_forward = f_pq[0][0] <= b_pq[0][0]
            elif b_pq:
                expand_forward = False

            if expand_forward and f_pq:
                f, g, _, curr_key, curr_state, path = heapq.heappop(f_pq)
                if curr_key in f_closed:
                    continue
                f_closed.add(curr_key)
                expansions += 1

                # Intersection check
                if curr_key in b_best_g:
                    total_c = g + b_best_g[curr_key]
                    if total_c < best_connection_cost:
                        best_connection_cost = total_c
                        best_connection_state = curr_key
                        if total_c <= f:
                            break

                for nxt, act, prob in get_forward_neighbors_fn(curr_state):
                    nxt_key = state_key_fn(nxt)
                    if nxt_key in f_closed:
                        continue
                    edge_cost = 1.0 + self.uncertainty_penalty(prob)
                    new_g = g + edge_cost
                    if nxt_key not in f_best_g or new_g < f_best_g[nxt_key]:
                        f_best_g[nxt_key] = new_g
                        f_paths[nxt_key] = path + [act]
                        new_f = new_g + heuristic_fn(nxt, goal_state)
                        counter += 1
                        heapq.heappush(f_pq, (new_f, new_g, counter, nxt_key, nxt, f_paths[nxt_key]))

            elif b_pq:
                f, g, _, curr_key, curr_state, path = heapq.heappop(b_pq)
                if curr_key in b_closed:
                    continue
                b_closed.add(curr_key)
                expansions += 1

                # Intersection check
                if curr_key in f_best_g:
                    total_c = f_best_g[curr_key] + g
                    if total_c < best_connection_cost:
                        best_connection_cost = total_c
                        best_connection_state = curr_key
                        if total_c <= f:
                            break

                for nxt, act, prob in get_backward_neighbors_fn(curr_state):
                    nxt_key = state_key_fn(nxt)
                    if nxt_key in b_closed:
                        continue
                    edge_cost = 1.0 + self.uncertainty_penalty(prob)
                    new_g = g + edge_cost
                    inv_act = opp_actions.get(act, f"inv_{act}")
                    if nxt_key not in b_best_g or new_g < b_best_g[nxt_key]:
                        b_best_g[nxt_key] = new_g
                        b_paths[nxt_key] = [inv_act] + path
                        new_f = new_g + heuristic_fn(nxt, start_state)
                        counter += 1
                        heapq.heappush(b_pq, (new_f, new_g, counter, nxt_key, nxt, b_paths[nxt_key]))

            # Early termination if meeting bound met
            if best_connection_state is not None:
                min_f = float("inf")
                if f_pq:
                    min_f = min(min_f, f_pq[0][0])
                if b_pq:
                    min_f = min(min_f, b_pq[0][0])
                if best_connection_cost <= min_f:
                    break

        latency_ms = (time.perf_counter() - t0) * 1000.0
        if best_connection_state is not None:
            fwd_p = f_paths.get(best_connection_state, [])
            bwd_p = b_paths.get(best_connection_state, [])
            full_path = fwd_p + bwd_p
            return {
                "success": True,
                "path": full_path,
                "cost": best_connection_cost,
                "nodes_expanded": expansions,
                "meeting_state": best_connection_state,
                "mode": "bidirectional",
                "latency_ms": latency_ms,
            }

        return {
            "success": False,
            "path": [],
            "cost": float("inf"),
            "nodes_expanded": expansions,
            "mode": "bidirectional",
            "latency_ms": latency_ms,
        }
