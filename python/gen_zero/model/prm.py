"""Gen-Zero Layer 1/2 Expert: Step-Level Process Reward Model (PRM) & Step Verifier.

Implements step-level verification and test-time compute scaling (OpenAI o1 / MCTSr philosophy):
- Evaluates individual step transitions: (s_t, a_t, s_{t+1}) -> PRMResult.
- Computes step viability, forward reachability, and irreversible deadlock/trap risk.
- Enables early branch pruning in MCTS and test-time compute allocation to contentious branches.
"""

import math
from typing import Dict, List, Any, Tuple, Optional, Callable, Union

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


class ProcessRewardModel:
    """Step-Level Process Reward Model & Dynamic Step Verifier."""

    def __init__(
        self,
        prune_threshold: float = 0.20,
        deadlock_risk_cutoff: float = 0.80,
        prm_weight: float = 1.0
    ):
        self.prune_threshold = prune_threshold
        self.deadlock_risk_cutoff = deadlock_risk_cutoff
        self.prm_weight = prm_weight

    def _flood_fill_reachability(self, grid_size: int, obstacles: set, start_pos: Tuple[int, int]) -> int:
        """Computes BFS reachable free space count from start_pos within grid."""
        if (
            not isinstance(start_pos, (tuple, list))
            or len(start_pos) < 2
            or start_pos in obstacles
            or not (0 <= start_pos[0] < grid_size and 0 <= start_pos[1] < grid_size)
        ):
            return 0

        from collections import deque
        visited = {start_pos}
        queue = deque([start_pos])
        count = 0

        while queue:
            r, c = queue.popleft()
            count += 1
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                neighbor = (nr, nc)
                if (
                    0 <= nr < grid_size
                    and 0 <= nc < grid_size
                    and neighbor not in obstacles
                    and neighbor not in visited
                ):
                    visited.add(neighbor)
                    queue.append(neighbor)
        return count

    def verify_step(
        self,
        parent_state: Any,
        action: str,
        next_state: Any,
        is_done: bool = False,
        step_reward: float = 0.0
    ) -> Dict[str, Any]:
        """Evaluates single step transition (s_t, a_t, s_{t+1}) for validity and deadlock risk.
        
        Returns:
            Dict with 'viability' in [0, 1], 'deadlock_risk' in [0, 1],
            'should_prune' (bool), and 'reason' (str).
        """
        # 1. Immediate termination with severe penalty -> 100% fatal
        if is_done and step_reward < -1.0:
            return {
                "viability": 0.0,
                "deadlock_risk": 1.0,
                "should_prune": True,
                "reason": f"Fatal terminal step (collision/failure reward = {step_reward})"
            }

        # 2. Geometric pocket trap & cul-de-sac deadlock analysis (for Snake/Grid domains)
        if isinstance(next_state, dict):
            size = next_state.get("size")
            body = next_state.get("body")
            if size and body and len(body) > 0:
                head = tuple(body[0])
                obstacles = {tuple(p) for p in body[1:]}
                available_space = self._flood_fill_reachability(size, obstacles, head)
                body_len = len(body)

                # Irreversible deadlock: reachable space strictly less than body length
                if available_space < body_len and available_space < (size * size * 0.25):
                    risk = max(0.0, min(1.0, 1.0 - (available_space / max(1, body_len))))
                    viability = round(1.0 - risk, 4)
                    return {
                        "viability": viability,
                        "deadlock_risk": round(risk, 4),
                        "should_prune": risk >= self.deadlock_risk_cutoff,
                        "reason": f"Pocket trap deadlock detected: reachable space {available_space} < body length {body_len}"
                    }

            # Financial stop-loss or extreme drawdowns
            unrealized_pnl = next_state.get("unrealized_pnl")
            stop_loss = next_state.get("stop_loss_pct")
            if unrealized_pnl is not None and stop_loss is not None:
                if float(unrealized_pnl) <= -float(stop_loss):
                    return {
                        "viability": 0.05,
                        "deadlock_risk": 0.95,
                        "should_prune": True,
                        "reason": f"Financial risk envelope breached: PnL {unrealized_pnl} <= -{stop_loss}"
                    }

        # 3. String / Business Workflow verification
        if isinstance(next_state, str):
            s_low = next_state.lower()
            if "deadlock" in s_low or "fatal" in s_low or "blocked" in s_low:
                return {
                    "viability": 0.10,
                    "deadlock_risk": 0.90,
                    "should_prune": True,
                    "reason": "Text state explicitly flags deadlock or fatal condition"
                }

        # 4. Standard healthy transition
        return {
            "viability": 0.95,
            "deadlock_risk": 0.05,
            "should_prune": False,
            "reason": "Healthy forward step"
        }

    def modulate_priors(
        self,
        priors: Dict[str, float],
        step_evaluations: Dict[str, Dict[str, Any]],
        temperature: float = 1.0
    ) -> Dict[str, float]:
        """Modulates policy priors using process reward viability scores: P(a) ~ pi(a) * V(a)^beta."""
        if not priors:
            return {}

        modulated = {}
        for a, p in priors.items():
            ev = step_evaluations.get(a, {})
            viability = float(ev.get("viability", 0.95))
            if ev.get("should_prune", False):
                # Severely suppress pruned/fatal actions
                score = 1e-4
            else:
                score = p * (viability ** self.prm_weight)
            modulated[a] = score

        sum_score = sum(modulated.values()) or 1.0
        return {a: round(v / sum_score, 4) for a, v in modulated.items()}
