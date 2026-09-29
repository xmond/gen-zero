"""Gen-Zero Planning Layer: Multi-Step Beam Search Planner with Diversity Penalty.

RFC-079 Implementation:
Extends sequential forward search with explicit repetition and cost penalization:
S(a_t | a_{1:t-1}, s_t) = log P_theta(a_t | s_t) - lambda * sum_{tau=1}^{t-1} I(a_tau == a_t) - mu * Cost(a_t)

Repetition penalties discourage repeated actions within the finite search horizon.
They do not prove deadlock freedom or prevent all loops during execution.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Sequence, Callable, Any
import math
import time
import numpy as np

from gen_zero.capability.descriptor import CapabilityDescriptor


@dataclass
class BeamNode:
    """Represents an active search path within the diversity beam tree."""
    sequence: List[str]
    cum_score: float
    state: Any
    total_cost: float = 0.0
    depth: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def last_action(self) -> Optional[str]:
        return self.sequence[-1] if self.sequence else None

    def count_occurrences(self, action: str) -> int:
        return sum(1 for a in self.sequence if a == action)


@dataclass
class DiversityBeamPlanResult:
    """Outcome and telemetry of a diversity beam search execution."""
    selected_sequence: List[str]
    selected_score: float
    total_cost: float
    beam_width: int
    horizon_depth: int
    repetition_count: int
    explored_nodes: int
    planning_time_ms: float
    all_top_beams: List[BeamNode]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_sequence": self.selected_sequence,
            "selected_score": round(self.selected_score, 4),
            "total_cost": round(self.total_cost, 4),
            "beam_width": self.beam_width,
            "horizon_depth": self.horizon_depth,
            "repetition_count": self.repetition_count,
            "explored_nodes": self.explored_nodes,
            "planning_time_ms": round(self.planning_time_ms, 3),
        }


class DiversityBeamPlanner:
    """Fast multi-step sequence planner incorporating diversity penalties and execution cost constraints."""

    def __init__(
        self,
        beam_width: int = 4,
        max_depth: int = 4,
        lambda_diversity: float = 1.2,
        mu_cost: float = 0.2,
        transition_model: Optional[Any] = None,
    ) -> None:
        self.beam_width = max(1, beam_width)
        self.max_depth = max(1, max_depth)
        self.lambda_diversity = float(lambda_diversity)
        self.mu_cost = float(mu_cost)
        self.transition_model = transition_model

    def plan(
        self,
        initial_state: Any,
        candidate_capabilities: Sequence[CapabilityDescriptor],
        prior_policy_fn: Optional[Callable[[Any, Sequence[str]], Dict[str, float]]] = None,
        terminal_check_fn: Optional[Callable[[Any, int], bool]] = None,
    ) -> DiversityBeamPlanResult:
        """Executes diversity-penalized beam search over candidate capability descriptors."""
        t0 = time.perf_counter()

        if not candidate_capabilities:
            return DiversityBeamPlanResult(
                selected_sequence=[],
                selected_score=0.0,
                total_cost=0.0,
                beam_width=self.beam_width,
                horizon_depth=0,
                repetition_count=0,
                explored_nodes=0,
                planning_time_ms=0.0,
                all_top_beams=[],
            )

        candidate_ids = [c.capability_id for c in candidate_capabilities]
        cost_map = {c.capability_id: float(c.cost_weight) for c in candidate_capabilities}

        # Initialize root beam
        current_beams: List[BeamNode] = [
            BeamNode(sequence=[], cum_score=0.0, state=initial_state, total_cost=0.0, depth=0)
        ]

        total_explored = 0

        for d in range(self.max_depth):
            next_candidates: List[BeamNode] = []

            for beam in current_beams:
                # Check terminal condition
                if terminal_check_fn is not None and terminal_check_fn(beam.state, d):
                    next_candidates.append(beam)
                    continue

                # Obtain prior probability distribution over candidate actions
                if prior_policy_fn is not None:
                    probs = prior_policy_fn(beam.state, candidate_ids)
                else:
                    # Uniform baseline
                    probs = {cid: 1.0 / len(candidate_ids) for cid in candidate_ids}

                for cid in candidate_ids:
                    p = max(1e-6, probs.get(cid, 1e-6))
                    log_p = math.log(p)

                    # Repetition penalty: lambda * sum_{tau=1}^{t-1} I(a_tau == a_t)
                    rep_count = beam.count_occurrences(cid)
                    diversity_penalty = self.lambda_diversity * rep_count

                    # Cost penalty: mu * Cost(a_t)
                    action_cost = cost_map.get(cid, 1.0)
                    cost_penalty = self.mu_cost * action_cost

                    # Net step score
                    step_score = log_p - diversity_penalty - cost_penalty
                    new_cum_score = beam.cum_score + step_score
                    new_total_cost = beam.total_cost + action_cost

                    # Transition state forward
                    if self.transition_model is not None and hasattr(self.transition_model, "step"):
                        try:
                            next_state, _, _ = self.transition_model.step(beam.state, cid)
                        except Exception:
                            next_state = beam.state
                    else:
                        next_state = beam.state

                    child_node = BeamNode(
                        sequence=beam.sequence + [cid],
                        cum_score=new_cum_score,
                        state=next_state,
                        total_cost=new_total_cost,
                        depth=d + 1,
                    )
                    next_candidates.append(child_node)
                    total_explored += 1

            if not next_candidates:
                break

            # Prune and retain top B beams
            next_candidates.sort(key=lambda b: b.cum_score, reverse=True)
            current_beams = next_candidates[: self.beam_width]

        # Select best beam
        best_beam = current_beams[0] if current_beams else BeamNode([], 0.0, initial_state)

        # Count repetitions in selected path
        selected_seq = best_beam.sequence
        unique_actions = set(selected_seq)
        rep_count = len(selected_seq) - len(unique_actions)

        dur_ms = (time.perf_counter() - t0) * 1000.0

        return DiversityBeamPlanResult(
            selected_sequence=selected_seq,
            selected_score=best_beam.cum_score,
            total_cost=best_beam.total_cost,
            beam_width=self.beam_width,
            horizon_depth=len(selected_seq),
            repetition_count=rep_count,
            explored_nodes=total_explored,
            planning_time_ms=dur_ms,
            all_top_beams=current_beams,
        )
