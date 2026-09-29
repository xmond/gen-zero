"""Gen-Zero Layer 3: Hard Sample Mining & Attribution Engine.

Identifies, filters, and attributes valuable learning samples from online rollouts:
1. Failure/Collision points: Steps directly leading to termination/collision.
2. High-Entropy decision points: States where model policy is uncertain (H(pi) > tau).
3. Large TD-Error points: States where final outcome z diverges from predicted V(s).
4. Synthesizes AlphaZero soft training targets: (s, pi_MCTS, z).
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Any, Optional


@dataclass
class MinedSample:
    state_id: str
    state_data: Dict[str, Any]
    candidate_ids: List[str]
    pi_target: Dict[str, float]       # Soft policy distribution from MCTS
    value_target: float               # Scalar game return z in [-1, 1]
    mining_reason: str                # 'collision', 'high_entropy', 'high_td_error', 'abstain', 'causal_decision_error'
    entropy: float
    td_error: float
    best_counterfactual_action: Optional[str] = None
    individual_treatment_effect: float = 0.0
    is_causal_culprit: bool = False


class HardSampleMiner:
    """Extracts high-value training samples from rollouts."""
    def __init__(
        self,
        entropy_threshold: float = 1.0,
        td_error_threshold: float = 0.5,
        history_window: int = 5
    ):
        self.entropy_threshold = entropy_threshold
        self.td_error_threshold = td_error_threshold
        self.history_window = history_window

    @staticmethod
    def calculate_entropy(probs: Dict[str, float]) -> float:
        h = 0.0
        for p in probs.values():
            if p > 1e-6:
                h -= p * math.log(p)
        return float(h)

    def mine_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
        final_outcome: str,
        final_score: float
    ) -> List[MinedSample]:
        """Analyzes an entire rollout episode and extracts hard learning samples.

        Args:
            trajectory: List of step records containing:
                - 'state': state dict
                - 'candidates': list of candidate actions
                - 'mcts_probs': visit distribution from search
                - 'model_probs': prior distribution from model
                - 'model_value': predicted scalar value V(s)
            final_outcome: Outcome string ('collision', 'trapped', 'horizon_survived', 'win').
            final_score: Numeric score achieved in episode.
        """
        mined = []
        n_steps = len(trajectory)
        is_failure = final_outcome in ("collision", "trapped", "fail", "timeout")

        # Map final outcome to scalar value z in [-1, 1]
        if final_outcome in ("win", "horizon_survived"):
            z = min(1.0, final_score / 25.0)
        else:
            z = -1.0

        from ..adaptive_engine import AdaptiveParameterEngine
        from ..causal.counterfactual_engine import CounterfactualEngine

        # Detect value cliff causal turning point
        causal_cliff_step = AdaptiveParameterEngine.detect_value_cliff_step(trajectory) if is_failure else -1

        # Perform 3-step Pearlian counterfactual attribution
        cf_engine = CounterfactualEngine()
        causal_attributions = cf_engine.attribute_trajectory_failures(
            trajectory=trajectory,
            final_outcome=final_outcome,
            final_score=final_score
        ) if is_failure else []

        # Collect trajectory TD errors for dynamic percentile calculation
        traj_td_errs = [abs(z - float(s.get("model_value") if s.get("model_value") is not None else 0.0)) for s in trajectory]

        for i, step in enumerate(trajectory):
            state = step["state"]
            cands = step["candidates"]
            mcts_pi = step.get("mcts_probs") or {c: 1.0 / len(cands) for c in cands}
            model_pi = step.get("model_probs") or mcts_pi
            v_val = step.get("model_value")
            v_pred = float(v_val if v_val is not None else 0.0)

            norm_ent = AdaptiveParameterEngine.get_normalized_entropy(model_pi)
            td_err = abs(z - v_pred)

            is_high_ent, is_high_td = AdaptiveParameterEngine.dynamic_mining_criteria(
                model_pi, td_err, history_td_errors=traj_td_errs
            )

            # Causal metadata
            c_attr = causal_attributions[i] if i < len(causal_attributions) else None
            is_causal_culprit = c_attr.is_decision_culprit if c_attr else False
            best_cf_act = c_attr.best_counterfactual_action if c_attr else None
            ite_val = c_attr.individual_treatment_effect if c_attr else 0.0

            reasons = []

            # 1. Pearlian Causal Decision Error (highest priority)
            if is_causal_culprit:
                reasons.append("causal_decision_error")

            # 2. Failure attribution: causal turning point to termination
            if is_failure and i >= causal_cliff_step:
                reasons.append("causal_failure_cliff" if i == causal_cliff_step else "collision_precursor")

            # 3. Dynamic high entropy (normalized relative threshold)
            if is_high_ent:
                reasons.append("high_entropy")

            # 4. Dynamic TD error (80th percentile surprise)
            if is_high_td:
                reasons.append("high_td_error")

            if reasons:
                mined.append(MinedSample(
                    state_id=f"mined:{step.get('step_idx', i)}",
                    state_data=state,
                    candidate_ids=cands,
                    pi_target=mcts_pi,
                    value_target=float(z),
                    mining_reason=",".join(reasons),
                    entropy=round(norm_ent, 3),
                    td_error=round(td_err, 3),
                    best_counterfactual_action=best_cf_act,
                    individual_treatment_effect=round(ite_val, 4),
                    is_causal_culprit=is_causal_culprit
                ))

        return mined
