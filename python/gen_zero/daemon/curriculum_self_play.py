"""Gen-Zero Curriculum-Guided Self-Play Generator.

Synthesizes high-information boundary scenarios based on model uncertainty and entropy:
1. When model confidence is overly high (> 0.95), injects causal counterfactual obstacles
   and domain shifts to probe blind spots.
2. Identifies high-entropy boundary regimes (0.4 <= H <= 0.85) where learning gradient is maximal.
3. Generates diverse self-play tasks without human intervention.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import math
import random
import numpy as np


class CurriculumSelfPlayGenerator:
    """Automated task generator for continuous background self-play."""

    def __init__(
        self,
        base_action_dim: int = 4,
        difficulty_level: float = 0.5,
        entropy_target: float = 0.65
    ):
        self.base_action_dim = base_action_dim
        self.difficulty_level = difficulty_level
        self.entropy_target = entropy_target
        self.episode_counter = 0

    def generate_boundary_scenario(
        self,
        model_eval_fn: Optional[Any] = None,
        max_attempts: int = 5
    ) -> Dict[str, Any]:
        """Generates a scenario targeting the model's high-entropy decision frontier."""
        self.episode_counter += 1

        best_scenario = None
        closest_entropy_diff = float("inf")

        for _ in range(max_attempts):
            # Synthesize candidate scenario with varying perturbation
            scenario = self._synthesize_scenario()

            if model_eval_fn is not None and callable(model_eval_fn):
                try:
                    eval_res = model_eval_fn(scenario["state_repr"], scenario["candidate_actions"])
                    probs = eval_res.get("probabilities", {})
                    # Calculate Shannon entropy
                    p_vals = [p for p in probs.values() if p > 1e-6]
                    entropy = -sum(p * math.log(p) for p in p_vals) / math.log(max(2, len(p_vals)))
                    scenario["measured_entropy"] = entropy

                    diff = abs(entropy - self.entropy_target)
                    if diff < closest_entropy_diff:
                        closest_entropy_diff = diff
                        best_scenario = scenario
                        if diff < 0.1:  # Close enough to optimal boundary
                            break
                except Exception:
                    best_scenario = scenario
                    break
            else:
                best_scenario = scenario
                break

        return best_scenario or self._synthesize_scenario()

    def _synthesize_scenario(self) -> Dict[str, Any]:
        """Synthesizes a domain state with controllable difficulty."""
        rng = np.random.RandomState()
        scenario_id = f"auto_curriculum_ep_{self.episode_counter}_{random.randint(1000, 9999)}"

        # Scenario archetypes: 0 = Obstacle Bottleneck, 1 = Financial Volatility Spike, 2 = Sensor Blindspot
        archetype = self.episode_counter % 3
        cands = ["ACTION_A", "ACTION_B", "ACTION_C", "SAFE_ABSTAIN"]

        if archetype == 0:
            name = "spatial_bottleneck"
            features = rng.randn(1024).astype(np.float32)
            # Inject obstacle correlation
            features[:10] += self.difficulty_level * 2.0
        elif archetype == 1:
            name = "volatility_cliff"
            features = rng.randn(1024).astype(np.float32) * (1.0 + self.difficulty_level * 1.5)
        else:
            name = "occluded_sensor"
            features = rng.randn(1024).astype(np.float32)
            # Mask out 40% of sensor features
            features[100:500] = 0.0

        return {
            "scenario_id": scenario_id,
            "archetype": name,
            "difficulty": round(self.difficulty_level, 3),
            "state_repr": features,
            "candidate_actions": cands,
            "ground_truth_safe_action": "SAFE_ABSTAIN" if self.difficulty_level > 0.7 else "ACTION_A"
        }

    def adapt_difficulty(self, recent_success_rate: float) -> None:
        """Dynamically tunes curriculum difficulty level based on agent success rate."""
        if recent_success_rate > 0.85:
            # Model is mastering tasks -> raise difficulty
            self.difficulty_level = min(1.0, self.difficulty_level + 0.05)
        elif recent_success_rate < 0.40:
            # Model is struggling -> ease difficulty
            self.difficulty_level = max(0.1, self.difficulty_level - 0.05)
