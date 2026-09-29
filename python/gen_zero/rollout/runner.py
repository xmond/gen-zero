"""Gen-Zero Layer 3: Unified Multi-Environment Rollout Runner.

Executes online rollouts without artificial candidate narrowing (SM / S2 discipline),
calling MCTS search for test-time deliberation and recording full step trajectories.
"""

import copy
import math
from typing import Dict, List, Any, Optional, Callable
from .hard_miner import HardSampleMiner, MinedSample


class UnifiedEnvironmentRunner:
    """Runs closed-loop episodes and mines hard samples."""
    def __init__(self, miner: Optional[HardSampleMiner] = None):
        self.miner = miner or HardSampleMiner()

    def run_episode(
        self,
        env: Any,
        policy_fn: Callable[[Any, List[str]], Dict[str, Any]],
        max_steps: int = 256
    ) -> Dict[str, Any]:
        """Runs a single episode under the policy/planner.

        Args:
            env: Object implementing:
                - reset() -> state
                - step(action) -> (next_state, reward, done, outcome)
                - get_legal_actions(state) -> list of strings
            policy_fn: (state, legal_actions) -> {
                'action': chosen_action,
                'mcts_probs': dict or None,
                'model_probs': dict or None,
                'model_value': float
            }
        """
        state = env.reset()
        trajectory = []
        steps = 0
        final_outcome = "horizon_survived"
        total_reward = 0.0

        while steps < max_steps:
            legal = env.get_legal_actions(state)
            if not legal:
                final_outcome = "trapped"
                break

            decision = policy_fn(state, legal)
            action = decision.get("action")
            if action is None or action not in legal:
                action = legal[0]

            trajectory.append({
                "step_idx": steps,
                "state": copy.deepcopy(state),
                "candidates": legal,
                "action": action,
                "mcts_probs": decision.get("mcts_probs"),
                "model_probs": decision.get("model_probs"),
                "model_value": decision.get("model_value", 0.0)
            })

            next_state, reward, done, outcome = env.step(action)
            total_reward += reward
            steps += 1
            state = next_state

            if done:
                final_outcome = outcome or ("win" if reward > 0 else "collision")
                break

        # Mine hard learning samples from this trajectory
        mined_samples = self.miner.mine_trajectory(
            trajectory=trajectory,
            final_outcome=final_outcome,
            final_score=total_reward
        )

        return {
            "steps": steps,
            "outcome": final_outcome,
            "total_reward": total_reward,
            "trajectory_length": len(trajectory),
            "mined_samples": mined_samples
        }
