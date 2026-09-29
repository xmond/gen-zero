"""Gen-Zero World Model: Lightweight Discrete/Text World Model (heuristic baseline).

Provides virtual simulation and consequence scores in black-box environments:
1. Scores p_fail(s, a), p_reward(s, a), and space_margin(s, a).

These scores come from hand-set rules, not from a model fit to data. They are a heuristic
baseline: no calibration against observed outcomes has been done, so do not read them as
probabilities. Text states use keyword matching and fixed constants (0.3 / 0.05 / 0.8 / 0.4).
2. Generates imagined state transitions for planning without environment access.
3. Supports adaptive simulation horizon and multi-step lookahead rollouts.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, List, Optional, Tuple


class GenZeroTextWorldModel:
    """Heuristic-baseline text/attribute world model for virtual lookahead (not trained, not calibrated).

    Every number ``predict_consequences`` returns is a hand-set constant or a hand-set
    linear-logistic weight, never fit or calibrated against observed outcomes. Constructing
    this class requires ``acknowledge_uncalibrated=True`` so a caller cannot end up quoting
    these numbers as calibrated confidence (e.g. to a user, a safety gate threshold, or a
    metric dashboard) without having read this warning first.
    """

    #: Text-state consequence constants (keyword-matched branch of predict_consequences).
    #: Not fit or calibrated against any observed outcome; see class docstring.
    TEXT_P_FAIL_RISK_KEYWORD_ACTION: float = 0.3
    TEXT_P_FAIL_DEFAULT: float = 0.05
    TEXT_P_REWARD_SAFE_ACTION: float = 0.8
    TEXT_P_REWARD_DEFAULT: float = 0.4
    TEXT_SPACE_MARGIN_CONSTANT: float = 0.8
    TEXT_RISK_KEYWORDS = ("reboot", "delete", "drop", "kill", "heater")
    TEXT_HIGH_RISK_ACTIONS = ("approve", "execute")
    TEXT_SAFE_VERDICT_ACTIONS = ("escalate", "verify")

    def __init__(self, feature_weights: Optional[List[float]] = None, *, acknowledge_uncalibrated: bool = False):
        if not acknowledge_uncalibrated:
            raise ValueError(
                "GenZeroTextWorldModel returns hand-set, uncalibrated heuristic constants "
                "(see class docstring) -- never numbers fit or calibrated against observed "
                "outcomes. Construct with acknowledge_uncalibrated=True to confirm the caller "
                "will not present predict_consequences() output as calibrated confidence."
            )
        # Heuristic baseline: hand-set linear-logistic weights, never fit or calibrated on data.
        self.w_fail = feature_weights or [1.8, -2.4, 0.2, -5.0]
        self.b_fail = 1.2
        self.w_reward = [0.0, 0.0, -2.5, -3.0]
        self.b_reward = 1.5

    def predict_consequences(self, state: Any, action: str) -> Dict[str, float]:
        """Scores consequences of an action with the heuristic baseline (uncalibrated scores).

        Every returned value is a hand-set constant or hand-set linear-logistic score -- see
        the class docstring. Not a calibrated probability.
        """
        # Generic state-action consequence extractor
        if isinstance(state, str):
            # Text state: keyword match plus fixed constants, no learned semantics.
            is_risk = any(w in state.lower() for w in self.TEXT_RISK_KEYWORDS)
            return {
                "p_fail": self.TEXT_P_FAIL_RISK_KEYWORD_ACTION if is_risk and action in self.TEXT_HIGH_RISK_ACTIONS else self.TEXT_P_FAIL_DEFAULT,
                "p_reward": self.TEXT_P_REWARD_SAFE_ACTION if action in self.TEXT_SAFE_VERDICT_ACTIONS or not is_risk else self.TEXT_P_REWARD_DEFAULT,
                "space_margin": self.TEXT_SPACE_MARGIN_CONSTANT,
            }

        state_dict = state if isinstance(state, dict) else {}
        body = state_dict.get("body") or [[0, 0]]
        head = body[0] if body else [0, 0]
        size = state_dict.get("size", 8)

        # Approximate direction offset
        dirs = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}
        dr, dc = dirs.get(action, (0, 0))
        nr, nc = head[0] + dr, head[1] + dc

        dist_wall = min(nr, size - 1 - nr, nc, size - 1 - nc)
        is_oob = 1.0 if not (0 <= nr < size and 0 <= nc < size) else 0.0

        body_set = {tuple(c) for c in state_dict.get("body", [])[:-1]}
        obs_set = {tuple(c) for c in state_dict.get("obstacles", [])}
        is_body = 1.0 if (nr, nc) in body_set else 0.0
        is_obs = 1.0 if (nr, nc) in obs_set else 0.0

        food = state_dict.get("food", [0, 0])
        curr_dist = abs(head[0] - food[0]) + abs(head[1] - food[1])
        next_dist = abs(nr - food[0]) + abs(nc - food[1])
        dist_delta = float(next_dist - curr_dist)

        opp = {"north": "south", "south": "north", "east": "west", "west": "east"}
        is_reverse = 1.0 if action == opp.get(state_dict.get("direction")) else 0.0

        feats = [float(dist_wall), float(is_oob * 5.0 + is_body * 5.0 + is_obs * 5.0), dist_delta, is_reverse]

        z_fail = sum(w * f for w, f in zip(self.w_fail, feats)) + self.b_fail
        p_fail = 1.0 - (1.0 / (1.0 + math.exp(-max(min(z_fail, 15.0), -15.0))))

        z_rew = sum(w * f for w, f in zip(self.w_reward, feats)) + self.b_reward
        p_rew = 1.0 / (1.0 + math.exp(-max(min(z_rew, 15.0), -15.0)))

        space_margin = max(0.0, min(1.0, (dist_wall + 1.0) / (size / 2.0)))

        return {
            "p_fail": p_fail,
            "p_reward": p_rew,
            "space_margin": space_margin,
        }

    def virtual_step(self, state: Any, action: str) -> Tuple[Any, float, bool]:
        """Performs a virtual step inside the world model."""
        conseq = self.predict_consequences(state, action)
        if conseq["p_fail"] > 0.5:
            return state, -5.0, True

        if isinstance(state, str):
            reward = 1.0 if conseq["p_reward"] > 0.5 else 0.1
            return state + f" -> {action}", reward, True

        dirs = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}
        dr, dc = dirs.get(action, (0, 0))

        if isinstance(state, (tuple, list)):
            next_pos = (state[0] + dr, state[1] + dc)
            return next_pos, 0.1, False

        state_dict = state if isinstance(state, dict) else {}
        body = state_dict.get("body") or [[0, 0]]
        head = body[0] if body else [0, 0]
        new_head = [head[0] + dr, head[1] + dc]
        ate_reward = conseq["p_reward"] > 0.65

        new_body = [new_head] + [list(c) for c in state_dict.get("body", [])]
        if not ate_reward and len(new_body) > len(state_dict.get("body", [])):
            new_body.pop()

        virtual_next_state = copy.deepcopy(state_dict)
        virtual_next_state["body"] = new_body
        virtual_next_state["direction"] = action
        virtual_next_state["score"] = state_dict.get("score", 0) + (1 if ate_reward else 0)
        virtual_next_state["steps"] = state_dict.get("steps", 0) + 1

        reward = (2.0 if ate_reward else 0.1) + conseq["space_margin"] * 0.4
        return virtual_next_state, reward, False

    def determine_adaptive_horizon(
        self,
        state: Any,
        candidates: List[str],
        complexity_score: Optional[float] = None,
        task_hint: Optional[str] = None,
    ) -> int:
        """Determines the optimal simulation horizon adaptively based on state complexity."""
        hint_str = (task_hint or "").lower()

        # Check explicit trap depth in state
        if isinstance(state, dict) and "trap_depth" in state:
            return max(3, min(12, int(state["trap_depth"]) + 2))

        # Check financial / trading volatility or sequence
        if "trade" in hint_str or "quant" in hint_str or "stock" in hint_str or (isinstance(state, dict) and "prices" in state):
            return 5

        # Check complexity score
        if complexity_score is not None:
            raw_h = int(1 + 9 * complexity_score)
            return max(1, min(10, raw_h))

        # Fallback inspection of obstacles
        if isinstance(state, dict) and "obstacles" in state:
            obs_cnt = len(state.get("obstacles", []))
            return max(3, min(8, 3 + obs_cnt // 2))

        return 4

    def adaptive_rollout(
        self,
        state: Any,
        action: str,
        horizon: int,
        trans_fn: Callable,
        gamma: float = 0.90,
    ) -> Dict[str, Any]:
        """Performs dynamic-horizon rollout with early-stopping & quiescence extension."""
        curr_state = state
        curr_action = action
        total_return = 0.0
        discount = 1.0
        steps = 0
        early_stopped = False
        quiescence_extended = False
        terminal = False

        max_steps = max(1, horizon)
        step_rewards = []
        has_traps = isinstance(state, dict) and ("obstacles" in state or "trap_depth" in state)
        opp = {"north": "south", "south": "north", "east": "west", "west": "east"}

        while steps < max_steps:
            next_state, reward, done = trans_fn(curr_state, curr_action)
            total_return += discount * reward
            step_rewards.append(reward)
            steps += 1
            if done:
                if reward < 0:
                    total_return += discount * -10.0
                terminal = True
                break

            discount *= gamma
            curr_state = next_state

            if not has_traps and steps >= 2 and len(step_rewards) >= 2:
                if step_rewards[-1] > 0.5 and step_rewards[-2] > 0.5 and abs(step_rewards[-1] - step_rewards[-2]) < 0.05:
                    early_stopped = True
                    break

            if isinstance(curr_state, dict) and "body" in curr_state:
                best_next_a = curr_action
                best_next_score = -1e9
                for a_cand in ["north", "south", "east", "west"]:
                    if a_cand == opp.get(curr_state.get("direction")):
                        continue
                    con = self.predict_consequences(curr_state, a_cand)
                    cand_score = con["p_reward"] * 2.0 - 10.0 * con["p_fail"]
                    if cand_score > best_next_score:
                        best_next_score = cand_score
                        best_next_a = a_cand
                curr_action = best_next_a

        if not terminal and not early_stopped and steps == max_steps:
            conseq = self.predict_consequences(curr_state, curr_action)
            if conseq["p_fail"] > 0.35:
                quiescence_extended = True
                for _ in range(2):
                    next_state, reward, done = trans_fn(curr_state, curr_action)
                    total_return += discount * reward
                    steps += 1
                    if done:
                        if reward < 0:
                            total_return += discount * -10.0
                        terminal = True
                        break
                    discount *= gamma
                    curr_state = next_state

        return {
            "cumulative_return": round(total_return, 4),
            "steps_simulated": steps,
            "horizon_budget": max_steps,
            "early_stopped": early_stopped,
            "quiescence_extended": quiescence_extended,
            "terminal": terminal,
        }
