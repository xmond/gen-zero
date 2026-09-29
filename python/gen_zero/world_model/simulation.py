"""Explicit world-model rollouts behind GenZero.simulate / what_if / audit_action.

One rollout loop serves every endpoint. The transition function is injected, so the
same loop runs over the neural dynamics model, the heuristic text world model, or a
caller-supplied simulator.

Two terminal notions are kept apart on purpose:

- ``done``: the transition ended the episode. The heuristic text world model ends
  every text episode after one step with a positive reward, so ``done`` alone does
  not mean death.
- ``hazard``: the transition was lethal or irreversible. Neural dynamics: ``done``
  (defined there as ``r_hat < done_threshold``). Heuristic model: ``done`` with a
  negative reward (the ``p_fail > 0.5`` branch). Caller simulator: ``done``, because
  nothing else is known about it, so an unknown terminal counts as a hazard.

``is_safe``, ``first_hazard_step`` and trap detection read ``hazard``. The rollout
stops at the first ``done`` of either kind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

SOURCE_NEURAL = "neural_dynamics"
SOURCE_HEURISTIC = "heuristic_text_world_model"
SOURCE_CALLER = "caller_transition_fn"

# Provenance: which simulator produced a step, for callers that must not treat every
# source as equally trustworthy (e.g. audit_action refusing APPROVED on an uncalibrated
# text heuristic). Distinct from `source` above, which only tags safe_prob semantics.
PROVENANCE_NEURAL = "neural_residual_dynamics"
PROVENANCE_GRID = "symbolic_grid_world"
PROVENANCE_TEXT = "text_heuristic"
PROVENANCE_CALLER = "caller_custom_transition"


@dataclass(frozen=True)
class StepOutcome:
    next_state: Any
    reward: float
    done: bool
    # None when the simulator exposes no safety estimate (caller transition_fn).
    safe_prob: Optional[float]
    hazard: bool
    source: str
    provenance: str


StepFn = Callable[[Any, Any], StepOutcome]
Policy = Callable[[int, Any], Any]


def to_jsonable(value: Any) -> Any:
    """ndarray / numpy scalars / tuples -> plain lists and floats; recurse into containers."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def wrap_caller_transition(fn: Callable[[Any, Any], Any]) -> StepFn:
    """Adapt a (state, action) -> (next_state, reward, done) simulator to StepOutcome."""

    def step(state: Any, action: Any) -> StepOutcome:
        result = fn(state, action)
        if not isinstance(result, tuple) or len(result) != 3:
            raise ValueError(f"transition_fn must return (next_state, reward, done), got {type(result).__name__}")
        next_state, reward, done = result
        done = bool(done)
        return StepOutcome(next_state, _finite_reward(reward), done, None, done, SOURCE_CALLER, PROVENANCE_CALLER)

    return step


def _finite_reward(reward: Any) -> float:
    value = float(reward)
    if not math.isfinite(value):
        raise ValueError(f"transition returned a non-finite reward: {value}")
    return value


def validate_horizon(horizon: Any) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon <= 0:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
    return int(horizon)


def rollout(state: Any, horizon: int, step_fn: StepFn, policy: Policy) -> Dict[str, Any]:
    """Run up to ``horizon`` imagined steps; ``policy(step_idx, state)`` names each action.

    ``step_idx`` starts at 1. The loop stops at the first ``done``. ``survival_horizon``
    counts steps completed before the first hazard; a non-lethal episode end counts as
    survived.
    """
    horizon = validate_horizon(horizon)
    trajectory: List[Dict[str, Any]] = []
    current = state
    cumulative = 0.0
    termination_step: Optional[int] = None
    first_hazard_step: Optional[int] = None
    sources = set()
    provenances = set()
    safe_probs: List[float] = []

    for step_idx in range(1, horizon + 1):
        action = policy(step_idx, current)
        outcome = step_fn(current, action)
        reward = _finite_reward(outcome.reward)
        cumulative += reward
        sources.add(outcome.source)
        provenances.add(outcome.provenance)
        if outcome.safe_prob is not None:
            safe_probs.append(float(outcome.safe_prob))
        trajectory.append({
            "step_idx": step_idx,
            "state": to_jsonable(outcome.next_state),
            "action": to_jsonable(action),
            "reward": reward,
            "safe_prob": None if outcome.safe_prob is None else float(outcome.safe_prob),
            "done": bool(outcome.done),
            "hazard": bool(outcome.hazard),
        })
        current = outcome.next_state
        if outcome.hazard and first_hazard_step is None:
            first_hazard_step = step_idx
        if outcome.done:
            termination_step = step_idx
            break

    hazard = first_hazard_step is not None
    return {
        "trajectory": trajectory,
        "steps_simulated": len(trajectory),
        "survival_horizon": (first_hazard_step - 1) if hazard else len(trajectory),
        "cumulative_return": cumulative,
        "terminated_early": termination_step is not None,
        "termination_step": termination_step,
        "is_safe": not hazard,
        "first_hazard_step": first_hazard_step,
        "min_safe_prob": min(safe_probs) if safe_probs else None,
        "safe_prob_source": sorted(sources),
        "provenance": sorted(provenances),
        "final_state": to_jsonable(current),
    }


def fixed_plan_policy(actions: Sequence[Any]) -> Policy:
    """Replay a given action list; step_idx is 1-based."""
    return lambda step_idx, _state: actions[step_idx - 1]


def greedy_lookahead_policy(first_action: Any, action_set: Sequence[Any], step_fn: StepFn) -> Policy:
    """First step plays ``first_action``; later steps pick the one-step best in ``action_set``.

    Ranking: non-hazard first, then higher safe_prob, then higher reward. Candidates
    without a safe_prob (caller simulator) rank on hazard and reward only. Ties keep
    ``action_set`` order.
    """
    if not action_set:
        raise ValueError("greedy continuation needs a non-empty action set")

    def score(outcome: StepOutcome):
        safe = outcome.safe_prob if outcome.safe_prob is not None else 0.0
        return (not outcome.hazard, safe, outcome.reward)

    def policy(step_idx: int, state: Any) -> Any:
        if step_idx == 1:
            return first_action
        return max(action_set, key=lambda a: score(step_fn(state, a)))

    return policy


def summarize_candidate(result: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("survival_horizon", "cumulative_return", "terminated_early", "is_safe",
            "first_hazard_step", "final_state", "min_safe_prob", "trajectory", "provenance")
    return {k: result[k] for k in keys}


def rank_candidates(outcomes: Dict[str, Dict[str, Any]]) -> List[str]:
    """Safe before trapped, then longer survival, then higher return. Stable on ties."""
    return sorted(
        outcomes,
        key=lambda c: (outcomes[c]["is_safe"], outcomes[c]["survival_horizon"], outcomes[c]["cumulative_return"]),
        reverse=True,
    )
