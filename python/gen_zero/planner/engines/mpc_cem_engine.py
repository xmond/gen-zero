"""Gen-Zero Planning Engine 3: Unified MpcCemEngine.

Convergence of continuous_mpc.py and mpc_cem.py:
1. Unified Continuous & Discrete CEM:
   - Continuous latent/action trajectory optimization in R^{H x D} with covariance adaptation.
   - Discrete categorical action distribution CEM for discrete action candidate selection.
2. Receding Horizon Rolling Window:
   - receding_step() executes the first action and rolls the horizon forward.
3. Covariance Adaptation & Elite Filtering:
   - Momentum-smoothed distribution updates: mu^(k) = beta * mu^(k-1) + (1-beta) * mu_elites.
   - Simplex projection (sum(w_i)=1, w_i>=0) and box constraint clipping.
"""

from __future__ import annotations

import math
import random
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

class MpcMode(str, Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


class MpcCemEngine:
    """Unified Model Predictive Control & Cross-Entropy Method Optimization Engine."""

    def __init__(
        self,
        action_dim: int = 4,
        horizon: int = 8,
        num_samples: int = 48,
        elite_ratio: float = 0.20,
        iterations: int = 4,
        momentum: float = 0.50,
        latent_model: Optional[Any] = None,
        seed: int = 42,
    ):
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.elite_ratio = elite_ratio
        self.iterations = iterations
        self.momentum = momentum
        self.latent_model = latent_model
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def plan(
        self,
        state: Any,
        candidate_actions_or_bounds: Any = None,
        transition_fn: Optional[Callable[[Any, Any], Tuple[Any, float, bool]]] = None,
        reward_fn: Optional[Callable[[Any, Any, Any], float]] = None,
        horizon: Optional[int] = None,
        num_samples: Optional[int] = None,
        simplex: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Polymorphic planning entrypoint.

        Dispatches to:
        - Discrete action planning if candidate_actions_or_bounds is a list of strings/actions.
        - Continuous trajectory planning if candidate_actions_or_bounds is a tuple of (min, max) or None.
        """
        if isinstance(candidate_actions_or_bounds, (list, tuple)) and (
            not candidate_actions_or_bounds or isinstance(candidate_actions_or_bounds[0], (str, int))
        ):
            return self.plan_discrete(
                initial_state=state,
                candidate_actions=list(candidate_actions_or_bounds),
                transition_fn=transition_fn or (lambda s, a: (s, 0.0, False)),
                reward_fn=reward_fn,
                horizon=horizon,
                num_samples=num_samples,
                **kwargs,
            )

        # Pop these out of kwargs before re-passing: plan_continuous receives
        # them as explicit keyword args below, and forwarding both would raise
        # "got multiple values for keyword argument".
        kwarg_bounds = kwargs.pop("bounds", (-1.0, 1.0))
        bounds = candidate_actions_or_bounds if isinstance(candidate_actions_or_bounds, tuple) else kwarg_bounds
        custom_reward_fn = kwargs.pop("custom_reward_fn", None)
        return self.plan_continuous(
            state=state,
            bounds=bounds,
            simplex=simplex,
            horizon=horizon,
            num_samples=num_samples,
            custom_reward_fn=custom_reward_fn,
            **kwargs,
        )

    def plan_continuous(
        self,
        state: Any,
        action_dim: Optional[int] = None,
        bounds: Optional[Tuple[float, float]] = (-1.0, 1.0),
        simplex: bool = False,
        horizon: Optional[int] = None,
        num_samples: Optional[int] = None,
        custom_reward_fn: Optional[Callable[[List[float], List[float]], float]] = None,
        policy_entropy: Optional[float] = None,
        volatility: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Plans optimal continuous action trajectory a_{0:H-1} via iterative CEM."""
        t0 = time.perf_counter()
        H = horizon or self.horizon
        N = num_samples or self.num_samples
        num_elites = max(1, int(N * self.elite_ratio))
        D = action_dim if action_dim is not None else self.action_dim

        # Initial distribution
        mu = np.zeros((H, D), dtype=np.float32)
        std = np.ones((H, D), dtype=np.float32)

        # Volatility adaptation
        if volatility is not None and volatility > 0.05:
            std *= min(2.0, 1.0 + volatility * 5.0)

        best_overall_score = -float("inf")
        best_overall_trajectory = mu.copy()

        # State latent representation
        z_init = self._extract_latent(state, D)

        for it in range(self.iterations):
            # Sample N action trajectories
            samples = self.np_rng.normal(mu, std, size=(N, H, D)).astype(np.float32)

            if simplex:
                # Softmax projection onto simplex
                exp_s = np.exp(samples - np.max(samples, axis=-1, keepdims=True))
                samples = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
            elif bounds is not None:
                samples = np.clip(samples, bounds[0], bounds[1])

            # Evaluate trajectory returns
            scores = np.zeros(N, dtype=np.float32)
            for i in range(N):
                traj = samples[i]
                r_sum = 0.0
                curr_z = z_init.copy()
                for step in range(H):
                    act = traj[step]
                    if custom_reward_fn is not None:
                        r = custom_reward_fn(curr_z.tolist(), act.tolist())
                    else:
                        # Default reward: energy penalty + alignment
                        r = float(-0.5 * np.sum(act**2) + np.dot(curr_z[:D], act))
                    r_sum += r
                    # Step latent state
                    curr_z = curr_z + 0.1 * act
                scores[i] = r_sum

            # Fail-closed: a NaN/Inf return anywhere in this iteration's batch
            # (from custom_reward_fn or the default reward) must never be
            # ranked with argsort (NaN keys sort unpredictably) or allowed to
            # refit mu/std from garbage elites. Abort the whole plan instead
            # of silently returning a trajectory chosen on corrupt scores.
            if not np.all(np.isfinite(scores)):
                return self._continuous_abstain_result(
                    H, N, t0, "NON_FINITE_TRANSITION_REWARD"
                )

            # Elite selection
            elite_indices = np.argsort(scores)[-num_elites:]
            elites = samples[elite_indices]

            # Track global best
            if scores[elite_indices[-1]] > best_overall_score:
                best_overall_score = float(scores[elite_indices[-1]])
                best_overall_trajectory = samples[elite_indices[-1]].copy()

            # Refit distribution with momentum smoothing
            mu_elites = np.mean(elites, axis=0)
            std_elites = np.std(elites, axis=0) + 1e-4

            mu = self.momentum * mu + (1.0 - self.momentum) * mu_elites
            std = self.momentum * std + (1.0 - self.momentum) * std_elites

        # Defensive fail-closed guard: best_overall_score only leaves -inf once
        # an iteration's elites are ranked above it. With iterations=0 (or an
        # otherwise degenerate N=0 run) no ranking ever happens, so there is no
        # verified best trajectory to report as if a plan had actually run.
        if not math.isfinite(best_overall_score):
            return self._continuous_abstain_result(
                H, N, t0, "NO_FEASIBLE_CANDIDATES"
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        best_first_action = best_overall_trajectory[0].tolist()

        return {
            "mode": MpcMode.CONTINUOUS.value,
            "action": best_first_action,
            "best_action": best_first_action,
            "best_trajectory": best_overall_trajectory.tolist(),
            "expected_return": float(best_overall_score),
            "horizon": H,
            "num_samples": N,
            "iterations": self.iterations,
            "latency_ms": latency_ms,
            "status": "OK",
        }

    def _continuous_abstain_result(self, H: int, N: int, t0: float, status: str) -> Dict[str, Any]:
        """Fail-closed continuous-plan result: no verified action to report."""
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "mode": MpcMode.CONTINUOUS.value,
            "action": None,
            "best_action": None,
            "best_trajectory": None,
            "expected_return": float("nan"),
            "horizon": H,
            "num_samples": N,
            "iterations": self.iterations,
            "latency_ms": latency_ms,
            "status": status,
        }

    def plan_discrete(
        self,
        initial_state: Any,
        candidate_actions: List[str],
        transition_fn: Callable[[Any, str], Tuple[Any, float, bool]],
        reward_fn: Optional[Callable[[Any, str, Any], float]] = None,
        horizon: Optional[int] = None,
        num_samples: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Plans discrete action sequence using categorical CEM distribution fitting."""
        t0 = time.perf_counter()
        if not candidate_actions:
            return {
                "mode": MpcMode.DISCRETE.value,
                "best_action": None,
                "plan_trajectory": [],
                "expected_return": 0.0,
                "action_probs": {},
                "latency_ms": 0.0,
                "status": "NO_CANDIDATES",
            }

        H = horizon or self.horizon
        N = num_samples or self.num_samples
        num_elites = max(1, int(N * self.elite_ratio))
        num_actions = len(candidate_actions)

        # Categorical probabilities per horizon step
        probs = [[1.0 / num_actions for _ in range(num_actions)] for _ in range(H)]

        best_overall_score = -float("inf")
        best_overall_seq: List[str] = []

        for it in range(self.iterations):
            samples: List[Tuple[List[str], float]] = []

            for _ in range(N):
                seq: List[str] = []
                for t in range(H):
                    p_vec = probs[t]
                    chosen_idx = self.rng.choices(range(num_actions), weights=p_vec, k=1)[0]
                    seq.append(candidate_actions[chosen_idx])

                # Rollout sequence
                curr_s = initial_state
                cum_return = 0.0
                gamma = 1.0

                for act in seq:
                    next_s, r, done = transition_fn(curr_s, act)
                    if reward_fn is not None:
                        step_r = reward_fn(curr_s, act, next_s)
                    else:
                        step_r = r
                    step_r = float(step_r)
                    # Fail-closed: a NaN/Inf step reward from transition_fn or
                    # reward_fn must abort the whole plan immediately. Letting
                    # it propagate would poison cum_return, and later sorting
                    # samples on a NaN key is undefined (NaN compares False to
                    # everything), silently corrupting elite selection.
                    if not math.isfinite(step_r):
                        return self._discrete_abstain_result(
                            H, t0, "NON_FINITE_REWARD"
                        )
                    cum_return += gamma * step_r
                    gamma *= 0.95
                    curr_s = next_s
                    if done:
                        break

                if not math.isfinite(cum_return):
                    return self._discrete_abstain_result(
                        H, t0, "NON_FINITE_REWARD"
                    )

                samples.append((seq, cum_return))

            # Rank and select elites
            samples.sort(key=lambda x: x[1], reverse=True)
            elites = samples[:num_elites]

            if elites[0][1] > best_overall_score:
                best_overall_score = elites[0][1]
                best_overall_seq = elites[0][0]

            # Update categorical distribution
            for t in range(H):
                counts = [0.1] * num_actions
                for seq, _ in elites:
                    act = seq[t]
                    idx = candidate_actions.index(act)
                    counts[idx] += 1.0
                total_c = sum(counts)
                new_p = [c / total_c for c in counts]
                probs[t] = [
                    self.momentum * probs[t][a] + (1.0 - self.momentum) * new_p[a]
                    for a in range(num_actions)
                ]

        # Fail-closed: candidate_actions[0] is an arbitrary, unverified pick,
        # not a plan result. If no elite sequence was ever ranked (e.g.
        # iterations=0), report an explicit abstain status instead of a
        # fabricated "best" action.
        if not best_overall_seq:
            return self._discrete_abstain_result(H, t0, "NO_FEASIBLE_CANDIDATES")

        latency_ms = (time.perf_counter() - t0) * 1000.0
        best_action = best_overall_seq[0]
        # Real fitted first-step categorical distribution from the CEM elite
        # updates above (probs[0]), not an assumed constant: this is the actual
        # empirical action frequency among elite rollouts after `iterations` of
        # cross-entropy refinement.
        action_probs = {candidate_actions[i]: float(probs[0][i]) for i in range(num_actions)}

        return {
            "mode": MpcMode.DISCRETE.value,
            "best_action": best_action,
            "plan_trajectory": best_overall_seq,
            "expected_return": float(best_overall_score),
            "action_probs": action_probs,
            "horizon": H,
            "latency_ms": latency_ms,
            "status": "OK",
        }

    def _discrete_abstain_result(self, H: int, t0: float, status: str) -> Dict[str, Any]:
        """Fail-closed discrete-plan result: no verified action to report."""
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "mode": MpcMode.DISCRETE.value,
            "best_action": None,
            "plan_trajectory": [],
            "expected_return": float("nan"),
            "action_probs": {},
            "horizon": H,
            "latency_ms": latency_ms,
            "status": status,
        }

    def receding_step(
        self,
        current_state: Any,
        transition_fn: Callable[[Any, Any], Tuple[Any, float, bool]],
        candidate_actions_or_bounds: Any = None,
        reward_fn: Optional[Callable[[Any, Any, Any], float]] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """Executes a single receding horizon step.

        Plans full trajectory, executes first action, and transitions to next state.
        Returns: (first_action, next_state, plan_info).
        """
        plan_res = self.plan(
            state=current_state,
            candidate_actions_or_bounds=candidate_actions_or_bounds,
            transition_fn=transition_fn,
            reward_fn=reward_fn,
            **kwargs,
        )
        action = plan_res.get("best_action")
        if action is None:
            # Fail-closed: the plan produced no verified action (see its
            # "status", e.g. NON_FINITE_REWARD / NON_FINITE_TRANSITION_REWARD /
            # NO_FEASIBLE_CANDIDATES / NO_CANDIDATES). Executing transition_fn with a None/blind action
            # would be a step no plan actually endorsed. Propagate the abstain
            # status untouched and stay at the current state instead.
            plan_res["step_reward"] = None
            plan_res["done"] = False
            return None, current_state, plan_res
        next_state, reward, done = transition_fn(current_state, action)
        plan_res["step_reward"] = reward
        plan_res["done"] = done
        return action, next_state, plan_res

    def _extract_latent(self, state: Any, dim: int) -> np.ndarray:
        """Extracts fixed-dimensional numpy array from arbitrary state."""
        if isinstance(state, np.ndarray):
            if state.shape[0] >= dim:
                return state[:dim].astype(np.float32)
            res = np.zeros(dim, dtype=np.float32)
            res[: len(state)] = state
            return res
        if isinstance(state, list):
            res = np.zeros(dim, dtype=np.float32)
            for i in range(min(dim, len(state))):
                res[i] = float(state[i])
            return res
        return np.zeros(dim, dtype=np.float32)
