"""Gen-Zero Planning Engine 4: Unified ManifoldGFlowNetEngine.

Convergence and full upgrade over gflownet.py via RFC-085 Continuous Manifold GFlowNet:
1. Simplex ETF Geodesic Flow:
   - Eliminates discrete random walks. Actions projected onto regular simplex ETF on S^(d-1).
   - Continuous SLERP geodesic evolution preserves permutation equivariance.
2. Trajectory Balance (TB) Objective:
   - Log Z_0 + sum log P_F = log R(s_T) + sum log P_B.
   - Strictly samples trajectories proportional to terminal reward P(tau) propto R(s_T).
3. Diversity & Mode Coverage (measured, not guaranteed):
   - ``mode_coverage`` is the fraction of candidate first actions that start at least one
     sampled trajectory with reward >= 70% of the sampled max. It is an empirical figure over
     the drawn samples, not proof of full coverage of the true reward modes.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from gen_zero.planner.continuous_gflownet import (
    ContinuousGFlowNetAdapter,
    ContinuousManifoldGFlowNetSampler,
    GFlowNetSamplingResult,
    GFlowNetTrajectory,
)


class ManifoldGFlowNetEngine:
    """Unified Continuous Manifold GFlowNet Planning Engine."""

    def __init__(
        self,
        dim: int = 128,
        temperature: float = 1.0,
        slerp_step_size: float = 0.35,
        exploration_noise: float = 0.03,
        blend_alpha: float = 0.95,
        seed: int = 42,
    ):
        self.dim = dim
        self.temperature = temperature
        self.sampler = ContinuousManifoldGFlowNetSampler(
            dim=dim,
            temperature=temperature,
            slerp_step_size=slerp_step_size,
            exploration_noise=exploration_noise,
            blend_alpha=blend_alpha,
            seed=seed,
        )
        self.adapter = ContinuousGFlowNetAdapter(temperature=temperature, dim=dim)

    def plan(
        self,
        state: Any,
        candidate_actions: List[str],
        transition_fn: Callable[[Any, str], Tuple[Any, float, bool]],
        sample_count: int = 16,
        max_horizon: int = 3,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Unified planning entrypoint generating diverse trajectories and best action."""
        t0 = time.perf_counter()
        if temperature is not None:
            self.sampler.temperature = float(temperature)

        result: GFlowNetSamplingResult = self.sampler.sample_diverse_trajectories(
            initial_state=state,
            candidate_actions=candidate_actions,
            transition_fn=transition_fn,
            sample_count=sample_count,
            max_horizon=max_horizon,
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "best_action": result.best_action,
            "action_probs": result.action_probs,
            "trajectory_entropy": round(result.trajectory_entropy, 4),
            "mode_coverage": round(result.mode_coverage, 4),
            "modes_discovered": result.modes_discovered,
            "counterfactual_branches": [t.to_dict() for t in result.counterfactual_branches],
            "total_flow_z": round(result.total_flow_z, 4),
            "permutation_flip_rate": round(result.permutation_flip_rate, 4),
            "latency_ms": latency_ms,
            "mode": "manifold_gflownet",
        }

    def sample_trajectories(
        self,
        state: Any,
        candidate_actions: List[str],
        transition_fn: Callable[[Any, str], Tuple[Any, float, bool]],
        sample_count: int = 16,
        max_horizon: int = 3,
    ) -> GFlowNetSamplingResult:
        """Directly samples full GFlowNetSamplingResult dataclass."""
        return self.sampler.sample_diverse_trajectories(
            initial_state=state,
            candidate_actions=candidate_actions,
            transition_fn=transition_fn,
            sample_count=sample_count,
            max_horizon=max_horizon,
        )
