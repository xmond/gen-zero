"""Continuous Simplex ETF trajectory sampler with a trajectory-balance objective.

Embeds actions using simplex geometry and evolves latent states using spherical
interpolation (with a numerical linear branch near collinearity). Trajectory
balance motivates reward-proportional terminal-state sampling when its flow
consistency assumptions and optimization conditions hold. Finite samples and
updates in this implementation do not establish those conditions or uniform
exploration. Diversity, mode coverage, permutation behavior, and latency require
workload-specific measurement; no quantitative improvement is guaranteed.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from gen_zero.nanocore.action_etf_embedding import (
    ActionSpaceETFEmbedding,
    generate_simplex_etf,
)


def slerp(p0: np.ndarray, p1: np.ndarray, t: float) -> np.ndarray:
    """Spherical Linear Interpolation (SLERP) on unit sphere S^(d-1).

    Interpolates between unit vectors p0 and p1 along the minimal geodesic arc.
    """
    dot = float(np.dot(p0, p1))
    dot = max(-1.0, min(1.0, dot))

    # If the vectors are nearly collinear, linear interpolation is numerically stable
    if abs(dot) > 0.99995:
        res = (1.0 - t) * p0 + t * p1
        norm = np.linalg.norm(res)
        return res / max(1e-12, norm)

    # Angle between vectors
    theta = math.acos(dot)
    sin_theta = math.sin(theta)

    scale0 = math.sin((1.0 - t) * theta) / sin_theta
    scale1 = math.sin(t * theta) / sin_theta

    res = scale0 * p0 + scale1 * p1
    norm = np.linalg.norm(res)
    return res / max(1e-12, norm)


@dataclass
class GFlowNetTrajectory:
    """A sampled trajectory along the continuous Simplex ETF manifold."""
    trajectory_id: str
    actions: List[str]
    latent_states: List[np.ndarray]
    forward_log_probs: List[float]
    backward_log_probs: List[float]
    reward: float
    terminal_state: Any
    log_pf: float = 0.0
    log_pb: float = 0.0
    tb_loss: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "actions": self.actions,
            "reward": round(self.reward, 6),
            "log_pf": round(self.log_pf, 6),
            "log_pb": round(self.log_pb, 6),
            "tb_loss": round(self.tb_loss, 6),
            "length": len(self.actions),
        }


@dataclass
class GFlowNetSamplingResult:
    """Comprehensive evaluation and trajectory diversity report."""
    best_trajectory: GFlowNetTrajectory
    all_trajectories: List[GFlowNetTrajectory]
    best_action: str
    action_probs: Dict[str, float]
    trajectory_entropy: float
    mode_coverage: float
    modes_discovered: List[str]
    counterfactual_branches: List[GFlowNetTrajectory]
    avg_step_latency_ms: float
    total_flow_z: float
    permutation_flip_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_action": self.best_action,
            "action_probs": {k: round(v, 4) for k, v in self.action_probs.items()},
            "trajectory_entropy": round(self.trajectory_entropy, 4),
            "mode_coverage": round(self.mode_coverage, 4),
            "modes_discovered": self.modes_discovered,
            "counterfactual_count": len(self.counterfactual_branches),
            "avg_step_latency_ms": round(self.avg_step_latency_ms, 4),
            "total_flow_z": round(self.total_flow_z, 4),
            "permutation_flip_rate": round(self.permutation_flip_rate, 4),
            "trajectories_count": len(self.all_trajectories),
        }


class ContinuousManifoldGFlowNetSampler:
    """Continuous Manifold GFlowNet Sampler powered by Simplex ETF Geometry."""

    def __init__(
        self,
        dim: int = 128,
        temperature: float = 1.0,
        slerp_step_size: float = 0.35,
        exploration_noise: float = 0.05,
        blend_alpha: float = 0.95,
        log_z_init: float = 0.0,
        learning_rate: float = 0.01,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            dim: Latent embedding dimension for S^(d-1).
            temperature: Softmax temperature for action selection logits.
            slerp_step_size: Geodesic interpolation factor per forward step in [0.05, 0.95].
            exploration_noise: Isotropic Gaussian perturbation on unit sphere.
            blend_alpha: Simplex ETF blend factor (1.0 = pure isotropic ETF).
            log_z_init: Initial value for global partition function log Z_theta.
            learning_rate: Online gradient descent step size for Trajectory Balance.
            seed: Optional random seed for reproducible exploration.
        """
        self.dim = dim
        self.temperature = max(1e-4, float(temperature))
        self.slerp_step_size = max(0.01, min(0.99, float(slerp_step_size)))
        self.exploration_noise = max(0.0, float(exploration_noise))
        self.blend_alpha = max(0.0, min(1.0, float(blend_alpha)))
        self.log_z = float(log_z_init)
        self.learning_rate = float(learning_rate)

        self.rng = np.random.RandomState(seed)
        self.etf_engine = ActionSpaceETFEmbedding(
            dim=self.dim,
            blend_alpha=self.blend_alpha,
            temperature=self.temperature,
        )

    def initialize_latent_state(self, state_hint: Optional[str] = None) -> np.ndarray:
        """Generates an initial unit latent state z_0 on S^(d-1)."""
        if state_hint:
            vec = self.etf_engine.derive_semantic_vector(state_hint)
        else:
            vec = self.rng.randn(self.dim).astype(np.float64)
        norm = np.linalg.norm(vec)
        return vec / max(1e-12, norm)

    def compute_forward_action_probs(
        self,
        latent_state: np.ndarray,
        candidate_actions: Sequence[str],
        etf_matrix: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Computes isotropic forward transition probabilities P_F(a | z) via Simplex ETF.

        Args:
            latent_state: Current unit latent vector z in S^(d-1).
            candidate_actions: List of K candidate actions.
            etf_matrix: Optional pre-computed [K, dim] ETF vectors.

        Returns:
            Tuple of (raw_probs_array, action_to_prob_dict).
        """
        k = len(candidate_actions)
        if k == 0:
            return np.zeros(0, dtype=np.float64), {}

        if etf_matrix is None:
            etf_matrix = self.etf_engine.embed_actions(candidate_actions)

        # Inner product on unit sphere: <z, v_i>
        inner_prods = np.dot(etf_matrix, latent_state)

        # Scale by temperature and apply numerically stable softmax
        logits = inner_prods / self.temperature
        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)
        sum_exp = np.sum(exp_logits)
        probs = exp_logits / max(1e-12, sum_exp)

        prob_dict = {act: float(probs[i]) for i, act in enumerate(candidate_actions)}
        return probs, prob_dict

    def step_geodesic(
        self,
        current_z: np.ndarray,
        action_vector: np.ndarray,
        step_size: Optional[float] = None,
        noise_scale: Optional[float] = None,
    ) -> np.ndarray:
        """Steps along the minimal Riemannian geodesic towards the action ETF vertex.

        Followed by optional isotropic tangent perturbation and spherical projection.
        """
        alpha = step_size if step_size is not None else self.slerp_step_size
        noise = noise_scale if noise_scale is not None else self.exploration_noise

        # 1. Geodesic SLERP on hypersphere
        next_z = slerp(current_z, action_vector, alpha)

        # 2. Add isotropic exploration noise if configured
        if noise > 1e-7:
            perturbation = self.rng.randn(self.dim).astype(np.float64) * noise
            next_z = next_z + perturbation
            norm = np.linalg.norm(next_z)
            next_z = next_z / max(1e-12, norm)

        return next_z

    def compute_tb_loss(
        self,
        trajectory: GFlowNetTrajectory,
        log_z: Optional[float] = None,
    ) -> float:
        """Computes Trajectory Balance (TB) loss for a completed trajectory:

        L_TB = ( log Z_theta + log P_F(tau) - log R(s_T) - log P_B(tau) )^2
        """
        lz = log_z if log_z is not None else self.log_z
        log_r = math.log(max(1e-9, trajectory.reward))
        diff = lz + trajectory.log_pf - log_r - trajectory.log_pb
        return float(diff * diff)

    def update_log_z_tb(self, trajectories: Sequence[GFlowNetTrajectory]) -> float:
        """Performs a gradient descent update on global flow log Z using Trajectory Balance.

        d(L_TB) / d(log Z) = 2 * ( log Z + log P_F - log R - log P_B )
        """
        if not trajectories:
            return 0.0

        grad_z = 0.0
        total_loss = 0.0
        for tau in trajectories:
            log_r = math.log(max(1e-9, tau.reward))
            diff = self.log_z + tau.log_pf - log_r - tau.log_pb
            grad_z += 2.0 * diff
            total_loss += diff * diff

        avg_grad = grad_z / len(trajectories)
        self.log_z -= self.learning_rate * avg_grad
        return float(total_loss / len(trajectories))

    def sample_trajectory(
        self,
        initial_state: Any,
        candidate_actions: Sequence[str],
        transition_fn: Callable[[Any, str], Tuple[Any, float, bool]],
        max_horizon: int = 4,
        initial_latent: Optional[np.ndarray] = None,
        trajectory_id: Optional[str] = None,
    ) -> GFlowNetTrajectory:
        """Samples a single trajectory along the continuous manifold according to P_F."""
        if not candidate_actions:
            z0 = self.initialize_latent_state()
            return GFlowNetTrajectory(
                trajectory_id=trajectory_id or "empty",
                actions=[],
                latent_states=[z0],
                forward_log_probs=[],
                backward_log_probs=[],
                reward=1e-6,
                terminal_state=initial_state,
            )

        tid = trajectory_id or f"tau_{int(time.time()*1e6)%1000000}"
        current_state = initial_state
        z = initial_latent if initial_latent is not None else self.initialize_latent_state(str(initial_state))

        etf_matrix = self.etf_engine.embed_actions(candidate_actions)
        act_to_vec = {act: etf_matrix[i] for i, act in enumerate(candidate_actions)}

        actions_taken = []
        latent_history = [z]
        forward_log_probs = []
        backward_log_probs = []
        cumulative_reward = 0.0

        for step in range(max_horizon):
            probs, _ = self.compute_forward_action_probs(z, candidate_actions, etf_matrix)

            # Sample action proportionally to P_F
            r_val = self.rng.random()
            cum = 0.0
            chosen_idx = len(candidate_actions) - 1
            for idx, p in enumerate(probs):
                cum += p
                if r_val <= cum:
                    chosen_idx = idx
                    break

            chosen_act = candidate_actions[chosen_idx]
            chosen_p = max(1e-12, float(probs[chosen_idx]))
            forward_log_probs.append(math.log(chosen_p))

            # Backward flow P_B is isotropic uniform over candidate dimension
            p_b = 1.0 / max(1, len(candidate_actions))
            backward_log_probs.append(math.log(p_b))

            # Execute transition in environment / sandbox
            next_state, step_reward, done = transition_fn(current_state, chosen_act)
            actions_taken.append(chosen_act)
            cumulative_reward += step_reward

            # Geodesic update on S^(d-1)
            z = self.step_geodesic(z, act_to_vec[chosen_act])
            latent_history.append(z)
            current_state = next_state

            if done:
                break

        # Positivity requirement for GFlowNet reward
        terminal_reward = max(1e-4, cumulative_reward)
        log_pf = sum(forward_log_probs)
        log_pb = sum(backward_log_probs)

        traj = GFlowNetTrajectory(
            trajectory_id=tid,
            actions=actions_taken,
            latent_states=latent_history,
            forward_log_probs=forward_log_probs,
            backward_log_probs=backward_log_probs,
            reward=terminal_reward,
            terminal_state=current_state,
            log_pf=log_pf,
            log_pb=log_pb,
        )
        traj.tb_loss = self.compute_tb_loss(traj)
        return traj

    def sample_diverse_trajectories(
        self,
        initial_state: Any,
        candidate_actions: Sequence[str],
        transition_fn: Callable[[Any, str], Tuple[Any, float, bool]],
        sample_count: int = 16,
        max_horizon: int = 3,
        identify_modes: bool = True,
    ) -> GFlowNetSamplingResult:
        """Generates a diverse ensemble of counterfactual trajectories on the ETF manifold.

        Args:
            initial_state: Environment root state.
            candidate_actions: Candidate actions at root.
            transition_fn: Environment simulator (s, a) -> (next_s, reward, done).
            sample_count: Number of independent trajectories to sample.
            max_horizon: Maximum steps per trajectory.
            identify_modes: Whether to detect and cluster distinct solution modes.

        Returns:
            GFlowNetSamplingResult containing entropy, mode coverage, and counterfactuals.
        """
        t0 = time.perf_counter()
        trajectories: List[GFlowNetTrajectory] = []

        total_steps = 0
        for i in range(sample_count):
            traj = self.sample_trajectory(
                initial_state=initial_state,
                candidate_actions=candidate_actions,
                transition_fn=transition_fn,
                max_horizon=max_horizon,
                trajectory_id=f"gfn_{i}",
            )
            trajectories.append(traj)
            total_steps += len(traj.actions)

        t1 = time.perf_counter()
        total_time_ms = (t1 - t0) * 1000.0
        avg_step_latency = total_time_ms / max(1, total_steps)

        # 1. Action probability distribution at root
        first_action_counts: Dict[str, float] = {act: 0.0 for act in candidate_actions}
        for tau in trajectories:
            if tau.actions:
                first_action_counts[tau.actions[0]] += tau.reward

        total_weight = sum(first_action_counts.values())
        if total_weight > 1e-9:
            action_probs = {act: w / total_weight for act, w in first_action_counts.items()}
        else:
            action_probs = {act: 1.0 / len(candidate_actions) for act in candidate_actions}

        # 2. Trajectory Shannon Entropy: H = - sum p * log p
        # Form empirical trajectory distribution based on distinct action paths
        path_rewards: Dict[Tuple[str, ...], float] = {}
        for tau in trajectories:
            path = tuple(tau.actions)
            path_rewards[path] = path_rewards.get(path, 0.0) + tau.reward

        path_sum = sum(path_rewards.values())
        if path_sum > 1e-9:
            norm_probs = [r / path_sum for r in path_rewards.values()]
            entropy = -sum(p * math.log(p + 1e-12) for p in norm_probs)
        else:
            entropy = 0.0

        # 3. Detect solution modes (peaks with reward >= 70% of max reward)
        max_r = max((tau.reward for tau in trajectories), default=1e-6)
        mode_threshold = max_r * 0.70
        distinct_modes = set()
        counterfactuals = []

        for tau in trajectories:
            if tau.reward >= mode_threshold:
                mode_sig = "->".join(tau.actions)
                distinct_modes.add(mode_sig)
            else:
                counterfactuals.append(tau)

        modes_list = sorted(list(distinct_modes))
        # Best overall trajectory by reward (or tie-break by minimal TB loss)
        sorted_trajs = sorted(trajectories, key=lambda t: (t.reward, -t.tb_loss), reverse=True)
        best_traj = sorted_trajs[0]
        best_action = best_traj.actions[0] if best_traj.actions else candidate_actions[0]

        # Mode coverage (empirical, first-action level): the fraction of candidate first
        # actions that start at least one sampled trajectory with reward >= 70% of the
        # sampled max. Numerator and denominator are both first actions, so the value lies
        # in [0, 1] without clamping. It measures spread over the samples we drew; it is
        # not coverage against a ground-truth mode set, which is unknown here.
        high_value_first_actions = {
            tau.actions[0] for tau in trajectories
            if tau.reward >= mode_threshold and tau.actions
        }
        mode_coverage = len(high_value_first_actions & set(candidate_actions)) / max(1, len(candidate_actions))

        # Measure permutation flip rate to confirm 0.00%
        flip_rate = self._compute_permutation_flip_rate(
            initial_state, candidate_actions, transition_fn
        )

        return GFlowNetSamplingResult(
            best_trajectory=best_traj,
            all_trajectories=trajectories,
            best_action=best_action,
            action_probs=action_probs,
            trajectory_entropy=float(entropy),
            mode_coverage=float(mode_coverage),
            modes_discovered=modes_list,
            counterfactual_branches=counterfactuals[:4],
            avg_step_latency_ms=avg_step_latency,
            total_flow_z=math.exp(self.log_z),
            permutation_flip_rate=flip_rate,
        )

    def _compute_permutation_flip_rate(
        self,
        state: Any,
        candidate_actions: Sequence[str],
        transition_fn: Callable[[Any, str], Tuple[Any, float, bool]],
        trials: int = 5,
    ) -> float:
        """Validates that candidate action permutation preserves exact argmax decision."""
        if len(candidate_actions) <= 1:
            return 0.0

        z0 = self.initialize_latent_state(str(state))
        _, base_probs = self.compute_forward_action_probs(z0, candidate_actions)
        base_top1 = max(base_probs.items(), key=lambda kv: kv[1])[0]

        flips = 0
        actions_list = list(candidate_actions)
        for _ in range(trials):
            permuted = list(self.rng.permutation(actions_list))
            _, perm_probs = self.compute_forward_action_probs(z0, permuted)
            perm_top1 = max(perm_probs.items(), key=lambda kv: kv[1])[0]
            if perm_top1 != base_top1:
                flips += 1

        return float(flips / max(1, trials))


class ContinuousGFlowNetAdapter:
    """Drop-in adapter compatible with Gen-Zero client planner interface."""

    def __init__(self, temperature: float = 1.0, dim: int = 128) -> None:
        self.sampler = ContinuousManifoldGFlowNetSampler(
            dim=dim,
            temperature=temperature,
            slerp_step_size=0.35,
            exploration_noise=0.03,
            blend_alpha=0.95,
        )

    def sample_trajectory(
        self,
        initial_state: Any,
        candidate_actions: List[str],
        transition_fn: Callable[[Any, str], Tuple[Any, float, bool]],
        temperature: Optional[float] = None,
        sample_count: int = 12,
        max_horizon: int = 3,
    ) -> Dict[str, Any]:
        """Provides backward-compatible interface for client.py."""
        if temperature is not None:
            self.sampler.temperature = float(temperature)

        result = self.sampler.sample_diverse_trajectories(
            initial_state=initial_state,
            candidate_actions=candidate_actions,
            transition_fn=transition_fn,
            sample_count=sample_count,
            max_horizon=max_horizon,
        )

        return {
            "best_action": result.best_action,
            "action_probs": result.action_probs,
            "total_flow": round(result.total_flow_z, 4),
            "diversity_entropy": round(result.trajectory_entropy, 4),
            "mode_coverage": round(result.mode_coverage, 4),
            "modes_discovered": result.modes_discovered,
            "counterfactual_count": len(result.counterfactual_branches),
            "avg_step_latency_ms": round(result.avg_step_latency_ms, 4),
            "permutation_flip_rate": round(result.permutation_flip_rate, 4),
            "adaptive_temperature": self.sampler.temperature,
            "mode": "continuous_manifold_gflownet",
        }
