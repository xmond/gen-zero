r"""Gen-Zero Latent Dynamics & Pearl Structural Causal Transition World Model.

Implements non-autoregressive latent state transition:
  z_{t+1} = z_t + \Delta_\theta(z_t, do(a_t)) + u_t
Predicts:
1. Next latent state embedding z_{t+1} in high-dimensional feature space (1024-dim).
2. Expected immediate transition reward / outcome delta r_t in [-1, 1].
3. Epistemic uncertainty / variance sigma^2(z, a) for surprise detection.
4. Exogenous causal shock attribution u_t = z_{t+1} - \hat{z}_{t+1}.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, Any
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = object
    F = None
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


def resolve_torch_device(device: Any) -> str:
    """Preserve indexed CUDA and MPS requests; retain CUDA's CPU fallback."""
    if not HAS_TORCH:
        return "cpu"
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return str(requested)


if HAS_TORCH:
    class PyTorchLatentDynamicsNet(nn.Module):
        """Pure PyTorch feedforward latent transition dynamics network."""

        def __init__(
            self,
            latent_dim: int = 1024,
            action_dim: int = 32,
            hidden_dim: int = 128
        ):
            super().__init__()
            self.latent_dim = latent_dim
            self.action_dim = action_dim

            # Action embedder
            self.action_embed = nn.Sequential(
                nn.Linear(action_dim, 32, bias=False),
                nn.GELU()
            )

            # Bottleneck Residual Dynamics: [z_t, a_emb] (1024 + 32 = 1056) -> 128 -> 1024
            self.down_proj = nn.Linear(latent_dim + 32, hidden_dim, bias=False)
            self.act = nn.GELU()
            self.up_proj = nn.Linear(hidden_dim, latent_dim, bias=False)

            # Reward & Uncertainty head from bottleneck features (128-dim)
            self.reward_head = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Tanh()
            )
            self.uncertainty_head = nn.Sequential(
                nn.Linear(hidden_dim, 1)
            )

            # Small init for stable residual transitions
            nn.init.normal_(self.up_proj.weight, std=0.01)

        def forward(
            self,
            z: torch.Tensor,
            a: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Forward step.
            
            Args:
                z: [Batch, latent_dim]
                a: [Batch, action_dim]
            Returns:
                next_z: [Batch, latent_dim] (residual: z + Delta)
                reward: [Batch, 1]
                variance: [Batch, 1]
            """
            a_emb = self.action_embed(a)
            concat = torch.cat([z, a_emb], dim=-1)

            feat = self.act(self.down_proj(concat))
            delta_z = self.up_proj(feat)
            next_z = z + delta_z  # Residual jump connection for smooth dynamics

            reward = self.reward_head(feat)
            variance = F.softplus(self.uncertainty_head(feat)) + 1e-4

            return next_z, reward, variance


class LatentTransitionModel:
    """High-level coordinator for latent state transition and causal rollout."""

    def __init__(
        self,
        latent_dim: int = 1024,
        action_dim: int = 32,
        hidden_dim: int = 128,
        device: Any = "cpu"
    ):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = resolve_torch_device(device)

        if HAS_TORCH:
            # Optimize CPU intra-op threads for small MLP latency
            if self.device == "cpu" and hasattr(torch, "get_num_threads") and torch.get_num_threads() > 4:
                try:
                    torch.set_num_threads(4)
                except Exception:
                    pass
            self.net = PyTorchLatentDynamicsNet(
                latent_dim=latent_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim
            ).to(self.device)
            self.net.eval()
        else:
            self.net = None

    def step(
        self,
        latent_state: Any,
        action: Union[int, List[float], Any],
        preserve_entropy: bool = False
    ) -> Tuple[Any, float, float]:
        """Performs a 1-step latent transition.
        
        Args:
            latent_state: Input latent representation.
            action: Selected action or action vector.
            preserve_entropy: If True, enforces so(D) norm-preserving geodesic dynamics
                to prevent representation collapse over long lookahead horizons (RFC-069).
                
        Returns:
            (next_latent_state, predicted_reward, predicted_variance)
        """
        if HAS_TORCH and self.net is not None:
            return self._step_torch(latent_state, action, preserve_entropy=preserve_entropy)
        return self._step_numpy(latent_state, action, preserve_entropy=preserve_entropy)

    def rollout(
        self,
        initial_latent: Any,
        action_sequence: List[Union[int, List[float], Any]],
        preserve_entropy: bool = False
    ) -> Dict[str, Any]:
        """Rolls out a sequence of actions entirely inside the latent space.
        
        Args:
            initial_latent: Starting latent state.
            action_sequence: List of sequential candidate actions.
            preserve_entropy: If True, uses geodesic dynamics maintaining constant norm.
        """
        states = [initial_latent]
        rewards = []
        variances = []

        curr_z = initial_latent
        for act in action_sequence:
            next_z, r, var = self.step(curr_z, act, preserve_entropy=preserve_entropy)
            states.append(next_z)
            rewards.append(r)
            variances.append(var)
            curr_z = next_z

        cumulative_reward = sum(rewards)
        mean_uncertainty = sum(variances) / max(1, len(variances))

        # Calculate norm statistics along trajectory
        norms = []
        for s in states:
            if HAS_TORCH and isinstance(s, torch.Tensor):
                norms.append(float(torch.norm(s.detach().cpu().flatten(), p=2).item()))
            elif HAS_NUMPY and isinstance(s, np.ndarray):
                norms.append(float(np.linalg.norm(s.flatten())))
            else:
                try:
                    norms.append(float(sum(x**2 for x in s)**0.5))
                except Exception:
                    norms.append(1.0)
        norm_var = float(np.var(norms)) if (HAS_NUMPY and len(norms) > 1) else 0.0

        return {
            "latent_states": states,
            "latent_trajectory": states,
            "rewards": rewards,
            "variances": variances,
            "cumulative_reward": cumulative_reward,
            "cumulative_return": cumulative_reward,
            "total_return": cumulative_reward,
            "mean_uncertainty": mean_uncertainty,
            "horizon": len(action_sequence),
            "trajectory_norms": norms,
            "norm_variance": norm_var,
        }

    def step_batch(
        self,
        latent_state: Any,
        actions: List[Any]
    ) -> List[Tuple[Any, float, float]]:
        """Batched 1-step latent transition for all candidate actions in a single forward pass."""
        if not actions:
            return []
        if HAS_TORCH and self.net is not None:
            return self._step_batch_torch(latent_state, actions)
        return [self._step_numpy(latent_state, a) for a in actions]

    def compute_causal_shock(
        self,
        prior_latent: Any,
        action: Any,
        real_next_latent: Any
    ) -> Tuple[float, Any]:
        r"""Calculates exogenous causal shock ||u_t|| = ||z_{t+1} - \hat{z}_{t+1}||."""
        pred_next_z, _, _ = self.step(prior_latent, action)

        if HAS_TORCH and isinstance(real_next_latent, torch.Tensor):
            parameter = next(self.net.parameters())
            if isinstance(pred_next_z, torch.Tensor):
                pred_t = pred_next_z.to(device=parameter.device, dtype=parameter.dtype)
            else:
                pred_t = torch.tensor(pred_next_z, device=parameter.device, dtype=parameter.dtype)
            shock_vec = real_next_latent.to(device=parameter.device, dtype=parameter.dtype) - pred_t
            if not bool(torch.isfinite(shock_vec).all()):
                raise ValueError("causal shock contains non-finite values")
            shock_norm = float(torch.norm(shock_vec, p=2).item())
            return shock_norm, shock_vec
        elif HAS_NUMPY and isinstance(real_next_latent, np.ndarray):
            if HAS_TORCH and isinstance(pred_next_z, torch.Tensor):
                pred_arr = pred_next_z.detach().cpu().numpy()
            else:
                pred_arr = np.asarray(pred_next_z, dtype=np.float32)
            real_arr = np.asarray(real_next_latent, dtype=np.float32)
            shock_vec = real_arr - pred_arr
            shock_norm = float(np.linalg.norm(shock_vec))
            return shock_norm, shock_vec

        return 0.0, None

    # ---- Internal Engine Handlers ----
    def _step_torch(
        self,
        z_input: Any,
        a_input: Any,
        preserve_entropy: bool = False
    ) -> Tuple[torch.Tensor, float, float]:
        with torch.no_grad():
            parameter = next(self.net.parameters())
            model_device, model_dtype = parameter.device, parameter.dtype
            # Format Z
            if not isinstance(z_input, torch.Tensor):
                z_t = torch.tensor(z_input, dtype=model_dtype, device=model_device)
            else:
                z_t = z_input.to(device=model_device, dtype=model_dtype)
            if z_t.ndim == 1:
                z_t = z_t.unsqueeze(0)
            if z_t.ndim != 2 or z_t.shape[-1] != self.latent_dim:
                raise ValueError(f"latent state must have final dimension {self.latent_dim}, got {tuple(z_t.shape)}")
            if z_t.shape[0] != 1:
                raise ValueError("step expects one latent state; use step_batch for batched transitions")
            if not bool(torch.isfinite(z_t).all()):
                raise ValueError("latent state contains non-finite values")

            # Format A
            a_t = self._encode_action_torch(a_input, dtype=model_dtype)
            if a_t.shape[0] != 1:
                raise ValueError("step expects one action; use step_batch for batched transitions")
            if not bool(torch.isfinite(a_t).all()):
                raise ValueError("action contains non-finite values")

            next_z, reward, var = self.net(z_t, a_t)
            if preserve_entropy:
                orig_norm = torch.norm(z_t, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                next_norm = torch.norm(next_z, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                next_z = next_z * (orig_norm / next_norm)

            if not (bool(torch.isfinite(next_z).all()) and bool(torch.isfinite(reward).all())
                    and bool(torch.isfinite(var).all())):
                raise FloatingPointError("latent transition produced non-finite values")

            return (
                next_z.squeeze(0),
                float(reward.item()),
                float(var.item())
            )

    def _step_batch_torch(
        self,
        z_input: Any,
        actions: List[Any],
        preserve_entropy: bool = False
    ) -> List[Tuple[torch.Tensor, float, float]]:
        with torch.no_grad():
            parameter = next(self.net.parameters())
            model_device, model_dtype = parameter.device, parameter.dtype
            if not isinstance(z_input, torch.Tensor):
                z_t = torch.tensor(z_input, dtype=model_dtype, device=model_device)
            else:
                z_t = z_input.to(device=model_device, dtype=model_dtype)
            if z_t.ndim == 1:
                z_t = z_t.unsqueeze(0)
            if z_t.ndim != 2 or z_t.shape[-1] != self.latent_dim:
                raise ValueError(f"latent state must have final dimension {self.latent_dim}, got {tuple(z_t.shape)}")
            if not bool(torch.isfinite(z_t).all()):
                raise ValueError("latent state contains non-finite values")

            b = len(actions)
            if z_t.shape[0] not in (1, b):
                raise ValueError(f"latent batch size {z_t.shape[0]} must be 1 or match action count {b}")
            z_batch = z_t.expand(b, -1) if z_t.shape[0] == 1 else z_t
            a_list = [self._encode_action_torch(a, dtype=model_dtype) for a in actions]
            if any(a.shape[0] != 1 for a in a_list):
                raise ValueError("Each candidate action must contain exactly one vector")
            a_batch = torch.cat(a_list, dim=0)
            if not bool(torch.isfinite(a_batch).all()):
                raise ValueError("action contains non-finite values")

            next_z, reward, var = self.net(z_batch, a_batch)
            if preserve_entropy:
                orig_norm = torch.norm(z_batch, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                next_norm = torch.norm(next_z, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                next_z = next_z * (orig_norm / next_norm)

            if not (bool(torch.isfinite(next_z).all()) and bool(torch.isfinite(reward).all())
                    and bool(torch.isfinite(var).all())):
                raise FloatingPointError("latent transition produced non-finite values")

            results = []
            for i in range(b):
                results.append((
                    next_z[i],
                    float(reward[i].item()),
                    float(var[i].item())
                ))
            return results

    def _step_numpy(
        self,
        z_input: Any,
        a_input: Any,
        preserve_entropy: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        # Lightweight CPU fallback
        z_arr = np.array(z_input, dtype=np.float32).flatten()
        if len(z_arr) < self.latent_dim:
            padded = np.zeros(self.latent_dim, dtype=np.float32)
            padded[:len(z_arr)] = z_arr
            z_arr = padded
        elif len(z_arr) > self.latent_dim:
            z_arr = z_arr[:self.latent_dim]

        # Deterministic pseudo transition
        act_hash = (hash(str(a_input)) % 1000) / 1000.0
        delta = 0.01 * np.sin(z_arr * (1.0 + act_hash))
        next_z = z_arr + delta
        if preserve_entropy:
            orig_norm = float(np.linalg.norm(z_arr))
            if orig_norm > 1e-8:
                next_norm = float(np.linalg.norm(next_z))
                if next_norm > 1e-8:
                    next_z = next_z * (orig_norm / next_norm)

        reward = float(np.clip(np.mean(next_z) * 2.0, -1.0, 1.0))
        var = float(0.01 + 0.05 * act_hash)
        return next_z, reward, var

    def _encode_action_torch(self, a_input: Any, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        parameter = next(self.net.parameters())
        model_device = parameter.device
        model_dtype = parameter.dtype if dtype is None else dtype
        if isinstance(a_input, int):
            # One-hot or sinusoidal encoding
            one_hot = torch.zeros((1, self.action_dim), device=model_device, dtype=model_dtype)
            idx = a_input % self.action_dim
            one_hot[0, idx] = 1.0
            return one_hot
        elif isinstance(a_input, (list, tuple)):
            arr = torch.tensor(a_input, device=model_device, dtype=model_dtype)
            if arr.ndim == 1:
                arr = arr.unsqueeze(0)
            if arr.ndim != 2:
                raise ValueError(f"action must be 1-D or 2-D, got {tuple(arr.shape)}")
            if arr.shape[-1] < self.action_dim:
                pad = torch.zeros((arr.shape[0], self.action_dim - arr.shape[-1]), device=model_device, dtype=model_dtype)
                arr = torch.cat([arr, pad], dim=-1)
            return arr[:, :self.action_dim]
        elif isinstance(a_input, torch.Tensor):
            arr = a_input.to(device=model_device, dtype=model_dtype)
            if arr.ndim == 1:
                arr = arr.unsqueeze(0)
            if arr.ndim != 2:
                raise ValueError(f"action must be 1-D or 2-D, got {tuple(arr.shape)}")
            if arr.shape[-1] < self.action_dim:
                pad = torch.zeros((arr.shape[0], self.action_dim - arr.shape[-1]), device=model_device, dtype=model_dtype)
                arr = torch.cat([arr, pad], dim=-1)
            return arr[:, :self.action_dim]
        else:
            # String or object hash
            h = (hash(str(a_input)) % self.action_dim)
            one_hot = torch.zeros((1, self.action_dim), device=model_device, dtype=model_dtype)
            one_hot[0, h] = 1.0
            return one_hot
