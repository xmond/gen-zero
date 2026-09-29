"""Gen-Zero Global Configuration."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


def _checkpoint_from_env(name: str) -> Optional[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path: {raw!r}")
    return str(path.resolve(strict=False))


@dataclass
class GenZeroConfig:
    # Model architecture
    hidden_dim: int = 4096
    embed_dim: int = 128
    num_attention_layers: int = 2
    num_attention_heads: int = 4
    use_value_head: bool = True
    enable_abstain: bool = True
    backbone_name: str = "Qwen/Qwen3.5-9B"
    # Trained dual-head state_dict. None => model stays random-init and the HTTP service
    # fails closed (503) on /v1/decisions, /v1/decide_step and /v1/score.
    dual_head_checkpoint: Optional[str] = field(
        default_factory=lambda: _checkpoint_from_env("GENZERO_DUAL_HEAD_CHECKPOINT")
    )

    # Planner settings
    astar_lambda: float = 1.0
    mcts_simulations: int = 64
    mcts_depth: int = 6
    mcts_cpuct: float = 1.4
    use_hamiltonian_dynamics: bool = False
    # Explicit checkpoint only; absence means no trained dynamics model is mounted.
    neural_dynamics_checkpoint: Optional[str] = field(
        default_factory=lambda: _checkpoint_from_env("GENZERO_NEURAL_DYNAMICS_CHECKPOINT")
    )
    
    # Online Exploration & Hard Mining
    exploration_entropy_threshold: float = 1.2
    td_error_threshold: float = 0.8
    hard_sample_history_steps: int = 5  # Capture 5 steps prior to failure

    # Replay Buffer & Training
    enable_causal_replay_buffer: bool = True
    causal_replay_capacity: int = 10000
    hard_to_gold_ratio: float = 0.25    # 1:3 ratio = 1 hard per 3 gold (25% hard)
    learning_rate: float = 2e-5
    value_loss_weight: float = 0.5
    abstain_loss_weight: float = 0.2
    max_train_steps: int = 1200
    batch_size: int = 16

    # Automated Safety Gate & RSI (I-24 Meta^n)
    gate_min_accuracy_retention: float = 0.995  # Must retain >= 99.5% of base accuracy
    gate_min_score_gain: float = 0.05           # Positive score gain on long-horizon games
    metan_convergence_delta: float = 0.005      # Stop if delta < 0.5% for 2 consecutive rounds

    # Cloud-Edge GPU Arbiter Fallback
    enable_gpu_arbiter_fallback: bool = True
    arbiter_confidence_threshold: float = 0.40  # Trigger fallback when confidence < 0.40
    arbiter_entropy_threshold: float = 0.75     # Trigger fallback when entropy > 0.75
    arbiter_endpoint: str = "http://localhost:8090/arbitrate"
    arbiter_timeout_s: float = 2.0

    # Instance-Adaptive Gating & Log-linear Opinion Pool (multi-expert consensus fusion)
    enable_adaptive_gating: bool = False
    adaptive_gating_pool_type: str = "log_linear"  # "log_linear" or "arithmetic"
    # Path to an InstanceAdaptiveRouter.save() artifact (.npz). Required when
    # enable_adaptive_gating is True; construction fails closed without it.
    adaptive_gating_artifact: Optional[str] = field(
        default_factory=lambda: _checkpoint_from_env("GENZERO_ADAPTIVE_GATING_ARTIFACT")
    )

    # Storage paths
    base_dir: str = field(default_factory=lambda: os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    checkpoint_dir: str = "runs/gen_zero_ckpt"
    results_dir: str = "results/gen_zero"

    def __post_init__(self) -> None:
        for key in ("dual_head_checkpoint", "neural_dynamics_checkpoint", "adaptive_gating_artifact"):
            value = getattr(self, key)
            if value is not None:
                path = Path(value).expanduser()
                if not path.is_absolute():
                    raise ValueError(f"{key} must be an absolute path: {value!r}")
                setattr(self, key, str(path.resolve(strict=False)))
        valid_pools = {"log_linear", "arithmetic"}
        if self.adaptive_gating_pool_type not in valid_pools:
            raise ValueError(
                f"invalid adaptive_gating_pool_type: {self.adaptive_gating_pool_type!r}; "
                f"must be one of {sorted(valid_pools)}"
            )
