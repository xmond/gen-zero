"""Neural dynamics world model: learned residual transition + safety/reward head.

Given a latent state z in R^{D_s} and an action a in R^{D_a}:

    delta_z = transition_net([z; a])        # residual MLP, LayerNorm + GELU
    z_next  = z + delta_z
    r_hat   = sigmoid(reward_net([z; a]))   # outcome / safety probability in [0, 1]

Training minimises  L = ||z_next - z'||^2 + lambda * BCE(r_hat, r)
(see ``joint_loss`` and ``scripts/train_world_model_dynamics.py``).

Fail-closed contract: a freshly constructed model holds random weights and is
NOT usable for inference. ``step`` raises until ``load_checkpoint`` has
succeeded. Training a model in memory does not unlock it either; the only
path to inference is a checkpoint that passed format and shape validation.

``done`` has no label in (s, a, s', r) data. It is defined as a rule, not a
learned output: ``done = r_hat < done_threshold`` (default 0.5, stored in the
checkpoint). A predicted unsafe/failed transition terminates the rollout.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .torus_codec import TORUS_ACTION_DIM, TORUS_ACTION_NAMES, encode_torus_action

logger = logging.getLogger("gen_zero.world_model.neural_dynamics")

CHECKPOINT_FORMAT = "gen_zero.neural_dynamics.v1"
FAIL_CLOSED_MESSAGE = "World model weights not loaded - fail closed"

ActionLike = Union[str, np.ndarray, Sequence[float]]


class _ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


class ResidualMLP(nn.Module):
    """Linear -> N pre-norm residual blocks (LayerNorm, GELU) -> LayerNorm -> Linear."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_blocks: int, zero_init_out: bool):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.Sequential(*[_ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        if zero_init_out:
            # Start as the identity transition (delta = 0) so early training is stable.
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.norm(self.blocks(self.inp(x))))


class NeuralDynamicsWorldModel(nn.Module):
    """Residual latent transition model with a sigmoid outcome/safety head."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        action_vocab: Optional[Sequence[str]] = None,
        done_threshold: float = 0.5,
    ):
        super().__init__()
        if state_dim <= 0 or action_dim <= 0 or hidden_dim <= 0 or num_blocks < 0:
            raise ValueError("state_dim, action_dim, hidden_dim must be positive and num_blocks >= 0")
        if action_vocab is not None and len(action_vocab) != action_dim:
            raise ValueError(f"action_vocab has {len(action_vocab)} names but action_dim={action_dim}")
        if not 0.0 <= done_threshold <= 1.0:
            raise ValueError("done_threshold must lie in [0, 1]")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        self.action_vocab: Optional[List[str]] = list(action_vocab) if action_vocab is not None else None
        self.done_threshold = float(done_threshold)
        in_dim = self.state_dim + self.action_dim
        self.transition_net = ResidualMLP(in_dim, self.hidden_dim, self.state_dim, self.num_blocks, zero_init_out=True)
        self.reward_net = ResidualMLP(in_dim, self.hidden_dim, 1, self.num_blocks, zero_init_out=False)
        self._weights_loaded = False
        self.checkpoint_path: Optional[Path] = None

    def _parameter_device_dtype(self) -> Tuple[torch.device, torch.dtype]:
        """Return the device and floating dtype used by this model's parameters."""
        parameter = next(self.parameters())
        return parameter.device, parameter.dtype

    def _as_model_tensor(self, value: torch.Tensor) -> torch.Tensor:
        """Convert an input tensor to the model parameter device and dtype."""
        device, dtype = self._parameter_device_dtype()
        return value.to(device=device, dtype=dtype)

    @property
    def weights_loaded(self) -> bool:
        return self._weights_loaded

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (z_next, reward_prob, reward_logit). Batched: z (B, D_s), a (B, D_a)."""
        z = self._as_model_tensor(z)
        a = self._as_model_tensor(a)
        za = torch.cat([z, a], dim=-1)
        z_next = z + self.transition_net(za)
        logit = self.reward_net(za).squeeze(-1)
        return z_next, torch.sigmoid(logit), logit

    def set_action_embeddings(self, mapping: Dict[str, np.ndarray]) -> None:
        """Register custom continuous/projected vectors for named actions."""
        self._action_embeddings = {k: np.asarray(v, dtype=np.float32).reshape(-1) for k, v in mapping.items()}

    def encode_action(self, action: ActionLike) -> np.ndarray:
        """Named action -> embedding vector or one-hot; vectors pass through after a shape check."""
        if hasattr(self, "_action_embeddings") and isinstance(action, str) and action in self._action_embeddings:
            return self._action_embeddings[action]
        if isinstance(action, str):
            if self.action_dim == TORUS_ACTION_DIM and action in TORUS_ACTION_NAMES:
                return encode_torus_action(action)
            if self.action_vocab is None:
                raise KeyError(f"action {action!r} is a name but this model has no action_vocab")
            try:
                idx = self.action_vocab.index(action)
            except ValueError:
                raise KeyError(f"action {action!r} not in trained action_vocab {self.action_vocab}") from None
            vec = np.zeros(self.action_dim, dtype=np.float32)
            vec[idx] = 1.0
            return vec
        vec = np.asarray(action, dtype=np.float32).reshape(-1)
        if vec.shape != (self.action_dim,):
            raise ValueError(f"action vector has shape {vec.shape}, expected ({self.action_dim},)")
        return vec

    def step(self, state: np.ndarray, action: ActionLike) -> Tuple[np.ndarray, float, bool]:
        """One imagined transition. Returns (z_next, r_hat, done) with done = r_hat < done_threshold."""
        if not self._weights_loaded:
            raise RuntimeError(FAIL_CLOSED_MESSAGE)
        z = np.asarray(state, dtype=np.float32).reshape(-1)
        if z.shape != (self.state_dim,):
            raise ValueError(f"state has shape {z.shape}, expected ({self.state_dim},)")
        a = self.encode_action(action)
        if not (np.all(np.isfinite(z)) and np.all(np.isfinite(a))):
            raise ValueError("state/action contain non-finite values")
        device, dtype = self._parameter_device_dtype()
        self.eval()
        with torch.no_grad():
            z_tensor = torch.as_tensor(z[None], device=device, dtype=dtype)
            a_tensor = torch.as_tensor(a[None], device=device, dtype=dtype)
            z_next, prob, _ = self(z_tensor, a_tensor)
        r_hat = float(prob[0].detach().cpu().item())
        next_state = z_next[0].detach().cpu().numpy().astype(np.float32)
        return next_state, r_hat, bool(r_hat < self.done_threshold)

    def _config(self) -> Dict[str, object]:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "num_blocks": self.num_blocks,
            "action_vocab": self.action_vocab,
            "done_threshold": self.done_threshold,
        }

    def save_checkpoint(self, path: Union[str, Path]) -> Path:
        """Atomic write of weights + architecture config. Does not unlock ``step`` on this instance."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": CHECKPOINT_FORMAT,
            "config": self._config(),
            "state_dict": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
        }
        tmp = path.with_name(path.name + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return path

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load and validate weights. Any mismatch raises and leaves the model locked."""
        path = Path(path)
        self._weights_loaded = False
        if not path.is_file():
            raise FileNotFoundError(f"world model checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
        cfg = payload["config"]
        for key in ("state_dim", "action_dim", "hidden_dim", "num_blocks"):
            if int(cfg[key]) != getattr(self, key):
                raise ValueError(f"checkpoint {key}={cfg[key]} does not match model {key}={getattr(self, key)}")
        self.load_state_dict(payload["state_dict"], strict=True)
        vocab = cfg.get("action_vocab")
        if vocab is not None and len(vocab) != self.action_dim:
            raise ValueError("checkpoint action_vocab length does not match action_dim")
        self.action_vocab = list(vocab) if vocab is not None else None
        self.done_threshold = float(cfg["done_threshold"])
        self.checkpoint_path = path
        self._weights_loaded = True
        logger.info("Loaded neural dynamics world model from %s (%s)", path, cfg)

    @classmethod
    def from_checkpoint(cls, path: Union[str, Path]) -> "NeuralDynamicsWorldModel":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"world model checkpoint not found: {path}")
        cfg = torch.load(path, map_location="cpu", weights_only=True)["config"]
        model = cls(
            state_dim=int(cfg["state_dim"]),
            action_dim=int(cfg["action_dim"]),
            hidden_dim=int(cfg["hidden_dim"]),
            num_blocks=int(cfg["num_blocks"]),
            action_vocab=cfg.get("action_vocab"),
            done_threshold=float(cfg["done_threshold"]),
        )
        model.load_checkpoint(path)
        return model


# ---------------------------------------------------------------------------
# Data + training primitives (shared by the training script and tests)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransitionDataset:
    """(s, a, s', r) arrays. ``rewards`` are binary outcome/safety labels in {0, 1}.

    ``episode_ids`` (N,) integers, optional. Rows of one episode keep their
    original step order through ``subset`` and ``split_indices``.
    """

    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    rewards: np.ndarray
    action_vocab: Optional[List[str]] = None
    episode_ids: Optional[np.ndarray] = None

    def __post_init__(self):
        n = self.states.shape[0]
        if self.states.ndim != 2 or self.next_states.shape != self.states.shape:
            raise ValueError("states and next_states must both be (N, D_s)")
        if self.actions.ndim != 2 or self.actions.shape[0] != n:
            raise ValueError("actions must be (N, D_a)")
        if self.rewards.shape != (n,):
            raise ValueError("rewards must be (N,)")
        if not np.all((self.rewards >= 0.0) & (self.rewards <= 1.0)):
            raise ValueError("rewards must lie in [0, 1] for the BCE head")
        if self.action_vocab is not None and len(self.action_vocab) != self.actions.shape[1]:
            raise ValueError("action_vocab length must equal D_a")
        if self.episode_ids is not None:
            if self.episode_ids.shape != (n,) or not np.issubdtype(self.episode_ids.dtype, np.integer):
                raise ValueError("episode_ids must be one integer per transition, shape (N,)")
        for name in ("states", "actions", "next_states", "rewards"):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f"{name} contains non-finite values")

    def __len__(self) -> int:
        return int(self.states.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    @classmethod
    def from_npz(cls, path: Union[str, Path]) -> "TransitionDataset":
        with np.load(path, allow_pickle=False) as f:
            missing = {"states", "actions", "next_states", "rewards"} - set(f.files)
            if missing:
                raise KeyError(f"{path} lacks arrays {sorted(missing)}")
            vocab = [str(x) for x in f["action_vocab"]] if "action_vocab" in f.files else None
            episode_ids = f["episode_ids"] if "episode_ids" in f.files else None
            return cls(
                states=f["states"].astype(np.float32),
                actions=f["actions"].astype(np.float32),
                next_states=f["next_states"].astype(np.float32),
                rewards=f["rewards"].astype(np.float32).reshape(-1),
                action_vocab=vocab,
                episode_ids=episode_ids,
            )

    def save_npz(self, path: Union[str, Path]) -> None:
        arrays = dict(states=self.states, actions=self.actions, next_states=self.next_states, rewards=self.rewards)
        if self.action_vocab is not None:
            arrays["action_vocab"] = np.asarray(self.action_vocab, dtype=np.str_)
        if self.episode_ids is not None:
            arrays["episode_ids"] = self.episode_ids
        np.savez(path, **arrays)

    def subset(self, idx: np.ndarray) -> "TransitionDataset":
        return TransitionDataset(
            self.states[idx], self.actions[idx], self.next_states[idx], self.rewards[idx], self.action_vocab,
            None if self.episode_ids is None else self.episode_ids[idx],
        )

    def split_indices(
        self, val_fraction: float, seed: int, group_by_episode: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Row indices (train, val).

        Grouped mode permutes the unique episode ids and sends round(E * val_fraction)
        whole episodes to val, so no episode is cut across the two sets. Indices come
        back ascending, which keeps each episode in step order. Grouped mode without
        ``episode_ids`` raises; a row split must be asked for with group_by_episode=False.
        """
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must lie in (0, 1)")
        rng = np.random.default_rng(seed)
        if not group_by_episode:
            perm = rng.permutation(len(self))
            n_val = int(round(len(self) * val_fraction))
            if n_val == 0 or n_val == len(self):
                raise ValueError(f"dataset of {len(self)} rows is too small for a {val_fraction:.0%} split")
            return perm[n_val:], perm[:n_val]
        if self.episode_ids is None:
            raise ValueError("group_by_episode=True needs episode_ids; pass group_by_episode=False for a row split")
        episodes = np.unique(self.episode_ids)
        n_val_eps = int(round(len(episodes) * val_fraction))
        if n_val_eps == 0 or n_val_eps == len(episodes):
            raise ValueError(f"{len(episodes)} episodes are too few for a {val_fraction:.0%} grouped split")
        val_eps = rng.permutation(episodes)[:n_val_eps]
        in_val = np.isin(self.episode_ids, val_eps)
        train_idx, val_idx = np.flatnonzero(~in_val), np.flatnonzero(in_val)
        leaked = np.intersect1d(self.episode_ids[train_idx], self.episode_ids[val_idx])
        if leaked.size or len(train_idx) + len(val_idx) != len(self):
            raise AssertionError(f"grouped split leaked episodes {leaked[:10].tolist()} across train/val")
        return train_idx, val_idx

    def split(
        self, val_fraction: float, seed: int, group_by_episode: bool = True
    ) -> Tuple["TransitionDataset", "TransitionDataset"]:
        train_idx, val_idx = self.split_indices(val_fraction, seed, group_by_episode)
        return self.subset(train_idx), self.subset(val_idx)

    def as_tensors(self) -> Dict[str, torch.Tensor]:
        return {
            "s": torch.from_numpy(np.ascontiguousarray(self.states, dtype=np.float32)),
            "a": torch.from_numpy(np.ascontiguousarray(self.actions, dtype=np.float32)),
            "s_next": torch.from_numpy(np.ascontiguousarray(self.next_states, dtype=np.float32)),
            "r": torch.from_numpy(np.ascontiguousarray(self.rewards, dtype=np.float32)),
        }


def make_synthetic_transitions(
    num_samples: int, state_dim: int, action_vocab: Sequence[str], seed: int = 0, noise: float = 0.01
) -> TransitionDataset:
    """Self-contained synthetic (s, a, s', r) with a known generator.

    s' = s + shift[a] + 0.1 * tanh(W s) + noise;  r = 1 if ||s'|| < median radius else 0.
    Used for tests and for an explicitly opted-in pipeline check only. It says
    nothing about how the model does on real trajectories.
    """
    rng = np.random.default_rng(seed)
    n_actions = len(action_vocab)
    shifts = rng.normal(0.0, 0.5, size=(n_actions, state_dim)).astype(np.float32)
    w = rng.normal(0.0, 1.0 / np.sqrt(state_dim), size=(state_dim, state_dim)).astype(np.float32)
    states = rng.normal(0.0, 1.0, size=(num_samples, state_dim)).astype(np.float32)
    a_idx = rng.integers(0, n_actions, size=num_samples)
    actions = np.eye(n_actions, dtype=np.float32)[a_idx]
    next_states = states + shifts[a_idx] + 0.1 * np.tanh(states @ w.T)
    next_states += rng.normal(0.0, noise, size=next_states.shape).astype(np.float32)
    radius = np.linalg.norm(next_states, axis=1)
    rewards = (radius < np.median(radius)).astype(np.float32)
    return TransitionDataset(states, actions, next_states.astype(np.float32), rewards, list(action_vocab))


def joint_loss(
    model: NeuralDynamicsWorldModel, batch: Dict[str, torch.Tensor], bce_weight: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """L = mean_i ||s_hat'_i - s'_i||^2 + lambda * BCE(r_hat, r). Returns (total, state_sq_err, bce)."""
    device, dtype = model._parameter_device_dtype()
    states = batch["s"].to(device=device, dtype=dtype)
    actions = batch["a"].to(device=device, dtype=dtype)
    next_states = batch["s_next"].to(device=device, dtype=dtype)
    rewards = batch["r"].to(device=device, dtype=dtype)
    pred_next, _, logit = model(states, actions)
    state_sq_err = ((pred_next - next_states) ** 2).sum(dim=-1).mean()
    bce = F.binary_cross_entropy_with_logits(logit, rewards)
    return state_sq_err + bce_weight * bce, state_sq_err, bce


def train_step(
    model: NeuralDynamicsWorldModel,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    bce_weight: float,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, sq_err, bce = joint_loss(model, batch, bce_weight)
    if not torch.isfinite(total):
        raise FloatingPointError(f"non-finite training loss: {total.item()}")
    total.backward()
    optimizer.step()
    return {"loss": float(total.item()), "state_sq_err": float(sq_err.item()), "bce": float(bce.item())}
