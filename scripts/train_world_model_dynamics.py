#!/usr/bin/env python3
"""Train NeuralDynamicsWorldModel on (s, a, s', r) transitions.

Input npz arrays: states (N, D_s), actions (N, D_a), next_states (N, D_s),
rewards (N,) in {0, 1}, episode_ids (N,) integers; optional action_vocab (D_a,)
names for one-hot actions.

File data is split by episode (split_mode="grouped_by_episode"): every row of an
episode lands in train or in val, never both. A file without episode_ids is a
hard error. Synthetic data has no episodes, so it takes a row split, logs a
WARNING and records split_mode="random_rows".

A missing --data file is a hard error. Synthetic data needs --allow-synthetic;
the run then logs a WARNING and the report records data_source="synthetic".
Metrics from synthetic data check the pipeline only, not real-world skill.

Usage:
  python scripts/train_world_model_dynamics.py --data benchmarks/artifacts/zero/trajectories_v1.npz
  python scripts/train_world_model_dynamics.py --allow-synthetic
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))

from sklearn.metrics import roc_auc_score  # noqa: E402

from gen_zero.world_model.neural_dynamics import (  # noqa: E402
    NeuralDynamicsWorldModel,
    TransitionDataset,
    joint_loss,
    make_synthetic_transitions,
    train_step,
)

logger = logging.getLogger("train_world_model_dynamics")

DEFAULT_DATA = REPO_ROOT / "benchmarks/artifacts/zero/trajectories_v1.npz"
DEFAULT_CHECKPOINT = REPO_ROOT / "benchmarks/artifacts/zero/world_model_dynamics_v1.pt"
DEFAULT_REPORT = REPO_ROOT / "benchmarks/results/world_model_training_report.json"


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--allow-synthetic", action="store_true",
                   help="if --data is missing, train on a self-contained synthetic generator instead")
    p.add_argument("--synthetic-samples", type=int, default=8192)
    p.add_argument("--state-dim", type=int, default=16, help="synthetic data only")
    p.add_argument("--num-actions", type=int, default=4, help="synthetic data only")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bce-weight", type=float, default=1.0, help="lambda in L = ||s_hat'-s'||^2 + lambda*BCE")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--done-threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return p.parse_args(argv)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning("could not read git HEAD: %s", exc)
        return None


def _load_data(args: argparse.Namespace) -> Dict[str, object]:
    if args.data.is_file():
        return {"dataset": TransitionDataset.from_npz(args.data), "data_source": "file",
                "data_path": str(args.data), "data_sha256": _sha256(args.data)}
    if not args.allow_synthetic:
        raise FileNotFoundError(
            f"training data not found: {args.data}. Pass --allow-synthetic to train on synthetic data instead."
        )
    logger.warning("DATA FILE MISSING (%s): training on SYNTHETIC transitions because --allow-synthetic was given. "
                   "Resulting metrics say nothing about real trajectories.", args.data)
    vocab = [f"action_{i}" for i in range(args.num_actions)]
    ds = make_synthetic_transitions(args.synthetic_samples, args.state_dim, vocab, seed=args.seed)
    return {"dataset": ds, "data_source": "synthetic", "data_path": None, "data_sha256": None}


def _evaluate(model: NeuralDynamicsWorldModel, ds: TransitionDataset, bce_weight: float) -> Dict[str, Optional[float]]:
    batch = ds.as_tensors()
    model.eval()
    with torch.no_grad():
        total, sq_err, bce = joint_loss(model, batch, bce_weight)
        pred_next, prob, _ = model(batch["s"], batch["a"])
    labels = ds.rewards
    if np.unique(labels).size < 2:
        logger.warning("validation labels contain a single class; AUC is undefined and reported as null")
        auc = None
    else:
        auc = float(roc_auc_score(labels, prob.numpy()))
    return {
        "loss": float(total),
        "state_sq_err": float(sq_err),
        "mse": float(((pred_next - batch["s_next"]) ** 2).mean()),
        "bce": float(bce),
        "auc": auc,
        "accuracy": float(((prob.numpy() >= 0.5) == (labels >= 0.5)).mean()),
    }


def _baselines(train: TransitionDataset, val: TransitionDataset) -> Dict[str, float]:
    """Trivial predictors the model must beat: identity transition and constant base-rate reward."""
    p = float(np.clip(train.rewards.mean(), 1e-6, 1 - 1e-6))
    r = val.rewards
    return {
        "identity_transition_val_mse": float(((val.next_states - val.states) ** 2).mean()),
        "constant_reward_val_bce": float(-(r * np.log(p) + (1 - r) * np.log(1 - p)).mean()),
    }


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    torch.manual_seed(args.seed)
    t0 = time.perf_counter()

    loaded = _load_data(args)
    ds: TransitionDataset = loaded["dataset"]
    grouped = loaded["data_source"] == "file"
    if grouped and ds.episode_ids is None:
        raise KeyError(f"{args.data} lacks episode_ids; the grouped-by-episode split needs them")
    if not grouped:
        logger.warning("SYNTHETIC data has no episodes: using a random ROW split (split_mode=random_rows)")
    train_idx, val_idx = ds.split_indices(args.val_fraction, args.seed, group_by_episode=grouped)
    train, val = ds.subset(train_idx), ds.subset(val_idx)
    split_info = {"split_mode": "grouped_by_episode" if grouped else "random_rows"}
    if grouped:
        split_info.update({
            "train_episodes": int(np.unique(train.episode_ids).size),
            "val_episodes": int(np.unique(val.episode_ids).size),
            "val_episode_ids": np.unique(val.episode_ids).tolist(),
        })
    logger.info("data_source=%s split_mode=%s N=%d train=%d val=%d train_episodes=%s val_episodes=%s D_s=%d D_a=%d",
                loaded["data_source"], split_info["split_mode"], len(ds), len(train), len(val),
                split_info.get("train_episodes"), split_info.get("val_episodes"), ds.state_dim, ds.action_dim)

    model = NeuralDynamicsWorldModel(
        state_dim=ds.state_dim, action_dim=ds.action_dim, hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks, action_vocab=ds.action_vocab, done_threshold=args.done_threshold,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_t = train.as_tensors()
    gen = torch.Generator().manual_seed(args.seed)
    history = []
    for epoch in range(args.epochs):
        perm = torch.randperm(len(train), generator=gen)
        stats = []
        for start in range(0, len(train), args.batch_size):
            idx = perm[start:start + args.batch_size]
            stats.append(train_step(model, optimizer, {k: v[idx] for k, v in train_t.items()}, args.bce_weight))
        epoch_train = float(np.mean([s["loss"] for s in stats]))
        epoch_val = _evaluate(model, val, args.bce_weight)
        history.append({"epoch": epoch + 1, "train_loss": epoch_train, "val_loss": epoch_val["loss"],
                        "val_auc": epoch_val["auc"]})
        logger.info("epoch %d/%d train_loss=%.5f val_loss=%.5f val_auc=%s",
                    epoch + 1, args.epochs, epoch_train, epoch_val["loss"], epoch_val["auc"])
    train_seconds = time.perf_counter() - t0

    model.save_checkpoint(args.checkpoint)
    # Metrics come from the exported artifact, reloaded through the fail-closed loader.
    exported = NeuralDynamicsWorldModel.from_checkpoint(args.checkpoint)
    final_train = _evaluate(exported, train, args.bce_weight)
    final_val = _evaluate(exported, val, args.bce_weight)

    report = {
        "schema": "gen_zero.world_model_training_report.v1",
        "model": "NeuralDynamicsWorldModel",
        "data_source": loaded["data_source"],
        "data_path": loaded["data_path"],
        "data_sha256": loaded["data_sha256"],
        "claim_scope": ("synthetic generator: pipeline-correctness evidence only, no claim about real trajectories"
                        if loaded["data_source"] == "synthetic" else "measured on the given trajectory file"),
        "num_samples": len(ds),
        "num_train": len(train),
        "num_val": len(val),
        **split_info,
        "state_dim": ds.state_dim,
        "action_dim": ds.action_dim,
        "action_vocab": ds.action_vocab,
        "hyperparameters": {k: vars(args)[k] for k in (
            "hidden_dim", "num_blocks", "epochs", "batch_size", "lr", "bce_weight",
            "val_fraction", "done_threshold", "seed")},
        "loss_definition": "mean_i ||s_hat'_i - s'_i||^2 (sum over state dims) + bce_weight * BCE(r_hat, r)",
        "final_train_loss": final_train["loss"],
        "final_val_loss": final_val["loss"],
        "train_mse": final_train["mse"],
        "val_mse": final_val["mse"],
        "val_state_sq_err": final_val["state_sq_err"],
        "val_bce": final_val["bce"],
        "val_auc": final_val["auc"],
        "val_reward_accuracy": final_val["accuracy"],
        "baselines": _baselines(train, val),
        "history": history,
        "wall_time_s": time.perf_counter() - t0,
        "train_time_s": train_seconds,
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "git_head": _git_head(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": "cpu",
        "loadavg": list(os.getloadavg()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("checkpoint=%s sha256=%s report=%s", args.checkpoint, report["checkpoint_sha256"], args.report)
    logger.info("val_mse=%.6f (identity baseline %.6f) val_bce=%.5f (constant baseline %.5f) val_auc=%s",
                report["val_mse"], report["baselines"]["identity_transition_val_mse"],
                report["val_bce"], report["baselines"]["constant_reward_val_bce"], report["val_auc"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
