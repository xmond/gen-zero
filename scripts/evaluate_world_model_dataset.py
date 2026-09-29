#!/usr/bin/env python3
"""Evaluate the exported neural world model on its recorded trajectory dataset.

The split is the trainer's grouped-by-episode split, rebuilt with
TransitionDataset.split_indices and checked against the training report
(same checkpoint, same data, same val episode ids). Val rollouts therefore run
only on episodes the model never saw in training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from gen_zero.world_model.neural_dynamics import NeuralDynamicsWorldModel, TransitionDataset  # noqa: E402

LOG = logging.getLogger(__name__)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def classification(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if not np.all(np.isin(labels, [0, 1])) or not np.all(np.isfinite(probabilities)) or not np.all((0 <= probabilities) & (probabilities <= 1)):
        raise ValueError("classification requires binary labels and finite probabilities in [0,1]")
    pred = probabilities >= threshold
    positive = labels == 1
    tp = int(np.sum(pred & positive))
    tn = int(np.sum(~pred & ~positive))
    fp = int(np.sum(pred & ~positive))
    fn = int(np.sum(~pred & positive))
    clipped = np.clip(probabilities.astype(np.float64), 1e-12, 1 - 1e-12)
    return {
        "bce": float(np.mean(-(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)))),
        "accuracy": (tp + tn) / len(labels),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def state_metrics(pred, target):
    if not np.all(np.isfinite(pred)):
        raise ValueError("model produced non-finite states")
    dot = np.sum(pred.astype(np.float64) * target, axis=1)
    norm = np.linalg.norm(pred.astype(np.float64), axis=1) * np.linalg.norm(target.astype(np.float64), axis=1)
    cosine = np.divide(dot, norm, out=np.zeros_like(dot), where=norm > 0)
    return {"mse": float(np.mean((pred.astype(np.float64) - target) ** 2)),
            "cosine_similarity": float(np.mean(cosine))}


def predict(model, ds, batch_size=256):
    states, probs = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(ds), batch_size):
            end = start + batch_size
            s, p, _ = model(torch.from_numpy(ds.states[start:end]), torch.from_numpy(ds.actions[start:end]))
            states.append(s.numpy())
            probs.append(p.numpy())
    return np.concatenate(states), np.concatenate(probs)


def rollout(model, ds, episode_ids, indices, horizon=10):
    """Autoregressive rollout from each episode's first row, over whole episodes in ``indices``.

    Step k feeds the model's own k-1 state back in; actions are the recorded ones
    (open loop, teacher-forced actions). A partial episode in ``indices`` means the
    split cut an episode, so it raises instead of being skipped.
    """
    selected = np.zeros(len(episode_ids), dtype=bool)
    selected[np.asarray(indices, dtype=np.int64)] = True
    episodes = []
    for eid in np.unique(episode_ids[selected]):
        rows = np.flatnonzero(episode_ids == eid)
        if not selected[rows].all():
            raise ValueError(f"episode {int(eid)} is split across train/val; rollout needs whole episodes")
        episodes.append(rows)
    mse = [[] for _ in range(horizon)]
    identity_mse = [[] for _ in range(horizon)]
    cosine = [[] for _ in range(horizon)]
    true_alive = [[] for _ in range(horizon)]
    pred_alive = [[] for _ in range(horizon)]
    with torch.inference_mode():
        for rows in episodes:
            s0 = ds.states[rows[0]].astype(np.float64)
            z = ds.states[rows[0]].copy()
            alive_true, alive_pred = True, True
            for k, row in enumerate(rows[:horizon]):
                z, _, done = model.step(z, ds.actions[row])
                if not np.all(np.isfinite(z)):
                    raise ValueError(f"rollout diverged to non-finite state at episode {episode_ids[row]} step {k + 1}")
                zk, target = z.astype(np.float64), ds.next_states[row].astype(np.float64)
                mse[k].append(float(np.mean((zk - target) ** 2)))
                identity_mse[k].append(float(np.mean((s0 - target) ** 2)))
                denom = np.linalg.norm(zk) * np.linalg.norm(target)
                cosine[k].append(float(zk @ target / denom) if denom > 0 else 0.0)
                alive_true = alive_true and bool(ds.rewards[row] >= 0.5)
                alive_pred = alive_pred and not done
                true_alive[k].append(alive_true)
                pred_alive[k].append(alive_pred)
    steps = []
    for k in range(horizon):
        n = len(mse[k])
        if n == 0:
            steps.append({"step": k + 1, "n": 0})
            continue
        t, p_ = np.array(true_alive[k]), np.array(pred_alive[k])
        steps.append({
            "step": k + 1, "n": n, "reach_rate": n / len(episodes),
            "mse": float(np.mean(mse[k])), "mse_std": float(np.std(mse[k])),
            "cumulative_mean_mse": float(np.mean([x for j in range(k + 1) for x in mse[j]])),
            "identity_baseline_mse": float(np.mean(identity_mse[k])),
            "cosine_similarity": float(np.mean(cosine[k])),
            "true_survival_rate": float(t.mean()), "predicted_survival_rate": float(p_.mean()),
            "survival_agreement": float((t == p_).mean()),
        })
    return {"episodes": len(episodes), "horizon": horizon, "steps": steps}


ROLLOUT_DEFINITIONS = {
    "n": "episodes with at least k transitions",
    "reach_rate": "n / episodes in the split",
    "mse": "mean over episodes of MSE(z_hat_k, s'_k); z_hat_k is the k-step autoregressive state, so this is accumulated drift",
    "cumulative_mean_mse": "mean of all per-episode MSE values at steps 1..k",
    "identity_baseline_mse": "MSE of predicting the episode's first state s_0 at every step",
    "cosine_similarity": "mean cosine(z_hat_k, s'_k)",
    "true_survival_rate": "share of the n episodes whose recorded reward is 1 at every step 1..k",
    "predicted_survival_rate": "share of the n episodes where the model's done flag stayed false at every step 1..k",
    "survival_agreement": "share of the n episodes where true and predicted survival at step k agree",
    "actions": "recorded actions (open loop); rollout continues after a predicted done",
}


def _checked_split(ds, model_path, data_path, training_report_path):
    """Rebuild the trainer's grouped split and refuse if it is not the one the checkpoint was trained on."""
    if ds.episode_ids is None:
        raise KeyError("episode_ids required for the grouped-by-episode split")
    report = json.loads(Path(training_report_path).read_text())
    if report.get("split_mode") != "grouped_by_episode":
        raise ValueError(f"training report split_mode={report.get('split_mode')!r}, expected grouped_by_episode")
    for key, path in (("checkpoint_sha256", model_path), ("data_sha256", data_path)):
        if report.get(key) != sha256(path):
            raise ValueError(f"training report {key} does not match {path}; the split cannot be trusted")
    hp = report["hyperparameters"]
    train_idx, val_idx = ds.split_indices(hp["val_fraction"], hp["seed"], group_by_episode=True)
    val_eps = np.unique(ds.episode_ids[val_idx]).tolist()
    if val_eps != report["val_episode_ids"]:
        raise ValueError("recomputed val episodes differ from the training report")
    return {"train": train_idx, "val": val_idx}, hp


def evaluate(model_path, data_path, training_report_path):
    model = NeuralDynamicsWorldModel.from_checkpoint(model_path)
    ds = TransitionDataset.from_npz(data_path)
    if (ds.state_dim, ds.action_dim) != (model.state_dim, model.action_dim):
        raise ValueError("dataset dimensions do not match checkpoint")
    splits, hp = _checked_split(ds, model_path, data_path, training_report_path)
    episode_ids = ds.episode_ids
    train_rate = float(np.mean(ds.rewards[splits["train"]]))
    report = {"schema": "gen_zero.world_model_dataset_eval.v2", "model_sha256": sha256(model_path),
              "data_sha256": sha256(data_path), "num_samples": len(ds),
              "split": {"split_mode": "grouped_by_episode",
                        "method": "TransitionDataset.split_indices: permute unique episode ids, "
                                  "first round(E*val_fraction) episodes to val; verified against the training report",
                        "seed": hp["seed"], "val_fraction": hp["val_fraction"],
                        "train_count": len(splits["train"]), "val_count": len(splits["val"]),
                        "train_episodes": int(np.unique(episode_ids[splits["train"]]).size),
                        "val_episodes": int(np.unique(episode_ids[splits["val"]]).size),
                        "shared_episodes": int(np.intersect1d(episode_ids[splits["train"]],
                                                              episode_ids[splits["val"]]).size)},
              "reward_positive_means": "safe or goal; 0 means dead or doomed",
              "done_rule": "reward probability below checkpoint threshold", "constant_reward_probability": train_rate,
              "results": {}}
    for name, idx in splits.items():
        part = ds.subset(idx)
        pred, prob = predict(model, part)
        constant_state = np.broadcast_to(ds.next_states[splits["train"]].mean(axis=0), part.next_states.shape)
        constant_prob = np.full(len(part), train_rate)
        report["results"][name] = {
            "n": len(part), "model": {"state": state_metrics(pred, part.next_states),
                                   "reward": classification(part.rewards, prob, model.done_threshold)},
            "identity_state_baseline": state_metrics(part.states, part.next_states),
            "constant_state_baseline": state_metrics(constant_state, part.next_states),
            "constant_reward_baseline": classification(part.rewards, constant_prob, model.done_threshold),
            "cpu_latency": latency(model, part),
        }
    report["rollout_definitions"] = ROLLOUT_DEFINITIONS
    report["val_episodes_rollout_drift"] = rollout(model, ds, episode_ids, splits["val"])
    report["train_episodes_rollout_drift"] = rollout(model, ds, episode_ids, splits["train"])
    return report


def _print_rollout(title, r):
    print(f"\n### {title} ({r['episodes']} whole episodes)\n")
    print("| Step | n | Reach | MSE | Cum. MSE | Identity MSE | Cosine | True surv. | Pred. surv. | Agree |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for x in r["steps"]:
        if x["n"]:
            print(f'| {x["step"]} | {x["n"]} | {x["reach_rate"]:.3f} | {x["mse"]:.5f} | {x["cumulative_mean_mse"]:.5f} | '
                  f'{x["identity_baseline_mse"]:.5f} | {x["cosine_similarity"]:.4f} | {x["true_survival_rate"]:.3f} | '
                  f'{x["predicted_survival_rate"]:.3f} | {x["survival_agreement"]:.3f} |')


def latency(model, ds, repeats=200, warmup=20):
    n = min(len(ds), repeats)
    for i in range(min(warmup, n)):
        model.step(ds.states[i], ds.actions[i])
    durations = []
    for i in range(n):
        start = time.perf_counter_ns()
        model.step(ds.states[i], ds.actions[i])
        durations.append((time.perf_counter_ns() - start) / 1000)
    start = time.perf_counter()
    predict(model, ds)
    elapsed = time.perf_counter() - start
    return {"step_samples": n, "step_us_mean": float(np.mean(durations)),
            "step_us_p50": float(np.percentile(durations, 50)),
            "step_us_p90": float(np.percentile(durations, 90)),
            "step_us_p99": float(np.percentile(durations, 99)),
            "batch_samples": len(ds), "batch_size": 256,
            "batch_samples_per_sec": len(ds) / elapsed}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("benchmarks/artifacts/zero/world_model_dynamics_v1.pt"))
    parser.add_argument("--data", type=Path, default=Path("benchmarks/artifacts/zero/trajectories_v1.npz"))
    parser.add_argument("--training-report", type=Path,
                        default=Path("benchmarks/results/world_model_training_report.json"),
                        help="source of the grouped split; must name the same checkpoint and data")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results/world_model_dataset_eval_report.json"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(1)
    result = evaluate(args.model, args.data, args.training_report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    sp = result["split"]
    print(f'split: {sp["split_mode"]}; train={sp["train_count"]} rows/{sp["train_episodes"]} episodes, '
          f'val={sp["val_count"]} rows/{sp["val_episodes"]} episodes, shared episodes={sp["shared_episodes"]}\n')
    print("| Split | N | Model MSE | Identity MSE | Constant MSE | Cosine | BCE | Accuracy | F1 | AUC |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, r in result["results"].items():
        s, c = r["model"]["state"], r["model"]["reward"]
        print(f'| {name} | {r["n"]} | {s["mse"]:.6f} | {r["identity_state_baseline"]["mse"]:.6f} | {r["constant_state_baseline"]["mse"]:.6f} | {s["cosine_similarity"]:.4f} | {c["bce"]:.5f} | {c["accuracy"]:.4f} | {c["f1"]:.4f} | {c["roc_auc"]:.4f} |')
    for name, r in result["results"].items():
        print(f'{name}: step mean/p50/p90/p99 us=' + "/".join(f'{r["cpu_latency"][k]:.1f}' for k in ("step_us_mean", "step_us_p50", "step_us_p90", "step_us_p99")) + f'; batch samples/sec={r["cpu_latency"]["batch_samples_per_sec"]:.1f}')
    _print_rollout("Held-out val episodes: multi-step autoregressive rollout drift", result["val_episodes_rollout_drift"])
    _print_rollout("Train episodes: multi-step autoregressive rollout drift", result["train_episodes_rollout_drift"])
    print(f"\nreport: {args.out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        LOG.exception("evaluation failed")
        sys.exit(1)
