#!/usr/bin/env python3
"""Train the Zero task head on the extracted calibration features.

Selects ``weight_decay`` by task-stratified k-fold cross-validation on the
312 calibration records only (never touches the frozen test set), comparing
against the plain-cosine baseline (``W = I``, i.e. the unsupervised manifold
score every other Zero decision already uses) fold by fold. The winning
hyperparameters are then refit on all 312 records and saved. A CV table and
the final artifact's own provenance are the evidence trail.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

from gen_zero.causal.zero_task_head import ZeroTaskHead  # noqa: E402

FEATURES = REPO / "benchmarks" / "artifacts" / "zero" / "zero_calibration_features_v1.npz"
OUT = REPO / "benchmarks" / "artifacts" / "zero" / "zero_task_head_v1.npz"
REPORT = REPO / "benchmarks" / "results" / "zero_task_head_training_report.json"
WEIGHT_DECAY_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
FOLDS = 4


def load_features(path: Path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        z0 = data["z0"]
        flat = data["candidates_flat"]
        offsets = data["offsets"]
        counts = data["counts"]
        positive = data["positive_indices"]
        tasks = data["tasks"]
        sample_ids = data["sample_ids"]
    blocks = [flat[o:o + c] for o, c in zip(offsets, counts)]
    return meta, z0, blocks, positive, tasks, sample_ids


def task_stratified_folds(tasks: np.ndarray, *, folds: int, seed: int) -> np.ndarray:
    """Deterministic fold id per record, balanced within each task."""
    rng = np.random.default_rng(seed)
    fold_id = np.full(len(tasks), -1, dtype=np.int64)
    for task in sorted(set(tasks.tolist())):
        idx = np.flatnonzero(tasks == task)
        rng.shuffle(idx)
        fold_id[idx] = np.arange(len(idx)) % folds
    assert (fold_id >= 0).all()
    return fold_id


def accuracy(head: ZeroTaskHead, z0, blocks, positive, idx) -> float:
    correct = 0
    for i in idx:
        scores = head.score(z0[i], blocks[i])
        correct += int(np.argmax(scores) == positive[i])
    return correct / len(idx) if len(idx) else float("nan")


def cosine_baseline_accuracy(z0, blocks, positive, idx) -> float:
    correct = 0
    for i in idx:
        scores = blocks[i] @ z0[i]
        correct += int(np.argmax(scores) == positive[i])
    return correct / len(idx) if len(idx) else float("nan")


def cross_validate(z0, blocks, positive, tasks, *, epochs: int, lr: float, seed: int, split: str = "train") -> dict:
    fold_id = task_stratified_folds(tasks, folds=FOLDS, seed=seed)
    table = []
    for wd in WEIGHT_DECAY_GRID:
        fold_accs, baseline_accs = [], []
        for fold in range(FOLDS):
            train_idx = np.flatnonzero(fold_id != fold)
            held_idx = np.flatnonzero(fold_id == fold)
            head, _history = ZeroTaskHead.fit(
                [z0[i] for i in train_idx], [blocks[i] for i in train_idx],
                [int(positive[i]) for i in train_idx], manifold_sha256="cv", source="cv",
                split=split, encoder_id="cv", weight_decay=wd, epochs=epochs, lr=lr, seed=seed)
            fold_accs.append(accuracy(head, z0, blocks, positive, held_idx))
            baseline_accs.append(cosine_baseline_accuracy(z0, blocks, positive, held_idx))
        table.append({"weight_decay": wd, "held_out_accuracy_mean": float(np.mean(fold_accs)),
                      "held_out_accuracy_per_fold": fold_accs,
                      "cosine_baseline_accuracy_mean": float(np.mean(baseline_accs))})
        print(f"[cv] weight_decay={wd:<6} held_out_acc={np.mean(fold_accs):.3f} "
              f"cosine_baseline={np.mean(baseline_accs):.3f}", flush=True)
    return {"folds": FOLDS, "grid": table}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    meta, z0, blocks, positive, tasks, sample_ids = load_features(args.features)
    if meta.get("label_free_encoder_input") is not True:
        raise ValueError("feature producer must attest label-free encoder inputs")
    split = meta.get("split", "train")
    if split not in ("calibration", "train"):
        raise ValueError("task head training requires train or calibration split")

    cv = cross_validate(z0, blocks, positive, tasks, epochs=args.epochs, lr=args.lr, seed=args.seed, split=split)
    best = max(cv["grid"], key=lambda row: row["held_out_accuracy_mean"])
    best_wd = best["weight_decay"]
    cosine_overall = float(np.mean([row["cosine_baseline_accuracy_mean"] for row in cv["grid"]]))
    print(f"[select] weight_decay={best_wd} cv_held_out_acc={best['held_out_accuracy_mean']:.3f} "
          f"vs cosine_baseline={cosine_overall:.3f}")

    head, history = ZeroTaskHead.fit(
        list(z0), blocks, [int(p) for p in positive], manifold_sha256=meta["manifold_sha256"],
        source=meta["source"], split=split, encoder_id=meta["encoder_id"],
        weight_decay=best_wd, epochs=args.epochs, lr=args.lr, seed=args.seed)
    train_acc = accuracy(head, z0, blocks, positive, np.arange(len(z0)))
    train_cosine_acc = cosine_baseline_accuracy(z0, blocks, positive, np.arange(len(z0)))
    print(f"[final] train (all {len(z0)} records) accuracy={train_acc:.3f} "
          f"cosine_baseline={train_cosine_acc:.3f}")

    per_task_cv = defaultdict(list)
    fold_id = task_stratified_folds(tasks, folds=FOLDS, seed=args.seed)
    for fold in range(FOLDS):
        train_idx = np.flatnonzero(fold_id != fold)
        held_idx = np.flatnonzero(fold_id == fold)
        fold_head, _ = ZeroTaskHead.fit(
            [z0[i] for i in train_idx], [blocks[i] for i in train_idx],
            [int(positive[i]) for i in train_idx], manifold_sha256="cv", source="cv",
            split=split, encoder_id="cv", weight_decay=best_wd, epochs=args.epochs,
            lr=args.lr, seed=args.seed)
        for i in held_idx:
            task = str(tasks[i])
            pred_head = int(np.argmax(fold_head.score(z0[i], blocks[i])))
            pred_cos = int(np.argmax(blocks[i] @ z0[i]))
            per_task_cv[task].append((pred_head == positive[i], pred_cos == positive[i]))

    per_task_report = {}
    for task, outcomes in sorted(per_task_cv.items()):
        head_correct = sum(o[0] for o in outcomes)
        cos_correct = sum(o[1] for o in outcomes)
        per_task_report[task] = {"n": len(outcomes), "head_cv_accuracy": head_correct / len(outcomes),
                                 "cosine_baseline_accuracy": cos_correct / len(outcomes)}
        print(f"[cv/task] {task:<16} n={len(outcomes):<3} head={head_correct / len(outcomes):.3f} "
              f"cosine={cos_correct / len(outcomes):.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    head.save(args.out)
    report = {
        "features_file": str(args.features),
        "chosen_weight_decay": best_wd,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "cross_validation": cv,
        "cross_validation_per_task": per_task_report,
        "final_train_fit": {
            "records": len(z0),
            "head_accuracy_on_training_records": train_acc,
            "cosine_baseline_accuracy_on_training_records": train_cosine_acc,
            "initial_loss": history[0],
            "final_loss": history[-1],
        },
        "artifact": str(args.out),
        "artifact_provenance": head.provenance,
        "honesty_note": (
            "head_accuracy_on_training_records is an in-sample number (the head "
            "was fit on these same 312 rows) and is reported only as a training "
            "diagnostic; cross_validation/cross_validation_per_task are the only "
            "numbers in this report that estimate out-of-sample accuracy, and even "
            "those are held out within the 312-row calibration split, not the "
            "frozen 930-row test set."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[final] wrote head -> {args.out}")
    print(f"[final] wrote report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
