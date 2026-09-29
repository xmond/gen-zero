#!/usr/bin/env python3
"""Persisted downstream probe + manifold evaluation for the Qwen2.5-72B 13-task feature extraction.

Replaces the one-off, no-source run that produced
``benchmarks/results/qwen72b_linear_probe_massive_en_report.json``: any future re-run of any of
the 13 tasks goes through this file instead of a REPL command nobody kept.

``--task massive_en --probe-c 0.01`` does NOT bit-exactly reproduce that earlier report's
``test_accuracy: 0.8971428571428571``. This script gets ``0.8914285714285715`` instead -- a 2/350
prediction delta, verified STABLE across ``max_iter`` in {2000, 5000, 20000} and ``tol`` down to
1e-10 (so it is not an unconverged-optimizer artifact; both runs converge to the same digit at
every reported precision). ``majority_class_baseline_test_accuracy`` (0.16285714285714287) and
``train_accuracy`` (1.0) match the original exactly, meaning the label reconstruction below (test
labels come from ``grand_challenge_data.load_test`` + ``candidates.index(ground_truth)``, since
the npz itself carries no test labels) is almost certainly correct: the class balance and the
n=1000 perfectly-separable training fit both agree bit-for-bit. The likely source of the 2-row
delta is sklearn version drift (the original run's sklearn version was never recorded, unlike
this one) or a difference in how LogisticRegression's `lbfgs` solver was invoked; it was NOT
chased further here, because tuning solver knobs until a number matches would defeat the point of
having a source-of-truth script. Whoever needs the earlier exact number should treat this file's
output as the current ground truth, not retroactively adjust it to match a report with no source.

Two independent measurements, either or both:

1. **Linear probe** (``--probe``): fits ``sklearn.linear_model.LogisticRegression`` on
   ``train_full``/``train_label`` from the task's ``<task>.npz`` and scores it on ``test_full``
   against ground truth pulled from ``grand_challenge_data.load_test`` -- the npz itself carries
   no test labels, only ``test_ids``, so the label for each test row is recovered the same way
   the extractor built ``train_label``: ``row["candidates"].index(row["ground_truth"])``
   (``gpu_extract_qwen72b_13tasks.py:212``), read per test row rather than assumed shared, and
   reordered to match the npz's own ``test_ids`` order.

2. **Manifold alignment** (``--reference-npz``): linear CKA + orthogonal-Procrustes residual
   against a second model's feature file for the same task, via
   ``cross_model_manifold_alignment.compare`` (reused unchanged, not reimplemented -- it already
   re-verifies row-id alignment before computing any geometry). Only tasks with a second real
   extraction locally support this; today that is exactly ``massive_en`` against
   ``data/extracted_features/gte7b_cpu/massive_en.npz``.

Every report records numpy/sklearn versions and the exact CLI args, so a number without a
matching report is not trusted, and a report can be told apart from a future run with different
library versions.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[2]
SUITES = REPO / "benchmarks" / "suites"
sys.path.insert(0, str(SUITES))

import grand_challenge_data as gd  # noqa: E402
from cross_model_manifold_alignment import compare as manifold_compare, load_features  # noqa: E402

DEFAULT_DATA_DIR = REPO / "data" / "extracted_features" / "qwen72b"
DEFAULT_GTE7B_MASSIVE_EN = REPO / "data" / "extracted_features" / "gte7b_cpu" / "massive_en.npz"
EXPECTED_FEATURE_DIM = 8192


def npz_path(data_dir: Path, task: str) -> Path:
    return data_dir / "features" / f"{task}.npz"


def build_test_labels(task: str, test_ids: np.ndarray) -> np.ndarray:
    """y_test[i] = candidates.index(ground_truth) for the test row whose id is test_ids[i], in
    that exact order -- the same encoding gpu_extract_qwen72b_13tasks.py used for train_label."""
    rows_by_id = {r["id"]: r for r in gd.load_test(task)}
    labels = np.empty(test_ids.shape[0], dtype=np.int64)
    for i, tid in enumerate(test_ids.tolist()):
        row = rows_by_id.get(tid)
        if row is None:
            raise ValueError(f"{task}: test_ids[{i}]={tid!r} has no matching row in gd.load_test('{task}')")
        labels[i] = row["candidates"].index(row["ground_truth"])
    return labels


def run_linear_probe(task: str, data_dir: Path, probe_c: float, max_iter: int) -> Dict[str, object]:
    path = npz_path(data_dir, task)
    data = load_features(path)  # raises on any internal row-count disagreement
    X_train, y_train = data["train_full"], data["train_label"].astype(np.int64)
    X_test = data["test_full"]
    if X_train.shape[1] != EXPECTED_FEATURE_DIM or X_test.shape[1] != EXPECTED_FEATURE_DIM:
        raise ValueError(f"{task}: feature dim {X_train.shape[1]}/{X_test.shape[1]} != {EXPECTED_FEATURE_DIM}")
    y_test = build_test_labels(task, data["test_ids"])
    K = int(max(y_train.max(), y_test.max())) + 1

    # sklearn >=1.5 dropped `multi_class`: lbfgs always fits multinomial softmax directly now.
    clf = LogisticRegression(C=probe_c, solver="lbfgs", max_iter=max_iter)
    t0 = time.time()
    clf.fit(X_train, y_train)
    fit_seconds = time.time() - t0

    train_accuracy = float(clf.score(X_train, y_train))
    test_accuracy = float(clf.score(X_test, y_test))
    counts = np.bincount(y_test, minlength=K)
    majority_baseline = float(counts.max() / y_test.shape[0])
    y_pred = clf.predict(X_test)
    confusion = np.zeros((K, K), dtype=np.int64)
    for true, pred in zip(y_test.tolist(), y_pred.tolist()):
        confusion[true, pred] += 1

    return {
        "task": task,
        "model": "qwen72b",
        "feature_dim": EXPECTED_FEATURE_DIM,
        "K": K,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "probe_C": probe_c,
        "fit_info": {"device": "cpu", "solver": "sklearn-lbfgs", "max_iter": max_iter, "fit_seconds": fit_seconds},
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "majority_class_baseline_test_accuracy": majority_baseline,
        "confusion_matrix": confusion.tolist(),
    }


def run_manifold_alignment(task: str, data_dir: Path, reference_npz: Path) -> Dict[str, object]:
    return manifold_compare(npz_path(data_dir, task), reference_npz, blocks=("train_full", "test_full"))


def build_report(task: str, args: argparse.Namespace) -> Dict[str, object]:
    report: Dict[str, object] = {
        "task": task,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {
            "script": "benchmarks/suites/run_qwen72b_downstream_eval.py",
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()} | {"task": task},
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
        },
    }
    if args.probe:
        report["linear_probe"] = run_linear_probe(task, args.data_dir, args.probe_c, args.max_iter)
    if args.reference_npz is not None:
        report["manifold_alignment"] = run_manifold_alignment(task, args.data_dir, args.reference_npz)
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", action="append", dest="tasks", choices=gd.TASKS,
                        help="one of the 13 task names; repeat for multiple, omit for all 13")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="directory holding manifest.json + features/<task>.npz "
                             f"(default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--probe", action=argparse.BooleanOptionalAction, default=True,
                        help="fit + score the linear probe (default: on)")
    parser.add_argument("--probe-c", type=float, default=0.01, dest="probe_c",
                        help="LogisticRegression inverse regularization strength (default: 0.01, "
                             "matching the original massive_en report)")
    parser.add_argument("--max-iter", type=int, default=2000, dest="max_iter",
                        help="LogisticRegression lbfgs max_iter (default: 2000, enough for "
                             "8192-D features to converge without the ConvergenceWarning)")
    parser.add_argument("--reference-npz", type=Path, default=None,
                        help="a second model's <task>.npz to compute CKA + Procrustes against "
                             f"(e.g. {DEFAULT_GTE7B_MASSIVE_EN} for --task massive_en)")
    parser.add_argument("--out", type=Path, default=None,
                        help="write JSON report(s) here; a single path for one task, or a "
                             "directory (one file per task) when --task is repeated/omitted")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    tasks = args.tasks or list(gd.TASKS)
    reports = {task: build_report(task, args) for task in tasks}

    if args.out is None:
        print(json.dumps(reports if len(tasks) > 1 else reports[tasks[0]], indent=2))
    elif len(tasks) == 1:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(reports[tasks[0]], indent=2) + "\n")
        print(f"wrote {args.out}")
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        for task, report in reports.items():
            p = args.out / f"qwen72b_downstream_eval_{task}.json"
            p.write_text(json.dumps(report, indent=2) + "\n")
            print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
