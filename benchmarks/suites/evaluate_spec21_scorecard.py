#!/usr/bin/env python3
"""Spec 21 grand scorecard: the Spec 21 heads inside the 13-task 1-SE evaluation.

Candidates (one shared 5-fold OOF pool per task, the same folds as evaluate_full_13_grand_scorecard.py):
  linear_probe, bbp_probe (Gavish-Donoho truncated probe), lw_lda (Ledoit-Wolf LDA),
  adapter (Spec 19 residual adapter), supcon (adapter + SupCon)   x   logit-adjustment tau in {0, 0.5, 1}.

Selection is Breiman's 1-SE rule on OOF accuracy: among candidates that pass the prior-collapse gate, take the
best OOF accuracy, then the SIMPLEST candidate (fewest supervised parameters, Spec 21 S6 counting) whose OOF
accuracy is within one SE of the best. This is the canonical rule. The legacy ladder in the Spec 19 script
(challenger must beat the incumbent by one SE) is a different, stricter rule, so the legacy pool is re-run under
the canonical rule as the `legacy3_tau0` arm instead of being compared to the old numbers by assumption.

Two deliberate readings of the brief, stated so nobody has to guess:
  * The prior-collapse gate that DECIDES the selection runs on OOF predictions (train labels only). Using test
    accuracy to pick or drop an expert would be selection on the test set. The test-set gate
    (`check_prior_collapse_gate(y_test, pred, train_priors)`) is applied AFTER selection: a chosen expert that
    fails it is reported `admitted = False` and is left out of the admitted macro. It is never swapped for
    another expert.
  * Admission is strict (fail-closed): only candidates whose OOF gate `passed` is True can be chosen. When none
    passes (collapse, below prior, or a class with recall < 10% for K >= 3), the task has NO chosen expert:
    it is reported under `tasks_without_champion`, it is left out of every macro, and a warning goes to stderr.
    There is no fallback to the best inadmissible candidate.
  * The 5-fold CV is not nested: rank / tau / expert are all chosen on the same OOF pool, so CV accuracy is
    optimistic and the pool has 15 candidates per task. The 1-SE rule limits, and does not remove, that bias.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks" / "suites"))

import evaluate_full_13_grand_scorecard as base  # noqa: E402
import grand_challenge_data as gd  # noqa: E402
import spec21_advanced_heads as s21  # noqa: E402

TAUS: Tuple[float, ...] = (0.0, 0.5, 1.0)
FAMILIES: Tuple[str, ...] = ("linear_probe", "bbp_probe", "lw_lda", "adapter", "supcon")
LEGACY_FAMILIES: Tuple[str, ...] = ("linear_probe", "adapter", "supcon")
ARMS: Dict[str, Tuple[Tuple[str, ...], Tuple[float, ...]]] = {
    "legacy3_tau0": (LEGACY_FAMILIES, (0.0,)),
    "plus_new_heads_tau0": (FAMILIES, (0.0,)),
    "legacy3_plus_tau": (LEGACY_FAMILIES, TAUS),
    "full": (FAMILIES, TAUS),
}
_TIE_RANK = {"linear_probe": 0, "bbp_probe": 1, "lw_lda": 2, "adapter": 3, "supcon": 4}
ABSENT_SHIFT = -1e9          # a class with no training rows must never win the argmax
wilson = base.wilson


@dataclass(frozen=True)
class EvalConfig:
    adapter_epochs: int = 50
    full_epochs: int = 60
    cv_folds: int = base.N_FOLDS
    bbp_inner_cv: int = 4
    n_boot: int = 10000
    latency_rows: Optional[int] = None      # None = time every test row
    class_weight: Optional[str] = None      # None | "balanced": loss weighting for bbp_probe and linear_probe


@dataclass(frozen=True)
class CandStat:
    cid: str
    family: str
    rank: Optional[int]
    tau: float
    cv_acc: float
    cv_se: float
    params: int
    tie_rank: int
    gate_ok: bool


# ----------------------------------------------------------------------------- logit adjustment

def adjust_scores(raw: np.ndarray, priors: np.ndarray, tau: float) -> np.ndarray:
    """raw - tau * log(pi) through spec21.logit_adjust on the classes seen in training; classes with no
    training rows (pi == 0) get ABSENT_SHIFT so log(0 + eps) cannot turn them into the boosted winner."""
    raw = np.asarray(raw, dtype=np.float64)
    priors = np.asarray(priors, dtype=np.float64)
    present = priors > 0
    out = np.full(raw.shape, ABSENT_SHIFT, dtype=np.float64)
    out[:, present] = s21.logit_adjust(raw[:, present], priors[present], tau)
    return out


# ----------------------------------------------------------------------------- complexity + selection

def tie_rank(family: str) -> int:
    return _TIE_RANK[family]


def head_param_count(family: str, D: int, K: int, rank: Optional[int], tau: float, bbp_rank: int = 0) -> int:
    """Supervised parameter count, Spec 21 S6 convention (unsupervised maps are not counted, tau costs one)."""
    if family in ("linear_probe", "lw_lda"):
        n = K * (D + 1)
    elif family == "bbp_probe":
        n = K * (bbp_rank + 1)
    elif family in ("adapter", "supcon"):
        if not rank:
            raise ValueError(f"{family} needs a rank")
        n = 2 * D * rank + rank + D + K * (D + 1)      # W_down, b_down, W_up, b_up, then the K x D head
    else:
        raise ValueError(f"unknown family {family!r}")
    return n + (1 if tau > 0 else 0)


def breiman_select(stats: Sequence[CandStat]) -> dict:
    """Breiman 1-SE: best OOF accuracy over the gate-passing pool, then the simplest candidate within one SE of it.
    Only gate-passing candidates are eligible. If none passes, `chosen` is None: there is no champion."""
    if not stats:
        raise ValueError("no candidates to select from")
    admissible = [s for s in stats if s.gate_ok]
    if not admissible:
        return {"chosen": None, "best_cid": None, "threshold": None, "n_admissible": 0, "within_cids": []}
    best = max(admissible, key=lambda s: (s.cv_acc, -s.params, -s.tie_rank))
    threshold = best.cv_acc - best.cv_se
    within = [s for s in admissible if s.cv_acc >= threshold - 1e-12]
    chosen = min(within, key=lambda s: (s.params, s.tie_rank, -s.cv_acc, s.cid))
    return {"chosen": chosen, "best_cid": best.cid, "threshold": threshold,
            "n_admissible": len(admissible), "within_cids": sorted(s.cid for s in within)}


def _selection_json(sel: dict) -> dict:
    return {"chosen_cid": None if sel["chosen"] is None else sel["chosen"].cid, "best_cid": sel["best_cid"],
            "threshold": sel["threshold"], "n_admissible": sel["n_admissible"], "within_cids": sel["within_cids"]}


# ----------------------------------------------------------------------------- heads

class FittedHead:
    """A head fitted on the classes present in its training rows; `scores` is K-wide (absent columns are 0 and
    are masked by adjust_scores). All scoring is NumPy, so the CPU path is what gets timed."""

    def __init__(self, family: str, K: int, present: np.ndarray, priors: np.ndarray, scores_fn, fold_fn, model=None):
        self.family, self.K, self.present, self.priors = family, K, present, priors
        self._scores_fn, self._fold_fn, self.model = scores_fn, fold_fn, model

    def scores(self, X: np.ndarray) -> np.ndarray:
        compact = np.asarray(self._scores_fn(np.asarray(X)), dtype=np.float64)
        out = np.zeros((compact.shape[0], self.K))
        out[:, self.present] = compact
        return out

    def fold(self) -> Tuple[np.ndarray, np.ndarray]:
        """(W, b) with scores(X) == X @ W.T + b; only the single-GEMV heads have one."""
        if self._fold_fn is None:
            raise ValueError(f"{self.family} is not a single-GEMV head")
        W, b = self._fold_fn()
        Wf, bf = np.zeros((self.K, W.shape[1])), np.zeros(self.K)
        Wf[self.present], bf[self.present] = W, b
        return Wf, bf


def compact_labels(y: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Relabel to 0..k-1 over the classes present. int64 is explicit: on Windows np.cumsum(bool) is int32, which
    torch's CUDA cross-entropy rejects."""
    return (np.cumsum(present, dtype=np.int64) - 1)[np.asarray(y, dtype=np.int64)].astype(np.int64)


def fit_head(family: str, Xn: np.ndarray, y: np.ndarray, K: int, rank: Optional[int], cfg: EvalConfig,
             device: str, seed: int = base.FOLD_SEED, lr: float = 1e-3, epochs: Optional[int] = None) -> FittedHead:
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y, minlength=K)
    present = counts > 0
    kp = int(present.sum())
    if kp < 2:
        raise ValueError("fewer than two classes in the training rows")
    yc = compact_labels(y, present)
    priors = counts / counts.sum()
    Xn32 = np.asarray(Xn, dtype=np.float32)

    if family == "linear_probe":
        W_t, b_t = base.fit_linear_probe_torch(torch.as_tensor(Xn32, device=device),
                                                torch.as_tensor(yc, device=device), kp, C=1.0, device=device,
                                                class_weight=cfg.class_weight)
        W, b = W_t.cpu().numpy().astype(np.float64), b_t.cpu().numpy().astype(np.float64)
        return FittedHead(family, K, present, priors, lambda X: X @ W.T + b, lambda: (W, b))
    if family == "bbp_probe":
        inner_c = None if counts[present].min() >= 2 else 1.0     # inner CV needs 2 rows per class
        m = s21.BBPAdaptiveProbe.fit(Xn32.astype(np.float64), yc, kp, C=inner_c,
                                     cv_folds=cfg.bbp_inner_cv, seed=seed % (2 ** 31), class_weight=cfg.class_weight)
        return FittedHead(family, K, present, priors, m.scores, lambda: (m.W_fold, m.b_fold), model=m)
    if family == "lw_lda":
        m = s21.LedoitWolfLDAHead.fit(Xn32.astype(np.float64), yc, kp)
        return FittedHead(family, K, present, priors, m.scores, lambda: (m.W_fold, m.b_fold), model=m)
    if family in ("adapter", "supcon"):
        weights, _, _ = base.train_adapter_gpu(
            torch.as_tensor(Xn32, device=device), torch.as_tensor(yc, device=device), kp, rank=rank, lr=lr,
            wd=0.01, lambda_up=1e-3, max_epochs=epochs or cfg.adapter_epochs, batch_size=256, device=device,
            rng_seed=seed, lambda_supcon=0.5 if family == "supcon" else 0.0, tau=0.1, feature_dropout=0.1)
        return FittedHead(family, K, present, priors, lambda X: base.score_folded_numpy(X.astype(np.float32), weights),
                          None)
    raise ValueError(f"unknown family {family!r}")


# ----------------------------------------------------------------------------- statistics helpers

def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact two-sided McNemar on paired correctness masks (a vs b)."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("masks differ in length")
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = only_a + only_b
    if n == 0:
        p = 1.0
    else:
        k = min(only_a, only_b)
        p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)
    return {"only_a": only_a, "only_b": only_b, "both": int(np.sum(a & b)), "neither": int(np.sum(~a & ~b)),
            "p_two_sided": float(p)}


def _macro_boot(a_list: Sequence[np.ndarray], b_list: Optional[Sequence[np.ndarray]], n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    acc = np.zeros(n_boot)
    for i, a in enumerate(a_list):
        a = np.asarray(a, dtype=np.float64)
        idx = rng.integers(0, a.size, size=(n_boot, a.size))         # same rows for both arms: paired
        v = a[idx].mean(axis=1)
        if b_list is not None:
            v = v - np.asarray(b_list[i], dtype=np.float64)[idx].mean(axis=1)
        acc += v
    return 100.0 * acc / len(a_list)


def bootstrap_macro_ci(corr: Sequence[np.ndarray], n_boot: int = 10000, seed: int = 0) -> dict:
    """Macro accuracy with a row-level bootstrap inside every task (within-task sampling noise only)."""
    dist = _macro_boot(corr, None, n_boot, seed)
    return {"point": 100.0 * float(np.mean([np.mean(c) for c in corr])),
            "ci95": [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))]}


def paired_bootstrap_macro_delta(a: Sequence[np.ndarray], b: Sequence[np.ndarray], n_boot: int = 10000,
                                 seed: int = 0) -> dict:
    """macro(a) - macro(b) in points, resampling the SAME test rows for both arms."""
    dist = _macro_boot(a, b, n_boot, seed)
    return {"point": 100.0 * float(np.mean([np.mean(x) for x in a]) - np.mean([np.mean(x) for x in b])),
            "ci95": [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))]}


# ----------------------------------------------------------------------------- per-task evaluation

def _strategy_label(family: str, rank: Optional[int], tau: float) -> str:
    s = family if rank is None else f"{family}_r{rank}"
    return s if tau == 0 else f"{s}+LA{tau:g}"


def _cid(family: str, rank: Optional[int], tau: float) -> str:
    return f"{family if rank is None else f'{family}_r{rank}'}|tau={float(tau)}"


def _metrics(pred: np.ndarray, y: np.ndarray, K: int, priors: np.ndarray) -> dict:
    cs = base.collapse_stats(pred, y, K)
    gate = s21.check_prior_collapse_gate(y, pred, priors)
    correct = int((pred == y).sum())
    return {"correct": correct, "accuracy": 100.0 * correct / len(y), "balanced_accuracy": cs["balanced_accuracy"],
            "macro_f1": cs["macro_f1"], "max_pred_class_frac": cs["max_pred_class_frac"], "gate": gate}


def _shift(priors: np.ndarray, tau: float) -> np.ndarray:
    return adjust_scores(np.zeros((1, priors.size)), priors, tau)[0]


def evaluate_task_arrays(task: str, X_tr_raw: np.ndarray, y_tr: np.ndarray, X_te_raw: np.ndarray,
                         y_te: np.ndarray, K: int, *, ranks: Sequence[int], device: str,
                         cfg: EvalConfig = EvalConfig()) -> dict:
    y_tr, y_te = np.asarray(y_tr, dtype=np.int64), np.asarray(y_te, dtype=np.int64)
    if y_tr.max() >= K or y_te.max() >= K:
        raise ValueError(f"{task}: label out of range for K={K}")
    N, D = X_tr_raw.shape
    mu = X_tr_raw.astype(np.float32).mean(0)
    sd = X_tr_raw.astype(np.float32).std(0) + 1e-6
    Xtr = ((X_tr_raw - mu) / sd).astype(np.float32)
    Xte = ((X_te_raw - mu) / sd).astype(np.float32)
    train_priors = s21.compute_class_priors(y_tr, K)
    majority_prior = 100.0 * float(train_priors.max())
    lr_override = base.TASK_ADAPTER_OVERRIDES.get(task, {})
    bbp_rank = int(s21.estimate_gd_rank(Xtr)["rank"])                 # label-free; only orders complexity

    folds = np.random.default_rng(base.FOLD_SEED).permutation(N) % cfg.cv_folds
    fit_errors: Dict[str, str] = {}
    notes: List[str] = []

    # ---- OOF raw scores + fold-train priors per head variant (family, rank)
    variants: List[Tuple[str, Optional[int]]] = [("linear_probe", None), ("bbp_probe", None), ("lw_lda", None)]
    variants += [("adapter", r) for r in ranks] + [("supcon", r) for r in ranks]
    oof: Dict[Tuple[str, Optional[int]], np.ndarray] = {}
    fold_priors: List[np.ndarray] = []
    for k in range(cfg.cv_folds):
        tr = np.flatnonzero(folds != k)
        c = np.bincount(y_tr[tr], minlength=K)
        fold_priors.append(c / c.sum())
        if int((c == 0).sum()):
            notes.append(f"fold {k}: classes {np.flatnonzero(c == 0).tolist()} absent from fold-train rows")
    for fam, r in variants:
        raw = np.zeros((N, K))
        try:
            for k in range(cfg.cv_folds):
                tr, ho = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
                lr = lr_override.get(r, {}).get("lr", 1e-3) if fam in ("adapter", "supcon") else 1e-3
                h = fit_head(fam, Xtr[tr], y_tr[tr], K, r, cfg, device, seed=base.FOLD_SEED + k, lr=lr)
                raw[ho] = h.scores(Xtr[ho])
        except ValueError as e:                                       # e.g. BBP: nothing above the GD threshold
            fit_errors[_cid(fam, r, 0.0).split("|")[0]] = str(e)
            continue
        oof[(fam, r)] = raw

    def oof_pred(raw: np.ndarray, tau: float) -> np.ndarray:
        adj = np.zeros_like(raw)
        for k in range(cfg.cv_folds):
            ho = folds == k
            adj[ho] = adjust_scores(raw[ho], fold_priors[k], tau)
        return adj.argmax(axis=1)

    per_rank: Dict[Tuple[str, Optional[int], float], dict] = {}
    for (fam, r), raw in oof.items():
        for tau in TAUS:
            pred = oof_pred(raw, tau)
            fold_accs = [float(np.mean(pred[folds == k] == y_tr[folds == k])) for k in range(cfg.cv_folds)]
            gate = s21.check_prior_collapse_gate(y_tr, pred, train_priors)
            per_rank[(fam, r, tau)] = {
                "family": fam, "rank": r, "tau": tau, "cid": _cid(fam, r, tau),
                "cv_acc": 100.0 * float(np.mean(pred == y_tr)),
                "cv_se": 100.0 * float(np.std(fold_accs, ddof=1) / math.sqrt(cfg.cv_folds)),
                "cv_balanced_accuracy": 100.0 * gate["balanced_accuracy"],
                "oof_max_pred_class_frac": gate["max_pred_class_frac"], "oof_min_class_recall": gate["min_class_recall"],
                "gate_ok": bool(gate["passed"]), "oof_gate_reasons": gate["reasons"],
                "params": head_param_count(fam, D, K, r, tau, bbp_rank), "tie_rank": tie_rank(fam)}

    # ---- best rank per (family, tau): gate-passing first, then OOF accuracy, then the smaller rank
    cands: List[dict] = []
    for fam in FAMILIES:
        for tau in TAUS:
            rows = [v for (f, _, t), v in per_rank.items() if f == fam and t == tau]
            if rows:
                cands.append(max(rows, key=lambda v: (v["gate_ok"], v["cv_acc"], -(v["rank"] or 0))))
    if not cands:
        raise ValueError(f"{task}: no candidate could be fitted: {fit_errors}")
    stats = {c["cid"]: CandStat(c["cid"], c["family"], c["rank"], c["tau"], c["cv_acc"], c["cv_se"],
                                c["params"], c["tie_rank"], c["gate_ok"]) for c in cands}

    # ---- full-train fits + test predictions for every candidate (diagnostic: never used for selection)
    full: Dict[Tuple[str, Optional[int]], FittedHead] = {}
    test_pred: Dict[str, np.ndarray] = {}
    posthoc: Dict[str, dict] = {}
    for c in cands:
        key = (c["family"], c["rank"])
        if key not in full:
            lr = lr_override.get(c["rank"], {}).get("lr", 1e-3) if c["family"] in ("adapter", "supcon") else 1e-3
            full[key] = fit_head(c["family"], Xtr, y_tr, K, c["rank"], cfg, device, seed=base.FOLD_SEED, lr=lr,
                                 epochs=cfg.full_epochs)
        pred = adjust_scores(full[key].scores(Xte), full[key].priors, c["tau"]).argmax(axis=1)
        test_pred[c["cid"]] = pred
        m = _metrics(pred, y_te, K, train_priors)
        posthoc[c["cid"]] = {"family": c["family"], "rank": c["rank"], "tau": c["tau"], **m}

    # ---- arms: nested pools over the same OOF pool
    arms: Dict[str, dict] = {}
    for name, (fams, taus) in ARMS.items():
        pool = [stats[c["cid"]] for c in cands if c["family"] in fams and c["tau"] in taus]
        sel = breiman_select(pool)
        cid = None if sel["chosen"] is None else sel["chosen"].cid
        arms[name] = {"chosen_cid": cid, "n_pool": len(pool), "n_admissible": sel["n_admissible"],
                      "accuracy": None if cid is None else posthoc[cid]["accuracy"],
                      "correct_mask": None if cid is None else (test_pred[cid] == y_te).tolist()}

    rec = {"task": task, "n": int(len(y_te)), "n_train_rows": int(N), "K": int(K), "D": int(D),
           "majority_class_train_prior_acc": majority_prior,
           "bbp_gd_rank_full_train": bbp_rank,
           "cv_candidates": cands,
           "posthoc_test_all_candidates": posthoc,
           "arms": arms,
           "fit_errors": fit_errors, "notes": notes}
    sel = breiman_select(list(stats.values()))
    rec["selection"] = _selection_json(sel)
    ch = sel["chosen"]
    if ch is None:
        # Fail-closed: no gate-passing candidate means no champion, never the best inadmissible one.
        reasons = sorted({r for c in cands for r in c["oof_gate_reasons"]})
        print(f"WARNING {task}: no candidate passed the OOF gate ({', '.join(reasons)}); task has no chosen expert "
              f"and is left out of every macro", file=sys.stderr, flush=True)
        rec.update({"chosen_cid": None, "chosen_family": None, "chosen_rank": None, "chosen_tau": None,
                    "chosen_strategy": None, "cv_acc_chosen": None, "cv_se_chosen": None,
                    "correct": None, "accuracy": None, "wilson95": None, "balanced_accuracy": None,
                    "macro_f1": None, "max_pred_class_frac": None, "gate": None, "admitted": False,
                    "no_champion_reason": "no_admissible_candidate", "oof_gate_reasons": reasons,
                    "correct_mask": None, "decision_latency_us": None})
        return rec

    ch_cand = next(c for c in cands if c["cid"] == ch.cid)
    ch_post = posthoc[ch.cid]
    gate = ch_post["gate"]

    # ---- CPU latency of the chosen head: one row at a time, NumPy only, adjustment folded as a bias shift
    head = full[(ch.family, ch.rank)]
    shift = _shift(head.priors, ch.tau)
    n_lat = len(Xte) if cfg.latency_rows is None else min(cfg.latency_rows, len(Xte))
    lat_ms = []
    for i in range(n_lat):
        t0 = time.perf_counter()
        _ = head.scores(Xte[i:i + 1])[0] + shift
        lat_ms.append((time.perf_counter() - t0) * 1e3)

    rec.update({
        "chosen_cid": ch.cid, "chosen_family": ch.family, "chosen_rank": ch.rank, "chosen_tau": ch.tau,
        "chosen_strategy": _strategy_label(ch.family, ch.rank, ch.tau),
        "cv_acc_chosen": ch_cand["cv_acc"], "cv_se_chosen": ch_cand["cv_se"],
        "correct": ch_post["correct"], "accuracy": ch_post["accuracy"],
        "wilson95": wilson(ch_post["correct"], len(y_te)),
        "balanced_accuracy": ch_post["balanced_accuracy"], "macro_f1": ch_post["macro_f1"],
        "max_pred_class_frac": ch_post["max_pred_class_frac"],
        "gate": gate, "admitted": bool(gate["passed"]), "no_champion_reason": None,
        "correct_mask": (test_pred[ch.cid] == y_te).tolist(),
        "decision_latency_us": {"median": float(np.median(lat_ms)) * 1e3, "p95": float(np.percentile(lat_ms, 95)) * 1e3,
                                "rows_timed": n_lat},
    })
    return rec


# ----------------------------------------------------------------------------- IO

def load_features(npz_path: str) -> dict:
    with np.load(npz_path, allow_pickle=False) as z:
        info = json.loads(str(z["info_json"]))
        return {"task": Path(npz_path).stem, "X_train": z["train_full"].astype(np.float32),
                "y_train": z["train_label"].astype(np.int64), "X_test": z["test_full"].astype(np.float32),
                "K": int(z["cands"].shape[0]), "variant": str(info.get("variant", "unknown"))}


def evaluate_benchmark_spec21(npz_path: str, ranks: Sequence[int], device: str, cfg: EvalConfig) -> dict:
    d = load_features(npz_path)
    task = d["task"]
    test_records = gd.load_test(task)
    if len(test_records) != len(d["X_test"]):
        raise ValueError(f"{task}: test rows {len(test_records)} != feature test rows {len(d['X_test'])}")
    cands_list = test_records[0]["candidates"]
    y_test = np.array([cands_list.index(r["ground_truth"]) for r in test_records], dtype=np.int64)
    rec = evaluate_task_arrays(task, d["X_train"], d["y_train"], d["X_test"], y_test, d["K"],
                               ranks=ranks, device=device, cfg=cfg)
    name, n_expected, nimble, jev = base.PNG[task]
    rec.update({"dataset": name, "n_expected_01png": n_expected, "nimble": nimble, "jev": jev,
                "source_variant": d["variant"], "test_file": gd.test_file_digest(task)})
    return rec


# ----------------------------------------------------------------------------- comparison + report

def compare_to_baseline(rows: Dict[str, dict], baseline: dict, n_boot: int = 10000) -> dict:
    per_task, new_corr, base_corr, skipped = {}, [], [], []
    for t, r in rows.items():
        b = baseline["tasks"].get(t)
        if b is None:
            continue
        if r["chosen_cid"] is None:
            skipped.append(t)
            continue
        per_task[t] = {"baseline_strategy": b["chosen_strategy"], "new_strategy": r["chosen_strategy"],
                       "strategy_changed": b["chosen_strategy"] != r["chosen_strategy"],
                       "baseline_acc": b["accuracy"], "new_acc": r["accuracy"],
                       "delta_acc": r["accuracy"] - b["accuracy"],
                       "baseline_wilson95": b["wilson95"], "new_wilson95": r["wilson95"]}
        new_corr.append(np.asarray(r["correct_mask"], dtype=bool))
        base_corr.append(np.arange(b["n"]) < b["correct"])          # marginal bootstrap needs only k of n
    if not per_task:
        return {"per_task": {}, "n_tasks_compared": 0, "tasks_skipped_no_champion": skipped}
    base_macro = float(np.mean([v["baseline_acc"] for v in per_task.values()]))
    new_macro = float(np.mean([v["new_acc"] for v in per_task.values()]))
    new_ci = bootstrap_macro_ci(new_corr, n_boot, 1)
    base_ci = bootstrap_macro_ci(base_corr, n_boot, 2)
    return {"per_task": per_task, "n_tasks_compared": len(per_task), "baseline_macro": base_macro,
            "new_macro": new_macro, "macro_delta": new_macro - base_macro,
            "baseline_macro_ci95": base_ci["ci95"], "new_macro_ci95": new_ci["ci95"],
            "n_changed": sum(v["strategy_changed"] for v in per_task.values()),
            "tasks_skipped_no_champion": skipped,
            "delta_ci_note": "the baseline report stores only correct/n per task, so its predictions cannot be "
                             "paired with ours; the two macro CIs are independent bootstraps and the delta has no "
                             "paired test. The paired test is full vs legacy3_tau0, re-run in this script."}


def _loadavg() -> Optional[List[float]]:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def build_report(rows: Dict[str, dict], baseline: Optional[dict], features_dir: str, cfg: EvalConfig,
                 ranks: Sequence[int], device: str, command: str) -> dict:
    tasks = list(rows)
    champ = [t for t in tasks if rows[t]["chosen_cid"] is not None]      # tasks with a gate-passing chosen expert
    admitted = [t for t in tasks if rows[t]["admitted"]]

    def mean_of(key: str, ts: Sequence[str]) -> Optional[float]:
        return float(np.mean([rows[t][key] for t in ts])) if ts else None

    def masks(ts: Sequence[str], arm: Optional[str] = None) -> List[np.ndarray]:
        return [np.asarray(rows[t]["correct_mask"] if arm is None else rows[t]["arms"][arm]["correct_mask"],
                           dtype=bool) for t in ts]

    arm_agg = {}
    for a in ARMS:
        ts = [t for t in tasks if rows[t]["arms"][a]["chosen_cid"] is not None]
        arm_agg[a] = {"n_tasks": len(ts),
                      "macro_acc": float(np.mean([rows[t]["arms"][a]["accuracy"] for t in ts])) if ts else None,
                      "macro_ci95": bootstrap_macro_ci(masks(ts, a), cfg.n_boot, 3)["ci95"] if ts else None,
                      "chosen": {t: rows[t]["arms"][a]["chosen_cid"] for t in tasks}}
    paired_ts = [t for t in tasks if rows[t]["arms"]["full"]["chosen_cid"] is not None
                 and rows[t]["arms"]["legacy3_tau0"]["chosen_cid"] is not None]
    paired = {"tasks": paired_ts,
              "full_vs_legacy3_tau0": paired_bootstrap_macro_delta(masks(paired_ts, "full"),
                                                                    masks(paired_ts, "legacy3_tau0"), cfg.n_boot, 4)
              if paired_ts else None,
              "mcnemar_per_task": {t: mcnemar_exact(np.asarray(rows[t]["arms"]["full"]["correct_mask"]),
                                                    np.asarray(rows[t]["arms"]["legacy3_tau0"]["correct_mask"]))
                                   for t in paired_ts}}

    def not_admitted(r: dict) -> dict:
        g = r["gate"]
        return {"no_admissible_candidate": r["chosen_cid"] is None,
                "oof_gate_reasons": r.get("oof_gate_reasons", []),
                "test_below_prior": None if g is None else g["below_prior"],
                "test_collapsed": None if g is None else g["collapsed"],
                "test_class_recall_below_min": None if g is None else g["class_recall_below_min"]}

    agg = {
        "n_tasks": len(tasks),
        "n_tasks_with_champion": len(champ),
        "tasks_without_champion": {t: rows[t]["oof_gate_reasons"] for t in tasks if rows[t]["chosen_cid"] is None},
        "macro_all": mean_of("accuracy", champ),
        "macro_all_ci95": bootstrap_macro_ci(masks(champ), cfg.n_boot, 0)["ci95"] if champ else None,
        "micro_acc": 100.0 * sum(rows[t]["correct"] for t in champ) / sum(rows[t]["n"] for t in champ)
        if champ else None,
        "macro_balanced_accuracy": mean_of("balanced_accuracy", champ),
        "macro_f1": mean_of("macro_f1", champ),
        "n_admitted": len(admitted),
        "macro_admitted": mean_of("accuracy", admitted),
        "tasks_not_admitted": {t: not_admitted(rows[t]) for t in tasks if not rows[t]["admitted"]},
        "tasks_selected_family": {f: sorted(t for t in tasks if rows[t]["chosen_family"] == f) for f in FAMILIES},
        "tasks_selected_with_logit_adjust": sorted(t for t in champ if rows[t]["chosen_tau"] > 0),
        "arms": arm_agg, "paired": paired,
    }
    variants = sorted({r.get("source_variant", "") for r in rows.values()})
    report = {
        "title": "Spec 21 grand scorecard: LW-LDA, BBP probe, logit adjustment vs linear / adapter / SupCon (1-SE)",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": command,
        "host": {"platform": platform.platform(), "cpu_count": os.cpu_count(), "loadavg_at_report": _loadavg(),
                 "device": device, "torch": torch.__version__},
        "protocol": {"features_dir": features_dir, "feature_variants": variants, "ranks": list(ranks),
                     "taus": list(TAUS), "cv_folds": cfg.cv_folds, "fold_seed": base.FOLD_SEED,
                     "config": cfg.__dict__, "candidates_per_task": len(FAMILIES) * len(TAUS),
                     "selection": "canonical Breiman 1-SE on pooled OOF accuracy over gate-passing candidates; "
                                  "simplest by Spec 21 S6 supervised parameter count, ties by fixed family order",
                     "gate": "OOF gate (selection): pooled OOF acc >= train majority prior, no class > 95% of "
                             f"predictions, and for K >= 3 every class recall >= {s21.MIN_CLASS_RECALL:g}. Only "
                             "gate-passing candidates can be chosen; with none the task has no chosen expert "
                             "(no fallback). Test gate (admission, post-selection): same test on the test split."},
        "aggregate": agg, "tasks": rows,
    }
    if baseline is not None:
        cmp_ = compare_to_baseline(rows, baseline, cfg.n_boot)
        b_var = (baseline.get("protocol", {}).get("feature_source") or {}).get("variant")
        cmp_["baseline_feature_variant"] = b_var
        cmp_["baseline_source_matches"] = bool(b_var and variants == [b_var])
        report["baseline_comparison"] = cmp_
    return report


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.2f}%"


def _ci(ci: Optional[Sequence[float]]) -> str:
    return "n/a" if ci is None else f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def _not_admitted_text(v: dict) -> str:
    parts = []
    if v["no_admissible_candidate"]:
        parts.append("no gate-passing CV candidate: " + "/".join(v["oof_gate_reasons"]))
    if v["test_below_prior"]:
        parts.append("test acc below train prior")
    if v["test_collapsed"]:
        parts.append("collapsed")
    if v["test_class_recall_below_min"]:
        parts.append("test class recall below min")
    return ", ".join(parts)


def render_md(report: dict) -> str:
    agg, tasks = report["aggregate"], report["tasks"]
    L: List[str] = [f"# {report['title']}", "", f"Generated: {report['generated_utc']}", "",
                    f"Command: `{report['command']}`", ""]
    L += ["## Headline", "",
          f"- Macro accuracy, chosen expert, {agg['n_tasks_with_champion']}/{agg['n_tasks']} tasks with a chosen "
          f"expert: **{_pct(agg['macro_all'])}** "
          f"(bootstrap 95% CI {_ci(agg['macro_all_ci95'])}, row-level within-task "
          f"noise only). Micro {_pct(agg['micro_acc'])}, macro balanced acc {_pct(agg['macro_balanced_accuracy'])}, "
          f"macro F1 {_pct(agg['macro_f1'])}."]
    if agg["tasks_without_champion"]:
        L.append("- NO chosen expert (no candidate passed the OOF gate, left out of every macro): " + "; ".join(
            f"`{t}` ({'/'.join(r)})" for t, r in agg["tasks_without_champion"].items()))
    if agg["macro_admitted"] is not None:
        L.append(f"- Admitted tasks only ({agg['n_admitted']}/{agg['n_tasks']}, chosen expert also passed the "
                 f"test-split prior gate and had a gate-passing CV pool): macro {_pct(agg['macro_admitted'])}.")
    bc = report.get("baseline_comparison")
    if bc and bc.get("n_tasks_compared"):
        L.append(f"- Baseline (Spec 19 Phase 4 report, macro over the same {bc['n_tasks_compared']} tasks): "
                 f"{_pct(bc['baseline_macro'])} (CI [{bc['baseline_macro_ci95'][0]:.2f}, {bc['baseline_macro_ci95'][1]:.2f}]). "
                 f"Delta: **{bc['macro_delta']:+.2f} pp**, {bc['n_changed']} task(s) changed winner. "
                 f"Baseline feature source matches this run: {bc['baseline_source_matches']} "
                 f"(baseline `{bc['baseline_feature_variant']}`).")
        L.append(f"- {bc['delta_ci_note']}")
    if agg["tasks_not_admitted"]:
        L.append("- Not admitted: " + "; ".join(
            f"`{t}` ({_not_admitted_text(v)})" for t, v in agg["tasks_not_admitted"].items()))
    L.append("")

    L += ["## Per-task results (chosen expert, test split)", "",
          "| Task | n | Chosen | tau | CV acc | Test acc | Wilson 95% | Bal.acc | Macro F1 | Train prior | Admitted |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t, r in tasks.items():
        if r["chosen_cid"] is None:
            L.append(f"| {t} | {r['n']} | NONE (no admissible candidate) | - | - | - | - | - | - | "
                     f"{_pct(r['majority_class_train_prior_acc'])} | False |")
            continue
        ci = r["wilson95"]
        L.append(f"| {t} | {r['n']} | {r['chosen_strategy']} | {r['chosen_tau']:g} | {_pct(r['cv_acc_chosen'])} | "
                 f"{_pct(r['accuracy'])} | [{ci[0]:.2f}, {ci[1]:.2f}] | {_pct(r['balanced_accuracy'])} | "
                 f"{_pct(r['macro_f1'])} | {_pct(r['majority_class_train_prior_acc'])} | {r['admitted']} |")
    L.append("")

    if bc and bc.get("per_task"):
        L += ["## Winner change vs the 76.06% baseline", "",
              "| Task | Baseline winner | New winner | Baseline acc [Wilson] | New acc [Wilson] | Delta pp |",
              "|---|---|---|---|---|---|"]
        for t, v in bc["per_task"].items():
            bw, nw = v["baseline_wilson95"], v["new_wilson95"]
            L.append(f"| {t} | {v['baseline_strategy']} | {v['new_strategy']}{' (changed)' if v['strategy_changed'] else ''} | "
                     f"{_pct(v['baseline_acc'])} [{bw[0]:.2f}, {bw[1]:.2f}] | {_pct(v['new_acc'])} [{nw[0]:.2f}, {nw[1]:.2f}] | "
                     f"{v['delta_acc']:+.2f} |")
        L.append("")

    L += ["## Ablation arms (same OOF pool, same test predictions, different candidate sets)", "",
          "| Arm | Tasks with a chosen expert | Macro acc | Bootstrap 95% CI |", "|---|---|---|---|"]
    for a, v in agg["arms"].items():
        L.append(f"| {a} | {v['n_tasks']}/{agg['n_tasks']} | {_pct(v['macro_acc'])} | {_ci(v['macro_ci95'])} |")
    p = agg["paired"]["full_vs_legacy3_tau0"]
    L += ["", "Paired bootstrap, full minus legacy3_tau0: no task where both arms have a chosen expert."
          if p is None else
          f"Paired bootstrap, full minus legacy3_tau0 (same test rows, {len(agg['paired']['tasks'])} tasks where "
          f"both arms have a chosen expert): {p['point']:+.2f} pp, "
          f"95% CI [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}].", "",
          "Per-task exact McNemar, full vs legacy3_tau0 (a = full only correct, b = legacy only correct):", "",
          "| Task | Full pick | Legacy pick | only full | only legacy | p (two-sided) |", "|---|---|---|---|---|---|"]
    for t, m in agg["paired"]["mcnemar_per_task"].items():
        L.append(f"| {t} | {tasks[t]['arms']['full']['chosen_cid']} | {tasks[t]['arms']['legacy3_tau0']['chosen_cid']} | "
                 f"{m['only_a']} | {m['only_b']} | {m['p_two_sided']:.3f} |")
    L.append("")

    L += ["## Logit adjustment on the linear probe (test split, DIAGNOSTIC, never used for selection)", "",
          "| Task | Train prior | tau=0 acc / bal / F1 | tau=0.5 acc / bal / F1 | tau=1 acc / bal / F1 |", "|---|---|---|---|---|"]
    for t, r in sorted(tasks.items(), key=lambda kv: -kv[1]["majority_class_train_prior_acc"]):
        cells = []
        for tau in TAUS:
            v = r["posthoc_test_all_candidates"].get(f"linear_probe|tau={float(tau)}")
            cells.append("n/a" if v is None else f"{v['accuracy']:.1f} / {v['balanced_accuracy']:.1f} / {v['macro_f1']:.1f}")
        L.append(f"| {t} | {r['majority_class_train_prior_acc']:.1f}% | " + " | ".join(cells) + " |")
    L.append("")

    L += ["## Selection ladder (OOF pool, best rank per family and tau)", "",
          "| Task | Best CV cand. | Threshold | Admissible | Chosen | Chosen params |", "|---|---|---|---|---|---|"]
    for t, r in tasks.items():
        s = r["selection"]
        if s["chosen_cid"] is None:
            L.append(f"| {t} | - | - | 0/{len(r['cv_candidates'])} (none passed: {'/'.join(r['oof_gate_reasons'])}) "
                     f"| NONE | - |")
            continue
        chosen = next(c for c in r["cv_candidates"] if c["cid"] == s["chosen_cid"])
        L.append(f"| {t} | {s['best_cid']} | {_pct(s['threshold'])} | {s['n_admissible']}/{len(r['cv_candidates'])}"
                 f" | {s['chosen_cid']} | {chosen['params']:,} |")
    L.append("")

    L += ["## Latency (chosen head, CPU NumPy, one row per call, includes Python overhead)", "",
          "| Task | Median us | p95 us | rows timed |", "|---|---|---|---|"]
    for t, r in tasks.items():
        lu = r["decision_latency_us"]
        if lu is None:
            L.append(f"| {t} | n/a | n/a | 0 |")
            continue
        L.append(f"| {t} | {lu['median']:.1f} | {lu['p95']:.1f} | {lu['rows_timed']} |")
    L.append("")

    errs = {t: r["fit_errors"] for t, r in tasks.items() if r["fit_errors"]}
    L += ["## Status", "", "Verified in this run:",
          "- Every candidate was fitted per fold and on the full train split; OOF and test predictions come from "
          "those fits (see `cv_candidates`, `posthoc_test_all_candidates` in the JSON).",
          "- The OOF prior gate, not test accuracy, decides which candidates may be selected; the test gate only "
          "decides admission of the already-chosen expert.",
          "- Permuting the test labels leaves the chosen candidate unchanged (unit test "
          "`test_selection_does_not_depend_on_test_labels`).", "",
          "NOT verified / limits:",
          "- CV is 5-fold OOF on one seed and NOT nested; 15 candidates per task share it, so CV accuracy is optimistic.",
          "- No leakage audit was re-run here: train/test overlap checks are inherited from feature extraction and "
          "the run does not re-derive them.",
          "- Baseline predictions are not available, so there is no paired test against the 76.06% report itself; "
          "the paired test is against the re-run `legacy3_tau0` arm, which uses the canonical rule, not the old ladder.",
          "- Macro CIs resample test rows inside each task; they do not include between-task or training-set variance.",
          f"- Latency figures come from whatever box ran this (loadavg {report['host']['loadavg_at_report']}).", "",
          "NOT done:",
          "- No conformal / abstain layer (Design E), no ETF head, no class-balanced SupCon sampler, no GD-2017 "
          "singular-value shrinkage (see `spec21_advanced_heads.py` scope notes)."]
    if errs:
        L += ["", "Fit errors (candidate dropped, not replaced): " + json.dumps(errs, ensure_ascii=False)]
    L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Spec 21 grand scorecard (13 tasks, 5 heads x logit adjustment, 1-SE)")
    ap.add_argument("--features-dir", default="D:/genz/features_uncap_v1/q9b_diff_compact/features")
    ap.add_argument("--results-dir", default=str(REPO / "benchmarks" / "results"))
    ap.add_argument("--baseline", default=str(REPO / "benchmarks" / "results" / "01png_sota_ensemble_report.json"))
    ap.add_argument("--out-stem", default="spec21_grand_scorecard_report")
    ap.add_argument("--ranks", default="32,64,128")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tasks", default="", help="comma list of task keys (default: all 13)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--class-weight", choices=("none", "balanced"), default="none",
                    help="loss weighting for bbp_probe and linear_probe (default none: unweighted loss)")
    args = ap.parse_args()

    ranks = [int(r) for r in args.ranks.split(",") if r.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] or list(base.PNG)
    bad = [t for t in tasks if t not in base.PNG]
    if bad:
        raise SystemExit(f"unknown task keys {bad}; valid: {sorted(base.PNG)}")
    cfg = EvalConfig(n_boot=args.n_boot, class_weight=None if args.class_weight == "none" else args.class_weight)
    baseline_path = Path(args.baseline)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else None
    print(f"device={args.device} features={args.features_dir} ranks={ranks} loadavg={_loadavg()} "
          f"baseline={'yes' if baseline else 'MISSING'}", flush=True)

    rows: Dict[str, dict] = {}
    for i, task in enumerate(tasks, 1):
        path = Path(args.features_dir) / f"{task}.npz"
        if not path.exists():
            raise SystemExit(f"missing feature file {path}")     # a partial 13-task board must not look complete
        t0 = time.perf_counter()
        rec = evaluate_benchmark_spec21(str(path), ranks, args.device, cfg)
        rows[task] = rec
        if rec["chosen_cid"] is None:
            print(f"[{i}/{len(tasks)}] {task}: NO chosen expert ({'/'.join(rec['oof_gate_reasons'])}) "
                  f"admitted=False [{time.perf_counter() - t0:.0f}s]", flush=True)
        else:
            print(f"[{i}/{len(tasks)}] {task}: {rec['chosen_strategy']} cv={rec['cv_acc_chosen']:.2f} "
                  f"test={rec['accuracy']:.2f} bal={rec['balanced_accuracy']:.2f} admitted={rec['admitted']} "
                  f"[{time.perf_counter() - t0:.0f}s]", flush=True)

    report = build_report(rows, baseline, args.features_dir, cfg, ranks, args.device, " ".join(sys.argv))
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.out_stem}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"{args.out_stem}.md").write_text(render_md(report), encoding="utf-8")
    print(f"macro_all={_pct(report['aggregate']['macro_all'])} "
          f"with_champion={report['aggregate']['n_tasks_with_champion']}/{len(rows)} "
          f"admitted={report['aggregate']['n_admitted']}/{len(rows)} "
          f"-> {out / args.out_stem}.[json|md]")


if __name__ == "__main__":
    main()
