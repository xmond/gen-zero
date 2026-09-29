#!/usr/bin/env python3
"""13-task dual-70B (Qwen-72B + Llama-70B) manifold-fusion evaluation with two selection tracks.

Pipeline per task, on the FULL 8192-d features (no random projection, all fits on training rows only):
  representations  qwen | llama | concat | geo(residual weight 0.35) | geo(1.0)   (geometric_latent_fusion)
  expert heads     bbp (Gavish-Donoho probe) | lda (Ledoit-Wolf LDA)              (spec21_advanced_heads)
  fusion           logit pooling of the qwen and llama heads, weight in {0.25, 0.5, 0.75}
  prior shift      raw | static logit adjustment | CALA gates entropy/margin/centroid  (confidence_adaptive_logit_adjustment)
  selection        5-fold stratified OOF on the training split, one 1-SE rule per track:
    Peak SOTA        objective = OOF accuracy,             gate = OOF accuracy >= train majority prior, no >95% collapse
    Certified Robust objective = OOF (balanced acc + macro F1)/2, gate = no >95% collapse and no class with zero recall
  conformal        the chosen config is refitted on 80% of the training rows and calibrated on the other
                   20% (split conformal, marginal AND class-conditional/Mondrian); test coverage is measured.

Test labels are read AFTER every selection is frozen and are used only for scoring. The selection
code path (`select_task`) has no label argument for the test split. Test metrics of non-selected
candidates (controls, Pareto front) are post-hoc diagnostics and never feed back into a choice.

Offload: `--remote HOST` rsyncs code, test data and feature files to HOST, bootstraps a private venv,
runs there (tasks in parallel processes), pulls the report back, verifies the feature hashes against
local files and deletes the remote work directory.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import hashlib
import json
import math
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SUITES = Path(__file__).resolve().parent
ROOT = SUITES.parents[1]
sys.path.insert(0, str(SUITES))

import confidence_adaptive_logit_adjustment as cala  # noqa: E402
import geometric_latent_fusion as glf  # noqa: E402
import spec21_advanced_heads as s21  # noqa: E402

TASKS = ("massive_en", "massive_de", "multinli", "pubmedqa", "vitaminc", "boolq", "squad2", "paws",
         "civil_comments", "aegis_safety", "helpsteer2", "summeval_relevance", "summeval_consistency")
SEED = 20260925
FOLDS = 5
REPS = ("qwen", "llama", "concat", "geo035", "geo100")
GEO_WEIGHT = {"geo035": 0.35, "geo100": 1.0}
HEADS = ("bbp", "lda")
FUSION_WEIGHTS = (0.25, 0.5, 0.75)
STATIC_TAUS = (0.5, 1.0, 2.0)
GATES = ("entropy", "margin", "centroid")
GATE_GAMMAS = (1.0, 2.0)
GATE_TAUS = (0.5, 1.0, 2.0)
ABLATION_REP = "geo100"
ABLATION_GATES = (cala.CalaConfig("static", 1.0, 1.0), cala.CalaConfig("entropy", 2.0, 1.0),
                  cala.CalaConfig("margin", 2.0, 1.0), cala.CalaConfig("centroid", 2.0, 1.0))
HEAVY_TASKS = ("massive_de", "boolq")                      # 11k / 9k training rows; the other tasks have <= 1000
CONF_ALPHA = 0.1
CAL_FRACTION = 0.2
COLLAPSE_FRAC = s21.COLLAPSE_FRAC
DEFAULT_QDIR = Path(os.environ.get("MASTER_QWEN_DIR", "/ebs/data/extracted_features/qwen72b/features"))
DEFAULT_LDIR = Path(os.environ.get("MASTER_LLAMA_DIR", "/ebs/data/extracted_features/llama70b"))
DEFAULT_OUT = ROOT / "benchmarks/results/spec21_manifold_pareto_ensemble_report"


# ------------------------------------------------------------------------------------------ data

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_pair(task: str, qdir: Path, ldir: Path):
    """Both views, with the id / label / finiteness checks the earlier suites made."""
    paths = (Path(qdir) / f"{task}.npz", Path(ldir) / f"{task}.npz")
    views = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            views.append({k: z[k] for k in ("train_full", "test_full", "train_label", "train_ids", "test_ids", "cands")})
    a, b = views
    for key in ("train_ids", "test_ids", "train_label"):
        if not np.array_equal(a[key], b[key]):
            raise ValueError(f"{task}: the two views disagree on {key}")
    if len(set(a["train_ids"])) != len(a["train_ids"]) or set(a["train_ids"]) & set(a["test_ids"]):
        raise ValueError(f"{task}: duplicate or overlapping row ids")
    for v in views:
        if not (np.isfinite(v["train_full"]).all() and np.isfinite(v["test_full"]).all()):
            raise ValueError(f"{task}: non-finite feature")
    return paths, a, b


def load_gold(task: str, test_ids: Sequence[str], n_classes: int) -> np.ndarray:
    """Test labels. Called only after selection is frozen."""
    import grand_challenge_data as gd
    records = gd.load_test(task)
    if [r["id"] for r in records] != list(test_ids):
        raise ValueError(f"{task}: test ids differ from the feature files")
    cands = records[0]["candidates"]
    if len(cands) != n_classes or any(r["candidates"] != cands for r in records):
        raise ValueError(f"{task}: candidate lists differ")
    return np.array([cands.index(r["ground_truth"]) for r in records])


# --------------------------------------------------------------------------------------- metrics

def metrics_from_pred(y: np.ndarray, pred: np.ndarray, k: int) -> Dict[str, float]:
    """acc / balanced acc / macro-F1 (in %), plus collapse diagnostics. `pred == -1` counts as wrong."""
    y, pred = np.asarray(y), np.asarray(pred)
    n = len(y)
    conf = np.bincount(y * k + np.where(pred >= 0, pred, 0), minlength=k * k).reshape(k, k).astype(float)
    wrong_abstain = pred < 0
    if wrong_abstain.any():                                    # abstentions are neither TP nor FP
        conf -= np.bincount(y[wrong_abstain] * k, minlength=k * k).reshape(k, k)
    tp = np.diag(conf)
    support = np.bincount(y, minlength=k).astype(float)
    pred_count = conf.sum(0)
    present = support > 0
    recall = np.divide(tp, support, out=np.zeros(k), where=present)
    prec = np.divide(tp, pred_count, out=np.zeros(k), where=pred_count > 0)
    f1 = np.divide(2 * prec * recall, prec + recall, out=np.zeros(k), where=(prec + recall) > 0)
    labels = present | (pred_count > 0)
    valid_pred = pred[pred >= 0]
    return {"accuracy": 100 * float((pred == y).mean()),
            "balanced_accuracy": 100 * float(recall[present].mean()),
            "macro_f1": 100 * float(f1[labels].mean()),
            "min_class_recall": 100 * float(recall[present].min()),
            "zero_recall_classes": int((recall[present] == 0).sum()),
            "max_pred_class_frac": float(np.bincount(valid_pred, minlength=k).max() / n) if valid_pred.size else 0.0,
            "n": int(n)}


def robust_objective(m: Dict[str, float]) -> float:
    return 0.5 * (m["balanced_accuracy"] + m["macro_f1"])


# ------------------------------------------------------------------------------------ candidates

@dataclass(frozen=True)
class Cand:
    kind: str                 # 'single' | 'fusion'
    head: str
    rep: str = ""             # single only
    w: float = 0.0            # fusion: weight of the qwen head
    gate: cala.CalaConfig = cala.CalaConfig()

    @property
    def key(self) -> str:
        base = f"{self.rep}+{self.head}" if self.kind == "single" else f"fuse{self.w:g}+{self.head}"
        return f"{base}|{self.gate.key}"

    @property
    def complexity(self) -> Tuple:
        shift = 0 if self.gate.tau == 0 else (1 if self.gate.mode == "static" else 2)
        family = 3 if self.kind == "fusion" else {"qwen": 0, "llama": 0, "concat": 1}.get(self.rep, 2)
        return (shift, family, HEADS.index(self.head), self.key)


def gate_grid(with_centroid: bool) -> List[cala.CalaConfig]:
    grid = [cala.CalaConfig()]
    grid += [cala.CalaConfig("static", 1.0, t) for t in STATIC_TAUS]
    for mode in GATES:
        if mode == "centroid" and not with_centroid:
            continue
        grid += [cala.CalaConfig(mode, g, t) for g in GATE_GAMMAS for t in GATE_TAUS]
    return grid


def enumerate_candidates(pool: str = "all") -> List[Cand]:
    """pool 'all' = every representation and fusion; 'single' = the two single-model reps only."""
    out = []
    reps = REPS if pool == "all" else ("qwen", "llama")
    for rep in reps:
        for head in HEADS:
            out += [Cand("single", head, rep=rep, gate=g) for g in gate_grid(True)]
    if pool == "all":
        for head in HEADS:
            for w in FUSION_WEIGHTS:
                out += [Cand("fusion", head, w=w, gate=g) for g in gate_grid(False)]
    return out


def candidate_logits(c: Cand, raw: Dict[Tuple[str, str], np.ndarray], ratio: Dict[str, np.ndarray],
                     scales: Dict, priors: np.ndarray) -> np.ndarray:
    """Normalised, prior-shifted logits of one candidate for one block of rows."""
    if c.kind == "single":
        z = raw[(c.rep, c.head)] / scales[(c.rep, c.head)]
        r = ratio.get(c.rep)
    else:
        z = (c.w * raw[("qwen", c.head)] / scales[("qwen", c.head)]
             + (1 - c.w) * raw[("llama", c.head)] / scales[("llama", c.head)]) / scales[("fuse", c.head, c.w)]
        r = None
    return c.gate.apply(z, priors, 1.0, centroid_ratio=r)


# ----------------------------------------------------------------------------------------- engine

def _standardizer(x: np.ndarray):
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    return mu, np.where(sd > 1e-12, sd, 1.0)


def _centroid_ratio(f_tr: np.ndarray, y: np.ndarray, k: int, evals: List[np.ndarray], standardize: bool):
    mu, sd = _standardizer(f_tr) if standardize else (np.zeros(f_tr.shape[1]), np.ones(f_tr.shape[1]))
    z = (f_tr - mu) / sd
    means = np.stack([z[y == c].mean(axis=0) for c in range(k)])
    return [cala.gate_from_centroids((f - mu) / sd, means, 1.0) for f in evals]


def engine(xq: np.ndarray, xl: np.ndarray, y: np.ndarray, k: int, seed: int,
           evals: List[Tuple[np.ndarray, np.ndarray]]):
    """Fit every (representation, head) on the training rows and score each eval block.

    Returns (raw, ratio, info): raw[(rep, head)] -> list of (n_eval x K) score arrays,
    ratio[rep] -> list of centroid-gate ratios, info -> ranks / chosen C (diagnostics).
    A fit that raises is recorded in info['failures'] and its key is absent from raw."""
    xq = xq.astype(np.float64)
    xl = xl.astype(np.float64)
    ev = [(a.astype(np.float64), b.astype(np.float64)) for a, b in evals]
    fus = glf.GeometricLatentFusion.fit(xq, xl)
    info = {"failures": [], "gd_rank_q": fus.view_q.rank, "gd_rank_l": fus.view_l.rank,
            "core_dim": fus.core_dim, "geo_dim": fus.output_dim}

    def representations():
        """One representation at a time (concat is 16384-d; do not hold all five in memory)."""
        yield "qwen", xq, [e[0] for e in ev], True
        yield "llama", xl, [e[1] for e in ev], True
        yield "concat", np.hstack([xq, xl]), [np.hstack(e) for e in ev], True
        for name, wgt in GEO_WEIGHT.items():
            yield name, fus.transform(xq, xl, wgt), [fus.transform(a, b, wgt) for a, b in ev], False

    raw = {}
    ratio = {}
    for rep, f_tr, f_ev, standardize in representations():
        ratio[rep] = _centroid_ratio(f_tr, y, k, f_ev, standardize)
        for head in HEADS:
            try:
                if head == "bbp":
                    model = s21.BBPAdaptiveProbe.fit(f_tr, y, k, standardize=standardize, seed=seed)
                    info[f"bbp_rank_{rep}"], info[f"bbp_C_{rep}"] = int(model.rank), float(model.C_)
                else:
                    model = s21.LedoitWolfLDAHead.fit(f_tr, y, k, standardize=standardize)
                    info[f"lda_rho_{rep}"] = float(model.rho)
                raw[(rep, head)] = [model.scores(f) for f in f_ev]
            except Exception as exc:                          # recorded, never masked: the key is simply absent
                info["failures"].append(f"{rep}+{head}:{type(exc).__name__}:{exc}")
    return raw, ratio, info


# ------------------------------------------------------------------------------------ selection

def _fold_metric_values(y, pred, fold_ids, k, objective):
    vals = []
    for f in range(FOLDS):
        idx = fold_ids == f
        m = metrics_from_pred(y[idx], pred[idx], k)
        vals.append(m["accuracy"] if objective == "peak" else robust_objective(m))
    return np.array(vals)


def score_candidates(cands, oof_raw, oof_ratio, scales, y, folds, priors_by_fold, k):
    """OOF metrics of every candidate; folds is a list of (train_idx, holdout_idx)."""
    fold_ids = np.empty(len(y), dtype=int)
    for f, (_, ho) in enumerate(folds):
        fold_ids[ho] = f
    majority = float(np.bincount(y, minlength=k).max() / len(y))
    raw_f = [{key: v[ho] for key, v in oof_raw.items()} for _, ho in folds]
    rat_f = [{key: v[ho] for key, v in oof_ratio.items()} for _, ho in folds]
    rows = []
    for c in cands:
        logits = np.empty((len(y), k))
        for f, (_, ho) in enumerate(folds):
            logits[ho] = candidate_logits(c, raw_f[f], rat_f[f], scales, priors_by_fold[f])
        pred = logits.argmax(1)
        m = metrics_from_pred(y, pred, k)
        rows.append({"cand": c, "metrics": m,
                     "peak_folds": _fold_metric_values(y, pred, fold_ids, k, "peak"),
                     "robust_folds": _fold_metric_values(y, pred, fold_ids, k, "robust"),
                     "gate_peak": bool(m["accuracy"] >= 100 * majority and m["max_pred_class_frac"] <= COLLAPSE_FRAC),
                     "gate_robust": bool(m["max_pred_class_frac"] <= COLLAPSE_FRAC and m["zero_recall_classes"] == 0)})
    return rows


def select_track(rows, track: str):
    """Breiman 1-SE rule on the track objective over gate-passing candidates; simplest wins.
    Returns (row, admitted). If nothing passes the gate the best raw objective is returned with
    admitted=False so the failure is visible, never hidden."""
    folds_key, gate_key = ("peak_folds", "gate_peak") if track == "peak" else ("robust_folds", "gate_robust")
    admitted = True
    pool = [r for r in rows if r[gate_key]]
    if not pool:
        pool, admitted = list(rows), False
    best = min(pool, key=lambda r: (-r[folds_key].mean(), r["cand"].complexity))
    se = float(best[folds_key].std(ddof=1) / math.sqrt(len(best[folds_key])))
    eligible = [r for r in pool if r[folds_key].mean() >= best[folds_key].mean() - se]
    chosen = min(eligible, key=lambda r: r["cand"].complexity)
    return chosen, admitted, se


def pareto_front(rows) -> List[dict]:
    """Non-dominated candidates in (OOF accuracy, OOF robust objective), collapse-free only."""
    pts = [(r["metrics"]["accuracy"], robust_objective(r["metrics"]), r) for r in rows
           if r["metrics"]["max_pred_class_frac"] <= COLLAPSE_FRAC]
    front = []
    for a, b, r in pts:
        if not any((a2 >= a and b2 >= b) and (a2 > a or b2 > b) for a2, b2, _ in pts):
            front.append((a, b, r))
    front.sort(key=lambda t: (-t[0], -t[1], t[2]["cand"].complexity))
    seen, out = set(), []
    for a, b, r in front:
        if (round(a, 9), round(b, 9)) not in seen:            # identical objective pairs: keep the simplest
            seen.add((round(a, 9), round(b, 9)))
            out.append(r)
    return out


def fit_scales(oof_raw: Dict) -> Dict:
    """Label-free logit scales: standard deviation of the pooled OOF scores (fusion: of the pooled logits)."""
    scales = {key: max(float(v.std()), 1e-8) for key, v in oof_raw.items()}
    for head in HEADS:
        if ("qwen", head) in oof_raw and ("llama", head) in oof_raw:
            for w in FUSION_WEIGHTS:
                z = (w * oof_raw[("qwen", head)] / scales[("qwen", head)]
                     + (1 - w) * oof_raw[("llama", head)] / scales[("llama", head)])
                scales[("fuse", head, w)] = max(float(z.std()), 1e-8)
    return scales


def select_task(xq, xl, y, k, seed=SEED):
    """Everything up to and including frozen selection. Takes NO test labels.
    Returns a dict holding the frozen choices plus the OOF bank needed for post-hoc diagnostics."""
    from sklearn.model_selection import StratifiedKFold
    if np.bincount(y, minlength=k).min() < FOLDS:
        raise ValueError("a class has fewer rows than folds")
    folds = list(StratifiedKFold(FOLDS, shuffle=True, random_state=seed).split(xq, y))
    oof_raw: Dict = {}
    oof_ratio: Dict = {}
    infos = []
    for f, (tr, ho) in enumerate(folds):
        raw, ratio, info = engine(xq[tr], xl[tr], y[tr], k, seed + f, [(xq[ho], xl[ho])])
        infos.append(info)
        for key, v in raw.items():
            oof_raw.setdefault(key, {})[f] = v[0]
        for key, v in ratio.items():
            oof_ratio.setdefault(key, {})[f] = v[0]
    complete = {key for key, d in oof_raw.items() if len(d) == FOLDS}
    dropped = sorted(f"{a}+{b}" for (a, b) in set(oof_raw) - complete)
    bank = {}
    for key in complete:
        arr = np.empty((len(y), k))
        for f, (_, ho) in enumerate(folds):
            arr[ho] = oof_raw[key][f]
        bank[key] = arr
    rbank = {}
    for rep in oof_ratio:
        arr = np.empty(len(y))
        for f, (_, ho) in enumerate(folds):
            arr[ho] = oof_ratio[rep][f]
        rbank[rep] = arr
    scales = fit_scales(bank)
    priors_by_fold = [s21.compute_class_priors(y[tr], k) for tr, _ in folds]
    result = {"folds": folds, "oof_raw": bank, "oof_ratio": rbank, "scales": scales,
              "priors_by_fold": priors_by_fold, "fold_infos": infos, "dropped_bases": dropped}
    for pool in ("all", "single"):
        cands = [c for c in enumerate_candidates(pool)
                 if ((c.rep, c.head) in bank if c.kind == "single"
                     else ("qwen", c.head) in bank and ("llama", c.head) in bank)]
        rows = score_candidates(cands, bank, rbank, scales, y, folds, priors_by_fold, k)
        result[f"rows_{pool}"] = rows
        for track in ("peak", "robust"):
            chosen, admitted, se = select_track(rows, track)
            result[f"{pool}_{track}"] = {"row": chosen, "admitted": admitted, "se": se}
    result["pareto"] = pareto_front(result["rows_all"])
    return result


# ---------------------------------------------------------------------------------- conformal

def mondrian_conformal(cal_logits, cal_labels, test_logits, test_labels, alpha=CONF_ALPHA) -> Dict:
    """Class-conditional split conformal with the LAC score 1 - softmax[y]. Class c's threshold is the
    ceil((n_c + 1)(1 - alpha))-th smallest calibration score of class c; with too few rows the threshold
    is +inf (full set for that class) and the class is listed in `trivial_classes` instead of a fake guarantee."""
    cal_logits, test_logits = np.asarray(cal_logits, float), np.asarray(test_logits, float)
    cal_labels, test_labels = np.asarray(cal_labels), np.asarray(test_labels)
    k = cal_logits.shape[1]
    p_cal = s21._softmax(cal_logits)
    p_test = s21._softmax(test_logits)
    qhat, trivial = np.empty(k), []
    for c in range(k):
        s = np.sort(1.0 - p_cal[cal_labels == c, c])
        need = math.ceil(round((len(s) + 1) * (1 - alpha), 9))
        if len(s) == 0 or need > len(s):
            qhat[c] = math.inf
            trivial.append(c)
        else:
            qhat[c] = s[max(need, 1) - 1]
    sets = (1.0 - p_test) <= qhat[None, :] + 1e-12
    size = sets.sum(1)
    covered = sets[np.arange(len(test_labels)), test_labels]
    single = size == 1
    return {"alpha": alpha, "marginal_coverage": float(covered.mean()),
            "class_coverage": {str(c): float(covered[test_labels == c].mean()) for c in range(k) if (test_labels == c).any()},
            "class_support": {str(c): int((test_labels == c).sum()) for c in range(k) if (test_labels == c).any()},
            "mean_set_size": float(size.mean()), "singleton_rate": float(single.mean()),
            "empty_rate": float((size == 0).mean()),
            "singleton_accuracy": float((sets.argmax(1)[single] == test_labels[single]).mean()) if single.any() else None,
            "trivial_classes": trivial, "n_cal": int(len(cal_labels)), "n_test": int(len(test_labels))}


def marginal_conformal(cal_logits, cal_labels, test_logits, test_labels, alpha=CONF_ALPHA) -> Dict:
    cp = s21.SplitConformalPredictor().calibrate(cal_logits, cal_labels, alpha)
    sets = cp.predict_set(test_logits)
    covered = np.array([int(t) in s for s, t in zip(sets, test_labels)])
    size = np.array([len(s) for s in sets])
    return {"alpha": alpha, "coverage": float(covered.mean()), "mean_set_size": float(size.mean()),
            "singleton_rate": float((size == 1).mean()), "trivial": bool(cp.is_trivial), "q_hat": float(cp.q_hat)}


def conformal_for(cands: Dict[str, Cand], xq, xl, y, k, xq_te, xl_te, gold, scales, seed):
    """Refit on 80% of training rows, calibrate on 20%, evaluate every named config on the test split."""
    from sklearn.model_selection import StratifiedShuffleSplit
    tr, cal = next(StratifiedShuffleSplit(1, test_size=CAL_FRACTION, random_state=seed).split(xq, y))
    raw, ratio, _ = engine(xq[tr], xl[tr], y[tr], k, seed, [(xq[cal], xl[cal]), (xq_te, xl_te)])
    priors = s21.compute_class_priors(y[tr], k)
    out = {}
    for name, c in cands.items():
        need = [(c.rep, c.head)] if c.kind == "single" else [("qwen", c.head), ("llama", c.head)]
        if any(n not in raw for n in need):
            out[name] = {"error": "a required base fit failed on the calibration split"}
            continue
        logits = [candidate_logits(c, {kk: v[i] for kk, v in raw.items()}, {kk: v[i] for kk, v in ratio.items()}, scales, priors)
                  for i in (0, 1)]
        out[name] = {"config": c.key,
                     "mondrian": mondrian_conformal(logits[0], y[cal], logits[1], gold, CONF_ALPHA),
                     "marginal": marginal_conformal(logits[0], y[cal], logits[1], gold, CONF_ALPHA)}
    return out


# ---------------------------------------------------------------------------------------- task

def _cand_summary(row, admitted, se, oof_obj):
    c = row["cand"]
    return {"key": c.key, "kind": c.kind, "rep": c.rep, "head": c.head, "qwen_weight": c.w if c.kind == "fusion" else None,
            "gate": {"mode": c.gate.mode, "gamma": c.gate.gamma, "tau": c.gate.tau},
            "oof": row["metrics"], "oof_objective": oof_obj, "one_se": se, "admitted_by_gate": admitted}


def _test_block(c: Cand, tst_raw, tst_ratio, scales, prior, gold, k):
    logits = candidate_logits(c, tst_raw, tst_ratio, scales, prior)
    pred = logits.argmax(1)
    return metrics_from_pred(gold, pred, k), pred, logits


def _bits(pred, gold) -> str:
    return "".join("1" if a == b else "0" for a, b in zip(pred, gold))


def run_task(task: str, qdir: str, ldir: str, threads: int) -> Dict:
    from threadpoolctl import threadpool_limits
    t0 = time.time()
    with threadpool_limits(limits=threads):
        paths, a, b = load_pair(task, Path(qdir), Path(ldir))
        y = a["train_label"].astype(int)
        k = len(a["cands"])
        xq, xl, xq_te, xl_te = a["train_full"], b["train_full"], a["test_full"], b["test_full"]
        sel = select_task(xq, xl, y, k)
        t_sel = time.time() - t0
        prior_full = s21.compute_class_priors(y, k)
        raw_te, ratio_te, info_te = engine(xq, xl, y, k, SEED, [(xq_te, xl_te)])
        tst_raw = {key: v[0] for key, v in raw_te.items()}
        tst_ratio = {key: v[0] for key, v in ratio_te.items()}
        # ---- everything above is frozen; test labels enter here and only here ----
        gold = load_gold(task, a["test_ids"].tolist(), k)
        frozen = {}
        for pool in ("all", "single"):
            for track in ("peak", "robust"):
                frozen[f"{pool}_{track}"] = sel[f"{pool}_{track}"]
        picks = {"peak": frozen["all_peak"]["row"]["cand"], "robust": frozen["all_robust"]["row"]["cand"],
                 "peak_single": frozen["single_peak"]["row"]["cand"], "robust_single": frozen["single_robust"]["row"]["cand"]}
        res: Dict = {"task": task, "n_train": int(len(y)), "n_test": int(len(gold)), "classes": k,
                     "train_class_counts": np.bincount(y, minlength=k).tolist(),
                     "majority_prior_pct": 100 * float(np.bincount(y).max() / len(y)),
                     "dropped_bases": sel["dropped_bases"], "test_fit_failures": info_te["failures"],
                     "fold_failures": [i["failures"] for i in sel["fold_infos"]],
                     "ranks": {kk: v for kk, v in info_te.items() if kk != "failures"}}
        correct = {}
        for name, cand in picks.items():
            track = "peak" if name.startswith("peak") else "robust"
            pool = "single" if name.endswith("single") else "all"
            fr = frozen[f"{pool}_{track}"]
            obj = fr["row"]["metrics"]["accuracy"] if track == "peak" else robust_objective(fr["row"]["metrics"])
            m, pred, _ = _test_block(cand, tst_raw, tst_ratio, sel["scales"], prior_full, gold, k)
            res[name] = {**_cand_summary(fr["row"], fr["admitted"], fr["se"], obj), "test": m,
                         "oof_minus_test_objective": obj - (m["accuracy"] if track == "peak" else robust_objective(m))}
            correct[name] = _bits(pred, gold)
        # ---- fixed, unselected controls (no candidate search at all) ----
        controls = {}
        for rep in REPS:
            for head in HEADS:
                if (rep, head) in tst_raw and (rep, head) in sel["oof_raw"]:
                    c = Cand("single", head, rep=rep)
                    m, pred, _ = _test_block(c, tst_raw, tst_ratio, sel["scales"], prior_full, gold, k)
                    row = next(r for r in sel["rows_all"] if r["cand"] == c)
                    controls[c.key] = {"test": m, "oof": row["metrics"]}
                    if head == "bbp" and rep in ("qwen", "llama", "concat", "geo100"):
                        correct[f"ctl_{rep}_bbp"] = _bits(pred, gold)
        # CALA ablation on fixed bases: does the gate spare accuracy that the static shift costs?
        for head in HEADS:
            for gate in ABLATION_GATES:
                c = Cand("single", head, rep=ABLATION_REP, gate=gate)
                if (ABLATION_REP, head) in tst_raw:
                    m, _, _ = _test_block(c, tst_raw, tst_ratio, sel["scales"], prior_full, gold, k)
                    controls[c.key] = {"test": m, "oof": next(r for r in sel["rows_all"] if r["cand"] == c)["metrics"]}
        res["controls"] = controls
        # ---- Pareto front (post-hoc test metrics for reference only) ----
        front = []
        for r in sel["pareto"]:
            m, _, _ = _test_block(r["cand"], tst_raw, tst_ratio, sel["scales"], prior_full, gold, k)
            front.append({"key": r["cand"].key, "oof_accuracy": r["metrics"]["accuracy"],
                          "oof_robust_objective": robust_objective(r["metrics"]), "test": m})
        res["pareto_front"] = front
        res["n_candidates"] = {"all": len(sel["rows_all"]), "single": len(sel["rows_single"])}
        res["correct_bits"] = correct
        res["conformal"] = conformal_for({"peak": picks["peak"], "robust": picks["robust"]}, xq, xl, y, k,
                                         xq_te, xl_te, gold, sel["scales"], SEED + 77)
        res["feature_sha256"] = {str(p): digest(p) for p in paths}
        import grand_challenge_data as gd
        res["test_sha256"] = digest(gd.TEST_DIR / f"{task}.jsonl")
        res["seconds"] = {"selection": round(t_sel, 1), "total": round(time.time() - t0, 1)}
    return res


# -------------------------------------------------------------------------------------- report

def paired_macro_bootstrap(bits_a: Dict[str, str], bits_b: Dict[str, str], n_boot=5000, seed=SEED) -> Dict:
    """Macro-accuracy difference (a - b, pp) with rows resampled independently inside each task."""
    tasks = [t for t in bits_a if t in bits_b]
    da = [np.frombuffer(bits_a[t].encode(), dtype=np.uint8) - 48 for t in tasks]
    db = [np.frombuffer(bits_b[t].encode(), dtype=np.uint8) - 48 for t in tasks]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        acc = 0.0
        for a, b in zip(da, db):
            idx = rng.integers(0, len(a), len(a))
            acc += float(a[idx].mean() - b[idx].mean())
        diffs[i] = 100 * acc / len(tasks)
    point = 100 * float(np.mean([a.mean() - b.mean() for a, b in zip(da, db)]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"delta_pp": point, "ci95": [float(lo), float(hi)], "n_boot": n_boot, "tasks": len(tasks)}


def macro(rows: Dict[str, Dict], get) -> Dict[str, float]:
    return {m: float(np.mean([get(r)[m] for r in rows.values()])) for m in ("accuracy", "balanced_accuracy", "macro_f1")}


def aggregate(rows: Dict[str, Dict]) -> Dict:
    agg = {name: macro(rows, lambda r, n=name: r[n]["test"]) for name in ("peak", "robust", "peak_single", "robust_single")}
    ctl_names = sorted({n for r in rows.values() for n in r["controls"]})
    agg["controls"] = {n: macro({t: r for t, r in rows.items() if n in r["controls"]}, lambda r, n=n: r["controls"][n]["test"])
                       for n in ctl_names if all(n in r["controls"] for r in rows.values())}
    boots = {}
    for a, b in (("peak", "peak_single"), ("peak", "ctl_qwen_bbp"), ("peak", "ctl_llama_bbp"),
                 ("robust", "robust_single"), ("peak", "ctl_concat_bbp"), ("ctl_geo100_bbp", "ctl_concat_bbp")):
        if all(a in r["correct_bits"] and b in r["correct_bits"] for r in rows.values()):
            boots[f"{a}_minus_{b}"] = paired_macro_bootstrap({t: r["correct_bits"][a] for t, r in rows.items()},
                                                             {t: r["correct_bits"][b] for t, r in rows.items()})
    agg["paired_bootstrap_accuracy"] = boots
    return agg


def references() -> Dict:
    out = {}
    for n in ("qwen72b", "llama70b"):
        p = ROOT / f"benchmarks/results/spec21_{n}_scorecard.json"
        out[n] = json.loads(p.read_text())["aggregate"]["macro_all"] if p.exists() else None
    p = ROOT / "benchmarks/results/spec21_dual_70b_72b_advanced_ensemble_report.json"
    out["prior_256d_dual_selected"] = json.loads(p.read_text())["aggregate_macro"]["selected"]["accuracy"] if p.exists() else None
    return out


def build_report(rows: Dict[str, Dict], args_line: str, host: Dict, ref: Dict) -> Dict:
    agg = aggregate(rows)
    pk, rb = agg["peak"], agg["robust"]
    conf = {t: r["conformal"] for t, r in rows.items()}
    ok_prior = {trk: {t: v[trk] for t, v in conf.items() if "error" not in v[trk]} for trk in ("peak", "robust")}
    return {"title": "Manifold-Pareto dual 70B/72B ensemble: geometric fusion + CALA gate + conformal control",
            "generated_utc": datetime.now(timezone.utc).isoformat(), "command": args_line, "host": host,
            "protocol": {"folds": FOLDS, "seed": SEED, "representations": REPS, "heads": HEADS,
                         "fusion_weights_qwen": FUSION_WEIGHTS, "static_taus": STATIC_TAUS,
                         "gates": {"modes": GATES, "gammas": GATE_GAMMAS, "taus": GATE_TAUS},
                         "cala_sign": "logit - tau * Phi * log(prior); the brief's '+' would boost the majority class",
                         "selection": "1-SE rule on 5-fold stratified OOF of the training split; simplest candidate within 1 SE",
                         "peak_track": "objective OOF accuracy; gate: OOF accuracy >= train majority prior and no class > 95% of predictions",
                         "robust_track": "objective OOF (balanced accuracy + macro F1)/2; gate: no class > 95% of predictions and no zero-recall class",
                         "conformal": f"refit on {100 * (1 - CAL_FRACTION):.0f}% of train rows, calibrate on {100 * CAL_FRACTION:.0f}%, alpha={CONF_ALPHA}; marginal and class-conditional",
                         "test_labels": "read after every selection and every test-score computation; scoring only",
                         "features": "full 8192-d per view, fits on training rows only, no random projection"},
            "references_macro_accuracy": ref,
            "aggregate_macro": agg,
            "targets": {"peak_macro_accuracy_ge_82_5": pk["accuracy"] >= 82.5,
                        "peak_beats_published_qwen72b": (pk["accuracy"] > ref["qwen72b"]) if ref["qwen72b"] else None,
                        "robust_macro_balanced_accuracy_ge_76": rb["balanced_accuracy"] >= 76.0,
                        "robust_macro_f1_ge_76": rb["macro_f1"] >= 76.0,
                        "robust_tasks_with_collapse_or_zero_recall": [t for t, r in rows.items()
                                                                     if r["robust"]["test"]["zero_recall_classes"] > 0
                                                                     or r["robust"]["test"]["max_pred_class_frac"] > COLLAPSE_FRAC],
                        "robust_tasks_not_admitted_by_oof_gate": [t for t, r in rows.items() if not r["robust"]["admitted_by_gate"]],
                        "peak_tasks_not_admitted_by_oof_gate": [t for t, r in rows.items() if not r["peak"]["admitted_by_gate"]]},
            "conformal_summary": {trk: {"mean_marginal_coverage": float(np.mean([v["mondrian"]["marginal_coverage"] for v in ok_prior[trk].values()])) if ok_prior[trk] else None,
                                        "mean_set_size": float(np.mean([v["mondrian"]["mean_set_size"] for v in ok_prior[trk].values()])) if ok_prior[trk] else None,
                                        "tasks_with_trivial_class": [t for t, v in ok_prior[trk].items() if v["mondrian"]["trivial_classes"]]}
                                  for trk in ("peak", "robust")},
            "tasks": rows}


def render_md(rep: Dict) -> str:
    agg, ref, tgt = rep["aggregate_macro"], rep["references_macro_accuracy"], rep["targets"]
    f = lambda v: "n/a" if v is None else f"{v:.2f}"
    L = [f"# {rep['title']}", "", f"Generated {rep['generated_utc']} on `{rep['host'].get('name')}` "
         f"(cores {rep['host'].get('cores')}, loadavg at start {rep['host'].get('loadavg')}).", "",
         f"Command: `{rep['command']}`", "", f"## Headline (test split, macro over {len(rep['tasks'])} tasks, %)", "",
         "| Arm | Accuracy | Balanced acc | Macro F1 |", "|---|---:|---:|---:|"]
    labels = {"peak": "Peak SOTA track (selected on OOF accuracy)", "peak_single": "  matched control: same search, single models only",
              "robust": "Certified Robust track (selected on OOF BA+F1)", "robust_single": "  matched control: same search, single models only"}
    for k, lab in labels.items():
        L.append(f"| {lab} | {agg[k]['accuracy']:.2f} | {agg[k]['balanced_accuracy']:.2f} | {agg[k]['macro_f1']:.2f} |")
    for n, v in agg["controls"].items():
        if n.endswith("|raw"):
            L.append(f"| fixed control `{n[:-4]}` (no search) | {v['accuracy']:.2f} | {v['balanced_accuracy']:.2f} | {v['macro_f1']:.2f} |")
    L += ["", f"External references (different protocol, not matched): published Qwen-72B {f(ref['qwen72b'])}, Llama-70B {f(ref['llama70b'])}, "
          f"earlier 256-d dual ensemble {f(ref['prior_256d_dual_selected'])}.", "", "## Target check", ""]
    L += [f"- Peak macro accuracy >= 82.5: **{tgt['peak_macro_accuracy_ge_82_5']}** ({agg['peak']['accuracy']:.2f})",
          f"- Peak beats published Qwen-72B scorecard: **{tgt['peak_beats_published_qwen72b']}**",
          f"- Robust macro balanced accuracy >= 76: **{tgt['robust_macro_balanced_accuracy_ge_76']}** ({agg['robust']['balanced_accuracy']:.2f})",
          f"- Robust macro F1 >= 76: **{tgt['robust_macro_f1_ge_76']}** ({agg['robust']['macro_f1']:.2f})",
          f"- Robust tasks with a collapse or a zero-recall class on test: {tgt['robust_tasks_with_collapse_or_zero_recall'] or 'none'}",
          f"- Robust tasks that no candidate could admit through the OOF gate: {tgt['robust_tasks_not_admitted_by_oof_gate'] or 'none'}",
          f"- Peak tasks that no candidate could admit through the OOF gate: {tgt['peak_tasks_not_admitted_by_oof_gate'] or 'none'}", "",
          "## Paired bootstrap on test accuracy (macro, pp, rows resampled inside tasks)", "",
          "| Comparison | Delta pp | 95% CI |", "|---|---:|---|"]
    for k, v in agg["paired_bootstrap_accuracy"].items():
        L.append(f"| {k} | {v['delta_pp']:+.2f} | [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] |")
    L += ["", "## CALA ablation (fixed base, no search; macro test %, delta vs the raw head in pp)", "",
          "| Base head | Prior shift | Accuracy | d Acc | Balanced acc | d BA | Macro F1 | d F1 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for head in HEADS:
        base = agg["controls"].get(f"{ABLATION_REP}+{head}|raw")
        for gate in (cala.CalaConfig(),) + ABLATION_GATES:
            v = agg["controls"].get(f"{ABLATION_REP}+{head}|{gate.key}")
            if base and v:
                L.append(f"| {ABLATION_REP}+{head} | {gate.key} | {v['accuracy']:.2f} | {v['accuracy'] - base['accuracy']:+.2f} | "
                         f"{v['balanced_accuracy']:.2f} | {v['balanced_accuracy'] - base['balanced_accuracy']:+.2f} | "
                         f"{v['macro_f1']:.2f} | {v['macro_f1'] - base['macro_f1']:+.2f} |")
    L += ["", "## Per task", "", "| Task | n_train | Peak config | Peak OOF acc | Peak test acc | Single-model control acc | Robust config | Robust OOF obj | Robust test BA | Robust test F1 | min recall | Robust admitted |",
          "|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---|"]
    for t, r in rep["tasks"].items():
        L.append(f"| {t} | {r['n_train']} | `{r['peak']['key']}` | {r['peak']['oof']['accuracy']:.2f} | {r['peak']['test']['accuracy']:.2f} | "
                 f"{r['peak_single']['test']['accuracy']:.2f} | `{r['robust']['key']}` | {r['robust']['oof_objective']:.2f} | "
                 f"{r['robust']['test']['balanced_accuracy']:.2f} | {r['robust']['test']['macro_f1']:.2f} | {r['robust']['test']['min_class_recall']:.1f} | {r['robust']['admitted_by_gate']} |")
    L += ["", "## Per-task fixed controls, test accuracy (no search)", "", "| Task | " + " | ".join(c[:-4] for c in agg["controls"] if c.endswith("|raw")) + " |",
          "|---|" + "---:|" * sum(1 for c in agg["controls"] if c.endswith("|raw"))]
    for t, r in rep["tasks"].items():
        L.append(f"| {t} | " + " | ".join(f"{r['controls'][c]['test']['accuracy']:.2f}" for c in agg["controls"] if c.endswith("|raw")) + " |")
    L += ["", "## Conformal (alpha 0.1; model refit on 80% of train, calibrated on 20%)", "",
          "| Task | Track | Marginal coverage | Min class coverage | Mean set size | Singleton rate | Singleton acc | Trivial classes |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for t, r in rep["tasks"].items():
        for trk in ("peak", "robust"):
            c = r["conformal"][trk]
            if "error" in c:
                L.append(f"| {t} | {trk} | error | | | | | |")
                continue
            m = c["mondrian"]
            sa = "n/a" if m["singleton_accuracy"] is None else f"{100 * m['singleton_accuracy']:.1f}"
            L.append(f"| {t} | {trk} | {100 * m['marginal_coverage']:.1f} | {100 * min(m['class_coverage'].values()):.1f} | "
                     f"{m['mean_set_size']:.2f} | {100 * m['singleton_rate']:.1f} | {sa} | {m['trivial_classes'] or '-'} |")
    L += ["", "## Pareto fronts (OOF accuracy vs OOF BA+F1; test columns are post-hoc, never used to choose)", ""]
    for t, r in rep["tasks"].items():
        L.append(f"- **{t}**: " + "; ".join(f"`{p['key']}` OOF {p['oof_accuracy']:.1f}/{p['oof_robust_objective']:.1f} test {p['test']['accuracy']:.1f}/{robust_objective(p['test']):.1f}"
                                            for p in r["pareto_front"][:6]))
    L += ["", "## Honesty notes", "",
          "- CALA sign: the term is subtracted (`f - tau*Phi*log pi`). The brief's `+` would favour the majority class.",
          "- Selection scans a few hundred candidates per task on OOF. Read the OOF-minus-test gap in the JSON as the selection optimism; the fixed controls need no selection.",
          "- Conformal guarantees hold for the 80%-train refit model under exchangeability; the reported point metrics come from the full-train model, which the theorem does not cover.",
          "- Abstention (singleton-only prediction) is reported separately and never feeds the accuracy numbers.",
          "- The earlier 256-d dual ensemble files (`evaluate_dual_70b_72b_*.py`) are untracked files of another task and were left in place; this suite does not import them.", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------------------- offload

def _run(cmd: List[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, **kw)


def offload(args, argv_line: str) -> int:
    """rsync -> ssh run (detached, polled) -> rsync back -> verify hashes -> delete remote work dir."""
    host, wd = args.remote, args.remote_dir
    ssh = ["ssh", "-o", "BatchMode=yes", host]
    ok = False
    files = ["confidence_adaptive_logit_adjustment.py", "geometric_latent_fusion.py", "evaluate_manifold_pareto_ensemble.py",
             "spec21_advanced_heads.py", "grand_challenge_data.py"]
    try:
        _run(ssh + [f"mkdir -p {wd}/suites {wd}/data/full_13 {wd}/features/q {wd}/features/l {wd}/out"])
        _run(["rsync", "-a", *[str(SUITES / f) for f in files], f"{host}:{wd}/suites/"])
        _run(["rsync", "-a", str(ROOT / "benchmarks/data/full_13") + "/", f"{host}:{wd}/data/full_13/"])
        _run(["rsync", "-a", "--partial", args.qwen_dir.rstrip("/") + "/", f"{host}:{wd}/features/q/"])
        _run(["rsync", "-a", "--partial", "--exclude=SHA256SUMS", "--exclude=manifest.json", args.llama_dir.rstrip("/") + "/", f"{host}:{wd}/features/l/"])
        boot = (f"cd {wd} && (test -x venv/bin/python || (python3 -m venv venv && venv/bin/pip install -q numpy scipy scikit-learn threadpoolctl)) "
                "&& venv/bin/python -c 'import numpy,scipy,sklearn,threadpoolctl'")
        _run(ssh + [boot])
        tasks = f" --tasks {' '.join(args.tasks)}" if args.tasks else ""
        remote_cmd = (f"cd {wd} && rm -rf done.flag run.log out && mkdir -p out && GC_TEST_DIR={wd}/data/full_13 setsid nohup sh -c "
                      f"'venv/bin/python suites/evaluate_manifold_pareto_ensemble.py --qwen-dir {wd}/features/q --llama-dir {wd}/features/l "
                      f"--out {wd}/out/report --workers {args.workers}{tasks} --skip-references > run.log 2>&1; echo $? > done.flag' > /dev/null 2>&1 &")
        _run(ssh + [remote_cmd])
        print(f"[offload] started on {host}:{wd}; polling", flush=True)
        while True:
            time.sleep(60)
            r = subprocess.run(ssh + [f"if [ -f {wd}/done.flag ]; then echo DONE $(cat {wd}/done.flag); fi; tail -n 1 {wd}/run.log 2>/dev/null"],
                               capture_output=True, text=True)
            lines = r.stdout.strip().split("\n")
            if lines and lines[0].startswith("DONE "):
                code = int(lines[0].split()[1])
                break
            print("[offload]", lines[-1][:160], flush=True)
        _run(["rsync", "-a", f"{host}:{wd}/run.log", str(Path(str(args.out) + ".run.log"))])
        if code != 0:
            print(f"[offload] remote run failed with exit code {code}; see {args.out}.run.log", file=sys.stderr)
            return code
        for ext in (".json", ".md"):
            _run(["rsync", "-a", f"{host}:{wd}/out/report{ext}", str(args.out) + ext])
        rc = verify_hashes(Path(str(args.out) + ".json"), args)
        ok = rc == 0
        return rc
    finally:
        if ok and not args.keep_remote:
            subprocess.run(ssh + [f"rm -rf {wd}"], check=False)
            print(f"[offload] removed {host}:{wd}", flush=True)
        elif not ok:
            print(f"[offload] run not clean: leaving {host}:{wd} in place for inspection", flush=True)


def verify_hashes(report_path: Path, args) -> int:
    """Remote feature hashes must equal the local files' hashes (proves the copies were faithful)."""
    rep = json.loads(report_path.read_text())
    bad = 0
    for t, r in rep["tasks"].items():
        for remote_path, h in r["feature_sha256"].items():
            local = (Path(args.qwen_dir) if "/features/q/" in remote_path else Path(args.llama_dir)) / f"{t}.npz"
            if digest(local) != h:
                print(f"HASH MISMATCH {t}: {local}", file=sys.stderr)
                bad += 1
    print(f"[offload] feature hash check: {'FAILED ' + str(bad) if bad else 'all match'}", flush=True)
    return 1 if bad else 0


# ------------------------------------------------------------------------------------------ main

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--qwen-dir", default=str(DEFAULT_QDIR))
    ap.add_argument("--llama-dir", default=str(DEFAULT_LDIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output stem (writes .json and .md)")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=3, help="tasks run in parallel processes")
    ap.add_argument("--remote", default=None, help="ssh host to offload the run to (e.g. ai-wsl, dev)")
    ap.add_argument("--remote-dir", default="~/gz_offload_b0925h")
    ap.add_argument("--keep-remote", action="store_true")
    ap.add_argument("--resume", action="store_true", help="reuse per-task results left by an interrupted run")
    ap.add_argument("--skip-references", action="store_true", help="do not read reference scorecards (remote runs)")
    args = ap.parse_args(argv)
    args.out = Path(args.out)
    argv_line = " ".join(shlex.quote(a) for a in (argv if argv is not None else sys.argv))
    tasks = list(args.tasks) if args.tasks else list(TASKS)
    unknown = set(tasks) - set(TASKS)
    if unknown:
        raise SystemExit(f"unknown tasks: {sorted(unknown)}")
    if args.remote:
        args.tasks = tasks if args.tasks else None
        return offload(args, argv_line)
    cores = os.cpu_count() or 1
    workers = max(1, min(args.workers, len(tasks)))
    threads = max(1, cores // 4)
    heavy_threads = max(threads, int(cores * 0.375))          # two large-n tasks run beside one small-task worker
    order = sorted(tasks, key=lambda t: t not in HEAVY_TASKS)
    host = {"name": os.uname().nodename, "cores": cores, "loadavg": [round(x, 2) for x in os.getloadavg()], "workers": workers,
            "threads_small_task": threads, "threads_heavy_task": heavy_threads, "python": sys.version.split()[0],
            "code_sha256": {f: hashlib.sha256((SUITES / f).read_bytes()).hexdigest() for f in
                            ("geometric_latent_fusion.py", "confidence_adaptive_logit_adjustment.py", "evaluate_manifold_pareto_ensemble.py")}}
    print(f"[run] {len(tasks)} tasks, {workers} workers, {threads}/{heavy_threads} threads (small/heavy), loadavg {host['loadavg']}", flush=True)
    rows: Dict[str, Dict] = {}
    rows_dir = args.out.parent / f"{args.out.name}.rows"       # per-task results survive a crash in the aggregation
    rows_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        rows.update({t: json.loads((rows_dir / f"{t}.json").read_text()) for t in tasks if (rows_dir / f"{t}.json").exists()})
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futs = {pool.submit(run_task, t, args.qwen_dir, args.llama_dir, heavy_threads if t in HEAVY_TASKS else threads): t for t in order if t not in rows}
        for fut in as_completed(futs):
            t = futs[fut]
            rows[t] = fut.result()
            (rows_dir / f"{t}.json").write_text(json.dumps(rows[t], default=float))
            r = rows[t]
            print(f"[{t}] peak {r['peak']['key']} test {r['peak']['test']['accuracy']:.2f} | robust {r['robust']['key']} "
                  f"BA {r['robust']['test']['balanced_accuracy']:.2f} F1 {r['robust']['test']['macro_f1']:.2f} | {r['seconds']}", flush=True)
    rows = {t: rows[t] for t in tasks}
    refs = ({"qwen72b": None, "llama70b": None, "prior_256d_dual_selected": None}
            if args.skip_references else references())
    report = build_report(rows, argv_line, host, refs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=float) + "\n")
    args.out.with_suffix(".md").write_text(render_md(report))
    shutil.rmtree(rows_dir)
    print("WROTE", args.out.with_suffix(".json"), args.out.with_suffix(".md"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
