"""Spec 20 P4 end-to-end: Qwen (8192-D) + Gemma (2816-D) dual manifold on the 13 grand-challenge tasks.

Per task, on the SAME rows (ids and labels checked array-equal across both sources and the test file):

  qwen_lp            L2 logistic probe on Qwen alone (sx.LinearProbe, kind "full", inner-CV C). This is
                     the matched single-model baseline: the same solver the dual head uses on top.
  gemma_lp           the same probe on Gemma alone.
  concat_lp          the same probe on [X_Q ; X_G] (11,008-D): naive fusion, capacity control. With the
                     opt-in --concat-pca K it runs on the whitened top-K principal components of the
                     standardized concat instead. That is a DIFFERENT method (unit-variance components
                     change what the C grid means, and K is capped at n_fit_rows - 1), not a faster run
                     of the same one; reports say which one produced the number.
  dual_qwen_only     DualManifoldHead with gate = 0: Qwen principal subspace only. Same projection
                     pipeline and top probe as dual_manifold, innovation block switched off. This is
                     the ablation that isolates the Gemma innovation.
  dual_manifold      DualManifoldHead (gate 1): fit on each fold's train rows only
                     Z_Q = whiten(X_Q) P_Q, B = ridge(Z_Q -> X_G), E_G = X_G - Z_Q B, R_G = top-k_g
                     right singular vectors of E_G, top probe on [Z_Q ; E_G R_G], then folded into
                     logits = X_Q W_fold_Q^T + X_G W_fold_G^T + b_fold (two NumPy GEMVs on the CPU).
  dual_gemma_permuted  dual_manifold with Gemma rows shuffled within train and within test (seeded).
                     Alignment control: if this matches dual_manifold, the Gemma block adds nothing
                     row-specific.

Protocol. 5-fold OOF on train (fold ids as evaluate_full_13_grand_scorecard: seed 20260924) for every
method except the permutation control; then one fit on all train rows and ONE scoring of the test
rows. Test labels are read from the test jsonl once, after all fitting. Nothing is tuned on test.

Selection (no test peeking). dual_selected_1se = dual_manifold if its OOF accuracy beats qwen_lp's
by at least qwen_lp's fold standard error (Breiman 1-SE), else qwen_lp.

The 76.06% reference is the phase-4 scorecard (linear / adapter / supcon, 1-SE ladder, torch solver);
it is read from its report and shown next to these numbers, never recomputed or mixed in.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grand_challenge_data as gd  # noqa: E402
import sota_enhanced_heads as eh  # noqa: E402
import sota_ensemble_experts as sx  # noqa: E402

FOLD_SEED, N_FOLDS = 20260924, 5           # = evaluate_full_13_grand_scorecard
PERM_SEED = 20260926
QWEN_DIM, GEMMA_DIM = 8192, 2816
METHODS = ("qwen_lp", "gemma_lp", "concat_lp", "dual_qwen_only", "dual_manifold")
REFERENCE_MACRO = 76.06
CONCAT_PCA_MIN_DIM = 4096                  # --concat-pca only bites when the concat is wider than this


# ----------------------------------------------------------------------------- data

def load_pair(task: str, qwen_path: Path, gemma_path: Path, test_records: Optional[List[dict]],
              qwen_dim: int = QWEN_DIM, gemma_dim: int = GEMMA_DIM) -> Dict[str, object]:
    """Paired features for one task. Refuses on any misalignment instead of guessing.

    test_records None = labels not loaded yet (fitting never needs them)."""
    with np.load(qwen_path, allow_pickle=False) as zq, np.load(gemma_path, allow_pickle=False) as zg:
        q = {k: zq[k] for k in ("train_full", "test_full", "train_label", "train_ids", "test_ids", "cands")}
        g = {k: zg[k] for k in ("train_full", "test_full", "train_label", "train_ids", "test_ids", "cands")}
        g_info = json.loads(str(zg["info_json"]))
    for k in ("train_ids", "test_ids", "train_label"):
        if q[k].shape != g[k].shape or not np.array_equal(q[k], g[k]):
            raise ValueError(f"{task}: Qwen and Gemma {k} differ; the sources are not row-aligned")
    if len(set(q["train_ids"].tolist())) != len(q["train_ids"]):
        raise ValueError(f"{task}: duplicate train ids")
    if set(q["train_ids"].tolist()) & set(q["test_ids"].tolist()):
        raise ValueError(f"{task}: train and test ids overlap")
    for name, src, dim in (("Qwen", q, qwen_dim), ("Gemma", g, gemma_dim)):
        for split in ("train_full", "test_full"):
            X = src[split]
            n = len(src["train_ids"] if split == "train_full" else src["test_ids"])
            if X.shape != (n, dim):
                raise ValueError(f"{task}: {name} {split} has shape {X.shape}, expected {(n, dim)}")
            if not np.all(np.isfinite(X)):
                raise ValueError(f"{task}: {name} {split} holds non-finite values")
    K = int(q["cands"].shape[0])
    if int(g["cands"].shape[0]) != K:
        raise ValueError(f"{task}: Qwen K={K}, Gemma K={g['cands'].shape[0]}")
    out = {"task": task, "K": K, "Xq_tr": q["train_full"].astype(np.float32),
           "Xg_tr": g["train_full"].astype(np.float32), "y_tr": q["train_label"].astype(np.int64),
           "Xq_te": q["test_full"].astype(np.float32), "Xg_te": g["test_full"].astype(np.float32),
           "train_ids": q["train_ids"], "test_ids": q["test_ids"], "gemma_info": g_info, "y_te": None}
    if test_records is not None:
        out["y_te"] = test_labels(task, test_records, q["test_ids"], K)
    return out


def test_labels(task: str, records: List[dict], test_ids: np.ndarray, K: int) -> np.ndarray:
    ids = np.array([r["id"] for r in records])
    if ids.shape != test_ids.shape or not np.array_equal(ids, test_ids):
        raise ValueError(f"{task}: test jsonl ids differ from the feature test ids")
    cands = records[0]["candidates"]
    if len(cands) != K or any(r["candidates"] != cands for r in records):
        raise ValueError(f"{task}: test candidate lists are not the fixed K={K} label set")
    return np.array([cands.index(r["ground_truth"]) for r in records], dtype=np.int64)


def fold_ids(n: int) -> np.ndarray:
    return np.random.default_rng(FOLD_SEED).permutation(n) % N_FOLDS


# -------------------------------------------------------------------------- methods

Scorer = Callable[[np.ndarray, np.ndarray], np.ndarray]


def fit_method(method: str, Xq: np.ndarray, Xg: np.ndarray, y: np.ndarray, K: int, *, seed: int,
               device: str, dual_cfg: eh.DualManifoldConfig,
               concat_pca: Optional[int] = None) -> Tuple[Scorer, dict]:
    """Fit one method on the rows given (and only those). Returns (scores(Xq, Xg) -> (N, K), info).

    concat_pca (None = off) only touches concat_lp, and only when the concat is wider than
    CONCAT_PCA_MIN_DIM. LinearProbe then caps the width at min(concat_pca, n_fit_rows - 1), so on small
    folds feature_dim in info can be far below concat_pca."""
    if method in ("qwen_lp", "gemma_lp", "concat_lp"):
        pick = {"qwen_lp": lambda a, b: a, "gemma_lp": lambda a, b: b,
                "concat_lp": lambda a, b: np.concatenate([a, b], axis=1)}[method]
        pca_k = concat_pca if method == "concat_lp" and concat_pca and Xq.shape[1] + Xg.shape[1] > CONCAT_PCA_MIN_DIM else None
        m = sx.LinearProbe.fit(pick(Xq, Xg), y, K, pair=False, kind="full", pca_k=pca_k, seed=seed, device=device)
        return (lambda a, b: m.scores(pick(a, b))), {"C": m.cfg["C"], "feature_dim": m.cfg["feature_dim"],
                                                     "pca_k": pca_k}
    if method in ("dual_manifold", "dual_qwen_only"):
        cfg = dual_cfg if method == "dual_manifold" else replace(dual_cfg, gate=0.0)
        h = eh.DualManifoldHead.fit(Xq, Xg, y, K, pair=False, seed=seed, config=cfg, device=device)
        keep = ("probe_C", "actual_projection_rank", "rank_truncated", "qwen_variance_kept",
                "innovation_variance_kept", "gemma_variance_explained_by_qwen",
                "orthogonality_identity_max_abs_err", "export_max_abs_err", "train_acc_folded", "train_seconds")
        info = {k: h.info[k] for k in keep}
        info["supervised_params"], info["exported_params"] = h.supervised_param_count(), h.exported_param_count()
        info["_head"] = h
        return h.scores, info
    raise ValueError(f"unknown method {method!r}")


def oof_scores(method: str, pair: Dict[str, object], folds: np.ndarray, *, seed: int, device: str,
               dual_cfg: eh.DualManifoldConfig, concat_pca: Optional[int] = None) -> Tuple[np.ndarray, List[dict]]:
    Xq, Xg, y, K = pair["Xq_tr"], pair["Xg_tr"], pair["y_tr"], pair["K"]
    S = np.full((len(y), K), np.nan, dtype=np.float32)
    infos = []
    for k in range(N_FOLDS):
        tr, ho = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        t0 = time.perf_counter()
        score, info = fit_method(method, Xq[tr], Xg[tr], y[tr], K, seed=seed + k, device=device, dual_cfg=dual_cfg,
                                 concat_pca=concat_pca)
        S[ho] = score(Xq[ho], Xg[ho])
        print(f"  [oof] {pair['task']}/{method} fold {k}: {time.perf_counter() - t0:.1f}s", flush=True)
        info.pop("_head", None)
        infos.append(info)
    if np.isnan(S).any():
        raise RuntimeError(f"{method}: some rows got no OOF score")
    return S, infos


# -------------------------------------------------------------------------- scoring

def fold_accuracy(S: np.ndarray, y: np.ndarray, folds: np.ndarray) -> Tuple[float, float, List[float]]:
    """(pooled OOF accuracy, standard error of the per-fold accuracies, per-fold accuracies)."""
    pred = S.argmax(1)
    per = [float(np.mean(pred[folds == k] == y[folds == k])) for k in range(N_FOLDS)]
    return float(np.mean(pred == y)), float(np.std(per, ddof=1) / math.sqrt(N_FOLDS)), per


def one_se_pick(cand_acc: float, base_acc: float, base_se: float) -> bool:
    """True when the candidate clears the baseline by at least one baseline SE (and strictly)."""
    delta = cand_acc - base_acc
    return delta > 0 and delta >= base_se


def wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 2), round(100 * (c + h), 2)]


def mcnemar_exact(a_ok: np.ndarray, b_ok: np.ndarray) -> Dict[str, float]:
    """Exact two-sided McNemar on paired correctness vectors (a = baseline, b = candidate)."""
    only_a = int(np.sum(a_ok & ~b_ok))
    only_b = int(np.sum(~a_ok & b_ok))
    n = only_a + only_b
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n)
    return {"only_baseline_correct": only_a, "only_candidate_correct": only_b, "p_value": p}


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n)


def gemv_latency_us(head: "eh.DualManifoldHead", xq: np.ndarray, xg: np.ndarray, reps: int = 200) -> float:
    """Median wall time of one single-row folded scores() call, microseconds."""
    head.scores(xq, xg)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        head.scores(xq, xg)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e6)


def permuted(X: np.ndarray, seed: int) -> np.ndarray:
    return X[np.random.default_rng(seed).permutation(len(X))]


def run_task(pair: Dict[str, object], *, device: str, dual_cfg: eh.DualManifoldConfig, seed: int = 0,
             label_loader: Optional[Callable[[], np.ndarray]] = None, run_oof: bool = True,
             concat_pca: Optional[int] = None) -> dict:
    """OOF on train, final fit on all train, then test labels are loaded and read exactly once."""
    task, K, y = pair["task"], pair["K"], pair["y_tr"]
    folds = fold_ids(len(y))
    t0 = time.perf_counter()
    oof: Dict[str, dict] = {}
    if run_oof:
        for m in METHODS:
            S, infos = oof_scores(m, pair, folds, seed=seed, device=device, dual_cfg=dual_cfg,
                                  concat_pca=concat_pca)
            acc, se, per = fold_accuracy(S, y, folds)
            oof[m] = {"cv_acc": acc, "cv_se": se, "fold_accs": per, "fold_info": infos}
    finals: Dict[str, Tuple[Scorer, dict]] = {}
    for m in METHODS:
        t1 = time.perf_counter()
        finals[m] = fit_method(m, pair["Xq_tr"], pair["Xg_tr"], y, K, seed=seed, device=device, dual_cfg=dual_cfg,
                               concat_pca=concat_pca)
        print(f"  [final] {task}/{m}: {time.perf_counter() - t1:.1f}s", flush=True)
    perm_score, perm_info = fit_method("dual_manifold", pair["Xq_tr"], permuted(pair["Xg_tr"], PERM_SEED), y, K,
                                       seed=seed, device=device, dual_cfg=dual_cfg)
    # ----- test: labels enter here, after every fit is done.
    y_te = pair["y_te"] if pair.get("y_te") is not None else label_loader()
    test_pred = {m: f(pair["Xq_te"], pair["Xg_te"]).argmax(1) for m, (f, _) in finals.items()}
    test_pred["dual_gemma_permuted"] = perm_score(pair["Xq_te"], permuted(pair["Xg_te"], PERM_SEED + 1)).argmax(1)
    head = finals["dual_manifold"][1]["_head"]
    res = {"task": task, "K": K, "n_train": int(len(y)), "n_test": int(len(y_te)), "methods": {}}
    for m, pred in test_pred.items():
        ok = pred == y_te
        res["methods"][m] = {"test_acc": round(100 * float(ok.mean()), 2), "correct": int(ok.sum()),
                             "wilson95": wilson(int(ok.sum()), len(ok)),
                             "max_pred_class_frac": round(float(np.bincount(pred, minlength=K).max() / len(pred)), 4)}
        if m in oof:
            res["methods"][m].update(cv_acc=round(100 * oof[m]["cv_acc"], 2), cv_se=round(100 * oof[m]["cv_se"], 3))
    base_ok = test_pred["qwen_lp"] == y_te
    for m in ("dual_manifold", "concat_lp", "dual_qwen_only", "gemma_lp", "dual_gemma_permuted"):
        res["methods"][m]["mcnemar_vs_qwen_lp"] = mcnemar_exact(base_ok, test_pred[m] == y_te)
    res["dual_manifold_info"] = {k: v for k, v in finals["dual_manifold"][1].items() if k != "_head"}
    res["dual_gemma_permuted_info"] = {k: v for k, v in perm_info.items() if k != "_head"}
    res["concat_lp_info"] = dict(finals["concat_lp"][1])
    if run_oof:
        pick = one_se_pick(oof["dual_manifold"]["cv_acc"], oof["qwen_lp"]["cv_acc"], oof["qwen_lp"]["cv_se"])
        chosen = "dual_manifold" if pick else "qwen_lp"
        res["selection_1se"] = {"chosen": chosen, "dual_cv_minus_qwen_cv": round(
            100 * (oof["dual_manifold"]["cv_acc"] - oof["qwen_lp"]["cv_acc"]), 3),
            "qwen_cv_se": round(100 * oof["qwen_lp"]["cv_se"], 3)}
        res["methods"]["dual_selected_1se"] = dict(res["methods"][chosen], chosen=chosen)
        res["oof_fold_info"] = {m: oof[m]["fold_info"] for m in ("dual_manifold", "dual_qwen_only", "concat_lp")}
    res["cpu_fold_check"] = {
        "test_max_abs_err_fold_vs_unfolded": float(np.max(np.abs(
            head.scores(pair["Xq_te"], pair["Xg_te"]) - head.scores_unfolded(pair["Xq_te"], pair["Xg_te"])))),
        "single_row_gemv_us_median": round(gemv_latency_us(head, pair["Xq_te"][0], pair["Xg_te"][0]), 1)}
    res["seconds"] = round(time.perf_counter() - t0, 1)
    return res


# -------------------------------------------------------------------------- report

def aggregate(rows: Dict[str, dict], reference: Optional[dict]) -> dict:
    names = list(next(iter(rows.values()))["methods"].keys())
    agg = {"n_tasks": len(rows), "macro_test_acc": {}}
    for m in names:
        agg["macro_test_acc"][m] = round(float(np.mean([r["methods"][m]["test_acc"] for r in rows.values()])), 2)
    deltas = {t: r["methods"]["dual_manifold"]["test_acc"] - r["methods"]["qwen_lp"]["test_acc"] for t, r in rows.items()}
    wins = sum(d > 0 for d in deltas.values())
    losses = sum(d < 0 for d in deltas.values())
    agg["dual_vs_qwen_lp"] = {"per_task_delta_pp": {t: round(d, 2) for t, d in deltas.items()},
                              "macro_delta_pp": round(float(np.mean(list(deltas.values()))), 2),
                              "wins": wins, "losses": losses, "ties": len(deltas) - wins - losses,
                              "sign_test_p": round(sign_test(wins, losses), 4),
                              "tasks_mcnemar_p_lt_0.05": sorted(
                                  t for t, r in rows.items()
                                  if r["methods"]["dual_manifold"]["mcnemar_vs_qwen_lp"]["p_value"] < 0.05)}
    if reference:
        ref = {t: reference["tasks"][t]["accuracy"] for t in rows if t in reference.get("tasks", {})}
        agg["reference_phase4"] = {"macro_13_reported": reference["aggregate"]["macro_avg_13"],
                                   "per_task": ref,
                                   "macro_over_these_tasks": round(float(np.mean(list(ref.values()))), 2) if ref else None}
        for m in ("dual_manifold", "dual_selected_1se"):
            if m in names and ref:
                agg[f"{m}_minus_reference_macro_pp"] = round(
                    float(np.mean([rows[t]["methods"][m]["test_acc"] - ref[t] for t in ref])), 2)
    return agg


def render_md(rep: dict) -> str:
    rows, agg = rep["tasks"], rep["aggregate"]
    names = list(agg["macro_test_acc"].keys())
    L = [f"# {rep['title']}", "", f"Generated {rep['generated_utc']} on {rep['host']['platform']}.", "",
         f"Command: `{rep['command']}`", "", "## Macro test accuracy (%)", "",
         "| method | macro over %d tasks |" % agg["n_tasks"], "|---|---|"]
    L += [f"| {m} | {agg['macro_test_acc'][m]:.2f} |" for m in names]
    if "reference_phase4" in agg:
        r = agg["reference_phase4"]
        L += [f"| reference phase-4 scorecard (not recomputed) | {r['macro_over_these_tasks']} "
              f"(reported macro-13 {r['macro_13_reported']}) |"]
    d = agg["dual_vs_qwen_lp"]
    L += ["", "## Dual manifold vs matched Qwen-only probe", "",
          f"- macro delta: **{d['macro_delta_pp']:+.2f} pp**; wins/losses/ties {d['wins']}/{d['losses']}/{d['ties']}; "
          f"sign test p = {d['sign_test_p']}",
          f"- tasks with McNemar p < 0.05: {', '.join(d['tasks_mcnemar_p_lt_0.05']) or 'none'}"]
    for k in ("dual_manifold_minus_reference_macro_pp", "dual_selected_1se_minus_reference_macro_pp"):
        if k in agg:
            L.append(f"- {k.replace('_', ' ')}: {agg[k]:+.2f} pp")
    L += ["", "## Per task test accuracy (%)", "",
          "| task | n | " + " | ".join(names) + " | ref | dual-qwen | McNemar p |",
          "|---|---|" + "---|" * len(names) + "---|---|---|"]
    ref = agg.get("reference_phase4", {}).get("per_task", {})
    for t, r in rows.items():
        cells = " | ".join(f"{r['methods'][m]['test_acc']:.2f}" for m in names)
        mc = r["methods"]["dual_manifold"]["mcnemar_vs_qwen_lp"]["p_value"]
        L.append(f"| {t} | {r['n_test']} | {cells} | {ref.get(t, '-')} | "
                 f"{d['per_task_delta_pp'][t]:+.2f} | {mc:.3g} |")
    L += ["", "## Dual manifold diagnostics (final fit on all train rows)", "",
          "| task | k_q, k_g | Gemma var explained by Qwen | innovation var kept | fold err | test fold-vs-chain err | GEMV us | 1-SE choice |",
          "|---|---|---|---|---|---|---|---|"]
    for t, r in rows.items():
        i, c = r["dual_manifold_info"], r["cpu_fold_check"]
        sel = r.get("selection_1se", {}).get("chosen", "-")
        L.append(f"| {t} | {i['actual_projection_rank']} | {i['gemma_variance_explained_by_qwen']:.3f} | "
                 f"{i['innovation_variance_kept']:.3f} | {i['export_max_abs_err']:.2e} | "
                 f"{c['test_max_abs_err_fold_vs_unfolded']:.2e} | {c['single_row_gemv_us_median']} | {sel} |")
    L += ["", "## Protocol and limits", ""] + [f"- {x}" for x in rep["protocol"]["notes"]]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------- resume

def run_fingerprint(args: argparse.Namespace, dual_cfg: eh.DualManifoldConfig) -> dict:
    """Everything that changes a task's numbers. A partial file is only resumed under the same one."""
    return {"device": args.device, "oof": not args.no_oof, "concat_pca": args.concat_pca,
            "dual_config": {"k_qwen": dual_cfg.k_qwen, "k_gemma": dual_cfg.k_gemma,
                            "target_dim": dual_cfg.target_dim, "regularization": dual_cfg.regularization,
                            "rank_rtol": dual_cfg.rank_rtol, "gate": dual_cfg.gate},
            "folds": {"n": N_FOLDS, "seed": FOLD_SEED}, "methods": list(METHODS)}


def load_partial(path: Path, fingerprint: dict) -> Dict[str, dict]:
    """Rows finished by an earlier run, or {} when there is no partial file.

    Refuses (ValueError) when the partial was written under another configuration, so a resumed report
    never mixes tasks computed with different settings. A partial from before fingerprints existed has
    no "fingerprint" key: it is accepted, with a warning, only when concat_pca is off, because that is
    the one setting those runs could not have used."""
    if not path.exists():
        return {}
    saved = json.loads(path.read_text(encoding="utf-8"))
    old = saved.get("fingerprint")
    if old is None:
        if fingerprint["concat_pca"]:
            raise ValueError(f"{path} is a legacy partial (no fingerprint) and cannot be resumed under "
                             f"--concat-pca {fingerprint['concat_pca']}; pass --no-resume or another --partial-json")
        print(f"[resume] WARNING {path} has no fingerprint (legacy); its settings are unverified", flush=True)
    elif old != fingerprint:
        diff = sorted(k for k in fingerprint if old.get(k) != fingerprint[k])
        raise ValueError(f"{path} was written under different settings ({', '.join(diff)}); pass --no-resume "
                         f"to start over or --partial-json to use another file")
    return saved["tasks"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--qwen-dir", type=Path, required=True)
    ap.add_argument("--gemma-dir", type=Path, required=True)
    ap.add_argument("--test-dir", type=Path, default=gd.TEST_DIR)
    ap.add_argument("--tasks", default=",".join(gd.TASKS),
                    help="comma-separated task names (default: all 13); an unknown name is refused up front")
    ap.add_argument("--device", default="cpu", help="fit-time device of the probes (cpu | cuda | auto)")
    ap.add_argument("--k-qwen", type=int, default=1536)
    ap.add_argument("--k-gemma", type=int, default=512)
    ap.add_argument("--target-dim", type=int, default=2048)
    ap.add_argument("--regularization", type=float, default=eh.DualManifoldConfig.regularization)
    ap.add_argument("--rank-rtol", type=float, default=eh.DualManifoldConfig.rank_rtol)
    ap.add_argument("--reference", type=Path, default=None, help="phase-4 scorecard json (76.06%%)")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--no-oof", action="store_true", help="skip the 5-fold OOF (no 1-SE selection)")
    ap.add_argument("--concat-pca", type=int, default=None, metavar="K",
                    help=f"concat_lp only: whitened PCA to K components when the concat is wider than "
                         f"{CONCAT_PCA_MIN_DIM}-D (K is capped at n_fit_rows - 1). A different method from the "
                         f"full-width probe, off by default; the report records it")
    ap.add_argument("--partial-json", type=Path, default=None,
                    help="per-task checkpoint, resumed automatically when present (default: <out-json>.partial.json)")
    ap.add_argument("--no-resume", action="store_true", help="ignore an existing partial file and overwrite it")
    args = ap.parse_args(argv)
    gd.TEST_DIR = args.test_dir
    tasks = [t for t in args.tasks.split(",") if t]
    unknown = sorted(set(tasks) - set(gd.TASKS))
    if unknown or not tasks:
        ap.error(f"--tasks: unknown {unknown}" if unknown else "--tasks is empty")
    if args.concat_pca is not None and args.concat_pca < 2:
        ap.error("--concat-pca must be >= 2")
    dual_cfg = eh.DualManifoldConfig(k_qwen=args.k_qwen, k_gemma=args.k_gemma, target_dim=args.target_dim,
                                     regularization=args.regularization, rank_rtol=args.rank_rtol)
    reference = json.loads(args.reference.read_text(encoding="utf-8")) if args.reference else None
    fingerprint = run_fingerprint(args, dual_cfg)
    partial = args.partial_json or args.out_json.with_suffix(".partial.json")
    try:
        rows = {} if args.no_resume else load_partial(partial, fingerprint)
    except ValueError as e:
        ap.error(str(e))
    if rows:
        print(f"[resume] done: {sorted(rows)}; todo: {[t for t in tasks if t not in rows]}", flush=True)
    for task in tasks:
        if task in rows:
            continue
        pair = load_pair(task, args.qwen_dir / f"{task}.npz", args.gemma_dir / f"{task}.npz", None)
        loader = lambda task=task, pair=pair: test_labels(task, gd.load_test(task), pair["test_ids"], pair["K"])  # noqa: E731
        print(f"[task] {task}: n_train={len(pair['y_tr'])} n_test={len(pair['test_ids'])} K={pair['K']}", flush=True)
        rows[task] = run_task(pair, device=args.device, dual_cfg=dual_cfg, label_loader=loader, run_oof=not args.no_oof,
                            concat_pca=args.concat_pca)
        print(f"[task] {task}: " + json.dumps({m: v["test_acc"] for m, v in rows[task]["methods"].items()}), flush=True)
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps({"fingerprint": fingerprint, "tasks": rows}, indent=1), encoding="utf-8")
    rep = {
        "title": "Spec 20 P4 dual manifold (Qwen3.5-9B q9b_diff_compact 8192-D + Gemma-4 26B-A4B 2816-D), 13 tasks",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": " ".join([Path(sys.argv[0]).name] + list(argv if argv is not None else sys.argv[1:])),
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "protocol": {
            "qwen_dir": str(args.qwen_dir), "gemma_dir": str(args.gemma_dir), "device": args.device,
            "dual_config": {"k_qwen": args.k_qwen, "k_gemma": args.k_gemma, "target_dim": args.target_dim,
                            "regularization": dual_cfg.regularization, "rank_rtol": dual_cfg.rank_rtol,
                            "gate": dual_cfg.gate},
            "folds": {"n": N_FOLDS, "seed": FOLD_SEED}, "oof": not args.no_oof,
            "concat_pca": args.concat_pca, "concat_pca_min_dim": CONCAT_PCA_MIN_DIM,
            "tasks_requested": tasks, "partial_json": str(partial),
            "notes": [
                "Rows: train/test ids and train labels checked array-equal between the Qwen and Gemma files; "
                "test ids checked equal to the test jsonl. Any mismatch aborts the task.",
                "Every projection (P_Q, B, R_G) and every probe is fitted on the rows of one fold (OOF) or on all "
                "train rows (final); test rows are scored once, after all fits, and test labels are loaded then.",
                "qwen_lp is the matched single-model baseline (same LinearProbe solver as the dual top probe). "
                f"The {REFERENCE_MACRO}% reference is a different pipeline (linear/adapter/supcon 1-SE ladder) "
                "and is shown for context only.",
                "Only the whole-context vectors (train_full/test_full) are used on every task, pair tasks included, "
                "as in the 76.06% scorecard.",
                "Gemma vectors are llama-server last-token pooled final states (raw, embd_normalize=-1); Qwen "
                "vectors are [last@24 ; last@24 - last@16]. Different models, layers and tokenizers; the same text "
                "and the same (max_tok, head_tok) cut, each in its own tokenizer.",
                "dual_gemma_permuted shuffles Gemma rows (train and test separately): the alignment control.",
                "concat_lp: full-width 11,008-D probe unless concat_pca is set above; with it, concat_lp is a "
                "whitened-PCA probe (different method, feature_dim per fit in concat_lp_info / oof_fold_info).",
                "Resume: rows already in the partial file are reused as computed; the partial carries the run "
                "settings and a mismatch is refused. The aggregate covers every row in the partial file, "
                "including tasks outside a narrower --tasks.",
                "GEMV latency is one Python call on the evaluation host, NumPy BLAS, not an isolated benchmark.",
            ]},
        "tasks": rows,
    }
    rep["aggregate"] = aggregate(rows, reference)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    args.out_md.write_text(render_md(rep), encoding="utf-8")
    print(json.dumps(rep["aggregate"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
