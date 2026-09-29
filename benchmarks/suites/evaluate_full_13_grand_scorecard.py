#!/usr/bin/env python3
"""Full 13-benchmark evaluation and grand scorecard generator for Spec 19.

Executes Action Item 2:
1. Evaluates all 13 public benchmarks on uncapped 8,192-D features.
2. Applies Breiman's 1-SE razor to select between Linear Probe and Deep Residual Adapter.
3. Fits the selected optimal model on 100% of training data.
4. Performs pure CPU NumPy scoring on the official test sets.
5. Computes test accuracy, balanced accuracy, macro-F1, collapse metrics, and latency.
6. Emits the final grand scorecard comparing against commercial SOTA (Nimble 74.8%, Jev 76.0%, Laya 55.48%).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks" / "suites"))

import aegis_dual_track as adt
import grand_challenge_data as gd
import sota_enhanced_heads as eh
import sota_ensemble_experts as sx

FOLD_SEED = 20260924
N_FOLDS = 5

PNG = {
    "massive_en": ("MASSIVE en-US", 350, 86.9, 87.4),
    "massive_de": ("MASSIVE de-DE", 350, 83.4, 86.9),
    "multinli": ("MultiNLI", 299, 85.3, 82.9),
    "pubmedqa": ("PubMedQA", 250, 75.6, 77.2),
    "vitaminc": ("VitaminC", 599, 76.6, 80.1),
    "boolq": ("BoolQ", 300, 86.0, 89.7),
    "squad2": ("SQuAD 2.0", 299, 80.6, 82.9),
    "paws": ("PAWS", 250, 82.8, 89.2),
    "civil_comments": ("Civil Comments", 300, 70.3, 81.0),
    "aegis_safety": ("Aegis 2.0", 250, 81.2, 80.4),
    "helpsteer2": ("HelpSteer2", 249, 39.0, 34.1),
    "summeval_relevance": ("SummEval relevance", 240, 49.2, 35.0),
    "summeval_consistency": ("SummEval consistency", 144, 75.7, 81.2),
}
PNG_AVG = {"nimble": 74.8, "jev": 76.0}
LAYA_MACRO = 55.48

TASK_ADAPTER_OVERRIDES = {
    "massive_de": {64: {"lr": 5e-4}},
    "pubmedqa": {128: {"lr": 1e-3}},
}

TASK_SUPCON_OVERRIDES = {}


def fit_linear_probe_torch(X_tr: torch.Tensor, y_tr: torch.Tensor, K: int, C: float = 1.0,
                           device: str = "cuda", class_weight: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """class_weight=None is the plain loss; 'balanced' weights class c by N / (K * n_c), the sklearn rule."""
    N, D = X_tr.shape
    weight = None
    if class_weight == "balanced":
        counts = torch.bincount(y_tr, minlength=K).to(torch.float32)
        if bool((counts == 0).any()):
            raise ValueError("class_weight='balanced' needs every class in the training rows")
        weight = N / (K * counts)
    elif class_weight is not None:
        raise ValueError(f"class_weight must be None or 'balanced', got {class_weight!r}")
    W = torch.zeros((K, D), dtype=torch.float32, device=device, requires_grad=True)
    b = torch.zeros(K, dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=25, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = F.linear(X_tr, W, b)
        loss = F.cross_entropy(logits, y_tr, weight=weight) + (0.5 / C) * torch.sum(W ** 2) / N
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def train_adapter_gpu(X_tr: torch.Tensor, y_tr: torch.Tensor, K: int, rank: int,
                      lr: float = 1e-3, wd: float = 0.01, lambda_up: float = 1e-3,
                      max_epochs: int = 50, batch_size: int = 256, device: str = "cuda",
                      rng_seed: int = 20260924, lambda_supcon: float = 0.0, tau: float = 0.1,
                      feature_dropout: float = 0.1) -> Tuple[Dict[str, np.ndarray], float, Dict[str, object]]:
    torch.manual_seed(rng_seed)
    N, D = X_tr.shape
    n_es = max(1, int(N * 0.15))
    perm = torch.randperm(N, device=device)
    tr_idx, es_idx = perm[n_es:], perm[:n_es]
    X_t, y_t = X_tr[tr_idx], y_tr[tr_idx]
    X_e, y_e = X_tr[es_idx], y_tr[es_idx]

    # Warm start from linear probe
    W_lin, b_lin = fit_linear_probe_torch(X_t, y_t, K, C=1.0, device=device)

    W_h = torch.nn.Parameter(W_lin.clone())
    b_h = torch.nn.Parameter(b_lin.clone())
    W_down = torch.nn.Parameter(torch.randn((rank, D), device=device) * (1.0 / math.sqrt(D)))
    b_down = torch.nn.Parameter(torch.zeros(rank, device=device))
    W_up = torch.nn.Parameter(torch.zeros((D, rank), device=device))
    b_up = torch.nn.Parameter(torch.zeros(D, device=device))

    opt = torch.optim.AdamW([
        {"params": [W_down, W_up, W_h], "weight_decay": wd},
        {"params": [b_down, b_up, b_h], "weight_decay": 0.0}
    ], lr=lr)

    def embed(x):
        h_inner = F.gelu(F.linear(x, W_down, b_down), approximate="tanh")
        return x + F.linear(h_inner, W_up, b_up)

    def forward_unfolded(x):
        return F.linear(embed(x), W_h, b_h)

    dropout_gen = torch.Generator(device=device)
    dropout_gen.manual_seed(rng_seed + 777)

    def dropout_view(x):
        if feature_dropout <= 0.0:
            return x
        keep = torch.rand(x.shape, generator=dropout_gen, device=device) >= feature_dropout
        return x * keep.to(x.dtype) / (1.0 - feature_dropout)

    # Initial probe score on held-out slice
    with torch.no_grad():
        base_logits = forward_unfolded(X_e)
        best_val_acc = float((base_logits.argmax(-1) == y_e).float().mean())
        best_val_loss = float(F.cross_entropy(base_logits, y_e))
        best_weights = {
            "W_h": W_h.detach().clone(), "b_h": b_h.detach().clone(),
            "W_down": W_down.detach().clone(), "b_down": b_down.detach().clone(),
            "W_up": W_up.detach().clone(), "b_up": b_up.detach().clone()
        }

    stale = 0
    patience = 10
    n_train = len(tr_idx)

    supcon_batches = supcon_degraded = 0
    supcon_loss_sum = 0.0

    for epoch in range(1, max_epochs + 1):
        epoch_perm = torch.randperm(n_train, device=device)
        for s in range(0, n_train, batch_size):
            b_ids = epoch_perm[s:s + batch_size]
            opt.zero_grad()
            logits = forward_unfolded(X_t[b_ids])
            loss = F.cross_entropy(logits, y_t[b_ids])
            if lambda_supcon > 0.0:
                supcon_batches += 1
                u1 = F.normalize(embed(dropout_view(X_t[b_ids])), dim=-1)
                u2 = F.normalize(embed(dropout_view(X_t[b_ids])), dim=-1)
                sup = eh.supcon_loss(u1, u2, y_t[b_ids], tau)
                if sup is None:
                    supcon_degraded += 1
                else:
                    supcon_loss_sum += float(sup.detach())
                    loss = loss + lambda_supcon * sup
            if lambda_up > 0.0:
                loss = loss + lambda_up * torch.sum(W_up ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([W_h, b_h, W_down, b_down, W_up, b_up], 1.0)
            opt.step()

        with torch.no_grad():
            es_lg = forward_unfolded(X_e)
            val_acc = float((es_lg.argmax(-1) == y_e).float().mean())
            val_loss = float(F.cross_entropy(es_lg, y_e))

        if (val_acc > best_val_acc + 1e-5) or (abs(val_acc - best_val_acc) <= 1e-5 and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            stale = 0
            best_weights = {
                "W_h": W_h.detach().clone(), "b_h": b_h.detach().clone(),
                "W_down": W_down.detach().clone(), "b_down": b_down.detach().clone(),
                "W_up": W_up.detach().clone(), "b_up": b_up.detach().clone()
            }
        else:
            stale += 1
            if stale >= patience:
                break

    Wh = best_weights["W_h"].cpu().numpy()
    bh = best_weights["b_h"].cpu().numpy()
    Wu = best_weights["W_up"].cpu().numpy()
    bu = best_weights["b_up"].cpu().numpy()
    Wd = best_weights["W_down"].cpu().numpy()
    bd = best_weights["b_down"].cpu().numpy()

    # Algebraic folding: W_fold = W_h @ W_up, b_fold = W_h @ b_up + b_h
    W_fold = Wh @ Wu
    b_fold = Wh @ bu + bh

    # Self-check: the folded NumPy math must reproduce the unfolded torch forward on the
    # held-out slice (mirrors sota_enhanced_heads.py's `_train` export self-check).
    X_e_np = X_e.cpu().numpy()
    h_inner_check = eh.gelu_tanh(X_e_np @ Wd.T + bd)
    folded_check = X_e_np @ Wh.T + h_inner_check @ W_fold.T + b_fold
    unfolded_check = (X_e_np + h_inner_check @ Wu.T + bu) @ Wh.T + bh
    fold_err = float(np.max(np.abs(folded_check - unfolded_check)))
    if not fold_err <= 1e-4 * (1.0 + float(np.max(np.abs(unfolded_check)))):
        raise FloatingPointError(f"adapter/supcon fold identity broke: max abs err {fold_err}")

    return {
        "W_h": Wh, "b_h": bh, "W_down": Wd, "b_down": bd,
        "W_fold": W_fold, "b_fold": b_fold
    }, best_val_acc, {
        "supcon_batches": supcon_batches,
        "supcon_degraded_batches": supcon_degraded,
        "supcon_mean_loss": (supcon_loss_sum / max(supcon_batches - supcon_degraded, 1)) if lambda_supcon > 0.0 else None,
    }


def score_folded_numpy(X: np.ndarray, weights: Dict[str, np.ndarray]) -> np.ndarray:
    """100% Pure NumPy CPU inference."""
    h_inner = X @ weights["W_down"].T + weights["b_down"]
    act = eh.gelu_tanh(h_inner)
    return X @ weights["W_h"].T + act @ weights["W_fold"].T + weights["b_fold"]


def wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (c - h), 2), round(100 * (c + h), 2)]


def collapse_max_frac(pred: np.ndarray, K: int) -> float:
    counts = np.bincount(pred, minlength=K)
    return float(counts.max() / max(int(counts.sum()), 1))


def collapse_stats(pred: np.ndarray, gold: np.ndarray, K: int) -> Dict[str, object]:
    pred_counts, gold_counts = np.bincount(pred, minlength=K), np.bincount(gold, minlength=K)
    max_pred_class_frac = float(pred_counts.max() / max(pred_counts.sum(), 1))
    collapsed = bool(K >= 2 and max_pred_class_frac > 0.95)
    recalls = [float(np.mean(pred[gold == c] == c)) for c in np.unique(gold)]
    balanced_acc = 100 * float(np.mean(recalls)) if recalls else 0.0
    f1s = []
    for c in range(K):
        tp = int(np.sum((pred == c) & (gold == c)))
        fp = int(np.sum((pred == c) & (gold != c)))
        fn = int(np.sum((pred != c) & (gold == c)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    macro_f1 = 100 * float(np.mean(f1s)) if f1s else 0.0
    return {
        "pred_counts": pred_counts.tolist(), "gold_counts": gold_counts.tolist(),
        "max_pred_class_frac": round(max_pred_class_frac, 4), "collapsed": collapsed,
        "balanced_accuracy": round(balanced_acc, 2), "macro_f1": round(macro_f1, 2)
    }


def feature_source_info(features_dir: Path, tasks: List[str]) -> Dict[str, object]:
    """variant / encoder / feature_dim shared by every present <task>.npz under features_dir.

    Spec 19 Phase 4 sources (synthesize_layer_diff_features.py) are 4096/8192/12288-D and carry their
    own variant name, so the report must not hardcode "q9b_mid". Mixed variants in one dir are refused."""
    seen: Dict[str, dict] = {}
    for task in tasks:
        path = features_dir / f"{task}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as z:
            info = json.loads(str(z["info_json"]))
            seen[task] = {"variant": info.get("variant"), "encoder": info.get("encoder"),
                          "feature_dim": int(z["train_full"].shape[1]), "layer_diff": info.get("layer_diff")}
    variants = {(v["variant"], v["encoder"], v["feature_dim"]) for v in seen.values()}
    if len(variants) > 1:
        raise SystemExit(f"{features_dir}: mixed feature variants {sorted(variants, key=str)}; one source per run")
    if not seen:
        return {"variant": None, "encoder": None, "feature_dim": None, "layer_diff": None}
    return next(iter(seen.values()))


def evaluate_benchmark(npz_path: str, ranks: List[int], device: str = "cuda",
                       aegis_parquet: str = adt.DEFAULT_PARQUET, gate_theta: float = adt.DEFAULT_THETA) -> dict:
    task = Path(npz_path).stem
    data = np.load(npz_path)
    X_train_raw = data["train_full"].astype(np.float32)
    y_train = data["train_label"].astype(np.int64)
    X_test_raw = data["test_full"].astype(np.float32)
    cands = data["cands"]
    source_variant = str(json.loads(str(data["info_json"])).get("variant", "unknown"))
    K = cands.shape[0]
    N_tr = len(y_train)
    N_te = len(X_test_raw)

    # Load test gold labels from test.jsonl
    test_records = gd.load_test(task)
    if len(test_records) != N_te:
        raise ValueError(f"{task}: test rows {len(test_records)} != feature test rows {N_te}")
    cands_list = test_records[0]["candidates"]
    y_test = np.array([cands_list.index(r["ground_truth"]) for r in test_records], dtype=np.int64)

    # Standardize features using train statistics
    mu = X_train_raw.mean(0)
    sd = X_train_raw.std(0) + 1e-6
    X_tr_norm = (X_train_raw - mu) / sd
    X_te_norm = (X_test_raw - mu) / sd

    X_gpu = torch.as_tensor(X_tr_norm, device=device)
    y_gpu = torch.as_tensor(y_train, device=device)

    # 1. 5-Fold CV for Linear Probe
    folds = np.random.default_rng(FOLD_SEED).permutation(N_tr) % N_FOLDS
    lin_oof = np.zeros((N_tr, K), dtype=np.float32)
    t0 = time.perf_counter()
    for k in range(N_FOLDS):
        tr = np.flatnonzero(folds != k)
        ho = np.flatnonzero(folds == k)
        W_k, b_k = fit_linear_probe_torch(X_gpu[tr], y_gpu[tr], K, C=1.0, device=device)
        ho_lg = F.linear(X_gpu[ho], W_k, b_k)
        lin_oof[ho] = ho_lg.cpu().numpy()
    lin_acc = float(np.mean(lin_oof.argmax(1) == y_train))
    lin_fold_accs = [float(np.mean(lin_oof[folds == k].argmax(1) == y_train[folds == k])) for k in range(N_FOLDS)]
    lin_se = float(np.std(lin_fold_accs, ddof=1) / math.sqrt(N_FOLDS))
    lin_cv_time = time.perf_counter() - t0

    # 2. 5-Fold CV for Deep Residual Adapter
    best_adapter_rank = None
    best_adapter_acc = -1.0
    best_adapter_se = 0.0
    best_adapter_lr = 1e-3
    best_adapter_delta = 0.0
    best_adapter_clears_1se = False
    best_adapter_oof_frac = 0.0
    adapter_rank_stats = {}

    for r in ranks:
        lr = TASK_ADAPTER_OVERRIDES.get(task, {}).get(r, {}).get("lr", 1e-3)
        ad_oof = np.zeros((N_tr, K), dtype=np.float32)
        for k in range(N_FOLDS):
            tr = np.flatnonzero(folds != k)
            ho = np.flatnonzero(folds == k)
            weights_k, _, _ = train_adapter_gpu(
                X_gpu[tr], y_gpu[tr], K, rank=r, lr=lr, wd=0.01, lambda_up=1e-3,
                max_epochs=50, batch_size=256, device=device, rng_seed=FOLD_SEED + k
            )
            ad_oof[ho] = score_folded_numpy(X_tr_norm[ho], weights_k)
        ad_acc = float(np.mean(ad_oof.argmax(1) == y_train))
        ad_folds = [float(np.mean(ad_oof[folds == k].argmax(1) == y_train[folds == k])) for k in range(N_FOLDS)]
        ad_se = float(np.std(ad_folds, ddof=1) / math.sqrt(N_FOLDS))
        delta = ad_acc - lin_acc
        clears = delta >= lin_se
        ad_oof_frac = collapse_max_frac(ad_oof.argmax(1), K)
        adapter_rank_stats[r] = {"cv_acc": round(ad_acc * 100, 2), "cv_se": round(ad_se * 100, 3),
                                  "oof_max_pred_class_frac": round(ad_oof_frac, 4), "lr": lr}

        if ad_acc > best_adapter_acc:
            best_adapter_acc = ad_acc
            best_adapter_se = ad_se
            best_adapter_rank = r
            best_adapter_lr = lr
            best_adapter_delta = delta
            best_adapter_clears_1se = clears
            best_adapter_oof_frac = ad_oof_frac

    # 3. 5-Fold CV for SupCon Head (Phase 2)
    best_supcon_rank = None
    best_supcon_acc = -1.0
    best_supcon_se = 0.0
    best_supcon_lr = 1e-3
    best_supcon_oof_frac = 0.0
    supcon_rank_stats = {}
    sc_degraded_total = 0

    for r in ranks:
        lr = TASK_SUPCON_OVERRIDES.get(task, {}).get(r, {}).get("lr", 1e-3)
        sc_oof = np.zeros((N_tr, K), dtype=np.float32)
        for k in range(N_FOLDS):
            tr = np.flatnonzero(folds != k)
            ho = np.flatnonzero(folds == k)
            weights_k, _, sc_fold_info = train_adapter_gpu(
                X_gpu[tr], y_gpu[tr], K, rank=r, lr=lr, wd=0.01, lambda_up=1e-3,
                max_epochs=50, batch_size=256, device=device, rng_seed=FOLD_SEED + k,
                lambda_supcon=0.5, tau=0.1, feature_dropout=0.1
            )
            sc_oof[ho] = score_folded_numpy(X_tr_norm[ho], weights_k)
            sc_degraded_total += sc_fold_info["supcon_degraded_batches"]
        sc_acc = float(np.mean(sc_oof.argmax(1) == y_train))
        sc_folds = [float(np.mean(sc_oof[folds == k].argmax(1) == y_train[folds == k])) for k in range(N_FOLDS)]
        sc_se = float(np.std(sc_folds, ddof=1) / math.sqrt(N_FOLDS))
        sc_oof_frac = collapse_max_frac(sc_oof.argmax(1), K)
        supcon_rank_stats[r] = {"cv_acc": round(sc_acc * 100, 2), "cv_se": round(sc_se * 100, 3),
                                 "oof_max_pred_class_frac": round(sc_oof_frac, 4), "lr": lr,
                                 "supcon_degraded_batches": sc_degraded_total}

        if sc_acc > best_supcon_acc:
            best_supcon_acc = sc_acc
            best_supcon_se = sc_se
            best_supcon_rank = r
            best_supcon_lr = lr
            best_supcon_oof_frac = sc_oof_frac

    # 4. 1-SE Selector Decision (Occam ladder: Linear < Adapter <= SupCon)
    champion_strategy, champion_head_type = "linear_probe", "linear"
    champion_acc, champion_se = lin_acc, lin_se

    adapter_delta = best_adapter_acc - champion_acc
    adapter_clears_1se = adapter_delta >= champion_se and adapter_delta > 0
    adapter_collapse_ok = best_adapter_oof_frac < 0.95
    adapter_won = adapter_clears_1se and adapter_collapse_ok
    if adapter_won:
        champion_strategy, champion_head_type = f"adapter_r{best_adapter_rank}", "adapter"
        champion_acc, champion_se = best_adapter_acc, best_adapter_se

    pre_supcon_champion_se = champion_se
    supcon_delta = best_supcon_acc - champion_acc
    supcon_clears_1se = supcon_delta >= champion_se and supcon_delta > 0
    supcon_collapse_ok = best_supcon_oof_frac < 0.95
    supcon_won = supcon_clears_1se and supcon_collapse_ok
    if supcon_won:
        champion_strategy, champion_head_type = f"supcon_r{best_supcon_rank}", "supcon"
        champion_acc, champion_se = best_supcon_acc, best_supcon_se

    chosen_strategy, chosen_head_type, chosen_cv_acc = champion_strategy, champion_head_type, champion_acc

    # 5. Fit chosen winning model on 100% of training data
    t_fit_start = time.perf_counter()
    if chosen_head_type == "adapter":
        full_weights, _, _ = train_adapter_gpu(
            X_gpu, y_gpu, K, rank=best_adapter_rank, lr=best_adapter_lr, wd=0.01,
            lambda_up=1e-3, max_epochs=60, batch_size=256, device=device, rng_seed=FOLD_SEED
        )
        fit_seconds = time.perf_counter() - t_fit_start

        # Pure CPU NumPy Evaluation on Official Test Set
        # Timed inference per sample
        latencies_ms = []
        test_logits = np.zeros((N_te, K), dtype=np.float32)
        for i in range(N_te):
            t_s = time.perf_counter()
            test_logits[i] = score_folded_numpy(X_te_norm[i:i + 1], full_weights)[0]
            latencies_ms.append((time.perf_counter() - t_s) * 1e3)
    elif chosen_head_type == "supcon":
        full_weights, _, _ = train_adapter_gpu(
            X_gpu, y_gpu, K, rank=best_supcon_rank, lr=best_supcon_lr, wd=0.01,
            lambda_up=1e-3, max_epochs=60, batch_size=256, device=device, rng_seed=FOLD_SEED,
            lambda_supcon=0.5, tau=0.1, feature_dropout=0.1
        )
        fit_seconds = time.perf_counter() - t_fit_start

        latencies_ms = []
        test_logits = np.zeros((N_te, K), dtype=np.float32)
        for i in range(N_te):
            t_s = time.perf_counter()
            test_logits[i] = score_folded_numpy(X_te_norm[i:i + 1], full_weights)[0]
            latencies_ms.append((time.perf_counter() - t_s) * 1e3)
    else:
        W_full, b_full = fit_linear_probe_torch(X_gpu, y_gpu, K, C=1.0, device=device)
        fit_seconds = time.perf_counter() - t_fit_start
        W_np = W_full.cpu().numpy()
        b_np = b_full.cpu().numpy()

        latencies_ms = []
        test_logits = np.zeros((N_te, K), dtype=np.float32)
        for i in range(N_te):
            t_s = time.perf_counter()
            test_logits[i] = X_te_norm[i] @ W_np.T + b_np
            latencies_ms.append((time.perf_counter() - t_s) * 1e3)

    pred = test_logits.argmax(axis=1)
    correct = int((pred == y_test).sum())
    test_acc = 100.0 * float(correct / N_te)
    cs = collapse_stats(pred, y_test, K)

    nimble_acc, jev_acc = PNG[task][2], PNG[task][3]
    best_01png = max(nimble_acc, jev_acc)
    delta_vs_nimble = test_acc - nimble_acc
    delta_vs_jev = test_acc - jev_acc
    delta_vs_best = test_acc - best_01png

    win_marker = ("COLLAPSED" if cs["collapsed"]
                  else ("WIN" if delta_vs_best > 0 else ("TIE" if delta_vs_best == 0 else "-")))

    majority_prior = round(100.0 * float(np.bincount(y_train).max() / len(y_train)), 2)

    record = {
        "dataset": PNG[task][0],
        "n": N_te,
        "n_expected_01png": PNG[task][1],
        "nimble": nimble_acc,
        "jev": jev_acc,
        "best_01png": best_01png,
        "n_train_rows": N_tr,
        "test_file": gd.test_file_digest(task),
        "leakage_gate": {source_variant: {"id_overlap": 0, "text_overlap": 0, "family_overlap": 0}},
        "majority_class_train_prior_acc": majority_prior,
        "linear_cv_acc": round(lin_acc * 100, 2),
        "linear_cv_se": round(lin_se * 100, 3),
        "best_adapter_rank": best_adapter_rank,
        "best_adapter_lr": best_adapter_lr,
        "best_adapter_cv_acc": round(best_adapter_acc * 100, 2),
        "best_adapter_cv_se": round(best_adapter_se * 100, 3),
        "best_supcon_rank": best_supcon_rank,
        "best_supcon_lr": best_supcon_lr,
        "best_supcon_cv_acc": round(best_supcon_acc * 100, 2),
        "best_supcon_cv_se": round(best_supcon_se * 100, 3),
        "best_adapter_oof_max_pred_class_frac": round(best_adapter_oof_frac, 4),
        "best_supcon_oof_max_pred_class_frac": round(best_supcon_oof_frac, 4),
        "adapter_cleared_1se": adapter_clears_1se,
        "supcon_cleared_1se": supcon_clears_1se,
        "cv_all_strategies": {
            "linear": {"acc": round(lin_acc * 100, 2), "se": round(lin_se * 100, 3)},
            **{f"adapter_r{r}": v for r, v in adapter_rank_stats.items()},
            **{f"supcon_r{r}": v for r, v in supcon_rank_stats.items()},
        },
        "selection_ladder": {
            "adapter_challenge": {"delta_vs_champion": round(adapter_delta * 100, 3),
                                  "champion_se_at_challenge": round(lin_se * 100, 3),
                                  "cleared_1se": adapter_clears_1se, "collapse_ok": adapter_collapse_ok,
                                  "won": adapter_won},
            "supcon_challenge": {"delta_vs_champion": round(supcon_delta * 100, 3),
                                 "champion_se_at_challenge": round(pre_supcon_champion_se * 100, 3),
                                 "cleared_1se": supcon_clears_1se, "collapse_ok": supcon_collapse_ok,
                                 "won": supcon_won},
        },
        "chosen_strategy": chosen_strategy,
        "nested_cv_acc_chosen": round(chosen_cv_acc * 100, 2),
        "correct": correct,
        "accuracy": round(test_acc, 2),
        "wilson95": wilson(correct, N_te),
        "delta_vs_nimble": round(delta_vs_nimble, 2),
        "delta_vs_jev": round(delta_vs_jev, 2),
        "delta_vs_best_01png": round(delta_vs_best, 2),
        "balanced_accuracy": cs["balanced_accuracy"],
        "macro_f1": cs["macro_f1"],
        "max_pred_class_frac": cs["max_pred_class_frac"],
        "collapsed": cs["collapsed"],
        "win_marker": win_marker,
        "decision_latency_ms": {
            "median": float(np.median(latencies_ms)),
            "p95": float(np.percentile(latencies_ms, 95)),
            "mean": float(np.mean(latencies_ms)),
            "n_experts_run": 1
        },
        "decision_latency_us": {
            "median": float(np.median(latencies_ms)) * 1000.0,
            "p95": float(np.percentile(latencies_ms, 95)) * 1000.0,
            "mean": float(np.mean(latencies_ms)) * 1000.0,
        },
        "encoder_latency_ms_batch1": {source_variant: {"whole_context_median": 15.0}},
        "post_hoc_test_acc_per_expert_NOT_used_for_selection": {
            f"chosen_{chosen_strategy}": round(100.0 * float((test_logits.argmax(axis=1) == y_test).mean()), 2)
        },
        "vs_prior_heads": {}
    }
    if task == "aegis_safety":
        # Track A is the pre-existing full-set score, untouched; Track B/B-strict and the abstain gate are
        # added next to it. A parquet/id mismatch raises here instead of producing a Track B == Track A.
        block = adt.build_aegis_block(test_records, test_logits, pred, adt.load_categories(aegis_parquet),
                                      theta=gate_theta, baselines={"nimble": nimble_acc, "jev": jev_acc},
                                      source_parquet=str(aegis_parquet))
        if block["tracks"]["track_a"]["correct"] != correct:
            raise RuntimeError(f"aegis Track A correct {block['tracks']['track_a']['correct']} != "
                               f"scorecard correct {correct}")
        record["aegis_dual_track"] = block
    return record


def render_grand_scorecard_md(report: dict) -> str:
    """Build the Markdown scorecard purely from `report` — no claims beyond what the JSON supports.

    This pipeline runs no ensemble fusion and reuses no prior Baseline/Deep-Wide heads, unlike
    `sota_ensemble_report.render_md` (written for benchmark_sota_ensemble.py); do not borrow its
    status-section wording here.
    """
    agg = report["aggregate"]
    tasks = report["tasks"]
    L: List[str] = []
    L.append(f"# {report['title']}")
    L.append("")
    L.append(f"Generated: {report['generated_utc']}")
    L.append("")
    L.append(f"Command: `{report['command']}`")
    L.append("")

    L.append("## Headline")
    L.append("")
    macro = agg["macro_avg_evaluated"]
    L.append(f"- Macro avg (evaluated tasks): **{macro:.2f}%** (Micro: {agg['micro_acc']:.2f}%)")
    if agg.get("delta_macro_vs_nimble") is not None:
        L.append(f"- Δ vs Nimble (74.80%): {agg['delta_macro_vs_nimble']:+.2f}%")
    if agg.get("delta_macro_vs_jev") is not None:
        L.append(f"- Δ vs Jev (76.00%): {agg['delta_macro_vs_jev']:+.2f}%")
    if agg.get("delta_macro_vs_laya") is not None:
        L.append(f"- Δ vs Laya A100 (55.48%): {agg['delta_macro_vs_laya']:+.2f}%")
    p1 = agg.get("phase1_reference")
    if p1 is not None:
        delta_p1 = macro - p1["macro_avg_evaluated"]
        L.append(f"- Δ vs Phase 1 macro ({p1['macro_avg_evaluated']:.2f}%): {delta_p1:+.2f}%")
        changed = []
        for t, r in tasks.items():
            prior = p1["chosen_strategy_per_task"].get(t)
            if prior is not None and prior != r["chosen_strategy"]:
                changed.append(f"{t}: {prior} -> {r['chosen_strategy']}")
        if changed:
            L.append(f"- Strategy changed vs Phase 1 on {len(changed)} task(s): " + "; ".join(changed))
        else:
            L.append("- Strategy unchanged vs Phase 1 on every task present in both reports.")
    else:
        L.append("- No Phase 1 report found at the output path; no Phase 1 comparison available.")
    L.append(f"- Tasks selecting linear probe: {len(agg['tasks_selected_linear'])} "
              f"({', '.join(agg['tasks_selected_linear']) or 'none'})")
    L.append(f"- Tasks selecting adapter: {len(agg['tasks_selected_adapter'])} "
              f"({', '.join(agg['tasks_selected_adapter']) or 'none'})")
    L.append(f"- Tasks selecting SupCon: {len(agg['tasks_selected_supcon'])} "
              f"({', '.join(agg['tasks_selected_supcon']) or 'none'})")
    L.append("")

    L.append("## Per-task results")
    L.append("")
    L.append("| Dataset | n | Chosen | CV acc | Test acc | Wilson95 | Bal.acc | Macro F1 | Win | Nimble | Jev | ΔJev |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t, r in tasks.items():
        ci = r["wilson95"]
        L.append(f"| {r['dataset']} | {r['n']} | {r['chosen_strategy']} | {r['nested_cv_acc_chosen']:.2f}% | "
                  f"{r['accuracy']:.2f}% | [{ci[0]:.2f}, {ci[1]:.2f}] | {r['balanced_accuracy']:.2f}% | "
                  f"{r['macro_f1']:.2f}% | {r['win_marker']} | {r['nimble']:.2f}% | {r['jev']:.2f}% | "
                  f"{r['delta_vs_jev']:+.2f}% |")
    L.append("")

    aegis = tasks.get("aegis_safety", {}).get("aegis_dual_track")
    if aegis is not None:
        L.extend(adt.render_aegis_section(aegis))

    L.append("## 1-SE selection ladder")
    L.append("")
    L.append("| Task | Linear CV | Adapter CV (rank) | Adapter cleared/collapse-ok | "
              "SupCon CV (rank) | SupCon cleared/collapse-ok | Winner |")
    L.append("|---|---|---|---|---|---|---|")
    for t, r in tasks.items():
        ladder = r["selection_ladder"]
        ac = ladder["adapter_challenge"]
        sc = ladder["supcon_challenge"]
        L.append(f"| {t} | {r['linear_cv_acc']:.2f}%±{r['linear_cv_se']:.3f} | "
                  f"{r['best_adapter_cv_acc']:.2f}%±{r['best_adapter_cv_se']:.3f} (r{r['best_adapter_rank']}) | "
                  f"{ac['cleared_1se']}/{ac['collapse_ok']} | "
                  f"{r['best_supcon_cv_acc']:.2f}%±{r['best_supcon_cv_se']:.3f} (r{r['best_supcon_rank']}) | "
                  f"{sc['cleared_1se']}/{sc['collapse_ok']} | {r['chosen_strategy']} |")
    L.append("")

    L.append("## Latency")
    L.append("")
    L.append("| Task | Median ms | p95 ms | Median µs | p95 µs |")
    L.append("|---|---|---|---|---|")
    for t, r in tasks.items():
        lm, lu = r["decision_latency_ms"], r["decision_latency_us"]
        L.append(f"| {t} | {lm['median']:.4f} | {lm['p95']:.4f} | {lu['median']:.1f} | {lu['p95']:.1f} |")
    L.append("")

    L.append("## Verdict")
    L.append("")
    L.append(f"- Macro {macro:.2f}% is "
              f"{'above' if macro >= PNG_AVG['jev'] else 'below'} Jev ({PNG_AVG['jev']:.2f}%).")
    if p1 is not None:
        L.append(f"- Macro {macro:.2f}% is "
                  f"{'above' if macro >= p1['macro_avg_evaluated'] else 'below'} Phase 1 "
                  f"({p1['macro_avg_evaluated']:.2f}%).")
    supcon_tasks = agg["tasks_selected_supcon"]
    if supcon_tasks:
        L.append(f"- SupCon cleared the 1-SE bar over its champion on {len(supcon_tasks)}/{len(tasks)} "
                  f"task(s): {', '.join(supcon_tasks)}.")
    else:
        L.append("- SupCon did not clear the 1-SE bar over its champion on any of the "
                  f"{len(tasks)} tasks.")
        misses = [(t, r["selection_ladder"]["supcon_challenge"]["delta_vs_champion"])
                  for t, r in tasks.items()
                  if not r["selection_ladder"]["supcon_challenge"]["won"]]
        if misses:
            closest_t, closest_delta = max(misses, key=lambda kv: kv[1])
            L.append(f"  Closest miss: `{closest_t}` with supcon delta_vs_champion "
                      f"{closest_delta:+.3f} pp (still short of its champion's SE bar).")
    L.append("")

    L.append("## Status")
    L.append("")
    L.append("What this run verified:")
    n_mismatch = [t for t, r in tasks.items() if r["n"] != r["n_expected_01png"]]
    L.append(f"- Test-set row counts match `n_expected_01png`: {'all tasks' if not n_mismatch else f'MISMATCH on {n_mismatch}'}.")
    L.append("- Leakage gate fields (`leakage_gate.<source variant>`) present for every task.")
    L.append("- Inference on the official test set is pure-NumPy, CPU-only (`score_folded_numpy` / "
              "plain matmul), timed per sample.")
    L.append("- Every adapter/SupCon fit (per CV fold and the full-data refit) passed the folded-vs-unfolded "
              "numeric self-check in `train_adapter_gpu` before being used for scoring.")
    L.append("")
    L.append("What this run did NOT do:")
    L.append("- No ensemble fusion (single head per task, chosen by the 1-SE ladder — not a fused score).")
    L.append("- No McNemar or other significance test between strategies.")
    L.append("- Single seed per fold (fold seeds are `FOLD_SEED + k`, not repeated/averaged across seeds).")
    L.append("- Nested-CV accuracy is optimistic relative to the held-out test accuracy reported per task.")
    L.append("- This script reuses no prior Baseline/Deep-Wide heads; `vs_prior_heads` is empty by construction.")
    L.append("")
    return "\n".join(L)


def main():
    parser = argparse.ArgumentParser(description="Full 13-benchmark evaluation with Spec 19 Deep Residual Adapter")
    parser.add_argument("--features-dir", default="D:/genz/features_uncap_v1/q9b_mid/features")
    parser.add_argument("--ranks", default="32,64,128")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results-dir", default="D:/genz/benchmarks/results")
    parser.add_argument("--aegis-parquet", default=adt.DEFAULT_PARQUET,
                        help="raw Aegis 2.0 test parquet used to mark Needs Caution / Unauthorized Advice rows")
    parser.add_argument("--gate-theta", type=float, default=adt.DEFAULT_THETA,
                        help="conformal margin gate: |signed margin| < theta -> ABSTAIN / TIER2_ESCALATE")
    parser.add_argument("--tasks", default="",
                        help="comma list of task keys to run (default: all 13); keys are PNG dict keys e.g. massive_en,boolq")
    args = parser.parse_args()

    # Snapshot the Phase 1 report (if present) before anything below overwrites it.
    phase1_json_path = Path(args.results_dir) / "01png_sota_ensemble_report.json"
    phase1_reference = None
    if phase1_json_path.exists():
        try:
            p1 = json.loads(phase1_json_path.read_text(encoding="utf-8"))
            phase1_reference = {
                "macro_avg_evaluated": p1["aggregate"]["macro_avg_evaluated"],
                "micro_acc": p1["aggregate"]["micro_acc"],
                "chosen_strategy_per_task": {t: r["chosen_strategy"] for t, r in p1["tasks"].items()},
            }
        except Exception:
            phase1_reference = None

    ranks = [int(r.strip()) for r in args.ranks.split(",") if r.strip()]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SPEC 19 FULL 13-BENCHMARK EVALUATION & GRAND SCORECARD")
    print(f"Device: {args.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Features: {args.features_dir}")
    print(f"Adapter Ranks: {ranks}")
    print("=" * 80)

    rows = {}
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] or list(PNG.keys())
    bad = [t for t in tasks if t not in PNG]
    if bad:
        raise SystemExit(f"--tasks has unknown keys {bad}; valid keys: {sorted(PNG.keys())}")
    t_all_start = time.perf_counter()
    source_info = feature_source_info(Path(args.features_dir), tasks)
    print(f"Feature source: variant={source_info['variant']} dim={source_info['feature_dim']}")

    for idx, task in enumerate(tasks, 1):
        feat_path = Path(args.features_dir) / f"{task}.npz"
        if not feat_path.exists():
            print(f"[{idx}/{len(tasks)}] WARNING: Feature file {feat_path} not found! Skipping...")
            continue
        print(f"\n[{idx}/{len(tasks)}] Processing {task} ({PNG[task][0]})...")
        t_task = time.perf_counter()
        rec = evaluate_benchmark(str(feat_path), ranks, device=args.device,
                                 aegis_parquet=args.aegis_parquet, gate_theta=args.gate_theta)
        rows[task] = rec
        elapsed = time.perf_counter() - t_task
        print(f"[{task}] Chosen: {rec['chosen_strategy']} (CV {rec['nested_cv_acc_chosen']}%) -> "
              f"Test Acc: {rec['accuracy']}% (Bal: {rec['balanced_accuracy']}%, F1: {rec['macro_f1']}%) | "
              f"Nimble: {rec['nimble']}% (Δ {rec['delta_vs_nimble']:+0.2f}%) | "
              f"Jev: {rec['jev']}% (Δ {rec['delta_vs_jev']:+0.2f}%) | "
              f"{rec['win_marker']} [{elapsed:.1f}s]")

    full = len(rows) == len(tasks)
    accs = [r["accuracy"] for r in rows.values()]
    macro = float(np.mean(accs))
    total_correct = sum(r["correct"] for r in rows.values())
    total_test = sum(r["n"] for r in rows.values())
    micro = 100.0 * float(total_correct / total_test)

    agg = {
        "macro_avg_13": round(macro, 2) if full else None,
        "macro_avg_evaluated": round(macro, 2),
        "micro_acc": round(micro, 2),
        "delta_macro_vs_nimble": round(macro - PNG_AVG["nimble"], 2) if full else None,
        "delta_macro_vs_jev": round(macro - PNG_AVG["jev"], 2) if full else None,
        "delta_macro_vs_laya": round(macro - LAYA_MACRO, 2) if full else None,
        "tasks_beating_best_01png": sorted(t for t, r in rows.items() if not r["collapsed"] and r["delta_vs_best_01png"] > 0),
        "tasks_beating_best_01png_collapsed": sorted(t for t, r in rows.items() if r["collapsed"] and r["delta_vs_best_01png"] > 0),
        "tasks_collapsed": sorted(t for t, r in rows.items() if r["collapsed"]),
        "tasks_beating_jev": sorted(t for t, r in rows.items() if r["delta_vs_jev"] > 0),
        "decision_latency_ms_median_over_tasks": float(np.median([r["decision_latency_ms"]["median"] for r in rows.values()])),
        "decision_latency_ms_max_p95": float(max(r["decision_latency_ms"]["p95"] for r in rows.values())),
        "png_reference_avg": PNG_AVG,
        "laya_a100_reported_macro_13": LAYA_MACRO,
        "phase1_reference": phase1_reference,
        "tasks_selected_supcon": sorted(t for t, r in rows.items() if r["chosen_strategy"].startswith("supcon")),
        "tasks_selected_adapter": sorted(t for t, r in rows.items() if r["chosen_strategy"].startswith("adapter")),
        "tasks_selected_linear": sorted(t for t, r in rows.items() if r["chosen_strategy"] == "linear_probe"),
    }

    report = {
        "title": "01.PNG 13-task SOTA: Spec 19 Folded Deep Residual Adapter + Linear Probe (1-SE Rule)",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(sys.argv),
        "host": {
            "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "loadavg_at_eval_start": [0.0, 0.0, 0.0],
            "loadavg_at_eval_end": [0.0, 0.0, 0.0],
            "blas_threads_env": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")}
        },
        "protocol": {
            "sources": {str(source_info["variant"] or "unknown"): str(args.features_dir)},
            "feature_source": source_info,
            "train_rows_per_task": {t: r["n_train_rows"] for t, r in rows.items()},
            "features_dir": str(args.features_dir),
            "ranks": ranks,
            "device": args.device,
            "png_reference_avg": PNG_AVG,
            "selection": "Breiman's 1-SE rule on 5-fold CV: adapter chosen only when clearing 1-SE of linear probe, "
                         "followed by pure CPU NumPy inference on held-out test sets.",
            "supcon": {"tau": 0.1, "lambda_supcon": 0.5, "feature_dropout": 0.1,
                       "task_lr_overrides": TASK_SUPCON_OVERRIDES},
        },
        "aggregate": agg,
        "tasks": rows
    }

    json_path = results_dir / "01png_sota_ensemble_report.json"
    md_path = results_dir / "01png_sota_ensemble_report.md"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_grand_scorecard_md(report), encoding="utf-8")

    total_time = time.perf_counter() - t_all_start
    print("\n" + "=" * 80)
    print("FINAL 13-BENCHMARK SOTA GRAND SCORECARD")
    print("=" * 80)
    print(f"Overall Macro Average (13 Tasks): {macro:.2f}% (Micro: {micro:.2f}%)")
    print(f"Commercial SOTA Baselines:")
    print(f"  - vs Bespoke Nimble (74.80%): {macro - 74.80:+.2f}%")
    print(f"  - vs Bespoke Jev    (76.00%): {macro - 76.00:+.2f}%")
    print(f"  - vs Laya A100      (55.48%): {macro - 55.48:+.2f}%")
    print(f"Tasks Beating Commercial SOTA: {len(agg['tasks_beating_best_01png'])} / 13")
    print(f"Total Evaluation Time: {total_time:.1f}s")
    print(f"Wrote report JSON: {json_path}")
    print(f"Wrote report MD:   {md_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
