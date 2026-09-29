"""Experts and combiners for benchmark_sota_ensemble.py (01.PNG 13-task suite).

Every expert maps one record's *source vector* x to K candidate scores, where
x is what a frozen encoder produced for that record:

    non-pair task : x = full                      (D,)
    pair task     : x = [full; field_A; field_B]  (3D,), fields encoded separately

Experts
  LinearProbe   L2 multinomial logistic regression (C picked by inner CV on the
                training rows only) on one of three feature maps:
                  full    standardized full-context vector
                  pair    [a; b; |a - b|; a * b] of the standardized fields
                  hybrid  [full; a; b; |a - b|; a * b]
                optionally after an unsupervised PCA "manifold" (fit on the
                training rows, whitened), which is the dimension fallback for
                the small-sample tasks. Pure NumPy at inference.
  RNN heads     ParallelRNNSetAdapter ("baseline") and DeepWideRNNSetAdapter
                ("deep_wide": pair path on pair tasks, whole context elsewhere;
                "deep_wide_full": whole context on pair tasks), trained with the
                exact recipe of benchmark_01png_grand_challenge.train_one
                (copied, not imported: that file is being edited by another
                session) and run at test time by the NumPy runtimes.

Engine adapters (duck-typed, so the engines are used unmodified)
  EngineExpert     .name/.in_dim/.score(x, C) -> raw scores; used by
                   MultiModelCausalMoE (which applies its own temperatures).
  CalibratedExpert .score(x, C) -> w * log_softmax(raw / T); summed by
                   ensemble_causal_engine.score_ensemble("logits_sum") this is
                   the weighted log-linear pool sum_e w_e log P_e.
  ReplayExpert     returns a precomputed score row (cursor set per record), so
                   the out-of-fold selection runs the SAME engine code as test.
"""
from __future__ import annotations

import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

NEG = np.float32(-1e4)          # default GAP below the lowest real logit for a class absent from a training fold
DEFAULT_CS_GRID = tuple(np.logspace(-4, 2, 7).tolist())
DEFAULT_LR_N_JOBS = 4


def cs_grid(override: Optional[Sequence[float]] = None) -> List[float]:
    """LogisticRegressionCV `Cs`: explicit argument, else env GC_CS_GRID ("1e-3,1e-2,1"), else the log grid."""
    if override is not None:
        vals = [float(v) for v in override]
    elif os.environ.get("GC_CS_GRID", "").strip():
        vals = [float(v) for v in os.environ["GC_CS_GRID"].split(",") if v.strip()]
    else:
        vals = list(DEFAULT_CS_GRID)
    if not vals or any(not math.isfinite(v) or v <= 0.0 for v in vals):
        raise ValueError(f"Cs grid must be non-empty positive finite floats, got {vals!r}")
    return vals


def lr_n_jobs(override: Optional[int] = None) -> int:
    """LogisticRegressionCV worker count: explicit argument, else env GC_LR_N_JOBS, else DEFAULT_LR_N_JOBS."""
    v = override if override is not None else int(os.environ.get("GC_LR_N_JOBS", DEFAULT_LR_N_JOBS))
    if v == 0:
        raise ValueError("n_jobs must be non-zero (use -1 for all cores)")
    return int(v)


DEVICE_CHOICES = ("cpu", "cuda", "auto")


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def check_device_name(device: str) -> str:
    """Validate the spelling only ('cpu', 'cuda', 'cuda:N', 'auto'); it never touches CUDA."""
    if device in DEVICE_CHOICES or (isinstance(device, str) and device.startswith("cuda:") and device[5:].isdigit()):
        return device
    raise ValueError(f"device must be one of {DEVICE_CHOICES} or 'cuda:N', got {device!r}")


def resolve_device(device: str = "auto") -> str:
    """Fit-time device -> a concrete torch device string, 'cpu' or 'cuda[:N]'.

    'auto' picks CUDA when torch sees one, else CPU. An explicit 'cuda' on a box without CUDA
    raises: silently running the multi-hour fit on the CPU instead would hide the misconfiguration.
    Only fitting uses a device; the exported arrays and every score() stay CPU NumPy.
    """
    check_device_name(device)
    if device == "cpu":
        return "cpu"
    if cuda_available():
        return "cuda" if device == "auto" else device
    if device == "auto":
        return "cpu"
    raise RuntimeError(f"device={device!r} was requested but torch.cuda.is_available() is False")


def absent_class_logit(F_logits: np.ndarray, gap: Optional[float] = None) -> float:
    """Logit for a class absent from the training fold, scaled to the real logits: `gap` below
    min(all real logits, 0), so it stays below every reachable logit whatever their magnitude.
    `gap` defaults to env GC_NEG_GAP, else -NEG (1e4)."""
    g = float(gap if gap is not None else os.environ.get("GC_NEG_GAP", -float(NEG)))
    if not math.isfinite(g) or g <= 0.0:
        raise ValueError(f"absent-class gap must be positive and finite, got {g!r}")
    lo = float(np.min(F_logits)) if np.size(F_logits) else 0.0
    return min(lo, 0.0) - g
EPS = 1e-12


def log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    s = x - m
    return s - np.log(np.sum(np.exp(s), axis=-1, keepdims=True))


def split_source(X: np.ndarray, pair: bool) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """(N, D) or (N, 3D) source rows -> (full, a, b); a, b are None off the pair tasks."""
    if not pair:
        return X, None, None
    if X.shape[-1] % 3:
        raise ValueError(f"pair source vector width {X.shape[-1]} is not 3*D")
    d = X.shape[-1] // 3
    return X[..., :d], X[..., d:2 * d], X[..., 2 * d:]


# --------------------------------------------------- GPU solver for the linear probe

LOGREG_TOL = 1e-6              # L-BFGS stops when max |grad| of the mean-loss objective is below this
LOGREG_MAX_ITER = 3000         # same cap as the scikit-learn path
LOGREG_GRAD_OK = 1e-4          # reported as not converged above this (scikit-learn's own default tol)


def _logreg_lbfgs(F, y, n_cls: int, C: float, *, W0=None, b0=None, max_iter: int = LOGREG_MAX_ITER,
                  tol: float = LOGREG_TOL):
    """L2 logistic regression on torch tensors, the objective scikit-learn's lbfgs minimizes:

        mean_i CE(F_i W^T + b, y_i) + ||W||_F^2 / (2 C n)          (intercept not penalized)

    F (n, D) float64 and y (n,) int64 in [0, n_cls) sit on the target device. n_cls == 2 fits ONE
    logit z with softmax over [0, z], as scikit-learn's binary path does (a 2-row softmax would
    penalize the same boundary twice and change the effective C). Returns torch tensors
    (coef (m, D), icpt (m,)) with m = 1 if n_cls == 2 else n_cls, and a dict with n_iter / grad.
    """
    import torch
    import torch.nn.functional as TF
    n, D = F.shape
    m = 1 if n_cls == 2 else n_cls
    W = (torch.zeros(m, D, dtype=F.dtype, device=F.device) if W0 is None else W0.detach().clone()).requires_grad_(True)
    b = (torch.zeros(m, dtype=F.dtype, device=F.device) if b0 is None else b0.detach().clone()).requires_grad_(True)
    half_l2 = 0.5 / (C * n)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, tolerance_grad=tol, tolerance_change=1e-14,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = F @ W.T + b
        if m == 1:
            z = torch.cat([torch.zeros_like(z), z], dim=1)
        loss = TF.cross_entropy(z, y) + half_l2 * W.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    closure()
    grad = float(max(W.grad.abs().max(), b.grad.abs().max()))
    n_iter = int(opt.state[opt._params[0]].get("n_iter", 0))
    if not math.isfinite(grad) or not bool(torch.isfinite(W).all()):
        raise FloatingPointError(f"torch logistic regression diverged at C={C}")
    return W.detach(), b.detach(), {"n_iter": n_iter, "max_abs_grad": grad}


def _logreg_cv_torch(F: np.ndarray, y: np.ndarray, Cs: Sequence[float], cv, dev: str):
    """LogisticRegressionCV(scoring="neg_log_loss", refit=True) on `dev`: same folds (`cv.split`),
    mean held-out log-loss per C (first best on a tie), then one refit on all rows at the winning C.

    Returns (classes, coef, icpt, best_C, info) with coef / icpt float64 NumPy in scikit-learn's
    layout: one row for a binary problem, one per class otherwise.
    """
    import torch
    import torch.nn.functional as TF
    classes, y_idx = np.unique(y, return_inverse=True)
    n_cls = len(classes)
    if n_cls < 2:
        raise ValueError(f"need samples of at least 2 classes, got {n_cls}")
    Ft = torch.as_tensor(np.ascontiguousarray(F, dtype=np.float64), device=dev)
    yt = torch.as_tensor(y_idx.astype(np.int64), device=dev)
    scores = np.zeros((cv.get_n_splits(), len(Cs)))
    worst_grad = 0.0
    for f, (tr, te) in enumerate(cv.split(F, y)):
        tr_t, te_t = torch.as_tensor(tr, device=dev), torch.as_tensor(te, device=dev)
        Ftr, ytr, Fte, yte = Ft[tr_t], yt[tr_t], Ft[te_t], yt[te_t]
        W = b = None
        for ci, C in enumerate(Cs):                   # warm start along the C path, as scikit-learn does
            W, b, d = _logreg_lbfgs(Ftr, ytr, n_cls, C, W0=W, b0=b)
            z = Fte @ W.T + b
            if W.shape[0] == 1:
                z = torch.cat([torch.zeros_like(z), z], dim=1)
            scores[f, ci] = -float(TF.cross_entropy(z, yte))
            worst_grad = max(worst_grad, d["max_abs_grad"])
    best = int(np.argmax(scores.mean(axis=0)))
    W, b, d = _logreg_lbfgs(Ft, yt, n_cls, Cs[best])
    worst_grad = max(worst_grad, d["max_abs_grad"])
    if worst_grad > LOGREG_GRAD_OK:
        warnings.warn(f"torch logistic regression stopped with max|grad|={worst_grad:.2e} > {LOGREG_GRAD_OK:.0e} "
                      f"after {LOGREG_MAX_ITER} iterations", RuntimeWarning, stacklevel=2)
    info = {"device": dev, "solver": "torch-lbfgs", "n_iter_refit": d["n_iter"], "max_abs_grad": worst_grad,
            "cv_neg_log_loss": scores.mean(axis=0).tolist()}
    return (classes, W.cpu().numpy().astype(np.float64), b.cpu().numpy().astype(np.float64),
            float(Cs[best]), info)


# --------------------------------------------------------------- linear probe

FEATURE_MAPS = ("full", "pair", "hybrid")


class LinearProbe:
    """Standardize -> feature map -> optional whitened PCA -> linear logits. NumPy only."""

    def __init__(self, arrays: Dict[str, np.ndarray], cfg: dict) -> None:
        self.a, self.cfg = arrays, cfg
        self.K, self.pair = int(cfg["K"]), bool(cfg["pair"])
        self.in_dim = int(cfg["in_dim"])

    # feature map, shared by fit and inference so they cannot drift apart
    @staticmethod
    def _map(X: np.ndarray, pair: bool, kind: str, stats: Dict[str, np.ndarray]) -> np.ndarray:
        full, a, b = split_source(X, pair)
        zf = (full - stats["mu_full"]) / stats["sd_full"]
        if kind == "full":
            return zf
        if not pair:
            raise ValueError(f"feature map {kind!r} needs a pair task")
        za = (a - stats["mu_a"]) / stats["sd_a"]
        zb = (b - stats["mu_b"]) / stats["sd_b"]
        cross = [za, zb, np.abs(za - zb), za * zb]
        return np.concatenate(([zf] if kind == "hybrid" else []) + cross, axis=-1)

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, K: int, *, pair: bool, kind: str,
            pca_k: Optional[int], seed: int, Cs: Optional[Sequence[float]] = None,
            n_jobs: Optional[int] = None, neg_gap: Optional[float] = None,
            device: str = "cpu") -> "LinearProbe":
        """device 'cpu' (default) runs scikit-learn's LogisticRegressionCV exactly as before. 'cuda' / 'auto'
        (with CUDA) runs the same inner-CV over Cs and the same objective with a torch L-BFGS solver
        (`_logreg_lbfgs`) on the GPU. Either way the stored W, b are float32 NumPy."""
        dev = resolve_device(device)          # first: a bad --device must fail before any work
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.model_selection import StratifiedKFold
        if kind not in FEATURE_MAPS:
            raise ValueError(f"unknown feature map {kind!r}")
        X = np.asarray(X, dtype=np.float64)
        full, a, b = split_source(X, pair)
        stats = {"mu_full": full.mean(0), "sd_full": full.std(0) + 1e-6}
        if pair:
            stats.update(mu_a=a.mean(0), sd_a=a.std(0) + 1e-6, mu_b=b.mean(0), sd_b=b.std(0) + 1e-6)
        F = cls._map(X, pair, kind, stats)
        arrays = {k: v.astype(np.float32) for k, v in stats.items()}
        if pca_k:
            mu_f = F.mean(0)
            _, s, Vt = np.linalg.svd(F - mu_f, full_matrices=False)
            k = min(pca_k, Vt.shape[0] - 1)
            sd = s[:k] / math.sqrt(max(F.shape[0] - 1, 1))
            P = Vt[:k].T / (sd + 1e-6)
            F = (F - mu_f) @ P
            arrays.update(pca_mu=mu_f.astype(np.float32), pca_P=P.astype(np.float32))
        classes, counts = np.unique(y, return_counts=True)
        n_splits = int(max(2, min(4, counts.min()))) if len(classes) > 1 else 2
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        if dev == "cpu":
            clf = LogisticRegressionCV(Cs=cs_grid(Cs), cv=cv, scoring="neg_log_loss", max_iter=3000,
                                       n_jobs=lr_n_jobs(n_jobs))
            clf.fit(F, y)
            fit_classes, coef, icpt = clf.classes_, clf.coef_, clf.intercept_
            best_C, fit_info = float(np.ravel(clf.C_)[0]), {"device": "cpu", "solver": "sklearn-lbfgs"}
        else:
            fit_classes, coef, icpt, best_C, fit_info = _logreg_cv_torch(F, y, cs_grid(Cs), cv, dev)
        W = np.zeros((K, F.shape[1]), dtype=np.float64)
        if len(fit_classes) == 2 and coef.shape[0] == 1:    # binary LR: logits [0, z]
            coef = np.vstack([np.zeros_like(coef), coef])
            icpt = np.array([0.0, icpt[0]])
        # Absent-class logit follows the scale of the fitted logits on the training rows.
        b0 = np.full(K, absent_class_logit(F @ coef.T + icpt, neg_gap), dtype=np.float64)
        for row, c in enumerate(fit_classes):
            W[int(c)], b0[int(c)] = coef[row], icpt[row]
        arrays.update(W=W.astype(np.float32), b=b0.astype(np.float32))
        cfg = {"K": K, "pair": pair, "kind": kind, "pca_k": pca_k, "in_dim": int(X.shape[1]),
               "C": best_C, "feature_dim": int(F.shape[1])}
        probe = cls(arrays, cfg)
        probe.fit_info = fit_info        # not saved: the .npz layout is unchanged
        return probe

    def scores(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.shape[-1] != self.in_dim or not np.all(np.isfinite(X)):
            raise ValueError(f"LinearProbe expects finite (..., {self.in_dim}) input, got {X.shape}")
        F = self._map(X, self.pair, self.cfg["kind"], self.a)
        if "pca_P" in self.a:
            F = (F - self.a["pca_mu"]) @ self.a["pca_P"]
        return (F @ self.a["W"].T + self.a["b"]).astype(np.float32)

    def score(self, x: np.ndarray, C: np.ndarray) -> np.ndarray:
        if np.asarray(C).shape[0] != self.K:
            raise ValueError(f"LinearProbe trained on K={self.K} candidates, got {np.asarray(C).shape[0]}")
        return self.scores(x)

    def save(self, path: Path) -> None:
        np.savez(path, cfg_json=np.array(json.dumps(self.cfg)), **self.a)

    @classmethod
    def load(cls, path: Path) -> "LinearProbe":
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k != "cfg_json"}
            return cls(arrays, json.loads(str(z["cfg_json"])))


# ------------------------------------------------------------------ RNN heads
# Recipe copied from benchmark_01png_grand_challenge.py (build_model, _pca_init,
# train_one) so the out-of-fold heads are trained exactly like the reused
# full-train heads. Hyperparameters below must stay equal to that file's.

EPOCHS, PATIENCE, BATCH_TRAIN, LR, WD = 80, 10, 32, 1e-3, 0.01
EARLY_STOP_FRACTION = 0.15
TRAIN_SEED = 0
RNN_CONFIGS = ("baseline", "deep_wide", "deep_wide_full")


def build_rnn(config: str, in_dim: int):
    if config == "baseline":
        from rnn_set_adapter_torch import ParallelRNNSetAdapter
        return ParallelRNNSetAdapter(in_dim, d=256, rank=16, think_steps=6, n_heads=4, n_layers=1, dropout=0.1)
    from deep_wide_rnn_set_torch import DeepWideRNNSetAdapter
    return DeepWideRNNSetAdapter(in_dim, d=512, rank=16, think_steps=6, rnn_layers=3,
                                 rho_max_schedule=[0.95, 0.85, 0.75], n_heads=8, set_layers=2, ffn_mult=4)


def rnn_param_count(config: str, in_dim: int) -> int:
    """Trainable parameters of an RNN head, for the complexity order of the 1-SE rule."""
    return int(sum(p.numel() for p in build_rnn(config, in_dim).parameters() if p.requires_grad))


def rnn_query(config: str, X: np.ndarray, pair: bool) -> Tuple[np.ndarray, ...]:
    full, a, b = split_source(X, pair)
    if config == "deep_wide" and pair:
        return (a, b)
    return (full,)


def _pca_init(model, states: np.ndarray) -> None:
    import torch
    X = torch.as_tensor(states, dtype=torch.float32)
    mu = X.mean(0)
    Xc = X - mu
    evals, evecs = torch.linalg.eigh((Xc.T @ Xc) / (X.shape[0] - 1))
    evals, evecs = evals.flip(0)[: model.d].clamp_min(0.0), evecs.flip(1)[:, : model.d]
    eps = 1e-5 * float(evals[0].clamp_min(1e-12)) + 1e-12
    with torch.no_grad():
        model.mu.copy_(mu)
        model.W_in.copy_(evecs / torch.sqrt(evals + eps))


def _rnn_logits(model, q, C):
    import torch
    B = q[0].shape[0]
    Cb = C.unsqueeze(0).expand(B, -1, -1)
    mask = torch.ones(B, C.shape[0], dtype=torch.bool, device=C.device)
    if len(q) == 2:
        return model.forward_pair(q[0], q[1], Cb, mask)
    return model(q[0], Cb, mask)


def train_rnn(config: str, q_np: Tuple[np.ndarray, ...], y_np: np.ndarray, cands: np.ndarray,
              rng_key: Sequence[int], device: str = "auto"):
    """train_one's loop: 15% early-stop slice drawn first from rng_key, AdamW, patience 10.

    device: 'cpu', 'cuda' or 'auto' (see resolve_device). Training runs there; the model that comes
    back is on the CPU with its best-epoch weights, so export_npz and rnn_scores are unchanged.
    """
    import torch
    import torch.nn.functional as F
    dev = resolve_device(device)
    torch.manual_seed(TRAIN_SEED)
    rng = np.random.default_rng(list(rng_key))
    n = len(y_np)
    es = rng.random(n) < EARLY_STOP_FRACTION
    tr_idx, es_idx = np.flatnonzero(~es), np.flatnonzero(es)
    model = build_rnn(config, cands.shape[1])
    _pca_init(model, np.concatenate([q[tr_idx] for q in q_np] + [cands]))   # on the CPU, so init is device-independent
    model.to(dev)
    q_all = [torch.as_tensor(q, device=dev) for q in q_np]
    C = torch.as_tensor(cands, device=dev)
    y = torch.as_tensor(y_np.astype(np.int64), device=dev)
    es_t = torch.as_tensor(es_idx, device=dev)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD},
                             {"params": no_decay, "weight_decay": 0.0}], lr=LR)
    best, best_state, best_epoch, stale = (-1.0, float("inf")), None, 0, 0
    t0 = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.as_tensor(rng.permutation(tr_idx), device=dev)
        for s in range(0, len(perm), BATCH_TRAIN):
            b = perm[s:s + BATCH_TRAIN]
            loss = F.cross_entropy(_rnn_logits(model, [q[b] for q in q_all], C), y[b])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{config}: non-finite loss at epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            b = es_t
            lg = _rnn_logits(model, [q[b] for q in q_all], C)
            key = (float((lg.argmax(-1) == y[b]).float().mean()), float(F.cross_entropy(lg, y[b])))
        if (key[0], -key[1]) > (best[0], -best[1]):
            best, best_epoch, stale = key, epoch, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.to("cpu")
    model.eval()
    info = {"best_epoch": best_epoch, "early_stop_acc": best[0], "n_train": int(len(tr_idx)),
            "n_early_stop": int(len(es_idx)), "train_seconds": time.perf_counter() - t0, "device": dev}
    return model, info


def rnn_scores(model, q_np: Tuple[np.ndarray, ...], cands: np.ndarray) -> np.ndarray:
    import torch
    with torch.no_grad():
        return _rnn_logits(model, [torch.as_tensor(q) for q in q_np], torch.as_tensor(cands)).numpy()


class RNNRuntimeExpertCore:
    """NumPy runtime of a trained RNN head, fed with the source vector x."""

    def __init__(self, config: str, path: Path, pair: bool) -> None:
        from gen_zero.causal.deep_wide_rnn_set import DeepWideRNNSetRuntime
        from gen_zero.causal.rnn_set_adapter import RNNSetAdapterRuntime
        self.config, self.pair = config, pair
        self.rt = (RNNSetAdapterRuntime if config == "baseline" else DeepWideRNNSetRuntime).from_npz(path)
        self.in_dim = int(self.rt.cfg["in_dim"]) * (3 if pair else 1)

    def score(self, x: np.ndarray, C: np.ndarray) -> np.ndarray:
        q = rnn_query(self.config, np.asarray(x, dtype=np.float32), self.pair)
        return self.rt.score_pair(q[0], q[1], C) if len(q) == 2 else self.rt.score(q[0], C)


# ------------------------------------------------------------ engine adapters

class EngineExpert:
    def __init__(self, name: str, core, in_dim: int) -> None:
        self.name, self.core, self.in_dim = name, core, int(in_dim)

    def score(self, x: np.ndarray, C: np.ndarray) -> np.ndarray:
        return np.asarray(self.core.score(x, C), dtype=np.float32)


class CalibratedExpert:
    """score = w * log_softmax(raw / T): score_ensemble's logits_sum becomes sum_e w_e log P_e."""

    def __init__(self, core, T: float, w: float) -> None:
        self.core, self.T, self.w = core, float(T), float(w)

    def score(self, x: np.ndarray, C: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.core.score(x, C), dtype=np.float64)
        return (self.w * log_softmax(raw / self.T)).astype(np.float32)


class ReplayExpert:
    """Returns table[cursor]: lets out-of-fold selection run through the real engine code."""

    def __init__(self, name: str, table: np.ndarray, in_dim: int) -> None:
        self.name, self.table, self.in_dim, self.cursor = name, table, int(in_dim), 0

    def score(self, x: np.ndarray, C: np.ndarray) -> np.ndarray:
        return self.table[self.cursor]


# ------------------------------------------------------------------ combiners

def nll(logp: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(logp[np.arange(len(y)), y]))


GREEDY_TOL = 1e-4               # nats: an addition must lower the pool NLL by more than this


def greedy_pool_weights(L: Dict[str, np.ndarray], y: np.ndarray, rounds: int = 25,
                        tol: float = GREEDY_TOL) -> Dict[str, float]:
    """Caruana ensemble selection with replacement on the log-linear pool, minimizing NLL.

    Early stop: round 1 takes the best single expert; every later round adds the best
    candidate only if it lowers the pool NLL by more than `tol`, else selection ends.
    A fixed round count would keep adding the least-bad expert after the pool stopped
    improving, and that is how weak experts leaked noise into the strong one.
    `rounds` is only a cap; weights are normalized by the rounds actually taken.
    """
    names = list(L)
    counts = {n: 0 for n in names}
    acc = np.zeros_like(next(iter(L.values())))
    cur, taken = math.inf, 0
    for r in range(1, rounds + 1):
        cand = {n: nll(log_softmax((acc + L[n]) / r), y) for n in names}
        best = min(names, key=lambda n: cand[n])
        if taken and not cand[best] < cur - tol:
            break
        counts[best] += 1
        acc = acc + L[best]
        cur, taken = cand[best], r
    return {n: c / taken for n, c in counts.items() if c}


# ---------------------------------------------------- fractal / causal experts
# Label-free geometry and the MCTS System-1 engine, exposed with the same
# .score(x, C) contract as every other expert, so their out-of-fold tables
# enter the same nested-CV selection as the supervised probes.

class FractalArbitrationExpert:
    """Per-candidate race energy from BifurcatedFractalEngine, via continuous_causal_reasoning_expert.

    For record x and candidate embeddings C (K, D), in the encoder space the
    RNN heads use: q0 = the whole-context vector, standardized with training-row
    statistics (label-free) and so is every C row. For each candidate k, CCRE
    seeds a K<=4 momentum pool (orthogonal perturbations of c_k - q0), races it
    by damped leapfrog with every other candidate as a repulsor, refines the
    survivor by hard-forced affine steps, and scores -||s_k - c_k||.

    Gotcha: all of that is built from inner products and linear combinations of
    {q0, c_1..c_K}, so it is run in an orthonormal basis of their span
    (<= K+1 dims). That is exact, and it keeps the per-call cost and the
    engine's working-set budget independent of D.

    The returned row is centered over K: a relative potential difference with
    no learned parameter. Nothing here reads a label.

    Known defect of the wrapped CCRE (not of this adapter): its score is
    anti-aligned with proximity; the candidate q0 sits on gets the lowest
    score. The arbitration strategies therefore fit a signed weight.
    """

    def __init__(self, arrays: Dict[str, np.ndarray], cfg: dict) -> None:
        self.a, self.cfg = arrays, cfg
        self.pair, self.in_dim = bool(cfg["pair"]), int(cfg["in_dim"])
        self.name = "fractal"

    @classmethod
    def fit(cls, X: np.ndarray, *, pair: bool) -> "FractalArbitrationExpert":
        full, _, _ = split_source(np.asarray(X, dtype=np.float64), pair)
        arrays = {"mu": full.mean(0).astype(np.float32), "sd": (full.std(0) + 1e-6).astype(np.float32)}
        return cls(arrays, {"pair": pair, "in_dim": int(X.shape[1]), "type": "fractal"})

    @staticmethod
    def core_scores(q0: np.ndarray, C: np.ndarray, *, seed: int = 0) -> np.ndarray:
        from gen_zero.causal.continuous_causal_reasoning_expert import continuous_causal_reasoning_expert
        q0 = np.asarray(q0, dtype=np.float64)
        C = np.asarray(C, dtype=np.float64)
        if C.ndim != 2 or C.shape[0] < 2 or C.shape[1] != q0.shape[0]:
            raise ValueError(f"FractalArbitrationExpert needs (K>=2, {q0.shape[0]}) candidates, got {C.shape}")
        M = np.vstack([q0[None, :], C])
        U, s, _ = np.linalg.svd(M.T, full_matrices=False)
        B = U[:, s > 1e-9 * max(float(s[0]), 1e-300)]
        if B.shape[1] < 2:
            raise ValueError("FractalArbitrationExpert: query and candidates span fewer than 2 directions")
        res = continuous_causal_reasoning_expert(q0 @ B, C @ B, domain_prototype=None, seed=seed)
        sc = np.asarray(res.scores, dtype=np.float64)
        if not np.all(np.isfinite(sc)):
            raise FloatingPointError("FractalArbitrationExpert: non-finite CCRE score")
        return sc - sc.mean()

    def score(self, x: np.ndarray, C: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.shape[-1] != self.in_dim or not np.all(np.isfinite(x)):
            raise ValueError(f"FractalArbitrationExpert expects finite ({self.in_dim},) input, got {x.shape}")
        full, _, _ = split_source(x, self.pair)
        mu, sd = self.a["mu"].astype(np.float64), self.a["sd"].astype(np.float64)
        return self.core_scores((full - mu) / sd, (np.asarray(C, dtype=np.float64) - mu) / sd).astype(np.float32)

    def scores(self, X: np.ndarray, C: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.shape[0] == 0:
            return np.zeros((0, np.asarray(C).shape[0]), dtype=np.float32)
        return np.stack([self.score(x, C) for x in X])

    def save(self, path: Path) -> None:
        np.savez(path, cfg_json=np.array(json.dumps(self.cfg)), **self.a)

    @classmethod
    def load(cls, path: Path) -> "FractalArbitrationExpert":
        with np.load(path, allow_pickle=False) as z:
            return cls({k: z[k] for k in z.files if k != "cfg_json"}, json.loads(str(z["cfg_json"])))


class CausalMCTSExpertCore:
    """CausalMCTSRNN.classify on a trained Baseline head: log of its counterfactually adjusted policy.

    classify() runs the ACT think loop + Set block (System 1), measures how far
    the choice moves under perturbations orthogonal to span(q, C), and mixes
    toward uniform by that amount. With no latent world model there is nothing
    to search, so the mode is always "fast"; that is the engine's own contract.
    """

    def __init__(self, path: Path, pair: bool) -> None:
        from gen_zero.causal.causal_mcts_rnn import CausalMCTSRNN
        from gen_zero.causal.rnn_set_adapter import RNNSetAdapterRuntime
        self.pair = pair
        self.rt = RNNSetAdapterRuntime.from_npz(path)
        self.engine = CausalMCTSRNN(self.rt)
        self.in_dim = int(self.rt.cfg["in_dim"]) * (3 if pair else 1)

    def score(self, x: np.ndarray, C: np.ndarray) -> np.ndarray:
        full, _, _ = split_source(np.asarray(x, dtype=np.float32), self.pair)
        d = self.engine.classify(full, np.asarray(C, dtype=np.float32))
        return np.log(np.clip(d.probs, EPS, None)).astype(np.float32)


# ------------------------------------------------------- fractal arbitration

FRACTAL_SUFFIX = ":fractal"
# Signed on purpose: continuous_causal_reasoning_expert ranks the candidate the query sits
# on LOWEST (measured, see test_fractal_score_is_inverted_against_the_nearest_candidate), so
# the sign of beta is fitted on training rows and reported, never assumed.
FRACTAL_BETAS = (0.0, 0.25, -0.25, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0)
FRACTAL_MARGINS = (0.05, 0.1, 0.2, 0.35, 1.01)     # 1.01: every record passes the margin gate
FRACTAL_ENTROPIES = (0.5, 0.7, 0.9, 1.01)          # 1.01: the entropy gate never fires alone


def fractal_names(names: Sequence[str]) -> List[str]:
    return [n for n in names if n.endswith(FRACTAL_SUFFIX)]


def fractal_gate(logp: np.ndarray, tau_m: float, tau_h: float) -> np.ndarray:
    """(N, K) log-probs -> (N,) bool: top-1 minus top-2 prob below tau_m, or normalized entropy above tau_h."""
    p = np.exp(logp - logp.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    top2 = np.sort(p, axis=-1)[..., -2:]
    margin = top2[..., 1] - top2[..., 0]
    H = -np.sum(p * np.log(p + EPS), axis=-1) / math.log(p.shape[-1])
    return (margin < tau_m) | (H > tau_h)


def fractal_residual(rows: np.ndarray, scale: float) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64) / scale


def fractal_arbitrate(base_logp: np.ndarray, resid: np.ndarray, fit: dict) -> np.ndarray:
    """Add beta * residual to the base log-probs on gated records only."""
    gate = fractal_gate(base_logp, fit["tau_m"], fit["tau_h"])
    return base_logp + (fit["beta"] * gate)[..., None] * resid


def fit_fractal_gate(base_logp: np.ndarray, resid: np.ndarray, y: np.ndarray, *,
                     betas: Optional[Sequence[float]] = None, margins: Optional[Sequence[float]] = None,
                     entropies: Optional[Sequence[float]] = None) -> dict:
    """Grid over (beta, tau_m, tau_h) on these rows: max accuracy, then min NLL; beta=0 wins exact ties
    (when 0.0 is listed first in `betas`). Grids default to the FRACTAL_* module constants."""
    betas = FRACTAL_BETAS if betas is None else tuple(betas)
    margins = FRACTAL_MARGINS if margins is None else tuple(margins)
    entropies = FRACTAL_ENTROPIES if entropies is None else tuple(entropies)
    if not (betas and margins and entropies):
        raise ValueError("fractal gate grids must all be non-empty")
    best, best_key = None, None
    for beta in betas:
        for tau_m in margins:
            for tau_h in entropies:
                fit = {"beta": beta, "tau_m": tau_m, "tau_h": tau_h}
                lp = log_softmax(fractal_arbitrate(base_logp, resid, fit))
                key = (-float(np.mean(lp.argmax(1) == y)), nll(lp, y))
                if best_key is None or key < best_key:
                    best, best_key = fit, key
                if beta == 0.0:
                    break                      # the gate is irrelevant at beta=0
            if beta == 0.0:
                break
    gated = fractal_gate(base_logp, best["tau_m"], best["tau_h"])
    return dict(best, fit_acc=-best_key[0], fit_nll=best_key[1], fit_gated_frac=float(gated.mean()))
