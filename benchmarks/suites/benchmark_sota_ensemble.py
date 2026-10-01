"""01.PNG 13-task SOTA attempt: cross-fitted causal ensemble over frozen-encoder experts.

Same 3,880 test records, same leakage-gated train rows and same feature files
as benchmark_01png_grand_challenge.py. What is new:

  1. More experts per task (sota_ensemble_experts.py): L2 logistic probes on
     full / pair [a; b; |a-b|; a*b] / hybrid features, each raw or after an
     unsupervised PCA manifold, next to the existing Baseline and Deep-Wide
     RNN+Set heads, plus Deep-Wide on the whole context for the pair tasks.
  2. Any number of feature SOURCES (--source name=ART, the grand-challenge
     feature layout). A stronger encoder (Qwen3.5-9B, a 27B/26B GGUF server)
     is added by pointing a source at its feature dir; whether its experts are
     used is decided by the out-of-fold selection, per task.
  3. Cross-fitting: every expert is trained 5 times on 4/5 of TRAIN and scores
     the held-out 1/5, giving an out-of-fold (OOF) score for every train row.
     Temperatures (causal_moe_engine.calibrate_temperature), pool weights,
     the MoE router and the choice of fusion strategy are all fitted on OOF
     rows and compared by a nested 5-fold CV over those rows. Test labels are
     read once, by the final scoring line of stage_eval.
  4. Fusion runs through the unmodified engines: ensemble_causal_engine.
     score_ensemble (weighted log-linear pool, consensus_veto) and
     causal_moe_engine.MultiModelCausalMoE (dense / consensus arbitration).

Stages (resumable, artifacts under --out-art):
  fit      OOF + full-train experts -> oof/<task>/<expert>.npz, models/<task>/...
  combine  OOF-only selection       -> combine/<task>.json (+ router npz)
  eval     test once, NumPy on CPU  -> results/01png_sota_ensemble_report.{json,md}
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "python"))

import grand_challenge_data as gd  # noqa: E402
import sota_ensemble_experts as sx  # noqa: E402
import sota_enhanced_heads as eh  # noqa: E402
from gen_zero.causal.causal_moe_engine import (CausalMoERouter, MultiModelCausalMoE,  # noqa: E402
                                               calibrate_temperature)
from gen_zero.causal.ensemble_causal_engine import score_ensemble  # noqa: E402

log = logging.getLogger("sota_ensemble")
N_FOLDS, FOLD_SEED, NESTED_SEED = 5, 20260924, 20260925
DEFAULT_RAW_MAX_DIM = 48000
# Above this width a raw (non-PCA) feature map is skipped and only its PCA manifold is used.
# Precedence: --raw-max-dim, then env GC_RAW_MAX_DIM, then DEFAULT_RAW_MAX_DIM. A skip is always logged.
RAW_MAX_DIM = DEFAULT_RAW_MAX_DIM
MOE_THREADS = 1              # n_threads for MultiModelCausalMoE; main() sets it from --torch-threads
_WARNED_SKIPS: set = set()


def resolve_raw_max_dim(cli: "int | None" = None) -> int:
    v = cli if cli is not None else int(os.environ.get("GC_RAW_MAX_DIM", DEFAULT_RAW_MAX_DIM))
    if v <= 0:
        raise ValueError(f"raw max dim must be positive, got {v}")
    return v
ROUTER_RANK = 4
OUT_ART = Path(os.environ.get("SOTA_ART", "/ebs2/gen-zero-sota-ensemble"))
RESULTS = REPO / "benchmarks" / "results"
SOURCES: Dict[str, Path] = {}
RNN_SOURCES: List[str] = []
RNN_HEADS: Dict[str, Path] = {}
# Spec 19 Phase 1: ranks of the deep residual adapter registered per source (--adapter-ranks).
# Empty = off, so the expert list of an existing run does not change under it.
ADAPTER_RANKS: Tuple[int, ...] = ()
DEFAULT_ADAPTER_RANKS = (32, 64, 128)
# Spec 19 Phase 2 direction 4: ranks of the SupCon joint-training head per source (--supcon-ranks).
SUPCON_RANKS: Tuple[int, ...] = ()
DEFAULT_SUPCON_RANKS = (32, 64, 128)
# Spec 20 P3: ranks of the Formulation B folded residual adapter (frozen f0 + C GELU(Ux + a)) per
# source (--adapter-b-ranks). Spec 20 S4.3: first-round candidates 32/64; 128 is a pre-registered extension.
ADAPTER_B_RANKS: Tuple[int, ...] = ()
DEFAULT_ADAPTER_B_RANKS = (32, 64)
# Spec 19 Phase 3 direction 3: RDA / Nystrom register one expert per source, not one per rank/config
# (their hyperparameters are picked by inner CV inside fit()), so a plain on/off flag is enough.
ENABLE_RDA = False
ENABLE_NYSTROM = False
# Spec 20 P4: (qwen_source, gemma_source) of the dual-manifold expert (--dual-manifold Q,G). None = off.
# Registered ONCE per task (not per source); its wire input is the concatenation of the two sources.
DUAL_MANIFOLD: Optional[Tuple[str, str]] = None
# Per-task adapter hyperparameter overrides, from the A100 sweep in tune_spec19_adapter_a100.py:
# pubmedqa rank 128 lr 1e-3 (67.07% -> 70.80%), massive_de rank 64 lr 5e-4 (88.97% -> 89.32%),
# both past the 1-SE bound over the linear probe. Every other task/rank keeps AdapterConfig's default lr.
TASK_ADAPTER_OVERRIDES: Dict[str, Dict[int, dict]] = {
    "massive_de": {64: {"lr": 5e-4}},
    "pubmedqa": {128: {"lr": 1e-3}},
}
PRIOR_REPORT = REPO / "benchmarks" / "results" / "01png_grand_challenge_report.json"

# 01.PNG, Bespoke Labs "Nimble vs Jev on 13 public benchmarks" (name, n, Nimble %, Jev %).
PNG = {
    "massive_en": ("MASSIVE en-US", 350, 86.9, 87.4), "massive_de": ("MASSIVE de-DE", 350, 83.4, 86.9),
    "multinli": ("MultiNLI", 299, 85.3, 82.9), "pubmedqa": ("PubMedQA", 250, 75.6, 77.2),
    "vitaminc": ("VitaminC", 599, 76.6, 80.1), "boolq": ("BoolQ", 300, 86.0, 89.7),
    "squad2": ("SQuAD 2.0", 299, 80.6, 82.9), "paws": ("PAWS", 250, 82.8, 89.2),
    "civil_comments": ("Civil Comments", 300, 70.3, 81.0), "aegis_safety": ("Aegis 2.0", 250, 81.2, 80.4),
    "helpsteer2": ("HelpSteer2", 249, 39.0, 34.1), "summeval_relevance": ("SummEval relevance", 240, 49.2, 35.0),
    "summeval_consistency": ("SummEval consistency", 144, 75.7, 81.2),
}
PNG_AVG = {"nimble": 74.8, "jev": 76.0}
LAYA_MACRO = 55.48           # laya_full_13_report.json (A100), copied, not re-run


def loadavg() -> List[float]:
    try:
        return [float(x) for x in os.getloadavg()]
    except (AttributeError, OSError):
        try:
            import psutil
            return [float(x) for x in psutil.getloadavg()]
        except (ImportError, AttributeError, OSError):
            return [float("nan")] * 3


# ------------------------------------------------------------------- sources

def load_source(src: str, task: str) -> Dict[str, object]:
    with np.load(SOURCES[src] / "features" / f"{task}.npz", allow_pickle=False) as z:
        f = {k: z[k] for k in z.files}
    f["info"] = json.loads(str(f.pop("info_json")))
    pair = task in gd.PAIR_FIELDS
    for split in ("train", "test"):
        parts = [f[f"{split}_full"]] + ([f[f"{split}_a"], f[f"{split}_b"]] if pair else [])
        f[f"X_{split}"] = np.concatenate(parts, axis=1).astype(np.float32)
    return f


def load_sources(task: str) -> Dict[str, Dict[str, object]]:
    fs = {s: load_source(s, task) for s in SOURCES}
    ref = next(iter(fs.values()))
    for s, f in fs.items():
        for k in ("train_ids", "test_ids", "train_label"):
            if not np.array_equal(f[k], ref[k]):
                raise ValueError(f"{task}: source {s!r} {k} differs from {next(iter(fs))!r}; "
                                 "sources must be encoded from the same train/test rows")
    return fs


def expert_specs(task: str, fs: Dict[str, Dict[str, object]]) -> List[Tuple[str, str, dict]]:
    """(name, source, spec). Fixed before any score is seen; the same list on every task."""
    pair = task in gd.PAIR_FIELDS
    specs = []
    for s, f in fs.items():
        D = f["train_full"].shape[1]
        maps = [("full", D)] + ([("pair", 4 * D), ("hybrid", 5 * D)] if pair else [])
        for kind, width in maps:
            if width <= RAW_MAX_DIM:
                specs.append((f"{s}:lin_{kind}", s, {"type": "linear", "kind": kind, "pca_k": None}))
            elif (task, s, kind) not in _WARNED_SKIPS:
                _WARNED_SKIPS.add((task, s, kind))
                log.warning("SKIPPED expert %s:lin_%s on task %s: feature width %d > raw max dim %d "
                            "(only its PCA variant runs; raise --raw-max-dim or GC_RAW_MAX_DIM to include it)",
                            s, kind, task, width, RAW_MAX_DIM)
            specs.append((f"{s}:lin_{kind}_pca", s, {"type": "linear", "kind": kind,
                                                      "pca_k": 128 if kind == "full" else 256}))
        specs.append((f"{s}{sx.FRACTAL_SUFFIX}", s, {"type": "fractal"}))
        for r in ADAPTER_RANKS:
            spec = {"type": "adapter", "rank": int(r)}
            spec.update(TASK_ADAPTER_OVERRIDES.get(task, {}).get(r, {}))
            specs.append((f"{s}:adapter_r{r}", s, spec))
        for r in SUPCON_RANKS:
            specs.append((f"{s}:supcon_r{r}", s, {"type": "supcon", "rank": int(r)}))
        for r in ADAPTER_B_RANKS:
            specs.append((f"{s}:adapter_b_r{r}", s, {"type": "adapter_b", "rank": int(r)}))
        if ENABLE_RDA:
            specs.append((f"{s}:rda", s, {"type": "rda"}))
        if ENABLE_NYSTROM:
            specs.append((f"{s}:nystrom", s, {"type": "nystrom"}))
        if s in RNN_SOURCES:
            for config in sx.RNN_CONFIGS:
                if config == "deep_wide_full" and not pair:
                    continue          # off the pair tasks deep_wide already reads the whole context
                specs.append((f"{s}:{config}", s, {"type": "rnn", "config": config}))
            specs.append((f"{s}:baseline_mcts", s, {"type": "mcts", "config": "baseline"}))
    if DUAL_MANIFOLD is not None:
        q, g = DUAL_MANIFOLD
        if q == g or q not in fs or g not in fs:
            raise ValueError(f"--dual-manifold needs two DIFFERENT registered sources, got {DUAL_MANIFOLD!r} "
                             f"among {sorted(fs)}")
        specs.append((f"{q}+{g}:dual_manifold", q, {"type": "dual_manifold", "sources": [q, g]}))
    return specs


def head_spec(spec: dict) -> dict:
    """The config-only part of a head spec: the dual expert's `sources` is dispatcher wiring, not a knob."""
    return {k: v for k, v in spec.items() if k != "sources"}


def expert_query(spec: dict, src: str, fs: Dict[str, Dict[str, object]], split: str) -> np.ndarray:
    """The (N, in_dim) rows an expert consumes on `split` ("train" | "test"): its own source, or for
    the dual-manifold expert the concatenation [X_Q ; X_G] its score(x, C) splits again."""
    if spec["type"] == "dual_manifold":
        q, g = spec["sources"]
        return np.concatenate([fs[q][f"X_{split}"], fs[g][f"X_{split}"]], axis=1)
    return fs[src][f"X_{split}"]


def fold_ids(task: str, n: int) -> np.ndarray:
    return np.random.default_rng([FOLD_SEED, gd.TASKS.index(task)]).permutation(n) % N_FOLDS


def safe(name: str) -> str:
    return name.replace(":", "__")


# ----------------------------------------------------------------------- fit

def _fit_predict(spec: dict, task: str, f: dict, tr: np.ndarray, rows_eval: np.ndarray, X_eval: np.ndarray,
                 rng_key: List[int], device: str = "cpu", fs: Optional[Dict[str, Dict[str, object]]] = None):
    """Fit one expert on rows `tr` and score X_eval. `device` ("cpu" | "cuda" | "auto") is where the
    fit runs; the returned model and its scores are CPU NumPy whatever it is.

    The dual-manifold expert needs both sources (`fs`, the dict load_sources returns; `f` is its Qwen
    source): it fits on rows `tr` of both and scores rows `rows_eval` of both, ignoring X_eval."""
    pair, K = task in gd.PAIR_FIELDS, f["cands"].shape[0]
    X, y = f["X_train"], f["train_label"]
    if spec["type"] == "dual_manifold":
        if fs is None:
            raise ValueError("dual_manifold needs fs= (all loaded sources) in _fit_predict")
        q, g = spec["sources"]
        fq, fg = fs[q], fs[g]
        if fq is not f:
            raise ValueError(f"dual_manifold expert must be dispatched on its Qwen source {q!r}")
        m = eh.DualManifoldHead.fit(fq["X_train"][tr], fg["X_train"][tr], y[tr], K, pair=pair, seed=rng_key[-1],
                                    config=eh.head_config(head_spec(spec)), rng_key=rng_key, device=device)
        s = m.scores(fq["X_train"][rows_eval], fg["X_train"][rows_eval])
        return m, s, dict(m.info, params=m.supervised_param_count())
    if spec["type"] == "linear":
        m = sx.LinearProbe.fit(X[tr], y[tr], K, pair=pair, kind=spec["kind"], pca_k=spec["pca_k"],
                               seed=rng_key[-1], device=device)
        return m, m.scores(X_eval), {"C": m.cfg["C"], "feature_dim": m.cfg["feature_dim"]}
    if spec["type"] in eh.HEADS:
        m = eh.HEADS[spec["type"]].fit(X[tr], y[tr], K, pair=pair, seed=rng_key[-1],
                                       config=eh.head_config(spec), rng_key=rng_key,
                                       device=device)
        return m, m.scores(X_eval), dict(m.info, params=m.supervised_param_count())
    if spec["type"] == "fractal":
        # Label-free: only the standardization statistics come from the rows in `tr`.
        m = sx.FractalArbitrationExpert.fit(X[tr], pair=pair)
        return m, m.scores(X_eval, f["cands"]), {"label_free": True}
    if spec["type"] == "mcts":
        # Fold-correct: the Baseline head is retrained on `tr` with the Baseline expert's own
        # rng_key, exported, and CausalMCTSRNN scores only the held-out rows. Scoring train rows
        # with the reused full-train head would be in-sample and leak into the selection.
        import tempfile
        model, info = sx.train_rnn(spec["config"], sx.rnn_query(spec["config"], X[tr], pair), y[tr], f["cands"],
                                   rng_key, device=device)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fold_head.npz"
            model.export_npz(path, {"task": task, "config": spec["config"], "fold_rng_key": list(rng_key)})
            core = sx.CausalMCTSExpertCore(path, pair)
        return core, np.stack([core.score(x, f["cands"]) for x in X_eval]), info
    model, info = sx.train_rnn(spec["config"], sx.rnn_query(spec["config"], X[tr], pair), y[tr], f["cands"],
                               rng_key, device=device)
    return model, sx.rnn_scores(model, sx.rnn_query(spec["config"], X_eval, pair), f["cands"]), info


def head_rows_match(head: Path, n: int) -> bool:
    """A reused head must have been trained on the same n train rows as the features.

    After GC_N_TRAIN grows, an old head sits on a prefix of the new rows and would pass
    every other check. Returns False when the head has no _train.json to check against.
    """
    log = head.with_name(head.stem + "_train.json")
    if not log.exists():
        return False
    meta = json.loads(log.read_text(encoding="utf-8"))
    rows = int(meta["n_train"]) + int(meta["n_early_stop"])
    if rows != n:
        raise ValueError(f"{head} was trained on {rows} rows, the features have {n}: re-run its train stage")
    return True


def stage_fit(tasks: List[str], only: List[str], device: str = "cpu") -> None:
    for task in tasks:
        fs = load_sources(task)
        ti = gd.TASKS.index(task)
        for name, src, spec in expert_specs(task, fs):
            if only and not any(o in name for o in only):
                continue
            f = fs[src]
            y, n = f["train_label"], len(f["train_label"])
            odir, mdir = OUT_ART / "oof" / task, OUT_ART / "models" / task
            odir.mkdir(parents=True, exist_ok=True)
            mdir.mkdir(parents=True, exist_ok=True)
            opath = odir / f"{safe(name)}.npz"
            if opath.exists():
                print(f"[fit] {task}/{name}: cached", flush=True)
                continue
            t0 = time.perf_counter()
            folds = fold_ids(task, n)
            oof = np.zeros((n, f["cands"].shape[0]), dtype=np.float32)
            fold_info = []
            for k in range(N_FOLDS):
                tr, ho = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
                _, s, info = _fit_predict(spec, task, f, tr, ho, f["X_train"][ho], [sx.TRAIN_SEED, ti, 100 + k],
                                       device=device, fs=fs)
                oof[ho] = s
                fold_info.append(info)
            # Full-train model for test. The Baseline / Deep-Wide heads of the grand challenge
            # were trained by the same recipe on the same rows: reuse them instead of retraining.
            full_info: dict = {}
            if spec["type"] in ("linear", "fractal") or spec["type"] in eh.HEADS:
                m, _, full_info = _fit_predict(spec, task, f, np.arange(n), np.arange(0), f["X_train"][:0],
                                               [sx.TRAIN_SEED, ti, 0], device=device, fs=fs)
                m.save(mdir / f"{safe(name)}.npz")
            elif spec["config"] in ("baseline", "deep_wide"):
                head_dir = RNN_HEADS.get(src) or (SOURCES[src] / "heads")
                head = head_dir / f"{task}_{spec['config']}.npz"
                if not head.exists():
                    raise FileNotFoundError(f"{head}: the reused grand-challenge head is missing")
                full_info = {"reused_head": str(head), "head_rows_checked": head_rows_match(head, n)}
            else:
                model, full_info = sx.train_rnn(spec["config"], sx.rnn_query(spec["config"], f["X_train"], True),
                                                y, f["cands"], [sx.TRAIN_SEED, ti], device=device)
                model.export_npz(mdir / f"{safe(name)}.npz", {"task": task, "config": spec["config"],
                                                              "encoder": f["info"].get("encoder")})
            acc = float(np.mean(oof.argmax(1) == y))
            meta = {"name": name, "source": src, "spec": spec, "oof_acc": acc, "folds": fold_info,
                    "full": full_info, "device": device, "seconds": time.perf_counter() - t0}
            np.savez(opath, oof=oof, meta_json=np.array(json.dumps(meta, default=float)))
            print(f"[fit] {task}/{name}: oof_acc {acc:.3f} {meta['seconds']:.0f}s", flush=True)


def load_oof(task: str, names: List[str]) -> Dict[str, Tuple[np.ndarray, dict]]:
    out = {}
    for n in names:
        p = OUT_ART / "oof" / task / f"{safe(n)}.npz"
        if not p.exists():
            raise FileNotFoundError(f"{p}: run the fit stage for {task}/{n} first")
        with np.load(p, allow_pickle=False) as z:
            out[n] = (z["oof"], json.loads(str(z["meta_json"])))
    return out


# ------------------------------------------------------------------- combine

# The fractal strategies come last: a fit whose residual weight is 0 ties its base
# strategy, and stage_combine resolves ties to the earlier entry.
FRACTAL_STRATEGIES = ("fractal_tiebreak", "moe_fractal_consensus")
STRATEGIES_LEARNED = ("pool_all", "pool_greedy", "veto_greedy", "moe_dense", "moe_consensus") + FRACTAL_STRATEGIES


# Complexity order of the 1-SE rule, fixed a priori and the same on every task: any single
# expert is simpler than any fusion; fusions rank by how much they fit on top of the experts.
# Spec 19 S6.5 gate 2: the majority-class reference. It outputs the class prior of the rows it
# was fitted on, so "the head only learned the prior" shows up as a named, selectable strategy.
PRIOR_STRATEGY = "single:prior"
LEARNED_TIER = {"pool_greedy": 1, "veto_greedy": 1, "pool_all": 2, "moe_dense": 3, "moe_consensus": 3,
                "fractal_tiebreak": 4, "moe_fractal_consensus": 4}


def strategy_complexity(st: str, specs: Dict[str, dict], meta: Dict[str, dict], K: int,
                        cand_dim: Dict[str, int]) -> Tuple[int, int]:
    """(tier, supervised parameters). Unsupervised maps (standardization, PCA) are not counted."""
    if st == PRIOR_STRATEGY:
        return 0, K - 1                          # Spec 19 S6.5 gate 2: K-1 free class frequencies
    if not st.startswith("single:"):
        return LEARNED_TIER[st], 0
    name = st[7:]
    spec = specs[name]
    if spec["type"] == "linear":
        return 0, K * (1 + max(int(f["feature_dim"]) for f in meta[name]["folds"]))
    if spec["type"] in eh.HEADS:
        return 0, max(int(f["params"]) for f in meta[name]["folds"])
    if spec["type"] == "fractal":
        return 0, 0                              # label-free: no supervised parameter
    return 0, sx.rnn_param_count(spec["config"], cand_dim[name])     # rnn and mcts heads


def select_one_se(strategies: List[str], fold_acc: Dict[str, np.ndarray],
                  complexity: Dict[str, Tuple[int, int]]) -> Tuple[str, dict]:
    """Breiman's one-standard-error rule over nested-CV fold accuracies.

    top = best mean accuracy; SE = std(top's 5 fold accuracies, ddof=1) / sqrt(5). The band is
    every strategy within 1 SE of top; the least complex one in the band wins, then the higher
    mean accuracy, then the earlier entry of `strategies`. A fusion that beats a single expert
    by less than 1 SE is noise at this sample size, and on test it paid for its extra fit.
    """
    mean = {s: float(np.mean(fold_acc[s])) for s in strategies}
    top = max(strategies, key=lambda s: (mean[s], -strategies.index(s)))
    se = float(np.std(fold_acc[top], ddof=1) / np.sqrt(len(fold_acc[top])))
    band = [s for s in strategies if mean[s] >= mean[top] - se - 1e-12]
    chosen = min(band, key=lambda s: (complexity[s], -mean[s], strategies.index(s)))
    return chosen, {"rule": "one_standard_error", "top": top, "top_acc": round(100 * mean[top], 2),
                    "se": round(100 * se, 3), "threshold": round(100 * (mean[top] - se), 3),
                    "band": sorted(band, key=strategies.index), "chosen": chosen,
                    "complexity": {s: list(complexity[s]) for s in band}}


def fit_temperatures(S: Dict[str, np.ndarray], y: np.ndarray, rows: np.ndarray) -> Dict[str, float]:
    return {n: calibrate_temperature(list(S[n][rows]), list(y[rows])) for n in S}


def fit_strategy(strategy: str, S: Dict[str, np.ndarray], Xq: Dict[str, np.ndarray], y: np.ndarray,
                 rows: np.ndarray, T_all: Dict[str, float]) -> dict:
    """Everything a strategy learns, learned from `rows` only; T_all was fitted on the same rows."""
    if strategy == PRIOR_STRATEGY:
        K = next(iter(S.values())).shape[1]
        return {"T": {}, "weights": {}, "prior": class_prior(y[rows], K)}
    names = list(S) if not strategy.startswith("single:") else [strategy[7:]]
    T = {n: T_all[n] for n in names}
    if strategy.startswith("single:"):
        return {"T": T, "weights": {names[0]: 1.0}}
    if strategy in FRACTAL_STRATEGIES:
        return fit_fractal_strategy(strategy, S, Xq, y, rows, T_all)
    # A fractal row is a centered potential difference, not a log-likelihood: it enters
    # only through single:<name> and the fractal strategies, never the legacy pools/router.
    names = [n for n in names if n not in sx.fractal_names(names)]
    T = {n: T_all[n] for n in names}
    L = {n: sx.log_softmax(S[n][rows] / T[n]) for n in names}
    if strategy == "pool_all":
        return {"T": T, "weights": {n: 1.0 / len(names) for n in names}}
    if strategy in ("pool_greedy", "veto_greedy"):
        return {"T": T, "weights": sx.greedy_pool_weights(L, y[rows])}
    router = CausalMoERouter.fit_projections(names, {n: Xq[n][rows] for n in names}, rank=ROUTER_RANK)
    router.temperatures = np.array([T[n] for n in names], dtype=np.float64)
    phis = np.stack([router.features({n: Xq[n][i] for n in names}) for i in rows])
    P_true = np.stack([np.exp(L[n][np.arange(len(rows)), y[rows]]) for n in names], axis=1)
    router.fit(phis, P_true)
    return {"T": T, "weights": None, "router": router}


def class_prior(y: np.ndarray, K: int) -> List[float]:
    """Class frequencies of the given labels; argmax is the majority class (ties -> lower index)."""
    return (np.bincount(np.asarray(y, dtype=np.int64), minlength=K) / max(len(y), 1)).tolist()


def fit_fractal_strategy(strategy: str, S: Dict[str, np.ndarray], Xq: Dict[str, np.ndarray], y: np.ndarray,
                         rows: np.ndarray, T_all: Dict[str, float]) -> dict:
    """Base fusion over the non-fractal experts, then (beta, tau_m, tau_h) of the gated residual, all on `rows`."""
    fr = sx.fractal_names(list(S))
    if not fr:
        raise ValueError(f"{strategy} needs a '<source>{sx.FRACTAL_SUFFIX}' expert, none among {sorted(S)}")
    base = [n for n in S if n not in fr]
    if not base:
        raise ValueError(f"{strategy} needs at least one non-fractal expert to arbitrate")
    base_strategy = "pool_greedy" if strategy == "fractal_tiebreak" else "moe_consensus"
    fit = fit_strategy(base_strategy, {n: S[n] for n in base}, {n: Xq[n] for n in base}, y, rows, T_all)
    # A fractal row is a centered, label-free potential difference; its spread on these
    # rows sets the unit of beta. An all-zero table stays all zero (beta then has no effect).
    spread = float(np.std(np.concatenate([S[n][rows] for n in fr], axis=0)))
    scale = spread if spread > 0.0 else 1.0
    K = S[base[0]].shape[1]
    base_logp = replay_rows(lambda f, e, i: fractal_base_logp(strategy, f, e, i), fit,
                            {n: S[n] for n in base}, {n: Xq[n] for n in base}, rows, K, moe=strategy.startswith("moe_"))
    resid = sx.fractal_residual(np.mean([S[n][rows] for n in fr], axis=0), scale)
    gate = sx.fit_fractal_gate(base_logp, resid, y[rows])
    return dict(fit, fractal=dict(gate, names=fr, scale=scale, base_strategy=base_strategy))


def fractal_base_logp(strategy: str, fit: dict, experts: Dict[str, object], inputs: Dict[str, Tuple]) -> np.ndarray:
    """Log-probs of the base fusion (non-fractal experts only) for one record."""
    if strategy == "moe_fractal_consensus":
        res = fit["moe"].infer({n: inputs[n] for n in fit["router"].names}, "consensus")
        return np.log(np.clip(np.asarray(res.probs, dtype=np.float64), sx.EPS, None))
    w = fit["weights"]
    adapters = {n: sx.CalibratedExpert(experts[n], fit["T"][n], w[n]) for n in w}
    res = score_ensemble(adapters, {n: {"query": inputs[n][0], "candidates": inputs[n][1]} for n in w}, "logits_sum")
    return sx.log_softmax(np.asarray(res["fused_logits"], dtype=np.float64))


def apply_strategy(strategy: str, fit: dict, experts: Dict[str, object], inputs: Dict[str, Tuple]) -> int:
    """One record through the engines. experts: name -> object with .score(x, C)."""
    if strategy in FRACTAL_STRATEGIES:
        fr = fit["fractal"]
        base = fractal_base_logp(strategy, fit, experts, inputs)
        raw = np.mean([np.asarray(experts[n].score(*inputs[n]), dtype=np.float64) for n in fr["names"]], axis=0)
        return int(np.argmax(sx.fractal_arbitrate(base, sx.fractal_residual(raw, fr["scale"]), fr)))
    if strategy == PRIOR_STRATEGY:
        return int(np.argmax(fit["prior"]))
    if strategy.startswith("single:"):
        n = strategy[7:]
        return int(np.argmax(experts[n].score(*inputs[n])))
    if strategy.startswith("moe_"):
        moe = fit["moe"]
        return moe.infer(inputs, "consensus" if strategy == "moe_consensus" else "dense").choice
    w = fit["weights"]
    adapters = {n: sx.CalibratedExpert(experts[n], fit["T"][n], w[n]) for n in w}
    res = score_ensemble(adapters, {n: {"query": inputs[n][0], "candidates": inputs[n][1]} for n in w},
                         "consensus_veto" if strategy == "veto_greedy" else "logits_sum")
    return int(res["pred"])


def build_moe(fit: dict, experts: Dict[str, object]) -> MultiModelCausalMoE:
    router = fit["router"]
    return MultiModelCausalMoE([sx.EngineExpert(n, experts[n], experts[n].in_dim) for n in router.names],
                               router, n_threads=MOE_THREADS, blas_threads=None)


def replay_rows(fn, fit: dict, S: Dict[str, np.ndarray], Xq: Dict[str, np.ndarray], rows: np.ndarray, K: int,
                *, moe: bool) -> np.ndarray:
    """fn(fit, experts, inputs) on every row, with ReplayExperts standing in for the trained experts."""
    names = list(S)
    rep = {n: sx.ReplayExpert(n, S[n], Xq[n].shape[1]) for n in names}
    if moe:
        fit = dict(fit, moe=build_moe(fit, rep))
    C = np.zeros((K, 1), dtype=np.float32)     # replay experts ignore C; score_ensemble needs K rows
    out = []
    for i in rows:
        for r in rep.values():
            r.cursor = i
        out.append(fn(fit, rep, {n: (Xq[n][i], C) for n in names}))
    return np.asarray(out)


def replay_eval(strategy: str, fit: dict, S: Dict[str, np.ndarray], Xq: Dict[str, np.ndarray],
                rows: np.ndarray, K: int) -> np.ndarray:
    return replay_rows(lambda f, e, i: apply_strategy(strategy, f, e, i), fit, S, Xq, rows, K,
                       moe=strategy.startswith("moe_")).astype(np.int64).reshape(len(rows))


def stage_combine(tasks: List[str], learned: Tuple[str, ...] = STRATEGIES_LEARNED) -> None:
    for task in tasks:
        fs = load_sources(task)
        specs = expert_specs(task, fs)
        names = [n for n, _, _ in specs if (OUT_ART / "oof" / task / f"{safe(n)}.npz").exists()]
        if not names:
            raise FileNotFoundError(f"no fitted experts found in {OUT_ART / 'oof' / task}")
        specs = [s for s in specs if s[0] in names]
        oof = load_oof(task, names)
        S = {n: oof[n][0].astype(np.float64) for n in names}
        Xq = {n: expert_query(sp, src, fs, "train") for n, src, sp in specs}
        y = next(iter(fs.values()))["train_label"]
        K, n = S[names[0]].shape[1], len(y)
        strategies = [PRIOR_STRATEGY] + [f"single:{m}" for m in names] + list(learned)
        folds = np.random.default_rng([NESTED_SEED, gd.TASKS.index(task)]).permutation(n) % N_FOLDS
        table, fold_acc = {}, {}
        t0 = time.perf_counter()
        T_fold = {k: fit_temperatures(S, y, np.flatnonzero(folds != k)) for k in range(N_FOLDS)}
        for st in strategies:
            correct = np.zeros(n, dtype=bool)
            for k in range(N_FOLDS):
                fit_rows, ev = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
                fit = fit_strategy(st, S, Xq, y, fit_rows, T_fold[k])
                correct[ev] = replay_eval(st, fit, S, Xq, ev, K) == y[ev]
            table[st] = round(100 * float(correct.mean()), 2)
            fold_acc[st] = np.array([correct[folds == k].mean() for k in range(N_FOLDS)])
        cand_dim = {nm: fs[src]["cands"].shape[1] for nm, src, _ in specs}
        complexity = {st: strategy_complexity(st, {nm: sp for nm, _, sp in specs}, {m: oof[m][1] for m in names},
                                              K, cand_dim) for st in strategies}
        chosen, selection = select_one_se(strategies, fold_acc, complexity)
        final = fit_strategy(chosen, S, Xq, y, np.arange(n), fit_temperatures(S, y, np.arange(n)))
        cdir = OUT_ART / "combine"
        cdir.mkdir(parents=True, exist_ok=True)
        rec = {"task": task, "chosen": chosen, "nested_cv_acc": table, "selection": selection,
               "nested_cv_fold_acc": {s: [round(100 * float(a), 2) for a in v] for s, v in fold_acc.items()},
               "n_train_rows": n,
               "temperatures": final["T"], "weights": final["weights"],
               "oof_acc_single": {m: round(100 * oof[m][1]["oof_acc"], 2) for m in names},
               "seconds": round(time.perf_counter() - t0, 1)}
        if "prior" in final:
            rec["prior"] = final["prior"]
        if "fractal" in final:
            rec["fractal"] = final["fractal"]
        if "router" in final:
            final["router"].save_npz(cdir / f"{task}_router.npz")
            rec["router_npz"] = str(cdir / f"{task}_router.npz")
        (cdir / f"{task}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"[combine] {task}: chosen {chosen} nested-cv {table[chosen]:.2f} (top {selection['top']} "
              f"{selection['top_acc']:.2f}, 1-SE {selection['se']:.2f}, band {len(selection['band'])}; "
              f"best single {max(v for k, v in table.items() if k.startswith('single:')):.2f})", flush=True)


# ---------------------------------------------------------------------- eval

def load_expert(task: str, name: str, src: str, spec: dict, fs: dict):
    pair = task in gd.PAIR_FIELDS
    if spec["type"] == "linear":
        return sx.LinearProbe.load(OUT_ART / "models" / task / f"{safe(name)}.npz")
    if spec["type"] in eh.HEADS:
        return eh.HEADS[spec["type"]].load(OUT_ART / "models" / task / f"{safe(name)}.npz")
    if spec["type"] == "fractal":
        return sx.FractalArbitrationExpert.load(OUT_ART / "models" / task / f"{safe(name)}.npz")
    head_dir = RNN_HEADS.get(src) or (SOURCES[src] / "heads")
    if spec["type"] == "mcts":
        return sx.CausalMCTSExpertCore(head_dir / f"{task}_{spec['config']}.npz", pair)
    path = (head_dir / f"{task}_{spec['config']}.npz" if spec["config"] in ("baseline", "deep_wide")
            else OUT_ART / "models" / task / f"{safe(name)}.npz")
    return sx.RNNRuntimeExpertCore(spec["config"], path, pair)


def wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (c - h), 2), round(100 * (c + h), 2)]


def mcnemar_exact(a_ok: np.ndarray, b_ok: np.ndarray) -> Dict[str, float]:
    from math import comb
    b01, b10 = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    m = b01 + b10
    p = 1.0 if m == 0 else min(1.0, 2 * sum(comb(m, i) for i in range(min(b01, b10) + 1)) / 2 ** m)
    return {"only_ensemble_correct": b01, "only_reference_correct": b10, "p_value": p}


def collapse_stats(pred: np.ndarray, gold: np.ndarray, K: int) -> Dict[str, object]:
    """Spec 19 S6.5 honesty gate: majority-class collapse flag, balanced accuracy, macro F1.

    collapsed = the most-predicted class is > 95% of all predictions (K >= 2 only: with a
    single candidate class every prediction is "the majority class" by construction, which is
    not a collapse). balanced_accuracy is the unweighted mean of per-class recall (classes
    present in `gold` only, sklearn's definition): a head that always predicts the train
    majority class scores near-chance here even when raw accuracy looks like a win. macro_f1
    is the unweighted mean of per-class F1 over all K candidate classes (zero for a class with
    no predicted and no gold rows), so a class the head never predicts still pulls the average
    down. Plain NumPy (matches sklearn.metrics.balanced_accuracy_score / f1_score(average=
    "macro", labels=range(K), zero_division=0) exactly); stage_eval must not gain a hard
    dependency on sklearn, which is only otherwise needed by the fit stage's LinearProbe.
    """
    pred_counts, gold_counts = np.bincount(pred, minlength=K), np.bincount(gold, minlength=K)
    max_pred_class_frac = float(pred_counts.max() / pred_counts.sum())
    collapsed = bool(K >= 2 and max_pred_class_frac > 0.95)
    recalls = [float(np.mean(pred[gold == c] == c)) for c in np.unique(gold)]
    balanced_acc = 100 * float(np.mean(recalls))
    f1s = []
    for c in range(K):
        tp = int(np.sum((pred == c) & (gold == c)))
        fp = int(np.sum((pred == c) & (gold != c)))
        fn = int(np.sum((pred != c) & (gold == c)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    macro_f1 = 100 * float(np.mean(f1s))
    return {"pred_counts": pred_counts.tolist(), "gold_counts": gold_counts.tolist(),
            "max_pred_class_frac": round(max_pred_class_frac, 4), "collapsed": collapsed,
            "balanced_accuracy": round(balanced_acc, 2), "macro_f1": round(macro_f1, 2)}


def stage_eval(tasks: List[str]) -> None:
    la0 = loadavg()
    prior = json.loads(PRIOR_REPORT.read_text(encoding="utf-8"))["tasks"] if PRIOR_REPORT.exists() else {}
    rows = {}
    for task in tasks:
        fs = load_sources(task)
        specs = expert_specs(task, fs)
        comb = json.loads((OUT_ART / "combine" / f"{task}.json").read_text(encoding="utf-8"))
        specs = [s for s in expert_specs(task, fs) if s[0] in comb["oof_acc_single"]]
        experts = {n: load_expert(task, n, s, sp, fs) for n, s, sp in specs}
        test = gd.load_test(task)
        ref = next(iter(fs.values()))
        if [r["id"] for r in test] != ref["test_ids"].tolist():
            raise ValueError(f"{task}: feature rows do not match the test file order")
        X_test = {n: expert_query(sp, s, fs, "test") for n, s, sp in specs}
        inputs_all = [{n: (X_test[n][i], fs[s]["cands"]) for n, s, _ in specs} for i in range(len(test))]
        chosen = comb["chosen"]
        fit = {"T": comb["temperatures"], "weights": comb["weights"]}
        if chosen == PRIOR_STRATEGY:
            fit["prior"] = comb["prior"]
        if chosen in FRACTAL_STRATEGIES:
            fit["fractal"] = comb["fractal"]
        if chosen.startswith("moe_"):
            fit["router"] = CausalMoERouter.from_npz(comb["router_npz"])
            fit["moe"] = build_moe(fit, experts)
        pred, ms = np.empty(len(test), dtype=np.int64), np.empty(len(test))
        for i, inp in enumerate(inputs_all):             # the timed decision: every expert + fusion
            t0 = time.perf_counter()
            pred[i] = apply_strategy(chosen, fit, experts, inp)
            ms[i] = (time.perf_counter() - t0) * 1e3
        expert_pred = {n: np.array([int(np.argmax(experts[n].score(*inp[n]))) for inp in inputs_all])
                       for n in experts}
        # ---- the ONLY place test labels are read ----
        cands = test[0]["candidates"]
        gold = np.array([cands.index(r["ground_truth"]) for r in test])
        ok = pred == gold
        acc = 100 * float(ok.mean())
        cs = collapse_stats(pred, gold, len(cands))
        win_marker = ("COLLAPSED(majority_collapse_win_excluded)" if cs["collapsed"]
                      else ("WIN" if acc - max(PNG[task][2:4]) > 0 else "-"))
        exp_acc = {n: round(100 * float(np.mean(p == gold)), 2) for n, p in expert_pred.items()}
        refs = {}
        for cfg in ("baseline", "deep_wide"):
            ref_src = RNN_SOURCES[0] if RNN_SOURCES else "q05"
            n = f"{ref_src}:{cfg}"
            if n in expert_pred:
                prior_acc = prior.get(task, {}).get("configs", {}).get(cfg, {}).get("accuracy")
                refs[cfg] = {"expert": n, "accuracy": exp_acc[n], "prior_report_accuracy": prior_acc,
                             "reproduces_prior_report": prior_acc is not None and abs(exp_acc[n] - prior_acc) < 1e-6,
                             "mcnemar": mcnemar_exact(ok, expert_pred[n] == gold),
                             "delta": round(acc - exp_acc[n], 2)}
        prior_row = prior.get(task, {})
        rows[task] = {
            "dataset": PNG[task][0], "n": len(test), "n_expected_01png": PNG[task][1],
            "nimble": PNG[task][2], "jev": PNG[task][3], "best_01png": max(PNG[task][2:4]),
            "laya_a100_reported": prior_row.get("laya_a100_reported"),
            "majority_class_train_prior_acc": prior_row.get("majority_class_train_prior_acc"),
            "test_file": gd.test_file_digest(task),
            "leakage_gate": {s: f["info"]["gate"] for s, f in fs.items()},
            "n_train_rows": int(len(ref["train_label"])),
            "chosen_strategy": chosen, "nested_cv_acc_chosen": comb["nested_cv_acc"][chosen],
            "correct": int(ok.sum()), "accuracy": round(acc, 2), "wilson95": wilson(int(ok.sum()), len(ok)),
            "delta_vs_nimble": round(acc - PNG[task][2], 2), "delta_vs_jev": round(acc - PNG[task][3], 2),
            "delta_vs_best_01png": round(acc - max(PNG[task][2:4]), 2),
            "balanced_accuracy": cs["balanced_accuracy"], "macro_f1": cs["macro_f1"],
            "max_pred_class_frac": cs["max_pred_class_frac"], "collapsed": cs["collapsed"],
            "win_marker": win_marker,
            "vs_prior_heads": refs,
            "decision_latency_ms": {"median": float(np.median(ms)), "p95": float(np.percentile(ms, 95)),
                                    "mean": float(ms.mean()),
                                    "n_experts_run": n_experts_run(chosen, comb, len(experts))},
            "encoder_latency_ms_batch1": {s: {"whole_context_median": float(np.median(f["info"]["latency_full_ms"]))
                                              if "latency_full_ms" in f["info"] else None,
                                              "encoder": f["info"].get("encoder")} for s, f in fs.items()},
            "post_hoc_test_acc_per_expert_NOT_used_for_selection": exp_acc,
            "nested_cv_acc_all_strategies": comb["nested_cv_acc"],
            "pred_counts": cs["pred_counts"], "gold_counts": cs["gold_counts"],
        }
        print(f"[eval] {task}: {chosen} -> {acc:.2f} (baseline {refs.get('baseline', {}).get('accuracy')}) "
              f"best01png {max(PNG[task][2:4])}", flush=True)
    report = build_report(rows, la0)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "01png_sota_ensemble_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    from sota_ensemble_report import render_md
    (RESULTS / "01png_sota_ensemble_report.md").write_text(render_md(report), encoding="utf-8")
    print(f"[eval] wrote {RESULTS / '01png_sota_ensemble_report.json'}", flush=True)


def n_experts_run(chosen: str, comb: dict, n_all: int) -> int:
    if chosen == PRIOR_STRATEGY:
        return 0
    if chosen == "moe_fractal_consensus":
        return len(comb["temperatures"]) + len(comb["fractal"]["names"])
    base = len(comb["weights"]) if comb["weights"] else n_all
    return base + (len(comb["fractal"]["names"]) if chosen == "fractal_tiebreak" else 0)


def build_report(rows: Dict[str, dict], la0: List[float]) -> dict:
    full = len(rows) == len(gd.TASKS)
    accs = [r["accuracy"] for r in rows.values()]
    macro = float(np.mean(accs))
    agg = {
        "macro_avg_13": round(macro, 2) if full else None, "macro_avg_evaluated": round(macro, 2),
        "micro_acc": round(100 * sum(r["correct"] for r in rows.values()) / sum(r["n"] for r in rows.values()), 2),
        "delta_macro_vs_nimble": round(macro - PNG_AVG["nimble"], 2) if full else None,
        "delta_macro_vs_jev": round(macro - PNG_AVG["jev"], 2) if full else None,
        "delta_macro_vs_laya": round(macro - LAYA_MACRO, 2) if full else None,
        "tasks_beating_best_01png": sorted(t for t, r in rows.items()
                                           if not r["collapsed"] and r["delta_vs_best_01png"] > 0),
        "tasks_beating_best_01png_collapsed": sorted(t for t, r in rows.items()
                                                      if r["collapsed"] and r["delta_vs_best_01png"] > 0),
        "tasks_collapsed": sorted(t for t, r in rows.items() if r["collapsed"]),
        "tasks_beating_jev": sorted(t for t, r in rows.items() if r["delta_vs_jev"] > 0),
        "decision_latency_ms_median_over_tasks": float(np.median([r["decision_latency_ms"]["median"]
                                                                  for r in rows.values()])),
        "decision_latency_ms_max_p95": float(max(r["decision_latency_ms"]["p95"] for r in rows.values())),
        "png_reference_avg": PNG_AVG, "laya_a100_reported_macro_13": LAYA_MACRO,
    }
    for cfg in ("baseline", "deep_wide"):
        vals = [r["vs_prior_heads"][cfg]["accuracy"] for r in rows.values() if cfg in r["vs_prior_heads"]]
        if len(vals) == len(rows):
            agg[f"prior_{cfg}_macro"] = round(float(np.mean(vals)), 2)
            agg[f"tasks_sig_better_than_prior_{cfg}_p05"] = sorted(
                t for t, r in rows.items() if r["vs_prior_heads"][cfg]["mcnemar"]["p_value"] < 0.05
                and r["vs_prior_heads"][cfg]["delta"] > 0)
            agg[f"tasks_sig_worse_than_prior_{cfg}_p05"] = sorted(
                t for t, r in rows.items() if r["vs_prior_heads"][cfg]["mcnemar"]["p_value"] < 0.05
                and r["vs_prior_heads"][cfg]["delta"] < 0)
    return {
        "title": "01.PNG 13-task SOTA attempt: cross-fitted causal ensemble",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(sys.argv),
        "host": {"platform": platform.platform(), "cpu_count": os.cpu_count(), "loadavg_at_eval_start": la0,
                 "loadavg_at_eval_end": loadavg(), "blas_threads_env": {k: os.environ.get(k) for k in
                 ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
        "protocol": {
            "sources": {s: str(p) for s, p in SOURCES.items()}, "rnn_sources": RNN_SOURCES, "dual_manifold": list(DUAL_MANIFOLD) if DUAL_MANIFOLD else None,
            "rnn_heads_reused": {s: str(p) for s, p in RNN_HEADS.items()}, "n_folds": N_FOLDS, "fold_seed": FOLD_SEED,
            "nested_seed": NESTED_SEED, "raw_max_dim": RAW_MAX_DIM, "moe_n_threads": MOE_THREADS,
            "router_rank": ROUTER_RANK,
            "selection": "every temperature, weight, router and the strategy choice is fitted on out-of-fold "
                         "TRAIN scores; strategies are compared by nested 5-fold CV over those rows and chosen by "
                         "the one-standard-error rule (least complex strategy within 1 SE of the best mean, "
                         "SE from the best strategy's 5 fold accuracies); test labels are read once, after all "
                         "predictions are made",
            "strategies": [PRIOR_STRATEGY, "single:<expert>"] + list(STRATEGIES_LEARNED),
            "adapter_ranks": list(ADAPTER_RANKS),
            "supcon_ranks": list(SUPCON_RANKS),
            "adapter_b_ranks": list(ADAPTER_B_RANKS),
            "enable_rda": ENABLE_RDA,
            "enable_nystrom": ENABLE_NYSTROM,
            "comparability_caveat": "Nimble and Jev in 01.PNG are zero-shot judges; every expert here is "
                                    "supervised on the task's public train split (leakage-gated, train rows "
                                    "per task in train_rows_per_task). "
                                    "Laya numbers are copied from its A100 report, not re-run.",
            "train_rows_per_task": {t: r["n_train_rows"] for t, r in rows.items()},
            "png_reference_avg": PNG_AVG,
        },
        "aggregate": agg, "tasks": rows,
    }


def _parse_ranks(text: str, enable: bool, default: Tuple[int, ...], flag: str) -> Tuple[int, ...]:
    ranks = [int(t) for t in text.split(",") if t.strip()]
    if enable and not ranks:
        ranks = list(default)
    if any(r < 1 for r in ranks) or len(set(ranks)) != len(ranks):
        raise ValueError(f"{flag} must be distinct positive ints, got {ranks}")
    return tuple(ranks)


def parse_adapter_ranks(text: str, enable: bool) -> Tuple[int, ...]:
    return _parse_ranks(text, enable, DEFAULT_ADAPTER_RANKS, "--adapter-ranks")


def parse_supcon_ranks(text: str, enable: bool) -> Tuple[int, ...]:
    return _parse_ranks(text, enable, DEFAULT_SUPCON_RANKS, "--supcon-ranks")


def parse_adapter_b_ranks(text: str, enable: bool) -> Tuple[int, ...]:
    return _parse_ranks(text, enable, DEFAULT_ADAPTER_B_RANKS, "--adapter-b-ranks")


def parse_dual_manifold(text: str, sources: Sequence[str]) -> Optional[Tuple[str, str]]:
    """'' -> None; 'Q,G' -> (Q, G) with both registered --source names and Q != G."""
    names = [t.strip() for t in text.split(",") if t.strip()]
    if not names:
        return None
    if len(names) != 2 or names[0] == names[1] or any(n not in sources for n in names):
        raise ValueError(f"--dual-manifold must be QWEN_SOURCE,GEMMA_SOURCE, two different --source names "
                         f"among {sorted(sources)}, got {text!r}")
    return names[0], names[1]


def main() -> None:
    global OUT_ART, RESULTS, RNN_SOURCES, RNN_HEADS, PRIOR_REPORT, ADAPTER_RANKS, SUPCON_RANKS, ADAPTER_B_RANKS
    global DUAL_MANIFOLD
    global N_FOLDS, FOLD_SEED, NESTED_SEED, RAW_MAX_DIM, MOE_THREADS, ENABLE_RDA, ENABLE_NYSTROM
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "combine", "eval", "all"])
    ap.add_argument("--tasks", default=",".join(gd.TASKS))
    ap.add_argument("--source", action="append", required=True, help="name=ART (grand-challenge feature dir root)")
    ap.add_argument("--rnn-source", action="append", default=[], help="source whose Baseline/Deep-Wide heads join the ensemble")
    ap.add_argument("--rnn-heads", type=Path, default=None, help="dir with <task>_{baseline,deep_wide}.npz")
    ap.add_argument("--only", default="", help="fit stage: comma list of substrings of expert names")
    ap.add_argument("--out-art", type=Path, default=OUT_ART)
    ap.add_argument("--results-dir", type=Path, default=RESULTS)
    ap.add_argument("--prior-report", type=Path, default=PRIOR_REPORT)
    ap.add_argument("--test-dir", type=Path, default=gd.TEST_DIR)
    ap.add_argument("--torch-threads", type=int, default=4,
                    help="torch threads in the fit stage AND n_threads of the MultiModelCausalMoE in combine/eval")
    ap.add_argument("--raw-max-dim", type=int, default=None,
                    help=f"widest raw feature map fitted without PCA (env GC_RAW_MAX_DIM, default {DEFAULT_RAW_MAX_DIM}); "
                         "wider maps are skipped with a WARNING")
    ap.add_argument("--adapter-ranks", default="",
                    help="Spec 19 Phase 1: comma list of deep residual adapter ranks to register per source "
                         "(e.g. 32,64,128); empty = off")
    ap.add_argument("--enable-adapter", action="store_true",
                    help=f"shorthand for --adapter-ranks {','.join(map(str, DEFAULT_ADAPTER_RANKS))}")
    ap.add_argument("--supcon-ranks", default="",
                    help="Spec 19 Phase 2 direction 4: comma list of SupCon joint-training head ranks to register "
                         "per source (e.g. 32,64,128); empty = off")
    ap.add_argument("--enable-supcon", action="store_true",
                    help=f"shorthand for --supcon-ranks {','.join(map(str, DEFAULT_SUPCON_RANKS))}")
    ap.add_argument("--adapter-b-ranks", default="",
                    help="Spec 20 P3: comma list of Formulation B folded residual adapter ranks to register per "
                         "source (e.g. 32,64); empty = off")
    ap.add_argument("--enable-adapter-b", action="store_true",
                    help=f"shorthand for --adapter-b-ranks {','.join(map(str, DEFAULT_ADAPTER_B_RANKS))}")
    ap.add_argument("--dual-manifold", default="",
                    help="Spec 20 P4: QWEN_SOURCE,GEMMA_SOURCE (two --source names) registers one dual-manifold "
                         "expert per task (Qwen principal subspace + Gemma orthogonal innovation, folded); empty = off")
    ap.add_argument("--enable-rda", action="store_true",
                    help="Spec 19 Phase 3 direction 3: register a Ledoit-Wolf-shrunk RDA expert per source")
    ap.add_argument("--enable-nystrom", action="store_true",
                    help="Spec 19 Phase 3 direction 3: register a Nystrom low-rank RBF expert per source")
    ap.add_argument("--device", choices=sx.DEVICE_CHOICES, default="auto",
                    help="fit stage only: where the heads and probes are fitted. auto = CUDA if torch sees one, "
                         "else CPU; cuda errors out without CUDA. Scoring and every exported .npz stay CPU NumPy")
    ap.add_argument("--n-folds", type=int, default=N_FOLDS, help="cross-fitting / nested-CV folds")
    ap.add_argument("--fold-seed", type=int, default=FOLD_SEED)
    ap.add_argument("--nested-seed", type=int, default=NESTED_SEED)
    ap.add_argument("--strategies", default=",".join(STRATEGIES_LEARNED),
                    help="combine stage: comma list of learned strategies to compare, from "
                         + ",".join(STRATEGIES_LEARNED) + " (single:<expert> entries are always compared)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.n_folds < 2:
        raise SystemExit(f"--n-folds must be >= 2, got {args.n_folds}")
    if args.torch_threads < 1:
        raise SystemExit(f"--torch-threads must be >= 1, got {args.torch_threads}")
    try:
        RAW_MAX_DIM = resolve_raw_max_dim(args.raw_max_dim)
    except ValueError as e:
        raise SystemExit(str(e))
    N_FOLDS, FOLD_SEED, NESTED_SEED = args.n_folds, args.fold_seed, args.nested_seed
    try:
        ADAPTER_RANKS = parse_adapter_ranks(args.adapter_ranks, args.enable_adapter)
        SUPCON_RANKS = parse_supcon_ranks(args.supcon_ranks, args.enable_supcon)
        ADAPTER_B_RANKS = parse_adapter_b_ranks(args.adapter_b_ranks, args.enable_adapter_b)
    except ValueError as e:
        raise SystemExit(str(e))
    ENABLE_RDA, ENABLE_NYSTROM = args.enable_rda, args.enable_nystrom
    MOE_THREADS = args.torch_threads
    learned = tuple(t for t in args.strategies.split(",") if t)
    if set(learned) - set(STRATEGIES_LEARNED):
        raise SystemExit(f"unknown strategies: {sorted(set(learned) - set(STRATEGIES_LEARNED))}")
    for s in args.source:
        name, _, path = s.partition("=")
        if not name or not path or ":" in name:
            raise SystemExit(f"--source must be name=path without ':' in name, got {s!r}")
        SOURCES[name] = Path(path)
    try:
        DUAL_MANIFOLD = parse_dual_manifold(args.dual_manifold, list(SOURCES))   # needs SOURCES filled
    except ValueError as e:
        raise SystemExit(str(e))
    OUT_ART, RESULTS, PRIOR_REPORT, gd.TEST_DIR = args.out_art, args.results_dir, args.prior_report, args.test_dir
    RNN_SOURCES = [s.strip() for rs in args.rnn_source for s in rs.split(",") if s.strip()]
    for s in RNN_SOURCES:
        if s not in SOURCES:
            raise SystemExit(f"--rnn-source {s!r} is not a --source")
    if args.rnn_heads and len(RNN_SOURCES) != 1:
        # One override dir cannot hold the heads of two encoders; "--rnn-source a,b" used to
        # hand it to both.
        raise SystemExit("--rnn-heads needs exactly one --rnn-source; each source otherwise uses <ART>/heads")
    RNN_HEADS = {s: args.rnn_heads or SOURCES[s] / "heads" for s in RNN_SOURCES}
    tasks = [t for t in args.tasks.split(",") if t]
    if set(tasks) - set(gd.TASKS):
        raise SystemExit(f"unknown tasks: {sorted(set(tasks) - set(gd.TASKS))}")
    if args.stage in ("fit", "all"):
        import torch
        torch.set_num_threads(args.torch_threads)
        try:
            device = sx.resolve_device(args.device)
        except RuntimeError as e:
            raise SystemExit(str(e))
        print(f"[device] Fitting pipeline running on: {device} (CUDA available: {sx.cuda_available()})", flush=True)
        stage_fit(tasks, [o for o in args.only.split(",") if o], device)
    if args.stage in ("combine", "all"):
        stage_combine(tasks, learned)
    if args.stage in ("eval", "all"):
        stage_eval(tasks)


if __name__ == "__main__":
    main()
