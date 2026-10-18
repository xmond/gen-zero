"""Production MCTS layerwise early exit + sparse SwiGLU FFN on 28-layer Qwen2.5-1.5B-Instruct.

Data: the 100 balanced, distinct BoolQ rows in ``benchmarks/data/boolq.jsonl`` (first 50
"yes" + first 50 "no" by file order). Each prompt is the full ``row["context"]`` (it ends
with "Answer with yes or no based strictly on the passage.") wrapped as one user turn by
the tokenizer's chat template with ``add_generation_prompt=True``, encoded with
``add_special_tokens=False``. Labels are read through ``VerbalizerProbe`` with two token
groups: yes = {"yes", "Yes", " yes", " Yes"}, no = {"no", "No", " no", " No"}; every
string must encode to a single token.

Every arm runs ``MctsLayerwiseQwenRunner.predict`` from
``python/gen_zero/causal/qwen_inter_layer_mcts.py``; a fresh runner is built per fold and
per arm, and everything it fits sees only that fold's training rows:

* ``full_28_uncalibrated``: raw probe, ``min_exit_layer = 27`` (never exits).
* ``full_28_calibrated``: the dense fold fit's probe stats and memories, no layer weights,
  ``min_exit_layer = 27``.
* ``mcts_multihook_dense_ffn``: ``fit_memories(train, layers=checkpoints + [27],
  fit_mcts_layer_weights=True, top_k_hooks)`` then ``predict``.
* ``mcts_multihook_sparse_ffn[_<layers>_k<ratio>]``: ``fit_ffn_pruning(train, layers,
  keep_ratio)`` (``QwenSparseFfnHook`` physical slices), then the same ``fit_memories``
  on the sliced model, then ``predict``.

The headline sparse configuration is fixed by ``--headline-ffn`` before any run (default
layers 22..27, keep ratio 0.75). Other ``--ffn-grid`` configurations are evaluated on the
same rows and reported, never used to pick the headline.

Timing: per row, end-to-end = chat template + tokenizer + host-to-device copy +
``predict`` (forward, hooks, reads, MCTS decisions). On CUDA, ``torch.cuda.synchronize()``
brackets every timed region. Model loading, anchor passes and fitting are reported
separately. ``MctsLayerwiseQwenRunner`` caches one pre-sliced ``QwenSparseFfnHook`` (built
by ``fit_ffn_pruning``) and reuses it across ``predict`` calls; it re-slices only when the
channel masks change, so slicing is fit cost, not per-row sparse arm time. The
FFN operator microbenchmark times one masked layer's MLP on a real captured input, with
the hook built once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import gen_zero.causal.latent_mcts as latent_mcts_module
import gen_zero.causal.qwen_inter_layer_mcts as mcts_module
from gen_zero.causal.qwen_inter_layer_mcts import (
    MctsLayerwiseQwenRunner, QwenSparseFfnHook, VerbalizerProbe, decoder_layers,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ("/ebs/cargo/qwen-cpu-bench/hf/hub/models--Qwen--Qwen2.5-1.5B-Instruct/"
                 "snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
YES_STRINGS = ("yes", "Yes", " yes", " Yes")
NO_STRINGS = ("no", "No", " no", " No")
PROMPT_SUFFIX = "\nAnswer with yes or no based strictly on the passage."
NUM_LAYERS = 28
HIDDEN = 1536
INTERMEDIATE = 8960


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _loadavg():
    return list(os.getloadavg()) if hasattr(os, "getloadavg") else None


def _rows(path: Path, n: int) -> list[dict]:
    want = {"yes": n // 2, "no": n - n // 2}
    rows, seen = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        label = row["ground_truth"]
        if label not in want:
            raise ValueError(f"unsupported BoolQ label in {path}: {label!r}")
        if not row["context"].startswith("Passage: ") or not row["context"].endswith(PROMPT_SUFFIX):
            raise ValueError(f"unexpected BoolQ prompt structure in {path}: {row.get('id')}")
        question = row["metadata"]["question"].strip().lower().rstrip("?")
        if question not in seen and want[label] > 0:
            want[label] -= 1
            seen.add(question)
            rows.append(row)
    if any(want.values()):
        raise ValueError(f"BoolQ source lacks enough distinct balanced rows: missing {want}")
    return rows


def _single_token(tok, text: str) -> int:
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"verbalizer {text!r} is not a single token: {ids}")
    return ids[0]


def _encode(tok, row, device):
    if getattr(tok, "chat_template", None) is None:
        raise ValueError("Qwen2.5-1.5B-Instruct tokenizer has no chat template")
    text = tok.apply_chat_template([{"role": "user", "content": row["context"]}],
                                   tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _parse_ffn(spec: str) -> tuple[list[int], float]:
    layers, ratio = spec.split(":")
    lo, hi = (int(x) for x in layers.split(".."))
    if not 0 <= lo <= hi < NUM_LAYERS:
        raise ValueError(f"bad FFN layer range {layers!r}")
    keep = float(ratio)
    if not 0 < keep <= 1:
        raise ValueError(f"bad FFN keep ratio {ratio!r}")
    return list(range(lo, hi + 1)), keep


def _ffn_name(spec: str) -> str:
    layers, keep = _parse_ffn(spec)
    return f"mcts_multihook_sparse_ffn_L{layers[0]}-{layers[-1]}_k{keep}"


def _mcnemar_exact(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def _predict_timed(runner, tok, row, device):
    """End-to-end and inference-only seconds for one row through ``runner.predict``."""
    _sync(device)
    t0 = time.perf_counter()
    ids = _encode(tok, row, device)
    _sync(device)
    t1 = time.perf_counter()
    decision = runner.predict(ids)
    _sync(device)
    t2 = time.perf_counter()
    if not np.all(np.isfinite(decision.probs)):
        raise FloatingPointError("predict returned non-finite probabilities")
    return decision, t2 - t0, t2 - t1


def _plain_forward_timed(model, tok, row, device):
    """Reference: one plain 28-layer backbone forward, no hooks, same timing boundary."""
    _sync(device)
    t0 = time.perf_counter()
    ids = _encode(tok, row, device)
    _sync(device)
    t1 = time.perf_counter()
    with torch.no_grad():
        model.model(input_ids=ids, use_cache=False)
    _sync(device)
    t2 = time.perf_counter()
    return t2 - t0, t2 - t1


def _summary(decisions, labels, e2e, infer, reference=None):
    pred = np.array([d.label for d in decisions])
    y = np.asarray(labels)
    if len(pred) == 0:
        raise ValueError("arm evaluated zero rows")
    layers = np.array([d.layers_executed for d in decisions])
    ch = np.array([d.ffn_channels_executed for d in decisions], dtype=np.float64)
    ch_dense = np.array([d.ffn_dense_channels for d in decisions], dtype=np.float64)
    macs = np.array([d.ffn_macs_executed for d in decisions], dtype=np.float64)
    hist: dict[str, int] = {}
    for d in decisions:
        key = str(d.exit_layer) if d.early_exit else "full_28"
        hist[key] = hist.get(key, 0) + 1
    out = {
        "correct": int(np.sum(pred == y)), "n": int(len(y)),
        "accuracy": float(np.mean(pred == y)),
        "mean_layers_executed": float(layers.mean()),
        "layer_reduction_vs_28": float(1 - layers.mean() / NUM_LAYERS),
        "early_exits": int(sum(d.early_exit for d in decisions)),
        "exit_layer_histogram": dict(sorted(hist.items())),
        "mean_ffn_channels_executed_per_token": float(ch.mean()),
        "mean_ffn_channels_hooked_dense_per_token": float(ch_dense.mean()),
        "mean_ffn_macs_executed_per_token": float(macs.mean()),
        "ffn_channel_reduction_vs_dense28": float(1 - ch.mean() / (NUM_LAYERS * INTERMEDIATE)),
        "ffn_mac_reduction_vs_dense28": float(1 - macs.mean() / (3 * HIDDEN * INTERMEDIATE * NUM_LAYERS)),
        "e2e_seconds_total": float(np.sum(e2e)), "e2e_ms_per_row_mean": float(np.mean(e2e) * 1e3),
        "e2e_ms_per_row_median": float(np.median(e2e) * 1e3),
        "inference_seconds_total": float(np.sum(infer)),
        "inference_ms_per_row_mean": float(np.mean(infer) * 1e3),
        "inference_ms_per_row_median": float(np.median(infer) * 1e3),
        "predictions": pred.tolist(),
        "layers_executed": layers.tolist(),
    }
    for name, ref in (reference or {}).items():
        base = np.array([d.label for d in ref])
        wins = int(np.sum((pred == y) & (base != y)))
        losses = int(np.sum((pred != y) & (base == y)))
        out[f"vs_{name}"] = {"agreement": int(np.sum(pred == base)), "wins": wins,
                             "losses": losses, "mcnemar_exact_p": _mcnemar_exact(wins, losses)}
    return out


def _ffn_microbench(model, masks, x, device, reps):
    """Time one MLP forward on a real input, dense vs physically sliced (hook built once)."""
    layer = min(masks)
    mlp = decoder_layers(model)[layer].mlp

    def bench():
        times = []
        with torch.no_grad():
            for _ in range(3):
                y = mlp(x)
            _sync(device)
            for _ in range(reps):
                t0 = time.perf_counter()
                y = mlp(x)
                _sync(device)
                times.append(time.perf_counter() - t0)
        return y, float(np.median(times))

    dense_y, dense_s = bench()
    _sync(device)
    t0 = time.perf_counter()
    hook = QwenSparseFfnHook(model, masks)
    _sync(device)
    build_s = time.perf_counter() - t0
    with hook:
        sparse_y, sparse_s = bench()
    rel = float((sparse_y - dense_y).norm() / dense_y.norm())
    return {"layer": layer, "tokens": int(x.shape[1]), "reps": reps,
            "kept_channels": int(len(masks[layer])), "dense_channels": INTERMEDIATE,
            "dense_mlp_ms_median": dense_s * 1e3, "sparse_mlp_ms_median": sparse_s * 1e3,
            "speedup": dense_s / sparse_s, "relative_output_change": rel,
            "sparse_hook_build_ms_all_masked_layers": build_s * 1e3}


def _capture_mlp_input(model, ids, layer):
    box = {}
    mlp = decoder_layers(model)[layer].mlp
    handle = mlp.register_forward_pre_hook(lambda _m, args: box.setdefault("x", args[0].detach().clone()))
    try:
        with torch.no_grad():
            model.model(input_ids=ids, use_cache=False)
    finally:
        handle.remove()
    return box["x"]


def _nvidia_smi():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,"
                            "utilization.gpu,clocks.sm", "--format=csv"],
                           capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        return None
    return {"returncode": q.returncode, "stdout": q.stdout}


def main(argv=None) -> int:
    wall_start = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default=DEFAULT_MODEL)
    ap.add_argument("--data", default=str(ROOT / "benchmarks" / "data" / "boolq.jsonl"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--eval-folds", default="all", help="'all' or a JSON list of fold indices")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--checkpoints", default="[12,16,20,22,24,26]")
    ap.add_argument("--min-exit-layer", type=int, default=22)
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--mcts-margin-min", type=float, default=0.05)
    ap.add_argument("--top-k-hooks", type=int, default=4)
    ap.add_argument("--headline-ffn", default="22..27:0.75")
    ap.add_argument("--ffn-grid", default="22..27:0.625,22..27:0.75,20..27:0.625,20..27:0.75")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--microbench-reps", type=int, default=50)
    ap.add_argument("--label", default="", help="free-text run label stored in the report")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.n < 8 or args.n % 2 or args.folds < 2 or args.n // 2 < args.folds:
        raise ValueError("--n must be even, >= 8, and give every fold both labels")
    if args.threads > 8:
        raise ValueError("--threads is capped at 8 on shared boxes")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    load_start = _loadavg()
    smi_start = _nvidia_smi() if device.type == "cuda" else None

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.hf_model, local_files_only=True,
                                                 dtype=getattr(torch, args.dtype)).to(device).eval()
    _sync(device)
    load_seconds = time.perf_counter() - t0
    total = len(decoder_layers(model))
    cfg = model.config
    if (total, cfg.hidden_size, cfg.intermediate_size) != (NUM_LAYERS, HIDDEN, INTERMEDIATE):
        raise ValueError(f"expected 28-layer Qwen2.5-1.5B, got {total} layers, "
                         f"hidden {cfg.hidden_size}, intermediate {cfg.intermediate_size}")
    groups = [[_single_token(tok, s) for s in YES_STRINGS], [_single_token(tok, s) for s in NO_STRINGS]]
    probe = VerbalizerProbe(model, groups)
    checkpoints = [int(c) for c in json.loads(args.checkpoints)]
    hook_layers = sorted({*checkpoints, total - 1})
    headline = args.headline_ffn
    grid = [s for s in args.ffn_grid.split(",") if s]
    if headline not in grid:
        grid.insert(0, headline)

    rows = _rows(Path(args.data), args.n)
    labels = [0 if r["ground_truth"] == "yes" else 1 for r in rows]
    t0 = time.perf_counter()
    anchor_ids = [_encode(tok, r, device) for r in rows]
    tokenize_all_seconds = time.perf_counter() - t0
    rng = np.random.default_rng(args.seed)
    folds = np.empty(len(rows), dtype=int)
    for cls in (0, 1):
        members = rng.permutation([i for i, y in enumerate(labels) if y == cls])
        folds[members] = np.arange(len(members)) % args.folds
    eval_folds = list(range(args.folds)) if args.eval_folds == "all" else json.loads(args.eval_folds)

    def make_runner(min_exit=args.min_exit_layer, **kw):
        return MctsLayerwiseQwenRunner(model, probe, checkpoints=checkpoints, tau=args.tau,
                                       min_exit_layer=min_exit, mcts_margin_min=args.mcts_margin_min,
                                       **kw)

    # Warm up kernels and allocators on training rows of the first evaluated fold only.
    warm_rows = [i for i in range(len(rows)) if folds[i] != eval_folds[0]][:args.warmup]
    for i in warm_rows:
        _plain_forward_timed(model, tok, rows[i], device)
        make_runner(min_exit=total - 1).predict(anchor_ids[i])

    arm_names = ["full_28_uncalibrated", "full_28_calibrated", "mcts_multihook_dense_ffn",
                 *[_ffn_name(s) for s in grid]]
    decisions = {name: {} for name in arm_names}
    e2e = {name: {} for name in arm_names}
    infer = {name: {} for name in arm_names}
    plain_e2e, plain_infer = {}, {}
    fold_fits = []
    fit_seconds = 0.0
    for fold in eval_folds:
        train = [i for i in range(len(rows)) if folds[i] != fold]
        test = [i for i in range(len(rows)) if folds[i] == fold]
        if set(train) & set(test) or len(train) + len(test) != len(rows):
            raise RuntimeError("cross-fit train/test partition is invalid")
        tr_ids, tr_y = [anchor_ids[i] for i in train], [labels[i] for i in train]
        fold_info = {"fold": fold, "train": len(train), "test": len(test), "arms": {}}

        t0 = time.perf_counter()
        dense = make_runner()
        reports = dense.fit_memories(tr_ids, tr_y, layers=hook_layers,
                                     fit_mcts_layer_weights=True, top_k_hooks=args.top_k_hooks)
        fit_s = time.perf_counter() - t0
        fit_seconds += fit_s
        fold_info["arms"]["mcts_multihook_dense_ffn"] = {
            "fit_seconds": fit_s,
            "layer_weights": {str(k): v for k, v in dense.layer_weights.items()},
            "memory_lambdas": {str(k): dense.memories[k][1] for k in hook_layers},
            "probe_logit_mean": {str(k): reports[k]["probe_logit_mean"] for k in hook_layers},
            "probe_logit_std": {str(k): reports[k]["probe_logit_std"] for k in hook_layers}}
        runners = {
            "full_28_uncalibrated": make_runner(min_exit=total - 1),
            "full_28_calibrated": make_runner(min_exit=total - 1, memories=dense.memories,
                                              probe_stats=dense.probe_stats),
            "mcts_multihook_dense_ffn": dense,
        }
        for spec in grid:
            name = _ffn_name(spec)
            ffn_layers, keep = _parse_ffn(spec)
            t0 = time.perf_counter()
            sparse = make_runner()
            masks = sparse.fit_ffn_pruning(tr_ids, ffn_layers, keep_ratio=keep)
            sparse.fit_memories(tr_ids, tr_y, layers=hook_layers,
                                fit_mcts_layer_weights=True, top_k_hooks=args.top_k_hooks)
            fit_s = time.perf_counter() - t0
            fit_seconds += fit_s
            runners[name] = sparse
            fold_info["arms"][name] = {
                "fit_seconds": fit_s,
                "kept_channels": {str(k): int(len(v)) for k, v in masks.items()},
                "layer_weights": {str(k): v for k, v in sparse.layer_weights.items()},
                "memory_lambdas": {str(k): sparse.memories[k][1] for k in hook_layers}}
        for i in test:
            plain_e2e[i], plain_infer[i] = _plain_forward_timed(model, tok, rows[i], device)
            for name in arm_names:
                d, te, ti = _predict_timed(runners[name], tok, rows[i], device)
                decisions[name][i], e2e[name][i], infer[name][i] = d, te, ti
        fold_fits.append(fold_info)

    evaluated = sorted(decisions["full_28_uncalibrated"])
    y_eval = [labels[i] for i in evaluated]
    ordered = {name: [decisions[name][i] for i in evaluated] for name in arm_names}
    refs = {"full_28_uncalibrated": ordered["full_28_uncalibrated"],
            "full_28_calibrated": ordered["full_28_calibrated"],
            "mcts_multihook_dense_ffn": ordered["mcts_multihook_dense_ffn"]}
    arms = {}
    for name in arm_names:
        arms[name] = _summary(ordered[name], y_eval, [e2e[name][i] for i in evaluated],
                              [infer[name][i] for i in evaluated],
                              {k: v for k, v in refs.items() if k != name})
    headline_name = _ffn_name(headline)
    arms["mcts_multihook_sparse_ffn"] = dict(arms[headline_name], headline_of=headline_name)
    base_e2e = arms["full_28_uncalibrated"]["e2e_seconds_total"]
    base_inf = arms["full_28_uncalibrated"]["inference_seconds_total"]
    for name, arm in arms.items():
        arm["e2e_speedup_vs_full_28_uncalibrated"] = base_e2e / arm["e2e_seconds_total"]
        arm["inference_speedup_vs_full_28_uncalibrated"] = base_inf / arm["inference_seconds_total"]

    head_layers, head_keep = _parse_ffn(headline)
    probe_x = _capture_mlp_input(model, anchor_ids[evaluated[0]], head_layers[0])
    first_train = [i for i in range(len(rows)) if folds[i] != eval_folds[0]]
    micro_runner = make_runner()
    micro_masks = micro_runner.fit_ffn_pruning([anchor_ids[i] for i in first_train], head_layers,
                                               keep_ratio=head_keep)
    microbench = _ffn_microbench(model, micro_masks, probe_x, device, args.microbench_reps)

    report = {
        "label": args.label,
        "model": args.hf_model, "dtype": args.dtype, "device": str(device),
        "host": platform.node(), "platform": platform.platform(), "python": sys.version,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32) if device.type == "cuda" else None,
        "torch_threads": torch.get_num_threads(), "seed": args.seed,
        "module_sha256": {"qwen_inter_layer_mcts.py": _sha256(mcts_module.__file__),
                          "latent_mcts.py": _sha256(latent_mcts_module.__file__),
                          "benchmark_suite": _sha256(__file__)},
        "data": args.data, "data_sha256": _sha256(args.data), "n": len(rows),
        "folds": args.folds, "eval_folds": eval_folds, "rows_evaluated": len(evaluated),
        "fold_label_counts": [[int(np.sum((folds == f) & (np.asarray(labels) == c))) for c in (0, 1)]
                              for f in range(args.folds)],
        "prompt": "chat_template([{'role':'user','content':row['context']}], add_generation_prompt=True), "
                  "add_special_tokens=False",
        "prompt_tokens_mean": float(np.mean([x.shape[1] for x in anchor_ids])),
        "verbalizer_strings": [list(YES_STRINGS), list(NO_STRINGS)],
        "verbalizer_token_ids": groups, "labels": {"yes": 0, "no": 1},
        "total_layers": total, "checkpoints": checkpoints, "hook_layers": hook_layers,
        "min_exit_layer": args.min_exit_layer, "tau": args.tau,
        "mcts_margin_min": args.mcts_margin_min, "top_k_hooks": args.top_k_hooks,
        "headline_ffn": headline, "ffn_grid": grid,
        "ffn_grid_note": "headline fixed before the run; other grid arms evaluated on the same rows",
        "dense28_ffn_channels_per_token": NUM_LAYERS * INTERMEDIATE,
        "dense28_ffn_macs_per_token": 3 * HIDDEN * INTERMEDIATE * NUM_LAYERS,
        "timing_boundary": "e2e = chat template + tokenizer + H2D + runner.predict; inference = "
                           "runner.predict only; cuda synchronize brackets each region; model load, "
                           "anchor passes and fitting excluded and reported separately",
        "model_load_seconds": load_seconds, "tokenize_all_rows_seconds": tokenize_all_seconds,
        "fit_seconds_total": fit_seconds,
        "plain_backbone_forward_28": {
            "e2e_ms_per_row_mean": float(np.mean([plain_e2e[i] for i in evaluated]) * 1e3),
            "inference_ms_per_row_mean": float(np.mean([plain_infer[i] for i in evaluated]) * 1e3),
            "note": "model.model(input_ids) with no hooks and no probe; reference for hook overhead"},
        "ffn_operator_microbench": microbench,
        "fold_fits": fold_fits,
        "row_ids": [rows[i]["id"] for i in evaluated],
        "row_labels": y_eval,
        "arms": arms,
        "loadavg_start": load_start, "loadavg_end": _loadavg(),
        "nvidia_smi_start": smi_start,
        "nvidia_smi_end": _nvidia_smi() if device.type == "cuda" else None,
    }
    report["total_wall_seconds"] = time.perf_counter() - wall_start
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    brief = {k: {kk: v[kk] for kk in ("correct", "n", "mean_layers_executed",
                                      "ffn_mac_reduction_vs_dense28",
                                      "e2e_speedup_vs_full_28_uncalibrated")}
             for k, v in arms.items()}
    print(json.dumps({"out": args.out, "arms": brief, "microbench": microbench,
                      "total_wall_seconds": report["total_wall_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
