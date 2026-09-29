"""01.PNG grand challenge: 13 public benchmarks, 3,880 test records, pure CPU.

Compares two heads on the SAME frozen CPU text encoder and the SAME
leakage-gated public train data (see grand_challenge_data.py):

  a) baseline   : legacy ParallelRNNSetAdapter (1 think loop, 1-layer
                  4-head set block) on ONE pooled vector of the whole
                  serialized context ("single-direction concatenated pooling").
  b) deep_wide  : DeepWideRNNSetAdapter (3 residual Parallel RNN layers with
                  per-layer Lyapunov clamps 0.95/0.85/0.75, 2-layer 8-head
                  Set-Transformer with SwiGLU). On the paired tasks (PAWS,
                  MultiNLI, VitaminC) its query is the cross-difference
                  manifold [z_A; z_B; z_A - z_B; z_A * z_B] of the two fields
                  encoded SEPARATELY; on the other 10 tasks it scores the same
                  whole-context vector as the baseline.

Both are per-task supervised heads, trained here on public train splits,
checkpoint-selected on a slice of TRAIN, exported to npz, and evaluated with
the NumPy CPU runtimes in python/gen_zero/causal (argmax over the task's
candidates vs ground_truth, exact match). The test labels are read only by
the final scoring line. Nimble and Jev (01.PNG) are zero-shot LLM judges; the
comparison is therefore supervised-small-head vs zero-shot-large-model and is
reported as such.

Stages (each resumable, artifacts under ART):
  encode   Qwen2.5-0.5B fp32 on CPU -> features/<task>.npz
  train    both heads per task      -> heads/<task>_<config>.npz
  eval     NumPy runtimes on test   -> benchmarks/results/01png_grand_challenge_report.{json,md}
"""
from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "python"))

import grand_challenge_data as gd  # noqa: E402

ART = Path(os.environ.get("GC_ART", "/ebs2/gen-zero-grand-challenge"))
FEAT_DIR, HEAD_DIR = ART / "features", ART / "heads"
RESULTS = REPO / "benchmarks" / "results"

ENCODER = os.environ.get("GC_ENCODER", "Qwen/Qwen2.5-0.5B")
ENCODER_MODE = os.environ.get("GC_ENCODER_MODE", "hf")   # "hf" (baseline) | "server" (GGUF llama-server)
SERVER_URL = os.environ.get("GC_SERVER_URL", "")
MAX_TOK, HEAD_TOK = 256, 64          # head+tail truncation: first 64 + last 192 tokens
POOL = (("mean", 12), ("last", 18))  # chosen on a TRAIN-internal pilot, see report
TASK_MAX_TOK: Dict[str, int] = {}
SERVER_WORKERS, SERVER_TIMEOUT = 4, 600.0


def parse_pool_layers(spec: str) -> tuple[tuple[str, int], ...]:
    pools = []
    for entry in spec.split(","):
        kind, sep, layer = entry.strip().partition("@")
        if not sep or kind not in ("mean", "last") or not layer.isdecimal():
            raise ValueError("--pool-layers expects entries such as mean@12,last@18")
        pools.append((kind, int(layer)))
    if not pools or any(layer < 1 for _, layer in pools):
        raise ValueError("--pool-layers needs positive decoder layer numbers")
    return tuple(pools)
BATCH = 16
LAT_SAMPLES = 16                     # batch-1 CPU encoder latency probes per task


def loadavg() -> List[float]:
    try:
        return [float(x) for x in os.getloadavg()]
    except (AttributeError, OSError):
        try:
            import psutil
            return [float(x) for x in psutil.getloadavg()]
        except (ImportError, AttributeError, OSError):
            return [0.0, 0.0, 0.0]


# ------------------------------------------------------------------ encode

class Encoder:
    def __init__(self, threads: int) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(ENCODER, local_files_only=True)
        self.model = AutoModel.from_pretrained(ENCODER, dtype=torch.float32, local_files_only=True).eval()
        n_layers = self.model.config.num_hidden_layers
        if any(layer > n_layers for _, layer in POOL):
            raise ValueError(f"--pool-layers exceeds {n_layers} decoder layers")
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.dim = self.model.config.hidden_size * len(POOL)
        self.max_tok = MAX_TOK

    def ids(self, text: str) -> List[int]:
        ids = self.tok(text, add_special_tokens=False)["input_ids"] or [self.pad]
        return gd.truncate_ids(ids, self.max_tok, HEAD_TOK)

    def _forward(self, seqs: List[List[int]]) -> np.ndarray:
        torch = self.torch
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), self.pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
        with torch.no_grad():
            hs = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
        m = mask.unsqueeze(-1).float()
        last = mask.sum(1) - 1
        parts = []
        for kind, layer in POOL:
            h = hs[layer]
            parts.append((h * m).sum(1) / m.sum(1) if kind == "mean" else h[torch.arange(len(seqs)), last])
        return torch.cat(parts, dim=-1).float().numpy()

    def encode(self, texts: List[str]) -> Dict[str, object]:
        seqs = [self.ids(t) for t in texts]
        order = np.argsort([len(s) for s in seqs], kind="stable")
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        t0 = time.perf_counter()
        for s in range(0, len(order), BATCH):
            idx = order[s:s + BATCH]
            out[idx] = self._forward([seqs[i] for i in idx])
        dt = time.perf_counter() - t0
        if not np.all(np.isfinite(out)):
            raise FloatingPointError("encoder produced non-finite features")
        return {"X": out, "seconds": dt, "tokens": int(sum(len(s) for s in seqs))}

    def latency_ms(self, texts: List[str]) -> List[float]:
        ms = []
        for t in texts:
            t0 = time.perf_counter()
            self._forward([self.ids(t)])
            ms.append((time.perf_counter() - t0) * 1e3)
        return ms


def pair_texts(rows: List[dict], task: str):
    a, b = gd.PAIR_FIELDS[task]
    return [r["fields"][a] for r in rows], [r["fields"][b] for r in rows]


def make_encoder(threads: int):
    if ENCODER_MODE == "server":
        if not SERVER_URL:
            raise SystemExit("--encoder-mode server needs --server-url")
        from gguf_server_encoder import GGUFServerEncoder
        return GGUFServerEncoder(SERVER_URL, MAX_TOK, HEAD_TOK,
                                 workers=SERVER_WORKERS, timeout=SERVER_TIMEOUT)
    return Encoder(threads)


def _cached_encoder_label(path: Path) -> str:
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["info_json"])).get("encoder", "?")


def stage_encode(tasks: List[str], threads: int) -> None:
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    enc = make_encoder(threads)
    for task in tasks:
        task_cap = TASK_MAX_TOK.get(task, MAX_TOK)
        path = FEAT_DIR / f"{task}.npz"
        if path.exists():
            # A reused --art would otherwise re-evaluate another encoder's features under this label.
            have = _cached_encoder_label(path)
            if have != ENCODER:
                raise SystemExit(f"{path} was encoded by {have!r}, not {ENCODER!r}: use a fresh --art")
            # Caches written before the cap was configurable carry no n_train_max; they used 1000.
            if gd.cached_train_spec(path) != gd.train_spec(task):
                raise SystemExit(f"{path} was built with (n_train_max, pubmedqa_extra) "
                                 f"{gd.cached_train_spec(path)}, not {gd.train_spec(task)}: use a fresh --art")
            with np.load(path, allow_pickle=False) as z:
                cached = json.loads(str(z["info_json"]))
            legacy = (cached.get("max_tok", 256), cached.get("head_tok", 64),
                      cached.get("pool", [["mean", 12], ["last", 18]] if ENCODER_MODE == "hf" else "server_pooled_last"))
            if legacy != (task_cap, HEAD_TOK,
                           [list(p) for p in POOL] if ENCODER_MODE == "hf" else "server_pooled_last"):
                raise SystemExit(f"{path} has different token or pooling settings: use a fresh --art")
            print(f"[encode] {task}: cached", flush=True)
            continue
        enc.max_tok = task_cap
        la0 = loadavg()
        test = gd.load_test(task)
        n_train = gd.n_train_for(task)
        train, gate = gd.build_train(task, test, n_max=n_train)
        arrays, info = {}, {"task": task, "gate": gate, "n_train_max": n_train,
                            "pubmedqa_extra": gd.train_spec(task)[1],
                            "encoder": ENCODER, "encoder_mode": ENCODER_MODE,
                            "pool": POOL if ENCODER_MODE == "hf" else "server_pooled_last",
                            "max_tok": task_cap, "head_tok": HEAD_TOK, "feature_dim": int(enc.dim)}
        jobs = {"train_full": [r["context"] for r in train], "test_full": [r["context"] for r in test],
                "cands": test[0]["candidates"]}
        if task in gd.PAIR_FIELDS:
            jobs["train_a"], jobs["train_b"] = pair_texts(train, task)
            jobs["test_a"], jobs["test_b"] = pair_texts(test, task)
        for name, texts in jobs.items():
            res = enc.encode(texts)
            arrays[name] = res["X"]
            info[f"{name}_seconds"], info[f"{name}_tokens"] = res["seconds"], res["tokens"]
            print(f"[encode] {task}/{name}: {len(texts)} texts {res['tokens']} tok "
                  f"{res['seconds']:.1f}s", flush=True)
        probe = test[:LAT_SAMPLES]
        info["latency_full_ms"] = enc.latency_ms([r["context"] for r in probe])
        if task in gd.PAIR_FIELDS:
            pa, pb = pair_texts(probe, task)
            info["latency_pair_ms"] = [x + y for x, y in zip(enc.latency_ms(pa), enc.latency_ms(pb))]
        info["loadavg_before"], info["loadavg_after"] = la0, loadavg()
        arrays["train_label"] = np.array([r["candidates"].index(r["ground_truth"]) for r in train])
        arrays["train_ids"] = np.array([r["id"] for r in train])
        arrays["test_ids"] = np.array([r["id"] for r in test])
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, info_json=np.array(json.dumps(info)), **arrays)
        tmp.rename(path)
        print(f"[encode] {task}: done", flush=True)


# ------------------------------------------------------------------- train

CONFIGS = ("baseline", "deep_wide")
EPOCHS, PATIENCE, BATCH_TRAIN, LR, WD = 80, 10, 32, 1e-3, 0.01
EARLY_STOP_FRACTION = 0.15            # of TRAIN; picks the checkpoint, never test
TRAIN_SEED = 0


def load_features(task: str) -> Dict[str, np.ndarray]:
    with np.load(FEAT_DIR / f"{task}.npz", allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    out["info"] = json.loads(str(out.pop("info_json")))
    return out


def _pca_init(model, states: np.ndarray) -> None:
    """Same PCA/ZCA init for both configs: mu = mean, W_in = top-d eigvecs / sqrt(eval)."""
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


def build_model(config: str, in_dim: int):
    if config == "baseline":
        from rnn_set_adapter_torch import ParallelRNNSetAdapter
        return ParallelRNNSetAdapter(in_dim, d=256, rank=16, think_steps=6, n_heads=4, n_layers=1,
                                     dropout=0.1)
    from deep_wide_rnn_set_torch import DeepWideRNNSetAdapter
    return DeepWideRNNSetAdapter(in_dim, d=512, rank=16, think_steps=6, rnn_layers=3,
                                 rho_max_schedule=[0.95, 0.85, 0.75], n_heads=8, set_layers=2,
                                 ffn_mult=4)


def query_inputs(config: str, task: str, f: Dict[str, np.ndarray], split: str):
    """baseline: one whole-context vector. deep_wide: (A, B) on the pair tasks."""
    if config == "deep_wide" and task in gd.PAIR_FIELDS:
        return (f[f"{split}_a"], f[f"{split}_b"])
    return (f[f"{split}_full"],)


def _logits(model, config: str, task: str, q, C):
    import torch
    B = q[0].shape[0]
    Cb = C.unsqueeze(0).expand(B, -1, -1)
    mask = torch.ones(B, C.shape[0], dtype=torch.bool)
    if len(q) == 2:
        return model.forward_pair(q[0], q[1], Cb, mask)
    return model(q[0], Cb, mask)


def train_one(task: str, config: str, f: Dict[str, np.ndarray]) -> dict:
    import torch
    import torch.nn.functional as F
    torch.manual_seed(TRAIN_SEED)
    rng = np.random.default_rng([TRAIN_SEED, gd.TASKS.index(task)])
    q_np = query_inputs(config, task, f, "train")
    y_np = f["train_label"].astype(np.int64)
    n = len(y_np)
    es = rng.random(n) < EARLY_STOP_FRACTION
    tr_idx, es_idx = np.flatnonzero(~es), np.flatnonzero(es)
    model = build_model(config, f["cands"].shape[1])
    _pca_init(model, np.concatenate([q[tr_idx] for q in q_np] + [f["cands"]]))
    q_all = [torch.as_tensor(q) for q in q_np]
    C = torch.as_tensor(f["cands"])
    y = torch.as_tensor(y_np)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD},
                             {"params": no_decay, "weight_decay": 0.0}], lr=LR)
    best = (-1.0, float("inf"))
    best_state, best_epoch, stale, history = None, 0, 0, []
    t0 = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = rng.permutation(tr_idx)
        tot = 0.0
        for s in range(0, len(perm), BATCH_TRAIN):
            b = torch.as_tensor(perm[s:s + BATCH_TRAIN])
            loss = F.cross_entropy(_logits(model, config, task, [q[b] for q in q_all], C), y[b])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{task}/{config}: non-finite loss at epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(b)
        model.eval()
        with torch.no_grad():
            b = torch.as_tensor(es_idx)
            lg = _logits(model, config, task, [q[b] for q in q_all], C)
            es_loss = float(F.cross_entropy(lg, y[b]))
            es_acc = float((lg.argmax(-1) == y[b]).float().mean())
        history.append({"epoch": epoch, "train_loss": tot / len(tr_idx), "es_acc": es_acc, "es_loss": es_loss})
        if (es_acc, -es_loss) > (best[0], -best[1]):
            best, best_epoch, stale = (es_acc, es_loss), epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    HEAD_DIR.mkdir(parents=True, exist_ok=True)
    path = HEAD_DIR / f"{task}_{config}.npz"
    meta = {"task": task, "config": config, "encoder": ENCODER, "best_epoch": best_epoch,
            "n_train": int(len(tr_idx)), "n_early_stop": int(len(es_idx))}
    model.export_npz(path, meta)
    # Torch-side test logits: used ONLY to check torch<->NumPy parity at eval, never for selection.
    with torch.no_grad():
        tq = [torch.as_tensor(q) for q in query_inputs(config, task, f, "test")]
        torch_pred = _logits(model, config, task, tq, C).argmax(-1).numpy()
    np.save(HEAD_DIR / f"{task}_{config}_torch_pred.npy", torch_pred)
    n_params = int(sum(p.numel() for p in model.parameters()))
    return {"path": str(path), "best_epoch": best_epoch, "epochs_run": len(history),
            "early_stop_acc": best[0], "early_stop_loss": best[1], "n_params": n_params,
            "n_train": int(len(tr_idx)), "n_early_stop": int(len(es_idx)),
            "train_seconds": time.perf_counter() - t0, "history": history}


def stage_train(tasks: List[str], threads: int = 8) -> None:
    import torch
    torch.set_num_threads(threads)
    for task in tasks:
        f = load_features(task)
        for config in CONFIGS:
            out = HEAD_DIR / f"{task}_{config}_train.json"
            if out.exists():
                print(f"[train] {task}/{config}: cached", flush=True)
                continue
            res = train_one(task, config, f)
            out.write_text(json.dumps(res, indent=1))
            print(f"[train] {task}/{config}: best_epoch {res['best_epoch']}/{res['epochs_run']} "
                  f"es_acc {res['early_stop_acc']:.3f} {res['train_seconds']:.0f}s", flush=True)


# -------------------------------------------------------------------- eval

# 01.PNG, Bespoke Labs "Nimble vs Jev on 13 public benchmarks" (accuracy %, n).
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
LAYA_REPORT = Path(os.environ.get("GC_LAYA_REPORT", str(Path.home() / "inbox" / "decision-models-benchmark-shared" / "laya_full_13_report.json")))


def wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (c - h), 2), round(100 * (c + h), 2)]


def mcnemar_exact(a_ok: np.ndarray, b_ok: np.ndarray) -> Dict[str, float]:
    from math import comb
    b01 = int(np.sum(a_ok & ~b_ok))
    b10 = int(np.sum(~a_ok & b_ok))
    n = b01 + b10
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b01, b10) + 1)) / 2 ** n)
    return {"only_baseline_correct": b01, "only_deep_wide_correct": b10, "p_value": p}


def runtime_for(config: str, path: Path):
    if config == "baseline":
        from gen_zero.causal.rnn_set_adapter import RNNSetAdapterRuntime
        return RNNSetAdapterRuntime.from_npz(path)
    from gen_zero.causal.deep_wide_rnn_set import DeepWideRNNSetRuntime
    return DeepWideRNNSetRuntime.from_npz(path)


def eval_config(task: str, config: str, f: Dict[str, np.ndarray]):
    import contextlib
    rt = runtime_for(config, HEAD_DIR / f"{task}_{config}.npz")
    q = query_inputs(config, task, f, "test")
    C = f["cands"]
    pred, ms = np.empty(len(q[0]), dtype=np.int64), np.empty(len(q[0]))
    if sys.platform != "win32":
        try:
            from threadpoolctl import threadpool_limits
            ctx = threadpool_limits(limits=1)
        except Exception:
            ctx = contextlib.nullcontext()
    else:
        ctx = contextlib.nullcontext()

    def _eval_loop():
        for i in range(len(q[0])):
            t0 = time.perf_counter()
            s = rt.score_pair(q[0][i], q[1][i], C) if len(q) == 2 else rt.score(q[0][i], C)
            ms[i] = (time.perf_counter() - t0) * 1e3
            pred[i] = int(np.argmax(s))

    try:
        with ctx:
            _eval_loop()
    except Exception:
        _eval_loop()

    extra = {}
    if config == "deep_wide":
        extra["layer_spectral_radii"] = [round(float(r), 6) for r in rt.layer_spectral_radii()]
    return pred, ms, extra


def stage_eval(tasks: List[str]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    laya = json.loads(LAYA_REPORT.read_text())["tasks"]
    la0 = loadavg()
    rows = {}
    for task in tasks:
        f = load_features(task)
        test = gd.load_test(task)
        if [r["id"] for r in test] != f["test_ids"].tolist():
            raise ValueError(f"{task}: feature rows do not match the test file order")
        cands = test[0]["candidates"]
        # The ONLY place test labels are read.
        gold = np.array([cands.index(r["ground_truth"]) for r in test])
        prior = np.bincount(f["train_label"], minlength=len(cands))
        majority_ok = gold == int(np.argmax(prior))
        info = f["info"]
        row = {"dataset": PNG[task][0], "n": len(test), "n_expected_01png": PNG[task][1],
               "nimble": PNG[task][2], "jev": PNG[task][3], "best_01png": max(PNG[task][2:4]),
               "laya_a100_reported": laya.get(task, {}).get("accuracy_pct"),
               "test_file": gd.test_file_digest(task), "leakage_gate": info["gate"],
               "majority_class_train_prior_acc": round(100 * float(majority_ok.mean()), 2),
               "pair_cross_difference_used": task in gd.PAIR_FIELDS,
               "encoder_latency_ms_batch1": {
                   "whole_context_median": float(np.median(info["latency_full_ms"])),
                   "pair_fields_median": (float(np.median(info["latency_pair_ms"]))
                                          if "latency_pair_ms" in info else None),
                   "probe_n": len(info["latency_full_ms"])},
               "encoder_batch_seconds": {k[:-8]: round(v, 2) for k, v in info.items()
                                         if k.endswith("_seconds")},
               "configs": {}}
        oks = {}
        for config in CONFIGS:
            pred, ms, extra = eval_config(task, config, f)
            ok = pred == gold
            oks[config] = ok
            train = json.loads((HEAD_DIR / f"{task}_{config}_train.json").read_text())
            torch_pred = np.load(HEAD_DIR / f"{task}_{config}_torch_pred.npy")
            acc = 100 * float(ok.mean())
            row["configs"][config] = {
                "correct": int(ok.sum()), "accuracy": round(acc, 2), "wilson95": wilson(int(ok.sum()), len(ok)),
                "delta_vs_nimble": round(acc - PNG[task][2], 2), "delta_vs_jev": round(acc - PNG[task][3], 2),
                "delta_vs_best_01png": round(acc - max(PNG[task][2:4]), 2),
                "head_latency_ms": {"median": float(np.median(ms)), "p95": float(np.percentile(ms, 95)),
                                    "mean": float(ms.mean())},
                "torch_numpy_argmax_agreement": float(np.mean(torch_pred == pred)),
                "pred_counts": np.bincount(pred, minlength=len(cands)).tolist(),
                "best_epoch": train["best_epoch"], "epochs_run": train["epochs_run"],
                "early_stop_acc": round(100 * train["early_stop_acc"], 2), "n_params": train["n_params"],
                "n_train": train["n_train"], "train_seconds": round(train["train_seconds"], 1), **extra}
        row["gold_counts"] = np.bincount(gold, minlength=len(cands)).tolist()
        row["mcnemar_baseline_vs_deep_wide"] = mcnemar_exact(oks["baseline"], oks["deep_wide"])
        rows[task] = row
        print(f"[eval] {task}: baseline {row['configs']['baseline']['accuracy']:.1f} "
              f"deep_wide {row['configs']['deep_wide']['accuracy']:.1f} "
              f"best01png {row['best_01png']}", flush=True)
    report = build_report(rows, la0)
    (RESULTS / "01png_grand_challenge_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    (RESULTS / "01png_grand_challenge_report.md").write_text(render_md(report))
    print(f"[eval] wrote {RESULTS / '01png_grand_challenge_report.json'}", flush=True)


def build_report(rows: Dict[str, dict], la0: List[float]) -> dict:
    agg = {}
    full = len(rows) == len(gd.TASKS)
    for config in CONFIGS:
        accs = [r["configs"][config]["accuracy"] for r in rows.values()]
        macro = float(np.mean(accs))
        agg[config] = {
            "macro_avg_13": round(macro, 2) if full else None,
            "macro_avg_evaluated": round(macro, 2),
            "micro_acc": round(100 * sum(r["configs"][config]["correct"] for r in rows.values())
                               / sum(r["n"] for r in rows.values()), 2),
            "delta_macro_vs_nimble": round(macro - PNG_AVG["nimble"], 2) if full else None,
            "delta_macro_vs_jev": round(macro - PNG_AVG["jev"], 2) if full else None,
            "tasks_beating_best_01png": sorted(t for t, r in rows.items()
                                               if r["configs"][config]["delta_vs_best_01png"] > 0),
            "tasks_beating_best_01png_AND_majority": sorted(
                t for t, r in rows.items() if r["configs"][config]["delta_vs_best_01png"] > 0
                and r["configs"][config]["accuracy"] > r["majority_class_train_prior_acc"]),
            "head_latency_ms_median_over_tasks": float(np.median(
                [r["configs"][config]["head_latency_ms"]["median"] for r in rows.values()])),
        }
    agg["majority_class_macro"] = round(float(np.mean([r["majority_class_train_prior_acc"]
                                                       for r in rows.values()])), 2)
    agg["laya_a100_reported_macro_13"] = round(float(np.mean([r["laya_a100_reported"] for r in rows.values()])), 2) \
        if full else None
    agg["png_reference_avg"] = PNG_AVG
    return {
        "title": "01.PNG grand challenge: 13 public benchmarks, 3,880 test records",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(sys.argv),
        "host": {"platform": platform.platform(), "cpu_count": os.cpu_count(),
                 "loadavg_at_eval_start": la0, "loadavg_at_eval_end": loadavg(),
                 "note": "shared box; latency is indicative, not a clean-room number"},
        "protocol": {
            "encoder": ENCODER, "encoder_mode": ENCODER_MODE,
            "encoder_precision": "fp32" if ENCODER_MODE == "hf" else "GGUF quantized (see encoder label)",
            "device": "cpu" if ENCODER_MODE == "hf" else "gpu llama-server (encoder latency NOT comparable with cpu)",
            "pooling": ([f"{kind}@layer{layer}" for kind, layer in POOL] if ENCODER_MODE == "hf"
                        else ["server --pooling last (one pooled vector)"]), "max_tokens": MAX_TOK,
            "max_tokens_by_task": {task: TASK_MAX_TOK.get(task, MAX_TOK) for task in rows},
            "truncation": f"first {HEAD_TOK} + last {MAX_TOK - HEAD_TOK} tokens",
            "train_records_per_task_max": {t: gd.n_train_for(t) for t in gd.TASKS},
            "early_stop_fraction_of_train": EARLY_STOP_FRACTION,
            "optimizer": {"name": "AdamW", "lr": LR, "weight_decay": WD, "batch": BATCH_TRAIN,
                          "max_epochs": EPOCHS, "patience": PATIENCE, "seed": TRAIN_SEED},
            "configs": {
                "baseline": "ParallelRNNSetAdapter d=256 rank=16 T=6, 1 set layer x 4 heads; query = pooled whole context",
                "deep_wide": "DeepWideRNNSetAdapter d=512 rank=16 T=6, 3 residual RNN layers rho_max 0.95/0.85/0.75, "
                             "2 set layers x 8 heads SwiGLU; query = cross-difference of separately encoded "
                             "fields on multinli/paws/vitaminc, pooled whole context elsewhere"},
            "scoring": "argmax over the task's candidate list vs ground_truth, exact match",
            "comparability_caveat": "Nimble and Jev in 01.PNG are zero-shot judges; both configs here are "
                                    "per-task heads supervised on public train splits (leakage-gated). "
                                    "Laya numbers are copied from its A100 report, not re-run.",
        },
        "aggregate": agg,
        "tasks": rows,
    }


def render_md(rep: dict) -> str:
    a, rows = rep["aggregate"], rep["tasks"]
    L = [f"# {rep['title']}", "",
         f"Generated {rep['generated_utc']} by `{rep['command']}`. Raw data: "
         "`benchmarks/results/01png_grand_challenge_report.json`.", "",
         "## Headline", ""]
    for c in CONFIGS:
        L.append(f"- **{c}**: macro avg {a[c]['macro_avg_evaluated']}% over {len(rows)} tasks "
                 f"(micro {a[c]['micro_acc']}%); Nimble 74.8%, Jev 76.0% "
                 f"(delta vs Jev {a[c]['delta_macro_vs_jev']}). Beats the 01.PNG best on "
                 f"{len(a[c]['tasks_beating_best_01png'])} task(s): {', '.join(a[c]['tasks_beating_best_01png']) or 'none'}; "
                 f"of those, also above the majority-class floor: "
                 f"{', '.join(a[c]['tasks_beating_best_01png_AND_majority']) or 'none'}.")
    L += [f"- Majority-class (train prior) macro: {a['majority_class_macro']}%. "
          f"Laya (A100 report, not re-run) macro: {a['laya_a100_reported_macro_13']}%.", "",
          "## Per task (accuracy %, 95% Wilson CI)", "",
          "| Task | n | Baseline | Deep-Wide | McNemar p | Majority | Nimble | Jev | Best | Δ DW vs best | Laya |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t, r in rows.items():
        b, d = r["configs"]["baseline"], r["configs"]["deep_wide"]
        L.append(f"| {r['dataset']}{' (pair)' if r['pair_cross_difference_used'] else ''} | {r['n']} | "
                 f"{b['accuracy']} [{b['wilson95'][0]}, {b['wilson95'][1]}] | "
                 f"{d['accuracy']} [{d['wilson95'][0]}, {d['wilson95'][1]}] | "
                 f"{r['mcnemar_baseline_vs_deep_wide']['p_value']:.3g} | {r['majority_class_train_prior_acc']} | "
                 f"{r['nimble']} | {r['jev']} | {r['best_01png']} | {d['delta_vs_best_01png']:+.1f} | "
                 f"{r['laya_a100_reported']} |")
    L += ["", "## CPU latency (ms per record, median)", "",
          "| Task | encoder batch-1, whole context | encoder batch-1, pair fields | baseline head | deep-wide head |",
          "|---|---:|---:|---:|---:|"]
    for t, r in rows.items():
        e = r["encoder_latency_ms_batch1"]
        pair = f"{e['pair_fields_median']:.0f}" if e["pair_fields_median"] is not None else "-"
        L.append(f"| {r['dataset']} | {e['whole_context_median']:.0f} | {pair} | "
                 f"{r['configs']['baseline']['head_latency_ms']['median']:.2f} | "
                 f"{r['configs']['deep_wide']['head_latency_ms']['median']:.2f} |")
    L += ["", f"Host loadavg at eval start/end: {rep['host']['loadavg_at_eval_start']} / "
          f"{rep['host']['loadavg_at_eval_end']} on {rep['host']['cpu_count']} CPUs (shared box).", "",
          "## Protocol", "", "```json", json.dumps(rep["protocol"], indent=1, ensure_ascii=False), "```", ""]
    L += honesty_section(rep)
    return "\n".join(L)


def honesty_section(rep: dict) -> List[str]:
    a, rows = rep["aggregate"], rep["tasks"]
    dw, jev = a["deep_wide"]["macro_avg_evaluated"], PNG_AVG["jev"]
    verdict = "above" if dw > jev else "below"
    if len(rows) != len(gd.TASKS):
        return ["## Verdict", "", f"PARTIAL RUN ({len(rows)}/{len(gd.TASKS)} tasks): the macro average is "
                "not comparable to the 01.PNG 13-task averages; no verdict.", ""]
    gate_zero = all(r["leakage_gate"]["id_overlap"] == 0 and r["leakage_gate"]["text_overlap"] == 0
                    and r["leakage_gate"]["family_overlap"] == 0 for r in rows.values())
    parity = min(min(r["configs"][c]["torch_numpy_argmax_agreement"] for c in CONFIGS) for r in rows.values())
    n_ok = all(r["n"] == r["n_expected_01png"] for r in rows.values())
    sig = sorted(t for t, r in rows.items() if r["mcnemar_baseline_vs_deep_wide"]["p_value"] < 0.05)
    return [
        "## Verdict", "",
        f"Deep-Wide macro average is **{dw}%**, {verdict} Jev ({jev}%) by {dw - jev:+.2f} points and "
        f"{dw - PNG_AVG['nimble']:+.2f} vs Nimble ({PNG_AVG['nimble']}%). "
        f"Baseline macro is {a['baseline']['macro_avg_evaluated']}%. "
        f"Baseline vs Deep-Wide differs at p < 0.05 (exact McNemar) on: {', '.join(sig) or 'no task'}.", "",
        "## Status (three classes)", "",
        "### Implemented, with evidence", "",
        f"- Test set = 01.PNG record counts on every task: {n_ok} (sha256 per file in the JSON `test_file`).",
        f"- Leakage gate id/family/normalized-text overlap all zero: {gate_zero} (JSON `leakage_gate`).",
        f"- NumPy CPU runtime reproduces the trained torch head: min argmax agreement {parity:.4f} "
        "(JSON `torch_numpy_argmax_agreement`).",
        "- Per-task accuracy, Wilson 95% CI, deltas vs Nimble/Jev/best, majority-class floor, exact McNemar "
        "baseline vs Deep-Wide: all computed by `stage_eval` from runtime predictions.", "",
        "### Not verified", "",
        "- Latency: shared box (see loadavg); single run; numbers are indicative only.",
        "- One training seed per head; run-to-run variance is not measured.",
        "- Pooling (mean@12 + last@18) was picked on a TRAIN-internal pilot of 3 tasks, not tuned per task.",
        "- No per-item comparison with Nimble/Jev: 01.PNG publishes aggregates only, so no McNemar vs them.", "",
        "### Not done", "",
        "- `causal_moe_engine.py` consensus arbitration and `causal_mcts_rnn.py` search are NOT part of this "
        "run: the two compared configs are single heads.",
        "- Laya was not re-run (owner instruction); its column is copied from `laya_full_13_report.json`.",
        f"- Training capped at {sorted({gd.MAX_TOKEN if gd.n_train_for(t) is None else gd.n_train_for(t) for t in gd.TASKS}, key=str)} records per task "
        "and the encoder is a 0.5B model on CPU; "
        "larger encoders / more data were not tried.",
        "- Cross-difference path is used only on MultiNLI, PAWS, VitaminC (as specified), not on other "
        "two-field tasks (BoolQ, SQuAD 2.0, HelpSteer2, SummEval).", "",
        "### Comparability", "",
        "- Nimble and Jev are zero-shot judges; these heads are supervised on each task's public train split. "
        "A higher number here is not the same claim as theirs.",
        "- A task where accuracy beats 01.PNG but not the majority-class floor (e.g. a skewed label prior) is "
        "not evidence of skill; see `tasks_beating_best_01png_AND_majority`.", ""]


def main() -> None:
    global ART, FEAT_DIR, HEAD_DIR, RESULTS, ENCODER, LAYA_REPORT, ENCODER_MODE, SERVER_URL
    global MAX_TOK, HEAD_TOK, POOL, TASK_MAX_TOK, SERVER_WORKERS, SERVER_TIMEOUT
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["encode", "train", "eval", "all"])
    ap.add_argument("--tasks", default=",".join(gd.TASKS))
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--max-tok", type=int, default=MAX_TOK)
    ap.add_argument("--head-tok", type=int, default=HEAD_TOK)
    ap.add_argument("--task-max-tok", default=os.environ.get("GC_TASK_MAX_TOK", ""))
    ap.add_argument("--pool-layers", default=",".join(f"{kind}@{layer}" for kind, layer in POOL))
    ap.add_argument("--workers", type=int, default=SERVER_WORKERS)
    ap.add_argument("--timeout", type=float, default=SERVER_TIMEOUT)
    ap.add_argument("--art", type=Path, default=ART)
    ap.add_argument("--raw", type=Path, default=gd.RAW)
    ap.add_argument("--test-dir", type=Path, default=gd.TEST_DIR)
    ap.add_argument("--encoder-path", default=ENCODER)
    ap.add_argument("--encoder-mode", choices=["hf", "server"], default=ENCODER_MODE)
    ap.add_argument("--server-url", default=SERVER_URL)
    ap.add_argument("--results-dir", type=Path, default=RESULTS)
    ap.add_argument("--laya-report", type=Path, default=LAYA_REPORT)
    args = ap.parse_args()
    if not 0 <= args.head_tok < args.max_tok:
        ap.error("--max-tok must exceed nonnegative --head-tok")
    if args.threads < 1 or args.workers < 1 or args.timeout <= 0:
        ap.error("--threads and --workers must be positive; --timeout must exceed zero")
    MAX_TOK, HEAD_TOK = args.max_tok, args.head_tok
    TASK_MAX_TOK = gd.parse_task_max_tok(args.task_max_tok, HEAD_TOK)
    POOL = parse_pool_layers(args.pool_layers)
    SERVER_WORKERS, SERVER_TIMEOUT = args.workers, args.timeout
    ART, FEAT_DIR, HEAD_DIR = args.art, args.art / "features", args.art / "heads"
    RESULTS, ENCODER, LAYA_REPORT = args.results_dir, args.encoder_path, args.laya_report
    ENCODER_MODE, SERVER_URL = args.encoder_mode, args.server_url
    gd.RAW, gd.TEST_DIR = args.raw, args.test_dir
    tasks = [t for t in args.tasks.split(",") if t]
    unknown = sorted(set(tasks) - set(gd.TASKS))
    if unknown:
        raise SystemExit(f"unknown tasks: {unknown}")
    if args.stage in ("encode", "all"):
        stage_encode(tasks, args.threads)
    if args.stage in ("train", "all"):
        stage_train(tasks, args.threads)
    if args.stage in ("eval", "all"):
        stage_eval(tasks)


if __name__ == "__main__":
    main()
