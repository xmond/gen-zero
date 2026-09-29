#!/usr/bin/env python3
"""Laya (421M ModernBERT, single-step) vs Gen-Zero (Parallel RNN + Set-Transformer, +/- causal MCTS).

Two test beds, both run on this CPU:

1. Deadlock gridworld (benchmarks/suites/deadlock_torus_env.py), built here:
   * 30 deceptive irreversible traps (corridor depth 2..6): straight toward the
     food is safe now, but enters a closed no-reverse corridor. A trap counts as
     survived if the agent is alive AND still viable (exact graph search) after
     depth+3 steps.
   * 60 regular states where single-step information is enough. Accuracy =
     chosen action is on a shortest safe route.
   Every arm sees the same map. Laya gets it as ASCII text plus a description
   of each move's immediate cell; Gen-Zero gets the same map as linear readouts
   on its latent state.
2. Real multiple-choice text (ARC, MMLU-Pro, banking77, APPS execution):
   * Gen-Zero arms run on artifacts/qwen35_9b/parity_val200.npz, the 200 cached
     Qwen3.5-9B validation features of the trained adapter.
   * Laya runs on 200 rows of the SAME validation split with the same per-task
     counts, but NOT the same rows: the parity file stores no record ids and its
     order cannot be reconstructed, and the A100 host that could re-extract
     features is unreachable. The report states this next to the numbers.

Arms: Laya (real checkpoint convaiinnovations/laya-typed-decisions, CPU),
B = RNNSetAdapterRuntime alone, Full = CausalMCTSRNN (adaptive gate), plus
ablations (always-MCTS, entropy-only gate, dynamics fit from less data).

What is learned vs given (so nothing is hidden):
* The latent dynamics are FIT from random-walk transitions (Kabsch + Lie log).
  The true env step is used only to execute the chosen action and score it.
* The prior (Set-Transformer) is trained only on a one-step signal
  (+1 food, -1 death, -0.1 * distance change); it never sees survival labels.
* The grid encoder is a fixed random orthonormal code per state (no pretrained
  encoder exists for these states on this box).

Stages: --stage train | laya | eval | all. Laya calls are cached in a JSONL
file keyed by the exact prompt, so eval replays Laya without re-running it.
"""
from __future__ import annotations

import os

THREADS = os.environ.get("GZ_BENCH_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, THREADS)
os.environ.setdefault("USE_TF", "0")

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "benchmarks" / "suites"))

from deadlock_torus_env import (ACTION_NAMES, N_ACTIONS, Layout, TorusWorld,  # noqa: E402
                                latent_codes, make_regular, make_trap)
from gen_zero.causal.causal_mcts_rnn import (CausalMCTSRNN, ContractionDamper,  # noqa: E402
                                             LatentReadout, LieLatentDynamics)
from gen_zero.causal.causal_moe_engine import calibrate_temperature  # noqa: E402
from gen_zero.causal.rnn_set_adapter import RNNSetAdapterRuntime, _softmax  # noqa: E402

GRID_N = 10
LATENT_DIM = 408                 # >= 4*10*10 states, orthonormal codes
CODE_SEED = 7
ART = REPO / "benchmarks" / "artifacts" / "zero"
PRIOR_NPZ = ART / "deadlock_rnn_set_prior_v1.npz"
DYN_NPZ = ART / "deadlock_lie_dynamics_v1.npz"
CALIB_JSON = ART / "deadlock_calibration_v1.json"
RESULTS = REPO / "benchmarks" / "results"
LAYA_MODELS = {  # key -> (hub id, cache file)
    "typed": ("convaiinnovations/laya-typed-decisions", RESULTS / "laya_comparison_laya_cache.jsonl"),
    "general": ("convaiinnovations/laya", RESULTS / "laya_comparison_laya_general_cache.jsonl"),
}
REPORT = RESULTS / "laya_comparison_report.json"
REPORT_MD = RESULTS / "laya_comparison_report.md"
QWEN_ADAPTER = REPO / "artifacts" / "qwen35_9b" / "zero_rnn_set_adapter_qwen35_9b.npz"
QWEN_PARITY = REPO / "artifacts" / "qwen35_9b" / "parity_val200.npz"
MCQ_DATA = REPO / "benchmarks" / "data" / "open_training_pool_natural_5k.jsonl"
EXPLORE_FULL = 30000
EXPLORE_ABLATION = (1500, 4000)
TRAP_DEPTHS = (2, 3, 4, 5, 6)
TRAPS_PER_DEPTH = 6
N_REGULAR = 60


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- items

def trap_items(world: TorusWorld, seed: int, depths: Sequence[int], per_depth: int) -> List[Layout]:
    rng = np.random.default_rng(seed)
    out: List[Layout] = []
    for depth in depths:
        made = 0
        while made < per_depth:
            row = int(rng.integers(2, world.n - 1 - 3))
            lay = make_trap(world, depth, row, int(rng.integers(4)), rng,
                            clutter=int(rng.integers(2, 7)), name=f"trap_d{depth}_{made}")
            if lay is not None:
                out.append(lay)
                made += 1
    return out


def regular_items(world: TorusWorld, seed: int, count: int) -> List[Layout]:
    rng = np.random.default_rng(seed)
    out: List[Layout] = []
    while len(out) < count:
        lay = make_regular(world, rng, density=float(rng.uniform(0.06, 0.16)), name=f"reg_{len(out)}")
        if lay is not None:
            out.append(lay)
    return out


SPLITS = {"train": 1000, "calib": 2000, "test": 3000}


def test_items(world: TorusWorld) -> Tuple[List[Layout], List[Layout]]:
    return (trap_items(world, SPLITS["test"], TRAP_DEPTHS, TRAPS_PER_DEPTH),
            regular_items(world, SPLITS["test"] + 1, N_REGULAR))


def layout_digest(lays: Sequence[Layout]) -> str:
    h = hashlib.sha256()
    for l in lays:
        h.update(l.lethal.tobytes() + repr((l.food, l.start, l.kind, l.depth)).encode())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- latent world

def make_readout(world: TorusWorld, codes: np.ndarray, lay: Layout, G: Optional[np.ndarray] = None) -> LatentReadout:
    G = world.state_features(lay) if G is None else G
    lethal, goal = world.state_masks(lay)
    return LatentReadout(features=G.T @ codes, fail=codes.T @ lethal.astype(np.float64),
                         success=codes.T @ goal.astype(np.float64), potential=codes.T @ (-G[:, 2].astype(np.float64)))


def explore(world: TorusWorld, codes: np.ndarray, budget: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random-walk transitions in random layouts. Death restarts the walk.

    Only what an agent observes: the code before, the action, the code after.
    """
    rng = np.random.default_rng(seed)
    S, A, Sn = [], [], []
    lay, s, left = None, 0, 0
    while len(S) < budget:
        if lay is None or left == 0:
            lay = make_regular(world, rng, density=float(rng.uniform(0.05, 0.2)))
            if lay is None:
                continue
            s, left = lay.start, 60
        a = int(rng.integers(N_ACTIONS))
        s2, st = world.step(lay, s, a)
        S.append(s), A.append(a), Sn.append(s2)
        left -= 1
        s = s2
        if st != "alive":
            left = 0
    S, A, Sn = np.array(S), np.array(A), np.array(Sn)
    return codes[S], A, codes[Sn]


def coverage(world: TorusWorld, A: np.ndarray, Zs: np.ndarray, codes: np.ndarray) -> float:
    s_idx = np.argmax(Zs @ codes.T, axis=1)
    return len(set(zip(s_idx.tolist(), A.tolist()))) / float(world.S * N_ACTIONS)


# --------------------------------------------------------------------------- stage: train

def one_step_records(world: TorusWorld, lays: Sequence[Layout], rng, per_layout: int):
    q, C, off, lab, tasks = [], [], [0], [], []
    for lay in lays:
        G = world.state_features(lay)
        via = world.viable(lay)
        lethal, goal = world.state_masks(lay)
        pool = np.flatnonzero(~lethal & ~goal)
        picks = list(rng.choice(pool, size=min(per_layout, len(pool)), replace=False))
        if lay.kind == "trap":
            picks.append(lay.start)
        for s in picks:
            r = world.one_step_reward(lay, int(s))
            best = np.flatnonzero(r == r.max())
            q.append(G[s])
            C.extend(G[world.next[s]])
            off.append(off[-1] + N_ACTIONS)
            lab.append(int(rng.choice(best)))                    # ties: random, not index order
            tasks.append(lay.kind)
    return (np.asarray(q, np.float32), np.asarray(C, np.float32), np.asarray(off, np.int64),
            np.asarray(lab, np.int64), np.asarray(tasks))


def set_nll(probs: np.ndarray, opt: Sequence[int]) -> float:
    return -math.log(max(float(probs[list(opt)].sum()), 1e-12))


def fit_set_temperature(logits: Sequence[np.ndarray], opts: Sequence[Sequence[int]]) -> float:
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 81))
    losses = [np.mean([set_nll(_softmax(l / T), o) for l, o in zip(logits, opts)]) for T in grid]
    return float(grid[int(np.argmin(losses))])


def stage_train(args) -> dict:
    import torch
    from rnn_set_adapter_torch import AdapterData, train_adapter
    torch.set_num_threads(int(THREADS))
    world = TorusWorld(GRID_N)
    codes = latent_codes(world.S, LATENT_DIM, CODE_SEED)
    ART.mkdir(parents=True, exist_ok=True)
    rep: dict = {}

    # 1. latent dynamics from exploration only
    t0 = time.time()
    Z, A, Zn = explore(world, codes, EXPLORE_FULL, seed=11)
    dyn = LieLatentDynamics.fit(Z, A, Zn, N_ACTIONS)
    dyn.save_npz(DYN_NPZ)
    rep["dynamics"] = {"transitions": int(len(A)), "state_action_coverage": coverage(world, A, Z, codes),
                       "fit_sec": time.time() - t0, **dyn.fit_report,
                       "orthogonality_error": dyn.orthogonality_error}
    log(f"dynamics fit: {rep['dynamics']}")

    # 2. prior trained on the one-step signal only
    rng = np.random.default_rng(SPLITS["train"])
    lays = (regular_items(world, SPLITS["train"], 300)
            + trap_items(world, SPLITS["train"] + 1, (1, 2, 3, 4, 5, 6), 30))
    q, C, off, lab, tasks = one_step_records(world, lays, rng, per_layout=25)
    ids = [hashlib.sha256(f"{i}".encode()).digest()[0] for i in range(len(lab))]
    is_val = np.asarray([b % 5 == 0 for b in ids])
    data = AdapterData(q=q, cands=C, offsets=off, labels=lab, tasks=tasks, is_val=is_val)
    model, trep = train_adapter(data, d=12, rank=4, think_steps=6, n_heads=2, n_layers=1, epochs=40,
                                lr=3e-3, batch_size=128, dropout=0.0, device="cpu", seed=0,
                                log=lambda m: None)
    model.export_npz(PRIOR_NPZ, {"task": "deadlock one-step prior", "label": "argmax one-step reward",
                                 "records": int(len(lab))})
    rt = RNNSetAdapterRuntime.from_npz(PRIOR_NPZ)
    rep["prior"] = {k: trep[k] for k in ("n_train", "n_val", "train_acc", "val_acc", "sigma_max_A",
                                         "spectral_radius_A", "best_epoch")}
    rep["prior"]["records"] = int(len(lab))
    rep["prior"]["runtime_sigma_max_A"] = rt.sigma_max_A
    log(f"prior trained: {rep['prior']}")

    # 3. calibration split: temperature + gate thresholds (never the test items)
    cal_traps = trap_items(world, SPLITS["calib"], TRAP_DEPTHS, 3)
    cal_regs = regular_items(world, SPLITS["calib"] + 1, 40)
    logits, opts = [], []
    eng = CausalMCTSRNN(rt, dyn, None)
    ents, cfs = [], []
    for lay in cal_traps + cal_regs:
        G = world.state_features(lay)
        s = lay.start
        lg = rt.score(G[s], G[world.next[s]])
        logits.append(lg)
        opts.append(world.optimal_actions(lay, s))
    T = fit_set_temperature(logits, opts)
    eng.T = T
    for lay in cal_traps + cal_regs:
        G = world.state_features(lay)
        ro = make_readout(world, codes, lay, G)
        d = eng.decide(codes[lay.start].astype(np.float32), ro, mode="fast")
        ents.append(d.entropy_norm)
        qv, Cv = ro.features @ codes[lay.start], (ro.features @ dyn.step_all(codes[lay.start].astype(np.float32)).T).T
        cfs.append(eng.counterfactual_sensitivity(qv.astype(np.float32), Cv.astype(np.float32), d.prior)[0])
    calib = {"temperature": T, "tau_entropy": float(np.quantile(ents, 0.8)),
             "tau_cf": float(max(np.quantile(cfs, 0.9), 1e-6)),
             "rule": "T: min set-NLL on calib split; tau_entropy = q80(entropy); tau_cf = q90(cf sensitivity)",
             "calib_items": len(cal_traps) + len(cal_regs)}
    CALIB_JSON.write_text(json.dumps(calib, indent=2))
    rep["calibration"] = calib
    log(f"calibration: {calib}")
    return rep


# --------------------------------------------------------------------------- policies

@dataclass
class Choice:
    action: int
    probs: np.ndarray
    wall_ms: float
    cpu_ms: float
    info: dict


def timed(fn: Callable[[], Tuple[int, np.ndarray, dict]]) -> Choice:
    w0, c0 = time.perf_counter(), time.process_time()
    a, p, info = fn()
    return Choice(a, np.asarray(p, np.float64), (time.perf_counter() - w0) * 1e3,
                  (time.process_time() - c0) * 1e3, info)


class GridContext:
    """Per-layout cached features/readouts so arms pay only for decisions."""

    def __init__(self, world: TorusWorld, codes: np.ndarray, lay: Layout) -> None:
        self.G = world.state_features(lay)
        self.ro = make_readout(world, codes, lay, self.G)


def policy_B(world, codes, rt: RNNSetAdapterRuntime, T: float):
    def pol(lay, ctx: GridContext, s: int) -> Choice:
        def run():
            # Candidate descriptions come from the env (the cell each move enters), as Laya gets.
            p = _softmax(rt.score(ctx.G[s], ctx.G[world.next[s]]) / np.float32(T))
            return int(np.argmax(p)), p, {}
        return timed(run)
    return pol


def policy_full(codes, eng: CausalMCTSRNN, mode: str):
    def pol(lay, ctx: GridContext, s: int) -> Choice:
        z = codes[s].astype(np.float32)

        def run():
            d = eng.decide(z, ctx.ro, mode=mode)
            return d.action, d.probs, {"mode": d.mode, "triggers": d.triggers, "sims": d.n_simulations,
                                       "entropy": d.entropy_norm, "cf": d.cf_sensitivity,
                                       "probe": d.probe_return, "act_steps": d.act_steps}
        return timed(run)
    return pol


# --------------------------------------------------------------------------- Laya

def _import_laya():
    try:
        import laya  # noqa: F401
    except ImportError:
        src = os.environ.get("LAYA_SRC")
        if not src:
            raise SystemExit("laya is not importable: `pip install laya==0.3.7` or set LAYA_SRC to an "
                             "extracted laya wheel")
        sys.path.insert(0, src)
    import laya
    return laya


def prompt_key(state: str, questions: dict) -> str:
    return hashlib.sha256(json.dumps([state, questions], sort_keys=True).encode()).hexdigest()


class LayaCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: Dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.rows[r["key"]] = r

    def get(self, key: str) -> Optional[dict]:
        return self.rows.get(key)

    def put(self, row: dict) -> None:
        self.rows[row["key"]] = row
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")


class LayaRunner:
    """Calls the real checkpoint on CPU, or replays the cache (live=False)."""

    def __init__(self, cache: LayaCache, live: bool, model_id: str) -> None:
        self.model_id = model_id
        self.cache = cache
        self.live = live
        self.agent = None
        self.embed = None
        self.load_sec = None

    def _load(self):
        if self.agent is None:
            import torch
            torch.set_num_threads(int(THREADS))
            laya = _import_laya()
            t0 = time.time()
            self.agent = laya.load(self.model_id, device="cpu")
            self.load_sec = time.time() - t0
            self._laya = laya
            log(f"Laya {self.model_id} loaded in {self.load_sec:.1f}s (threads={torch.get_num_threads()})")

    def ask(self, state: str, criteria: Dict[str, str], instructions: str) -> dict:
        qs = {"q": {"type": "choice", "instructions": instructions, "criteria": criteria}}
        key = prompt_key(state, qs)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        if not self.live:
            raise KeyError(f"Laya cache miss for key {key[:12]}; run --stage laya first")
        self._load()
        load = os.getloadavg()[0]
        w0, c0 = time.perf_counter(), time.process_time()
        row = {"key": key, "labels": list(criteria.keys()), "error": None, "shortlist": False}
        try:
            out = self.agent.system_one(state, qs)
        except ValueError as e:
            # Laya's own documented route for large label sets: embedding shortlist, then the head.
            if "exceed head_max_len" not in str(e):
                raise
            if self.embed is None:
                self.embed = self._laya.embed_fn_from_agent(self.agent)
            try:
                out = self._laya.predict_shortlist(self.agent, state, qs, self.embed)
                row["shortlist"] = True
            except Exception as e2:  # recorded, scored as wrong
                out = None
                row["error"] = f"{type(e).__name__}: {e} | shortlist: {type(e2).__name__}: {e2}"
        row["wall_ms"] = (time.perf_counter() - w0) * 1e3
        row["cpu_ms"] = (time.process_time() - c0) * 1e3
        row["loadavg_1m"] = load
        if out is not None:
            ans = out["answers"]["q"]
            row["choice"] = ans["choice"]
            row["probabilities"] = ans["probabilities"]
            row["input_tokens"] = out["usage"]["input_tokens"]
        self.cache.put(row)
        return row


GRID_INSTRUCTIONS = "Which move should the agent make now? Choose the move that keeps it alive and gets it to the food."


def policy_laya(world: TorusWorld, runner: LayaRunner):
    def pol(lay, ctx, s: int) -> Choice:
        state, crit = world.render_text(lay, s)
        row = runner.ask(state, crit, GRID_INSTRUCTIONS)
        if row.get("error"):
            raise RuntimeError(f"Laya failed on a 3-option grid prompt: {row['error']}")
        p = np.array([row["probabilities"].get(n, 0.0) for n in ACTION_NAMES], np.float64)
        p = p / p.sum()
        return Choice(ACTION_NAMES.index(row["choice"]), p, row["wall_ms"], row["cpu_ms"],
                      {"loadavg_1m": row["loadavg_1m"], "cached": True})
    return pol


# --------------------------------------------------------------------------- episodes & metrics

def run_trap(world: TorusWorld, lay: Layout, ctx, pol) -> dict:
    s, steps, decisions = lay.start, lay.depth + 3, []
    status = "alive"
    for _ in range(steps):
        ch = pol(lay, ctx, s)
        decisions.append(ch)
        s, status = world.step(lay, s, ch.action)
        if status != "alive":
            break
    if status == "alive":
        status = "survived" if world.viable(lay)[s] else "doomed"
    elif status == "goal":
        status = "survived"
    first = decisions[0]
    return {"name": lay.name, "depth": lay.depth, "outcome": status, "first_action": first.action,
            "first_probs": first.probs.tolist(), "decisions": len(decisions),
            "wall_ms": [c.wall_ms for c in decisions], "cpu_ms": [c.cpu_ms for c in decisions],
            "info": [c.info for c in decisions]}


def run_goal_episode(world: TorusWorld, lay: Layout, ctx, pol, horizon: int = 40) -> str:
    s = lay.start
    for _ in range(horizon):
        s, st = world.step(lay, s, pol(lay, ctx, s).action)
        if st != "alive":
            return st
    return "timeout"


def ece_brier(conf: Sequence[float], correct: Sequence[bool], bins: int = 10) -> Tuple[float, float]:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= 0) & (conf <= edges[1])
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(ece), float(np.mean((conf - correct) ** 2))


def multiclass_brier(probs: Sequence[np.ndarray], labels: Sequence[int]) -> float:
    return float(np.mean([np.sum((p - np.eye(len(p))[y]) ** 2) for p, y in zip(probs, labels)]))


def lat_stats(xs: Sequence[float]) -> dict:
    a = np.asarray(xs, float)
    return {"n": int(a.size), "median": float(np.median(a)), "p90": float(np.quantile(a, 0.9)),
            "mean": float(a.mean()), "min": float(a.min())}


def eval_grid_arm(world, traps, regs, ctxs, pol, name: str, goal_episodes: bool) -> dict:
    load0 = os.getloadavg()[0]
    tr = [run_trap(world, l, ctxs[id(l)], pol) for l in traps]
    reg_rows, conf, correct = [], [], []
    for l in regs:
        ch = pol(l, ctxs[id(l)], l.start)
        opt = world.optimal_actions(l, l.start)
        reg_rows.append({"name": l.name, "action": ch.action, "opt": opt, "correct": ch.action in opt,
                         "probs": ch.probs.tolist(), "wall_ms": ch.wall_ms, "cpu_ms": ch.cpu_ms, "info": ch.info})
    for r in reg_rows:
        conf.append(r["probs"][r["action"]])
        correct.append(r["correct"])
    for t, l in zip(tr, traps):
        opt = world.optimal_actions(l, l.start)
        conf.append(t["first_probs"][t["first_action"]])
        correct.append(t["first_action"] in opt)
    ece, brier = ece_brier(conf, correct)
    by_depth = {}
    for d in sorted({t["depth"] for t in tr}):
        rows = [t for t in tr if t["depth"] == d]
        by_depth[str(d)] = sum(t["outcome"] == "survived" for t in rows) / len(rows)
    walls = [w for t in tr for w in t["wall_ms"]] + [r["wall_ms"] for r in reg_rows]
    cpus = [c for t in tr for c in t["cpu_ms"]] + [r["cpu_ms"] for r in reg_rows]
    out = {
        "arm": name,
        "trap_survival_rate": sum(t["outcome"] == "survived" for t in tr) / len(tr),
        "trap_survived": sum(t["outcome"] == "survived" for t in tr),
        "trap_total": len(tr),
        "trap_outcomes": {k: sum(t["outcome"] == k for t in tr) for k in ("survived", "dead", "doomed")},
        "trap_survival_by_depth": by_depth,
        "trap_first_move_straight_rate": sum(t["first_action"] == 1 for t in tr) / len(tr),
        "regular_accuracy": float(np.mean([r["correct"] for r in reg_rows])),
        "regular_total": len(reg_rows),
        "ece_10bin": ece, "brier_binary": brier, "calibration_items": len(conf),
        "latency_wall_ms": lat_stats(walls), "latency_cpu_ms": lat_stats(cpus),
        "loadavg_1m_at_start": load0, "loadavg_1m_at_end": os.getloadavg()[0],
        "traps": tr, "regular": reg_rows,
    }
    modes = [i.get("mode") for t in tr for i in t["info"]] + [r["info"].get("mode") for r in reg_rows]
    if any(m is not None for m in modes):
        out["mcts_escalation_rate"] = float(np.mean([m == "mcts" for m in modes]))
        trig = [i.get("triggers", {}) for t in tr for i in t["info"]] + [r["info"].get("triggers", {}) for r in reg_rows]
        out["trigger_rates"] = {k: float(np.mean([bool(t.get(k)) for t in trig]))
                                for k in ("entropy", "counterfactual", "probe_conflict")}
        acts = [i.get("act_steps") for t in tr for i in t["info"]] + [r["info"].get("act_steps") for r in reg_rows]
        out["act_steps"] = {"min": int(min(acts)), "max": int(max(acts)), "mean": float(np.mean(acts))}
        fast = [w for w, m in zip(walls, modes) if m == "fast"]
        slow = [w for w, m in zip(walls, modes) if m == "mcts"]
        out["latency_wall_ms_fast_path"] = lat_stats(fast) if fast else None
        out["latency_wall_ms_mcts_path"] = lat_stats(slow) if slow else None
    if goal_episodes:
        res = [run_goal_episode(world, l, ctxs[id(l)], pol) for l in traps + regs]
        out["goal_reach_rate_40_steps"] = float(np.mean([r == "goal" for r in res]))
        out["goal_episode_outcomes"] = {k: res.count(k) for k in ("goal", "dead", "timeout")}
    log(f"{name}: survival {out['trap_survived']}/{out['trap_total']} regular_acc {out['regular_accuracy']:.3f} "
        f"median {out['latency_wall_ms']['median']:.3f} ms")
    return out


# --------------------------------------------------------------------------- MCQ

def val_rows_for_laya(tasks_needed: Dict[str, int], seed: int = 0) -> List[dict]:
    def is_val(i: str) -> bool:
        return int.from_bytes(hashlib.sha256(i.encode()).digest()[:8], "big") % 5 == 0
    rows = [json.loads(l) for l in MCQ_DATA.read_text().splitlines() if l.strip()]
    val = [r for r in rows if is_val(r["id"])]
    rng = np.random.default_rng(seed)
    out = []
    for t in sorted(tasks_needed):
        pool = [r for r in val if r["task"] == t]
        idx = rng.choice(len(pool), size=tasks_needed[t], replace=False)
        out.extend(pool[i] for i in sorted(idx))
    return out


def mcq_prompt(row: dict) -> Tuple[str, Dict[str, str], int]:
    crit = {str(i + 1): c for i, c in enumerate(row["candidates"])}
    return row["context"], crit, row["candidates"].index(row["ground_truth"])


MCQ_INSTRUCTIONS = "Select the single correct option."


def parity_tasks() -> Dict[str, int]:
    with np.load(QWEN_PARITY, allow_pickle=False) as z:
        t = z["tasks"].astype(str)
    return {k: int((t == k).sum()) for k in sorted(set(t.tolist()))}


def eval_mcq_genzero() -> dict:
    rt = RNNSetAdapterRuntime.from_npz(QWEN_ADAPTER)
    with np.load(QWEN_PARITY, allow_pickle=False) as z:
        Q, Cs, off, lab, tasks = z["q"], z["cands"], z["offsets"], z["labels"], z["tasks"].astype(str)
    n = len(lab)
    cal = np.arange(n) % 2 == 0      # calibration half: even rows; evaluation half: odd rows
    scores, walls_b, cpus_b = [], [], []
    for i in range(n):
        C = Cs[off[i]:off[i + 1]]
        w0, c0 = time.perf_counter(), time.process_time()
        scores.append(rt.score(Q[i], C))
        walls_b.append((time.perf_counter() - w0) * 1e3)
        cpus_b.append((time.process_time() - c0) * 1e3)
    T = calibrate_temperature([scores[i] for i in range(n) if cal[i]], [int(lab[i]) for i in range(n) if cal[i]])
    eng = CausalMCTSRNN(rt, None, None, temperature=T)
    dec_cal = [eng.classify(Q[i], Cs[off[i]:off[i + 1]]) for i in range(n) if cal[i]]
    eng.tau_entropy = float(np.quantile([d.entropy_norm for d in dec_cal], 0.8))
    eng.tau_cf = float(max(np.quantile([d.cf_sensitivity for d in dec_cal], 0.9), 1e-6))
    full, walls_f, cpus_f = [], [], []
    for i in range(n):
        C = Cs[off[i]:off[i + 1]]
        w0, c0 = time.perf_counter(), time.process_time()
        full.append(eng.classify(Q[i], C))
        walls_f.append((time.perf_counter() - w0) * 1e3)
        cpus_f.append((time.process_time() - c0) * 1e3)

    def summarize(pred, probs, idx):
        idx = np.asarray(idx)
        corr = [pred[i] == lab[i] for i in idx]
        ece, bb = ece_brier([probs[i][pred[i]] for i in idx], corr)
        per = {}
        for t in sorted(set(tasks[idx].tolist())):
            m = [i for i in idx if tasks[i] == t]
            per[t] = {"n": len(m), "acc": float(np.mean([pred[i] == lab[i] for i in m]))}
        return {"n": int(len(idx)), "accuracy": float(np.mean(corr)), "ece_10bin": ece, "brier_binary": bb,
                "brier_multiclass": multiclass_brier([probs[i] for i in idx], [int(lab[i]) for i in idx]),
                "per_task": per}

    pb = [_softmax(s / np.float32(T)).astype(np.float64) for s in scores]
    predb = [int(np.argmax(s)) for s in scores]
    pf = [d.probs for d in full]
    predf = [d.action for d in full]
    ev = np.flatnonzero(~cal)
    return {
        "items": "artifacts/qwen35_9b/parity_val200.npz (200 cached Qwen3.5-9B val features)",
        "temperature_fit_on": "even rows (100)", "metrics_on": "odd rows (100); accuracy also on all 200",
        "temperature": T, "tau_entropy": eng.tau_entropy, "tau_cf": eng.tau_cf,
        "B": {"eval_half": summarize(predb, pb, ev), "all_200_accuracy": float(np.mean(np.array(predb) == lab)),
              "latency_wall_ms": lat_stats(walls_b), "latency_cpu_ms": lat_stats(cpus_b)},
        "Full": {"eval_half": summarize(predf, pf, ev), "all_200_accuracy": float(np.mean(np.array(predf) == lab)),
                 "argmax_agreement_with_B": float(np.mean(np.array(predf) == np.array(predb))),
                 "would_escalate_rate": float(np.mean([any(d.triggers.values()) for d in full])),
                 "act_steps": {"min": int(min(d.act_steps for d in full)), "max": int(max(d.act_steps for d in full))},
                 "latency_wall_ms": lat_stats(walls_f), "latency_cpu_ms": lat_stats(cpus_f),
                 "note": "no transition model exists for MCQ, so escalation cannot run a search; "
                         "Full = ACT think + counterfactual confidence adjustment"},
        "latency_excludes": "the Qwen3.5-9B forward pass that produced the 4096-D features (run on an A100, "
                            "not measured here). Gen-Zero MCQ latency is head-only.",
    }


def eval_mcq_laya(runner: LayaRunner) -> dict:
    rows = val_rows_for_laya(parity_tasks())
    preds, probs, labels, tasks, walls, cpus, errors, shortlisted, loads = [], [], [], [], [], [], 0, 0, []
    for r in rows:
        state, crit, y = mcq_prompt(r)
        out = runner.ask(state, crit, MCQ_INSTRUCTIONS)
        labels.append(y)
        tasks.append(r["task"])
        walls.append(out["wall_ms"])
        cpus.append(out["cpu_ms"])
        loads.append(out["loadavg_1m"])
        shortlisted += bool(out.get("shortlist"))
        if out.get("error"):
            errors += 1
            preds.append(-1)
            probs.append(np.full(len(crit), 1.0 / len(crit)))
            continue
        p = np.array([out["probabilities"].get(k, 0.0) for k in crit], np.float64)
        p = p / p.sum() if p.sum() > 0 else np.full(len(crit), 1.0 / len(crit))
        probs.append(p)
        preds.append(int(list(crit).index(out["choice"])))
    corr = [p == y for p, y in zip(preds, labels)]
    ece, bb = ece_brier([probs[i][preds[i]] if preds[i] >= 0 else 0.0 for i in range(len(rows))], corr)
    per = {t: {"n": tasks.count(t), "acc": float(np.mean([c for c, tt in zip(corr, tasks) if tt == t]))}
           for t in sorted(set(tasks))}
    return {"items": "200 rows of the same sha256(id)%5==0 validation split, same per-task counts, "
                     "seeded sample; NOT the same rows as the Gen-Zero parity file",
            "ids": [r["id"] for r in rows], "accuracy": float(np.mean(corr)), "ece_10bin": ece,
            "brier_binary": bb, "brier_multiclass": multiclass_brier(probs, labels), "per_task": per,
            "errors_scored_wrong": errors, "used_laya_shortlist": shortlisted,
            "latency_wall_ms": lat_stats(walls), "latency_cpu_ms": lat_stats(cpus),
            "loadavg_1m": lat_stats(loads),
            "note": "laya-typed-decisions is a specialist fine-tuned on four synthetic workflows; these "
                    "tasks are out of its domain. choice:11+ temperature was clamped by laya "
                    "itself (0.10 -> 0.5), so its confidence on >10 options is self-declared uncalibrated."}


# --------------------------------------------------------------------------- microbench

def microbench(dyn: LieLatentDynamics, eng: CausalMCTSRNN, world, codes, lay) -> dict:
    out = {"lie_step_us_single": {}, "lie_step_us_batched_per_state": {}}
    rng = np.random.default_rng(0)
    for d in (64, 128, 256, LATENT_DIM):
        R = np.linalg.qr(rng.standard_normal((d, d)))[0].astype(np.float32)
        z = rng.standard_normal(d).astype(np.float32)
        Zb = rng.standard_normal((d, 256)).astype(np.float32)
        for _ in range(100):
            R @ z
        n = 20000
        t0 = time.perf_counter()
        for _ in range(n):
            R @ z
        out["lie_step_us_single"][str(d)] = (time.perf_counter() - t0) / n * 1e6
        t0 = time.perf_counter()
        for _ in range(200):
            R @ Zb
        out["lie_step_us_batched_per_state"][str(d)] = (time.perf_counter() - t0) / (200 * 256) * 1e6
    ctx = GridContext(world, codes, lay)
    z = codes[lay.start].astype(np.float32)
    for m in ("fast", "mcts"):
        ts = []
        for _ in range(30):
            t0 = time.perf_counter()
            eng.decide(z, ctx.ro, mode=m)
            ts.append((time.perf_counter() - t0) * 1e3)
        out[f"decide_{m}_ms"] = lat_stats(ts)
    return out


def dynamics_check(world, codes, dyn, eng, traps) -> dict:
    """Decode R_a z_s for every (s, a) and compare with the true successor (evaluation only)."""
    Rc = np.einsum("aij,sj->asi", dyn.R64, codes)                   # (A, S, d)
    dec = np.argmax(Rc @ codes.T, axis=2)                           # (A, S)
    wrong = dec != world.next.T
    near = set()
    for lay in traps:
        frontier = {lay.start}
        for _ in range(lay.depth + 3):
            near |= frontier
            frontier = {int(t) for s in frontier for t in world.next[s]}
        near |= frontier
    wrong_near = int(sum(wrong[a, s] for s in near for a in range(N_ACTIONS)))
    # A planner only ever steps FROM non-terminal states; count wrong pairs whose source is
    # non-lethal in the trap's own layout (those are the ones that could mislead a search).
    wrong_usable = 0
    for lay in traps:
        lethal, _ = world.state_masks(lay)
        frontier, seen = {lay.start}, set()
        for _ in range(lay.depth + 3):
            seen |= frontier
            frontier = {int(t) for s in frontier if not lethal[s] for t in world.next[s]}
        seen |= frontier
        wrong_usable += int(sum(wrong[a, s] for s in seen if not lethal[s] for a in range(N_ACTIONS)))
    cells = np.arange(world.S) // 4
    ring = (cells // world.n == 0) | (cells % world.n == 0)
    ctx = GridContext(world, codes, traps[0])
    z0 = codes[traps[0].start].astype(np.float32)
    zd, zu = z0.copy(), z0.copy()
    for t in range(16):
        a = int(np.argmax(ctx.ro._scalars[2] @ dyn.step_all(zd).T))
        zd, zu = dyn.step(zd, a), dyn.step(zu, a)
        if (t + 1) % eng.damper_every == 0:
            zd = eng.damper.apply(zd)
    return {"exact_next_state_fraction": float(1.0 - wrong.mean()),
            "wrong_pairs": int(wrong.sum()), "total_pairs": int(wrong.size),
            "wrong_pairs_on_test_trap_paths": wrong_near,
            "wrong_pairs_from_ring_states": int(wrong[:, ring].sum()),
            "wrong_pairs_from_non_ring_states": int(wrong[:, ~ring].sum()),
            "wrong_pairs_from_non_lethal_states_on_trap_paths": wrong_usable,
            "damper_off_manifold_after_16_steps": eng.damper.off_manifold_norm(zd),
            "off_manifold_after_16_steps_without_damper": eng.damper.off_manifold_norm(zu)}


# --------------------------------------------------------------------------- stages

def load_models():
    world = TorusWorld(GRID_N)
    codes = latent_codes(world.S, LATENT_DIM, CODE_SEED)
    for p in (PRIOR_NPZ, DYN_NPZ, CALIB_JSON):
        if not p.exists():
            raise SystemExit(f"{p} missing: run --stage train first")
    rt = RNNSetAdapterRuntime.from_npz(PRIOR_NPZ)
    dyn = LieLatentDynamics.from_npz(DYN_NPZ)
    calib = json.loads(CALIB_JSON.read_text())
    return world, codes, rt, dyn, calib


def make_engine(rt, dyn, codes, calib, damper_seed: int = 0) -> CausalMCTSRNN:
    damper = ContractionDamper.random(codes.T, rank=4, rho_max=0.95, seed=damper_seed)
    return CausalMCTSRNN(rt, dyn, damper, temperature=calib["temperature"], top_k=2,
                         tau_entropy=calib["tau_entropy"], tau_cf=calib["tau_cf"])


def stage_laya(args) -> None:
    world = TorusWorld(GRID_N)
    traps, regs = test_items(world)
    model_id, cache = LAYA_MODELS[args.laya_model]
    runner = LayaRunner(LayaCache(cache), live=True, model_id=model_id)
    pol = policy_laya(world, runner)
    log(f"Laya grid run: {len(traps)} traps + {len(regs)} regular (items digest {layout_digest(traps + regs)})")
    for i, l in enumerate(traps):
        r = run_trap(world, l, None, pol)
        log(f"  trap {i + 1}/{len(traps)} {l.name}: {r['outcome']} first={ACTION_NAMES[r['first_action']]}")
    for i, l in enumerate(regs):
        pol(l, None, l.start)
        if (i + 1) % 10 == 0:
            log(f"  regular {i + 1}/{len(regs)}")
    if not args.skip_mcq:
        rows = val_rows_for_laya(parity_tasks())
        for i, r in enumerate(rows):
            state, crit, _ = mcq_prompt(r)
            out = runner.ask(state, crit, MCQ_INSTRUCTIONS)
            if (i + 1) % 10 == 0 or out.get("error"):
                log(f"  mcq {i + 1}/{len(rows)} {r['task']} err={out.get('error')}")
    log("Laya stage done")


def stage_eval(args, train_rep: Optional[dict]) -> dict:
    world, codes, rt, dyn, calib = load_models()
    traps, regs = test_items(world)
    ctxs = {id(l): GridContext(world, codes, l) for l in traps + regs}
    eng = make_engine(rt, dyn, codes, calib)
    grid = {}
    grid["B_rnn_set_only"] = eval_grid_arm(world, traps, regs, ctxs, policy_B(world, codes, rt, calib["temperature"]),
                                           "B: Parallel RNN + Set-Transformer (no MCTS)", True)
    grid["Full_adaptive"] = eval_grid_arm(world, traps, regs, ctxs, policy_full(codes, eng, "adaptive"),
                                          "Full: Causal MCTS + RNN + Set (adaptive gate)", True)
    grid["Ablation_always_mcts"] = eval_grid_arm(world, traps, regs, ctxs, policy_full(codes, eng, "mcts"),
                                                 "Ablation: always MCTS", True)
    grid["Ablation_entropy_only_gate"] = eval_grid_arm(world, traps, regs, ctxs,
                                                       policy_full(codes, eng, "adaptive_entropy_only"),
                                                       "Ablation: entropy-only gate", True)
    grid["Ablation_rollout_only_no_tree"] = eval_grid_arm(world, traps, regs, ctxs,
                                                          policy_full(codes, eng, "rollout_only"),
                                                          "Ablation: one rollout per action, no tree", True)
    for budget in EXPLORE_ABLATION:
        Z, A, Zn = explore(world, codes, budget, seed=11)
        dyn_b = LieLatentDynamics.fit(Z, A, Zn, N_ACTIONS)
        eng_b = make_engine(rt, dyn_b, codes, calib)
        key = f"Ablation_dynamics_{budget}_transitions"
        grid[key] = eval_grid_arm(world, traps, regs, ctxs, policy_full(codes, eng_b, "adaptive"),
                                  f"Ablation: dynamics fit from {budget} transitions", False)
        grid[key]["state_action_coverage"] = coverage(world, A, Z, codes)
    laya_notes = []
    mcq = {"genzero": eval_mcq_genzero()}
    for key, (model_id, cache) in LAYA_MODELS.items():
        arm = "A_laya" if key == "typed" else f"A_laya_{key}"
        mkey = "laya" if key == "typed" else f"laya_{key}"
        try:
            runner = LayaRunner(LayaCache(cache), live=False, model_id=model_id)
            grid[arm] = eval_grid_arm(world, traps, regs, ctxs, policy_laya(world, runner),
                                      f"A: Laya {model_id} (421M ModernBERT, single-step)", False)
            grid[arm]["goal_reach_rate_40_steps"] = None
            grid[arm]["goal_reach_note"] = "not run for Laya: seconds per call on this loaded host"
        except KeyError as e:
            laya_notes.append(f"{model_id} grid results missing: {e}")
        try:
            mcq[mkey] = eval_mcq_laya(LayaRunner(LayaCache(cache), live=False, model_id=model_id))
            mcq[mkey]["model_id"] = model_id
        except KeyError as e:
            mcq[mkey] = None
            laya_notes.append(f"{model_id} MCQ results missing: {e}")
    laya_note = " | ".join(laya_notes) or None
    bench = microbench(dyn, eng, world, codes, traps[0])
    dyn_check = dynamics_check(world, codes, dyn, eng, traps)
    arms = {k: v for k, v in grid.items()}
    findings = []
    ro, fu = arms.get("Ablation_rollout_only_no_tree"), arms["Full_adaptive"]
    if ro:
        findings.append(f"One greedy rollout per action with no tree survives {ro['trap_survived']}/{ro['trap_total']} "
                        f"traps (Full: {fu['trap_survived']}/{fu['trap_total']}). Survival comes from model-based "
                        f"look-ahead; the tree adds regular accuracy ({100 * ro['regular_accuracy']:.1f}% -> "
                        f"{100 * fu['regular_accuracy']:.1f}%) and calibration (ECE {ro['ece_10bin']:.3f} -> "
                        f"{fu['ece_10bin']:.3f}), not survival.")
    eo = arms["Ablation_entropy_only_gate"]
    findings.append(f"Entropy-only gating survives {eo['trap_survived']}/{eo['trap_total']}: deceptive traps are "
                    "confident mistakes, so a low-entropy gate sends them down the fast path. The probe-rollout "
                    f"trigger fires on {100 * fu['trigger_rates']['probe_conflict']:.1f}% of Full's decisions.")
    for b in EXPLORE_ABLATION:
        a = arms[f"Ablation_dynamics_{b}_transitions"]
        findings.append(f"Dynamics from {b} transitions ({100 * a['state_action_coverage']:.0f}% state-action "
                        f"coverage): survival {a['trap_survived']}/{a['trap_total']}.")
    mp = fu.get("latency_wall_ms_mcts_path") or {}
    findings.append(f"Latency: Full fast path median {fu['latency_wall_ms_fast_path']['median']:.2f} ms, MCTS path "
                    f"median {mp.get('median', float('nan')):.1f} ms (RFC-072 table claims 2.3-7.4 ms). Lie step: "
                    f"{bench['lie_step_us_single'][str(LATENT_DIM)]:.1f} us single at d={LATENT_DIM}, "
                    f"{bench['lie_step_us_single']['64']:.1f} us at d=64 (RFC claims <=3.2 us).")
    findings.append(f"ACT steps: grid {fu['act_steps']['min']}-{fu['act_steps']['max']}, MCQ "
                    f"{mcq['genzero']['Full']['act_steps']['min']}-{mcq['genzero']['Full']['act_steps']['max']}; "
                    "the halting step tracks the contraction of A, not task difficulty.")
    if mcq.get("laya_general"):
        findings.append(f"MCQ: general Laya {100 * mcq['laya_general']['accuracy']:.1f}%, grid regular "
                        f"accuracy {100 * arms['A_laya_general']['regular_accuracy']:.1f}%, trap survival "
                        f"{arms['A_laya_general']['trap_survived']}/{arms['A_laya_general']['trap_total']}.")
    findings.append(f"Learned dynamics: {100 * dyn_check['exact_next_state_fraction']:.1f}% of all state-action "
                    f"pairs decode to the true next state. Of the {dyn_check['wrong_pairs']} wrong pairs, "
                    f"{dyn_check['wrong_pairs_from_ring_states']} start on the always-lethal border ring (never "
                    f"observed as a source, terminal for the planner) and {dyn_check['wrong_pairs_from_non_ring_states']} "
                    f"elsewhere; {dyn_check['wrong_pairs_from_non_lethal_states_on_trap_paths']} wrong pairs start from a "
                    "non-lethal state reachable within depth+3 steps of a test trap. Damper: off-manifold norm "
                    f"after a 16-step rollout is {dyn_check['damper_off_manifold_after_16_steps']:.2e} with it, "
                    f"{dyn_check['off_manifold_after_16_steps_without_damper']:.2e} without, so in this benchmark "
                    "it has no measurable effect; its contraction is verified by unit tests only.")
    if mcq.get("laya"):
        findings.append(f"MCQ: Laya {100 * mcq['laya']['accuracy']:.1f}% on 200 val rows vs Gen-Zero B "
                        f"{100 * mcq['genzero']['B']['all_200_accuracy']:.1f}% on 200 different val rows. Both Laya "
                        "checkpoints (typed-decisions specialist and general) are near chance on these tasks with "
                        "this prompt format; Laya's own published numbers are on its typed-decisions benchmark, "
                        "which was not run here.")
    findings.append("Regular grid states are selected so that one-step greedy is optimal; 100% there shows "
                    "MCTS does not hurt easy cases, not general competence.")
    report = {
        "suite": "benchmark_laya_comparison",
        "spec": "docs/zero/12-causal-moe-mcts-parallel-rnn-spec.md (RFC-072)",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": {"platform": platform.platform(), "cpu_count": os.cpu_count(), "threads_per_arm": int(THREADS),
                 "loadavg_at_report": os.getloadavg(),
                 "note": "shared host; other sessions ran CPU-heavy jobs during measurement, so every "
                         "latency is inflated and noisy. Compare arms only within the same run."},
        "grid": {"world": f"{GRID_N}x{GRID_N} no-reverse torus with lethal ring, {world.S} states, "
                          f"latent dim {LATENT_DIM}",
                 "items_digest": layout_digest(traps + regs), "traps": len(traps), "regular": len(regs),
                 "trap_survival_rule": "alive and viable (exact search) after depth+3 steps",
                 "arms": {k: {kk: vv for kk, vv in v.items() if kk not in ("traps", "regular")}
                          for k, v in grid.items()},
                 "per_item": {k: {"traps": v["traps"], "regular": v["regular"]} for k, v in grid.items()}},
        "mcq": mcq,
        "microbench": bench,
        "dynamics_check": dyn_check,
        "training": train_rep or json.loads(CALIB_JSON.read_text()),
        "laya_missing": laya_note,
        "findings": findings,
        "honesty": [
            "Traps are built so that the one-step signal points into the corridor; B is trained on that "
            "one-step signal only. B failing deep traps is the expected behaviour of a single-step "
            "mapper, measured, not asserted.",
            "The dynamics are exact permutations on orthonormal codes, which is why a rotation can model "
            "them. A rotation cannot model many-to-one transitions; irreversibility here comes from "
            "terminal readouts.",
            "With full exploration the learned model is essentially exact, so MCTS then plans on a near-"
            "perfect simulator; the limited-exploration ablations show what happens when it is not.",
            "MCQ: Laya and Gen-Zero are scored on different rows of the same split, and Gen-Zero latency "
            "excludes the 9B feature extractor.",
            "Grid latency is head-only for Gen-Zero: per-layout feature and readout construction "
            "(GridContext, ~10 ms per map) is precomputed and not timed. Laya's time is end-to-end "
            "(tokenise + 421M forward). The latency columns are not like-for-like.",
        ],
    }
    return report


def to_markdown(rep: dict) -> str:
    g = rep["grid"]["arms"]
    order = [k for k in ("A_laya", "A_laya_general", "B_rnn_set_only", "Full_adaptive", "Ablation_always_mcts",
                         "Ablation_entropy_only_gate", "Ablation_rollout_only_no_tree") if k in g]
    order += [k for k in g if k.startswith("Ablation_dynamics")]
    lines = ["| Arm | Trap survival | Regular acc | ECE | Brier | Median ms wall (Gen-Zero head-only, Laya end-to-end) "
             "| Median ms CPU | MCTS rate |",
             "|---|---|---|---|---|---|---|---|"]
    for k in order:
        a = g[k]
        lines.append(f"| {a['arm']} | {a['trap_survived']}/{a['trap_total']} ({100 * a['trap_survival_rate']:.1f}%) "
                     f"| {100 * a['regular_accuracy']:.1f}% | {a['ece_10bin']:.3f} | {a['brier_binary']:.3f} "
                     f"| {a['latency_wall_ms']['median']:.3f} | {a['latency_cpu_ms']['median']:.3f} "
                     f"| {a.get('mcts_escalation_rate', 0.0) * 100:.1f}% |")
    m = rep["mcq"]
    lines += ["", "| MCQ arm | Items | Accuracy | ECE | Brier (binary) | Median ms (wall) |", "|---|---|---|---|---|---|"]
    gz = m["genzero"]
    for k in ("B", "Full"):
        e = gz[k]["eval_half"]
        lines.append(f"| Gen-Zero {k} | parity odd half (100) | {100 * e['accuracy']:.1f}% "
                     f"(all 200: {100 * gz[k]['all_200_accuracy']:.1f}%) | {e['ece_10bin']:.3f} | "
                     f"{e['brier_binary']:.3f} | {gz[k]['latency_wall_ms']['median']:.3f} (head only) |")
    for lk in ("laya", "laya_general"):
        if not m.get(lk):
            continue
        la = m[lk]
        lines.append(f"| Laya {la['model_id'].split('/')[-1]} | 200 other val rows | {100 * la['accuracy']:.1f}% | {la['ece_10bin']:.3f} | "
                     f"{la['brier_binary']:.3f} | {la['latency_wall_ms']['median']:.1f} (text end-to-end) |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", choices=("train", "laya", "eval", "all"), default="all")
    p.add_argument("--skip-mcq", action="store_true", help="laya stage: grid only")
    p.add_argument("--laya-model", choices=tuple(LAYA_MODELS), default="typed", help="laya stage: checkpoint")
    args = p.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    train_rep = None
    if args.stage in ("train", "all"):
        train_rep = stage_train(args)
        (ART / "deadlock_training_report_v1.json").write_text(json.dumps(train_rep, indent=2))
    elif (ART / "deadlock_training_report_v1.json").exists():
        train_rep = json.loads((ART / "deadlock_training_report_v1.json").read_text())
    if args.stage in ("laya", "all"):
        stage_laya(args)
    if args.stage in ("eval", "all"):
        rep = stage_eval(args, train_rep)
        REPORT.write_text(json.dumps(rep, indent=2, default=float))
        md = to_markdown(rep)
        REPORT_MD.write_text("# Laya vs Gen-Zero (RFC-072) measured report\n\n"
                             f"Generated {rep['generated']}; load average at report {rep['host']['loadavg_at_report']}.\n\n"
                             + md + "\n## Findings\n\n" + "".join(f"- {f}\n" for f in rep["findings"])
                             + "\n## Caveats\n\n" + "".join(f"- {h}\n" for h in rep["honesty"])
                             + "\nSee laya_comparison_report.json for per-item records.\n")
        print(md)
        log(f"report -> {REPORT}")


if __name__ == "__main__":
    main()
