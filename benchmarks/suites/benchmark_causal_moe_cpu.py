"""Single expert vs multi-model causal MoE on CPU: accuracy, decisions, latency -> JSON.

Two sources, never mixed in one table:

  real  Each --expert is NAME=ADAPTER.npz=FEATURES.npz. FEATURES holds q, cands,
        offsets, labels (and optionally tasks) in the parity_val200 layout. All
        experts must describe the same records with the same candidate order
        (offsets and labels are checked equal). Records are split by a seeded
        permutation into a router-fit half and a test half. Default: the one
        trained Qwen3.5-9B expert. With one expert every strategy reduces to it.

  sim   A generative simulation, tagged "source": "simulated" in the JSON.
        Latent candidates c_k ~ N(0, I_m); the query is c_y + noise + delta_g,
        where delta_g marks one of three hidden domains. Expert e sees a random
        linear lift of the latents to its own native width (4096 / 5120 / 2816)
        with its own latent noise per domain, and is a real RNN-set adapter
        trained on CPU on its own features only. Two scenarios:
          complementary  expert e is sharp on domain e, noisy elsewhere
          redundant      every expert has the same mid noise everywhere
        The router sees only query vectors; domain ids are used for diagnostics.
        This measures the routing mechanics, not the value of real teachers.

Splits: experts train on split A; the router, temperatures and consensus
thresholds are fitted on split B; every reported number comes from split C
(sim) or the test half (real), which nothing was fitted on.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "benchmarks" / "suites"))

from gen_zero.causal.causal_moe_engine import (  # noqa: E402
    CausalMoERouter, MultiModelCausalMoE, RNNSetExpert, calibrate_temperature)

DEFAULT_REAL = [f"qwen35_9b={REPO}/artifacts/qwen35_9b/zero_rnn_set_adapter_qwen35_9b.npz"
                f"={REPO}/artifacts/qwen35_9b/parity_val200.npz"]
SIM_WIDTHS = {"sim_9b": 4096, "sim_wide": 5120, "sim_narrow": 2816}
STRATEGIES = [("dense_prob", "dense", 2, "prob"), ("dense_logit", "dense", 2, "logit"),
              ("sparse_top1", "sparse", 1, "prob"), ("sparse_top2", "sparse", 2, "prob"),
              ("consensus", "consensus", 2, "prob")]


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# ------------------------------------------------------------------ data

def load_real(specs):
    experts, feats = [], {}
    ref = None
    for spec in specs:
        name, adapter, feat = spec.split("=", 2)
        experts.append(RNNSetExpert.from_npz(name, adapter))
        with np.load(feat, allow_pickle=False) as z:
            f = {k: z[k] for k in ("q", "cands", "offsets", "labels")}
            f["tasks"] = z["tasks"].astype(str) if "tasks" in z.files else np.full(len(f["labels"]), "all")
        if ref is None:
            ref = f
        elif not (np.array_equal(f["offsets"], ref["offsets"]) and np.array_equal(f["labels"], ref["labels"])):
            raise ValueError(f"{name}: records/candidates do not align with the first expert")
        feats[name] = f
    return experts, feats, ref["offsets"], ref["labels"], ref["tasks"]


def simulate(scenario, n, K, m, rng):
    """Latent task + per-expert native-width views. Returns feats per expert, labels, domains."""
    G = len(SIM_WIDTHS)
    dom = rng.integers(G, size=n)
    labels = rng.integers(K, size=n)
    delta = rng.normal(size=(G, m)) * 2.0
    lat_c = rng.normal(size=(n, K, m))
    lat_q = lat_c[np.arange(n), labels] + 0.4 * rng.normal(size=(n, m)) + delta[dom]
    offsets = np.arange(n + 1) * K
    feats = {}
    for e, (name, D) in enumerate(SIM_WIDTHS.items()):
        if scenario == "complementary":
            sig = np.where(np.arange(G) == e, 0.35, 1.6)
        else:
            sig = np.full(G, 0.9)
        s = sig[dom]
        M = rng.normal(size=(m, D)) / np.sqrt(m)
        q = (lat_q + s[:, None] * rng.normal(size=(n, m))) @ M
        c = (lat_c + s[:, None, None] * rng.normal(size=(n, K, m))) @ M
        q += 0.05 * rng.normal(size=q.shape)
        c += 0.05 * rng.normal(size=c.shape)
        feats[name] = {"q": q.astype(np.float32), "cands": c.reshape(n * K, D).astype(np.float32),
                       "offsets": offsets, "labels": labels}
    return feats, offsets, labels, dom


def train_sim_experts(feats, offsets, labels, train_idx, router_idx, out_dir, args, log):
    import torch
    from rnn_set_adapter_torch import AdapterData, train_adapter
    torch.set_num_threads(args.torch_threads)
    experts, reports = [], {}
    idx = np.concatenate([train_idx, router_idx])   # test records are never given to training
    for name, f in feats.items():
        rows = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in idx])
        sub_off = np.concatenate([[0], np.cumsum(np.diff(offsets)[idx])])
        data = AdapterData(q=f["q"][idx], cands=f["cands"][rows], offsets=sub_off, labels=labels[idx],
                           tasks=np.full(len(idx), "sim"),
                           is_val=np.r_[np.zeros(len(train_idx), bool), np.ones(len(router_idx), bool)])
        t0 = time.time()
        model, rep = train_adapter(data, d=args.d, rank=args.rank, think_steps=6, n_heads=4, n_layers=1,
                                   epochs=args.epochs, patience=8, device="cpu", seed=args.seed,
                                   log=lambda *_: None)
        path = Path(out_dir) / f"{name}.npz"
        model.export_npz(path, {"is_synthetic": True, "source": "simulated", "in_dim": int(f["q"].shape[1])})
        experts.append(RNNSetExpert.from_npz(name, path))
        reports[name] = {"train_seconds": round(time.time() - t0, 1),
                         "rho_max": float(experts[-1].runtime.rho_max),
                         "sigma_max_A": float(experts[-1].runtime.sigma_max_A),
                         **{k: rep[k] for k in ("best_epoch", "train_acc") if k in rep}}
        log(f"[train] {name} in_dim={f['q'].shape[1]} {reports[name]}")
    return experts, reports


# ------------------------------------------------------------------ fit + eval

def record(feats, offsets, i):
    return {n: (f["q"][i], f["cands"][offsets[i]:offsets[i + 1]]) for n, f in feats.items()}


def fit_router(experts, feats, offsets, labels, fit_idx, rank, log):
    names = [e.name for e in experts]
    if len(experts) == 1 or len(fit_idx) < rank + 1:
        router = CausalMoERouter.uniform(names)
    else:
        router = CausalMoERouter.fit_projections(names, {n: feats[n]["q"][fit_idx] for n in names}, rank)
    scores = {n: [] for n in names}
    for i in fit_idx:
        rec = record(feats, offsets, i)
        for e in experts:
            scores[e.name].append(e.score(*rec[e.name]).astype(np.float64))
    y = labels[fit_idx]
    router.temperatures = np.array([calibrate_temperature(scores[n], y) for n in names])
    if len(experts) > 1:
        P = np.array([[_softmax(scores[n][j] / T)[y[j]] for n, T in zip(names, router.temperatures)]
                      for j in range(len(fit_idx))])
        phis = np.stack([router.features({n: feats[n]["q"][i] for n in names}) for i in fit_idx])
        hist = router.fit(phis, P, l2=1e-2, steps=400, lr=0.05)
        log(f"[router] mixture NLL {hist[0]:.4f} -> {hist[-1]:.4f}")
        best = (-1.0, 0.15, 0.45)
        moe = MultiModelCausalMoE(experts, router)
        for agree in (0.0, 0.05, 0.1, 0.2, 0.3):
            for conflict in (0.3, 0.45, 0.6, 0.8, 1.0):
                if agree > conflict:
                    continue
                router.agree_tau, router.conflict_tau = agree, conflict
                acc = np.mean([moe.infer(record(feats, offsets, i), "consensus").choice == labels[i]
                               for i in fit_idx])
                if acc > best[0] + 1e-12:
                    best = (acc, agree, conflict)
        router.agree_tau, router.conflict_tau = best[1], best[2]
        log(f"[router] consensus taus agree={best[1]} conflict={best[2]} (fit-split acc {best[0]:.4f})")
    return router


def bootstrap_diff(a, b, rng, n=2000):
    d = a.astype(float) - b.astype(float)
    boots = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(n)]
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def evaluate(experts, router, feats, offsets, labels, test_idx, groups, args, rng):
    names = [e.name for e in experts]
    moe1 = MultiModelCausalMoE(experts, router, n_threads=1)
    moeN = MultiModelCausalMoE(experts, router, n_threads=len(experts))
    y = labels[test_idx]
    hits = {n: [] for n in names}
    out = {k: {"hits": [], "decisions": {}, "gate_argmax": [], "conf": []} for k, *_ in STRATEGIES}
    for i in test_idx:
        rec = record(feats, offsets, i)
        for key, strat, k, fusion in STRATEGIES:
            r = moe1.infer(rec, strat, top_k=k, fusion=fusion)
            o = out[key]
            o["hits"].append(r.choice == labels[i])
            o["decisions"][r.decision] = o["decisions"].get(r.decision, 0) + 1
            o["gate_argmax"].append(int(np.argmax(r.gates)))
            o["conf"].append(r.confidence)
            if key == "dense_prob":
                for n in names:
                    hits[n].append(int(np.argmax(r.expert_scores[n])) == labels[i])
    single = {n: float(np.mean(h)) for n, h in hits.items()}
    best_single = max(single, key=single.get)
    oracle = float(np.mean(np.any(np.stack([hits[n] for n in names]), axis=0)))
    res = {"n_test": int(len(test_idx)), "single_expert_acc": single, "best_single": best_single,
           "oracle_any_expert_acc": oracle, "strategies": {}}
    for key, *_ in STRATEGIES:
        o = out[key]
        h = np.array(o["hits"])
        entry = {"acc": float(h.mean()), "decisions": o["decisions"],
                 "vs_best_single": bootstrap_diff(h, np.array(hits[best_single]), rng),
                 "mean_confidence": float(np.mean(o["conf"]))}
        conf, ok = np.array(o["conf"]), h
        entry["acc_when_conf_top_half"] = float(ok[conf >= np.median(conf)].mean())
        if groups is not None:
            g = groups[test_idx]
            entry["per_group_acc"] = {str(v): float(h[g == v].mean()) for v in np.unique(g)}
            if len(names) > 1:
                entry["gate_argmax_matches_group"] = float(np.mean(np.array(o["gate_argmax"]) == g))
        res["strategies"][key] = entry
    if groups is not None:
        g = groups[test_idx]
        res["single_expert_per_group_acc"] = {n: {str(v): float(np.array(hits[n])[g == v].mean())
                                                  for v in np.unique(g)} for n in names}
    res["latency_ms"] = latency(moe1, moeN, record(feats, offsets, int(test_idx[0])), args.latency_runs)
    moeN.close()
    return res


def latency(moe1, moeN, rec, runs):
    out = {}
    for label, moe in (("threads_1", moe1), (f"threads_{len(moeN.names)}", moeN)):
        for key, strat, k, fusion in STRATEGIES:
            for _ in range(3):
                moe.infer(rec, strat, top_k=k, fusion=fusion)
            t = []
            for _ in range(runs):
                t0 = time.perf_counter()
                moe.infer(rec, strat, top_k=k, fusion=fusion)
                t.append((time.perf_counter() - t0) * 1e3)
            out[f"{label}/{key}"] = {"min": round(min(t), 3), "median": round(float(np.median(t)), 3),
                                     "p90": round(float(np.percentile(t, 90)), 3)}
    out["K"] = int(next(iter(rec.values()))[1].shape[0])
    return out


# ------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("real", "sim", "both"), default="both")
    ap.add_argument("--expert", action="append", help="NAME=ADAPTER.npz=FEATURES.npz (real mode)")
    ap.add_argument("--sim-n", type=int, default=1500)
    ap.add_argument("--sim-k", type=int, default=4)
    ap.add_argument("--sim-latent", type=int, default=48)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--router-rank", type=int, default=8)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--latency-runs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "benchmarks" / "results" / "causal_moe_cpu_benchmark.json"))
    args = ap.parse_args(argv)
    log = lambda s: print(s, flush=True)
    rng = np.random.default_rng(args.seed)
    report = {"env": {"loadavg_start": os.getloadavg(), "nproc": os.cpu_count(), "numpy": np.__version__,
                      "python": platform.python_version(), "blas_threads_per_inference": 1,
                      "argv": sys.argv[1:]}, "results": {}}
    t_start = time.time()

    if args.mode in ("real", "both"):
        experts, feats, offsets, labels, tasks = load_real(args.expert or DEFAULT_REAL)
        perm = rng.permutation(len(labels))
        fit_idx, test_idx = np.sort(perm[: len(perm) // 2]), np.sort(perm[len(perm) // 2:])
        router = fit_router(experts, feats, offsets, labels, fit_idx, args.router_rank, log)
        res = evaluate(experts, router, feats, offsets, labels, test_idx, tasks, args, rng)
        res.update({"source": "real", "experts": {e.name: e.in_dim for e in experts},
                    "temperatures": router.temperatures.tolist(),
                    "note": ("one expert: every strategy reduces to it; Qwen 27B and Gemma 26B features "
                             "are not on this box") if len(experts) == 1 else ""})
        report["results"]["real"] = res
        log(f"[real] experts={res['experts']} single={res['single_expert_acc']} "
            f"dense={res['strategies']['dense_prob']['acc']:.4f} n_test={res['n_test']}")

    if args.mode in ("sim", "both"):
        for scenario in ("complementary", "redundant"):
            srng = np.random.default_rng(args.seed + (1 if scenario == "complementary" else 2))
            feats, offsets, labels, dom = simulate(scenario, args.sim_n, args.sim_k, args.sim_latent, srng)
            perm = srng.permutation(args.sim_n)
            a, b = int(0.5 * args.sim_n), int(0.7 * args.sim_n)
            tr, fit, te = np.sort(perm[:a]), np.sort(perm[a:b]), np.sort(perm[b:])
            with tempfile.TemporaryDirectory() as tmp:
                experts, reports = train_sim_experts(feats, offsets, labels, tr, fit, tmp, args, log)
                router = fit_router(experts, feats, offsets, labels, fit, args.router_rank, log)
                res = evaluate(experts, router, feats, offsets, labels, te, dom, args, rng)
            res.update({"is_synthetic": True, "source": "simulated", "scenario": scenario,
                        "experts": {e.name: e.in_dim for e in experts}, "expert_training": reports,
                        "temperatures": router.temperatures.tolist(),
                        "consensus_taus": [router.agree_tau, router.conflict_tau],
                        "split_sizes": {"expert_train": len(tr), "router_fit": len(fit), "test": len(te)}})
            report["results"][f"sim_{scenario}"] = res
            log(f"[sim:{scenario}] single={res['single_expert_acc']} oracle={res['oracle_any_expert_acc']:.4f} "
                + " ".join(f"{k}={v['acc']:.4f}" for k, v in res["strategies"].items()))

    report["env"]["loadavg_end"] = os.getloadavg()
    report["env"]["wall_seconds"] = round(time.time() - t_start, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    log(f"[done] -> {args.out}")
    return report


if __name__ == "__main__":
    main()
