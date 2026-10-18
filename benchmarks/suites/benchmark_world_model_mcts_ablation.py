#!/usr/bin/env python3
"""Paired diagnostic of privileged exact-dynamics lookahead versus greedy navigation.

This is a benchmark entry point, not evidence of a trained neural world model.
The exact graph model is intentionally labelled privileged in every report.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
import zlib
from pathlib import Path

import numpy as np

from deadlock_torus_env import ACTIONS, DeadlockTorusEnv, latent_codes  # also puts python/ on sys.path
from gen_zero.causal.latent_mcts import LatentMctsPlanner
from gen_zero.config import GenZeroConfig
from gen_zero.planner.engines.mcts_engine import MctsEngine

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "results" / "world_model_mcts_ablation_report.json"


class ExactGraphWorldModel:
    """Deterministic transition and safety predictor with privileged map access."""

    def __init__(self, env: DeadlockTorusEnv):
        self.env = env
        self.calls = 0

    def predict(self, state: tuple[int, int], action: str) -> tuple[tuple[int, int], float]:
        self.calls += 1
        next_state = self.env.step(state, action)
        return next_state, 0.0 if next_state in self.env.traps else 1.0


def greedy(env: DeadlockTorusEnv, state: tuple[int, int]) -> str:
    # Coordinate-only heuristic: no transition function or trap map access.
    x, y = state
    gx, gy = env.goal
    size = env.size
    estimates = {
        "north": (x, (y - 1) % size), "east": ((x + 1) % size, y),
        "south": (x, (y + 1) % size), "west": ((x - 1) % size, y),
    }
    return min(ACTIONS, key=lambda a: (
        min((estimates[a][0] - gx) % size, (gx - estimates[a][0]) % size)
        + min((estimates[a][1] - gy) % size, (gy - estimates[a][1]) % size),
        ACTIONS.index(a),
    ))


def mcts_action(env: DeadlockTorusEnv, model: ExactGraphWorldModel,
                state: tuple[int, int], simulations: int = 32, depth: int = 6) -> str:
    if simulations < 4 or depth < 1:
        raise ValueError("simulations >= 4 and depth >= 1 required")
    visits = {a: 0 for a in ACTIONS}
    totals = {a: 0.0 for a in ACTIONS}
    for iteration in range(simulations):
        unvisited = [a for a in ACTIONS if not visits[a]]
        if unvisited:
            root = unvisited[0]
        else:
            root = max(ACTIONS, key=lambda a: totals[a] / visits[a]
                       + 1.4 * math.sqrt(math.log(iteration + 1) / visits[a]))
        node, safety = model.predict(state, root)
        if safety == 0:
            value = -10.0
        else:
            value = 0.0
            seen = {state, node}
            for step in range(depth):
                if node == env.goal:
                    value = 10.0 - step
                    break
                # Expand using predicted transitions and prune unsafe successors.
                candidates = [(a, *model.predict(node, a)) for a in ACTIONS]
                safe = [(a, nxt) for a, nxt, score in candidates if score > 0 and nxt not in seen]
                if not safe:
                    value = -float(env.distance(node))
                    break
                _, node = min(safe, key=lambda item: env.distance(item[1]))
                seen.add(node)
            else:
                value = -float(env.distance(node))
        visits[root] += 1
        totals[root] += value
    return max(ACTIONS, key=lambda a: (totals[a] / visits[a], -ACTIONS.index(a)))


def episode(env: DeadlockTorusEnv, with_model: bool, max_steps: int,
            simulations: int) -> dict:
    state = env.start
    model = ExactGraphWorldModel(env) if with_model else None
    latencies = []
    for step in range(max_steps):
        if state == env.goal or state in env.traps:
            break
        begin = time.perf_counter_ns()
        action = mcts_action(env, model, state, simulations) if model else greedy(env, state)
        latencies.append((time.perf_counter_ns() - begin) / 1e6)
        state = env.step(state, action)
    return {"success": state == env.goal, "trap": state in env.traps,
            "steps": len(latencies), "decision_latency_ms_mean": statistics.mean(latencies) if latencies else 0.0,
            "model_calls": model.calls if model else 0}


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n)


def run(episodes: int, seed: int, simulations: int, max_steps: int) -> dict:
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")
    rows = []
    for trial in range(episodes):
        env = DeadlockTorusEnv.generate(seed + trial)
        rows.append({"seed": seed + trial,
                     "baseline": episode(env, False, max_steps, simulations),
                     "world_model": episode(env, True, max_steps, simulations)})
    b = sum(r["baseline"]["success"] and not r["world_model"]["success"] for r in rows)
    c = sum(r["world_model"]["success"] and not r["baseline"]["success"] for r in rows)
    metrics = {}
    for name in ("baseline", "world_model"):
        data = [r[name] for r in rows]
        metrics[name] = {
            "success_rate_pct": 100 * sum(r["success"] for r in data) / episodes,
            "trap_rate_pct": 100 * sum(r["trap"] for r in data) / episodes,
            "mean_steps": statistics.mean(r["steps"] for r in data),
            "mean_decision_latency_ms": statistics.mean(r["decision_latency_ms_mean"] for r in data),
            "total_model_calls": sum(r["model_calls"] for r in data),
        }
    return {"is_synthetic": True,
            "notice": "Paired seeded torus diagnostic using privileged exact graph dynamics; no neural model was trained or evaluated.",
            "method": "paired seeded torus; greedy coordinate baseline vs MCTS with privileged exact graph dynamics; no neural training",
            "episodes": episodes, "seed_start": seed, "simulations_per_decision": simulations,
            "max_steps": max_steps, "metrics": metrics,
            "mcnemar_success": {"baseline_only": b, "world_model_only": c,
                                "p_exact_two_sided": mcnemar_exact(b, c)}, "paired_rows": rows}


# --------------------------------------------------------------------------- engine ablation
#
# The arms below call the production MctsEngine / LatentMctsPlanner, not the
# hand-written bandit above. "baseline" is the engine default (plain PUCT, full
# budget, no cache); every option under test is opt-in on the same engine.

ENGINE_DEPTH = 6
LATENT_DIM = GenZeroConfig.hidden_dim  # production latent width


class LatentCodeTorusModel:
    """Latent-code world model of the same torus, with privileged exact dynamics.

    States are orthonormal codes of width ``dim`` (default: the production latent
    width ``GenZeroConfig.hidden_dim``). A transition decodes the nearest code
    (one ``cells x dim`` mat-vec), applies the exact env transition and re-encodes.
    It plans exactly like the graph model; only the per-transition cost differs.
    The dynamics come from env.transitions, so this is not a learned model.
    """

    def __init__(self, env: DeadlockTorusEnv, dim: int = LATENT_DIM):
        self.env = env
        self.cells = sorted(env.transitions)
        index = {cell: i for i, cell in enumerate(self.cells)}
        self.codes = latent_codes(len(self.cells), dim, env.seed)
        self.index = index
        self.next_index = {a: [index[env.transitions[cell][a]] for cell in self.cells] for a in ACTIONS}
        self.calls = 0

    def encode(self, cell: tuple[int, int]) -> np.ndarray:
        return self.codes[self.index[cell]].copy()

    def decode(self, z: np.ndarray) -> tuple[int, int]:
        return self.cells[int(np.argmax(self.codes @ z))]

    def predict(self, z: np.ndarray, action: str) -> tuple[np.ndarray, tuple[int, int]]:
        self.calls += 1
        j = self.next_index[action][int(np.argmax(self.codes @ z))]
        return self.codes[j].copy(), self.cells[j]


def _torus_problem(env: DeadlockTorusEnv, model, regime: str):
    """transition_fn / eval_fn over cells (exact graph) or latent codes (linear model)."""
    def outcome(nxt, cell):
        if cell in env.traps:
            return nxt, -10.0, True
        if cell == env.goal:
            return nxt, 10.0, True
        return nxt, -0.1, False

    if regime == "exact_graph_dict":
        def transition(state, action):
            nxt, _ = model.predict(state, action)
            return outcome(nxt, nxt)

        def eval_fn(state, actions):
            return {a: 1.0 for a in actions}, -env.distance(state) / env.size
    else:
        def transition(state, action):
            return outcome(*model.predict(state, action))

        def eval_fn(state, actions):
            return {a: 1.0 for a in actions}, -env.distance(model.decode(state)) / env.size

    return transition, eval_fn


def _engine_decision(env, cell, engine: MctsEngine, model, regime: str, repeats: int) -> dict:
    """Times ``repeats`` identical plans (deterministic) and keeps the median."""
    root = cell if regime == "exact_graph_dict" else model.encode(cell)
    transition, eval_fn = _torus_problem(env, model, regime)
    times, res, calls = [], None, 0
    for _ in range(repeats):
        model.calls = 0
        begin = time.perf_counter_ns()
        res = engine.plan(root, list(ACTIONS), transition, legal_actions_fn=lambda s: ACTIONS,
                          eval_fn=eval_fn, max_depth=ENGINE_DEPTH)
        times.append((time.perf_counter_ns() - begin) / 1e6)
        calls = model.calls
    if res["status"] != "OK":
        raise RuntimeError(f"engine returned {res['status']} on a solvable torus state {cell}")
    if calls != res["transition_calls"]:
        raise AssertionError("engine transition_calls disagrees with the model's own counter")
    return {"action": res["best_action"], "latency_ms": statistics.median(times),
            "transition_calls": calls, "simulations_run": res["simulations_run"],
            "cache_hits": res["cache_hits"], "early_stopped": res["early_stopped"]}


SPEED_CONFIGS = {
    "default": {},
    "early_stop": {"early_stop": True},
    "cache": {"transposition_cache": True},
    "early_stop+cache": {"early_stop": True, "transposition_cache": True},
}


def _speed_regime(regime: str, episodes: int, seed: int, simulations: int, max_steps: int,
                  repeats: int) -> dict:
    engines = {name: MctsEngine(num_simulations=simulations, max_depth=ENGINE_DEPTH, **kw)
               for name, kw in SPEED_CONFIGS.items()}
    decisions, outcomes = [], {name: [] for name in SPEED_CONFIGS}
    for trial in range(episodes):
        env = DeadlockTorusEnv.generate(seed + trial)
        model = ExactGraphWorldModel(env) if regime == "exact_graph_dict" else LatentCodeTorusModel(env)
        # Paired decisions along the trajectory the default engine takes.
        cell = env.start
        for _ in range(max_steps):
            if cell == env.goal or cell in env.traps:
                break
            row = {name: _engine_decision(env, cell, eng, model, regime, repeats)
                   for name, eng in engines.items()}
            decisions.append(row)
            cell = env.step(cell, row["default"]["action"])
        # Each configuration also drives its own episode end to end.
        for name, eng in engines.items():
            cell = env.start
            for _ in range(max_steps):
                if cell == env.goal or cell in env.traps:
                    break
                cell = env.step(cell, _engine_decision(env, cell, eng, model, regime, 1)["action"])
            outcomes[name].append(cell == env.goal)
    if not decisions:
        raise RuntimeError("engine speed ablation produced zero decisions")
    metrics, paired = {}, {}
    for name in SPEED_CONFIGS:
        rows = [d[name] for d in decisions]
        metrics[name] = {
            "mean_latency_ms": statistics.mean(r["latency_ms"] for r in rows),
            "median_latency_ms": statistics.median(r["latency_ms"] for r in rows),
            "mean_transition_calls": statistics.mean(r["transition_calls"] for r in rows),
            "mean_simulations_run": statistics.mean(r["simulations_run"] for r in rows),
            "early_stop_rate_pct": 100 * sum(r["early_stopped"] for r in rows) / len(rows),
            "mean_cache_hits": statistics.mean(r["cache_hits"] for r in rows),
            "episode_success_rate_pct": 100 * sum(outcomes[name]) / episodes,
        }
        if name == "default":
            continue
        faster = sum(d[name]["latency_ms"] < d["default"]["latency_ms"] for d in decisions)
        slower = sum(d[name]["latency_ms"] > d["default"]["latency_ms"] for d in decisions)
        paired[name] = {
            "decision_agreement_with_default_pct": 100 * sum(
                d["default"]["action"] == d[name]["action"] for d in decisions) / len(decisions),
            "median_latency_ratio_vs_default": statistics.median(
                d[name]["latency_ms"] / d["default"]["latency_ms"] for d in decisions),
            "transition_call_ratio_vs_default": (
                sum(d[name]["transition_calls"] for d in decisions)
                / sum(d["default"]["transition_calls"] for d in decisions)),
            "decisions_faster": faster, "decisions_slower": slower,
            "sign_test_p_two_sided": mcnemar_exact(slower, faster),
        }
    return {"decisions": len(decisions), "metrics": metrics, "paired_vs_default": paired}


def engine_speed_ablation(episodes: int, seed: int, simulations: int, max_steps: int,
                          repeats: int = 3) -> dict:
    """Paired per-decision timing: same state, same budget, options off vs on."""
    return {
        "task": "DeadlockTorusEnv decisions planned by MctsEngine.plan",
        "episodes": episodes, "simulations_budget": simulations, "max_depth": ENGINE_DEPTH,
        "timing_repeats_per_decision": repeats,
        "configs": {k: v or {"note": "engine defaults: PUCT, full budget, no cache"}
                    for k, v in SPEED_CONFIGS.items()},
        "regimes": {
            "exact_graph_dict": _speed_regime("exact_graph_dict", episodes, seed, simulations,
                                              max_steps, repeats),
            f"latent_codes_dim{LATENT_DIM}": _speed_regime("latent_codes", episodes, seed, simulations,
                                                           max_steps, repeats),
        },
    }


LURE_ARMS = 8
LURE_DISCOUNT = 0.99


def _lure_instance(seed: int, prior_kind: str):
    """Multi-branch distractor trap with noisy leaf evaluations.

    Eight root arms. The optimal arm's true return is 0.8 * gamma. The lure pays an
    immediate +0.3 but is worth 0.3 + 0.2 * gamma. Other arms are worth
    U[0.3, 0.6] * gamma. ``eval_fn`` sees each state's true arm value plus
    deterministic noise in [-0.35, 0.35], so a good choice needs repeated looks.
    ``prior_kind`` sets the root prior: "lure" (0.4 on the lure, 0.02 on the
    optimum), "helpful" (the reverse) or "uniform".
    """
    rng = random.Random(seed)
    arms = [f"a{i}" for i in range(LURE_ARMS)]
    optimal, lure = rng.sample(range(LURE_ARMS), 2)
    mean = {a: rng.uniform(0.3, 0.6) for a in arms}
    mean[arms[optimal]], mean[arms[lure]] = 0.8, 0.2
    reward = {a: 0.0 for a in arms}
    reward[arms[lure]] = 0.3
    favoured, starved = (lure, optimal) if prior_kind == "lure" else (optimal, lure)
    prior = {a: 0.58 / (LURE_ARMS - 2) for a in arms}
    if prior_kind == "uniform":
        prior = {a: 1.0 / LURE_ARMS for a in arms}
    else:
        prior[arms[favoured]], prior[arms[starved]] = 0.4, 0.02

    def noise(state) -> float:
        return 0.7 * (zlib.crc32(repr((seed, state)).encode()) / 2**32 - 0.5)

    def legal(state):
        return arms if state == () else ["l", "r"]

    def transition(state, action):
        return state + (action,), (reward[action] if state == () else 0.0), False

    def eval_fn(state, actions):
        if state == ():
            return {a: prior[a] for a in actions}, 0.0
        return {a: 1.0 for a in actions}, mean[state[0]] + noise(state)

    truth = max(arms, key=lambda a: reward[a] + LURE_DISCOUNT * mean[a])
    if truth != arms[optimal]:
        raise AssertionError("lure instance construction broke its own ground truth")
    return arms, legal, transition, eval_fn, truth


def engine_accuracy_ablation(instances: int, seed: int, budgets=(8, 16, 32)) -> dict:
    """Optimal-arm hit rate at small budgets: PUCT (given / uniform prior) vs Sequential Halving."""
    arms_by_config = {
        "puct_prior": ("puct", None),
        "puct_uniform_prior": ("puct", "uniform"),
        "sequential_halving_prior": ("sequential_halving", None),
    }
    out = {}
    for prior_kind in ("lure", "helpful"):
        per_budget = {}
        for budget in budgets:
            hits = {name: [] for name in arms_by_config}
            for i in range(instances):
                for name, (strategy, override) in arms_by_config.items():
                    arms, legal, transition, eval_fn, truth = _lure_instance(seed + i, override or prior_kind)
                    res = MctsEngine(num_simulations=budget, max_depth=4, discount=LURE_DISCOUNT,
                                     search_strategy=strategy).plan(
                        (), arms, transition, legal_actions_fn=legal, eval_fn=eval_fn)
                    if res["status"] != "OK":
                        raise RuntimeError(f"{name} returned {res['status']} on lure instance {seed + i}")
                    hits[name].append(res["best_action"] == truth)
            sh = hits["sequential_halving_prior"]
            row = {name: 100 * sum(h) / instances for name, h in hits.items()}
            for base in ("puct_prior", "puct_uniform_prior"):
                b = sum(x and not y for x, y in zip(hits[base], sh))
                c = sum(y and not x for x, y in zip(hits[base], sh))
                row[f"mcnemar_vs_{base}"] = {"baseline_only": b, "sequential_halving_only": c,
                                             "p_exact_two_sided": mcnemar_exact(b, c)}
            per_budget[str(budget)] = row
        out[prior_kind] = per_budget
    return {"task": "8-arm distractor-trap tree, noisy eval_fn, MctsEngine", "instances": instances,
            "seed_start": seed, "chance_level_pct": 100.0 / LURE_ARMS, "metric": "optimal root arm hit rate %",
            "by_root_prior": out}


def latent_early_stop_ablation(problems: int, seed: int, simulations: int = 200) -> dict:
    """LatentMctsPlanner default vs early_stop on random ETF problems, same budget."""
    rng = np.random.default_rng(seed)
    rows = []
    full_planner = LatentMctsPlanner(num_simulations=simulations)
    fast_planner = LatentMctsPlanner(num_simulations=simulations, early_stop=True)
    for _ in range(problems):
        emb = rng.normal(size=(6, 12))
        z = rng.normal(size=12)
        cands = [str(i) for i in range(6)]
        t0 = time.perf_counter_ns()
        full = full_planner.plan(z, cands, emb)
        t1 = time.perf_counter_ns()
        fast = fast_planner.plan(z, cands, emb)
        t2 = time.perf_counter_ns()
        rows.append({"agree": full[0] == fast[0], "full_ms": (t1 - t0) / 1e6, "fast_ms": (t2 - t1) / 1e6,
                     "fast_sims": fast[2]["simulations_run"], "stopped": fast[2]["early_stopped"]})
    return {"task": "random 6-candidate latent problems, default transition", "problems": problems,
            "simulations_budget": simulations,
            "decision_agreement_pct": 100 * sum(r["agree"] for r in rows) / problems,
            "early_stop_rate_pct": 100 * sum(r["stopped"] for r in rows) / problems,
            "mean_simulations_run_early_stop": statistics.mean(r["fast_sims"] for r in rows),
            "mean_latency_ms_default": statistics.mean(r["full_ms"] for r in rows),
            "mean_latency_ms_early_stop": statistics.mean(r["fast_ms"] for r in rows)}


def run_engine_ablation(episodes: int, seed: int, simulations: int, max_steps: int,
                        instances: int) -> dict:
    return {"speed": engine_speed_ablation(episodes, seed, simulations, max_steps),
            "accuracy": engine_accuracy_ablation(instances, seed),
            "latent_early_stop": latent_early_stop_ablation(max(instances // 2, 1), seed)}


def render_engine_table(ablation: dict) -> str:
    sp, acc, lat = ablation["speed"], ablation["accuracy"], ablation["latent_early_stop"]
    lines = ["=" * 78, f"MctsEngine options ablation (production engine.plan, budget "
             f"{sp['simulations_budget']}, depth {sp['max_depth']})", "=" * 78]
    for regime, data in sp["regimes"].items():
        lines.append(f"[{regime}] {data['decisions']} paired decisions")
        lines.append(f"{'Config':<17} | {'Mean ms':>8} | {'Med ms':>7} | {'Transitions':>11} | "
                     f"{'Sims run':>8} | {'Success %':>9} | {'Agree %':>7} | {'Ratio':>5}")
        for name, m in data["metrics"].items():
            pd = data["paired_vs_default"].get(name)
            agree = f"{pd['decision_agreement_with_default_pct']:7.1f}" if pd else f"{'-':>7}"
            ratio = f"{pd['median_latency_ratio_vs_default']:5.2f}" if pd else f"{'-':>5}"
            lines.append(f"{name:<17} | {m['mean_latency_ms']:>8.3f} | {m['median_latency_ms']:>7.3f} | "
                         f"{m['mean_transition_calls']:>11.1f} | {m['mean_simulations_run']:>8.1f} | "
                         f"{m['episode_success_rate_pct']:>9.2f} | {agree} | {ratio}")
    lines.append("-" * 78)
    lines.append(f"Distractor-trap hit rate % (n={acc['instances']} per cell, chance "
                 f"{acc['chance_level_pct']:.1f}%)")
    for kind, per_budget in acc["by_root_prior"].items():
        for budget, row in per_budget.items():
            m1, m2 = row["mcnemar_vs_puct_prior"], row["mcnemar_vs_puct_uniform_prior"]
            lines.append(f"prior={kind:<7} sims={budget:>3} | PUCT {row['puct_prior']:6.1f} | "
                         f"PUCT-uniform {row['puct_uniform_prior']:6.1f} | SeqHalving "
                         f"{row['sequential_halving_prior']:6.1f} | p_vs_PUCT={m1['p_exact_two_sided']:.3g} "
                         f"p_vs_uniform={m2['p_exact_two_sided']:.3g}")
    lines.append("-" * 78)
    lines.append(f"LatentMctsPlanner early_stop (n={lat['problems']}, budget {lat['simulations_budget']}): "
                 f"agreement {lat['decision_agreement_pct']:.1f}%, sims {lat['mean_simulations_run_early_stop']:.1f}, "
                 f"ms {lat['mean_latency_ms_default']:.3f} -> {lat['mean_latency_ms_early_stop']:.3f}")
    lines.append("=" * 78)
    return "\n".join(lines)


METHOD_LABELS = {"baseline": "Greedy baseline (no model)",
                  "world_model": "Gen-Zero MCTS (exact-graph oracle)"}

CAVEAT = ('CAVEAT: "world model" here is env.step()\'s privileged exact graph dynamics, not the\n'
          "trained neural network. This is a planning diagnostic, not evidence of neural world-model\n"
          'or production-planner performance. See README.md "What is measured" before quoting it.')


def render_console_table(report: dict) -> str:
    """Human-readable ASCII/Markdown-style summary for terminal output."""
    m, p = report["metrics"], report["mcnemar_success"]
    rows = [(METHOD_LABELS[name], v) for name, v in m.items()]
    col = max(len("Method"), max(len(name) for name, _ in rows))
    header = (f"{'Method':<{col}} | {'Success %':>9} | {'Trapped %':>9} | "
              f"{'Mean Steps':>10} | {'Latency ms':>10}")
    rule = "=" * len(header)
    title = (f"Gen-Zero MCTS (exact-graph oracle) vs Greedy Baseline -- Deadlock Torus "
             f"({report['episodes']} episodes, seed {report['seed_start']}-"
             f"{report['seed_start'] + report['episodes'] - 1})")
    lines = [rule, title, rule, header, "-" * len(header)]
    for name, v in rows:
        lines.append(f"{name:<{col}} | {v['success_rate_pct']:>9.2f} | {v['trap_rate_pct']:>9.2f} | "
                      f"{v['mean_steps']:>10.2f} | {v['mean_decision_latency_ms']:>10.3f}")
    lines.append(rule)
    lines.append(f"McNemar exact (paired): baseline-only={p['baseline_only']}, "
                 f"world_model-only={p['world_model_only']}, p={p['p_exact_two_sided']:.3g}")
    lines.append(CAVEAT)
    lines.append(rule)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--engine-instances", type=int, default=200,
                        help="distractor-trap instances per (prior, budget) cell")
    args = parser.parse_args()
    report = run(args.episodes, args.seed, args.simulations, args.max_steps)
    report["engine_ablation"] = run_engine_ablation(args.episodes, args.seed, args.simulations,
                                                    args.max_steps, args.engine_instances)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    m = report["metrics"]
    p = report["mcnemar_success"]
    summary = ("# World model MCTS paired ablation\n\n"
               "This diagnostic uses privileged exact graph dynamics, not a trained neural model. "
               "Results cannot establish neural-world-model gains or production planner gains.\n\n"
               f"Episodes: {args.episodes}; seed range: {args.seed}–{args.seed + args.episodes - 1}.\n\n"
               "| Policy | Success % | Trap % | Mean steps | Mean decision latency ms | Model calls |\n"
               "|---|---:|---:|---:|---:|---:|\n"
               + "".join(f"| {name} | {v['success_rate_pct']:.2f} | {v['trap_rate_pct']:.2f} | "
                         f"{v['mean_steps']:.2f} | {v['mean_decision_latency_ms']:.4f} | {v['total_model_calls']} |\n"
                         for name, v in m.items())
               + f"\nExact paired McNemar: baseline-only={p['baseline_only']}, "
                 f"model-only={p['world_model_only']}, p={p['p_exact_two_sided']:.6g}.\n\n"
                 "Paired per-seed outcomes and timing are in the adjacent JSON.\n")
    summary += "\n## Production engine options\n\n```\n" + render_engine_table(report["engine_ablation"]) + "\n```\n"
    args.output.with_suffix(".md").write_text(summary)
    print(render_console_table(report))
    print()
    print(render_engine_table(report["engine_ablation"]))
    print()
    print(f"Full JSON report: {args.output}")
    print(f"Full Markdown report: {args.output.with_suffix('.md')}")
    print()
    print(json.dumps({"output": str(args.output), "episodes": args.episodes,
                      "mcnemar_success": p, "metrics": m}))


if __name__ == "__main__":
    main()
