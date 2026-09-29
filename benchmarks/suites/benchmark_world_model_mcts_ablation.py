#!/usr/bin/env python3
"""Paired diagnostic of privileged exact-dynamics lookahead versus greedy navigation.

This is a benchmark entry point, not evidence of a trained neural world model.
The exact graph model is intentionally labelled privileged in every report.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from deadlock_torus_env import ACTIONS, DeadlockTorusEnv

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
    args = parser.parse_args()
    report = run(args.episodes, args.seed, args.simulations, args.max_steps)
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
    args.output.with_suffix(".md").write_text(summary)
    print(render_console_table(report))
    print()
    print(f"Full JSON report: {args.output}")
    print(f"Full Markdown report: {args.output.with_suffix('.md')}")
    print()
    print(json.dumps({"output": str(args.output), "episodes": args.episodes,
                      "mcnemar_success": p, "metrics": m}))


if __name__ == "__main__":
    main()
