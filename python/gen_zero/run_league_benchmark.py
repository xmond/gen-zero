"""Empirical Elo convergence audit of the LeagueArena simulated game.

This measures the existing simulator, not poker strength or a learned Nash policy.
No monotonic Elo or fixed historical win-rate guarantee is assumed.
"""
import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen_zero.multiagent.league_arena import LeagueArena


def elo_convergence(history, window=20, score_tolerance=0.05):
    """Require a 95% CI for late-window Elo drift inside practical equivalence.

    A five percentage point expected-score change at equal rating corresponds to
    ~35 Elo. The tolerance is declared before matches, not fitted to results.
    Windows are descriptive: serially correlated training is not an IID proof.
    """
    if window < 2 or not 0 < score_tolerance < 0.5:
        raise ValueError("Invalid convergence window or score tolerance")
    margin = 400 * math.log10((0.5 + score_tolerance) / (0.5 - score_tolerance))
    if len(history) < 2 * window or not all(math.isfinite(x) for x in history):
        return {"passed": False, "reason": "insufficient_or_nonfinite_history"}
    earlier, later = history[-2 * window:-window], history[-window:]
    drift = statistics.mean(later) - statistics.mean(earlier)
    # Batch means reduce, but do not eliminate, serial-correlation sensitivity.
    half_width = 1.96 * math.sqrt(statistics.variance(earlier) / window + statistics.variance(later) / window)
    return {"passed": abs(drift) + half_width <= margin,
            "window": window, "score_tolerance": score_tolerance,
            "elo_equivalence_margin": margin, "drift_elo": drift,
            "drift_ci95": [drift - half_width, drift + half_width],
            "late_elo_std": statistics.stdev(later),
            "limitation": "descriptive interval; training samples are serially correlated"}


def run_league_benchmark(generations=100, seed=42):
    random.seed(seed)
    arena = LeagueArena(base_elo=1200.0, k_factor=32.0, pfsp_exponent=2.0)
    history = []
    for _ in range(generations):
        history.append(arena.evolve_league_generation(matches_per_agent=12))
    ratings = [r["main_elo"] for r in history]
    convergence = elo_convergence(ratings)
    robustness = arena.evaluate_historical_robustness()
    report = {"seed": seed, "generations": generations, "matches_per_agent": 12,
              "domain": "LeagueArena simulated payoff game",
              "initial_elo": arena.base_elo, "final_elo": arena.main_agent.elo_rating,
              "elo_gain": arena.main_agent.elo_rating - arena.base_elo,
              "convergence": convergence, "historical_evaluation": robustness,
              "history": history, "passed": convergence["passed"]}
    path = Path(__file__).resolve().parents[2] / "benchmarks/results/r4_evidence/league_after_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
    print(f"Report: {path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sys.exit(0 if run_league_benchmark(args.generations, args.seed)["passed"] else 1)
