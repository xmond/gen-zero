"""Focused end-to-end checks of paired benchmark and exact McNemar calculation."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/suites/benchmark_world_model_mcts_ablation.py"
sys.path.insert(0, str(SCRIPT.parent))
from deadlock_torus_env import DeadlockTorusEnv
from benchmark_world_model_mcts_ablation import mcnemar_exact  # noqa: E402


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1
    assert mcnemar_exact(0, 5) == 0.0625
    assert mcnemar_exact(2, 2) == 1


def test_cli_five_paired_episodes(tmp_path):
    output = tmp_path / "report.json"
    completed = subprocess.run([sys.executable, str(SCRIPT), "--episodes", "5",
                                "--output", str(output)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text())
    assert len(report["paired_rows"]) == 5
    assert output.with_suffix(".md").exists()
    assert all(row["world_model"]["model_calls"] > 0 for row in report["paired_rows"])
    assert all(row["baseline"]["model_calls"] == 0 for row in report["paired_rows"])


def test_cli_console_table_carries_the_oracle_caveat(tmp_path):
    output = tmp_path / "report.json"
    completed = subprocess.run([sys.executable, str(SCRIPT), "--episodes", "5",
                                "--output", str(output)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "Greedy baseline (no model)" in completed.stdout
    assert "Gen-Zero MCTS (exact-graph oracle)" in completed.stdout
    assert "McNemar exact (paired)" in completed.stdout
    assert "privileged exact graph dynamics" in completed.stdout
    assert "trained neural network" in completed.stdout


def test_trap_is_distinct_from_goal_and_absorbing():
    for seed in range(100):
        env = DeadlockTorusEnv.generate(seed)
        assert env.goal not in env.traps
        assert env.start not in env.traps
        assert env.step(env.start, "east") in env.traps
        trap = next(iter(env.traps))
        assert all(env.step(trap, action) == trap for action in ("north", "east", "south", "west"))
