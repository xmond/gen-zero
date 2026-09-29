#!/usr/bin/env python3
"""Run the production SDK CEM entrypoint with its built-in illustrative dynamics."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from gen_zero import GenZero


def main():
    engine = GenZero()
    result = engine.decide_continuous(
        state=[0.5, -0.25], action_dim=2, bounds=(-1.0, 1.0),
        horizon=4, num_samples=32,
    )
    print(json.dumps({"planner": "mpc_cem", "result": result}, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
