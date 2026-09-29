#!/usr/bin/env python3
"""Minimal SDK decisions. Outputs are demonstrations, not accuracy claims."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from gen_zero import GenZero


def main():
    engine = GenZero()
    discrete = engine.decide(
        {"size": 5, "body": [[2, 2]], "food": [2, 3]},
        ["north", "east", "south", "west"],
        mode="reflex",
    )
    continuous = engine.decide_continuous(
        [0.5, 0.2], action_dim=2, bounds=(-1.0, 1.0), horizon=3, num_samples=16
    )
    print(json.dumps({"discrete": discrete, "continuous": continuous}, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
