#!/usr/bin/env python3
"""Inspect model predictions and their provenance; no empirical safety claim."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from gen_zero import GenZero


def main():
    engine = GenZero()
    state = {"size": 5, "body": [[2, 2]], "food": [2, 3], "direction": "east"}
    actions = ["east", "north", "west"]
    simulation = engine.simulate(state, actions, horizon=len(actions))
    counterfactual = engine.what_if(state, ["east", "north", "west"], horizon=3)
    audit = engine.audit_action(state, "east", horizon=3, continuation_actions=["north", "west"])
    print(json.dumps({"simulation": simulation, "what_if": counterfactual, "audit": audit}, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
