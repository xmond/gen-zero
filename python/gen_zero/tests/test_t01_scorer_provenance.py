"""T01: decide() reports the scorer that actually ran, separate from checkpoint state.

A non-neural primary expert (grid rollout, graph search, the cp_sat Python predicate
filter) must keep its own scorer label whether or not a checkpoint is loaded. Only a
neural path without a checkpoint is labelled untrained_weights_fallback. A non-neural
primary fused with an untrained neural secondary must still come back degraded. A state
the world_model expert cannot simulate raises instead of returning fabricated scores.
"""
import logging

import pytest

from gen_zero.client import NON_NEURAL_SCORERS, UNTRAINED_WEIGHTS_SCORER, GenZero, UnsupportedStateError

UNTRAINED_LOG = "untrained random-init weights"
GRID = {"size": 5, "body": [[2, 2]], "direction": "north", "food": [2, 0]}


@pytest.fixture(scope="module")
def client():
    gz = GenZero()
    assert gz.weights_loaded_from_checkpoint is False
    return gz


def _untrained_logs(caplog):
    return [r for r in caplog.records if UNTRAINED_LOG in r.getMessage()]


def test_world_model_primary_keeps_heuristic_scorer(client, caplog):
    with caplog.at_level(logging.WARNING, logger="gen_zero.client"):
        res = client.decide(state=GRID, candidates=["north", "east"], mode="world_model")
    assert res["scorer"] == "heuristic_grid_rollout"
    assert res["scorer_expert"] == "world_model"
    assert res["confidence_kind"] == "uncalibrated_normalized_rollout_score"
    assert res["weights_loaded_from_checkpoint"] is False
    assert UNTRAINED_WEIGHTS_SCORER not in str(res.get("degraded_reason") or "")
    assert not _untrained_logs(caplog)


def _walk(state, action):
    dx, dy = {"east": (1, 0), "north": (0, 1)}[action]
    pos = (state["pos"][0] + dx, state["pos"][1] + dy)
    reached = pos == state["goal"]
    return {"pos": pos, "goal": state["goal"]}, (1.0 if reached else -0.1), reached


def test_astar_primary_reports_graph_search(client, caplog):
    with caplog.at_level(logging.WARNING, logger="gen_zero.client"):
        res = client.decide(
            state={"pos": (0, 0), "goal": (2, 0)}, candidates=["north", "east"],
            mode="astar", transition_fn=_walk,
        )
    meta = res["expert_outputs"]["astar"]["meta"]
    assert meta["scorer"] == "symbolic_graph_search"
    assert meta["confidence_kind"] == "heuristic_path_cost"
    assert meta["plan_length"] == 2
    assert res["action"] == "east"
    assert res["scorer"] == "symbolic_graph_search"
    assert res["weights_loaded_from_checkpoint"] is False
    assert not _untrained_logs(caplog)


def test_astar_without_goal_falls_back_to_labelled_lookahead(client):
    res = client.decide(state=GRID, candidates=["north", "east"], mode="astar")
    assert res["scorer"] == "symbolic_one_step_lookahead"


def test_bidirectional_is_labelled_lookahead_not_graph_search(client):
    # "bidirectional" never calls the graph planner at this call site.
    res = client.decide(state=GRID, candidates=["north", "east"], mode="bidirectional")
    assert res["scorer"] == "symbolic_one_step_lookahead"
    assert res["confidence_kind"] == "heuristic_lookahead_score"


def test_cp_sat_primary_reports_python_predicate_filter(client, caplog):
    # verify_and_prune runs Python rule predicates, not OR-Tools; the label must say so.
    with caplog.at_level(logging.WARNING, logger="gen_zero.client"):
        res = client.decide(state="some state text", candidates=["a", "b"], mode="cp_sat")
    assert res["scorer"] == "python_predicate_filter"
    assert res["scorer_expert"] == "cp_sat"
    assert res["confidence_kind"] == "rule_predicate_filtering"
    assert res["weights_loaded_from_checkpoint"] is False
    assert not _untrained_logs(caplog)


def test_world_model_scorer_survives_loaded_checkpoint(client, monkeypatch):
    monkeypatch.setattr(client, "weights_loaded_from_checkpoint", True)
    res = client.decide(state=GRID, candidates=["north", "east"], mode="world_model")
    assert res["weights_loaded_from_checkpoint"] is True
    assert res["scorer"] == "heuristic_grid_rollout"
    assert res["scorer_expert"] == "world_model"
    assert res["confidence_kind"] == "uncalibrated_normalized_rollout_score"


def test_astar_scorer_survives_loaded_checkpoint(client, monkeypatch):
    monkeypatch.setattr(client, "weights_loaded_from_checkpoint", True)
    res = client.decide(
        state={"pos": (0, 0), "goal": (2, 0)}, candidates=["north", "east"],
        mode="astar", transition_fn=_walk,
    )
    assert res["weights_loaded_from_checkpoint"] is True
    assert res["scorer"] == "symbolic_graph_search"
    assert res["scorer_expert"] == "astar"
    assert res["confidence_kind"] == "heuristic_path_cost"


def test_world_model_unsupported_state_fails_closed(client):
    # Before the fix this returned uniform 1/len(candidates) probs and candidates[0].
    with pytest.raises(UnsupportedStateError, match="Unsupported dict state schema"):
        client.decide(state={"foo": 1}, candidates=["north", "east"], mode="world_model")


def test_reflex_without_checkpoint_is_still_marked_untrained(client, caplog):
    with caplog.at_level(logging.WARNING, logger="gen_zero.client"):
        res = client.decide(state="some state text", candidates=["a", "b"], mode="reflex")
    assert res["scorer"] == UNTRAINED_WEIGHTS_SCORER
    assert res["degraded"] is True
    assert res["weights_loaded_from_checkpoint"] is False
    assert _untrained_logs(caplog)


def test_non_neural_primary_fused_with_untrained_reflex_is_degraded(client, monkeypatch, caplog):
    def fake_route(**kwargs):
        return {
            "k": 2,
            "selected_experts": [("world_model", 0.6), ("reflex", 0.4)],
            "complexity_score": 0.5,
            "pipeline_has_cpsat": False,
        }

    monkeypatch.setattr(client.moe_router, "route_dynamic", fake_route)
    with caplog.at_level(logging.WARNING, logger="gen_zero.client"):
        res = client.decide(state=GRID, candidates=["north", "east"], mode="auto")
    assert res["expert_outputs"]["reflex"]["meta"]["scorer"] == UNTRAINED_WEIGHTS_SCORER
    assert res["scorer"] == "heuristic_grid_rollout"
    assert res["degraded"] is True
    assert f"{UNTRAINED_WEIGHTS_SCORER}:reflex" in res["degraded_reason"]
    assert any("fused experts" in r.getMessage() for r in caplog.records)


def test_mark_untrained_leaves_non_neural_scorer_alone(client, caplog):
    for label in sorted(NON_NEURAL_SCORERS):
        res = {"scorer": label}
        with caplog.at_level(logging.WARNING, logger="gen_zero.client"):
            client._mark_untrained(res, "decide")
        assert res == {"scorer": label}
    assert not _untrained_logs(caplog)
