"""Round 6 remaining contracts: explicit experts never fall to reflex, and the top-level
record of which experts ran matches what actually ran.

1. A named expert (world_model here) with one candidate, given or left after legality
   pruning, runs its own branch. An unsupported state fails closed with
   UnsupportedStateError instead of being scored by reflex.
2. When auto mode reassigns world_model to reflex (non-grid dict state), experts_activated,
   expert_weights, expert_routed and k_experts all describe the executed plan, and the
   router's first choice stays visible in initial_routing_plan.
"""
import pytest

from gen_zero.client import GenZero, UnsupportedStateError

GRID = {"size": 5, "body": [[2, 2]], "direction": "north", "food": [2, 0]}
REASSIGNED = "routed_world_model_reassigned_to_reflex:non_grid_state"


@pytest.fixture(scope="module")
def client():
    return GenZero()


def test_unsupported_grid_state_two_candidates_fails_closed(client):
    with pytest.raises(UnsupportedStateError):
        client.decide(state={"foo": 1}, candidates=["north", "east"], mode="world_model")


def test_unsupported_grid_state_single_candidate_fails_closed(client):
    with pytest.raises(UnsupportedStateError):
        client.decide(state={"foo": 1}, candidates=["north"], mode="world_model")


def test_unsupported_grid_state_pruned_to_single_candidate_fails_closed(client):
    # decide() calls legal_actions_fn(state) with one argument (so does the MCTS
    # per-node mask), so the pruning callback takes the state only.
    calls = []

    def legal_actions_fn(s):
        calls.append(s)
        return ["north"]

    with pytest.raises(UnsupportedStateError):
        client.decide(
            state={"foo": 1}, candidates=["north", "east"], mode="world_model",
            legal_actions_fn=legal_actions_fn,
        )
    assert calls, "legal_actions_fn was never applied, so the single-candidate path was not exercised"


def test_valid_grid_state_single_candidate_executes_world_model(client):
    res = client.decide(state=GRID, candidates=["north"], mode="world_model")
    # decide() reports its choice as "action"; best_action lives per expert.
    assert res["action"] == "north"
    assert res["expert_outputs"]["world_model"]["best_action"] == "north"
    assert res["scorer"] == "heuristic_grid_rollout"
    assert res["confidence_kind"] == "uncalibrated_normalized_rollout_score"
    wm = res["expert_outputs"]["world_model"]
    assert wm["probs"] == {"north": 1.0}
    assert wm["meta"]["scorer"] == "heuristic_grid_rollout"
    assert isinstance(wm["meta"]["adaptive_horizon"], int) and wm["meta"]["adaptive_horizon"] > 0
    assert res["experts_activated"] == ["world_model"]
    assert res["expert_routed"] == "world_model"


def test_auto_mode_reassigned_reflex_fields_consistent(client):
    res = client.decide(state={"foo": 1}, candidates=["north", "east", "south"], mode="auto")
    assert res["expert_routed"] == "reflex"
    assert "world_model" not in res["experts_activated"]
    assert "reflex" in res["experts_activated"]
    assert set(res["expert_outputs"].keys()) == set(res["experts_activated"])
    assert set(res["expert_weights"].keys()) == set(res["experts_activated"])
    assert res["k_experts"] == len(res["experts_activated"])
    assert res["adaptive_params"]["k_experts"] == res["k_experts"]
    assert "world_model" in res["initial_routing_plan"]
    assert res["routing_reassignment"] == REASSIGNED


def test_auto_mode_reassignment_merges_with_coselected_reflex(client):
    # The router picks reflex and world_model together here. After reassignment reflex
    # must appear once, run once, and carry the merged weight.
    res = client.decide(
        state={"foo": 1}, candidates=["north", "east", "south"], mode="auto",
        policy_entropy=0.9, task_hint="reflex",
    )
    assert res["initial_routing_plan"] == ["reflex", "world_model"]
    assert res["experts_activated"] == ["reflex"]
    assert res["expert_weights"] == {"reflex": 1.0}
    assert res["expert_outputs"]["reflex"]["weight"] == 1.0
    assert res["k_experts"] == 1
    assert res["routing_reassignment"] == REASSIGNED


def test_infeasible_abstain_reports_only_cp_sat_as_executed(client):
    # Nothing survives the root legality filter, so only the Stage-1 cp_sat pre-filter runs.
    res = client.decide(
        state=GRID, candidates=["north", "east"], mode="world_model",
        legal_actions_fn=lambda s: [],
    )
    assert res["status"] == "INFEASIBLE_ABSTAIN"
    assert res["expert_routed"] == "cp_sat"
    assert res["experts_activated"] == ["cp_sat"]
    assert set(res["expert_outputs"].keys()) == set(res["experts_activated"])
    assert set(res["expert_weights"].keys()) == set(res["experts_activated"])
    assert res["k_experts"] == 1
    assert res["initial_routing_plan"] == ["world_model"]
