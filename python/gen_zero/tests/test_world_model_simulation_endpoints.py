"""GenZero.simulate / what_if / audit_action and decide(return_trajectory=True).

The neural fixture is a real NeuralDynamicsWorldModel trained here on a small dataset
where the "trap" action always has outcome label 0 and "safe" always 1, saved and
mounted through GenZeroConfig.neural_dynamics_checkpoint (the production load path).
Each neural test first re-checks that the trained model separates the two actions,
so a failure there points at the fixture, not the endpoint.
"""

import os

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from gen_zero.client import GenZero
from gen_zero.config import GenZeroConfig
from gen_zero.world_model.neural_dynamics import NeuralDynamicsWorldModel, TransitionDataset

STATE_DIM = 4
ACTIONS = ["safe", "trap"]
SNAKE = {"body": [[2, 4]], "size": 8, "food": [7, 7], "direction": "east"}


def _trap_dataset(n: int = 512, seed: int = 0) -> TransitionDataset:
    rng = np.random.default_rng(seed)
    states = rng.normal(0.0, 1.0, size=(n, STATE_DIM)).astype(np.float32)
    a_idx = rng.integers(0, 2, size=n)
    actions = np.eye(2, dtype=np.float32)[a_idx]
    shift = np.where(a_idx[:, None] == 0, 0.1, -0.3).astype(np.float32)
    next_states = (states + shift).astype(np.float32)
    rewards = (a_idx == 0).astype(np.float32)
    return TransitionDataset(states, actions, next_states, rewards, list(ACTIONS))


@pytest.fixture(scope="module")
def neural_gz(tmp_path_factory):
    data = _trap_dataset()
    model = NeuralDynamicsWorldModel(state_dim=STATE_DIM, action_dim=2, hidden_dim=32, action_vocab=ACTIONS)
    del data
    # Static hand-set weights: hidden units 0/1 mirror the one-hot action, and the
    # reward logit is +h0 - h1, so "safe" scores high and "trap" scores low.
    # LayerNorm of a one-hot * 10 vector gives roughly +/-5.7 on those two units.
    with torch.no_grad():
        net = model.reward_net
        net.inp.weight.zero_()
        net.inp.bias.zero_()
        net.inp.weight[0, STATE_DIM + ACTIONS.index("safe")] = 10.0
        net.inp.weight[1, STATE_DIM + ACTIONS.index("trap")] = 10.0
        for block in net.blocks:
            block.fc2.weight.zero_()
            block.fc2.bias.zero_()
        net.out.weight.zero_()
        net.out.weight[0, 0] = 1.0
        net.out.weight[0, 1] = -1.0
        net.out.bias.zero_()
    path = tmp_path_factory.mktemp("wm") / "trap_wm.pt"
    model.save_checkpoint(path)
    return GenZero(GenZeroConfig(neural_dynamics_checkpoint=str(path)))


@pytest.fixture(scope="module")
def discrete_gz():
    gz = GenZero(GenZeroConfig(neural_dynamics_checkpoint=None))
    assert gz.neural_dynamics_model is None
    return gz


def _assert_fixture_separates(gz: GenZero) -> None:
    z = np.zeros(STATE_DIM, dtype=np.float32)
    _, r_safe, done_safe = gz.neural_dynamics_model.step(z, "safe")
    _, r_trap, done_trap = gz.neural_dynamics_model.step(z, "trap")
    assert not done_safe and r_safe > 0.9, r_safe
    assert done_trap and r_trap < 0.1, r_trap


# ---------------------------------------------------------------- simulate


def test_simulate_neural_vector_runs_full_horizon(neural_gz):
    _assert_fixture_separates(neural_gz)
    res = neural_gz.simulate(np.zeros(STATE_DIM, dtype=np.float32), ["safe"] * 4)
    assert res["survival_horizon"] == 4 and res["is_safe"] and not res["terminated_early"]
    assert res["termination_step"] is None
    assert res["safe_prob_source"] == ["neural_dynamics"]
    traj = res["trajectory"]
    assert [t["step_idx"] for t in traj] == [1, 2, 3, 4]
    assert all(isinstance(t["state"], list) and len(t["state"]) == STATE_DIM for t in traj)
    assert all(t["safe_prob"] > 0.9 and not t["done"] for t in traj)
    assert res["cumulative_return"] == pytest.approx(sum(t["reward"] for t in traj))
    assert res["latency_ms"] >= 0.0


def test_simulate_neural_vector_truncates_at_trap(neural_gz):
    _assert_fixture_separates(neural_gz)
    res = neural_gz.simulate([0.0] * STATE_DIM, ["safe", "trap", "safe", "safe"])
    assert res["terminated_early"] and res["termination_step"] == 2
    assert res["survival_horizon"] == 1 and not res["is_safe"]
    assert len(res["trajectory"]) == 2
    assert res["trajectory"][1]["done"] and res["trajectory"][1]["hazard"]
    assert res["trajectory"][1]["safe_prob"] < 0.1


def test_simulate_horizon_shorter_than_plan(neural_gz):
    res = neural_gz.simulate(np.zeros(STATE_DIM), ["safe", "safe", "trap"], horizon=2)
    assert res["is_safe"] and res["steps_simulated"] == 2


def test_simulate_discrete_dict_state(discrete_gz):
    res = discrete_gz.simulate(SNAKE, ["south", "south", "east"])
    assert res["is_safe"] and res["survival_horizon"] == 3
    assert res["safe_prob_source"] == ["heuristic_text_world_model"]
    heads = [t["state"]["body"][0] for t in res["trajectory"]]
    assert heads == [[3, 4], [4, 4], [4, 5]]
    assert all(0.0 <= t["safe_prob"] <= 1.0 for t in res["trajectory"])


def test_simulate_discrete_lethal_reversal(discrete_gz):
    # Head at row 1 facing east: "west" reverses into the body, p_fail > 0.5.
    state = dict(SNAKE, body=[[1, 4]])
    res = discrete_gz.simulate(state, ["west", "south"])
    assert res["terminated_early"] and res["termination_step"] == 1
    assert res["survival_horizon"] == 0 and not res["is_safe"]
    assert res["trajectory"][0]["reward"] == -5.0


def test_simulate_text_episode_end_is_not_a_hazard(discrete_gz):
    # The text model ends every text episode after one step with positive reward.
    res = discrete_gz.simulate("summarize the report", ["verify", "verify"])
    assert res["terminated_early"] and res["termination_step"] == 1
    assert res["is_safe"] and res["survival_horizon"] == 1
    assert res["trajectory"][0]["done"] and not res["trajectory"][0]["hazard"]


# ---------------------------------------------------------------- what_if


def test_what_if_neural_flags_trap_and_picks_safe(neural_gz):
    _assert_fixture_separates(neural_gz)
    res = neural_gz.what_if(np.zeros(STATE_DIM), ["trap", "safe"], horizon=3)
    assert res["traps_detected"] == ["trap"]
    assert res["best_candidate"] == "safe"
    assert res["safety_ranking"] == ["safe", "trap"]
    trap, safe = res["candidate_outcomes"]["trap"], res["candidate_outcomes"]["safe"]
    assert trap["first_hazard_step"] == 1 and trap["survival_horizon"] == 0 and trap["terminated_early"]
    assert safe["is_safe"] and safe["survival_horizon"] == 3 and safe["first_hazard_step"] is None
    assert len(safe["final_state"]) == STATE_DIM
    assert not res["all_candidates_trapped"]


def test_what_if_discrete_flags_reversal_trap(discrete_gz):
    state = dict(SNAKE, body=[[1, 4]])
    res = discrete_gz.what_if(state, ["west", "south", "east"], horizon=4)
    assert res["traps_detected"] == ["west"]
    assert res["best_candidate"] in ("south", "east")
    assert res["safety_ranking"][-1] == "west"


def test_what_if_carries_provenance_per_candidate_and_top_level(discrete_gz):
    res = discrete_gz.what_if(SNAKE, ["south", "east"], horizon=2)
    assert res["provenance"] == ["symbolic_grid_world"]
    for outcome in res["candidate_outcomes"].values():
        assert outcome["provenance"] == ["symbolic_grid_world"]


def test_what_if_neural_provenance(neural_gz):
    _assert_fixture_separates(neural_gz)
    res = neural_gz.what_if(np.zeros(STATE_DIM), ["safe", "trap"], horizon=2)
    assert res["provenance"] == ["neural_residual_dynamics"]


# ---------------------------------------------------------------- audit_action


def test_audit_neural_safe_approved_and_trap_rejected(neural_gz):
    _assert_fixture_separates(neural_gz)
    ok = neural_gz.audit_action(np.zeros(STATE_DIM), "safe", horizon=3)
    assert ok["verdict"] == "APPROVED" and ok["is_safe"] and not ok["hazard_detected"]
    assert ok["survival_horizon"] == 3 and ok["risk_score"] < 0.1
    bad = neural_gz.audit_action(np.zeros(STATE_DIM), "trap", horizon=3)
    assert bad["verdict"] == "REJECT_LETHAL" and bad["hazard_detected"] and not bad["is_safe"]
    assert bad["first_hazard_step"] == 1 and bad["risk_score"] > 0.9
    assert "step 1" in bad["explanation"]


def test_audit_discrete_delayed_hazard_from_repeated_action(discrete_gz):
    # Repeating "north" from row 2 walks off the top edge after a few safe steps.
    res = discrete_gz.audit_action(SNAKE, "north", horizon=5)
    assert res["continuation_policy"] == "repeat_audited_action"
    assert res["verdict"] == "REJECT_LETHAL"
    assert res["first_hazard_step"] > 1
    assert res["survival_horizon"] == res["first_hazard_step"] - 1
    assert f"step {res['first_hazard_step']}" in res["explanation"]


def test_audit_discrete_greedy_continuation_avoids_the_edge(discrete_gz):
    res = discrete_gz.audit_action(SNAKE, "north", horizon=5, continuation_actions=["north", "south", "east"])
    assert res["verdict"] in ("APPROVED", "WARN_HAZARD") and not res["hazard_detected"]


def test_audit_discrete_warn_on_borderline_risk(discrete_gz):
    # "west" from row 2 facing east: reversal with p_fail exactly 0.5, not lethal.
    res = discrete_gz.audit_action(SNAKE, "west", horizon=1)
    assert res["verdict"] == "WARN_HAZARD" and not res["hazard_detected"]
    assert res["risk_score"] == pytest.approx(0.5, abs=1e-3)


def test_audit_text_heuristic_never_approves(discrete_gz):
    # An uncalibrated keyword-matching heuristic (fixed 0.05/0.8 constants, never fit to
    # data) must not be allowed to claim a confident APPROVED.
    res = discrete_gz.audit_action("summarize the quarterly report", "verify", horizon=1)
    assert res["verdict"] == "UNVERIFIED_HEURISTIC"
    assert res["provenance"] == ["text_heuristic"]
    assert "uncalibrated" in res["explanation"]
    assert not res["hazard_detected"]


def test_audit_text_heuristic_warn_risk_takes_precedence(discrete_gz):
    # A risk-flagged keyword still reports WARN_HAZARD ahead of the UNVERIFIED_HEURISTIC
    # downgrade: hazard/warn severity is never masked by the provenance-based verdict cap.
    res = discrete_gz.audit_action("reboot the production cluster", "execute", horizon=1)
    assert res["verdict"] == "WARN_HAZARD"
    assert res["provenance"] == ["text_heuristic"]


def test_audit_discrete_grid_can_still_approve(discrete_gz):
    # Grid provenance (symbolic_grid_world) is unaffected by the text-only downgrade.
    res = discrete_gz.audit_action(SNAKE, "east", horizon=1)
    assert res["verdict"] == "APPROVED"
    assert res["provenance"] == ["symbolic_grid_world"]


# ---------------------------------------------------------------- decide


def test_decide_return_trajectory_neural(neural_gz):
    _assert_fixture_separates(neural_gz)
    res = neural_gz.decide(np.zeros(STATE_DIM, dtype=np.float32), ["safe", "trap"], mode="mcts",
                           return_trajectory=True)
    assert res["action"] == "safe"
    assert res["trajectory_status"] == "OK"
    traj = res["trajectory"]
    assert traj["trajectory"][0]["action"] == "safe"
    assert traj["steps_simulated"] == res["adaptive_horizon"]
    assert traj["is_safe"] and traj["safe_prob_source"] == ["neural_dynamics"]
    assert all(t["safe_prob"] > 0.9 for t in traj["trajectory"])


def test_decide_without_flag_has_no_trajectory(neural_gz):
    res = neural_gz.decide(np.zeros(STATE_DIM, dtype=np.float32), ["safe", "trap"], mode="mcts")
    assert "trajectory" not in res and "trajectory_status" not in res


def test_decide_return_trajectory_caller_transition(discrete_gz):
    calls = []

    def sim(s, a):
        calls.append(a)
        return s, (1.0 if a == "go" else -1.0), a == "stop"

    res = discrete_gz.decide({"x": 0}, ["go", "stop"], mode="mcts", transition_fn=sim, return_trajectory=True)
    traj = res["trajectory"]
    assert traj["safe_prob_source"] == ["caller_transition_fn"]
    assert all(t["safe_prob"] is None for t in traj["trajectory"])
    assert traj["trajectory"][0]["action"] == res["action"]


def test_decide_empty_candidates_reports_no_trajectory(discrete_gz):
    res = discrete_gz.decide(SNAKE, [], return_trajectory=True)
    assert res["trajectory"] is None and res["trajectory_status"] == "NO_CANDIDATES"


# ---------------------------------------------------------------- fail-closed


@pytest.mark.parametrize("actions", [[], "north", None])
def test_simulate_rejects_empty_or_non_sequence_actions(discrete_gz, actions):
    with pytest.raises(ValueError):
        discrete_gz.simulate(SNAKE, actions)


@pytest.mark.parametrize("horizon", [0, -1, 2.5, True])
def test_simulate_rejects_bad_horizon(discrete_gz, horizon):
    with pytest.raises(ValueError):
        discrete_gz.simulate(SNAKE, ["south"], horizon=horizon)


def test_simulate_rejects_horizon_beyond_plan(discrete_gz):
    with pytest.raises(ValueError, match="exceeds"):
        discrete_gz.simulate(SNAKE, ["south"], horizon=3)


def test_vector_state_without_neural_model_fails_closed(discrete_gz):
    with pytest.raises(ValueError, match="neural dynamics"):
        discrete_gz.simulate(np.zeros(STATE_DIM), ["safe"])


def test_decide_vector_state_without_neural_model_fails_closed(discrete_gz):
    # The old decide closure swallowed this and fed the vector to the grid heuristic.
    with pytest.raises(ValueError, match="neural dynamics"):
        discrete_gz.decide(np.zeros(STATE_DIM, dtype=np.float32), ["safe", "trap"], mode="mcts")


def test_bare_grid_position_is_rejected(discrete_gz):
    with pytest.raises(ValueError):
        discrete_gz.simulate((3, 3), ["north"])


def test_neural_wrong_dim_and_unknown_action_fail_closed(neural_gz):
    with pytest.raises(ValueError, match="shape"):
        neural_gz.simulate(np.zeros(STATE_DIM + 1), ["safe"])
    with pytest.raises(ValueError, match="encode action"):
        neural_gz.simulate(np.zeros(STATE_DIM), ["jump"])
    with pytest.raises(ValueError, match="not numeric"):
        neural_gz.simulate(["a", "b", "c", "d"], ["safe"])


def test_unloaded_neural_model_raises_runtime_error(neural_gz):
    original = neural_gz.neural_dynamics_model
    neural_gz.neural_dynamics_model = NeuralDynamicsWorldModel(STATE_DIM, 2, hidden_dim=32, action_vocab=ACTIONS)
    try:
        with pytest.raises(RuntimeError, match="fail closed"):
            neural_gz.simulate(np.zeros(STATE_DIM), ["safe"])
    finally:
        neural_gz.neural_dynamics_model = original


def test_unsupported_state_type_and_non_string_discrete_action(discrete_gz):
    with pytest.raises(ValueError, match="unsupported state type"):
        discrete_gz.simulate(42, ["north"])
    with pytest.raises(ValueError, match="string action"):
        discrete_gz.simulate(SNAKE, [3])


def test_unknown_dict_schema_fails_closed_instead_of_silent_zero_body(discrete_gz):
    # Reviewer-flagged bug: an arbitrary dict used to fall through to `body=[[0, 0]]` and
    # step as if it were a snake/grid state, reporting a fake safe outcome.
    with pytest.raises(ValueError, match=r"Unsupported dict state schema: \['balance'\]"):
        discrete_gz.simulate({"balance": 100}, ["transfer_all_funds"])


def test_dict_schema_rejects_malformed_size_or_body(discrete_gz):
    with pytest.raises(ValueError, match="Unsupported dict state schema"):
        discrete_gz.simulate({"size": 0, "body": [[0, 0]]}, ["north"])
    with pytest.raises(ValueError, match="Unsupported dict state schema"):
        discrete_gz.simulate({"size": 8, "body": []}, ["north"])
    with pytest.raises(ValueError, match="Unsupported dict state schema"):
        discrete_gz.simulate({"size": 8, "body": [["a", "b"]]}, ["north"])
    with pytest.raises(ValueError, match="Unsupported dict state schema"):
        discrete_gz.simulate({"size": True, "body": [[0, 0]]}, ["north"])


def test_unknown_grid_action_fails_closed_instead_of_silent_noop(discrete_gz):
    # Reviewer-flagged bug: an unknown action used to map to a (0, 0) no-op step via
    # `dirs.get(action, (0, 0))` and report a fake safe outcome instead of raising.
    with pytest.raises(ValueError, match="Unsupported action 'UNKNOWN'"):
        discrete_gz.simulate(SNAKE, ["UNKNOWN"])
    with pytest.raises(ValueError, match="Unsupported action"):
        discrete_gz.audit_action(SNAKE, "transfer_all_funds")


@pytest.mark.parametrize("candidates", [[], ["a", "a"], "north"])
def test_what_if_rejects_bad_candidates(discrete_gz, candidates):
    with pytest.raises(ValueError):
        discrete_gz.what_if(SNAKE, candidates)


def test_audit_rejects_bad_warn_risk(discrete_gz):
    with pytest.raises(ValueError):
        discrete_gz.audit_action(SNAKE, "north", warn_risk=0.0)
    with pytest.raises(ValueError):
        discrete_gz.audit_action(SNAKE, "north", warn_risk=float("nan"))


def test_caller_transition_bad_return_fails_closed(discrete_gz):
    # T3-M01: a single candidate no longer bypasses the real MCTS trajectory rollout,
    # so the malformed (non-tuple) transition_fn return is now caught immediately
    # inside the engine's own unpacking, before the request ever reaches the separate
    # wrap_caller_transition-validated trajectory-reconstruction step below (which is
    # where the friendlier "transition_fn must return" message used to fire, only
    # because the single-candidate case previously skipped real MCTS calling
    # transition_fn at all -- a multi-candidate MCTS request already hit this same
    # raw error path before this fix). Either way it still fails closed with
    # ValueError; no wrong answer is ever silently returned.
    with pytest.raises(ValueError, match="not enough values to unpack"):
        discrete_gz.decide({"x": 0}, ["go"], mode="mcts", transition_fn=lambda s, a: s, return_trajectory=True)


# ---------------------------------------------------------------- HTTP service


@pytest.fixture(scope="module")
def http(neural_gz):
    import gen_zero.service.app as service

    token = "test-wm-endpoints-token"
    old_key, old_client = os.environ.get("GENZERO_API_KEY"), service.client
    os.environ["GENZERO_API_KEY"] = token
    neural_gz.weights_loaded_from_checkpoint = True
    service.client = neural_gz
    try:
        yield TestClient(service.app), {"Authorization": f"Bearer {token}"}
    finally:
        service.client = old_client
        if old_key is None:
            os.environ.pop("GENZERO_API_KEY", None)
        else:
            os.environ["GENZERO_API_KEY"] = old_key


def test_http_endpoints_route_to_client(http):
    tc, hdr = http
    zero = [0.0] * STATE_DIM
    sim = tc.post("/v1/simulate", json={"state": zero, "actions": ["safe", "trap"]}, headers=hdr)
    assert sim.status_code == 200, sim.text
    assert sim.json()["termination_step"] == 2
    wif = tc.post("/v1/what_if", json={"state": zero, "candidates": ["trap", "safe"], "horizon": 2}, headers=hdr)
    assert wif.status_code == 200 and wif.json()["best_candidate"] == "safe"
    aud = tc.post("/v1/audit_action", json={"state": zero, "action": "trap"}, headers=hdr)
    assert aud.status_code == 200 and aud.json()["verdict"] == "REJECT_LETHAL"


def test_http_bad_input_is_400_and_auth_required(http):
    tc, hdr = http
    bad = tc.post("/v1/simulate", json={"state": [0.0] * STATE_DIM, "actions": []}, headers=hdr)
    assert bad.status_code == 400
    assert bad.json()["detail"]["error"]["code"] == "invalid_world_model_request"
    assert tc.post("/v1/what_if", json={"state": "x", "candidates": ["a"]}).status_code == 401


def test_http_unsupported_dict_schema_is_400(http):
    tc, hdr = http
    resp = tc.post(
        "/v1/simulate",
        json={"state": {"balance": 100}, "actions": ["transfer_all_funds"]},
        headers=hdr,
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"]["error"]["code"] == "invalid_world_model_request"


def test_http_unknown_grid_action_is_400(http):
    tc, hdr = http
    resp = tc.post("/v1/simulate", json={"state": SNAKE, "actions": ["UNKNOWN"]}, headers=hdr)
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"]["error"]["code"] == "invalid_world_model_request"


def test_http_horizon_bool_is_rejected(http):
    tc, hdr = http
    zero = [0.0] * STATE_DIM
    resp = tc.post("/v1/simulate", json={"state": zero, "actions": ["safe"], "horizon": True}, headers=hdr)
    assert resp.status_code in (400, 422), resp.text


def test_http_decide_step_return_trajectory(http):
    # A grid-shaped state with grid-whitelisted affordances: the smooth rollout succeeds,
    # and its first step must be the exact action decide_step chose (`target`), not an
    # independently re-decided one from a second, unrelated planning pass.
    tc, hdr = http
    resp = tc.post(
        "/v1/decide_step",
        json={
            "state": SNAKE,
            "affordances": ["north", "south", "east", "west"],
            "return_trajectory": True,
        },
        headers=hdr,
    )
    assert resp.status_code == 200, resp.text
    step = resp.json()["step"]
    assert step["trajectory_status"] == "OK"
    assert step["trajectory"] is not None
    assert step["trajectory"]["provenance"] == ["symbolic_grid_world"]
    assert step["trajectory"]["trajectory"][0]["action"] == step["target"]


def test_http_decide_step_return_trajectory_unsupported_state_is_explicit_not_silent(http):
    # A UI state dict doesn't fit the symbolic grid schema; decide_step must say so
    # explicitly rather than faking a rollout (the exact silent fallback this task closes)
    # or crashing the whole request. The core target/action answer is unaffected.
    tc, hdr = http
    resp = tc.post(
        "/v1/decide_step",
        json={
            "state": {"page": "checkout", "total": 50},
            "affordances": ["#btn-submit", "#btn-cancel"],
            "return_trajectory": True,
        },
        headers=hdr,
    )
    assert resp.status_code == 200, resp.text
    step = resp.json()["step"]
    assert step["target"] in ("#btn-submit", "#btn-cancel")
    assert step["trajectory"] is None
    assert step["trajectory_status"].startswith("UNSUPPORTED_STATE_FOR_TRAJECTORY")


def test_http_decide_step_return_trajectory_text_state_succeeds(http):
    # decide_step's usual domain (UI automation) passes the raw text state straight to
    # the rollout (no modality-router normalization), so text_heuristic provenance works.
    tc, hdr = http
    resp = tc.post(
        "/v1/decide_step",
        json={
            "state": "Order summary page",
            "affordances": ["confirm", "cancel"],
            "return_trajectory": True,
        },
        headers=hdr,
    )
    assert resp.status_code == 200, resp.text
    step = resp.json()["step"]
    assert step["trajectory_status"] == "OK"
    assert step["trajectory"]["provenance"] == ["text_heuristic"]
    assert step["trajectory"]["trajectory"][0]["action"] == step["target"]


def test_http_decide_step_without_flag_has_no_trajectory(http):
    tc, hdr = http
    resp = tc.post(
        "/v1/decide_step",
        json={"state": "Order summary page", "affordances": ["confirm", "cancel"]},
        headers=hdr,
    )
    assert resp.status_code == 200, resp.text
    step = resp.json()["step"]
    assert "trajectory" not in step and "trajectory_status" not in step


def test_http_decisions_return_trajectory_single_state(http):
    # Free text is normalized to a non-grid dict by the modality router, so the bonus
    # rollout is explicitly unsupported here; the core 'choice' answer must still come
    # back with 200, not a 500 from decide()'s trailing rollout raising ValueError.
    tc, hdr = http
    resp = tc.post(
        "/v1/decisions",
        json={
            "state": "Order summary page",
            "questions": {
                "q1": {
                    "type": "choice",
                    "instructions": "pick one",
                    "criteria": {"confirm": "confirm the order", "cancel": "cancel the order"},
                }
            },
            "return_trajectory": True,
        },
        headers=hdr,
    )
    assert resp.status_code == 200, resp.text
    answer = resp.json()["answers"]["q1"]
    assert answer["choice"] in ("confirm", "cancel")
    assert answer["trajectory"] is None
    assert answer["trajectory_status"].startswith("UNSUPPORTED_STATE_FOR_TRAJECTORY")


def test_http_decisions_return_trajectory_rejected_for_batch_states(http):
    tc, hdr = http
    resp = tc.post(
        "/v1/decisions",
        json={
            "states": ["a", "b"],
            "questions": {"q1": {"type": "noul", "instructions": "risky?", "criteria": {"true": "yes", "false": "no"}}},
            "return_trajectory": True,
        },
        headers=hdr,
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"]["error"]["code"] == "unsupported_return_trajectory_batch"
