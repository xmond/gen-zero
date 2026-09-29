"""R13-03: per-node legal masks, real multi-level expansion, and no client-side revival of pruned MCTS branches."""
import numpy as np
import pytest

from gen_zero.causal.latent_mcts import LatentMctsPlanner
from gen_zero.client import GenZero
from gen_zero.config import GenZeroConfig
from gen_zero.planner.engines.mcts_engine import MctsEngine


# ------------------------------------------------------------------ engine: per-state mask


def _alternating_env():
    """legal(s0)={A}, legal(s1)={B}, legal(s2)={A}, ...: the legal action depends on the state depth."""
    calls = []
    legal_queries = []

    def legal(s):
        legal_queries.append(s)
        return ["A"] if s % 2 == 0 else ["B"]

    def transition(s, a):
        calls.append((s, a))
        assert a in legal(s), f"transition({s}, {a}) called with an action illegal in state {s}"
        return s + 1, 0.1, False

    return legal, transition, calls, legal_queries


def test_child_states_only_see_their_own_legal_actions():
    legal, transition, calls, legal_queries = _alternating_env()
    engine = MctsEngine(num_simulations=20, max_depth=4)
    res = engine.plan(0, ["A", "B"], transition, legal_actions_fn=legal,
                      reward_fn=lambda p, a, c: 0.5)

    assert res["status"] == "OK"
    assert res["best_action"] == "A"
    assert res["root_masked_actions"] == ["B"]
    assert (0, "B") not in calls
    assert (1, "A") not in calls and (1, "B") in calls
    assert all(a == ("A" if s % 2 == 0 else "B") for s, a in calls)
    assert {1, 2, 3} <= set(legal_queries)


def test_plan_expands_below_depth_one():
    legal, transition, calls, _ = _alternating_env()
    engine = MctsEngine(num_simulations=20, max_depth=4)
    res = engine.plan(0, ["A"], transition, legal_actions_fn=legal, reward_fn=lambda p, a, c: 0.5)

    assert res["max_depth_reached"] >= 2
    assert res["max_depth_reached"] == 4
    assert any(s >= 1 for s, _ in calls), "no transition was ever run from a child state"
    assert [s for s, _ in calls] == [0, 1, 2, 3]


def test_eval_fn_gets_state_specific_actions_and_drives_values():
    seen = []

    def legal(s):
        return ["up", "down"] if s == 0 else ["up"]

    def eval_fn(s, acts):
        seen.append((s, tuple(acts)))
        return {a: 1.0 for a in acts}, 0.0

    def transition(s, a):
        return s + (1 if a == "up" else 10), (1.0 if a == "up" else -1.0), False

    engine = MctsEngine(num_simulations=16, max_depth=3)
    res = engine.plan(0, ["up", "down"], transition, legal_actions_fn=legal, eval_fn=eval_fn)

    assert res["value_source"] == "eval_fn"
    assert res["best_action"] == "up"
    assert res["max_depth_reached"] >= 2
    assert (0, ("up", "down")) in seen
    assert all(acts == ("up",) for s, acts in seen if s != 0)
    assert any(s != 0 for s, _ in seen)


def test_search_uses_the_tree_and_state_specific_legality():
    legal, transition, calls, _ = _alternating_env()
    res = MctsEngine(num_simulations=12, max_depth=3).search(
        root_state=0, get_legal_actions_fn=legal, transition_fn=transition,
        eval_fn=lambda s, acts: ({a: 1.0 for a in acts}, 0.0),
    )
    assert res["best_action"] == "A"
    assert res["max_depth_reached"] >= 2
    assert (1, "A") not in calls and (1, "B") in calls


def test_legal_actions_fn_is_mandatory():
    engine = MctsEngine(num_simulations=4)
    with pytest.raises(TypeError):
        engine.plan(0, ["A"], lambda s, a: (s, 0.0, False))
    with pytest.raises(TypeError, match="callable"):
        engine.plan(0, ["A"], lambda s, a: (s, 0.0, False), legal_actions_fn=None)


def test_zero_simulations_fail_closed():
    with pytest.raises(ValueError, match="num_simulations"):
        MctsEngine(num_simulations=0).plan(0, ["A"], lambda s, a: (s, 0.0, False),
                                           legal_actions_fn=lambda s: ["A"])


# ------------------------------------------------------------------ engine: pruning


def test_all_root_branches_falsified_returns_no_action():
    engine = MctsEngine(num_simulations=8, max_depth=3)
    res = engine.plan(0, ["x", "y"], lambda s, a: (s + 1, -10.0, True),
                      legal_actions_fn=lambda s: ["x", "y"],
                      causal_invariant_fn=lambda s, a, ns, r, done: False)
    assert res["status"] == "ALL_PRUNED"
    assert res["best_action"] is None
    assert res["visit_distribution"] == {}


def test_prune_cascades_from_depth_two_to_the_root():
    """Every grandchild violates the invariant; the parents and the root branches must fall with them."""
    def transition(s, a):
        return s + (a,), 0.0, False

    engine = MctsEngine(num_simulations=10, max_depth=3)
    res = engine.plan((), ["a", "b"], transition, legal_actions_fn=lambda s: ["a", "b"],
                      reward_fn=lambda p, a, c: 1.0,
                      causal_invariant_fn=lambda s, a, ns, r, done: len(ns) < 2)
    assert res["status"] == "ALL_PRUNED"
    assert res["best_action"] is None
    assert res["falsified_pruned_count"] >= 6


def test_non_terminal_state_without_legal_actions_is_a_dead_end():
    def legal(s):
        return ["go", "trap"] if s == "root" else ([] if s == "pit" else ["go"])

    def transition(s, a):
        return ("pit" if a == "trap" else "road"), 0.0, False

    res = MctsEngine(num_simulations=10, max_depth=3).plan(
        "root", ["go", "trap"], transition, legal_actions_fn=legal, reward_fn=lambda p, a, c: 1.0)
    assert res["best_action"] == "go"
    assert "trap" not in res["visit_distribution"]
    assert res["dead_end_count"] >= 1


# ------------------------------------------------------------------ client: no revival


@pytest.fixture
def gz():
    return GenZero(GenZeroConfig(hidden_dim=32, embed_dim=4, mcts_simulations=12))


def test_client_abstains_when_mcts_prunes_every_branch(gz):
    calls = []

    def fatal(s, a):
        calls.append(a)
        return {"x": 1}, -10.0, True

    res = gz.decide({"x": 0}, ["go", "stay"], mode="mcts", transition_fn=fatal)
    assert res["action"] == "ABSTAIN"
    assert res["action"] != "go"
    assert res["status"] == "ALL_PRUNED_ABSTAIN"
    assert res["confidence"] == 0.0
    assert set(res["probs"].values()) == {0.0}
    meta = res["expert_outputs"]["mcts"]["meta"]
    assert meta["mcts_status"] == "ALL_PRUNED"
    assert meta["valid_set"] == []
    assert calls, "MCTS never ran the transitions"


def test_expert_distribution_does_not_refill_pruned_candidates(gz):
    probs, value, best, meta = gz._execute_expert_distribution(
        expert_name="mcts", state={"x": 0}, candidates=["go", "stay"],
        trans_fn=lambda s, a: ({"x": 1}, -10.0, True),
    )
    assert best == "ABSTAIN"
    assert probs == {"go": 0.0, "stay": 0.0}
    assert meta["status"] == "ALL_PRUNED"


def test_client_drops_a_pruned_branch_from_the_valid_set(gz):
    def sim(s, a):
        if a == "jump":
            return {"x": -1}, -10.0, True
        return {"x": s["x"] + 1}, 0.5, False

    res = gz.decide({"x": 0}, ["jump", "walk"], mode="mcts", transition_fn=sim)
    meta = res["expert_outputs"]["mcts"]["meta"]
    assert meta["valid_set"] == ["walk"]
    assert res["action"] == "walk"
    assert res["probs"]["jump"] == 0.0


def test_client_mcts_uses_per_state_mask_eval_fn_and_depth(gz):
    calls = []

    def legal(s):
        if s["d"] == 0:
            return ["A", "C"]
        return ["A"] if s["d"] % 2 == 0 else ["B"]

    def sim(s, a):
        calls.append((s["d"], a))
        return {"d": s["d"] + 1}, 0.1, False

    res = gz.decide({"d": 0}, ["A", "B", "C"], mode="mcts", transition_fn=sim, legal_actions_fn=legal)
    assert res["action"] in ("A", "C")
    assert res["probs"]["B"] == 0.0
    meta = res["expert_outputs"]["mcts"]["meta"]
    assert meta["action_mask_source"] == "caller_legal_actions_fn+hard_rules"
    assert meta["value_source"] == "eval_fn"
    assert meta["max_depth_reached"] >= 2
    assert (0, "B") not in calls
    assert all(a in legal({"d": d}) for d, a in calls)
    assert any(d >= 1 for d, _ in calls)


def test_client_hard_rules_mask_child_states(gz):
    """The hard-rule engine runs at every tree state, not only at the root."""
    gz.cp_sat_solver.register_hard_rule(lambda s, a: not (isinstance(s, dict) and s.get("d", 0) >= 1 and a == "B"))
    calls = []

    def sim(s, a):
        calls.append((s["d"], a))
        return {"d": s["d"] + 1}, 0.1, False

    gz.decide({"d": 0}, ["A", "B"], mode="mcts", transition_fn=sim)
    assert (0, "B") in calls
    assert not [c for c in calls if c[0] >= 1 and c[1] == "B"]


# ------------------------------------------------------------------ latent MCTS mask


def test_latent_mcts_applies_legal_mask_at_every_node():
    rng = np.random.default_rng(3)
    emb = rng.normal(size=(4, 6))
    z0 = rng.normal(size=6)
    z0[0] = abs(z0[0]) + 0.5

    def legal(z):
        return [0, 1] if z[0] >= 0 else [2, 3]

    calls = []

    def transition(z, a):
        assert a in legal(z), f"action {a} is illegal in this latent state"
        calls.append(a)
        nz = z.copy()
        nz[0] = -nz[0]
        return nz

    best, policy, info = LatentMctsPlanner(num_simulations=40, max_depth=4).plan(
        z0, list("abcd"), emb, transition_fn=transition, legal_actions_fn=legal)
    assert best in (0, 1)
    assert policy[2] == policy[3] == 0.0
    assert info["action_mask_source"] == "legal_actions_fn"
    assert info["max_depth"] >= 2
    assert {2, 3} & set(calls), "child states never used their own legal actions"


def test_latent_mcts_without_root_legal_action_raises():
    emb = np.eye(3)
    with pytest.raises(ValueError, match="root"):
        LatentMctsPlanner(num_simulations=4).plan(np.ones(3), list("abc"), emb, legal_actions_fn=lambda z: [])
