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


# ------------------------------------------------------------------ engine: module B acceleration
#
# Sequential Halving (Karnin et al. 2013; Danihelka et al. ICLR 2022), root
# count-dominance early stopping and bounded transposition memoization. All three
# are opt-in, so every test above keeps the default PUCT behaviour.

import zlib  # noqa: E402

ARMS = [f"d{i}" for i in range(7)] + ["opt"]


def _lure_env():
    """Eight root arms. The prior puts almost no mass on ``opt``, the only arm worth 1.0."""
    def legal(s):
        return ARMS if s == "root" else ["go"]

    def transition(s, a):
        nxt = (a,) if s == "root" else s + (a,)
        return nxt, 0.0, False

    def eval_fn(s, acts):
        if s == "root":
            return {a: (0.001 if a == "opt" else 1.0) for a in acts}, 0.0
        return {a: 1.0 for a in acts}, (1.0 if s[0] == "opt" else 0.5)

    return legal, transition, eval_fn


def test_plain_puct_misses_the_low_prior_optimum_under_a_small_budget():
    legal, transition, eval_fn = _lure_env()
    res = MctsEngine(num_simulations=16, max_depth=3).plan(
        "root", ARMS, transition, legal_actions_fn=legal, eval_fn=eval_fn)
    assert res["status"] == "OK"
    assert res["best_action"] != "opt"


def test_sequential_halving_finds_the_low_prior_optimum_under_the_same_budget():
    legal, transition, eval_fn = _lure_env()
    res = MctsEngine(num_simulations=16, max_depth=3, search_strategy="sequential_halving").plan(
        "root", ARMS, transition, legal_actions_fn=legal, eval_fn=eval_fn)
    assert res.get("search_strategy") == "sequential_halving"
    assert res["status"] == "OK"
    assert res["best_action"] == "opt"
    assert res.get("simulations_run") == 16
    # Every arm got at least one simulation in the first halving round.
    assert set(res["visit_distribution"]) == set(ARMS)
    assert all(v > 0 for v in res["visit_distribution"].values())


def test_sequential_halving_never_returns_a_falsified_arm():
    legal, transition, eval_fn = _lure_env()
    res = MctsEngine(num_simulations=16, max_depth=3, search_strategy="sequential_halving").plan(
        "root", ARMS, transition, legal_actions_fn=legal, eval_fn=eval_fn,
        causal_invariant_fn=lambda s, a, ns, r, done: ns[0] != "opt")
    assert res["status"] == "OK"
    assert res["best_action"] != "opt"
    assert "opt" not in res["visit_distribution"]


def test_sequential_halving_keeps_the_all_pruned_gate():
    legal, transition, eval_fn = _lure_env()
    res = MctsEngine(num_simulations=16, max_depth=3, search_strategy="sequential_halving").plan(
        "root", ARMS, transition, legal_actions_fn=legal, eval_fn=eval_fn,
        causal_invariant_fn=lambda s, a, ns, r, done: len(ns) < 2)
    assert res["status"] == "ALL_PRUNED"
    assert res["best_action"] is None


def test_sequential_halving_keeps_the_non_finite_reward_gate_below_the_root():
    legal, _, eval_fn = _lure_env()

    def transition(s, a):
        nxt = (a,) if s == "root" else s + (a,)
        return nxt, (float("nan") if s == ("opt",) else 0.0), False

    res = MctsEngine(num_simulations=16, max_depth=3, search_strategy="sequential_halving").plan(
        "root", ARMS, transition, legal_actions_fn=legal, eval_fn=eval_fn)
    assert res["status"] == "NON_FINITE_TRANSITION_REWARD"
    assert res["best_action"] is None


def test_unknown_search_strategy_fails_closed():
    with pytest.raises(ValueError, match="search_strategy"):
        MctsEngine(search_strategy="greedy")


def _two_arm_env(calls):
    def legal(s):
        return ["good", "bad"] if s == () else ["x", "y"]

    def transition(s, a):
        calls.append((s, a))
        return s + (a,), 0.0, False

    def eval_fn(s, acts):
        return {a: 1.0 for a in acts}, (0.0 if s and s[0] == "bad" else 1.0)

    return legal, transition, eval_fn


def test_default_engine_runs_the_full_budget_without_early_stop():
    legal, transition, eval_fn = _two_arm_env([])
    res = MctsEngine(num_simulations=100, max_depth=3).plan(
        (), ["good", "bad"], transition, legal_actions_fn=legal, eval_fn=eval_fn)
    assert res.get("early_stopped") is False
    assert res.get("simulations_run") == 100


def test_count_dominance_early_stop_keeps_the_decision_and_saves_work():
    full_calls, fast_calls = [], []
    legal, transition, eval_fn = _two_arm_env(full_calls)
    full = MctsEngine(num_simulations=100, max_depth=10).plan(
        (), ["good", "bad"], transition, legal_actions_fn=legal, eval_fn=eval_fn)
    legal, transition, eval_fn = _two_arm_env(fast_calls)
    fast = MctsEngine(num_simulations=100, max_depth=10, early_stop=True).plan(
        (), ["good", "bad"], transition, legal_actions_fn=legal, eval_fn=eval_fn)

    assert fast["status"] == full["status"] == "OK"
    assert fast["best_action"] == full["best_action"] == "good"
    assert fast.get("early_stopped") is True
    assert fast["simulations_run"] < 100
    assert len(fast_calls) < len(full_calls)
    counts = sorted(fast["root_visit_counts"].values(), reverse=True)
    # The certificate: the runner-up cannot catch up even with every remaining simulation.
    assert counts[0] - counts[1] > 100 - fast["simulations_run"]


def _random_tree_env(seed):
    def legal(s):
        return ["a", "b", "c"]

    def h(s):
        return zlib.crc32(repr((seed,) + s).encode()) / 2**32

    def transition(s, a):
        nxt = s + (a,)
        return nxt, h(nxt) - 0.5, False

    def eval_fn(s, acts):
        return {a: 0.2 + h(s + (a, "p")) for a in acts}, h(s + ("v",))

    return legal, transition, eval_fn


def test_count_dominance_early_stop_matches_the_full_search_decision_on_random_trees():
    stopped = 0
    for seed in range(40):
        legal, transition, eval_fn = _random_tree_env(seed)
        full = MctsEngine(num_simulations=80, max_depth=4).plan(
            (), ["a", "b", "c"], transition, legal_actions_fn=legal, eval_fn=eval_fn)
        fast = MctsEngine(num_simulations=80, max_depth=4, early_stop=True).plan(
            (), ["a", "b", "c"], transition, legal_actions_fn=legal, eval_fn=eval_fn)
        assert fast["best_action"] == full["best_action"], seed
        stopped += bool(fast.get("early_stopped"))
    assert stopped > 0


def test_early_stop_is_not_applied_when_a_causal_invariant_can_still_prune():
    legal, transition, eval_fn = _two_arm_env([])
    res = MctsEngine(num_simulations=100, max_depth=3, early_stop=True).plan(
        (), ["good", "bad"], transition, legal_actions_fn=legal, eval_fn=eval_fn,
        causal_invariant_fn=lambda s, a, ns, r, done: True)
    assert res.get("early_stopped") is False
    assert res.get("simulations_run") == 100


def _grid_env(transition_calls, legal_calls, state_type=tuple):
    """Moves R/U on a grid: R-then-U and U-then-R reach the same cell (a transposition)."""
    def legal(s):
        legal_calls.append(s)
        return ["R", "U"]

    def transition(s, a):
        transition_calls.append((s, a))
        x, y = (s["x"], s["y"]) if isinstance(s, dict) else s
        x, y = (x + 1, y) if a == "R" else (x, y + 1)
        nxt = {"x": x, "y": y} if state_type is dict else (x, y)
        return nxt, (0.1 if a == "R" else 0.0), False

    def eval_fn(s, acts):
        x, y = (s["x"], s["y"]) if isinstance(s, dict) else s
        return {a: 1.0 for a in acts}, -0.1 * abs(x - 2 * y)

    return legal, transition, eval_fn


_SAME_RESULT_KEYS = ("status", "best_action", "visit_distribution", "root_q", "expected_value",
                     "nodes_expanded", "max_depth_reached")


def test_transposition_cache_cuts_transitions_and_legal_queries_without_changing_the_result():
    plain_t, plain_l, cached_t, cached_l = [], [], [], []
    legal, transition, eval_fn = _grid_env(plain_t, plain_l)
    plain = MctsEngine(num_simulations=120, max_depth=5).plan(
        (0, 0), ["R", "U"], transition, legal_actions_fn=legal, eval_fn=eval_fn)
    legal, transition, eval_fn = _grid_env(cached_t, cached_l)
    cached = MctsEngine(num_simulations=120, max_depth=5, transposition_cache=True).plan(
        (0, 0), ["R", "U"], transition, legal_actions_fn=legal, eval_fn=eval_fn)

    assert cached.get("cache_hits", 0) > 0
    assert cached["transition_cache_hits"] > 0 and cached["legal_cache_hits"] > 0
    assert len(cached_t) < len(plain_t)
    assert len(cached_l) < len(plain_l)
    assert cached["transition_calls"] == len(cached_t)
    assert cached["legal_calls"] == len(cached_l)
    assert {k: cached[k] for k in _SAME_RESULT_KEYS} == {k: plain[k] for k in _SAME_RESULT_KEYS}


def test_transposition_cache_skips_unhashable_states():
    plain_t, cached_t = [], []
    legal, transition, eval_fn = _grid_env(plain_t, [], state_type=dict)
    plain = MctsEngine(num_simulations=40, max_depth=4).plan(
        {"x": 0, "y": 0}, ["R", "U"], transition, legal_actions_fn=legal, eval_fn=eval_fn)
    legal, transition, eval_fn = _grid_env(cached_t, [], state_type=dict)
    cached = MctsEngine(num_simulations=40, max_depth=4, transposition_cache=True).plan(
        {"x": 0, "y": 0}, ["R", "U"], transition, legal_actions_fn=legal, eval_fn=eval_fn)
    assert cached["status"] == "OK"
    assert cached.get("cache_hits") == 0
    assert cached["cache_uncacheable"] > 0
    assert len(cached_t) == len(plain_t)
    assert {k: cached[k] for k in _SAME_RESULT_KEYS} == {k: plain[k] for k in _SAME_RESULT_KEYS}


def test_transposition_cache_keeps_the_per_state_legal_mask():
    legal, transition, calls, _ = _alternating_env()
    res = MctsEngine(num_simulations=20, max_depth=4, transposition_cache=True).plan(
        0, ["A", "B"], transition, legal_actions_fn=legal, reward_fn=lambda p, a, c: 0.5)
    assert res["status"] == "OK"
    assert res["best_action"] == "A"
    assert all(a == ("A" if s % 2 == 0 else "B") for s, a in calls)
    assert res.get("transition_calls") == len(calls)


def test_transposition_cache_keeps_the_non_finite_reward_gate():
    res = MctsEngine(num_simulations=8, max_depth=3, transposition_cache=True).plan(
        0, ["A"], lambda s, a: (s + 1, float("inf") if s == 1 else 0.0, False),
        legal_actions_fn=lambda s: ["A"], reward_fn=lambda p, a, c: 0.5)
    assert res["status"] == "NON_FINITE_TRANSITION_REWARD"
    assert res["best_action"] is None


def test_count_dominance_waits_until_every_live_root_child_is_visited():
    from gen_zero.planner.engines.mcts_engine import MctsNode64B, _count_dominance

    root = MctsNode64B(0)
    nodes = {0: root}
    for cid, (visits, value) in enumerate(((60, 30.0), (3, 0.3), (0, 0.0)), start=1):
        node = MctsNode64B(cid, parent_id=0, action_idx=cid - 1, depth=1)
        node.visit_count, node.value_sum = visits, value
        nodes[cid] = node
        root.children_ids.append(cid)
    # The lead (57) exceeds the remaining budget (10), but child 3 was never tried.
    assert _count_dominance(nodes, root, remaining=10) is None
    nodes[3].visit_count, nodes[3].value_sum = 1, 0.1
    assert _count_dominance(nodes, root, remaining=10) == "count_dominance"
    # A falsified, unvisited child is not live and does not block the stop.
    nodes[3].visit_count, nodes[3].falsified = 0, True
    assert _count_dominance(nodes, root, remaining=10) == "count_dominance"
