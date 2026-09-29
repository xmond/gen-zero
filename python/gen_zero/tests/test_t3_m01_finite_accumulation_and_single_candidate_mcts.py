"""Regression tests for the ChatGPT 6 Pro third-round audit finding T3-M01:

1. Accumulated-return overflow: every individual step reward (or eval_fn/reward_fn
   value) can be finite on its own (e.g. 1e308) and still pass the existing X-M02
   per-step ``_require_finite_reward`` / ``math.isfinite`` checks. Repeated
   backpropagation across many simulations (``value_sum += v``) can then overflow the
   node's accumulator to +/-inf, and the previous code returned ``status = OK`` with
   ``best_action`` chosen on top of that inf-poisoned value. This must instead fail
   closed with a dedicated ``NON_FINITE_RETURN`` status.

2. Single-candidate MCTS bypass: ``GenZero._execute_expert_distribution`` used to
   short-circuit to a bare ``return {candidates[0]: 1.0}, 0.0, candidates[0], meta``
   whenever ``len(candidates) == 1``, *even when the caller explicitly requested the
   MCTS expert*. That skipped the entire trajectory-transition rollout and the
   causal/PRM safety verification MCTS performs at every simulated node, silently
   assuming a single candidate is safe by construction. It is not: this test proves
   both that the transition function is now actually invoked, and that a PRM-flagged
   sole candidate is correctly abstained rather than passed straight through.
"""
import math

import pytest

from gen_zero.client import GenZero
from gen_zero.config import GenZeroConfig
from gen_zero.planner.engines.mcts_engine import MctsEngine


# ============================================================= T3-M01 accumulation


def test_mcts_fails_closed_when_finite_step_reward_overflows_value_sum_via_reward_fn():
    """Single legal action, every simulation lands on the same depth-1 leaf, and
    reward_fn returns a large-but-finite 1e308 on every call. Each individual value
    passes math.isfinite, but node.value_sum accumulates 1e308 + 1e308 + 1e308 across
    3 simulations, which overflows float64 (max ~1.7977e308) to +inf. The old code had
    no post-accumulation check and would have returned status=OK with
    expected_value=inf.
    """
    def transition(state, action):
        return state + 1, 1.0, False

    def legal(state):
        return ["A"]

    engine = MctsEngine(num_simulations=3, max_depth=1, seed=1)
    res = engine.plan(0, ["A"], transition, legal_actions_fn=legal, reward_fn=lambda p, a, c: 1e308)

    assert res["status"] == "NON_FINITE_RETURN"
    assert res["best_action"] is None
    assert res["visit_distribution"] == {}
    assert math.isfinite(res["expected_value"])


def test_mcts_fails_closed_when_finite_leaf_value_overflows_value_sum_via_eval_fn():
    """Same overflow, reached through the eval_fn leaf-value path instead of reward_fn:
    eval_fn's returned value (1e308) is finite on its own, but repeated backprop across
    simulations still overflows the accumulator.
    """
    def transition(state, action):
        return state + 1, 1.0, False

    def legal(state):
        return ["A"]

    def eval_fn(state, actions):
        return {a: 1.0 for a in actions}, 1e308

    engine = MctsEngine(num_simulations=3, max_depth=1, seed=1)
    res = engine.plan(0, ["A"], transition, legal_actions_fn=legal, eval_fn=eval_fn)

    assert res["status"] == "NON_FINITE_RETURN"
    assert res["best_action"] is None
    assert math.isfinite(res["expected_value"])


def test_mcts_fails_closed_when_imagined_rollout_accumulation_overflows():
    """Dynamics-model path: each per-step reward from dynamics_model.step is finite
    (1e308) and passes _require_finite_reward, but _imagined_value's running sum
    (value += scale * reward) overflows across the rollout horizon.
    """
    import numpy as np

    class _HugeFiniteRewardDynamics:
        def step(self, z, action):
            return z + 1.0, 1e308, False

    engine = MctsEngine(num_simulations=2, max_depth=4, dynamics_model=_HugeFiniteRewardDynamics())
    root = np.array([0.0], dtype=np.float32)
    res = engine.plan(root, ["A"], transition_fn=None, legal_actions_fn=lambda s: ["A"])

    assert res["status"] == "NON_FINITE_RETURN"
    assert res["best_action"] is None


def test_mcts_stays_ok_and_finite_for_ordinary_finite_accumulation():
    """Sanity/no-regression: ordinary finite rewards across many simulations must still
    return status=OK with a finite expected_value. The new checks must not misfire on
    legitimate, non-overflowing accumulation.
    """
    def transition(state, action):
        return state + 1, 1.0, False

    def legal(state):
        return ["A"]

    engine = MctsEngine(num_simulations=20, max_depth=2, seed=1)
    res = engine.plan(0, ["A"], transition, legal_actions_fn=legal, reward_fn=lambda p, a, c: 1.0)

    assert res["status"] == "OK"
    assert res["best_action"] == "A"
    assert math.isfinite(res["expected_value"])


def test_client_decide_abstains_when_mcts_return_accumulation_overflows():
    """End-to-end through GenZero.decide(): the accumulation overflow must surface as
    an honest top-level ABSTAIN status, exactly like the existing X-M02
    NON_FINITE_TRANSITION_REWARD_ABSTAIN wiring, not a silent generic abstain.
    """
    gz = GenZero(GenZeroConfig(hidden_dim=32, embed_dim=4, mcts_simulations=3))

    def huge_finite_transition(s, a):
        return {"x": s["x"] + 1}, 1e308, False

    res = gz.decide({"x": 0}, ["A", "B"], mode="mcts", transition_fn=huge_finite_transition)

    assert res["action"] == "ABSTAIN"
    assert res["status"] == "NON_FINITE_RETURN_ABSTAIN"
    assert set(res["probs"].values()) == {0.0}
    assert not (isinstance(res.get("value"), float) and math.isnan(res["value"]))
    assert not (isinstance(res.get("value"), float) and math.isinf(res["value"]))


# ================================================== T3-M01 single-candidate bypass


def test_execute_expert_distribution_runs_real_mcts_for_a_single_candidate():
    """Directly reproduces the reported bypass: expert_name == "mcts" with exactly one
    candidate used to hit ``if expert_name in ("reflex", "fast") or len(candidates) ==
    1`` and short-circuit straight to ``return {candidates[0]: 1.0}, 0.0, ...`` without
    ever calling trans_fn. It must instead fall through to the real PUCT search.
    """
    gz = GenZero(GenZeroConfig(hidden_dim=32, embed_dim=4, mcts_simulations=4))

    calls = {"n": 0}

    def trans_fn(s, a):
        calls["n"] += 1
        return {"x": s["x"] + 1}, 1.0, False

    probs, val, best, meta = gz._execute_expert_distribution(
        expert_name="mcts",
        state={"x": 0},
        candidates=["only"],
        trans_fn=trans_fn,
    )

    assert calls["n"] > 0, "a real MCTS search must invoke the transition function at least once"
    assert "mcts_status" in meta, "only the real MCTS code path sets mcts_status in expert meta"
    assert meta["mcts_status"] == "OK"
    assert best == "only"


def test_execute_expert_distribution_single_candidate_reflex_still_uses_the_shortcut():
    """Non-regression: the reflex/fast experts must keep their single-candidate
    shortcut (they do not simulate trajectories; only mcts's contract requires it).
    """
    gz = GenZero(GenZeroConfig(hidden_dim=32, embed_dim=4, mcts_simulations=4))

    calls = {"n": 0}

    def trans_fn(s, a):
        calls["n"] += 1
        return {"x": s["x"] + 1}, 1.0, False

    probs, val, best, meta = gz._execute_expert_distribution(
        expert_name="reflex",
        state={"x": 0},
        candidates=["only"],
        trans_fn=trans_fn,
    )

    assert calls["n"] == 0, "reflex keeps its single-candidate shortcut and never calls trans_fn"
    assert best == "only"
    assert probs == {"only": 1.0}


def test_client_decide_mcts_abstains_when_prm_verifier_prunes_the_sole_candidate():
    """Proves the single-candidate MCTS path actually runs safety verification: when
    the process-reward-model verifier flags the sole candidate's transition as unsafe,
    the search must fall back to ABSTAIN, not blindly return the one candidate it was
    handed. Before the fix this path never called the verifier at all.
    """
    gz = GenZero(GenZeroConfig(hidden_dim=32, embed_dim=4, mcts_simulations=4))
    gz.prm_verifier.verify_step = lambda *a, **k: {"should_prune": True}

    def trans_fn(s, a):
        return {"x": s["x"] + 1}, 1.0, False

    res = gz.decide({"x": 0}, ["only"], mode="mcts", transition_fn=trans_fn)

    assert res["action"] == "ABSTAIN"
    assert res["status"] == "ALL_PRUNED_ABSTAIN"


def test_client_decide_mcts_single_candidate_end_to_end_when_safe():
    """Companion positive case: a genuinely safe sole candidate must still be returned
    as the decision, proving the fix does not turn every single-candidate MCTS request
    into an abstain -- only unsafe ones.
    """
    gz = GenZero(GenZeroConfig(hidden_dim=32, embed_dim=4, mcts_simulations=4))

    def trans_fn(s, a):
        return {"x": s["x"] + 1}, 1.0, False

    res = gz.decide({"x": 0}, ["only"], mode="mcts", transition_fn=trans_fn)

    assert res["action"] == "only"
