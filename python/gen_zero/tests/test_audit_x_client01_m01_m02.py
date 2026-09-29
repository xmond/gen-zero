"""Regression tests for the ChatGPT 6 Pro second-round audit findings:

X-CLIENT01: the consensus hard support set must propagate through the whole decision
pipeline. A candidate excluded by the consensus valid set must never resurface through
the arbiter fallback or ``_apply_action_constraints``, even if it still carries higher
stale probability mass and no caller constraint names it.

X-M01: MCTS backpropagation must implement the real Bellman discounted-return recursion
G_t = r_t + gamma * G_{t+1}. The previous code only discounted the accumulator
(``v *= discount``) and never added the transition reward of each edge on the path,
so a high-immediate-reward path (+100 then -1) lost its +100 entirely and could be
outscored by a low-then-high path (0 then +10).

X-M02: a transition/step (or ``reward_fn``) returning a non-finite (NaN/Inf) reward must
make MCTS fail closed with a dedicated non-OK status. It must never return
``status = OK`` with a best_action chosen on top of a poisoned NaN value.
"""
import math

import numpy as np
import pytest

from gen_zero.client import GenZero
from gen_zero.config import GenZeroConfig
from gen_zero.gate.constraint_compiler import ConstraintLinearProjectionCompiler
from gen_zero.planner.engines.mcts_engine import MctsEngine


# ==================================================================== X-CLIENT01


def test_apply_action_constraints_never_resurrects_excluded_candidate():
    """Direct reproduction of the reported scenario: consensus support = {B}, but
    accum_probs still carries A=0.6, B=0.4 and no caller constraint mentions either
    action. Without the hard_support_set gate, project_probabilities would echo A's
    stale 0.6 back unchanged and _apply_action_constraints would re-pick A because
    0.6 > 0.4 (top-probability argmax over the wrong, unfiltered universe).
    """
    compiler = ConstraintLinearProjectionCompiler()
    compiler.compile_action_constraints([])  # caller supplied zero constraints

    probs, best, confidence, report, disabled = GenZero._apply_action_constraints(
        compiler,
        ["A", "B"],
        {"A": 0.6, "B": 0.4},
        best_action="B",
        confidence=0.4,
        hard_support_set=["B"],
    )

    assert best == "B", "A candidate outside the hard support set must never be re-selected"
    assert probs["A"] == 0.0
    assert probs["B"] == pytest.approx(1.0)
    assert report["status"] == "OK"


def test_apply_action_constraints_fails_closed_when_support_set_conflicts_with_constraints():
    """A caller lower_bound constraint that demands mass on an already-excluded action
    must never be satisfied by reviving that action; the correct fail-closed answer is
    ABSTAIN with an all-zero distribution, not a silent expansion of the support set.
    """
    compiler = ConstraintLinearProjectionCompiler()
    compiler.compile_action_constraints([{"type": "lower_bound", "action": "A", "value": 0.5}])

    probs, best, confidence, report, disabled = GenZero._apply_action_constraints(
        compiler,
        ["A", "B"],
        {"A": 0.6, "B": 0.4},
        best_action="B",
        confidence=0.4,
        hard_support_set=["B"],
    )

    assert best == "ABSTAIN"
    assert set(probs.values()) == {0.0}


def test_apply_action_constraints_passthrough_when_pick_already_inside_support_set():
    """Sanity check: when the pick is legitimately the top candidate inside the hard
    support set, the gate must not perturb the outcome.
    """
    compiler = ConstraintLinearProjectionCompiler()
    compiler.compile_action_constraints([])

    probs, best, confidence, report, disabled = GenZero._apply_action_constraints(
        compiler,
        ["A", "B", "C"],
        {"A": 0.1, "B": 0.7, "C": 0.2},
        best_action="B",
        confidence=0.7,
        hard_support_set=["B", "C"],
    )

    assert best == "B"
    assert probs["A"] == 0.0
    assert probs["B"] + probs["C"] == pytest.approx(1.0)


@pytest.fixture
def gz():
    return GenZero(GenZeroConfig(hidden_dim=32, embed_dim=4, mcts_simulations=12))


@pytest.fixture
def gz_no_arbiter():
    """Arbiter fallback disabled: isolates the hard-support-set gate (X-CLIENT01) from
    the cloud-arbiter re-validation step, which already independently filters to
    ``valid_candidates`` and would otherwise mask a regression in the gate itself.
    """
    return GenZero(GenZeroConfig(
        hidden_dim=32, embed_dim=4, mcts_simulations=12, enable_gpu_arbiter_fallback=False,
    ))


def test_decide_end_to_end_never_leaks_a_candidate_outside_the_consensus_support_set(gz_no_arbiter):
    """End-to-end through the public ``decide()`` entrypoint: a single expert declares
    a hard valid_set of {B} while still emitting higher raw probability mass on A. With
    a caller constraint compiler attached (but zero actual constraints, matching the
    reported scenario), the final decision must be B with A's probability at exactly 0.
    """

    def fake_expert(**kwargs):
        candidates = kwargs["candidates"]
        probs = {c: 0.0 for c in candidates}
        probs["A"] = 0.6
        probs["B"] = 0.4
        return probs, 0.6, "A", {"valid_set": ["B"], "status": "OK"}

    gz_no_arbiter._execute_expert_distribution = fake_expert
    res = gz_no_arbiter.decide(
        {"x": 0}, ["A", "B"], mode="mcts",
        transition_fn=lambda s, a: (s, 0.0, False),
        constraints=[],
    )

    assert res["action"] == "B"
    assert res["probs"]["A"] == 0.0
    assert res["probs"]["B"] == pytest.approx(1.0)


# ==================================================================== X-M01


def _branching_path_env():
    """root -[A: +100]-> s_A -[X: -1, done]-> terminal
    root -[B: 0]-> s_B -[X: +10, done]-> terminal

    Correct discounted return: G(A) = 100 + gamma*(-1) ~= 99.0; G(B) = 0 + gamma*10 ~= 9.9.
    A must win despite B's larger *second*-step reward, because A's dominant first-step
    reward must not be dropped by backpropagation.
    """
    def transition(state, action):
        if state == "root" and action == "A":
            return "s_A", 100.0, False
        if state == "root" and action == "B":
            return "s_B", 0.0, False
        if state == "s_A" and action == "X":
            return "s_A_term", -1.0, True
        if state == "s_B" and action == "X":
            return "s_B_term", 10.0, True
        raise AssertionError(f"unexpected transition ({state!r}, {action!r})")

    def legal(state):
        if state == "root":
            return ["A", "B"]
        if state in ("s_A", "s_B"):
            return ["X"]
        return []

    def eval_fn(state, actions):
        return {a: 1.0 / len(actions) for a in actions}, 0.0

    return transition, legal, eval_fn


def test_mcts_backprop_keeps_the_high_immediate_reward_path():
    transition, legal, eval_fn = _branching_path_env()
    # A high c_puct forces exploration of both root branches (not just pure exploitation
    # of whichever looked best on the very first simulation) so the test genuinely
    # exercises a head-to-head comparison of both discounted-return computations.
    engine = MctsEngine(num_simulations=60, max_depth=3, discount=0.99, c_puct=200.0, seed=1)
    res = engine.plan("root", ["A", "B"], transition, legal_actions_fn=legal, eval_fn=eval_fn)

    assert res["status"] == "OK"
    assert res["best_action"] == "A"
    # True values are ~99.01 (A) and ~9.9 (B). The old bug (v *= discount, no reward
    # add) made every deep visit to A contribute approximately -0.99 instead of +99.01,
    # dragging A's average toward zero or negative as simulations accumulate.
    assert res["root_q"]["A"] > 90.0
    assert res["root_q"]["B"] < 15.0
    assert res["root_q"]["A"] > res["root_q"]["B"]


def test_mcts_backprop_folds_reward_at_every_ancestor_on_a_three_level_path():
    """A single deterministic 3-edge path (single candidate at every level) isolates the
    recursion G_t = r_t + gamma*G_{t+1} from any PUCT exploration dynamics: exact value
    is checked analytically at every depth.
    """
    rewards = {("root", "A"): 5.0, ("n1", "A"): 7.0, ("n2", "A"): -2.0}
    order = ["root", "n1", "n2", "n3"]

    def transition(state, action):
        idx = order.index(state)
        nxt = order[idx + 1]
        done = nxt == "n3"
        return nxt, rewards[(state, action)], done

    def legal(state):
        return ["A"] if state != "n3" else []

    def eval_fn(state, actions):
        return {a: 1.0 for a in actions}, 0.0

    gamma = 0.9
    num_sims = 6
    engine = MctsEngine(num_simulations=num_sims, max_depth=3, discount=gamma, seed=1)
    res = engine.plan("root", ["A"], transition, legal_actions_fn=legal, eval_fn=eval_fn)

    assert res["status"] == "OK"
    # This is a degree-1 tree (a single candidate at every level), so the sequence of
    # depths PUCT reaches is deterministic: simulation 1 reaches n1 (depth 1),
    # simulation 2 reaches n2 (depth 2), and simulation 3 onward always reaches the
    # terminal n3 (depth 3) since that is the only remaining descendant. root_q["A"] is
    # nodes[n1].q_value, i.e. the *average* of every G folded into n1 across all visits,
    # not a single-shot value, so the expected value is the average of the three distinct
    # per-simulation Gs (correct Bellman recursion G_t = r_t + gamma*G_{t+1} at each depth):
    g_at_n1_reaching_only_n1 = 5.0  # r(root->n1) + gamma*eval_fn(n1) = 5.0 + 0.9*0
    g_at_n1_reaching_n2 = 5.0 + gamma * 7.0  # + gamma*eval_fn(n2) term is 0
    g_at_n1_reaching_n3 = 5.0 + gamma * (7.0 + gamma * (-2.0))
    deep_sims = num_sims - 2  # simulations 3..6 all reach the terminal n3
    expected_avg = (g_at_n1_reaching_only_n1 + g_at_n1_reaching_n2 + deep_sims * g_at_n1_reaching_n3) / num_sims
    assert res["root_q"]["A"] == pytest.approx(expected_avg, abs=1e-9)
    # Regardless of the exact averaging, the fully-folded per-visit return must be a
    # strongly positive number close to 9.68, not close to 0 or negative: the old bug
    # (v *= discount with no reward add) would have made deep visits contribute
    # 0.9*(0.9*(-2.0)) = -1.62 each instead of +9.68, pulling this average toward zero.
    assert res["root_q"]["A"] > 8.0


# ==================================================================== X-M02


def test_mcts_fails_closed_on_non_finite_root_transition_reward():
    def transition(state, action):
        return state + 1, float("nan"), False

    def legal(state):
        return ["A"]

    engine = MctsEngine(num_simulations=5, max_depth=2)
    res = engine.plan(0, ["A"], transition, legal_actions_fn=legal, reward_fn=lambda p, a, c: 1.0)

    assert res["status"] == "NON_FINITE_TRANSITION_REWARD"
    assert res["best_action"] is None
    assert res["visit_distribution"] == {}


def test_mcts_fails_closed_on_non_finite_transition_reward_mid_search():
    def transition(state, action):
        if state == 0:
            return 1, 1.0, False
        return 2, float("inf"), True

    def legal(state):
        return ["A"] if state < 2 else []

    engine = MctsEngine(num_simulations=5, max_depth=3)
    res = engine.plan(0, ["A"], transition, legal_actions_fn=legal, reward_fn=lambda p, a, c: 1.0)

    assert res["status"] == "NON_FINITE_TRANSITION_REWARD"
    assert res["best_action"] is None


def test_mcts_fails_closed_on_non_finite_reward_fn_value():
    def transition(state, action):
        return state + 1, 1.0, False

    def legal(state):
        return ["A"]

    engine = MctsEngine(num_simulations=3, max_depth=2)
    res = engine.plan(0, ["A"], transition, legal_actions_fn=legal, reward_fn=lambda p, a, c: float("nan"))

    assert res["status"] == "NON_FINITE_TRANSITION_REWARD"
    assert res["best_action"] is None


def test_mcts_fails_closed_on_non_finite_reward_inside_dynamics_imagined_rollout():
    class _FakeDynamics:
        def __init__(self):
            self.calls = 0

        def step(self, z, action):
            self.calls += 1
            if self.calls == 1:
                return z + 1.0, 1.0, False
            return z + 1.0, float("nan"), False

    engine = MctsEngine(num_simulations=3, max_depth=4, dynamics_model=_FakeDynamics())
    root = np.array([0.0], dtype=np.float32)
    res = engine.plan(root, ["A"], transition_fn=None, legal_actions_fn=lambda s: ["A"])

    assert res["status"] == "NON_FINITE_TRANSITION_REWARD"
    assert res["best_action"] is None


def test_client_decide_abstains_when_mcts_transition_reward_is_non_finite(gz):
    def poisoned(s, a):
        return {"x": s["x"] + 1}, float("nan"), False

    res = gz.decide({"x": 0}, ["A", "B"], mode="mcts", transition_fn=poisoned)

    assert res["action"] == "ABSTAIN"
    assert res["status"] == "NON_FINITE_TRANSITION_REWARD_ABSTAIN"
    assert set(res["probs"].values()) == {0.0}
    assert not (isinstance(res.get("value"), float) and math.isnan(res["value"]))
