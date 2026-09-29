"""Real-engine regression checks. The 2ms performance target is audited separately."""
import copy
import random

import numpy as np
import pytest

from gen_zero.client import GenZero
from gen_zero.gate.constraint_compiler import ConstraintLinearProjectionCompiler
from gen_zero.multiagent.league_arena import LeagueArena
from gen_zero.planner.engines.mpc_cem_engine import MpcCemEngine
from gen_zero.run_league_benchmark import elo_convergence


def compiler(rules=(), budget=100):
    c = ConstraintLinearProjectionCompiler(latent_dim=8, hard_timeout_ms=budget)
    c.compile_rules(rules)
    return c


def test_bidirectional_client_reaches_goal():
    client = GenZero()
    result = client.plan_bidirectional(0, 5, lambda s: [(s+1, 'right', 1)],
                                      lambda s: [(s-1, 'left', 1)])
    assert result['success']
    assert result['path'] == ['right'] * 5


def test_cem_state_dispatch_optimizes_real_transition_rewards():
    engine = MpcCemEngine(seed=1)
    result = engine.plan(state=0, candidate_actions_or_bounds=['up', 'down'],
                         transition_fn=lambda s, a: (s + (1 if a == 'up' else -1),
                                                      1 if a == 'up' else -1, False))
    assert result['best_action'] == 'up'


def test_cem_discrete_non_finite_reward_reports_contract_status():
    """Reviewer fix: plan_discrete's non-finite-reward abstain must use the
    NON_FINITE_REWARD status name that client.py's EXPERT_ABSTAIN_STATUSES
    (and the fail-closed fusion check) actually recognize."""
    engine = MpcCemEngine(seed=1)
    result = engine.plan_discrete(
        initial_state=0,
        candidate_actions=['up', 'down'],
        transition_fn=lambda s, a: (s + 1, float('nan'), False),
    )
    assert result['status'] == 'NON_FINITE_REWARD'
    assert result['best_action'] is None


def test_cem_non_finite_reward_forces_global_abstain_through_client():
    """Reviewer fix: client.py's _execute_expert_distribution must copy CEM's
    abstain status into expert meta so the top-level fusion step's
    EXPERT_ABSTAIN_STATUSES check rejects the whole decision, instead of
    another expert silently outvoting a CEM NaN-reward abstain."""
    client = GenZero()

    def poisoned_transition(s, a):
        return s + 1, float('nan'), False

    res = client.decide(
        state=0,
        candidates=['up', 'down'],
        mode='mpc_cem',
        transition_fn=poisoned_transition,
    )
    cem_meta = res['expert_outputs']['mpc_cem']['meta']
    assert cem_meta['status'] == 'NON_FINITE_REWARD'
    assert res['action'] == 'ABSTAIN'
    assert res['status'] == 'NON_FINITE_REWARD_ABSTAIN'
    assert set(res['probs'].values()) == {0.0}
    # _sanitize_for_json must have scrubbed the NaN expected_return before it
    # reached the caller as a literal float("nan") value.
    import math
    assert not (isinstance(res.get('value'), float) and math.isnan(res['value']))


def test_small_utility_difference_preserved_and_rules_enforced():
    c = compiler(['FORBID DANGER IF x >= 1'])
    for values in [(0.10001, 0.10002), (-0.10002, -0.10001)]:
        result = c.solve_safest_action({'DANGER': 10, 'A': values[0], 'B': values[1]}, current_metrics={'x': 1})
        assert result.is_safe and result.selected_action == 'B'
        assert result.solver_status == 'CP_SAT_OPTIMAL' and not result.fallback_used


def test_deadline_rejects_instead_of_silent_argmax():
    result = compiler(budget=1e-9).solve_safest_action({'A': 1, 'B': 2})
    assert result.timed_out and result.fallback_used and not result.is_safe


def test_infeasible_and_unparseable_conditions_do_not_release_fallback():
    for rules in [('REQUIRE STOP WHEN x > 0',), ('ALLOW A ONLY IF unknown(x)',)]:
        result = compiler(rules).solve_safest_action({'A': 1}, current_metrics={'x': 1})
        assert not result.is_safe and result.fallback_used


def test_latent_compound_resolves_every_and_term_not_just_the_first():
    """C07: project_latent_propositions / evaluate_condition must consume every
    sub_condition of an AND, not silently answer from the primary term alone.
    x is coordinate 0, y is coordinate 1 (both referenced by the rule's
    sub_conditions, sorted alphabetically -- see compile_rules)."""
    c = compiler(['ALLOW A ONLY IF x > 0 and y > 0'])
    fp = c.schema_fingerprint

    # x=1.0, y=1.0: both terms true -> the AND is genuinely true -> A is authorized.
    # Before the C07 fix this incorrectly stayed forbidden because "y" was never
    # bound to a coordinate and the projection only ever looked at "x > 0".
    both_true = c.solve_safest_action({'A': 10, 'B': 1}, z_latent=np.ones(8), schema_fingerprint=fp)
    assert both_true.is_safe and both_true.selected_action == 'A'

    # x=1.0, y=-1.0: y > 0 is false, so the AND is false -> ALLOW_ONLY_IF denies A.
    z_y_false = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    y_false = c.solve_safest_action({'A': 10, 'B': 1}, z_latent=z_y_false, schema_fingerprint=fp)
    assert not (y_false.is_safe and y_false.selected_action == 'A')
    assert any('ALLOW_VIOLATION' in rule for rule in y_false.applied_constraints)
    assert y_false.is_safe and y_false.selected_action == 'B'


def test_latent_missing_and_term_dimension_fails_closed_not_zero():
    """X-C02: a z_latent too short to reach y's coordinate must abstain
    (Tristate.UNKNOWN / fail-closed), never silently treat the missing
    dimension as 0.0 (which would make "y > 0" false and "y <= 0" true)."""
    c = compiler(['ALLOW A ONLY IF x > 0 and y > 0'])
    fp = c.schema_fingerprint
    # y is bound to coordinate 1, but this vector only has coordinate 0.
    short_z = np.array([1.0])
    result = c.solve_safest_action({'A': 10, 'B': 1}, z_latent=short_z, schema_fingerprint=fp)
    # y is unresolved -> ALLOW_ONLY_IF's condition is UNKNOWN -> fail-closed deny.
    assert not (result.is_safe and result.selected_action == 'A')


@pytest.mark.parametrize('utilities', [{'A': float('nan')}, {'A': float('inf')}, {'A': 1, ' a ': 2}])
def test_invalid_utilities_are_errors(utilities):
    with pytest.raises(ValueError):
        compiler().solve_safest_action(utilities)


def test_league_evaluation_preserves_training_state():
    random.seed(42)
    arena = LeagueArena()
    arena.evolve_league_generation(2)
    before = copy.deepcopy((arena.archive, arena.main_agent, arena.head_to_head))
    assert arena.evaluate_historical_robustness()['tested_historical_snapshots'] == 1
    assert before == (arena.archive, arena.main_agent, arena.head_to_head)


def test_convergence_rejects_drift_and_missing_evidence():
    assert not elo_convergence([])['passed']
    assert not elo_convergence([1000 + i * 5 for i in range(40)])['passed']
    assert elo_convergence([1000 + (-1)**i for i in range(40)])['passed']
