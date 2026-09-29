"""Tests for the controlled continuous dynamical thinking engine.

"Correct" is always defined analytically by the constraints in the context c
(the unique candidate that satisfies them), never by the decoder itself.
"""

import ast
import dataclasses
import inspect

import numpy as np
import pytest

from gen_zero.causal import controlled_continuous_engine as cce
from gen_zero.causal.controlled_continuous_engine import (
    ContextTamperedError,
    ControlledContinuousEngine,
    EngineConfig,
    EngineStatus,
    FrozenProblemContext,
    NonFiniteInputError,
    context_residual,
)


def _unit(v):
    return v / np.linalg.norm(v)


def _row_space_projector(A):
    return A.T @ np.linalg.pinv(A.T)


def _make_problem(seed, d=24, n_prem=6, n_cf=3, n_ex=2):
    """Random constraint problem with one correct candidate and three wrong ones.

    Wrong candidates: (a) breaks a premise, (b) enters a counterfactual region,
    (c) makes two mutually exclusive propositions both true. (b) and (c) keep
    every premise exact, so only the contradiction terms can tell them apart.
    """
    rng = np.random.default_rng(seed)
    q_star = rng.normal(size=d)
    A = np.array([_unit(rng.normal(size=d)) for _ in range(n_prem)])
    U = np.array([_unit(rng.normal(size=d)) for _ in range(n_cf)])
    S, T = [], []
    for _ in range(n_ex):
        s, t = _unit(rng.normal(size=d)), _unit(rng.normal(size=d))
        if s @ q_star > 0 and t @ q_star > 0:
            s = -s
        S.append(s)
        T.append(t)
    S, T = np.array(S), np.array(T)
    b, m = A @ q_star, U @ q_star + 0.5
    ctx = FrozenProblemContext.from_arrays(A, b, U, m, S, T)

    G = np.vstack([A, U, S, T])
    null = np.eye(d) - _row_space_projector(G)
    correct = q_star + null @ rng.normal(size=d)
    p_null_a = np.eye(d) - _row_space_projector(A)

    wrong_premise = q_star + _unit(A.T @ rng.normal(size=n_prem))
    u_perp = _unit(p_null_a @ U[0])
    wrong_cf = q_star + ((m[0] + 0.5 - U[0] @ q_star) / (U[0] @ u_perp)) * u_perp
    v = _unit(p_null_a @ (S[0] + T[0]))
    assert S[0] @ v > 0 and T[0] @ v > 0
    alpha = max((0.5 - S[0] @ q_star) / (S[0] @ v), (0.5 - T[0] @ q_star) / (T[0] @ v), 0.0)
    wrong_ex = q_star + alpha * v

    cands = np.array([correct, wrong_premise, wrong_cf, wrong_ex])
    order = rng.permutation(4)
    cands = cands[order]
    correct_idx = int(np.where(order == 0)[0][0])
    wrong = {kind: int(np.where(order == k)[0][0])
             for k, kind in ((1, "premise"), (2, "counterfactual"), (3, "exclusion"))}
    return ctx, q_star, cands, correct_idx, wrong, rng


def _residual_norm(q, ctx):
    return float(np.sqrt(sum(float(v @ v) for v in context_residual(q, ctx).values())))


def _legacy_pull_step_for_contrast(zc, target):
    """The removed transition, kept here only to show its failure mode."""
    moved = np.tanh(zc + 0.3 * (target - zc))
    return moved / np.linalg.norm(moved)


# ---------------------------------------------------------------- 1. frozen context
def test_context_arrays_are_read_only_and_digest_pinned():
    ctx, *_ = _make_problem(0)
    for name in ("A", "b", "U", "m", "S", "T"):
        arr = getattr(ctx, name)
        with pytest.raises(ValueError):
            arr[...] = 0.0
        with pytest.raises(ValueError):
            arr.flags.writeable = True
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.b = np.zeros(3)
    ctx.verify()
    object.__setattr__(ctx, "b", np.asarray(ctx.b) + 1.0)  # deliberate tampering
    with pytest.raises(ContextTamperedError):
        ctx.verify()
    with pytest.raises(ContextTamperedError):
        ControlledContinuousEngine().think(np.zeros(ctx.dim), ctx)


def test_context_without_premises_is_rejected():
    with pytest.raises(ValueError):
        FrozenProblemContext.from_arrays(np.zeros((0, 4)), np.zeros(0))


def test_dynamics_api_takes_no_candidates():
    think = list(inspect.signature(ControlledContinuousEngine.think).parameters)
    step = list(inspect.signature(ControlledContinuousEngine.step).parameters)
    assert think == ["self", "q0", "ctx", "p0"]
    assert step == ["self", "q", "p", "ctx", "dt", "gamma", "branch"]


def test_module_has_no_tanh_pull_call():
    tree = ast.parse(inspect.getsource(cce))
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "tanh" not in called


def test_every_step_reads_the_context():
    ctx1, q_star, *_ = _make_problem(1)
    ctx2 = FrozenProblemContext.from_arrays(ctx1.A, ctx1.b + 0.25, ctx1.U, ctx1.m, ctx1.S, ctx1.T)
    eng = ControlledContinuousEngine()
    rng = np.random.default_rng(7)
    q, p = rng.normal(size=ctx1.dim), rng.normal(size=ctx1.dim)
    q1, p1 = eng.step(q, p, ctx1, 0.05, 0.1)
    q2, p2 = eng.step(q, p, ctx2, 0.05, 0.1)
    assert np.linalg.norm(p1 - p2) > 1e-3
    assert np.linalg.norm(q1 - q2) > 1e-5


# ---------------------------------------------------------------- 2. context retention
def _two_contexts():
    """Same candidates, contradicting premises: c1 is satisfied only by e0, c2 only by e1."""
    rng = np.random.default_rng(11)
    d = 16
    A = np.array([_unit(rng.normal(size=d)) for _ in range(4)])
    e0, e1 = rng.normal(size=d), rng.normal(size=d)
    ctx1 = FrozenProblemContext.from_arrays(A, A @ e0)
    ctx2 = FrozenProblemContext.from_arrays(A, A @ e1)
    return ctx1, ctx2, np.array([e0, e1]), rng.normal(size=d)


def test_changing_context_changes_trajectory_attractor_and_answer():
    ctx1, ctx2, cands, q0 = _two_contexts()
    eng = ControlledContinuousEngine(record_states=True)
    t1, d1 = eng.solve(q0, ctx1, cands)
    t2, d2 = eng.solve(q0, ctx2, cands)
    assert t1.status is EngineStatus.CONVERGED and t2.status is EngineStatus.CONVERGED
    s1, s2 = t1.episodes[0].states, t2.episodes[0].states
    assert np.linalg.norm(s1[1] - s2[1]) > 1e-6          # diverge from the first step
    assert np.linalg.norm(t1.q - t2.q) > 1e-1             # different attractors
    assert (d1.choice, d2.choice) == (0, 1)               # the answer flips with c
    # Each attractor reconstructs its own premises.
    assert np.linalg.norm(ctx1.A @ t1.q - ctx1.b) < 1e-6
    assert np.linalg.norm(ctx2.A @ t2.q - ctx2.b) < 1e-6


def test_same_context_is_deterministic():
    ctx1, _, cands, q0 = _two_contexts()
    eng = ControlledContinuousEngine()
    a, b = eng.think(q0, ctx1), eng.think(q0, ctx1)
    assert np.array_equal(a.q, b.q) and a.episodes[0].steps == b.episodes[0].steps


# ---------------------------------------------------------------- 3. wrong-option robustness
@pytest.mark.parametrize("kind", ["premise", "counterfactual", "exclusion"])
@pytest.mark.parametrize("seed", range(8))
def test_start_on_wrong_option_is_pulled_back(seed, kind):
    ctx, q_star, cands, correct, wrong, _ = _make_problem(seed)
    w = wrong[kind]
    q0 = cands[w].copy()                                   # worst case: start exactly on it
    assert _residual_norm(q0, ctx) > 0.1                   # the start really is wrong
    trace, dec = ControlledContinuousEngine().solve(q0, ctx, cands)

    assert trace.status is EngineStatus.CONVERGED
    assert trace.residual_norm < 1e-6
    assert dec.choice == correct and not dec.abstained
    assert not dec.admissible[w]
    # The candidate-free state itself left the wrong option: the part of z_K the
    # premises determine equals the correct answer, and no contradiction is left.
    P = _row_space_projector(ctx.A)
    assert np.linalg.norm(P @ (trace.q - q_star)) < 1e-5
    assert cce._contradiction(trace.q, ctx) < 1e-12


def test_legacy_pull_hypnotises_itself_and_engine_does_not():
    ctx, q_star, cands, correct, wrong, rng = _make_problem(3)
    w = wrong["premise"]
    unit = cands / np.linalg.norm(cands, axis=1, keepdims=True)
    z = _unit(rng.normal(size=ctx.dim))
    cos = [float(z @ unit[w])]
    for _ in range(10):
        z = _legacy_pull_step_for_contrast(z, unit[w])
        cos.append(float(z @ unit[w]))
    # Old transition: the score of the (wrong) target rises at every step.
    assert all(b > a for a, b in zip(cos, cos[1:])) and cos[-1] > 0.99

    # New engine from that same "confident" state still ends on the correct answer.
    trace, dec = ControlledContinuousEngine().solve(z * np.linalg.norm(cands[w]), ctx, cands)
    assert trace.status is EngineStatus.CONVERGED and dec.choice == correct


# ---------------------------------------------------------------- 4. reflection / backtracking
def _false_attractor_problem():
    """x + y = 2, not (x > 0 and y > 0), x <= 1. Only the branch x <= 0, y >= 2 is valid.

    Near (1.04, 0.04) the free potential has a true local minimum (checked in the
    test by its Hessian) that still breaks the premise: a false attractor. The
    start sits in its basin with zero momentum.
    """
    d = 6
    ex, ey = np.eye(d)[0], np.eye(d)[1]
    ctx = FrozenProblemContext.from_arrays(
        A=[(ex + ey) / np.sqrt(2)], b=[2 / np.sqrt(2)], U=[ex], m=[1.0], S=[ex], T=[ey])
    q0 = np.array([1.05, 0.04, 0.3, -0.2, 0.1, 0.4])
    return ctx, q0


def _hessian_2d(pot, q, h=1e-5):
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            def f(a, b):
                x = q.copy()
                x[i] += a
                x[j] += b
                return pot.energy_grad(x)[0]
            H[i, j] = (f(h, h) - f(h, -h) - f(-h, h) + f(-h, -h)) / (4 * h * h)
    return H


def test_false_attractor_triggers_backtracking_to_valid_branch():
    ctx, q0 = _false_attractor_problem()
    eng = ControlledContinuousEngine(record_states=True)
    trace = eng.think(q0, ctx)
    first = trace.episodes[0]
    q_fail = first.states[-1]
    free = cce._Potential(ctx, eng.config, None)
    # Episode 1 stopped at a stationary point that is a strict local minimum of V ...
    assert first.exit_reason in ("settled", "stalled")
    assert np.linalg.norm(free.energy_grad(q_fail)[1]) < 1e-5
    assert np.all(np.linalg.eigvalsh(_hessian_2d(free, q_fail)) > 1.0)
    # ... yet it breaks the premise: the reflector must not accept it.
    assert first.residual_norm > 0.5
    assert trace.backtracks >= 1
    assert trace.status is EngineStatus.CONVERGED and trace.residual_norm < 1e-6
    x, y = trace.q[:2]
    assert x <= 1e-6 and y >= 2 - 1e-6
    assert np.allclose(trace.q[2:], q0[2:])               # unconstrained axes untouched


def test_contradictory_context_is_unconverged_and_abstains():
    d = 5
    u = np.eye(d)[0]
    ctx = FrozenProblemContext.from_arrays(A=[u], b=[1.0], U=[u], m=[0.0])  # u.q = 1 and u.q <= 0
    cand = np.zeros((2, d))
    cand[0, 0] = 1.0
    trace, dec = ControlledContinuousEngine().solve(np.ones(d), ctx, cand)
    assert trace.status is EngineStatus.UNCONVERGED
    assert trace.residual_norm > 1e-2
    assert dec.abstained and dec.choice is None


def test_no_consistent_candidate_abstains():
    ctx, _, cands, correct, *_ = _make_problem(4)
    only_wrong = np.delete(cands, correct, axis=0)
    trace, dec = ControlledContinuousEngine().solve(np.zeros(ctx.dim), ctx, only_wrong)
    assert trace.status is EngineStatus.CONVERGED
    assert dec.abstained and dec.choice is None
    assert "no candidate" in dec.reason


def test_duplicate_correct_candidates_are_ambiguous():
    ctx, _, cands, correct, *_ = _make_problem(5)
    dup = np.vstack([cands, cands[correct]])
    _, dec = ControlledContinuousEngine().solve(np.zeros(ctx.dim), ctx, dup)
    assert dec.abstained and "ambiguous" in dec.reason


# ---------------------------------------------------------------- 5. energy dissipation
@pytest.mark.parametrize("seed", range(6))
def test_hamiltonian_is_monotone_non_increasing(seed):
    ctx, _, cands, _, wrong, rng = _make_problem(seed)
    trace = ControlledContinuousEngine().think(cands[wrong["exclusion"]] + rng.normal(size=ctx.dim), ctx)
    for ep in trace.episodes:
        H = np.array(ep.energy)
        assert H.size > 10
        assert np.all(np.diff(H) <= 0.0), f"max increase {np.diff(H).max()}"
        # The physics does the work, not the guard: at the default step no step was rejected.
        assert ep.rejected_steps == 0
    assert trace.episodes[-1].energy[-1] < 1e-10


def test_false_attractor_episodes_are_each_monotone():
    ctx, q0 = _false_attractor_problem()
    trace = ControlledContinuousEngine().think(q0, ctx)
    for ep in trace.episodes:
        assert np.all(np.diff(ep.energy) <= 0.0)


def test_energy_guard_is_needed_and_works_at_large_step():
    """Premise-only problem: the Lipschitz bound is tight, so dt_frac > 1 is really too big."""
    ctx, _, cands, q0 = _two_contexts()
    for dt_frac in (1.5, 2.5):
        raw = ControlledContinuousEngine(EngineConfig(dt_frac=dt_frac, energy_guard=False,
                                                      max_steps=400)).think(q0, ctx)
        assert np.diff(raw.episodes[0].energy).max() > 1e-3   # raw leapfrog gains energy
        guarded = ControlledContinuousEngine(EngineConfig(dt_frac=dt_frac)).think(q0, ctx)
        ep = guarded.episodes[0]
        assert ep.rejected_steps > 0
        assert np.all(np.diff(ep.energy) <= 0.0)
        assert guarded.status is EngineStatus.CONVERGED


# ---------------------------------------------------------------- 6. fail-closed boundaries
@pytest.mark.parametrize("field", ["A", "b", "U", "m", "S", "T"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_context_is_rejected(field, bad):
    ctx, *_ = _make_problem(0)
    arrays = {k: np.array(getattr(ctx, k)) for k in ("A", "b", "U", "m", "S", "T")}
    arrays[field].flat[0] = bad
    with pytest.raises(NonFiniteInputError):
        FrozenProblemContext.from_arrays(**arrays)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_state_or_candidates_are_rejected(bad):
    ctx, _, cands, *_ = _make_problem(0)
    eng = ControlledContinuousEngine()
    q0 = np.zeros(ctx.dim)
    q0[3] = bad
    with pytest.raises(NonFiniteInputError):
        eng.think(q0, ctx)
    with pytest.raises(NonFiniteInputError):
        eng.think(np.zeros(ctx.dim), ctx, p0=q0)
    trace = eng.think(np.zeros(ctx.dim), ctx)
    bad_c = cands.copy()
    bad_c[1, 2] = bad
    with pytest.raises(NonFiniteInputError):
        eng.decode(trace, ctx, bad_c)


def test_overflow_during_dynamics_fails_closed():
    d = 4
    ctx = FrozenProblemContext.from_arrays(A=np.full((1, d), 1e200), b=[1e200])
    trace, dec = ControlledContinuousEngine().solve(np.ones(d), ctx, np.ones((2, d)))
    assert trace.status is EngineStatus.NONFINITE
    assert trace.q is None
    assert dec.abstained and dec.choice is None


def test_decode_refuses_trace_from_other_context():
    ctx1, ctx2, cands, q0 = _two_contexts()
    eng = ControlledContinuousEngine()
    with pytest.raises(ContextTamperedError):
        eng.decode(eng.think(q0, ctx1), ctx2, cands)


@pytest.mark.parametrize("kw", [{"dt_frac": float("nan")}, {"kappa": float("inf")},
                                {"eps_tol": 0.0}, {"omega": -1.0}, {"max_steps": 0}])
def test_invalid_config_is_rejected(kw):
    with pytest.raises(ValueError):
        EngineConfig(**kw)


# ---------------------------------------------------------------- 7. no language / format assumptions
def test_rotation_and_permutation_equivariance():
    ctx, _, cands, correct, *_ , rng = _make_problem(6)
    q0 = rng.normal(size=ctx.dim)
    Q, _ = np.linalg.qr(rng.normal(size=(ctx.dim, ctx.dim)))
    rot = FrozenProblemContext.from_arrays(ctx.A @ Q.T, ctx.b, ctx.U @ Q.T, ctx.m, ctx.S @ Q.T, ctx.T @ Q.T)
    perm = rng.permutation(len(cands))
    eng = ControlledContinuousEngine()
    t0, d0 = eng.solve(q0, ctx, cands)
    t1, d1 = eng.solve(Q @ q0, rot, (cands @ Q.T)[perm])
    assert t0.backtracks == t1.backtracks == 0 or t0.backtracks == t1.backtracks
    assert np.linalg.norm(Q @ t0.q - t1.q) < 1e-6
    assert d0.choice == correct and perm[d1.choice] == correct


@pytest.mark.parametrize("d,n_prem,n_cf,n_ex", [(8, 2, 1, 1), (24, 6, 3, 2), (64, 20, 6, 4)])
def test_zero_shot_random_problem_families(d, n_prem, n_cf, n_ex):
    """No parameter is fitted. Each problem is new; the start is random or on a wrong option."""
    hits, total = 0, 0
    for seed in range(100, 110):
        ctx, _, cands, correct, wrong, rng = _make_problem(seed, d, n_prem, n_cf, n_ex)
        for q0 in (rng.normal(size=d), cands[wrong["exclusion"]]):
            trace, dec = ControlledContinuousEngine().solve(q0, ctx, cands)
            total += 1
            hits += trace.status is EngineStatus.CONVERGED and dec.choice == correct
    assert hits == total, f"{hits}/{total}"
