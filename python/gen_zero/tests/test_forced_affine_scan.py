"""Numerical evidence for docs/research/04-conditional-forcing-and-continuous-reread.md.

(a) impulse injection decays to the condition-free fixed point 0, fastest along
    small-|eigenvalue| directions; (b) hard forcing keeps the scan associative and
    its unique fixed point satisfies G s* = b; (c) soft forcing is exactly affine and
    fails closed outside its stability window; (d) infeasible conditions and
    non-finite input fail closed; (e) empty conditions reduce to the unforced scan;
    (f) per-chunk linearisation error grows with chunk length and is zero for
    affine g; (g) the projected step never forms a dense matrix and matches the
    dense elements; (h) the module imports no `re` and reads no labels.
"""
import ast
import inspect

import numpy as np
import pytest

from gen_zero.causal import forced_affine_scan as fas
from gen_zero.causal.forced_affine_scan import (
    ConditionSet, InfeasibleConditionError, UnstableForcingError,
    chunked_linearized_scan, fixed_point, hard_forced_elements, hard_forced_step,
    impulse_decay, linearize, soft_forced_elements,
)
from gen_zero.causal.parallel_rnn_lora import ParallelRNNLoRAAdapter, associative_scan, sequential_scan


def _adapter(seed=0, dim=16, rank=4):
    return ParallelRNNLoRAAdapter(dim=dim, rank=rank, rho_max=0.95, seed=seed)


def _cond(rng, dim=16, m=3):
    G = rng.standard_normal((m, dim))
    s_true = rng.standard_normal(dim)
    return ConditionSet.from_arrays(G, G @ s_true)


# (a) ---------------------------------------------------------------------
def test_impulse_injection_decays_to_zero_and_high_frequency_dies_first():
    ad = _adapter()
    A, B = ad.transition_matrix(), ad.input_matrix()
    rng = np.random.default_rng(1)
    norms = impulse_decay(A, B, rng.standard_normal(16), T=400)
    assert norms[0] > 0.0
    assert norms[-1] < 1e-6 * norms[0]            # anchoring to x_0 is gone
    # Spectral picture: eigen-directions of A decay as |lambda_i|^t.
    lam = np.abs(np.linalg.eigvals(A))
    assert lam.max() < 1.0
    assert lam.min() < 0.5 * lam.max()             # a fast (high-frequency) direction exists
    # Along the slowest direction the amplitude at t=50 is about lam_max^50; along
    # the fastest it is lam_min^50, orders of magnitude smaller.
    assert lam.min() ** 50 < 1e-3 * lam.max() ** 50


# (b) ---------------------------------------------------------------------
def test_hard_forcing_scan_is_exact_and_fixed_point_satisfies_conditions():
    ad = _adapter()
    A = ad.transition_matrix()
    rng = np.random.default_rng(2)
    cond = _cond(rng)
    T = 257
    u = rng.standard_normal((T, 16)) * 0.1
    A_f, u_f = hard_forced_elements(A, u, cond)
    h_par, h_seq = associative_scan(A_f, u_f), sequential_scan(A_f, u_f)
    assert np.allclose(h_par, h_seq, atol=1e-10)   # associativity preserved
    # every forced state satisfies the conditions, not just the limit
    assert np.max(np.abs(cond.G @ h_par.T - cond.b[:, None])) < 1e-9
    # contraction and unique fixed point
    assert np.linalg.svd(A_f[0], compute_uv=False)[0] <= np.linalg.svd(A, compute_uv=False)[0] + 1e-12
    s_star = fixed_point(A_f[0], u_f[-1])
    assert np.linalg.norm(cond.residual(s_star)) < 1e-9
    # the fixed point depends on the conditions: a different C gives a different s*
    other = _cond(np.random.default_rng(3))
    A_o, u_o = hard_forced_elements(A, u, other)
    assert np.linalg.norm(fixed_point(A_o[0], u_o[-1]) - s_star) > 1e-3


def test_hard_forced_fixed_point_reached_from_any_start():
    ad = _adapter(seed=5)
    A = ad.transition_matrix()
    rng = np.random.default_rng(6)
    cond = _cond(rng)
    u = np.tile(rng.standard_normal(16) * 0.1, (600, 1))  # held input
    A_f, u_f = hard_forced_elements(A, u, cond)
    s_star = fixed_point(A_f[0], u_f[0])
    for seed in range(3):
        s0 = np.random.default_rng(seed).standard_normal(16) * 10
        uu = u_f.copy(); uu[0] += A_f[0] @ s0
        h = associative_scan(A_f, uu)
        assert np.linalg.norm(h[-1] - s_star) < 1e-8


# (c) ---------------------------------------------------------------------
def test_soft_forcing_is_affine_exact_and_has_a_stability_window():
    ad = _adapter(seed=7)
    A = ad.transition_matrix()
    rng = np.random.default_rng(8)
    cond = _cond(rng)
    u = rng.standard_normal((64, 16)) * 0.1
    mu = np.linalg.eigvalsh(cond.G.T @ cond.G).max()
    eta = 0.3 / mu
    A_f, u_f = soft_forced_elements(A, u, cond, eta=eta)
    # exactly the penalty-gradient step, no approximation
    s = rng.standard_normal(16)
    manual = A @ s + u[3] - eta * cond.G.T @ (cond.G @ s - cond.b)
    assert np.allclose(A_f[3] @ s + u_f[3], manual, atol=1e-12)
    assert np.allclose(associative_scan(A_f, u_f), sequential_scan(A_f, u_f), atol=1e-10)
    # residual at the soft fixed point is small but NOT zero (penalty, not projection)
    s_star = fixed_point(A_f[0], u_f[0])
    r = np.linalg.norm(cond.residual(s_star))
    assert 0.0 < r
    # larger eta -> smaller residual, until the stability window closes
    A_g, u_g = soft_forced_elements(A, u, cond, eta=1.0 / mu)
    r_strong = np.linalg.norm(cond.residual(fixed_point(A_g[0], u_g[0])))
    assert r_strong < r
    assert r_strong > 0.3 * np.linalg.norm(cond.b)   # measured: inside its window the soft form removes < 60% of ||b||
    eta_bad = 1.5 / mu             # below the pure-damping (A = rho I) bound (1 + rho) / mu, yet unstable for this A
    raw = A - eta_bad * cond.G.T @ cond.G
    assert np.linalg.svd(raw, compute_uv=False)[0] >= 1.0   # "stronger" forcing really is unstable
    with pytest.raises(UnstableForcingError):
        soft_forced_elements(A, u, cond, eta=eta_bad)


# (d) ---------------------------------------------------------------------
def test_infeasible_conditions_fail_closed():
    G = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(InfeasibleConditionError):
        ConditionSet.from_arrays(G, np.array([1.0, 2.0]))
    ConditionSet.from_arrays(G, np.array([1.0, 1.0]))   # consistent duplicate is fine


def test_nonfinite_inputs_rejected():
    cond = ConditionSet.empty(4)
    with pytest.raises(ValueError):
        ConditionSet.from_arrays(np.array([[np.nan, 0.0]]), np.array([0.0]))
    with pytest.raises(ValueError):
        hard_forced_elements(np.eye(4) * np.inf, np.zeros((2, 4)), cond)
    with pytest.raises(UnstableForcingError):
        fixed_point(np.eye(4), np.zeros(4))


# (e) ---------------------------------------------------------------------
def test_empty_conditions_reduce_to_unforced_scan():
    ad = _adapter(seed=9)
    A = ad.transition_matrix()
    u = np.random.default_rng(10).standard_normal((32, 16))
    A_f, u_f = hard_forced_elements(A, u, ConditionSet.empty(16))
    assert np.allclose(A_f[0], A) and np.allclose(u_f, u)
    A_s, u_s = soft_forced_elements(A, u, ConditionSet.empty(16), eta=1.0)
    assert np.allclose(A_s[0], A) and np.allclose(u_s, u)


# (f) ---------------------------------------------------------------------
def test_chunked_linearization_error_grows_with_chunk_and_is_zero_for_affine():
    ad = _adapter(seed=11)
    A = ad.transition_matrix()
    rng = np.random.default_rng(12)
    T = 256
    u = rng.standard_normal((T, 16)) * 0.3
    s0 = rng.standard_normal(16)
    # affine g: chunking is irrelevant, all chunk sizes agree to round-off
    G = rng.standard_normal((3, 16)); b = rng.standard_normal(3)
    g_aff, j_aff = (lambda s: G @ s - b), (lambda s: G)
    ref = chunked_linearized_scan(A, u, g_aff, j_aff, s0, chunk=1)
    for c in (4, 32, 256):
        assert np.allclose(chunked_linearized_scan(A, u, g_aff, j_aff, s0, chunk=c), ref, atol=1e-9)
    # smooth single-sheet nonlinear g: deviation from the per-step reference grows with chunk
    W = rng.standard_normal((3, 16)) / 4; V = rng.standard_normal((3, 16)) / 4; c0 = np.array([1.0, 2.0, 0.5])
    g = lambda s: W @ s + 0.1 * (V @ s) ** 2 - c0
    jac = lambda s: W + 0.2 * (V @ s)[:, None] * V
    ref = chunked_linearized_scan(A, u, g, jac, s0, chunk=1)
    errs = [np.max(np.linalg.norm(chunked_linearized_scan(A, u, g, jac, s0, chunk=c) - ref, axis=1))
            for c in (2, 8, 64, 256)]
    assert errs[0] <= errs[1] <= errs[2] <= errs[3]
    assert errs[0] > 1e-3                                  # linearisation is NOT free for non-affine g
    # two-sheet g ((w.s)^2 = c, non-convex): the start state selects the sheet. Both
    # end states satisfy g2 = 0, yet they are different attractors. Forcing does NOT
    # remove this multiplicity; only the affine (single-sheet) case has a unique s*.
    W2 = rng.standard_normal((1, 16)) / 4
    g2 = lambda s: (W2 @ s) ** 2 - c0[:1]
    jac2 = lambda s: 2 * (W2 @ s)[:, None] * W2
    u_held = np.tile(u[0] * 0.1, (T, 1))
    s_plus = 4.0 * W2[0] / np.linalg.norm(W2[0]) ** 2
    h_p = chunked_linearized_scan(A, u_held, g2, jac2, s_plus, chunk=1)
    h_m = chunked_linearized_scan(A, u_held, g2, jac2, -s_plus, chunk=1)
    assert np.abs(g2(h_p[-1])).max() < 1e-8 and np.abs(g2(h_m[-1])).max() < 1e-8
    assert np.sign(W2 @ h_p[-1]) != np.sign(W2 @ h_m[-1])
    assert np.linalg.norm(h_p[-1] - h_m[-1]) > 1.0


# (g) ---------------------------------------------------------------------
def test_projected_step_matches_dense_elements_without_dense_matrix():
    ad = _adapter(seed=13)
    rng = np.random.default_rng(14)
    cond = _cond(rng)
    x = rng.standard_normal((20, 16)); s = np.zeros(16)
    A_f, u_f = hard_forced_elements(ad.transition_matrix(), x @ ad.input_matrix().T, cond)
    dense = sequential_scan(A_f, u_f)
    for t in range(20):
        s = hard_forced_step(ad.step, x[t], s, cond)
        assert np.allclose(s, dense[t], atol=1e-10)
    src = inspect.getsource(hard_forced_step)
    assert "projector()" not in src and "np.eye" not in src


# (h) ---------------------------------------------------------------------
def test_module_has_no_regex_and_no_label_access():
    tree = ast.parse(inspect.getsource(fas))
    imports = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    froms = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "re" not in imports and "re" not in froms
    src = inspect.getsource(fas).lower()
    for word in ("candidate", "label", "answer", "option"):
        assert f" {word}s" not in src.replace("candidates or labels", "")
