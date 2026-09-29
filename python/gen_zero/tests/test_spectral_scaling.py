"""Adaptive spectral scaling: N-invariance of the ridge/data energy ratio.

For z-scored features, ||Z^T Z||_F grows ~linearly in the effective sample count
N_eff (diag(Z^T Z) ~ N_eff by construction), while a raw ``lambda_reg`` is a fixed
constant. That means a fixed lambda gets diluted to ~1/N_eff of the data term as
N_eff grows -- across this project's 13 evaluation tasks N_eff spans ~144 to
~11,247 (~80x). ``adaptive_spectral_scaling`` restores a constant
``||Lambda||_F / ||Z^T M Z||_F`` energy ratio by scaling lambda with the data
Gram's own Frobenius energy (see the derivation in master_objective.py's module
docstring for why this drops the ``/ N_eff`` a naive reading of the literal
"lambda * ||Z^T Z||_F / (N * sqrt(d))" formula would suggest -- that division
cancels the very growth being compensated for and is a no-op on z-scored data).

The cleanest, exact way to test "the regularization's relative pull is unchanged
by a change in N" is literal row duplication: tiling Z/Y k times scales the data
Gram A and the RHS B by exactly k. If lambda_eff also scales by exactly k, the
whole normal-equations system scales by k and the solved W* is bit-for-bit
unchanged. That is what is tested below, with a negative control showing the
fixed (non-adaptive) lambda does *not* have this property -- the very dilution
bug this feature exists to fix.
"""

import numpy as np
import pytest

from gen_zero.manifold import MasterClosedFormSolver, MasterObjectiveError, MultiModelGraphLaplacian


def _regression_data(n=150, d=6, k=3, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))
    W = rng.normal(size=(d, k))
    Y = Z @ W + 0.5 + 0.1 * rng.normal(size=(n, k))
    return Z, Y


# --------------------------------------------------------------- N-invariance
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("k_dup", [2, 10])
def test_solution_exactly_invariant_under_row_duplication(fit_intercept, k_dup):
    """Duplicating every row k times must not move W*, lambda's relative pull, or
    the shrinkage toward W0 -- a stand-in for the real 144 -> 11,247 (~80x) spread."""
    Z, Y = _regression_data(seed=1)
    W0 = np.random.default_rng(2).normal(size=(6, 3))
    base = MasterClosedFormSolver()
    W_base = base.fit(Z, Y, W0=W0, lambda_reg=7.0, fit_intercept=fit_intercept,
                       adaptive_spectral_scaling=True)

    Zd, Yd = np.tile(Z, (k_dup, 1)), np.tile(Y, (k_dup, 1))
    dup = MasterClosedFormSolver()
    W_dup = dup.fit(Zd, Yd, W0=W0, lambda_reg=7.0, fit_intercept=fit_intercept,
                     adaptive_spectral_scaling=True)

    assert np.max(np.abs(W_base - W_dup)) < 1e-8
    assert np.max(np.abs(base.intercept_ - dup.intercept_)) < 1e-8
    # spectral_scale (rho = ||Add||_F / (N_eff*sqrt(d))) is defined to be exactly
    # N-invariant; it is the diagnostic, not the lambda actually used in the solve.
    assert abs(base.diagnostics_.spectral_scale - dup.diagnostics_.spectral_scale) < \
        1e-8 * max(1.0, base.diagnostics_.spectral_scale)
    # lambda_eff (the value actually added to the diagonal) scales by exactly
    # k_dup, tracking ||Add||_F's own growth 1:1 -- this is what keeps the ratio
    # ||Lambda||_F / ||Add||_F constant instead of diluting by 1/k_dup.
    assert abs(dup.diagnostics_.lambda_eff / base.diagnostics_.lambda_eff - k_dup) < 1e-6
    # The shrinkage toward the prior is identical, not diluted by the larger N.
    assert abs(np.linalg.norm(W_base - W0) - np.linalg.norm(W_dup - W0)) < 1e-8


def test_negative_control_without_adaptive_scaling_duplication_changes_solution():
    """Without adaptive scaling, duplicating rows 10x must visibly weaken the
    ridge's relative effect -- this is the exact dilution bug being fixed, and a
    test that can't fail (i.e. that would also pass on the broken code) is worthless."""
    Z, Y = _regression_data(seed=1)
    W0 = np.random.default_rng(2).normal(size=(6, 3))
    base = MasterClosedFormSolver().fit(Z, Y, W0=W0, lambda_reg=7.0,
                                         adaptive_spectral_scaling=False)
    Zd, Yd = np.tile(Z, (10, 1)), np.tile(Y, (10, 1))
    dup = MasterClosedFormSolver().fit(Zd, Yd, W0=W0, lambda_reg=7.0,
                                        adaptive_spectral_scaling=False)
    assert np.max(np.abs(base - dup)) > 1e-3
    # A fixed lambda is diluted by the 10x larger (unnormalised) data Gram: the
    # fit shrinks *less* toward W0 (moves closer to unregularised OLS), not the
    # same amount -- this is the exact dilution the adaptive scaling compensates.
    assert np.linalg.norm(dup - W0) > np.linalg.norm(base - W0)


def test_lambda_eff_and_spectral_scale_match_closed_form():
    """Pin the implementation to the exact formula, not just its invariance property."""
    Z, Y = _regression_data(n=40, d=5, k=2, seed=9)
    lam = 3.0
    s = MasterClosedFormSolver()
    s.fit(Z, Y, lambda_reg=lam, fit_intercept=False, adaptive_spectral_scaling=True)
    n, d = Z.shape
    expected_norm = np.linalg.norm(Z.T @ Z)
    expected_lambda_eff = lam * expected_norm / np.sqrt(d)
    expected_rho = expected_norm / (n * np.sqrt(d))
    assert abs(s.diagnostics_.lambda_eff - expected_lambda_eff) < 1e-6 * expected_lambda_eff
    assert abs(s.diagnostics_.spectral_scale - expected_rho) < 1e-6 * expected_rho


def test_adaptive_scaling_off_reproduces_literal_lambda():
    Z, Y = _regression_data(n=30, d=4, k=2, seed=4)
    s = MasterClosedFormSolver()
    s.fit(Z, Y, lambda_reg=5.0, adaptive_spectral_scaling=False)
    assert s.diagnostics_.lambda_eff == 5.0
    assert s.diagnostics_.adaptive_spectral_scaling is False


def test_eta_is_never_rescaled():
    """Za^T L Za is already a raw sum over O(N) graph edges (quadratic_operator),
    so it already tracks the data term's growth -- rescaling eta the same way
    lambda is rescaled would double-count that growth and over-regularise
    quadratically. eta_eff must equal eta regardless of the adaptive flag."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 5))
    y_lab = (X[:, 0] > 0).astype(int)
    L = MultiModelGraphLaplacian().build_sparse_laplacian(X, k_neighbors=5)
    Y = np.eye(2)[y_lab]
    for adaptive in (True, False):
        s = MasterClosedFormSolver()
        s.fit(X, Y, L=L, lambda_reg=1.0, eta=2.5, adaptive_spectral_scaling=adaptive)
        assert s.diagnostics_.eta_eff == 2.5


# --------------------------------------------------------------- fail closed
def test_fails_closed_on_zero_samples():
    Z0, Y0 = np.zeros((0, 5)), np.zeros((0, 3))
    with pytest.raises(MasterObjectiveError, match="N=0"):
        MasterClosedFormSolver().fit(Z0, Y0, lambda_reg=1.0, adaptive_spectral_scaling=True)


def test_fails_closed_on_zero_features():
    Z, Y = np.zeros((50, 6)), _regression_data(n=50, d=1, k=3, seed=2)[1]
    with pytest.raises(MasterObjectiveError, match="Z=0|degenerate"):
        MasterClosedFormSolver().fit(Z, Y, lambda_reg=1.0, adaptive_spectral_scaling=True)
    # Without adaptive scaling, all-zero Z is a legitimate (if useless) ridge fit:
    # the fix must not fail closed on inputs the base solver already accepts.
    W = MasterClosedFormSolver().fit(Z, Y, lambda_reg=1.0, adaptive_spectral_scaling=False)
    assert np.all(np.isfinite(W))


def test_fails_closed_on_zero_effective_weight():
    """An all-zero (but symmetric, PSD, otherwise-valid) full weight matrix gives
    N_eff=0; this must be rejected explicitly rather than silently producing a
    zero or non-finite lambda_eff."""
    Z, Y = _regression_data(n=20, d=4, k=2, seed=7)
    M = np.zeros((20, 20))
    with pytest.raises(MasterObjectiveError, match="non-positive"):
        MasterClosedFormSolver().fit(Z, Y, M=M, lambda_reg=1.0, adaptive_spectral_scaling=True)


def test_fails_closed_does_not_trigger_without_adaptive_scaling():
    """The new guards are scoped to the adaptive path; they must not fire (and
    change behaviour) for existing non-adaptive callers passing N=0 or Z=0."""
    Z0, Y0 = np.zeros((0, 5)), np.zeros((0, 3))
    with pytest.raises(MasterObjectiveError, match="positive definite"):
        # Still fails (0 samples, fit_intercept leaves the last row/col all zero),
        # but via the pre-existing generic PD check, not the new spectral guard.
        MasterClosedFormSolver().fit(Z0, Y0, lambda_reg=0.0, adaptive_spectral_scaling=False)
