"""Tests for the master closed-form objective and the sparse graph Laplacian."""

import json

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.linear_model import Ridge

from gen_zero import MasterClosedFormSolver as ExportedSolver
from gen_zero.cli import main as cli_main
from gen_zero.manifold import (
    MasterClosedFormSolver,
    MasterObjectiveError,
    MultiModelGraphLaplacian,
)


def _regression_data(n=200, d=8, k=3, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))
    W = rng.normal(size=(d, k))
    Y = Z @ W + 0.5 + 0.1 * rng.normal(size=(n, k))
    return Z, Y


def _two_moons_like(n_per=60, seed=0):
    """Two well separated 2-D clusters lifted into 6-D with small noise."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, np.pi, size=n_per)
    a = np.c_[np.cos(t), np.sin(t)]
    b = np.c_[1 - np.cos(t), 0.5 - np.sin(t)] + np.array([0.0, -1.5])
    X2 = np.vstack([a, b])
    lift = rng.normal(size=(2, 6))
    X = X2 @ lift + 0.02 * rng.normal(size=(2 * n_per, 6))
    y = np.r_[np.zeros(n_per, int), np.ones(n_per, int)]
    return X, y


# ----------------------------------------------------------------- ridge limit
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("lam", [1e-3, 1.0, 100.0])
def test_reduces_to_sklearn_ridge(fit_intercept, lam):
    Z, Y = _regression_data()
    solver = MasterClosedFormSolver()
    W = solver.fit(Z, Y, W0=None, M=None, L=None, lambda_reg=lam, eta=0.0,
                   fit_intercept=fit_intercept, adaptive_spectral_scaling=False)
    ref = Ridge(alpha=lam, fit_intercept=fit_intercept, solver="cholesky").fit(Z, Y)
    assert np.max(np.abs(W - ref.coef_.T)) < 1e-9
    assert np.max(np.abs(solver.intercept_ - np.atleast_1d(ref.intercept_))) < 1e-9
    assert np.max(np.abs(solver.predict(Z) - ref.predict(Z))) < 1e-9


def test_explicit_identity_weights_and_zero_prior_match_defaults():
    Z, Y = _regression_data()
    a = MasterClosedFormSolver().fit(Z, Y, lambda_reg=3.0, adaptive_spectral_scaling=False)
    b = MasterClosedFormSolver().fit(Z, Y, W0=np.zeros((8, 3)), M=np.eye(200),
                                     L=sp.csr_matrix((200, 200)), lambda_reg=3.0, eta=0.0,
                                     adaptive_spectral_scaling=False)
    c = MasterClosedFormSolver().fit(Z, Y, M=np.ones(200), lambda_reg=3.0, adaptive_spectral_scaling=False)
    assert np.max(np.abs(a - b)) < 1e-9
    assert np.max(np.abs(a - c)) < 1e-9


def test_diagonal_weights_match_sklearn_sample_weight():
    Z, Y = _regression_data(seed=3)
    m = np.random.default_rng(1).uniform(0.1, 3.0, size=len(Z))
    solver = MasterClosedFormSolver()
    W = solver.fit(Z, Y, M=m, lambda_reg=2.0, adaptive_spectral_scaling=False)
    ref = Ridge(alpha=2.0, solver="cholesky").fit(Z, Y, sample_weight=m)
    assert np.max(np.abs(W - ref.coef_.T)) < 1e-9
    assert np.max(np.abs(solver.intercept_ - ref.intercept_)) < 1e-9
    W_sparse = MasterClosedFormSolver().fit(Z, Y, M=sp.diags(m), lambda_reg=2.0, adaptive_spectral_scaling=False)
    assert np.max(np.abs(W - W_sparse)) < 1e-12


# ----------------------------------------------------------- prior shrinkage
def test_prior_is_residual_correction():
    """fit(Z, Y, W0) - W0 == fit(Z, Y - Z W0, W0=0): the solver learns a correction to the prior."""
    Z, Y = _regression_data(seed=5)
    W0 = np.random.default_rng(9).normal(size=(8, 3))
    for fit_intercept in (True, False):
        s1, s2 = MasterClosedFormSolver(), MasterClosedFormSolver()
        W_prior = s1.fit(Z, Y, W0=W0, lambda_reg=50.0, fit_intercept=fit_intercept,
                         adaptive_spectral_scaling=False)
        delta = s2.fit(Z, Y - Z @ W0, lambda_reg=50.0, fit_intercept=fit_intercept,
                       adaptive_spectral_scaling=False)
        assert np.max(np.abs((W_prior - W0) - delta)) < 1e-9
        assert np.max(np.abs(s1.intercept_ - s2.intercept_)) < 1e-9


def test_prior_shrinkage_limits():
    Z, Y = _regression_data(seed=6)
    W0 = np.random.default_rng(2).normal(size=(8, 3))
    ols = MasterClosedFormSolver().fit(Z, Y, lambda_reg=1e-8, adaptive_spectral_scaling=False)
    dists = [np.linalg.norm(MasterClosedFormSolver().fit(
                 Z, Y, W0=W0, lambda_reg=lam, adaptive_spectral_scaling=False) - W0)
             for lam in (1e-2, 1.0, 1e2, 1e4, 1e6)]
    assert all(a > b for a, b in zip(dists, dists[1:]))
    # ||W* - W0|| = O(1/lambda) as lambda -> inf: a 100x larger lambda gives ~100x closer.
    assert 90.0 < dists[-2] / dists[-1] < 110.0
    assert dists[-1] < 1e-3 * dists[0]
    near_ols = MasterClosedFormSolver().fit(Z, Y, W0=W0, lambda_reg=1e-8, adaptive_spectral_scaling=False)
    assert np.max(np.abs(near_ols - ols)) < 1e-6


# ------------------------------------------------------------ graph Laplacian
def test_laplacian_is_symmetric_psd_and_sparse():
    X, _ = _two_moons_like()
    rng = np.random.default_rng(0)
    other_model = X @ rng.normal(size=(6, 10)) + 0.01 * rng.normal(size=(len(X), 10))
    g = MultiModelGraphLaplacian()
    L = g.build_sparse_laplacian([X, other_model], k_neighbors=5, metric="cosine", weights=[2.0, 1.0])
    assert sp.issparse(L)
    assert L.nnz < 0.2 * L.shape[0] ** 2
    assert np.allclose(g.model_weights_, [2 / 3, 1 / 3])
    assert abs(L - L.T).max() < 1e-12
    assert np.max(np.abs(L @ np.ones(L.shape[0]))) < 1e-10
    for _ in range(50):
        x = rng.normal(size=L.shape[0])
        assert x @ (L @ x) >= -1e-10
    assert np.linalg.eigvalsh(L.toarray())[0] > -1e-10


def test_compact_operator_matches_dense_and_pairwise():
    X, _ = _two_moons_like()
    L = MultiModelGraphLaplacian().build_sparse_laplacian(X, k_neighbors=6, metric="euclidean")
    Z = np.random.default_rng(1).normal(size=(len(X), 5))
    dense = Z.T @ L.toarray() @ Z
    sparse_op = MultiModelGraphLaplacian.quadratic_operator(Z, L)
    pairwise = MultiModelGraphLaplacian.pairwise_quadratic_operator(Z, L)
    assert np.max(np.abs(sparse_op - dense)) < 1e-10
    assert np.max(np.abs(pairwise - dense)) < 1e-10


@pytest.mark.parametrize("bad, msg", [
    (sp.csr_matrix(np.array([[1.0, -1.0], [0.0, 0.0]])), "symmetric"),
    (sp.csr_matrix(np.array([[-1.0, 1.0], [1.0, -1.0]])), "positive off-diagonal"),
    (sp.csr_matrix(np.array([[2.0, -1.0], [-1.0, 2.0]])), "sum to zero"),
    (sp.csr_matrix(np.array([[np.nan, 0.0], [0.0, 0.0]])), "NaN"),
])
def test_verify_laplacian_rejects_non_laplacians(bad, msg):
    with pytest.raises(ValueError, match=msg):
        MultiModelGraphLaplacian.verify_laplacian(bad)


def test_graph_build_fails_closed_on_degenerate_input():
    g = MultiModelGraphLaplacian()
    X = np.random.default_rng(0).normal(size=(10, 3))
    with pytest.raises(ValueError, match="zero-norm"):
        g.build_sparse_laplacian(np.vstack([X, np.zeros((1, 3))]), k_neighbors=3)
    with pytest.raises(ValueError, match="duplicates"):
        g.build_sparse_laplacian(np.vstack([X, np.repeat(X[:1], 4, axis=0)]), k_neighbors=3)
    with pytest.raises(ValueError, match="k_neighbors"):
        g.build_sparse_laplacian(X, k_neighbors=10)
    with pytest.raises(ValueError, match="rows"):
        g.build_sparse_laplacian([X, X[:5]])
    with pytest.raises(ValueError, match="weights"):
        g.build_sparse_laplacian([X, X], weights=[0.0, 0.0])


def test_graph_term_smooths_and_propagates_labels():
    """Semi-supervised: 2 labels per cluster, the rest unlabeled (M = 0), graph carries the labels."""
    X, y = _two_moons_like(seed=4)
    n = len(X)
    L = MultiModelGraphLaplacian().build_sparse_laplacian(X, k_neighbors=6, metric="euclidean")
    labeled = np.r_[0, 1, 60, 61]
    m = np.zeros(n)
    m[labeled] = 1.0
    Y = np.zeros((n, 2))
    Y[labeled, y[labeled]] = 1.0

    energies, accs = [], []
    for eta in (0.0, 0.1, 1.0, 10.0):
        solver = MasterClosedFormSolver()
        solver.fit(X, Y, M=m, L=L, lambda_reg=1.0, eta=eta, adaptive_spectral_scaling=False)
        F = solver.predict(X)
        energies.append(MultiModelGraphLaplacian.dirichlet_energy(F, L))
        accs.append(np.mean(np.argmax(F, axis=1) == y))
        assert solver.diagnostics_.graph_term >= 0.0
        assert abs(solver.diagnostics_.graph_term - eta * energies[-1]) < 1e-8 * max(1, energies[-1])
        again = MasterClosedFormSolver().fit(X, Y, M=m, L=L, lambda_reg=1.0, eta=eta,
                                             adaptive_spectral_scaling=False)
        assert np.array_equal(again, solver.coef_)  # deterministic
    assert all(a > b for a, b in zip(energies, energies[1:]))
    assert accs[-1] >= accs[0]
    assert accs[-1] > 0.9


def test_graph_term_matches_textbook_formula():
    """With a combinatorial L the intercept drops out: W* = (Z^T M Z + lam I + eta Z^T L Z)^-1 (...)."""
    X, y = _two_moons_like(seed=7)
    L = MultiModelGraphLaplacian().build_sparse_laplacian(X, k_neighbors=5)
    Y = np.eye(2)[y]
    m = np.random.default_rng(0).uniform(0.5, 1.5, size=len(X))
    lam, eta = 0.7, 2.5
    W = MasterClosedFormSolver().fit(X, Y, M=m, L=L, lambda_reg=lam, eta=eta, fit_intercept=False,
                                     adaptive_spectral_scaling=False)
    A = X.T @ (m[:, None] * X) + lam * np.eye(6) + eta * X.T @ L.toarray() @ X
    B = X.T @ (m[:, None] * Y)
    assert np.max(np.abs(A @ W - B)) < 1e-9


# --------------------------------------------------------------- fail closed
def test_fail_closed_on_bad_numerics():
    Z, Y = _regression_data()
    s = MasterClosedFormSolver()
    Zn = Z.copy()
    Zn[3, 2] = np.nan
    with pytest.raises(MasterObjectiveError, match="NaN"):
        s.fit(Zn, Y)
    Yi = Y.copy()
    Yi[0, 0] = np.inf
    with pytest.raises(MasterObjectiveError, match="NaN or Inf"):
        s.fit(Z, Yi)
    # Rank deficient with no ridge: singular system must raise, not return lstsq garbage.
    Zr = np.c_[Z, Z[:, :1]]
    with pytest.raises(MasterObjectiveError, match="positive definite|ill-conditioned"):
        s.fit(Zr, Y, lambda_reg=0.0)
    # Nearly collinear with a tiny ridge: ill-conditioned must raise.
    Zc = np.c_[Z, Z[:, :1] + 1e-9 * np.random.default_rng(0).normal(size=(len(Z), 1))]
    with pytest.raises(MasterObjectiveError, match="ill-conditioned|positive definite"):
        s.fit(Zc, Y, lambda_reg=1e-12)
    with pytest.raises(ValueError, match="requires a graph Laplacian"):
        s.fit(Z, Y, eta=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        s.fit(Z, Y, M=-np.ones(len(Z)))
    with pytest.raises(ValueError, match="at least one"):
        s.fit(Z, Y, M=np.zeros(len(Z)))
    with pytest.raises(ValueError, match="lambda_reg"):
        s.fit(Z, Y, lambda_reg=-1.0)
    with pytest.raises(RuntimeError, match="not fitted"):
        s.predict(Z)


def test_predict_proba_softmax_and_guards():
    X, y = _two_moons_like()
    s = MasterClosedFormSolver()
    s.fit(X, np.eye(2)[y], lambda_reg=1.0)
    p = s.predict_proba(X)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.all(p > 0)
    sharp = s.predict_proba(X, temperature=0.1)
    assert np.mean(sharp.max(axis=1)) > np.mean(p.max(axis=1))
    with pytest.raises(ValueError, match="temperature"):
        s.predict_proba(X, temperature=0.0)
    with pytest.raises(ValueError, match="features"):
        s.predict(X[:, :3])


def test_save_load_roundtrip(tmp_path):
    Z, Y = _regression_data()
    s = MasterClosedFormSolver()
    s.fit(Z, Y, lambda_reg=1.0)
    path = tmp_path / "head.npz"
    s.save(path, mean=np.zeros(8))
    r = MasterClosedFormSolver.load(path)
    assert np.array_equal(r.predict(Z), s.predict(Z))
    with pytest.raises(ValueError, match="collide"):
        s.save(tmp_path / "x.npz", coef=np.zeros(1))


# --------------------------------------------------------- production wiring
def test_package_exports_solver():
    assert ExportedSolver is MasterClosedFormSolver


def _write_store(tmp_path):
    X, y = _two_moons_like(n_per=80, seed=11)
    path = tmp_path / "store.npz"
    np.savez(path, train_full=X[::2].astype(np.float32), train_label=y[::2].astype(np.int64),
             test_full=X[1::2].astype(np.float32))
    return path


def test_cli_manifold_fit_ridge_and_graph(tmp_path, capsys):
    store = _write_store(tmp_path)
    out = tmp_path / "head.npz"
    rc = cli_main(["manifold-fit", "--features", str(store), "--lambda-reg", "1.0",
                   "--holdout", "0.25", "--out", str(out), "--json"])
    assert rc == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["n_fit"] == 60 and rep["n_holdout"] == 20
    assert rep["holdout_accuracy"] > 0.9
    assert out.exists()

    rc = cli_main(["manifold-fit", "--features", str(store), "--lambda-reg", "1.0",
                   "--holdout", "0.25", "--eta", "1.0", "--unlabeled-block", "test_full",
                   "--prior", str(out), "--json"])
    assert rc == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["n_unlabeled"] == 80
    assert rep["diagnostics"]["graph_term"] > 0.0
    assert rep["holdout_accuracy"] > 0.9


def test_cli_manifold_fit_fails_closed(tmp_path, capsys):
    store = _write_store(tmp_path)
    rc = cli_main(["manifold-fit", "--features", str(store), "--block", "missing"])
    assert rc == 1
    assert "missing" in capsys.readouterr().err
