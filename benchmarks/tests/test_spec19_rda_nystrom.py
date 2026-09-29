"""Spec 19 Phase 3 direction 3: RDA and Nystrom heads (sota_enhanced_heads) and their pipeline
wiring (S4.2, S4.3, S6.1/S6.2).

Covers the task's test list: beta=0 RDA is exactly a linear discriminant, Nystrom classifies a
problem no linear head can, save/load is bit-exact, inference runs with torch import blocked,
malformed input/config is rejected, and training reads only the rows passed to `.fit()`.

All data is synthetic (Gaussian classes, XOR, concentric rings); no task files. Tests that call
sklearn (every `.fit()`) skip without it; inference itself needs neither sklearn nor torch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SUITES = REPO / "benchmarks" / "suites"
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(SUITES))

import benchmark_sota_ensemble as bse  # noqa: E402
import sota_enhanced_heads as eh  # noqa: E402
import sota_ensemble_experts as sx  # noqa: E402


def _gaussian_classes(n_per_class, D=6, K=3, seed=0, sep=3.0):
    """K well-separated Gaussian blobs in R^D, class-conditional covariance not shared."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=sep, size=(K, D))
    X, y = [], []
    for c in range(K):
        cov_scale = rng.uniform(0.3, 1.5, size=D)
        Xc = centers[c] + rng.normal(size=(n_per_class, D)) * cov_scale
        X.append(Xc)
        y.append(np.full(n_per_class, c))
    X, y = np.concatenate(X).astype(np.float32), np.concatenate(y).astype(np.int64)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def _xor_data(n, D=16, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, D)).astype(np.float32)
    return X, ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)


def _concentric_rings(n_per_class, seed=0, noise=0.15):
    """2 classes by radius only: inner ring (r~1) vs outer ring (r~3). No hyperplane separates
    them (angle is uninformative, and both rings are centered on the origin)."""
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(0, 2 * np.pi, n_per_class)
    theta1 = rng.uniform(0, 2 * np.pi, n_per_class)
    r0 = 1.0 + rng.normal(scale=noise, size=n_per_class)
    r1 = 3.0 + rng.normal(scale=noise, size=n_per_class)
    X0 = np.stack([r0 * np.cos(theta0), r0 * np.sin(theta0)], axis=1)
    X1 = np.stack([r1 * np.cos(theta1), r1 * np.sin(theta1)], axis=1)
    X = np.concatenate([X0, X1]).astype(np.float32)
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(np.int64)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ---------------------------------------------------- test 1: beta=0 RDA is exactly linear (LDA)

def _pairwise_diffs(g: np.ndarray) -> np.ndarray:
    """(N, K) class scores -> (N, K-1) each class's score minus class 0's. The K raw scores of a
    QDA share a per-class quadratic term only when beta=0 (every class then has the same Sigma),
    and that shared term cancels in g_i - g_j but not in g_i itself: the affine identity below
    holds for the DIFFERENCES, which is what determines argmax / the decision surface, not for
    the raw scores."""
    return g[..., 1:] - g[..., :1]


def test_beta_zero_rda_decision_surface_is_exactly_affine():
    """With beta=0 every class shares Sigma_pooled, so g_i(x) - g_j(x) is affine in x: its value
    at the midpoint of any two points must equal the average of its values at the endpoints. A
    genuine quadratic surface (beta > 0, class-specific curvature) fails this exact identity."""
    pytest.importorskip("sklearn")
    X, y = _gaussian_classes(80, D=5, K=3, seed=1)
    head0 = eh.RDAHead.fit(X, y, 3, pair=False, seed=0, config=eh.RDAConfig(pca_k=4, beta_grid=(0.0,)))
    assert head0.cfg["beta"] == 0.0
    rng = np.random.default_rng(2)
    x1 = rng.normal(scale=4.0, size=(30, 5)).astype(np.float32)
    x2 = rng.normal(scale=4.0, size=(30, 5)).astype(np.float32)
    xm = 0.5 * (x1 + x2)
    d1, d2, dm = (_pairwise_diffs(head0.scores(x).astype(np.float64)) for x in (x1, x2, xm))
    err = np.max(np.abs(dm - 0.5 * (d1 + d2)))
    assert err < 1e-3, err

    # Contrast: a genuinely curved (beta>0) surface must NOT satisfy this identity, or the test
    # above would be vacuous for any config.
    head_curved = eh.RDAHead.fit(X, y, 3, pair=False, seed=0, config=eh.RDAConfig(pca_k=4, beta_grid=(1.0,)))
    assert head_curved.cfg["beta"] == 1.0
    dc1, dc2, dcm = (_pairwise_diffs(head_curved.scores(x).astype(np.float64)) for x in (x1, x2, xm))
    assert np.max(np.abs(dcm - 0.5 * (dc1 + dc2))) > 1e-1


def test_beta_selected_by_inner_cv_is_one_of_the_grid_values():
    pytest.importorskip("sklearn")
    X, y = _gaussian_classes(60, D=4, K=2, seed=3)
    head = eh.RDAHead.fit(X, y, 2, pair=False, seed=0, config=eh.RDAConfig(pca_k=3))
    assert head.cfg["beta"] in eh.RDAConfig().beta_grid


# --------------------------------------------- test 2: Nystrom classifies a non-linear problem

def test_nystrom_classifies_concentric_rings_a_linear_probe_cannot():
    pytest.importorskip("sklearn")
    Xtr, ytr = _concentric_rings(220, seed=4)
    Xte, yte = _concentric_rings(120, seed=5)

    nys = eh.NystromHead.fit(Xtr, ytr, 2, pair=False, seed=0,
                              config=eh.NystromConfig(landmarks=48, rank=24))
    nys_acc = float(np.mean(nys.scores(Xte).argmax(1) == yte))

    lin = sx.LinearProbe.fit(Xtr, ytr, 2, pair=False, kind="full", pca_k=None, seed=0, device="cpu")
    lin_acc = float(np.mean(lin.scores(Xte).argmax(1) == yte))

    assert nys_acc > 0.9, nys_acc
    assert lin_acc < 0.65, lin_acc          # radius carries no linear direction: near chance
    assert nys_acc > lin_acc + 0.25


def test_nystrom_gamma_selected_from_the_multiplier_grid():
    """gamma must equal gamma_0 * one of the configured multipliers, gamma_0 the median-distance
    heuristic recomputed from the landmarks the head actually stored."""
    pytest.importorskip("sklearn")
    Xtr, ytr = _concentric_rings(80, seed=6)
    cfg = eh.NystromConfig(landmarks=32, rank=16, gamma_multipliers=(0.25, 1.0, 4.0))
    head = eh.NystromHead.fit(Xtr, ytr, 2, pair=False, seed=0, config=cfg)
    Z = head.a["Z"].astype(np.float64)
    d2 = eh._sqdist(Z, Z)
    med = np.median(d2[~np.eye(len(Z), dtype=bool)])
    gamma0 = 1.0 / med
    assert head.gamma > 0.0
    assert any(np.isclose(head.gamma, gamma0 * m, rtol=1e-6) for m in cfg.gamma_multipliers)
    assert head.cfg["config"]["gamma_multipliers"] == list(cfg.gamma_multipliers)


# ----------------------------------------------------------- test 3: save/load bit-exact

def test_rda_save_load_roundtrip_is_bit_exact(tmp_path):
    pytest.importorskip("sklearn")
    X, y = _gaussian_classes(50, D=5, K=3, seed=7)
    head = eh.RDAHead.fit(X, y, 3, pair=False, seed=0, config=eh.RDAConfig(pca_k=4))
    head.save(tmp_path / "rda.npz")
    back = eh.RDAHead.load(tmp_path / "rda.npz")
    for k in eh.RDAHead.ARRAY_KEYS:
        assert back.a[k].dtype == np.float32 and np.array_equal(back.a[k], head.a[k]), k
    assert back.cfg == head.cfg
    Xq = np.random.default_rng(8).normal(size=(20, head.in_dim)).astype(np.float32)
    assert np.array_equal(back.scores(Xq), head.scores(Xq))
    assert np.array_equal(back.score(Xq[0], np.zeros((head.K, 1))), head.scores(Xq[:1])[0])


def test_nystrom_save_load_roundtrip_is_bit_exact(tmp_path):
    pytest.importorskip("sklearn")
    Xtr, ytr = _concentric_rings(60, seed=9)
    head = eh.NystromHead.fit(Xtr, ytr, 2, pair=False, seed=0, config=eh.NystromConfig(landmarks=24, rank=12))
    head.save(tmp_path / "nys.npz")
    back = eh.NystromHead.load(tmp_path / "nys.npz")
    for k in eh.NystromHead.ARRAY_KEYS:
        assert back.a[k].dtype == np.float32 and np.array_equal(back.a[k], head.a[k]), k
    for k in head.probe.a:
        assert np.array_equal(back.probe.a[k], head.probe.a[k]), k
    assert back.cfg == head.cfg
    Xq = np.random.default_rng(10).normal(size=(15, head.in_dim)).astype(np.float32)
    assert np.array_equal(back.scores(Xq), head.scores(Xq))
    assert np.array_equal(back.score(Xq[0], np.zeros((head.K, 1))), head.scores(Xq[:1])[0])


# ----------------------------------------------------- test 4: NumPy-only inference (no torch)

def test_rda_load_and_score_run_with_torch_import_blocked(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    X, y = _gaussian_classes(40, D=4, K=2, seed=11)
    head = eh.RDAHead.fit(X, y, 2, pair=False, seed=0, config=eh.RDAConfig(pca_k=3))
    head.save(tmp_path / "rda.npz")
    Xq = np.random.default_rng(12).normal(size=(10, head.in_dim)).astype(np.float32)
    expect = head.scores(Xq)
    monkeypatch.setitem(sys.modules, "torch", None)
    back = eh.RDAHead.load(tmp_path / "rda.npz")
    out = back.scores(Xq)
    assert type(out) is np.ndarray and out.dtype == np.float32
    assert np.array_equal(out, expect)
    with pytest.raises(ImportError):
        import torch as _t  # noqa: F401


def test_nystrom_load_and_score_run_with_torch_import_blocked(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    Xtr, ytr = _concentric_rings(40, seed=13)
    head = eh.NystromHead.fit(Xtr, ytr, 2, pair=False, seed=0, config=eh.NystromConfig(landmarks=20, rank=10))
    head.save(tmp_path / "nys.npz")
    Xq = np.random.default_rng(14).normal(size=(10, head.in_dim)).astype(np.float32)
    expect = head.scores(Xq)
    monkeypatch.setitem(sys.modules, "torch", None)
    back = eh.NystromHead.load(tmp_path / "nys.npz")
    out = back.scores(Xq)
    assert type(out) is np.ndarray and out.dtype == np.float32
    assert np.array_equal(out, expect)
    with pytest.raises(ImportError):
        import torch as _t  # noqa: F401


# ----------------------------------------------------------- test 5: input/config defense

@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_rda_non_finite_input_is_rejected(bad):
    pytest.importorskip("sklearn")
    X, y = _gaussian_classes(30, D=4, K=2, seed=15)
    head = eh.RDAHead.fit(X, y, 2, pair=False, seed=0, config=eh.RDAConfig(pca_k=3))
    Xq = np.zeros((3, head.in_dim), dtype=np.float32)
    Xq[1, 0] = bad
    with pytest.raises(ValueError):
        head.scores(Xq)
    with pytest.raises(ValueError):
        head.score(Xq[1], np.zeros((head.K, 1)))
    with pytest.raises(ValueError):
        eh.RDAHead.fit(np.repeat(Xq, 20, 0), np.tile([0, 1, 0], 20), 2, pair=False, seed=0)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nystrom_non_finite_input_is_rejected(bad):
    pytest.importorskip("sklearn")
    Xtr, ytr = _concentric_rings(30, seed=16)
    head = eh.NystromHead.fit(Xtr, ytr, 2, pair=False, seed=0, config=eh.NystromConfig(landmarks=16, rank=8))
    Xq = np.zeros((3, head.in_dim), dtype=np.float32)
    Xq[1, 0] = bad
    with pytest.raises(ValueError):
        head.scores(Xq)
    with pytest.raises(ValueError):
        head.score(Xq[1], np.zeros((head.K, 1)))
    with pytest.raises(ValueError):
        eh.NystromHead.fit(np.repeat(Xq, 20, 0), np.tile([0, 1, 0], 20), 2, pair=False, seed=0)


def test_rda_wrong_width_wrong_k_and_bad_labels_are_rejected():
    pytest.importorskip("sklearn")
    X, y = _gaussian_classes(30, D=4, K=2, seed=17)
    head = eh.RDAHead.fit(X, y, 2, pair=False, seed=0, config=eh.RDAConfig(pca_k=3))
    with pytest.raises(ValueError):
        head.scores(np.zeros((2, head.in_dim + 1), dtype=np.float32))
    with pytest.raises(ValueError):
        head.score(np.zeros(head.in_dim, dtype=np.float32), np.zeros((head.K + 1, 1)))
    Xb = np.zeros((40, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        eh.RDAHead.fit(Xb, np.full(40, 3), 3, pair=False, seed=0)          # label == K
    with pytest.raises(ValueError):
        eh.RDAHead.fit(Xb, np.zeros(39, dtype=int), 3, pair=False, seed=0)  # N != len(y)
    with pytest.raises(ValueError):
        eh.RDAHead(dict(head.a, prec_c=head.a["prec_c"][:, :, :-1]), head.cfg)


def test_nystrom_wrong_width_wrong_k_and_bad_labels_are_rejected():
    pytest.importorskip("sklearn")
    Xtr, ytr = _concentric_rings(30, seed=18)
    head = eh.NystromHead.fit(Xtr, ytr, 2, pair=False, seed=0, config=eh.NystromConfig(landmarks=16, rank=8))
    with pytest.raises(ValueError):
        head.scores(np.zeros((2, head.in_dim + 1), dtype=np.float32))
    with pytest.raises(ValueError):
        head.score(np.zeros(head.in_dim, dtype=np.float32), np.zeros((head.K + 1, 1)))
    Xb = np.zeros((40, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        eh.NystromHead.fit(Xb, np.full(40, 3), 3, pair=False, seed=0)
    with pytest.raises(ValueError):
        eh.NystromHead.fit(Xb, np.zeros(39, dtype=int), 3, pair=False, seed=0)
    with pytest.raises(ValueError):
        eh.NystromHead(dict(head.a, Proj=head.a["Proj"][:, :-1]), head.cfg)


@pytest.mark.parametrize("bad_kwargs", [{"pca_k": 0}, {"pca_k": -1}, {"beta_grid": ()},
                                        {"beta_grid": (1.5,)}, {"beta_grid": (-0.1,)}, {"device": "cuda"}])
def test_rda_config_rejects_bad_values(bad_kwargs):
    with pytest.raises(ValueError):
        eh.RDAConfig(**bad_kwargs)


@pytest.mark.parametrize("bad_kwargs", [{"landmarks": 1}, {"rank": 0}, {"gamma_multipliers": ()},
                                        {"gamma_multipliers": (0.0,)}, {"gamma_multipliers": (-1.0,)},
                                        {"class_stratified_landmarks": "yes"}, {"device": "cuda"}])
def test_nystrom_config_rejects_bad_values(bad_kwargs):
    with pytest.raises(ValueError):
        eh.NystromConfig(**bad_kwargs)


# ------------------------------------------------------- test 6: training row isolation

def test_rda_training_reads_only_the_rows_passed():
    """S7: NaN sentinel rows sit outside `tr`; _fit_predict must neither fail nor read them."""
    pytest.importorskip("sklearn")
    X, y = _xor_data(240, D=6, seed=19)
    X[::6] = np.nan
    tr = np.flatnonzero(np.arange(240) % 6 != 0)
    f = {"X_train": X, "train_label": y, "cands": np.zeros((2, 1), dtype=np.float32)}
    m, s, info = bse._fit_predict({"type": "rda", "pca_k": 4}, "massive_en", f, tr, tr[:30], X[tr[:30]], [0, 0, 100])
    assert s.shape == (30, 2) and np.all(np.isfinite(s))
    assert m.supervised_param_count() == info["params"]


def test_nystrom_training_reads_only_the_rows_passed():
    pytest.importorskip("sklearn")
    X, y = _xor_data(240, D=6, seed=20)
    X[::6] = np.nan
    tr = np.flatnonzero(np.arange(240) % 6 != 0)
    f = {"X_train": X, "train_label": y, "cands": np.zeros((2, 1), dtype=np.float32)}
    spec = {"type": "nystrom", "landmarks": 32, "rank": 16}
    m, s, info = bse._fit_predict(spec, "massive_en", f, tr, tr[:30], X[tr[:30]], [0, 0, 100])
    assert s.shape == (30, 2) and np.all(np.isfinite(s))
    assert m.supervised_param_count() == info["params"]


# ------------------------------------------------------------ test 7: parameter counts

def test_rda_supervised_param_count_matches_the_spec_formula():
    pytest.importorskip("sklearn")
    X, y = _gaussian_classes(40, D=8, K=4, seed=21)
    head = eh.RDAHead.fit(X, y, 4, pair=False, seed=0, config=eh.RDAConfig(pca_k=5))
    k = head.k
    assert head.supervised_param_count() == 4 * (k + k * (k + 1) // 2)


def test_nystrom_supervised_param_count_matches_the_spec_formula():
    pytest.importorskip("sklearn")
    Xtr, ytr = _concentric_rings(60, seed=22)
    head = eh.NystromHead.fit(Xtr, ytr, 2, pair=False, seed=0,
                              config=eh.NystromConfig(landmarks=24, rank=12, class_stratified_landmarks=True))
    assert head.supervised_param_count() == 2 * (head.r + 1) + head.m
    head_unstratified = eh.NystromHead.fit(Xtr, ytr, 2, pair=False, seed=0,
                                           config=eh.NystromConfig(landmarks=24, rank=12,
                                                                    class_stratified_landmarks=False))
    assert head_unstratified.supervised_param_count() == 2 * (head_unstratified.r + 1)


# -------------------------------------------------------- test 8: pair-task source

def test_rda_pair_source_uses_the_full_context_third_only():
    pytest.importorskip("sklearn")
    D, K = 5, 3
    Xfull, y = _gaussian_classes(40, D=D, K=K, seed=25)
    rng = np.random.default_rng(26)
    Xpair = np.concatenate([Xfull, rng.normal(size=(len(y), D)).astype(np.float32),
                            rng.normal(size=(len(y), D)).astype(np.float32)], axis=1)
    head = eh.RDAHead.fit(Xpair, y, K, pair=True, seed=0, config=eh.RDAConfig(pca_k=4))
    assert head.in_dim == 3 * D
    Xq, Xq_a_b_changed, Xq_full_changed = Xpair[:10].copy(), Xpair[:10].copy(), Xpair[:10].copy()
    Xq_a_b_changed[:, D:] = 99.0
    Xq_full_changed[:, :D] = 99.0
    assert np.array_equal(head.scores(Xq), head.scores(Xq_a_b_changed))
    assert not np.array_equal(head.scores(Xq), head.scores(Xq_full_changed))


def test_nystrom_pair_source_uses_the_full_context_third_only():
    pytest.importorskip("sklearn")
    D = 2
    Xfull, y = _concentric_rings(60, seed=27)
    rng = np.random.default_rng(28)
    Xpair = np.concatenate([Xfull, rng.normal(size=(len(y), D)).astype(np.float32),
                            rng.normal(size=(len(y), D)).astype(np.float32)], axis=1)
    head = eh.NystromHead.fit(Xpair, y, 2, pair=True, seed=0, config=eh.NystromConfig(landmarks=24, rank=12))
    assert head.in_dim == 3 * D
    Xq, Xq_a_b_changed, Xq_full_changed = Xpair[:10].copy(), Xpair[:10].copy(), Xpair[:10].copy()
    Xq_a_b_changed[:, D:] = 99.0
    Xq_full_changed[:, :D] = 99.0
    assert np.array_equal(head.scores(Xq), head.scores(Xq_a_b_changed))
    assert not np.array_equal(head.scores(Xq), head.scores(Xq_full_changed))


# ------------------------------------------------- test 9: pipeline wiring (S6.2)

def test_expert_specs_register_rda_and_nystrom_only_when_enabled(monkeypatch):
    fs = {"enc": {"train_full": np.zeros((5, 16))}}
    monkeypatch.setattr(bse, "ADAPTER_RANKS", ())
    monkeypatch.setattr(bse, "SUPCON_RANKS", ())
    monkeypatch.setattr(bse, "ENABLE_RDA", False)
    monkeypatch.setattr(bse, "ENABLE_NYSTROM", False)
    names = [n for n, _, _ in bse.expert_specs("massive_en", fs)]
    assert "enc:rda" not in names and "enc:nystrom" not in names

    monkeypatch.setattr(bse, "ENABLE_RDA", True)
    monkeypatch.setattr(bse, "ENABLE_NYSTROM", True)
    specs = {n: sp for n, _, sp in bse.expert_specs("massive_en", fs)}
    assert specs["enc:rda"] == {"type": "rda"}
    assert specs["enc:nystrom"] == {"type": "nystrom"}
    assert eh.head_config(specs["enc:rda"]) == eh.RDAConfig()
    assert eh.head_config(specs["enc:nystrom"]) == eh.NystromConfig()


def test_strategy_complexity_counts_rda_and_nystrom_params():
    specs = {"e:rda": {"type": "rda"}, "e:nystrom": {"type": "nystrom"}}
    meta = {"e:rda": {"folds": [{"params": 100}, {"params": 120}]},
            "e:nystrom": {"folds": [{"params": 80}, {"params": 90}]}}
    assert bse.strategy_complexity("single:e:rda", specs, meta, 2, {}) == (0, 120)
    assert bse.strategy_complexity("single:e:nystrom", specs, meta, 2, {}) == (0, 90)


def test_load_expert_dispatches_rda_and_nystrom(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    monkeypatch.setattr(bse, "OUT_ART", tmp_path / "art")
    X, y = _gaussian_classes(30, D=6, K=2, seed=23)
    rda = eh.RDAHead.fit(X, y, 2, pair=False, seed=0, config=eh.RDAConfig(pca_k=4))
    Xr, yr = _concentric_rings(30, seed=24)
    nys = eh.NystromHead.fit(Xr, yr, 2, pair=False, seed=0, config=eh.NystromConfig(landmarks=16, rank=8))
    mdir = tmp_path / "art" / "models" / "massive_en"
    mdir.mkdir(parents=True)
    rda.save(mdir / "enc__rda.npz")
    nys.save(mdir / "enc__nystrom.npz")
    got_rda = bse.load_expert("massive_en", "enc:rda", "enc", {"type": "rda"}, {})
    got_nys = bse.load_expert("massive_en", "enc:nystrom", "enc", {"type": "nystrom"}, {})
    assert isinstance(got_rda, eh.RDAHead) and isinstance(got_nys, eh.NystromHead)
    assert np.array_equal(got_rda.scores(X[:5]), rda.scores(X[:5]))
    assert np.array_equal(got_nys.scores(Xr[:5]), nys.scores(Xr[:5]))


def _write_source(root: Path, task: str, X, y, Xte, K):
    (root / "features").mkdir(parents=True)
    np.savez(root / "features" / f"{task}.npz", train_full=X, test_full=Xte,
             train_ids=np.arange(len(y)), test_ids=np.arange(len(Xte)), train_label=y,
             cands=np.eye(K, 4, dtype=np.float32), info_json=np.array(json.dumps({"encoder": "synthetic"})))


def test_stage_fit_combine_and_load_expert_end_to_end_for_rda_and_nystrom(tmp_path, monkeypatch):
    """The S6.2 wiring points on a synthetic source, run through the real CLI stages (not called
    in isolation): fit (OOF + full-train save under stage_fit's `spec["type"] in eh.HEADS`
    branch), complexity from the saved fold params, combine, load_expert."""
    pytest.importorskip("sklearn")
    X, y = _concentric_rings(150, seed=29)
    Xte, _ = _concentric_rings(20, seed=30)
    _write_source(tmp_path / "src", "massive_en", X, y, Xte, 2)
    monkeypatch.setattr(bse, "SOURCES", {"syn": tmp_path / "src"})
    monkeypatch.setattr(bse, "OUT_ART", tmp_path / "art")
    monkeypatch.setattr(bse, "ADAPTER_RANKS", ())
    monkeypatch.setattr(bse, "SUPCON_RANKS", ())
    monkeypatch.setattr(bse, "ENABLE_RDA", True)
    monkeypatch.setattr(bse, "ENABLE_NYSTROM", True)
    monkeypatch.setattr(bse, "RNN_SOURCES", [])
    bse.stage_fit(["massive_en"], ["rda", "nystrom"])
    for kind in ("rda", "nystrom"):
        model = tmp_path / "art" / "models" / "massive_en" / f"syn__{kind}.npz"
        assert model.exists()
        with np.load(tmp_path / "art" / "oof" / "massive_en" / f"syn__{kind}.npz", allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"]))
        assert meta["spec"] == {"type": kind} and len(meta["folds"]) == bse.N_FOLDS
        assert all(fo["params"] > 0 for fo in meta["folds"])
    bse.stage_combine(["massive_en"], learned=())
    comb = json.loads((tmp_path / "art" / "combine" / "massive_en.json").read_text())
    assert "single:syn:rda" in comb["nested_cv_acc"]
    assert "single:syn:nystrom" in comb["nested_cv_acc"]
    fs = bse.load_sources("massive_en")
    specs = {n: (s, sp) for n, s, sp in bse.expert_specs("massive_en", fs)}
    rda_core = bse.load_expert("massive_en", "syn:rda", *specs["syn:rda"], fs)
    nys_core = bse.load_expert("massive_en", "syn:nystrom", *specs["syn:nystrom"], fs)
    assert isinstance(rda_core, eh.RDAHead) and isinstance(nys_core, eh.NystromHead)
    assert rda_core.score(Xte[0], fs["syn"]["cands"]).shape == (2,)
    assert nys_core.score(Xte[0], fs["syn"]["cands"]).shape == (2,)
