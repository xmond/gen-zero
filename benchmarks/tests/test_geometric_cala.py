"""Geometric latent fusion + CALA (benchmarks/suites/geometric_latent_fusion.py,
confidence_adaptive_logit_adjustment.py). All data is synthetic; no 01.PNG features.

Covers:
  1. Procrustes / core-residual decomposition: a known rotation between two views is recovered,
     core and residual coordinate blocks are orthogonal directions, transform() of held-out rows
     uses only train-fit statistics (changing held-out rows never changes what a train row maps to).
  2. CALA operator: phi == 0 leaves logits untouched, phi == 1 equals spec21 logit_adjust, the gates
     stay in [0, 1], confident rows keep their argmax, ambiguous minority rows are rescued.
  3. Conformal helpers of the evaluation suite: class-conditional (Mondrian) coverage.
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks" / "suites"))

import confidence_adaptive_logit_adjustment as cala  # noqa: E402
import geometric_latent_fusion as glf  # noqa: E402
import spec21_advanced_heads as s21  # noqa: E402


def _two_views(n=400, d=30, latent=6, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, latent)) * np.array([5, 4, 3, 2.5, 2, 1.5])[:latent]
    a = rng.standard_normal((latent, d))
    q_rot, _ = np.linalg.qr(rng.standard_normal((d, d)))
    xq = z @ a + noise * rng.standard_normal((n, d))
    xl = (z @ a) @ q_rot + noise * rng.standard_normal((n, d))       # same latent, rotated basis
    return xq, xl, z


# ---------------------------------------------------------------- 1. geometry

def test_procrustes_recovers_known_rotation():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((300, 5))
    rot, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    b = a @ rot
    est = glf.orthogonal_procrustes(a, b)
    assert np.allclose(est, rot, atol=1e-8)
    assert np.allclose(est.T @ est, np.eye(5), atol=1e-10)


def test_core_captures_shared_latent_and_residual_is_orthogonal():
    xq, xl, _ = _two_views()
    fus = glf.GeometricLatentFusion.fit(xq, xl, core_energy=0.95)
    assert 1 <= fus.core_dim <= 6 + 2                # 6 shared latents (+ a little noise energy)
    for basis in (fus.core_dirs_q_, fus.core_dirs_l_):
        assert np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-8)
    # core and residual directions of one model are orthogonal (columns of one orthogonal matrix)
    assert np.abs(fus.core_dirs_q_.T @ fus.resid_dirs_q_).max() < 1e-8
    assert np.abs(fus.core_dirs_l_.T @ fus.resid_dirs_l_).max() < 1e-8
    rep = fus.transform(xq, xl)
    assert rep.shape == (len(xq), fus.core_dim + fus.resid_dirs_q_.shape[1] + fus.resid_dirs_l_.shape[1])
    assert np.isfinite(rep).all()


def test_core_coordinates_agree_across_views():
    xq, xl, _ = _two_views(noise=0.02)
    fus = glf.GeometricLatentFusion.fit(xq, xl, core_energy=0.95)
    cq, cl = fus.core_views(xq, xl)
    for j in range(min(4, cq.shape[1])):
        assert abs(np.corrcoef(cq[:, j], cl[:, j])[0, 1]) > 0.95     # aligned axes carry the same signal


def test_transform_uses_only_train_statistics():
    xq, xl, _ = _two_views(n=300)
    fus = glf.GeometricLatentFusion.fit(xq[:200], xl[:200])
    first = fus.transform(xq[:200], xl[:200])
    again = fus.transform(np.vstack([xq[:200], xq[200:] * 50]), np.vstack([xl[:200], xl[200:] * 50]))[:200]
    assert np.allclose(first, again)                                  # held-out rows cannot move train rows
    mu = fus.transform(xq[:200], xl[:200]).mean(axis=0)
    assert np.abs(mu).max() < 1e-6                                    # centred by TRAIN mean


def test_fit_rejects_mismatched_rows_and_nonfinite():
    xq, xl, _ = _two_views(n=50)
    with pytest.raises(ValueError):
        glf.GeometricLatentFusion.fit(xq, xl[:-1])
    bad = xq.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        glf.GeometricLatentFusion.fit(bad, xl)


def test_residual_weight_scales_only_residual_blocks():
    xq, xl, _ = _two_views()
    fus = glf.GeometricLatentFusion.fit(xq, xl)
    a = fus.transform(xq, xl, residual_weight=1.0)
    b = fus.transform(xq, xl, residual_weight=0.25)
    k = fus.core_dim
    assert np.allclose(a[:, :k], b[:, :k])
    assert np.allclose(0.25 * a[:, k:], b[:, k:])


# ---------------------------------------------------------------- 2. CALA

def _logits(n=6, k=3, seed=3):
    return np.random.default_rng(seed).standard_normal((n, k))


PRIORS = np.array([0.8, 0.15, 0.05])


def test_phi_zero_is_identity_and_phi_one_is_static_logit_adjust():
    z = _logits()
    assert np.array_equal(cala.cala_adjust(z, PRIORS, 1.0, np.zeros(len(z))), z)
    assert np.allclose(cala.cala_adjust(z, PRIORS, 1.0, np.ones(len(z))), s21.logit_adjust(z, PRIORS, 1.0))
    assert np.allclose(cala.cala_adjust(z, PRIORS, 0.0, np.full(len(z), 0.7)), z)


def test_sign_convention_boosts_the_minority_class():
    z = np.zeros((1, 3))
    out = cala.cala_adjust(z, PRIORS, 1.0, np.ones(1))
    assert out[0, 2] > out[0, 1] > out[0, 0]                          # rarest class gains most


@pytest.mark.parametrize("mode", ["entropy", "margin"])
@pytest.mark.parametrize("gamma", [1.0, 2.0, 4.0])
def test_gate_range_and_monotone_in_confidence(mode, gamma):
    rng = np.random.default_rng(5)
    z = rng.standard_normal((200, 4)) * 3
    phi = cala.gate_from_logits(z, scale=1.0, mode=mode, gamma=gamma)
    assert phi.shape == (200,) and phi.min() >= 0.0 and phi.max() <= 1.0
    sharp, flat = np.array([[9.0, 0, 0, 0]]), np.array([[0.1, 0.0, 0.0, 0.0]])
    assert cala.gate_from_logits(sharp, 1.0, mode, gamma)[0] < cala.gate_from_logits(flat, 1.0, mode, gamma)[0]


def test_centroid_gate_range():
    rng = np.random.default_rng(6)
    means = rng.standard_normal((4, 10)) * 4
    f = np.vstack([means[0] + 0.01, (means[0] + means[1]) / 2])       # on a centroid vs on the bisector
    phi = cala.gate_from_centroids(f, means, gamma=1.0)
    assert 0.0 <= phi.min() and phi.max() <= 1.0
    assert phi[0] < 0.05 and phi[1] > 0.95


def test_confident_rows_keep_argmax_and_ambiguous_minority_is_rescued():
    # Row 0: confident majority; row 1: near-tie between majority and rare class.
    z = np.array([[8.0, 0.0, 0.0], [1.02, 0.5, 1.0]])
    phi = cala.gate_from_logits(z, scale=1.0, mode="margin", gamma=2.0)
    adj = cala.cala_adjust(z, PRIORS, 1.0, phi)
    assert adj[0].argmax() == 0 and phi[0] < 0.01
    assert z[1].argmax() == 0 and adj[1].argmax() == 2
    static = s21.logit_adjust(z, PRIORS, 1.0)
    assert static[0].argmax() == 0                                    # gate keeps confident rows sharp


def test_cala_validation():
    z = _logits()
    with pytest.raises(ValueError):
        cala.cala_adjust(z, PRIORS, -1.0, np.ones(len(z)))
    with pytest.raises(ValueError):
        cala.cala_adjust(z, PRIORS, 1.0, np.full(len(z), 1.5))
    with pytest.raises(ValueError):
        cala.cala_adjust(z, PRIORS, 1.0, np.ones(len(z) + 1))
    with pytest.raises(ValueError):
        cala.gate_from_logits(z, 1.0, "nonsense", 1.0)


# ---------------------------------------------------------------- 3. conformal (suite helper)

def test_mondrian_conformal_class_conditional_coverage():
    import evaluate_manifold_pareto_ensemble as ev
    rng = np.random.default_rng(11)
    K, n_cal, n_test = 3, 3000, 6000
    pri = np.array([0.7, 0.2, 0.1])

    def draw(n):
        y = rng.choice(K, n, p=pri)
        z = rng.standard_normal((n, K))
        z[np.arange(n), y] += 1.5
        return z, y
    zc, yc = draw(n_cal)
    zt, yt = draw(n_test)
    out = ev.mondrian_conformal(zc, yc, zt, yt, alpha=0.1)
    for c in range(K):
        assert out["class_coverage"][str(c)] >= 0.88                  # >= 1 - alpha up to sampling noise
    assert out["marginal_coverage"] >= 0.88
    assert out["mean_set_size"] >= 1.0


def test_mondrian_conformal_tiny_class_gets_full_set_not_a_fake_guarantee():
    import evaluate_manifold_pareto_ensemble as ev
    rng = np.random.default_rng(12)
    zc = rng.standard_normal((40, 3))
    yc = np.array([0] * 30 + [1] * 9 + [2] * 1)                       # class 2 has one calibration row
    zt = rng.standard_normal((20, 3))
    yt = rng.integers(0, 3, 20)
    out = ev.mondrian_conformal(zc, yc, zt, yt, alpha=0.1)
    assert out["trivial_classes"] == [2]


# ---------------------------------------------------------------- 4. evaluator internals

def test_metrics_match_sklearn():
    import evaluate_manifold_pareto_ensemble as ev
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    rng = np.random.default_rng(21)
    y = rng.integers(0, 4, 500)
    pred = np.where(rng.random(500) < 0.6, y, rng.integers(0, 4, 500))
    m = ev.metrics_from_pred(y, pred, 4)
    assert m["accuracy"] == pytest.approx(100 * accuracy_score(y, pred))
    assert m["balanced_accuracy"] == pytest.approx(100 * balanced_accuracy_score(y, pred))
    assert m["macro_f1"] == pytest.approx(100 * f1_score(y, pred, average="macro"))
    assert m["zero_recall_classes"] == 0


def test_collapse_flags_are_reported():
    import evaluate_manifold_pareto_ensemble as ev
    y = np.array([0] * 95 + [1] * 5)
    m = ev.metrics_from_pred(y, np.zeros(100, dtype=int), 2)
    assert m["zero_recall_classes"] == 1 and m["max_pred_class_frac"] == 1.0 and m["balanced_accuracy"] == 50.0


def _fake_row(name, folds_peak, folds_robust, gate=True, tier=0):
    import evaluate_manifold_pareto_ensemble as ev
    c = ev.Cand("single", "bbp", rep="qwen", gate=cala.CalaConfig("static", 1.0, [0.0, 0.5, 1.0][tier]))
    return {"cand": c, "peak_folds": np.array(folds_peak, float), "robust_folds": np.array(folds_robust, float),
            "gate_peak": gate, "gate_robust": gate,
            "metrics": {"accuracy": float(np.mean(folds_peak)), "balanced_accuracy": float(np.mean(folds_robust)),
                        "macro_f1": float(np.mean(folds_robust)), "max_pred_class_frac": 0.5}}


def test_one_se_prefers_simplest_within_one_standard_error_and_flags_ungated():
    import evaluate_manifold_pareto_ensemble as ev
    simple = _fake_row("simple", [80, 82, 79, 81, 81], [0] * 5, tier=0)     # mean 80.6
    fancy = _fake_row("fancy", [81, 83, 79, 82, 80], [0] * 5, tier=2)       # mean 81, SE ~ 0.71
    chosen, admitted, se = ev.select_track([simple, fancy], "peak")
    assert admitted and chosen is simple and se > 0.5
    far = _fake_row("far", [90, 92, 88, 91, 89], [0] * 5, tier=2)
    assert ev.select_track([simple, far], "peak")[0] is far                 # a real gain still wins
    bad = _fake_row("bad", [99] * 5, [0] * 5, gate=False)
    chosen, admitted, _ = ev.select_track([bad], "peak")
    assert chosen is bad and admitted is False                              # gate failure is visible, not hidden


def test_pareto_front_is_non_dominated():
    import evaluate_manifold_pareto_ensemble as ev
    a = _fake_row("a", [90] * 5, [50] * 5)
    b = _fake_row("b", [80] * 5, [70] * 5)
    c = _fake_row("c", [79] * 5, [60] * 5)                                  # dominated by b
    front = ev.pareto_front([a, b, c])
    assert {id(r) for r in front} == {id(a), id(b)}


def test_select_task_has_no_test_label_argument_and_is_deterministic():
    import inspect
    import evaluate_manifold_pareto_ensemble as ev
    assert list(inspect.signature(ev.select_task).parameters) == ["xq", "xl", "y", "k", "seed"]
    rng = np.random.default_rng(31)
    n, d, k = 150, 40, 3
    y = np.repeat(np.arange(k), [90, 40, 20])
    mu = rng.standard_normal((k, d)) * 1.2
    xq = mu[y] + rng.standard_normal((n, d))
    xl = mu[y] @ np.linalg.qr(rng.standard_normal((d, d)))[0] + rng.standard_normal((n, d))
    out1 = ev.select_task(xq, xl, y, k)
    out2 = ev.select_task(xq, xl, y, k)
    for pool in ("all", "single"):
        for trk in ("peak", "robust"):
            assert out1[f"{pool}_{trk}"]["row"]["cand"] == out2[f"{pool}_{trk}"]["row"]["cand"]
    assert not out1["dropped_bases"]
    assert len(out1["rows_all"]) > len(out1["rows_single"]) > 40
    chosen = out1["all_peak"]["row"]
    assert chosen["metrics"]["accuracy"] > 60                               # signal is real, selection found it
    assert out1["all_robust"]["row"]["metrics"]["zero_recall_classes"] == 0 if out1["all_robust"]["admitted"] else True


def test_paired_bootstrap_detects_a_real_difference():
    import evaluate_manifold_pareto_ensemble as ev
    rng = np.random.default_rng(41)
    a = {"t1": "".join("1" if rng.random() < 0.9 else "0" for _ in range(400)),
         "t2": "".join("1" if rng.random() < 0.9 else "0" for _ in range(400))}
    b = {"t1": "".join("1" if rng.random() < 0.6 else "0" for _ in range(400)),
         "t2": "".join("1" if rng.random() < 0.6 else "0" for _ in range(400))}
    r = ev.paired_macro_bootstrap(a, b, n_boot=500)
    assert r["ci95"][0] > 20 and r["delta_pp"] == pytest.approx(30, abs=6)
    same = ev.paired_macro_bootstrap(a, a, n_boot=200)
    assert same["delta_pp"] == 0 and same["ci95"] == [0.0, 0.0]
