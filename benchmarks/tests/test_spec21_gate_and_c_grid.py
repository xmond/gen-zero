"""BBP C grid, class_weight and the min_class_recall admission gate (spec21_advanced_heads.py,
evaluate_spec21_scorecard.py, evaluate_full_13_grand_scorecard.fit_linear_probe_torch).

  1. the inner-CV C grid reaches 1e-5 and the probe can actually pick a C below the old 1e-3 floor;
  2. class_weight: validation, 'balanced' raises minority recall for BBP and the torch linear probe,
     and the scorecard wires EvalConfig.class_weight into both heads;
  3. gate: a K >= 3 head with a zero-recall class is blocked (passed False, reason recorded), a head whose
     every class clears the threshold passes, binary tasks are exempt;
  4. selection: a zero-recall head with the best CV accuracy is never chosen, and with no admissible head
     nothing is chosen.
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks" / "suites"))

import spec21_advanced_heads as s21  # noqa: E402
import evaluate_full_13_grand_scorecard as base  # noqa: E402
import evaluate_spec21_scorecard as sc  # noqa: E402


# --------------------------------------------------------------------------- 1. C grid

def test_c_grid_spans_1e_minus_5_to_1e3():
    grid = np.asarray(s21.BBP_C_GRID)
    assert grid.size == 9
    assert np.allclose(grid, np.logspace(-5, 3, 9))
    assert np.isclose(grid.min(), 1e-5) and np.isclose(grid.max(), 1e3)
    assert np.any(np.isclose(grid, 1e-4))
    assert np.all(np.diff(grid) > 0)


def test_inner_cv_uses_the_extended_grid_by_default():
    assert inspect.signature(s21._select_c_by_cv).parameters["grid"].default == s21.BBP_C_GRID


def test_probe_picks_a_c_below_the_old_1e_minus_3_floor():
    # Strong low-rank structure that carries NO label information: held-out log-loss is best when the
    # probe is regularised hard, which the old grid (floor 1e-3) could not express.
    rng = np.random.default_rng(1)
    n, d, k = 300, 60, 3
    y = np.arange(n) % k
    X = rng.standard_normal((n, 5)) @ (2.0 * rng.standard_normal((5, d))) + rng.standard_normal((n, d))
    probe = s21.BBPAdaptiveProbe.fit(X, y, k)
    assert probe.C_ < 1e-3
    assert any(np.isclose(probe.C_, c) for c in s21.BBP_C_GRID)


# --------------------------------------------------------------------------- 2. class_weight

def _imbalanced_3class(seed=0, n=600, d=20):
    rng = np.random.default_rng(seed)
    y = rng.choice(3, size=n, p=[0.80, 0.15, 0.05])
    means = rng.standard_normal((3, d)) * 0.8
    return means[y] + rng.standard_normal((n, d)), y


def test_resolve_class_weight_accepts_known_forms_and_rejects_the_rest():
    assert s21.resolve_class_weight(None, 3) is None
    assert s21.resolve_class_weight("balanced", 3) == "balanced"
    assert s21.resolve_class_weight([1.0, 2.0, 4.0], 3) == {0: 1.0, 1: 2.0, 2: 4.0}
    assert s21.resolve_class_weight({0: 1, 2: 3}, 3) == {0: 1.0, 2: 3.0}
    for bad in ("uniform", [1.0, 2.0], [1.0, 0.0, 1.0], [1.0, np.nan, 1.0], {5: 1.0}, {0: -1.0}):
        with pytest.raises(ValueError):
            s21.resolve_class_weight(bad, 3)


def test_bbp_balanced_class_weight_raises_minority_recall():
    X, y = _imbalanced_3class()
    plain = s21.BBPAdaptiveProbe.fit(X, y, 3, C=1.0)
    bal = s21.BBPAdaptiveProbe.fit(X, y, 3, C=1.0, class_weight="balanced")
    assert plain.class_weight_ is None and bal.class_weight_ == "balanced"
    rec = lambda m: s21.check_prior_collapse_gate(y, m.predict(X))["per_class_recall"]
    assert rec(plain)[2] < s21.MIN_CLASS_RECALL <= rec(bal)[2]   # unweighted fit nearly ignores the 5% class


def test_bbp_inner_cv_runs_with_class_weight():
    X, y = _imbalanced_3class(seed=2)
    probe = s21.BBPAdaptiveProbe.fit(X, y, 3, class_weight=[1.0, 5.0, 16.0], cv_folds=3)
    assert probe.class_weight_ == {0: 1.0, 1: 5.0, 2: 16.0}
    assert any(np.isclose(probe.C_, c) for c in s21.BBP_C_GRID)


def test_torch_linear_probe_balanced_raises_minority_recall_and_validates():
    X, y = _imbalanced_3class(seed=3)
    Xt, yt = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.int64)

    def recalls(cw):
        W, b = base.fit_linear_probe_torch(Xt, yt, 3, C=1.0, device="cpu", class_weight=cw)
        pred = (Xt @ W.T + b).argmax(1).numpy()
        return s21.check_prior_collapse_gate(y, pred)["per_class_recall"]

    plain, bal = recalls(None), recalls("balanced")
    assert bal[2] > plain[2]
    with pytest.raises(ValueError):
        base.fit_linear_probe_torch(Xt, yt, 3, device="cpu", class_weight="uniform")
    with pytest.raises(ValueError):                           # class 3 has no rows: balanced weight is undefined
        base.fit_linear_probe_torch(Xt, yt, 4, device="cpu", class_weight="balanced")


def test_scorecard_config_wires_class_weight_into_both_heads():
    X, y = _imbalanced_3class(seed=4)
    Xn = ((X - X.mean(0)) / X.std(0)).astype(np.float32)
    cfg = sc.EvalConfig(bbp_inner_cv=3, class_weight="balanced")
    bbp = sc.fit_head("bbp_probe", Xn, y, 3, None, cfg, "cpu")
    assert bbp.model.class_weight_ == "balanced"
    W_bal, _ = sc.fit_head("linear_probe", Xn, y, 3, None, cfg, "cpu").fold()
    W_plain, _ = sc.fit_head("linear_probe", Xn, y, 3, None, sc.EvalConfig(bbp_inner_cv=3), "cpu").fold()
    assert not np.allclose(W_bal, W_plain)


# --------------------------------------------------------------------------- 3. gate

def test_gate_blocks_head_with_a_zero_recall_class():
    # beats the prior (0.85 > 0.5), not collapsed (max frac 0.55), yet never predicts class 2
    y_true = np.array([0] * 50 + [1] * 40 + [2] * 10)
    y_pred = y_true.copy()
    y_pred[90:] = 0
    y_pred[:5] = 1
    g = s21.check_prior_collapse_gate(y_true, y_pred)
    assert g["accuracy"] > g["majority_prior"] and g["collapsed"] is False and g["below_prior"] is False
    assert g["per_class_recall"][2] == 0.0 and g["min_class_recall"] == 0.0
    assert g["class_recall_below_min"] is True
    assert g["reasons"] == ["class_recall_below_min"]
    assert g["passed"] is False


def test_gate_passes_when_every_class_clears_the_recall_threshold():
    y_true = np.array([0] * 50 + [1] * 40 + [2] * 10)
    y_pred = y_true.copy()
    y_pred[91:] = 0                                            # class 2 recall 1/10 == threshold: passes
    g = s21.check_prior_collapse_gate(y_true, y_pred, train_priors=np.array([0.5, 0.4, 0.1]))
    assert g["min_class_recall"] == pytest.approx(0.10)
    assert g["class_recall_below_min"] is False
    assert g["reasons"] == [] and g["passed"] is True
    assert g["num_classes"] == 3 and g["min_class_recall_threshold"] == pytest.approx(0.10)


def test_gate_threshold_is_configurable_and_validated():
    y_true = np.array([0] * 50 + [1] * 40 + [2] * 10)
    y_pred = y_true.copy()
    y_pred[93:] = 0                                            # class 2 recall 0.3
    assert s21.check_prior_collapse_gate(y_true, y_pred)["passed"] is True
    g = s21.check_prior_collapse_gate(y_true, y_pred, min_class_recall=0.5)
    assert g["passed"] is False and "class_recall_below_min" in g["reasons"]
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError):
            s21.check_prior_collapse_gate(y_true, y_pred, min_class_recall=bad)


def test_gate_recall_rule_is_exempt_for_binary_tasks():
    y_true = np.array([0] * 70 + [1] * 30)
    y_pred = y_true.copy()
    y_pred[70:] = 0
    y_pred[:5] = 1                                             # class 1 recall 0, but acc 0.65 < 0.70 prior
    g = s21.check_prior_collapse_gate(y_true, y_pred)
    assert g["num_classes"] == 2 and g["class_recall_below_min"] is False
    assert g["reasons"] == ["below_prior"]


def test_gate_counts_classes_from_train_priors():
    # eval split holds only 2 of the 3 training classes: the task is still 3-class and the rule applies
    y_true = np.array([0] * 60 + [1] * 40)
    y_pred = np.zeros(100, dtype=int)
    y_pred[:45] = 2                                            # nothing collapsed, class 1 never predicted
    g = s21.check_prior_collapse_gate(y_true, y_pred, train_priors=np.array([0.3, 0.3, 0.4]))
    assert g["num_classes"] == 3 and g["class_recall_below_min"] is True and g["passed"] is False


# --------------------------------------------------------------------------- 4. selection

def _stat_from_gate(cid, family, cv_acc, params, y_true, y_pred):
    ok = bool(s21.check_prior_collapse_gate(y_true, y_pred)["passed"])
    return sc.CandStat(cid, family, None, 0.0, cv_acc, 0.01, params, sc.tie_rank(family), ok)


def test_zero_recall_head_is_never_selected_even_with_the_best_cv():
    y_true = np.array([0] * 50 + [1] * 40 + [2] * 10)
    zero = y_true.copy()
    zero[90:] = 0                                              # best accuracy, class 2 recall 0
    fair = y_true.copy()
    fair[80:95] = 0                                            # lower accuracy, class 2 recall 0.5
    stats = [_stat_from_gate("zero", "bbp_probe", 90.0, 10, y_true, zero),
             _stat_from_gate("fair", "adapter", 85.0, 5000, y_true, fair)]
    assert stats[0].gate_ok is False and stats[1].gate_ok is True
    sel = sc.breiman_select(stats)
    assert sel["chosen"].cid == "fair" and sel["n_admissible"] == 1


def test_no_admissible_head_means_no_champion():
    y_true = np.array([0] * 50 + [1] * 40 + [2] * 10)
    zero = y_true.copy()
    zero[90:] = 0
    stats = [_stat_from_gate("zero", "bbp_probe", 90.0, 10, y_true, zero)]
    sel = sc.breiman_select(stats)
    assert sel["chosen"] is None and sel["n_admissible"] == 0
    assert sc._selection_json(sel)["chosen_cid"] is None
