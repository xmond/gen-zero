"""Spec 21 scorecard (benchmarks/suites/evaluate_spec21_scorecard.py). Covers, on synthetic tasks:

  1. logit-adjust shift: additive shift == spec21.logit_adjust, tau=0 is the identity, classes absent from the
     training rows can never win the argmax after adjustment;
  2. Breiman 1-SE selection on hand-built candidate stats: simplest-within-1-SE (not the best), the
     prior-collapse gate removes a higher-CV candidate, an all-inadmissible pool yields NO champion,
     complexity ordering bbp < linear < lda < adapter < supcon, tau != 0 costs one parameter;
  3. end-to-end evaluate_task_arrays: all five heads x three taus are fitted and reported, every candidate
     the ladder can pick comes from the OOF gate, folded heads reproduce their explicit scores;
  4. no test-label leakage: permuting y_test cannot change the chosen candidate;
  5. ablation arms are nested subsets of the same OOF pool; paired McNemar / bootstrap helpers;
  6. baseline comparison + report/markdown builders + npz loader.
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

import spec21_advanced_heads as s21  # noqa: E402
import evaluate_spec21_scorecard as sc  # noqa: E402

CFG = sc.EvalConfig(adapter_epochs=8, full_epochs=10, cv_folds=5, bbp_inner_cv=3, latency_rows=5)
RANKS = [4]


def _separable(seed=0, n_tr=240, n_te=200, d=48, k=3, shift=0.45):
    rng = np.random.default_rng(seed)
    means = rng.standard_normal((k, d)) * shift

    def draw(n):
        y = rng.integers(0, k, n)
        return means[y] + rng.standard_normal((n, d)), y

    (X_tr, y_tr), (X_te, y_te) = draw(n_tr), draw(n_te)
    return X_tr, y_tr, X_te, y_te


def _imbalanced_noise(seed=1, n_tr=240, n_te=200, d=48, minority=0.05):
    rng = np.random.default_rng(seed)

    def draw(n):
        y = (rng.random(n) < minority).astype(int)
        return rng.standard_normal((n, d)), y

    (X_tr, y_tr), (X_te, y_te) = draw(n_tr), draw(n_te)
    return X_tr, y_tr, X_te, y_te


@pytest.fixture(scope="module")
def sep_record():
    X_tr, y_tr, X_te, y_te = _separable()
    return sc.evaluate_task_arrays("synthetic_sep", X_tr, y_tr, X_te, y_te, 3, ranks=RANKS, device="cpu", cfg=CFG)


# --------------------------------------------------------------------------- 1. logit adjustment

def test_shift_equals_library_logit_adjust_and_tau_zero_is_identity():
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((7, 4))
    pri = np.array([0.5, 0.3, 0.15, 0.05])
    for tau in (0.5, 1.0):
        np.testing.assert_allclose(sc.adjust_scores(raw, pri, tau), s21.logit_adjust(raw, pri, tau))
    np.testing.assert_allclose(sc.adjust_scores(raw, pri, 0.0), raw)


def test_absent_class_never_wins_after_adjustment():
    raw = np.array([[0.0, 0.0, 5.0], [1.0, 0.2, 9.0]])
    pri = np.array([0.7, 0.3, 0.0])                 # class 2 has no training rows
    for tau in (0.0, 0.5, 1.0):
        assert not np.any(sc.adjust_scores(raw, pri, tau).argmax(axis=1) == 2)


def test_adjustment_boosts_minority_scores():
    raw = np.zeros((1, 2))
    adj = sc.adjust_scores(raw, np.array([0.9, 0.1]), 1.0)
    assert adj[0, 1] > adj[0, 0]


# --------------------------------------------------------------------------- 2. Breiman selection

def _stat(cid, family, cv, se, params, gate_ok=True, tie=0, tau=0.0, rank=None):
    return sc.CandStat(cid=cid, family=family, rank=rank, tau=tau, cv_acc=cv, cv_se=se,
                       params=params, tie_rank=tie, gate_ok=gate_ok)


def test_breiman_picks_simplest_within_one_se_of_best():
    stats = [_stat("lin", "linear_probe", 0.80, 0.02, 100),
             _stat("ada", "adapter", 0.83, 0.02, 500),
             _stat("sup", "supcon", 0.815, 0.02, 500, tie=1)]
    sel = sc.breiman_select(stats)
    assert sel["best_cid"] == "ada"
    assert sel["threshold"] == pytest.approx(0.81)
    assert sel["chosen"].cid == "ada"      # ada and sup both in the band, same params; ada wins on tie rank (0 < 1)
    stats[0] = _stat("lin", "linear_probe", 0.815, 0.02, 100)           # now inside the band and simplest
    assert sc.breiman_select(stats)["chosen"].cid == "lin"


def test_breiman_gate_removes_higher_cv_candidate():
    stats = [_stat("bad", "adapter", 0.95, 0.01, 500, gate_ok=False),
             _stat("ok", "linear_probe", 0.70, 0.01, 100)]
    sel = sc.breiman_select(stats)
    assert sel["chosen"].cid == "ok"
    assert sel["n_admissible"] == 1


def test_breiman_all_inadmissible_chooses_nothing():
    stats = [_stat("a", "adapter", 0.9, 0.01, 500, gate_ok=False),
             _stat("l", "linear_probe", 0.8, 0.01, 100, gate_ok=False)]
    sel = sc.breiman_select(stats)
    assert sel["chosen"] is None and sel["n_admissible"] == 0
    assert sel["best_cid"] is None and sel["within_cids"] == []


def test_breiman_rejects_empty_input():
    with pytest.raises(ValueError):
        sc.breiman_select([])


def test_complexity_order_and_tau_cost():
    D, K, r = 8192, 18, 64
    p = lambda fam, tau=0.0, rank=None: sc.head_param_count(fam, D, K, rank, tau, bbp_rank=190)
    assert p("bbp_probe") == K * (190 + 1)
    assert p("bbp_probe") < p("linear_probe") == p("lw_lda") == K * (D + 1)
    assert p("linear_probe") < p("adapter", rank=r) == p("supcon", rank=r)
    assert p("adapter", rank=r) == 2 * D * r + r + D + K * (D + 1)      # Spec 21 S6-B count, dense W_down
    assert p("linear_probe", tau=0.5) == p("linear_probe") + 1
    assert sc.tie_rank("linear_probe") < sc.tie_rank("lw_lda")
    assert sc.tie_rank("adapter") < sc.tie_rank("supcon")


# --------------------------------------------------------------------------- 3. end to end

def test_all_five_heads_and_three_taus_are_fitted_and_reported(sep_record):
    r = sep_record
    fams = {c["family"] for c in r["cv_candidates"]}
    assert fams == {"linear_probe", "bbp_probe", "lw_lda", "adapter", "supcon"}
    assert {c["tau"] for c in r["cv_candidates"]} == {0.0, 0.5, 1.0}
    assert len(r["cv_candidates"]) == 5 * 3                               # best rank per family and tau
    post = r["posthoc_test_all_candidates"]
    assert {c["cid"] for c in r["cv_candidates"]} == set(post)
    assert all(0.0 <= v["accuracy"] <= 100.0 for v in post.values())


def test_separable_task_is_learned_and_admitted(sep_record):
    r = sep_record
    assert r["accuracy"] > 80.0
    assert r["gate"]["passed"] and r["admitted"] is True
    assert r["selection"]["n_admissible"] >= 1
    assert r["chosen_cid"] in {c["cid"] for c in r["cv_candidates"] if c["gate_ok"]}
    lo, hi = r["wilson95"]
    assert lo <= r["accuracy"] <= hi and r["accuracy"] < 100.0


def test_1se_rule_is_applied_to_reported_cv_numbers(sep_record):
    sel = sep_record["selection"]
    pool = [c for c in sep_record["cv_candidates"] if c["gate_ok"]]
    best = next(c for c in pool if c["cid"] == sel["best_cid"])
    assert best["cv_acc"] == max(c["cv_acc"] for c in pool)               # ties go to the simpler candidate
    assert sel["threshold"] == pytest.approx(best["cv_acc"] - best["cv_se"], abs=1e-6)
    within = [c for c in pool if c["cv_acc"] >= sel["threshold"] - 1e-9]
    assert sep_record["chosen_cid"] in {c["cid"] for c in within}
    chosen = next(c for c in pool if c["cid"] == sep_record["chosen_cid"])
    assert (chosen["params"], chosen["tie_rank"]) == min((c["params"], c["tie_rank"]) for c in within)


def test_noise_task_with_extreme_prior_has_no_admissible_expert():
    X_tr, y_tr, X_te, y_te = _imbalanced_noise()
    r = sc.evaluate_task_arrays("synthetic_imb", X_tr, y_tr, X_te, y_te, 2, ranks=RANKS, device="cpu", cfg=CFG)
    assert r["selection"]["chosen_cid"] is None and r["selection"]["n_admissible"] == 0
    assert r["chosen_cid"] is None and r["accuracy"] is None and r["correct_mask"] is None
    assert r["admitted"] is False and r["no_champion_reason"] == "no_admissible_candidate"
    assert r["oof_gate_reasons"]
    assert r["majority_class_train_prior_acc"] > 90.0
    # the task is left out of every macro instead of scoring an inadmissible head
    report = sc.build_report({"synthetic_imb": r}, baseline=None, features_dir="synthetic", cfg=CFG,
                             ranks=RANKS, device="cpu", command="pytest")
    agg = report["aggregate"]
    assert agg["n_tasks_with_champion"] == 0 and agg["macro_all"] is None
    assert agg["tasks_without_champion"] == {"synthetic_imb": r["oof_gate_reasons"]}
    assert agg["tasks_not_admitted"]["synthetic_imb"]["no_admissible_candidate"] is True
    assert "NO chosen expert" in sc.render_md(report)


def test_bbp_without_signal_is_dropped_and_recorded_not_replaced():
    rng = np.random.default_rng(11)
    X_tr, X_te = rng.standard_normal((160, 40)), rng.standard_normal((100, 40))
    y_tr, y_te = rng.integers(0, 2, 160), rng.integers(0, 2, 100)
    r = sc.evaluate_task_arrays("synthetic_noise", X_tr, y_tr, X_te, y_te, 2, ranks=RANKS, device="cpu", cfg=CFG)
    assert "bbp_probe" in r["fit_errors"] and "Gavish-Donoho" in r["fit_errors"]["bbp_probe"]
    assert "bbp_probe" not in {c["family"] for c in r["cv_candidates"]}
    assert {c["family"] for c in r["cv_candidates"]} == {"linear_probe", "lw_lda", "adapter", "supcon"}


def test_test_gate_marks_below_prior_choice_not_admitted():
    # features carry signal for the wrong reason: train is 60:40, test is flipped to 30:70 with the same rule
    rng = np.random.default_rng(3)
    d = 32
    mu = rng.standard_normal(d) * 1.5

    def draw(n, p1):
        y = (rng.random(n) < p1).astype(int)
        return rng.standard_normal((n, d)) + np.outer(2 * y - 1, mu), y

    X_tr, y_tr = draw(220, 0.4)
    X_te, y_te = draw(160, 0.5)
    y_te = 1 - y_te                                                      # labels inverted at test time: acc << prior
    r = sc.evaluate_task_arrays("synthetic_flip", X_tr, y_tr, X_te, y_te, 2, ranks=RANKS, device="cpu", cfg=CFG)
    assert r["gate"]["below_prior"] is True
    assert r["admitted"] is False
    assert r["accuracy"] < r["majority_class_train_prior_acc"]


def test_folded_heads_reproduce_explicit_scores():
    X_tr, y_tr, X_te, _ = _separable(seed=5)
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    Xn, Xt = ((X_tr - mu) / sd).astype(np.float32), ((X_te - mu) / sd).astype(np.float32)
    for fam in ("linear_probe", "bbp_probe", "lw_lda"):
        h = sc.fit_head(fam, Xn, y_tr, 3, rank=None, cfg=CFG, device="cpu")
        raw = h.scores(Xt)
        assert raw.shape == (len(Xt), 3) and np.all(np.isfinite(raw))
        W, b = h.fold()
        np.testing.assert_allclose(Xt @ W.T + b, raw, rtol=1e-4, atol=1e-4)
        m = h.model
        if fam == "bbp_probe":                                           # explicit project-then-logistic path
            Z = (Xt.astype(np.float64) - m.mean_) / m.scale_
            np.testing.assert_allclose(Z @ m.components_.T @ m.coef_.T + m.intercept_, raw, rtol=1e-6, atol=1e-8)
        if fam == "lw_lda":                                              # explicit Mahalanobis discriminant
            Z = (Xt.astype(np.float64) - m.mean_) / m.scale_
            P = np.linalg.inv(m.covariance_matrix())
            expl = Z @ P @ m.class_means_.T - 0.5 * np.einsum("kd,de,ke->k", m.class_means_, P, m.class_means_) + np.log(m.priors)
            np.testing.assert_allclose(expl, raw, rtol=1e-6, atol=1e-8)


def test_compact_labels_are_int64_and_dense():
    present = np.array([True, False, True, True])
    out = sc.compact_labels(np.array([0, 2, 3, 2], dtype=np.int32), present)
    assert out.dtype == np.int64 and out.tolist() == [0, 1, 2, 1]


def test_head_fit_handles_class_missing_from_training_rows():
    X_tr, y_tr, X_te, _ = _separable(seed=6, k=3)
    keep = y_tr != 2
    Xn = ((X_tr[keep] - X_tr.mean(0)) / X_tr.std(0)).astype(np.float32)
    for fam in ("linear_probe", "bbp_probe", "lw_lda"):
        h = sc.fit_head(fam, Xn, y_tr[keep], 3, rank=None, cfg=CFG, device="cpu")
        assert h.present.tolist() == [True, True, False]
        pred = sc.adjust_scores(h.scores(((X_te - X_tr.mean(0)) / X_tr.std(0)).astype(np.float32)),
                                h.priors, 0.5).argmax(1)
        assert not np.any(pred == 2)


# --------------------------------------------------------------------------- 4. no test-label leakage

def test_selection_does_not_depend_on_test_labels(sep_record):
    X_tr, y_tr, X_te, y_te = _separable()
    rng = np.random.default_rng(99)
    r2 = sc.evaluate_task_arrays("synthetic_sep", X_tr, y_tr, X_te, rng.permutation(y_te), 3,
                                 ranks=RANKS, device="cpu", cfg=CFG)
    assert r2["chosen_cid"] == sep_record["chosen_cid"]
    assert r2["selection"] == sep_record["selection"]
    assert [(c["cid"], c["cv_acc"]) for c in r2["cv_candidates"]] == \
           [(c["cid"], c["cv_acc"]) for c in sep_record["cv_candidates"]]


# --------------------------------------------------------------------------- 5. arms + paired stats

def test_arms_are_nested_pools_over_the_same_oof(sep_record):
    arms = sep_record["arms"]
    assert set(arms) == {"legacy3_tau0", "plus_new_heads_tau0", "legacy3_plus_tau", "full"}
    legacy = arms["legacy3_tau0"]["chosen_cid"]
    assert legacy.split("|")[0].split("_r")[0] in {"linear_probe", "adapter", "supcon"} and legacy.endswith("tau=0.0")
    assert arms["legacy3_tau0"]["n_pool"] < arms["plus_new_heads_tau0"]["n_pool"] < arms["full"]["n_pool"]
    assert arms["legacy3_plus_tau"]["n_pool"] < arms["full"]["n_pool"]
    assert arms["full"]["chosen_cid"] == sep_record["chosen_cid"]
    for a in arms.values():
        assert len(a["correct_mask"]) == sep_record["n"]


def test_mcnemar_exact_counts_and_pvalue():
    a = np.array([1, 1, 1, 1, 0, 0, 1, 0], dtype=bool)
    b = np.array([1, 0, 0, 0, 1, 0, 1, 0], dtype=bool)
    m = sc.mcnemar_exact(a, b)
    assert (m["only_a"], m["only_b"], m["both"], m["neither"]) == (3, 1, 2, 2)
    assert m["p_two_sided"] == pytest.approx(0.625)                      # 2 * P(Bin(4, .5) <= 1) = 2 * 5/16
    same = sc.mcnemar_exact(a, a)
    assert same["p_two_sided"] == 1.0 and same["only_a"] == same["only_b"] == 0


def test_paired_bootstrap_macro_delta_detects_a_real_gap_and_is_zero_for_identical_arms():
    rng = np.random.default_rng(0)
    a = [rng.random(200) < 0.9 for _ in range(4)]
    b = [rng.random(200) < 0.6 for _ in range(4)]
    d = sc.paired_bootstrap_macro_delta(a, b, n_boot=2000, seed=0)
    assert d["ci95"][0] > 0.0 and d["point"] == pytest.approx(100 * (np.mean([x.mean() for x in a]) - np.mean([x.mean() for x in b])))
    z = sc.paired_bootstrap_macro_delta(a, a, n_boot=500, seed=0)
    assert z["point"] == 0.0 and z["ci95"] == [0.0, 0.0]


def test_macro_ci_contains_the_point_estimate():
    rng = np.random.default_rng(1)
    corr = [rng.random(150) < 0.7 for _ in range(5)]
    m = sc.bootstrap_macro_ci(corr, n_boot=2000, seed=0)
    assert m["ci95"][0] <= m["point"] <= m["ci95"][1]


# --------------------------------------------------------------------------- 6. baseline + report + loader

def _fake_baseline(rows):
    tasks = {t: {"chosen_strategy": "linear_probe", "accuracy": 70.0, "correct": int(0.7 * r["n"]), "n": r["n"],
                 "wilson95": sc.wilson(int(0.7 * r["n"]), r["n"])} for t, r in rows.items()}
    return {"aggregate": {"macro_avg_evaluated": 70.0}, "tasks": tasks}


def test_baseline_comparison_reports_winner_changes_and_deltas(sep_record):
    rows = {"synthetic_sep": sep_record}
    cmp = sc.compare_to_baseline(rows, _fake_baseline(rows))
    t = cmp["per_task"]["synthetic_sep"]
    assert t["baseline_strategy"] == "linear_probe" and t["new_strategy"] == sep_record["chosen_strategy"]
    assert t["strategy_changed"] == (sep_record["chosen_strategy"] != "linear_probe")
    assert t["delta_acc"] == pytest.approx(sep_record["accuracy"] - 70.0)
    assert cmp["macro_delta"] == pytest.approx(sep_record["accuracy"] - 70.0)
    assert cmp["baseline_macro"] == 70.0 and cmp["n_changed"] in (0, 1)


def test_report_and_markdown_are_built_only_from_recorded_numbers(sep_record):
    rows = {"synthetic_sep": sep_record}
    report = sc.build_report(rows, baseline=_fake_baseline(rows), features_dir="synthetic", cfg=CFG,
                             ranks=RANKS, device="cpu", command="pytest")
    assert report["aggregate"]["macro_all"] == pytest.approx(sep_record["accuracy"])
    assert report["aggregate"]["n_admitted"] == 1
    import json
    json.loads(json.dumps(report))                                       # every recorded number is JSON-serialisable
    md = sc.render_md(report)
    assert "synthetic_sep" in md and "Wilson" in md and "NOT verified" in md
    assert f"{sep_record['accuracy']:.2f}" in md


def test_load_features_reads_npz_written_to_disk(tmp_path):
    p = tmp_path / "massive_en.npz"
    X_tr, y_tr, X_te, _ = _separable(seed=7)
    np.savez(p, train_full=X_tr.astype(np.float32), train_label=y_tr, test_full=X_te.astype(np.float32),
             cands=np.zeros((3, 2)), info_json=np.array('{"variant": "syn"}'))
    d = sc.load_features(str(p))
    assert d["X_train"].shape == X_tr.shape and d["K"] == 3 and d["variant"] == "syn"
    assert d["X_train"].dtype == np.float32 and d["y_train"].dtype == np.int64
