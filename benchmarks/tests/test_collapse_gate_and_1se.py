"""Spec 19 S6.5 honesty gates: majority-class collapse detection + the 1-SE selector.

Covers benchmark_sota_ensemble.collapse_stats (balanced_accuracy, macro_f1, the
`collapsed` flag and its exact threshold), the build_report split of
tasks_beating_best_01png into a non-collapsed and a collapsed list, the
render_md COLLAPSED marker, and a regression pin on select_one_se for the
concrete scenario Spec 19 S0.2 describes: a new candidate head that does not
beat the simplest linear probe by >= 1 standard error must lose to it.

All data here is synthetic and hand-derived (see comments); no text, no
01.PNG features, no real task files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "benchmarks" / "suites"))

import benchmark_sota_ensemble as bse  # noqa: E402
from sota_ensemble_report import render_md  # noqa: E402


# ------------------------------------------------------- collapse_stats: gate

def test_collapsed_head_that_always_predicts_one_class_is_flagged_and_scored_at_chance():
    """K=3, gold split 2/2/2, every prediction is class 0 (the civil_comments/helpsteer2 pattern:
    `pred_counts` all mass on one class). Hand-derived: balanced_accuracy = mean(1, 0, 0) = 1/3;
    macro_f1 = mean(0.5, 0, 0) = 1/6 (class 0: P=2/6, R=2/2, F1=2*(1/3)*1/(1/3+1)=0.5; classes
    1 and 2: never predicted, F1=0 under zero_division=0)."""
    gold = np.array([0, 0, 1, 1, 2, 2])
    pred = np.zeros(6, dtype=np.int64)
    cs = bse.collapse_stats(pred, gold, K=3)
    assert cs["collapsed"] is True
    assert cs["max_pred_class_frac"] == 1.0
    assert cs["pred_counts"] == [6, 0, 0] and cs["gold_counts"] == [2, 2, 2]
    assert cs["balanced_accuracy"] == pytest.approx(100 / 3, abs=0.01)
    assert cs["macro_f1"] == pytest.approx(100 / 6, abs=0.01)


def test_collapse_threshold_is_strictly_greater_than_95_percent():
    """96/100 on one class: collapsed. Exactly 95/100: NOT collapsed (Spec 19 S6.5 item 1 says
    `> 0.95`, not `>=`); this is the boundary the majority-reference-entering-1SE mechanism relies
    on to not fire on an honestly-close-to-balanced head."""
    gold = np.array([0] * 50 + [1] * 50)
    pred96 = np.array([0] * 96 + [1] * 4)
    assert bse.collapse_stats(pred96, gold, K=2)["collapsed"] is True
    pred95 = np.array([0] * 95 + [1] * 5)
    cs95 = bse.collapse_stats(pred95, gold, K=2)
    assert cs95["max_pred_class_frac"] == 0.95
    assert cs95["collapsed"] is False


def test_single_candidate_task_can_never_collapse():
    """K=1: every prediction is trivially 'the only class'; §6.5 item 1 requires class count >= 2."""
    pred = gold = np.zeros(10, dtype=np.int64)
    cs = bse.collapse_stats(pred, gold, K=1)
    assert cs["max_pred_class_frac"] == 1.0 and cs["collapsed"] is False


def test_imbalanced_dataset_balanced_accuracy_exposes_what_raw_accuracy_hides():
    """K=2, gold 90/10 imbalanced (the civil_comments shape). The head gets every majority-class
    row right and half the minority-class rows right: raw accuracy 95% looks excellent, but
    balanced_accuracy (mean of the two recalls) is only 75%. Not collapsed: predictions split
    95/5, at the boundary (not > 0.95). Hand-derived exactly, see comments."""
    gold = np.array([0] * 90 + [1] * 10)
    pred = np.array([0] * 90 + [1] * 5 + [0] * 5)     # 90 correct majority + 5 correct minority
    cs = bse.collapse_stats(pred, gold, K=2)
    assert cs["collapsed"] is False and cs["max_pred_class_frac"] == 0.95
    accuracy = 100 * float(np.mean(pred == gold))
    assert accuracy == pytest.approx(95.0)
    assert cs["balanced_accuracy"] == pytest.approx(75.0)                # mean(90/90, 5/10) = mean(1.0, 0.5)
    # class0: P=90/95, R=90/90=1 -> F1=2*(90/95)/(90/95+1); class1: P=5/5=1, R=5/10=0.5 -> F1=2/3
    f1_0 = 2 * (90 / 95) * 1.0 / (90 / 95 + 1.0)
    f1_1 = 2 * 1.0 * 0.5 / (1.0 + 0.5)
    assert cs["macro_f1"] == pytest.approx(100 * (f1_0 + f1_1) / 2, abs=0.01)
    assert cs["balanced_accuracy"] < accuracy - 15                       # the gap §6.5 exists to surface


def test_balanced_dataset_symmetric_confusion_makes_balanced_accuracy_equal_raw_accuracy():
    """K=3, gold perfectly balanced (30/30/30) with a symmetric confusion pattern (each class
    donates the same 5 misclassifications to the next). With equal class support, the support-
    weighted mean (= raw accuracy) and the unweighted mean (= balanced_accuracy) coincide exactly;
    the symmetric confusion also makes every per-class F1 equal, so macro_f1 matches too."""
    gold = np.array([0] * 30 + [1] * 30 + [2] * 30)
    pred = np.array([0] * 25 + [1] * 5 +          # class 0: 25 right, 5 mislabeled as 1
                     [1] * 25 + [2] * 5 +          # class 1: 25 right, 5 mislabeled as 2
                     [2] * 25 + [0] * 5)           # class 2: 25 right, 5 mislabeled as 0
    cs = bse.collapse_stats(pred, gold, K=3)
    accuracy = 100 * float(np.mean(pred == gold))
    assert cs["collapsed"] is False
    assert accuracy == pytest.approx(75 / 90 * 100, abs=0.01)
    assert cs["balanced_accuracy"] == pytest.approx(accuracy, abs=0.01)
    assert cs["macro_f1"] == pytest.approx(accuracy, abs=0.01)


# ---------------------------------------- build_report / render_md: the win gate

def _row(name: str, acc: float, best01png: float, collapsed: bool, balanced_acc: float, macro_f1: float,
         n: int = 100) -> dict:
    win_marker = "COLLAPSED(majority_collapse_win_excluded)" if collapsed else ("WIN" if acc > best01png else "-")
    return {
        "dataset": name, "n": n, "n_expected_01png": n, "nimble": best01png, "jev": best01png,
        "best_01png": best01png, "majority_class_train_prior_acc": None, "leakage_gate": {},
        "n_train_rows": 1000,
        "chosen_strategy": "single:lin_full", "nested_cv_acc_chosen": round(acc, 2),
        "correct": int(round(acc / 100 * n)), "accuracy": round(acc, 2), "wilson95": [round(acc, 2) - 1, round(acc, 2) + 1],
        "delta_vs_nimble": round(acc - best01png, 2), "delta_vs_jev": round(acc - best01png, 2),
        "delta_vs_best_01png": round(acc - best01png, 2),
        "balanced_accuracy": round(balanced_acc, 2), "macro_f1": round(macro_f1, 2),
        "max_pred_class_frac": 1.0 if collapsed else 0.5, "collapsed": collapsed, "win_marker": win_marker,
        "vs_prior_heads": {},
        "decision_latency_ms": {"median": 1.0, "p95": 2.0, "mean": 1.0, "n_experts_run": 1},
        "encoder_latency_ms_batch1": {}, "post_hoc_test_acc_per_expert_NOT_used_for_selection": {},
        "nested_cv_acc_all_strategies": {}, "pred_counts": [n, 0] if collapsed else [n // 2, n // 2],
        "gold_counts": [n // 2, n // 2],
    }


def test_build_report_splits_wins_by_collapse_and_render_md_marks_the_collapsed_one():
    rows = {
        "honest_win": _row("Honest Win", acc=90.0, best01png=85.0, collapsed=False,
                            balanced_acc=88.0, macro_f1=87.0),
        "collapsed_win": _row("Collapsed Win", acc=89.33, best01png=81.0, collapsed=True,
                               balanced_acc=50.0, macro_f1=47.2),
        "honest_loss": _row("Honest Loss", acc=70.0, best01png=85.0, collapsed=False,
                             balanced_acc=69.0, macro_f1=68.0),
    }
    rep = bse.build_report(rows, la0=[0.0, 0.0, 0.0])
    agg = rep["aggregate"]
    assert agg["tasks_beating_best_01png"] == ["honest_win"]
    assert agg["tasks_beating_best_01png_collapsed"] == ["collapsed_win"]
    assert agg["tasks_collapsed"] == ["collapsed_win"]
    assert "collapsed_win" not in agg["tasks_beating_best_01png"]

    md = render_md(rep)
    assert "COLLAPSED(majority_collapse_win_excluded)" in md
    assert "Majority-class collapse" in md
    assert "honest_win" not in md.split("Majority-class collapse")[1].split("\n")[0]  # only the collapsed task named there
    # the JSON report round-trips through the exact renderer path used by stage_eval
    import json
    json.dumps(rep, ensure_ascii=False)                # must not raise (e.g. numpy scalars leaking through)


# --------------------------------------------------- 1-SE selector: hardening
# Spec 19 S0.2 / S6.5 item 2: a new candidate head must beat the simplest linear probe already
# in the ensemble by >= 1 standard error on nested CV, or the 1-SE rule must reject it and fall
# back to the simplest linear head. select_one_se is exercised directly (as
# test_sota_ensemble.py's existing 1-SE tests do), with complexity tuples shaped like
# strategy_complexity's real output for a `single:linear` vs. a much heavier supervised head
# (e.g. a Spec-19-style residual adapter: ~8.4M params, matching S0.2's real numbers).

def _folds(*accs: float) -> np.ndarray:
    return np.array(accs, dtype=float) / 100


def test_new_candidate_head_inside_one_se_of_baseline_is_rejected_for_the_simplest_linear_head():
    strategies = ["single:lin_full", "single:new_head"]
    fold_acc = {"single:lin_full": _folds(84, 85, 84, 86, 84),      # mean 84.6
                "single:new_head": _folds(84, 86, 84, 86, 85)}      # mean 85.0 (nominally "better")
    complexity = {"single:lin_full": (0, 897 * 3), "single:new_head": (0, 8_400_000)}
    chosen, info = bse.select_one_se(strategies, fold_acc, complexity)
    assert info["top"] == "single:new_head"                         # new head has the higher raw mean
    assert set(info["band"]) == {"single:lin_full", "single:new_head"}   # but within 1 SE of it
    assert chosen == "single:lin_full"                               # so the simplest linear head wins


def test_new_candidate_head_that_clears_one_se_is_kept_despite_higher_complexity():
    """Counter-test: the gate is not "always pick the linear head" -- a real >= 1 SE gain wins."""
    strategies = ["single:lin_full", "single:new_head"]
    fold_acc = {"single:lin_full": _folds(70, 71, 70, 72, 71),      # mean 70.8, far below the band
                "single:new_head": _folds(84, 86, 84, 86, 85)}      # mean 85.0
    complexity = {"single:lin_full": (0, 897 * 3), "single:new_head": (0, 8_400_000)}
    chosen, info = bse.select_one_se(strategies, fold_acc, complexity)
    assert info["top"] == "single:new_head"
    assert info["band"] == ["single:new_head"]
    assert chosen == "single:new_head"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ----------------------------------- report wording follows the data, not literals
# Module B decoupling: no train-row cap and no Jev number is written into the prose.

import logging  # noqa: E402

import grand_challenge_data as gd  # noqa: E402
import sota_ensemble_experts as sx  # noqa: E402
from sota_ensemble_report import png_refs, train_rows_text  # noqa: E402


def _report(train_rows) -> dict:
    rows = {f"t{i}": _row(f"T{i}", acc=80.0, best01png=75.0, collapsed=False, balanced_acc=79.0, macro_f1=78.0)
            for i in range(len(train_rows))}
    for r, n in zip(rows.values(), train_rows):
        r["n_train_rows"] = n
    return bse.build_report(rows, la0=[0.0, 0.0, 0.0])


def test_train_rows_text_is_single_number_or_min_to_max():
    assert train_rows_text(_report([11247, 11247, 11247])) == "11,247"
    assert train_rows_text(_report([1000, 11247, 2500])) == "1,000 to 11,247"
    assert train_rows_text({"protocol": {"train_rows_per_task": {}}}) == "an unrecorded number of"


@pytest.mark.parametrize("train_rows,expect", [([11247, 11247], "11,247"), ([1000, 11247], "1,000 to 11,247")])
def test_render_md_never_claims_a_1000_row_cap_when_more_rows_were_used(train_rows, expect):
    md = render_md(_report(train_rows))
    assert "<=1000" not in md and "~1000" not in md
    assert f"supervised on {expect} leakage-gated public train rows per task" in md
    assert f"on {expect} train rows per task" in md


def test_verdict_and_headline_use_the_protocol_jev_reference_not_a_literal():
    rep = _report([1000, 1000])                    # macro 80.0 (every task 80.0)
    assert png_refs(rep)[0] == rep["protocol"]["png_reference_avg"]["jev"] == bse.PNG_AVG["jev"]
    assert "is above Jev (76.0%)" in render_md(rep)
    rep["protocol"]["png_reference_avg"] = {"nimble": 74.8, "jev": 88.5}   # move the bar above the 80.0 macro
    md = render_md(rep)
    assert "is below Jev (88.5%)" in md and "the SOTA target is NOT reached" in md
    assert "Jev 88.5%" in md and "76.0" not in md


# -------------------------------------------------- --raw-max-dim and friends

def _specs_for(monkeypatch, raw_max_dim: int, D: int = 100):
    monkeypatch.setattr(bse, "RAW_MAX_DIM", raw_max_dim)
    monkeypatch.setattr(bse, "_WARNED_SKIPS", set())
    pair_task = next(iter(gd.PAIR_FIELDS))
    fs = {"src": {"train_full": np.zeros((4, D), dtype=np.float32)}}
    return pair_task, [n for n, _, _ in bse.expert_specs(pair_task, fs)]


def test_raw_map_wider_than_raw_max_dim_is_skipped_with_an_explicit_warning(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="sota_ensemble"):
        task, names = _specs_for(monkeypatch, raw_max_dim=450)     # widths: full 100, pair 400, hybrid 500
    assert "src:lin_full" in names and "src:lin_pair" in names
    assert "src:lin_hybrid" not in names and "src:lin_hybrid_pca" in names   # PCA variant still runs
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(msgs) == 1
    assert "SKIPPED" in msgs[0] and "src:lin_hybrid" in msgs[0] and task in msgs[0]
    assert "500" in msgs[0] and "450" in msgs[0]


def test_no_warning_when_every_raw_map_fits(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger="sota_ensemble"):
        _, names = _specs_for(monkeypatch, raw_max_dim=bse.DEFAULT_RAW_MAX_DIM)
    assert {"src:lin_full", "src:lin_pair", "src:lin_hybrid"} <= set(names)
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_raw_max_dim_default_is_raised_and_precedence_is_cli_then_env_then_default(monkeypatch):
    assert bse.DEFAULT_RAW_MAX_DIM == 48000
    monkeypatch.delenv("GC_RAW_MAX_DIM", raising=False)
    assert bse.resolve_raw_max_dim(None) == 48000
    monkeypatch.setenv("GC_RAW_MAX_DIM", "20000")
    assert bse.resolve_raw_max_dim(None) == 20000
    assert bse.resolve_raw_max_dim(7000) == 7000
    with pytest.raises(ValueError):
        bse.resolve_raw_max_dim(0)


def _run_main(monkeypatch, *argv: str) -> None:
    for name in ("RAW_MAX_DIM", "N_FOLDS", "FOLD_SEED", "NESTED_SEED", "MOE_THREADS", "SOURCES", "RNN_SOURCES"):
        monkeypatch.setattr(bse, name, getattr(bse, name))      # restored after the test
    monkeypatch.setattr(bse, "stage_combine", lambda tasks, learned: None)
    monkeypatch.setattr(sys, "argv", ["prog", "combine", "--source", "src=/nonexistent", *argv])
    bse.main()


def test_cli_flags_override_raw_max_dim_folds_seeds_and_moe_threads(monkeypatch):
    monkeypatch.setenv("GC_RAW_MAX_DIM", "1234")
    _run_main(monkeypatch, "--raw-max-dim", "999", "--n-folds", "3", "--fold-seed", "11", "--nested-seed", "22",
              "--torch-threads", "6")
    assert (bse.RAW_MAX_DIM, bse.N_FOLDS, bse.FOLD_SEED, bse.NESTED_SEED, bse.MOE_THREADS) == (999, 3, 11, 22, 6)
    rep = _report([1000])
    assert rep["protocol"]["raw_max_dim"] == 999 and rep["protocol"]["n_folds"] == 3
    assert rep["protocol"]["fold_seed"] == 11 and rep["protocol"]["nested_seed"] == 22


def test_cli_raw_max_dim_falls_back_to_env_then_default(monkeypatch):
    monkeypatch.setenv("GC_RAW_MAX_DIM", "1234")
    _run_main(monkeypatch)
    assert bse.RAW_MAX_DIM == 1234
    monkeypatch.delenv("GC_RAW_MAX_DIM")
    _run_main(monkeypatch)
    assert bse.RAW_MAX_DIM == 48000 and bse.N_FOLDS == 5 and bse.FOLD_SEED == 20260924


def test_cli_rejects_nonsense_values(monkeypatch):
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, "--raw-max-dim", "0")
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, "--n-folds", "1")


def test_build_moe_passes_torch_threads_through_instead_of_a_hard_coded_one(monkeypatch):
    seen = {}

    class _Rec:
        def __init__(self, experts, router, n_threads, blas_threads):
            seen["n_threads"] = n_threads

    class _Router:
        names: list = []

    monkeypatch.setattr(bse, "MultiModelCausalMoE", _Rec)
    monkeypatch.setattr(bse, "MOE_THREADS", 6)
    bse.build_moe({"router": _Router()}, {})
    assert seen["n_threads"] == 6


# ------------------------------------------------- experts: grids / NEG / jobs

def test_cs_grid_env_override_and_validation(monkeypatch):
    monkeypatch.delenv("GC_CS_GRID", raising=False)
    assert sx.cs_grid() == pytest.approx(np.logspace(-4, 2, 7).tolist())
    monkeypatch.setenv("GC_CS_GRID", "0.5, 2")
    assert sx.cs_grid() == [0.5, 2.0]
    assert sx.cs_grid([3.0]) == [3.0]                         # explicit argument beats env
    monkeypatch.setenv("GC_CS_GRID", "1,-2")
    with pytest.raises(ValueError):
        sx.cs_grid()


def test_lr_n_jobs_argument_then_env_then_default(monkeypatch):
    monkeypatch.delenv("GC_LR_N_JOBS", raising=False)
    assert sx.lr_n_jobs() == sx.DEFAULT_LR_N_JOBS
    monkeypatch.setenv("GC_LR_N_JOBS", "2")
    assert sx.lr_n_jobs() == 2 and sx.lr_n_jobs(3) == 3
    with pytest.raises(ValueError):
        sx.lr_n_jobs(0)


def test_absent_class_logit_scales_with_real_logits():
    small = np.array([[-1.0, 2.0], [0.5, 1.0]])
    huge = small * 1e6
    assert sx.absent_class_logit(small) == pytest.approx(-1.0 - 1e4)
    assert sx.absent_class_logit(huge) == pytest.approx(-1e6 - 1e4)     # still below every real logit
    assert sx.absent_class_logit(huge) < huge.min()
    assert sx.absent_class_logit(small, gap=50.0) == pytest.approx(-51.0)
    assert sx.absent_class_logit(np.array([[3.0, 4.0]])) == pytest.approx(-1e4)   # all-positive: floor at 0


def test_linear_probe_uses_supplied_grid_jobs_and_adaptive_absent_class_logit():
    pytest.importorskip("sklearn")                      # LogisticRegressionCV; absent in the bare repo venv
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 30)
    X = rng.normal(size=(60, 6)).astype(np.float32) + y[:, None] * 2.0
    m = sx.LinearProbe.fit(X, y, K=3, pair=False, kind="full", pca_k=None, seed=1, Cs=[0.25], n_jobs=1, neg_gap=500.0)
    assert m.cfg["C"] == 0.25                                          # the only C offered
    b2 = float(m.a["b"][2])                                            # class 2 never seen in training
    F = m._map(X.astype(np.float64), False, "full", {k: v.astype(np.float64) for k, v in m.a.items()
                                                      if k in ("mu_full", "sd_full")})
    seen_logits = F @ m.a["W"][:2].T.astype(np.float64) + m.a["b"][:2].astype(np.float64)
    assert b2 == pytest.approx(min(seen_logits.min(), 0.0) - 500.0, rel=1e-3)
    assert np.all(m.scores(X).argmax(1) != 2)


def test_fractal_gate_grids_are_overridable_and_default_to_module_constants():
    rng = np.random.default_rng(1)
    logp = np.log(rng.dirichlet(np.ones(3), size=40))
    resid = rng.normal(size=(40, 3))
    y = rng.integers(0, 3, size=40)
    custom = sx.fit_fractal_gate(logp, resid, y, betas=(0.0, 3.0), margins=(0.3,), entropies=(0.8,))
    assert custom["beta"] in (0.0, 3.0) and custom["tau_m"] == 0.3 and custom["tau_h"] == 0.8
    default = sx.fit_fractal_gate(logp, resid, y)
    assert default["beta"] in sx.FRACTAL_BETAS and default["tau_m"] in sx.FRACTAL_MARGINS
    assert default == sx.fit_fractal_gate(logp, resid, y, betas=sx.FRACTAL_BETAS, margins=sx.FRACTAL_MARGINS,
                                          entropies=sx.FRACTAL_ENTROPIES)
    with pytest.raises(ValueError):
        sx.fit_fractal_gate(logp, resid, y, betas=())
