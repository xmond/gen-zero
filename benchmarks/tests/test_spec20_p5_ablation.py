"""Spec 20 P5: combined ablation & final grand scorecard (run_spec20_p5_ablation.py). Covers:
  1. wilson95(): matches an independently-derived (different algebraic route) reference solver, and
     basic interval properties (containment, monotonic widening as n shrinks, domain errors);
  2. collapsed(): the exact S8/evaluate_full_13_grand_scorecard.py gate formula;
  3. bootstrap_macro_ci(): reproducible under a fixed seed, contains the true mean, widens with more
     spread, and is NOT a Wilson interval (independent of any (k, n) pair);
  4. build_per_task_scorecard(): recomputes Wilson from raw counts and REFUSES (SystemExit) if the
     recomputed interval disagrees with a report's stored interval -- the anti-drift guard;
  5. build_axis_grid() on synthetic fixtures: every "not_evaluated" cell carries a `reason` and NO
     accuracy-shaped numeric field (the anti-fabrication regression guard: this is the one test that
     would fail if a future edit started inventing numbers for un-evaluated axis cells);
  6. build_report() + render_markdown() run end-to-end on a small synthetic phase4-like fixture and
     produce valid, self-consistent JSON/Markdown (selector distribution sums to n_tasks, etc.);
  7. the three real head/pooling latency benchmarks (bench_adapter_b, bench_dual_manifold,
     bench_pooling, bench_linear_probe) exercise the real production inference code paths
     (FoldedResidualAdapterBHead.scores / DualManifoldHead.scores / PartitionAnchorPooler.pool) at
     small synthetic shapes and return well-formed median_us/p95_us timing dicts.
Synthetic data throughout (fixture reports, random weights): this suite validates the report-generation
LOGIC, not any real accuracy claim -- run_spec20_p5_ablation.py's own report is the accuracy evidence.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SUITES = REPO / "benchmarks" / "suites"
sys.path.insert(0, str(SUITES))

import run_spec20_p5_ablation as p5  # noqa: E402


# --------------------------------------------------------------------------- wilson95

def _reference_wilson(k: int, n: int, z: float = 1.96) -> list:
    """Independent re-derivation: solve the Wilson quadratic (Newcombe 1998)
    (n + z^2) p^2 - (2k + z^2) p + k^2/n = 0 directly for its two roots, rather than using the
    closed-form center +/- halfwidth p5.wilson95 evaluates. Different algebra, same interval."""
    A = n + z * z
    B = -(2 * k + z * z)
    C = (k * k) / n
    roots = sorted(np.roots([A, B, C]))
    return [round(100 * roots[0], 2), round(100 * roots[1], 2)]


@pytest.mark.parametrize("k,n", [(50, 100), (5, 20), (1, 1), (0, 10), (10, 10), (289, 350), (1, 250)])
def test_wilson95_matches_independent_reference(k, n):
    got = p5.wilson95(k, n)
    ref = _reference_wilson(k, n)
    assert got == pytest.approx(ref, abs=0.02)


def test_wilson95_contains_point_estimate():
    lo, hi = p5.wilson95(37, 100)
    assert lo <= 37.0 <= hi


def test_wilson95_widens_as_n_shrinks_at_fixed_p():
    lo10, hi10 = p5.wilson95(5, 10)
    lo100, hi100 = p5.wilson95(50, 100)
    assert (hi10 - lo10) > (hi100 - lo100)


def test_wilson95_rejects_non_positive_n():
    with pytest.raises(ValueError):
        p5.wilson95(1, 0)
    with pytest.raises(ValueError):
        p5.wilson95(1, -5)


# --------------------------------------------------------------------------- collapse gate

@pytest.mark.parametrize("frac,K,expect", [
    (0.96, 2, True), (0.95, 2, False), (1.0, 18, True), (0.5, 2, False), (0.99, 1, False),
])
def test_collapsed_matches_s8_gate_formula(frac, K, expect):
    assert p5.collapsed(frac, K) is expect


# --------------------------------------------------------------------------- bootstrap macro CI

def test_bootstrap_macro_ci_reproducible_and_contains_mean():
    accs = [76.0, 82.5, 41.3, 88.9, 63.2, 90.0, 55.5, 70.1, 68.4, 80.1, 86.1, 41.2, 84.0]
    ci_a = p5.bootstrap_macro_ci(accs, n_resamples=5000, seed=0)
    ci_b = p5.bootstrap_macro_ci(accs, n_resamples=5000, seed=0)
    assert ci_a == ci_b  # deterministic under a fixed seed
    mean = float(np.mean(accs))
    assert ci_a[0] <= mean <= ci_a[1]


def test_bootstrap_macro_ci_widens_with_more_spread():
    tight = [70.0] * 12 + [71.0]
    wide = [10.0, 95.0] * 6 + [50.0]
    ci_tight = p5.bootstrap_macro_ci(tight, n_resamples=5000, seed=1)
    ci_wide = p5.bootstrap_macro_ci(wide, n_resamples=5000, seed=1)
    assert (ci_wide[1] - ci_wide[0]) > (ci_tight[1] - ci_tight[0])


def test_bootstrap_macro_ci_is_not_a_wilson_interval():
    """A macro average over heterogeneous tasks has no single (k, n); bootstrap_macro_ci must not
    silently degrade into wilson95 on some invented (k, n)."""
    accs = [76.06] * 13
    ci = p5.bootstrap_macro_ci(accs, n_resamples=2000, seed=0)
    assert ci[0] == pytest.approx(76.06, abs=0.01) and ci[1] == pytest.approx(76.06, abs=0.01)


# --------------------------------------------------------------------------- per-task scorecard + drift guard

def _fixture_phase4(wilson_override=None):
    task = {
        "dataset": "Fixture Task", "n": 100, "correct": 70, "accuracy": 70.0,
        "wilson95": wilson_override if wilson_override is not None else p5.wilson95(70, 100),
        "chosen_strategy": "linear_probe", "collapsed": False, "max_pred_class_frac": 0.3,
        "nimble": 65.0, "jev": 68.0, "delta_vs_nimble": 5.0, "delta_vs_jev": 2.0,
        "balanced_accuracy": 69.0, "macro_f1": 68.5,
    }
    return {
        "aggregate": {
            "macro_avg_13": 70.0, "micro_acc": 70.0, "delta_macro_vs_nimble": 5.0, "delta_macro_vs_jev": 2.0,
            "png_reference_avg": {"nimble": 65.0, "jev": 68.0},
            "tasks_beating_jev": [], "tasks_collapsed": [],
            "tasks_selected_linear": ["fixture_task"], "tasks_selected_adapter": [], "tasks_selected_supcon": [],
        },
        "tasks": {"fixture_task": task},
    }


def test_build_per_task_scorecard_recomputes_and_agrees():
    rows = p5.build_per_task_scorecard(_fixture_phase4())
    assert len(rows) == 1
    r = rows[0]
    assert r["task"] == "fixture_task" and r["correct"] == 70 and r["n"] == 100
    assert r["wilson95_recomputed"] == r["wilson95_stored"]


def test_build_per_task_scorecard_refuses_on_wilson_drift():
    bad = _fixture_phase4(wilson_override=[10.0, 20.0])  # nowhere near the real interval for 70/100
    with pytest.raises(SystemExit):
        p5.build_per_task_scorecard(bad)


def test_build_per_task_scorecard_flags_commercial_score_inside_our_ci():
    rows = p5.build_per_task_scorecard(_fixture_phase4())
    r = rows[0]
    lo, hi = p5.wilson95(70, 100)  # the fixture's own real (correct, n), independently recomputed here
    assert r["jev_inside_our_wilson95"] == (lo <= 68.0 <= hi)
    assert r["nimble_inside_our_wilson95"] == (lo <= 65.0 <= hi)


def test_build_headline_caveat_ci_note_names_which_reference_scores_are_inside():
    rows = [{"task": "x", "accuracy": 90.0}]
    agg = {"delta_macro_vs_jev": 1.0, "macro_avg_13": 76.0,
          "png_reference_avg": {"jev": 76.0, "nimble": 74.8}}
    mec = {"excluded_tasks": ["x"], "excluded_task_majority_class_prior_acc": {"x": 90.0},
          "n_tasks_remaining": 5, "ours": 70.0, "jev": 72.0, "nimble": 71.0,
          "delta_vs_jev": -2.0, "delta_vs_nimble": -1.0}

    both_inside = p5.build_headline_caveat(agg, mec, [70.0, 80.0], rows)
    assert "Jev" in both_inside and "Nimble" in both_inside and "not distinguishable" in both_inside

    neither_inside = p5.build_headline_caveat(agg, mec, [10.0, 20.0], rows)
    assert "neither Jev" in neither_inside and "nor" in neither_inside


# --------------------------------------------------------------------------- collapsed-task headline guard

def _fixture_phase4_with_collapsed_flip():
    """Two tasks: one normal, one collapsed where OUR score beats Jev/Nimble only because they
    score even worse on a degenerate majority-class task. Mirrors the real civil_comments shape."""
    normal = {
        "dataset": "Normal", "n": 200, "correct": 150, "accuracy": 75.0,
        "wilson95": p5.wilson95(150, 200), "chosen_strategy": "linear_probe", "collapsed": False,
        "max_pred_class_frac": 0.5, "majority_class_train_prior_acc": 50.0,
        "nimble": 80.0, "jev": 82.0, "delta_vs_nimble": -5.0, "delta_vs_jev": -7.0,
        "balanced_accuracy": 75.0, "macro_f1": 75.0,
    }
    collapsed = {
        "dataset": "Collapsed", "n": 100, "correct": 89, "accuracy": 89.0,
        "wilson95": p5.wilson95(89, 100), "chosen_strategy": "linear_probe", "collapsed": True,
        "max_pred_class_frac": 0.96, "majority_class_train_prior_acc": 92.0,
        "nimble": 70.0, "jev": 81.0, "delta_vs_nimble": 19.0, "delta_vs_jev": 8.0,
        "balanced_accuracy": 50.0, "macro_f1": 46.0,
    }
    return {
        "aggregate": {
            "macro_avg_13": round((75.0 + 89.0) / 2, 2), "micro_acc": 80.0,
            "delta_macro_vs_nimble": 7.0, "delta_macro_vs_jev": 0.5,
            "png_reference_avg": {"nimble": 75.0, "jev": 81.5},
            "tasks_beating_jev": ["collapsed"], "tasks_collapsed": ["collapsed"],
            "tasks_selected_linear": ["normal", "collapsed"], "tasks_selected_adapter": [],
            "tasks_selected_supcon": [],
        },
        "tasks": {"normal": normal, "collapsed": collapsed},
    }


def test_build_headline_caveat_says_flip_only_when_sign_actually_changes():
    rows = [{"task": "x", "accuracy": 90.0}]
    agg_positive = {"delta_macro_vs_jev": 1.0, "macro_avg_13": 76.0,
                   "png_reference_avg": {"jev": 76.0, "nimble": 74.8}}
    macro_excl_negative = {"excluded_tasks": ["x"], "excluded_task_majority_class_prior_acc": {"x": 90.0},
                            "n_tasks_remaining": 5, "ours": 70.0, "jev": 72.0, "nimble": 71.0,
                            "delta_vs_jev": -2.0, "delta_vs_nimble": -1.0}
    caveat_flip = p5.build_headline_caveat(agg_positive, macro_excl_negative, [50.0, 100.0], rows)
    assert "FLIPS from positive to negative" in caveat_flip

    macro_excl_still_positive = dict(macro_excl_negative, delta_vs_jev=0.5)
    caveat_no_flip = p5.build_headline_caveat(agg_positive, macro_excl_still_positive, [50.0, 100.0], rows)
    assert "FLIPS" not in caveat_no_flip and "stays positive" in caveat_no_flip

    caveat_none = p5.build_headline_caveat(agg_positive, None, [50.0, 100.0], rows)
    assert "No task was flagged collapsed" in caveat_none


def test_macro_excluding_collapsed_flips_sign_like_the_real_civil_comments_case():
    fixture = _fixture_phase4_with_collapsed_flip()
    unit_tests = {k: {"exit_code": 0, "summary": "n/a"} for k in
                 ("P1_partition_anchor_pooling", "P3_formulation_b_adapter", "P4_dual_manifold")}
    rows = p5.build_per_task_scorecard(fixture)
    axis_grid = p5.build_axis_grid(fixture, unit_tests)
    latency = _fixture_latency()
    report = p5.build_report(rows, fixture, axis_grid, latency, unit_tests)
    mec = report["macro_summary"]["macro_excluding_collapsed_tasks"]
    assert mec is not None
    assert mec["excluded_tasks"] == ["collapsed"]
    # overall (both tasks) our delta vs jev is positive (we "win"); excluding the collapsed task it must flip negative
    overall_delta_vs_jev = report["macro_summary"]["macro_avg_13_point_estimate"] - \
        report["macro_summary"]["png_reference_avg"]["jev"]
    assert overall_delta_vs_jev > 0
    assert mec["delta_vs_jev"] < 0
    caveat = report["final_verdict"]["headline_caveat"]
    assert "FLIPS from positive to negative" in caveat
    assert "collapsed" in caveat  # names the actual excluded task, not a hardcoded one


# --------------------------------------------------------------------------- axis grid: anti-fabrication guard

_ACCURACY_SHAPED_KEYS = {"accuracy", "macro_avg_13", "macro_avg_evaluated", "micro_acc", "wilson95",
                         "cv_acc", "nested_cv_acc_chosen", "delta_vs_nimble", "delta_vs_jev"}


def _walk_not_evaluated_cells(axis_grid: dict):
    for axis in axis_grid.values():
        for cell_name, info in axis.items():
            if info["status"].startswith("not_evaluated"):
                yield f"{cell_name}", info


def test_not_evaluated_cells_carry_a_reason_and_no_fabricated_accuracy():
    unit_tests = {
        "P1_partition_anchor_pooling": {"exit_code": 0, "summary": "38 passed in 2.0s"},
        "P3_formulation_b_adapter": {"exit_code": 0, "summary": "44 passed in 10.0s"},
        "P4_dual_manifold": {"exit_code": 0, "summary": "20 passed in 6.0s"},
    }
    axis_grid = p5.build_axis_grid(_fixture_phase4(), unit_tests)
    seen_not_evaluated = 0
    for cell_name, info in _walk_not_evaluated_cells(axis_grid):
        seen_not_evaluated += 1
        assert "reason" in info and len(info["reason"]) > 20, f"{cell_name} missing a real reason string"
        offending = _ACCURACY_SHAPED_KEYS & set(info.keys())
        assert not offending, f"{cell_name} is not_evaluated but carries accuracy-shaped keys {offending}"
    # the grid must actually contain not_evaluated cells for this guard to mean anything
    assert seen_not_evaluated >= 5


def test_evaluated_cells_carry_evidence_pointer():
    unit_tests = {k: {"exit_code": 0, "summary": "n/a"} for k in
                 ("P1_partition_anchor_pooling", "P3_formulation_b_adapter", "P4_dual_manifold")}
    axis_grid = p5.build_axis_grid(_fixture_phase4(), unit_tests)
    for axis in axis_grid.values():
        for cell_name, info in axis.items():
            if info["status"] in ("evaluated", "evaluated_implicitly"):
                assert "evidence" in info or "note" in info, f"{cell_name} evaluated with no evidence trail"


# --------------------------------------------------------------------------- end-to-end report + markdown

def _fixture_latency():
    return {
        "methodology": "fixture", "loadavg_1min": 0.0, "cpu_count": 1, "host_busy_caveat": "n/a",
        "k_values_observed_in_13_tasks": {}, "k_probe_points": [2],
        "adapter_formulation_a_D8192_rank64_shape_only": {"K2": {"median_us": 1.0, "p95_us": 2.0}},
        "adapter_formulation_b_D8192_rank64": {"K2": {"median_us": 1.0, "p95_us": 2.0}},
        "dual_manifold_DQ8192_DG2816": {"K2": {"median_us": 1.0, "p95_us": 2.0}},
        "linear_probe_D8192_fresh": {"K2": {"median_us": 1.0, "p95_us": 2.0}},
        "pooling_T1536_D4096": {"boundary": "fixture boundary note",
                                "partition_anchor_pooling": {"median_us": 1.0, "p95_us": 2.0},
                                "uniform_mean": {"median_us": 1.0, "p95_us": 2.0}},
    }


def test_build_report_and_render_markdown_end_to_end(tmp_path):
    fixture = _fixture_phase4()
    unit_tests = {
        "P1_partition_anchor_pooling": {"exit_code": 0, "summary": "38 passed in 2.0s"},
        "P3_formulation_b_adapter": {"exit_code": 0, "summary": "44 passed in 10.0s"},
        "P4_dual_manifold": {"exit_code": 0, "summary": "20 passed in 6.0s"},
    }
    rows = p5.build_per_task_scorecard(fixture)
    axis_grid = p5.build_axis_grid(fixture, unit_tests)
    latency = _fixture_latency()
    report = p5.build_report(rows, fixture, axis_grid, latency, unit_tests)

    # JSON round-trips
    json_path = tmp_path / "report.json"
    json_path.write_text(json.dumps(report, indent=2))
    reloaded = json.loads(json_path.read_text())
    assert reloaded["macro_summary"]["macro_avg_13_point_estimate"] == 70.0

    # selector distribution sums to the number of tasks in the fixture
    sd = report["one_se_selector_distribution"]
    assert sd["n_tasks_selected_sum_check"] == sd["n_tasks_total"] == 1

    # macro bootstrap CI is present and contains the point estimate for a degenerate 1-task fixture
    ci = report["macro_summary"]["macro_avg_13_task_level_bootstrap_ci95"]
    assert ci[0] == pytest.approx(70.0, abs=0.01) and ci[1] == pytest.approx(70.0, abs=0.01)

    # final verdict names the un-evaluated axes and never claims a number for them
    fv = report["final_verdict"]
    assert len(fv["not_yet_combinable_axes"]) >= 5
    for axis_name in fv["not_yet_combinable_axes"]:
        assert axis_name not in fv["only_real_end_to_end_combination"]

    md = p5.render_markdown(report)
    assert "not_evaluated" in md
    assert "## 8. Final verdict" in md
    md_path = tmp_path / "report.md"
    md_path.write_text(md)
    assert md_path.exists() and md_path.stat().st_size > 0


# --------------------------------------------------------------------------- real head/pooling latency benches

def test_build_per_task_scorecard_defaults_missing_decision_latency_to_none():
    rows = p5.build_per_task_scorecard(_fixture_phase4())  # fixture has no decision_latency_us key
    assert rows[0]["decision_latency_us_median"] is None
    assert rows[0]["decision_latency_us_p95"] is None


def test_bench_adapter_a_runs_real_head_and_times_it(monkeypatch):
    monkeypatch.setattr(p5, "_timeit", lambda fn, reps=2000: {"median_us": 0.0, "p95_us": 0.0, "n": _probe(fn)})
    out = p5.bench_adapter_a(D=16, K=3, rank=2, rng=np.random.default_rng(0))
    assert set(out) >= {"median_us", "p95_us"}


def test_bench_adapter_b_runs_real_head_and_times_it(monkeypatch):
    monkeypatch.setattr(p5, "_timeit", lambda fn, reps=2000: {"median_us": 0.0, "p95_us": 0.0, "n": _probe(fn)})
    out = p5.bench_adapter_b(D=16, K=3, rank=2, rng=np.random.default_rng(0))
    assert set(out) >= {"median_us", "p95_us"}


def _probe(fn):
    out = fn()
    assert np.all(np.isfinite(np.asarray(out)))
    return 1


def test_bench_dual_manifold_runs_real_head_and_times_it(monkeypatch):
    monkeypatch.setattr(p5, "_timeit", lambda fn, reps=2000: {"median_us": 0.0, "p95_us": 0.0, "n": _probe(fn)})
    out = p5.bench_dual_manifold(D_Q=16, D_G=8, K=3, rng=np.random.default_rng(0))
    assert set(out) >= {"median_us", "p95_us"}


def test_bench_linear_probe_runs_and_times_it(monkeypatch):
    monkeypatch.setattr(p5, "_timeit", lambda fn, reps=2000: {"median_us": 0.0, "p95_us": 0.0, "n": _probe(fn)})
    out = p5.bench_linear_probe(D=16, K=3, rng=np.random.default_rng(0))
    assert set(out) >= {"median_us", "p95_us"}


def test_bench_pooling_runs_real_pooler_and_uniform_mean(monkeypatch):
    monkeypatch.setattr(p5, "_timeit", lambda fn, reps=2000: {"median_us": 0.0, "p95_us": 0.0, "n": _probe(fn)})
    out = p5.bench_pooling(T=32, D=8, rng=np.random.default_rng(0))
    assert "partition_anchor_pooling" in out and "uniform_mean" in out


def test_bench_adapter_b_output_shape_is_real_not_mocked():
    """No monkeypatch here: runs the real head at a tiny shape with a tiny rep count via a direct
    call to the underlying scores(), independent of the timing wrapper, to prove the head itself
    (not a stand-in) is what gets timed in bench_adapter_b."""
    rng = np.random.default_rng(0)
    D, K, rank = 12, 4, 2
    f32 = np.float32
    arrays = {
        "mu_full": np.zeros(D, f32), "sd_full": np.ones(D, f32),
        "W0": (rng.standard_normal((K, D)) * 0.01).astype(f32), "b0": np.zeros(K, f32),
        "C": (rng.standard_normal((K, rank)) * 0.01).astype(f32),
        "U": (rng.standard_normal((rank, D)) * 0.01).astype(f32), "a": np.zeros(rank, f32),
    }
    cfg = {"K": K, "pair": False, "in_dim": D, "D": D, "rank": rank, "head_type": "adapter_b"}
    head = p5.eh.FoldedResidualAdapterBHead(arrays, cfg)
    x = rng.standard_normal(D).astype(f32)
    out = head.scores(x)
    assert out.shape == (K,) and np.all(np.isfinite(out))
