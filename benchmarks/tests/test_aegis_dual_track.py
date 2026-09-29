"""Aegis 2.0 dual-track scorecard (benchmarks/suites/aegis_dual_track.py).

Track A = every test row (public comparable score). Track B = Track A minus the "Needs Caution" rows;
Track B-strict = Track B minus "Unauthorized Advice". Covers: category parsing, fail-closed id lookup, that
denominators are computed from data and never shrink silently, that the gate abstentions stay in the
denominator, and (skipped only if the data is absent) the real 250-row test file against the real parquet.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "suites"))

import aegis_dual_track as adt  # noqa: E402


# ------------------------------------------------------------------ synthetic fixture

ROWS = [  # (id, gold label, categories string as stored in the parquet, margin)
    ("a1", "unsafe", "Violence", 3.0),                       # confident unsafe, right
    ("a2", "safe", "", -3.0),                                # confident safe, right
    ("a3", "unsafe", "Criminal Planning/Confessions", -0.4), # in band, high risk, ungated leak
    ("a4", "unsafe", "Needs Caution", -2.0),                 # NC, wrong (leak)
    ("a5", "safe", "Needs Caution", 0.5),                    # NC, in band
    ("a6", "unsafe", "Unauthorized Advice", 2.0),            # UA, right
    ("a7", "safe", "", 2.0),                                 # wrong (false block)
    ("a8", "unsafe", "Guns and Illegal Weapons,Threat", -1.5),  # high risk, confident safe -> released leak
]


@pytest.fixture()
def parquet(tmp_path):
    df = pd.DataFrame({
        "id": [r[0] for r in ROWS],
        "prompt_label": [r[1] for r in ROWS],
        "violated_categories": [r[2] if r[2] else None for r in ROWS],
    })
    path = tmp_path / "aegis.parquet"
    df.to_parquet(path)
    return path


def _records():
    return [{"id": f"aegis2-{r[0]}", "candidates": ["false", "true"],
             "ground_truth": "true" if r[1] == "unsafe" else "false"} for r in ROWS]


def _logits():
    m = np.array([r[3] for r in ROWS])
    return np.stack([-m / 2, m / 2], axis=1)   # margin = logit(true) - logit(false) = m


def _block(parquet, **kw):
    cats = adt.load_categories(parquet)
    logits = _logits()
    return adt.build_aegis_block(_records(), logits, logits.argmax(1), cats, **kw)


# ------------------------------------------------------------------ categories

def test_split_categories_handles_nan_empty_and_padding():
    assert adt.split_categories(None) == ()
    assert adt.split_categories(float("nan")) == ()
    assert adt.split_categories("") == ()
    assert adt.split_categories(" Violence ,Needs Caution") == ("Violence", "Needs Caution")


def test_load_categories_maps_ids(parquet):
    cats = adt.load_categories(parquet)
    assert cats["a3"] == ("Criminal Planning/Confessions",)
    assert cats["a8"] == ("Guns and Illegal Weapons", "Threat")
    assert cats["a2"] == ()


def test_missing_parquet_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="aegis"):
        adt.load_categories(tmp_path / "nope.parquet")


def test_test_id_absent_from_parquet_raises_instead_of_counting_as_not_excluded(parquet):
    cats = adt.load_categories(parquet)
    recs = _records() + [{"id": "aegis2-zzz", "candidates": ["false", "true"], "ground_truth": "true"}]
    logits = np.vstack([_logits(), [[0.0, 3.0]]])
    with pytest.raises(KeyError, match="zzz"):
        adt.build_aegis_block(recs, logits, logits.argmax(1), cats)


# ------------------------------------------------------------------ tracks

def test_track_a_is_the_full_set_and_matches_plain_argmax_accuracy(parquet):
    b = _block(parquet)
    a = b["tracks"]["track_a"]
    pred = _logits().argmax(1)
    gold = np.array([1 if r[1] == "unsafe" else 0 for r in ROWS])
    assert a["n"] == len(ROWS) and a["n_excluded"] == 0 and a["excluded_categories"] == []
    assert a["correct"] == int((pred == gold).sum())
    assert a["accuracy"] == round(100.0 * (pred == gold).mean(), 2)


def test_track_b_excludes_needs_caution_and_reports_the_denominators(parquet):
    t = _block(parquet)["tracks"]["track_b"]
    assert t["excluded_categories"] == ["Needs Caution"]
    assert (t["n_total"], t["n_excluded"], t["n"]) == (8, 2, 6)
    # a1,a2,a3,a6,a7,a8 remain; a3 (in-band unsafe, m<0) and a8 are wrong, a7 wrong -> 3/6
    assert t["correct"] == 3 and t["accuracy"] == 50.0
    assert len(t["wilson95"]) == 2


def test_track_b_strict_also_excludes_unauthorized_advice(parquet):
    t = _block(parquet)["tracks"]["track_b_strict"]
    assert set(t["excluded_categories"]) == {"Needs Caution", "Unauthorized Advice"}
    assert (t["n_excluded"], t["n"]) == (3, 5)


def test_expected_n_is_flagged_not_forced(parquet):
    a = _block(parquet)["tracks"]["track_a"]
    assert a["expected_n"] == 250 and a["n_matches_expected"] is False      # 8 rows, loud, not smoothed over
    assert _block(parquet)["tracks"]["track_b"]["expected_n"] == 225
    assert _block(parquet)["tracks"]["track_b_strict"]["expected_n"] == 221


def test_only_track_a_is_comparable_to_external_baselines(parquet):
    tr = _block(parquet)["tracks"]
    assert tr["track_a"]["comparable_to_external_baselines"] is True
    assert tr["track_b"]["comparable_to_external_baselines"] is False
    assert tr["track_b_strict"]["comparable_to_external_baselines"] is False


def test_row_count_mismatch_raises(parquet):
    cats = adt.load_categories(parquet)
    with pytest.raises(ValueError, match="rows"):
        adt.build_aegis_block(_records(), _logits()[:5], _logits().argmax(1)[:5], cats)


def test_non_binary_candidates_raise(parquet):
    cats = adt.load_categories(parquet)
    recs = _records()
    recs[0]["candidates"] = ["safe", "unsafe"]
    with pytest.raises(ValueError, match="candidates"):
        adt.build_aegis_block(recs, _logits(), _logits().argmax(1), cats)


def test_duplicate_test_ids_raise(parquet):
    cats = adt.load_categories(parquet)
    recs = _records()
    recs[1]["id"] = recs[0]["id"]
    with pytest.raises(ValueError, match="duplicate test id"):
        adt.build_aegis_block(recs, _logits(), _logits().argmax(1), cats)


def test_split_categories_handles_na_and_clean_tuples():
    assert adt.split_categories(None) == ()
    assert adt.split_categories(float("nan")) == ()
    assert adt.split_categories("<NA>") == ()
    assert adt.split_categories("nan") == ()
    assert adt.split_categories("Hate, Violence") == ("Hate", "Violence")


# ------------------------------------------------------------------ gate accounting

def test_gate_keeps_abstentions_in_the_denominator(parquet):
    g = _block(parquet, theta=1.0)["gate"]
    assert g["theta"] == 1.0
    a = g["views"]["deployable"]["track_a"]
    # in band (|m|<1): a3 (-0.4), a5 (0.5) -> 2 abstentions; confident: the other 6
    assert a["n"] == 8 and a["n_abstain"] == 2 and a["n_allow"] + a["n_block"] == 6
    assert a["coverage"] == 0.75
    # strict accuracy counts an abstention as wrong and stays over all 8 rows
    # confident rows: a1 ok, a2 ok, a4 wrong, a6 ok, a7 wrong, a8 wrong -> 3/8
    assert a["strict_correct"] == 3 and a["strict_accuracy"] == 37.5
    assert a["selective_accuracy"] == 50.0                                   # 3 of 6 covered


def test_gate_never_releases_more_unsafe_than_ungated_and_reports_both(parquet):
    a = _block(parquet)["gate"]["views"]["deployable"]["track_a"]
    # ungated releases (argmax says safe) gold-unsafe rows: a3, a4, a8 ; gated releases only a4, a8
    assert a["unsafe_released_ungated"] == 3 and a["unsafe_released_gated"] == 2
    assert a["unsafe_released_gated"] <= a["unsafe_released_ungated"]


def test_high_risk_leak_accounting_is_by_gold_category_slice_only(parquet):
    hr = _block(parquet)["gate"]["views"]["deployable"]["track_a"]["high_risk_gold_unsafe"]
    # gold-unsafe high-risk rows: a1 (Violence), a3 (Crime), a8 (Weapons)
    assert hr["n"] == 3 and hr["released_ungated"] == 2 and hr["released_gated"] == 1
    assert hr["categories"] == ["Criminal Planning/Confessions", "Guns and Illegal Weapons", "Violence"]


def test_oracle_view_is_labelled_and_blocks_high_risk_abstentions(parquet):
    views = _block(parquet)["gate"]["views"]
    o = views["oracle_high_risk"]
    assert o["deployable"] is False and "gold" in o["note"].lower()
    assert views["deployable"]["deployable"] is True
    a = o["track_a"]
    # a3 is high risk and in band: deployable ABSTAIN becomes BLOCK; a5 (not high risk) still abstains
    assert a["n_abstain"] == 1 and a["unsafe_released_gated"] == 2
    assert a["high_risk_gold_unsafe"]["released_gated"] == 1


def test_gate_metrics_present_for_every_track(parquet):
    v = _block(parquet)["gate"]["views"]["deployable"]
    assert set(v) >= {"track_a", "track_b", "track_b_strict"}
    assert v["track_b"]["n"] == 6 and v["track_b_strict"]["n"] == 5


# ------------------------------------------------------------------ markdown

def test_markdown_shows_both_tracks_and_flags_mismatch(parquet):
    md = "\n".join(adt.render_aegis_section(_block(parquet)))
    assert "Track A" in md and "Track B" in md and "Track B-strict" in md
    assert "n=8" in md and "n=6" in md and "excluded 2" in md
    assert "MISMATCH" in md                                  # 8 rows != 250 expected
    assert "ORACLE" in md and "TIER2_ESCALATE" in md


# ------------------------------------------------------------------ real data

def _real_paths():
    import grand_challenge_data as gd
    test_p = gd.TEST_DIR / "aegis_safety.jsonl"
    if not test_p.exists():
        fallback = Path(__file__).resolve().parents[1] / "data" / "full_13" / "aegis_safety.jsonl"
        if fallback.exists():
            test_p = fallback
    return test_p, Path(adt.DEFAULT_PARQUET)


_test_path, _parquet_path = _real_paths()


@pytest.mark.skipif(not (_test_path.exists() and _parquet_path.exists()),
                    reason=f"real Aegis test file ({_test_path}) or parquet ({_parquet_path}) absent; "
                           "set GC_TEST_DIR / AEGIS_RAW_PARQUET")
def test_real_aegis_counts_250_225_221():
    recs = [json.loads(line) for line in _test_path.open(encoding="utf-8")]
    cats = adt.load_categories(_parquet_path)
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(len(recs), 2)).astype(np.float32)
    b = adt.build_aegis_block(recs, logits, logits.argmax(1), cats)
    tr = b["tracks"]
    assert (tr["track_a"]["n"], tr["track_b"]["n"], tr["track_b_strict"]["n"]) == (250, 225, 221)
    assert (tr["track_b"]["n_excluded"], tr["track_b_strict"]["n_excluded"]) == (25, 29)
    assert all(t["n_matches_expected"] for t in tr.values())
    assert tr["track_a"]["n"] > tr["track_b"]["n"] > tr["track_b_strict"]["n"]


# ------------------------------------------------------------------ scorecard wiring

def test_scorecard_renders_the_dual_track_section_from_the_task_record(parquet):
    """The gate and both tracks must be reachable from the real scorecard renderer (no orphan module)."""
    import evaluate_full_13_grand_scorecard as sc
    rec = {"dataset": "Aegis 2.0", "n": 8, "n_expected_01png": 250, "nimble": 81.2, "jev": 80.4,
           "chosen_strategy": "linear_probe", "nested_cv_acc_chosen": 80.0, "accuracy": 37.5,
           "wilson95": [0.0, 1.0], "balanced_accuracy": 50.0, "macro_f1": 50.0, "win_marker": "-",
           "delta_vs_jev": -42.9, "linear_cv_acc": 80.0, "linear_cv_se": 1.0, "best_adapter_cv_acc": 80.0,
           "best_adapter_cv_se": 1.0, "best_adapter_rank": 32, "best_supcon_cv_acc": 80.0,
           "best_supcon_cv_se": 1.0, "best_supcon_rank": 32,
           "selection_ladder": {"adapter_challenge": {"cleared_1se": False, "collapse_ok": True, "won": False,
                                                       "delta_vs_champion": 0.0},
                                "supcon_challenge": {"cleared_1se": False, "collapse_ok": True, "won": False,
                                                     "delta_vs_champion": 0.0}},
           "decision_latency_ms": {"median": 0.1, "p95": 0.2}, "decision_latency_us": {"median": 100.0, "p95": 200.0},
           "aegis_dual_track": _block(parquet, baselines={"nimble": 81.2, "jev": 80.4})}
    report = {"title": "t", "generated_utc": "x", "command": "c",
              "aggregate": {"macro_avg_evaluated": 37.5, "micro_acc": 37.5, "phase1_reference": None,
                            "tasks_selected_linear": ["aegis_safety"], "tasks_selected_adapter": [],
                            "tasks_selected_supcon": []},
              "tasks": {"aegis_safety": rec}}
    md = sc.render_grand_scorecard_md(report)
    assert "## Aegis 2.0 dual-track" in md and "Track B-strict" in md and "vs Nimble" in md
    other = dict(rec)
    del other["aegis_dual_track"]
    assert "dual-track" not in sc.render_grand_scorecard_md({**report, "tasks": {"aegis_safety": other}})
