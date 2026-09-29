"""Fail-closed behaviour of the benchmark harness (latency, equivariance, error analysis, dataset fetch).

The stub "binaries" below are test doubles for the failure paths of the harness only. They never
produce numbers that end up in a committed report.
"""
import stat
import sys
import textwrap
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH / "suites"))
sys.path.insert(0, str(BENCH / "datasets"))

import error_analysis  # noqa: E402
import equivariance_suite as eqs  # noqa: E402
import fetch_real_datasets as frd  # noqa: E402
import latency_suite as lat  # noqa: E402


def _stub(tmp_path: Path, body: str) -> str:
    """Write an executable python stub that plays the `gen-zero reflex` CLI."""
    path = tmp_path / "gen-zero-stub"
    path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


FIRST_CANDIDATE_STUB = """
    import json, sys
    a = sys.argv
    cands = a[a.index("--candidates") + 1].split(",") if "--candidates" in a else ["proceed"]
    print(json.dumps({"meta": {"chosen_action": cands[0]}}))
"""


# ---------------------------------------------------------------- equivariance

def test_flips_are_really_counted(tmp_path):
    """A position-biased stub (always picks the first candidate) must show flips, not 0."""
    r = eqs.PermutationEquivarianceSuite(_stub(tmp_path, FIRST_CANDIDATE_STUB), num_trials=5, shuffles_per_trial=6).run()
    assert r["error_samples"] == 0 and r["status"] == "ok"
    assert r["valid_comparisons"] == 30
    assert r["observed_flips"] > 0
    assert r["flip_rate_percent"] == round(r["observed_flips"] / 30 * 100.0, 4)
    assert "llm_baseline_comparison" not in r and "mathematical_guarantee" not in r


@pytest.mark.parametrize("body", [
    'import sys\nprint("{}")',                                       # no meta.chosen_action
    'import sys\nprint("not json")',                                 # unparsable
    'import sys\nprint("{\\"meta\\": {\\"chosen_action\\": \\"\\"}}")',  # empty choice
    'import sys\nsys.exit(1)',                                       # refused decision
])
def test_all_bad_responses_raise_instead_of_reporting_zero_flips(tmp_path, body):
    suite = eqs.PermutationEquivarianceSuite(_stub(tmp_path, body), num_trials=2, shuffles_per_trial=3)
    with pytest.raises(RuntimeError, match="0 valid comparisons"):
        suite.run()


def test_partial_errors_are_reported_not_hidden(tmp_path):
    counter = tmp_path / "n"
    body = f"""
    import json, sys, pathlib
    c = pathlib.Path({str(counter)!r})
    n = int(c.read_text()) if c.exists() else 0
    c.write_text(str(n + 1))
    if n % 3 == 2:
        print("garbage")
    else:
        print(json.dumps({{"meta": {{"chosen_action": "execute_transfer"}}}}))
    """
    r = eqs.PermutationEquivarianceSuite(_stub(tmp_path, body), num_trials=3, shuffles_per_trial=4).run()
    assert r["error_samples"] > 0 and r["status"] == "partial_errors"
    assert r["observed_flips"] == 0
    assert r["valid_comparisons"] + r["error_samples"] + r["skipped_shuffles_after_base_failure"] >= 12
    assert r["error_details"] and "not JSON" in r["error_details"][0]["reason"]


def test_equivariance_missing_binary_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        eqs.PermutationEquivarianceSuite(str(tmp_path / "nope")).run()


# ---------------------------------------------------------------- latency

def test_latency_reports_only_measured_values(tmp_path):
    r = lat.LatencyBenchmarkSuite(_stub(tmp_path, FIRST_CANDIDATE_STUB), iterations=3).run()
    assert r["in_engine_reflex_latency_us"] == "not_measured"
    assert r["in_engine_mcts_latency_ms"] == "not_measured"
    row = r["competitors_comparison"][0]
    assert row["throughput_qps"] == r["cli_throughput_qps"]
    assert row["acc"] == "not_measured" and row["ece"] == "not_measured"
    assert row["provenance"] == "measured_this_run_cli_subprocess"
    assert len(r["competitors_comparison"]) == 1
    assert r["is_synthetic"] is True
    assert not any("Python Prototype" in c["model"] for c in r["competitors_comparison"])


def test_latency_missing_binary_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        lat.LatencyBenchmarkSuite(str(tmp_path / "nope"), iterations=1).run()


# ---------------------------------------------------------------- error analysis

def test_synthetic_generator_is_gone():
    assert not hasattr(error_analysis, "generate_realistic_benchmark_predictions")


def test_missing_real_predictions_raises_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # hide the repo-relative search paths
    target = tmp_path / "real_predictions.jsonl"
    with pytest.raises(FileNotFoundError, match="synthetic predictions are strictly forbidden per anti-cheat policy"):
        error_analysis.ErrorAnalysisSuite(predictions_path=target).load_dataset()
    assert not target.exists()


def test_external_store_file_is_not_silently_loaded(tmp_path, monkeypatch):
    """The old synthesizer mirrored fake data to ../gen-zero-eval-data; that path must not be searched."""
    work = tmp_path / "work"
    work.mkdir()
    ext = tmp_path / "gen-zero-eval-data"
    ext.mkdir()
    (ext / "real_predictions.jsonl").write_text('{"id": "x"}\n')
    (work / "eval-data").mkdir()
    (work / "eval-data" / "real_predictions.jsonl").write_text('{"id": "x"}\n')
    monkeypatch.chdir(work)
    with pytest.raises(FileNotFoundError):
        error_analysis.ErrorAnalysisSuite().load_dataset()


# ---------------------------------------------------------------- dataset fetch

def test_no_fixture_fallback_symbol():
    assert not hasattr(frd, "CURATED_FALLBACK_FIXTURES")


def test_download_failure_raises_not_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(frd, "HAS_PYARROW", True)
    fetcher = frd.DatasetFetcher(output_dir=tmp_path / "out", raw_cache_dir=tmp_path / "raw")

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(fetcher, "_get_parquet_download_url", boom)
    for source in ("auto", "huggingface"):
        with pytest.raises(RuntimeError, match="network down"):
            fetcher.fetch_task_samples("boolq", max_samples=3, source=source)


def test_missing_pyarrow_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(frd, "HAS_PYARROW", False)
    fetcher = frd.DatasetFetcher(output_dir=tmp_path / "out", raw_cache_dir=tmp_path / "raw")
    with pytest.raises(RuntimeError, match="pyarrow is required"):
        fetcher.fetch_task_samples("boolq", max_samples=3)


def test_curated_source_is_rejected(tmp_path):
    fetcher = frd.DatasetFetcher(output_dir=tmp_path / "out", raw_cache_dir=tmp_path / "raw")
    with pytest.raises(ValueError):
        fetcher.fetch_task_samples("boolq", max_samples=3, source="curated")


def _rec(i, task, gt, gp, gc, qp, qc, **extra):
    return {
        "id": str(i), "task": task, "input_text": "x", "ground_truth": gt,
        "gen_zero": {"prediction": gp, "confidence": gc, "latency_ms": 1.0, "correct": gp == gt},
        "qwen35_9b": {"prediction": qp, "confidence": qc, "latency_ms": 10.0, "correct": qp == gt},
        **extra,
    }


def test_error_analysis_report_contains_only_computed_statistics(tmp_path):
    import json

    src = tmp_path / "p.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in [
        _rec(1, "PAWS", "a", "a", 0.9, "b", 0.6),
        _rec(2, "PAWS", "b", "zzz", 0.55, "b", 1.0, category="syntactic_permutation"),
    ]))
    suite = error_analysis.ErrorAnalysisSuite(src)
    suite.load_dataset()
    analysis = suite.analyze()
    out = tmp_path / "r.md"
    suite.generate_markdown_report(analysis, out)
    text = out.read_text()

    for fabricated in ("+18.7%", "0.054", "99.4%", "0.892", "0.841", "0.925",
                       "Verified Production Impact", "Illustrative Failure Trace"):
        assert fabricated not in text
    # Slices absent from the input are labelled, not filled with 0.0%.
    assert "not measured" in text
    # Out-of-label predictions get their own column, not label[0].
    cm = analysis["confusion_matrices"]["PAWS"]
    assert cm["gen_zero_matrix"]["b"][error_analysis.ErrorAnalysisSuite.UNMAPPED_LABEL] == 1


@pytest.mark.parametrize("body", ['print("{}")', 'print("not json")',
    'print(\'{"meta": {"chosen_action": "ABSTAIN"}}\')',
    'print(\'{"meta": {"chosen_action": "proceed"}, "is_error": true}\')'])
def test_latency_rejects_zero_exit_non_decisions(tmp_path, body):
    with pytest.raises(RuntimeError, match="non-decision"):
        lat.LatencyBenchmarkSuite(_stub(tmp_path, body), iterations=1).run()


def test_equivariance_rejects_constant_non_candidate(tmp_path):
    body = 'print(\'{"meta": {"chosen_action": "invented"}}\')'
    with pytest.raises(RuntimeError, match="0 valid comparisons"):
        eqs.PermutationEquivarianceSuite(_stub(tmp_path, body), num_trials=1).run()


def test_error_analysis_recomputes_claimed_success(tmp_path):
    import json
    row = _rec(1, "PAWS", "a", "b", 0.9, "a", 0.8)
    row["gen_zero"]["correct"] = True
    row["qwen35_9b"]["correct"] = False
    src = tmp_path / "p.jsonl"
    src.write_text(json.dumps(row))
    suite = error_analysis.ErrorAnalysisSuite(src)
    suite.load_dataset()
    summary = suite.analyze()["dataset_summary"]
    assert summary["gen_zero_overall_acc"] == 0
    assert summary["qwen35_overall_acc"] == 100


@pytest.mark.parametrize("kind", ["synthetic", "metadata_synthetic", "duplicate", "nan", "negative_latency"])
def test_error_analysis_rejects_invalid_evidence_atomically(tmp_path, kind):
    import json
    good = _rec(1, "PAWS", "a", "a", 0.9, "b", 0.8)
    bad = _rec(2, "PAWS", "a", "a", 0.9, "b", 0.8)
    if kind == "synthetic":
        bad["is_synthetic"] = True
    elif kind == "metadata_synthetic":
        bad["metadata"] = {"is_synthetic": True}
    elif kind == "duplicate":
        bad["id"] = good["id"]
    elif kind == "nan":
        bad["gen_zero"]["confidence"] = float("nan")
    else:
        bad["gen_zero"]["latency_ms"] = -1
    src = tmp_path / "p.jsonl"
    src.write_text(json.dumps(good) + "\n" + json.dumps(bad))
    suite = error_analysis.ErrorAnalysisSuite(src)
    with pytest.raises(ValueError):
        suite.load_dataset()
    assert suite.records == []


def test_derived_paws_rows_are_explicitly_synthetic():
    import hashlib
    import json
    from rebuild_gsm8k_paws import minimal_pairs
    generated = minimal_pairs("p", "The house is blue.", "The house is blue.", "paraphrase")
    assert generated and all(r["is_synthetic"] is True for r in generated)
    path = BENCH / "data/aux/paws_minimal_pairs.jsonl"
    rows = [json.loads(line) for line in path.read_text().split("\n") if line]
    assert rows and all(r["is_synthetic"] is True for r in rows)
    manifest = json.loads((BENCH / "data/manifest.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["aux_files"]["aux/paws_minimal_pairs.jsonl"]["sha256"]


@pytest.mark.parametrize("criteria", ["invalid scalar", {"gold": "only answer"}, ["gold x", "gold y"]])
def test_bespoke_converter_cannot_derive_candidates_from_gold(tmp_path, criteria):
    import json
    from convert_new_benchmarks import convert_bespoke
    src = tmp_path / "source.jsonl"
    src.write_text(json.dumps({"input": {"questions": {"decision": {"criteria": criteria}}},
                               "reference": {"target": "gold"}}))
    with pytest.raises(ValueError):
        convert_bespoke(src, tmp_path / "converted.jsonl")


@pytest.mark.parametrize("relative_path", [
    "benchmarks/deprecated_unverified/real_eval_summary.json",
    "benchmarks/results/deep_adapter_cpu_training_report.json",
    "benchmarks/results/decision_foundation_eval_results.json",
    "benchmarks/results/world_model_mcts_ablation_report.json",
    "python/results/gen_zero/issue_76_constraint_compiler_benchmark_report.json",
    "python/results/gen_zero/issue_86_hamiltonian_benchmark_report.json",
    "python/results/gen_zero/qwen_foundation_benchmark_results.json",
    "python/results/gen_zero/web_agent_benchmark_results.json",
])
def test_historical_synthetic_artifacts_are_explicitly_flagged(relative_path):
    import json
    payload = json.loads((BENCH.parent / relative_path).read_text())
    assert payload["is_synthetic"] is True
