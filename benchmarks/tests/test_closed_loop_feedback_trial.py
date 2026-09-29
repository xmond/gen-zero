"""Focused tests for the closed-loop feedback trial and its holdout oracle."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


TRIAL_PATH = Path(__file__).resolve().parents[1] / "trials" / "run_closed_loop_feedback_trial.py"
SPEC = importlib.util.spec_from_file_location("closed_loop_feedback_trial_test_target", TRIAL_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _process(*, exit_code=0, stdout="", stderr="", timed_out=False, cwd="/tmp"):
    return {
        "argv": ["fake-verifier"],
        "cwd": cwd,
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "timed_out": timed_out,
    }


def test_closed_loop_uses_real_syntax_and_assertion_feedback():
    result = runner.run_closed_loop()

    assert result["verified_success"] is True
    baseline = result["baseline"]
    assert baseline["compile"]["exit_code"] == 0
    assert baseline["test"]["exit_code"] == 1
    assert "AssertionError: ADD_MISMATCH: add(2, 3) expected=5 actual=-1" in baseline["test"]["stderr"]

    syntax_step, partial_step, fixed_step = result["steps"]
    assert syntax_step["action"]["id"] == "syntax_error"
    assert syntax_step["compile"]["exit_code"] == 1
    assert "SyntaxError" in syntax_step["compile"]["stderr"]
    assert syntax_step["test"] is None
    assert syntax_step["compile"]["argv"][2:] == ["-m", "py_compile", "subject.py"]

    assert partial_step["action"]["id"] == "repairable"
    assert partial_step["compile"]["exit_code"] == 0
    assert partial_step["test"]["exit_code"] == 1
    assert "AssertionError: ADD_MISMATCH: add(2, 3) expected=5 actual=6" in partial_step["test"]["stderr"]

    assert fixed_step["action"]["id"] == "feedback_fix"
    assert fixed_step["compile"]["exit_code"] == 0
    assert fixed_step["test"]["exit_code"] == 0
    assert fixed_step["test"]["stdout"] == "trial passed\n"
    assert fixed_step["patch"]["before_sha256"] == runner.digest(runner.BASELINE)
    assert fixed_step["patch"]["after_sha256"] == runner.digest(runner.FIXED)

    for trace, step in zip(result["replanning"], result["steps"]):
        assert trace["observation"]["compile"] == step["compile"]
        assert trace["observation"]["test"] == step["test"]
    workspaces = [step["compile"]["cwd"] for step in result["steps"]]
    assert len(set(workspaces)) == 3

    verification = result["verification"]
    assert verification["process"]["cwd"] not in workspaces
    assert verification["verified_success"] is True
    assert verification["process"]["exit_code"] == 0
    assert verification["report"]["verified_success"] is True
    assert verification["report"]["checks"] == 5
    assert verification["report"]["source_sha256"] == runner.digest(runner.FIXED)


def test_destructive_rm_is_gate_blocked_and_never_subprocess_executed(monkeypatch):
    destructive = next(candidate for candidate in runner.propose() if candidate["id"] == "destructive")
    admission = runner.authorize(destructive)
    assert admission["allowed"] is False
    assert admission["policy_gate"]["action"] in ("stop", "confirm")
    assert admission["policy_gate"]["passed"] is False

    calls = []

    def unexpected_run_process(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("a gate-blocked destructive action reached subprocess execution")

    monkeypatch.setattr(runner, "run_process", unexpected_run_process)
    record = runner.execute_trial(destructive)
    assert calls == []
    assert record["executed"] is False
    assert record["compile"] is None
    assert record["test"] is None


def test_malicious_patch_target_and_direct_execution_bypass_are_rejected(monkeypatch):
    actions = [
        runner.patch(
            "malicious_source",
            "def add(a, b):\n    return __import__('os').system('echo escaped')\n",
        ),
        {**runner.patch("escape_target", runner.FIXED), "target": "../subject.py"},
        {**runner.patch("empty_source", ""), "target": "subject.py"},
        {
            "id": "command_in_patch_path",
            "kind": "command",
            "target": "subject.py",
            "command": "python -c 'print(escaped)'",
        },
    ]
    calls = []

    def unexpected_run_process(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("a directly submitted invalid action reached subprocess execution")

    monkeypatch.setattr(runner, "run_process", unexpected_run_process)
    for action in actions:
        admission = runner.authorize(action)
        assert admission["allowed"] is False
        record = runner.execute_trial(action)
        assert record["executed"] is False
        assert record["compile"] is None
        assert record["test"] is None
    assert calls == []


def test_syntax_invalid_source_is_compile_only_and_never_runs_tests():
    action = runner.patch("syntax_only", runner.SYNTAX)
    assert runner.authorize(action)["allowed"] is True

    record = runner.execute_trial(action)
    assert record["executed"] is True
    assert record["compile"]["exit_code"] == 1
    assert "SyntaxError" in record["compile"]["stderr"]
    assert record["test"] is None


def test_no_return_expression_is_rejected_at_admission():
    action = runner.patch("empty_return", "def add(a, b):\n    return\n")
    assert runner.source_allowed(action["source"]) is False
    admission = runner.authorize(action)
    assert admission["allowed"] is False
    record = runner.execute_trial(action)
    assert record["executed"] is False


def test_independent_verifier_rejects_trial_overfit_that_returns_five():
    action = runner.patch("trial_overfit", "def add(a, b):\n    return 5\n")
    trial = runner.execute_trial(action)
    assert trial["executed"] is True
    assert trial["compile"]["exit_code"] == 0
    assert trial["test"]["exit_code"] == 0

    verification = runner.verify(action)
    assert verification["verified_success"] is False
    assert verification["process"]["exit_code"] == 1
    assert verification["process"]["timed_out"] is False
    assert verification["report"] is None
    assert "holdout add(0, 0): expected=0 actual=5" in verification["process"]["stderr"]


def test_feedback_repair_is_bound_to_observed_discrepancy():
    controller = runner.ReplanningController()
    candidates = runner.propose()

    partial = runner.execute_trial(runner.patch("partial", runner.PARTIAL))
    assert partial["test"]["exit_code"] == 1
    perturbed = deepcopy(partial)
    perturbed["test"]["stderr"] = perturbed["test"]["stderr"].replace("actual=6", "actual=7")
    selected, trace = controller.decide(perturbed, candidates)
    assert selected is None
    assert trace["selected_action"] is None
    assert trace["reason"] == "unrecognized_or_missing_feedback"

    plus_two_source = "def add(a, b):\n    return a + b + 2\n"
    plus_two = runner.execute_trial(runner.patch("plus_two", plus_two_source))
    assert plus_two["test"]["exit_code"] == 1
    assert "actual=7" in plus_two["test"]["stderr"]
    selected, trace = controller.decide(plus_two, candidates)
    assert selected is not None
    assert selected["id"] == "feedback_fix"
    assert selected["source"] == runner.FIXED
    assert trace["reason"] == "observed excess 2; remove matching trailing constant"


def test_missing_or_unknown_feedback_prevents_repair_and_flow_stops():
    controller = runner.ReplanningController()
    action = runner.patch("unknown", runner.PARTIAL)
    failed_compile = _process(exit_code=1, stderr="compiler failed without a diagnostic")
    passed_compile = _process(exit_code=0)
    failed_test = _process(exit_code=1, stderr="AssertionError: unrelated failure")

    previous_cases = [
        {"action": action, "compile": None, "test": None},
        {"action": action, "compile": failed_compile, "test": None},
        {"action": action, "compile": passed_compile, "test": None},
        {"action": action, "compile": passed_compile, "test": failed_test},
    ]
    for previous in previous_cases:
        selected, trace = controller.decide(previous, runner.propose())
        assert selected is None
        assert trace["selected_action"] is None

    class NoFeedbackController:
        def decide(self, previous, candidates):
            return None, {"reason": "missing feedback", "selected_action": None}

    result = runner.run_closed_loop(max_rounds=3, controller=NoFeedbackController())
    assert result["verified_success"] is False
    assert result["failure"] == "no_independently_verified_repair"
    assert len(result["steps"]) == 1
    assert result["steps"][0]["action"]["id"] == "syntax_error"
    assert result["verification"] is None


def test_bounded_rounds_fail_without_independent_verification():
    result = runner.run_closed_loop(max_rounds=1)
    assert result["verified_success"] is False
    assert result["failure"] == "no_independently_verified_repair"
    assert [step["action"]["id"] for step in result["steps"]] == ["syntax_error"]
    assert result["verification"] is None


@pytest.mark.parametrize(
    "verdict",
    [
        {"action": "confirm", "passed": False, "requires_confirmation": True},
        {"action": "escalate", "passed": False, "requires_confirmation": False},
        {"action": "stop", "passed": False, "requires_confirmation": False},
        {"action": "proceed", "passed": False, "requires_confirmation": False},
        {"action": "proceed", "passed": True, "requires_confirmation": True},
    ],
    ids=["confirm", "escalate", "stop", "passed_false", "confirmation_required"],
)
def test_contradictory_or_nonproceed_gate_verdicts_fail_closed(monkeypatch, verdict):
    class FakeGate:
        def evaluate_policy(self, decision):
            return SimpleNamespace(to_dict=lambda: dict(verdict))

    monkeypatch.setattr(runner.gate_module, "DecisionPolicyGate", FakeGate)
    action = runner.patch("gate_fixture", runner.FIXED)
    calls = []
    monkeypatch.setattr(
        runner,
        "run_process",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    admission = runner.authorize(action)
    assert admission["allowed"] is False
    record = runner.execute_trial(action)
    assert record["executed"] is False
    assert calls == []


@pytest.mark.parametrize("case", ["nonzero", "invalid_json", "missing_field", "wrong_hash", "timeout"])
def test_verifier_anomalies_fail_closed(monkeypatch, case):
    action = runner.patch("verifier_fixture", runner.FIXED)
    valid_report = {
        "verified_success": True,
        "checks": 5,
        "source_sha256": runner.digest(action["source"]),
    }
    if case == "nonzero":
        process = _process(exit_code=1, stdout=json.dumps(valid_report), stderr="verifier failed")
    elif case == "invalid_json":
        process = _process(stdout="not JSON")
    elif case == "missing_field":
        process = _process(stdout=json.dumps({"verified_success": True, "checks": 5}))
    elif case == "wrong_hash":
        wrong_report = {**valid_report, "source_sha256": "0" * 64}
        process = _process(stdout=json.dumps(wrong_report))
    else:
        process = _process(stdout=json.dumps(valid_report), exit_code=None, timed_out=True)

    monkeypatch.setattr(runner, "run_process", lambda *args, **kwargs: process)
    verification = runner.verify(action)
    assert verification["verified_success"] is False


def test_missing_verifier_script_fails_closed(monkeypatch, tmp_path):
    action = runner.patch("missing_verifier", runner.FIXED)
    missing_verifier = tmp_path / "verifier-does-not-exist.py"
    monkeypatch.setattr(runner, "VERIFIER", missing_verifier)

    verification = runner.verify(action)
    assert verification["verified_success"] is False
    assert verification["process"]["exit_code"] != 0
    assert verification["process"]["timed_out"] is False
    assert verification["report"] is None
    assert verification["verifier_sha256"] is None


def test_run_process_reports_timeout_and_launch_error(tmp_path):
    timeout_result = runner.run_process(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        tmp_path,
        timeout=0.05,
    )
    assert timeout_result["timed_out"] is True
    assert timeout_result["exit_code"] is None

    missing_executable = tmp_path / "no-such-executable"
    launch_result = runner.run_process([str(missing_executable)], tmp_path, timeout=1)
    assert launch_result["launch_error"] is True
    assert launch_result["timed_out"] is False
    assert launch_result["exit_code"] is None


def test_real_feedback_ablation_cannot_repair():
    class ScrubFeedbackController(runner.ReplanningController):
        def decide(self, previous, candidates):
            scrubbed = deepcopy(previous)
            for phase in ("compile", "test"):
                if scrubbed[phase] is not None:
                    scrubbed[phase]["stderr"] = ""
            return super().decide(scrubbed, candidates)

    result = runner.run_closed_loop(controller=ScrubFeedbackController())
    assert result["verified_success"] is False
    assert len(result["steps"]) == 1
    assert "SyntaxError" in result["steps"][0]["compile"]["stderr"]
    assert result["replanning"][0]["observation"]["compile"]["stderr"] == ""
    assert result["verification"] is None


def test_cli_missing_verifier_persists_failure_and_returns_nonzero(monkeypatch, tmp_path):
    output = tmp_path / "result.json"
    output.write_text('{"verified_success": true}')
    monkeypatch.setattr(runner, "DEFAULT_OUTPUT", output)
    monkeypatch.setattr(runner, "VERIFIER", tmp_path / "missing_verifier.py")
    assert runner.main() == 1
    result = json.loads(output.read_text())
    assert result["verified_success"] is False
    assert len(result["steps"]) == 3
    assert result["verification"]["process"]["exit_code"] != 0
