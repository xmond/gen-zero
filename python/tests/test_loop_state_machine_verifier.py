"""Completion claims and independently verified success must stay distinct."""
from unittest.mock import Mock

import pytest

from gen_zero.runtime.composite_decision import CompositeStepDecision
from gen_zero.runtime.loop_state_machine import CompositeExecutionLoop, LoopExecutionStatus


def run(action="finish", done_prob=0.0, verifier=None, dry_run=False):
    decision = CompositeStepDecision("safe", .99, action, .99, done_prob, 0.0)
    engine = Mock()
    engine.decide_step.return_value = decision
    actuate = Mock(return_value={"step": 1})
    report = CompositeExecutionLoop(decision_engine=engine, max_steps=1).execute_loop(
        initial_state={}, get_affordances_fn=lambda state: ["safe"],
        actuate_fn=actuate, verify_fn=verifier, dry_run=dry_run,
    )
    return report, actuate


@pytest.mark.parametrize("action,done", [("finish", 0.0), ("click", .90), ("finish", 1.0)])
def test_no_verifier_never_claims_verified_success(action, done):
    report, actuate = run(action, done)
    assert report.status is LoopExecutionStatus.UNVERIFIED_FINISH
    assert report.agent_finished is True
    assert report.verified is False
    assert report.to_dict()["verified"] is False
    assert report.to_dict()["agent_finished"] is True
    assert len(report.history) == 1
    actuate.assert_not_called()


@pytest.mark.parametrize("result", [False, None, "true", 1, {"verified": True}])
def test_only_boolean_true_is_verification(result):
    verifier = Mock(return_value=result)
    report, actuate = run(verifier=verifier)
    assert report.status is LoopExecutionStatus.ESCALATED
    assert report.verified is False
    verifier.assert_called_once_with({})
    actuate.assert_not_called()


def test_independent_verification_succeeds():
    verifier = Mock(return_value=True)
    report, actuate = run(verifier=verifier)
    assert report.status is LoopExecutionStatus.SUCCESS
    assert report.verified is True and report.agent_finished is True
    assert report.to_dict()["verified"] is True
    verifier.assert_called_once_with({})
    actuate.assert_not_called()


@pytest.mark.parametrize("action", ["finish", "click"])
def test_verifier_error_fails_closed(action):
    report, _ = run(action=action, verifier=Mock(side_effect=RuntimeError("failed")))
    assert report.status is LoopExecutionStatus.ESCALATED
    assert report.verified is False


def test_periodic_verification_is_distinct_from_agent_finish():
    report, actuate = run(action="click", verifier=Mock(return_value=True))
    assert report.status is LoopExecutionStatus.SUCCESS
    assert report.verified is True and report.agent_finished is False
    actuate.assert_called_once()


@pytest.mark.parametrize("result", [False, None, "true", 1])
def test_periodic_check_rejects_truthy_non_boolean(result):
    report, _ = run(action="click", verifier=Mock(return_value=result))
    assert report.status is LoopExecutionStatus.MAX_STEPS_EXCEEDED
    assert report.verified is False


def test_dry_run_never_claims_verification():
    verifier = Mock(return_value=True)
    report, actuate = run(verifier=verifier, dry_run=True)
    assert report.status is LoopExecutionStatus.DRY_RUN
    assert report.verified is False
    verifier.assert_not_called()
    actuate.assert_not_called()
