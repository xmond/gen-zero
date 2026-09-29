"""Policy arithmetic tests with explicit evidence, not a simulated text classifier."""
from dataclasses import replace
import math
import pytest

from gen_zero.gate.safety_gate import (
    PolicyAssessment, PolicyVerdictAction, SafetyGate, TwoTierPolicyGate,
)


@pytest.mark.parametrize("prompt,choice", [("", ""), ("请求", "动作"), ("طلب", "إجراء"), ("requête", "action")])
def test_policy_is_language_independent(prompt, choice):
    gate = TwoTierPolicyGate()
    evidence = PolicyAssessment(prompt, choice, 0.01, 0.02)
    assert gate.evaluate_choice(prompt, choice, 0.9, assessment=evidence).passed
    hard = replace(evidence, violations=("operation_denied",), reason="trusted policy fact")
    verdict = gate.evaluate_choice(prompt, choice, 0.99, assessment=hard)
    assert verdict.action == PolicyVerdictAction.STOP
    assert verdict.triggered_rules == ("operation_denied",)


@pytest.mark.parametrize("confidence,action", [
    (0.0, PolicyVerdictAction.STOP), (0.449, PolicyVerdictAction.STOP),
    (0.45, PolicyVerdictAction.CONFIRM), (0.65, PolicyVerdictAction.CONFIRM),
    (0.651, PolicyVerdictAction.PROCEED), (1.0, PolicyVerdictAction.PROCEED),
])
def test_confidence_boundaries(confidence, action):
    evidence = PolicyAssessment("p", "c", 0.0, 0.0)
    assert TwoTierPolicyGate().evaluate_choice("p", "c", confidence, assessment=evidence).action == action


@pytest.mark.parametrize("risk,uncertainty,action", [
    (0.19, 0, PolicyVerdictAction.PROCEED),
    (0.20, 0, PolicyVerdictAction.CONFIRM),
    (0.10, 0.10, PolicyVerdictAction.CONFIRM),
    (0.5, 0, PolicyVerdictAction.STOP),
    (0.1, 0.4, PolicyVerdictAction.STOP),
    (1.0, 1.0, PolicyVerdictAction.STOP),
])
def test_risk_and_uncertainty_boundaries(risk, uncertainty, action):
    evidence = PolicyAssessment("p", "c", risk, uncertainty)
    verdict = TwoTierPolicyGate().evaluate_choice("p", "c", 1, assessment=evidence)
    assert verdict.action == action
    assert verdict.risk_score == min(1, risk + uncertainty)


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf, -0.1, 1.1, True, "0.9", None])
def test_invalid_numeric_evidence_never_passes(invalid):
    gate = TwoTierPolicyGate()
    evidence = PolicyAssessment("p", "c", 0, 0)
    assert not gate.evaluate_choice("p", "c", invalid, assessment=evidence).passed
    for field in ("risk", "uncertainty"):
        verdict = gate.evaluate_choice("p", "c", 1, assessment=replace(evidence, **{field: invalid}))
        assert verdict.triggered_rules == ("INVALID_POLICY_ASSESSMENT",)


def test_missing_and_mismatched_evidence_never_passes():
    gate = TwoTierPolicyGate()
    assert gate.evaluate_choice("p", "c", 1).triggered_rules == ("MISSING_POLICY_ASSESSMENT",)
    for evidence in (PolicyAssessment("other", "c", 0, 0), PolicyAssessment("p", "other", 0, 0), {}):
        assert gate.evaluate_choice("p", "c", 1, assessment=evidence).triggered_rules == ("INVALID_POLICY_ASSESSMENT",)


def test_integration_and_prompt_evidence():
    gate = SafetyGate()
    evidence = PolicyAssessment("请求", "", 0, 0)
    assert gate.evaluate_prompt_safety("请求", confidence=0.9, assessment=evidence).passed
    assert gate.evaluate_decision("请求", "", 0.9, assessment=evidence).passed
    assert not gate.evaluate_prompt_safety("请求", confidence=0.9).passed


@pytest.mark.parametrize("options", [
    {"min_confidence": float("nan")}, {"stop_risk": True},
    {"min_confidence": 0.8}, {"confirm_risk": 0.6}, {"evaluator": 1},
])
def test_invalid_configuration_rejected(options):
    with pytest.raises((ValueError, TypeError)):
        TwoTierPolicyGate(**options)
