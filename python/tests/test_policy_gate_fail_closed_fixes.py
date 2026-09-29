"""Direct regressions for the audit's missing-schema and whitelist bypasses."""
from dataclasses import replace
import math

import pytest

from gen_zero.gate.policy_gate import DecisionPolicyGate, DomainRiskProfile, PolicyVerdictAction
from gen_zero.runtime.composite_decision import CompositeStepDecision


def valid(**changes):
    return dict(action="click", target="safe", context={}, confidence=0.99, risk=0.0, **changes)


def assert_stopped(decision, profile=None, **kwargs):
    result = DecisionPolicyGate(profile).evaluate_policy(decision, **kwargs)
    assert result.action is PolicyVerdictAction.STOP
    assert result.passed is False
    assert result.requires_confirmation is False
    assert math.isfinite(result.confidence) and math.isfinite(result.risk_score)
    return result


@pytest.mark.parametrize("decision", [{}, None, "click", [], {"action": "click"}])
def test_incomplete_schema_rejected(decision):
    assert "INVALID_DECISION" in assert_stopped(decision).triggered_rules


@pytest.mark.parametrize("key", ["action", "target", "context", "confidence", "risk"])
def test_every_required_field_is_checked(key):
    decision = valid()
    del decision[key]
    assert_stopped(decision)


@pytest.mark.parametrize("key,value", [("action", ""), ("action", 7), ("target", None),
    ("context", None), ("risk", "0.0"), ("confidence", True), ("risk", -0.1),
    ("risk", 1.1), ("confidence", None), ("confidence", 2)])
def test_malformed_values_rejected(key, value):
    decision = valid()
    decision[key] = value
    assert_stopped(decision)


@pytest.mark.parametrize("profile_factory", [DomainRiskProfile.standard, DomainRiskProfile.read_only, DomainRiskProfile.critical])
@pytest.mark.parametrize("action,risk", [("delete", 1.0), ("delete", 0.0), ("click", 1.0), ("rm", 0.0)])
def test_whitelist_never_overrides_hard_stop(profile_factory, action, risk):
    profile = profile_factory()
    profile.whitelisted_targets.add("safe")
    decision = valid()
    decision.update(action=action, risk=risk)
    result = assert_stopped(decision, profile)
    assert "INVALID_DECISION" not in result.triggered_rules


@pytest.mark.parametrize("whitelisted", [False, True])
def test_risk_threshold_is_inclusive(whitelisted):
    profile = DomainRiskProfile(whitelisted_targets={"safe"} if whitelisted else set())
    decision = valid()
    decision["risk"] = profile.risk_threshold
    assert_stopped(decision, profile)
    decision["risk"] = math.nextafter(profile.risk_threshold, 0.0)
    assert DecisionPolicyGate(profile).evaluate_policy(decision).passed


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["risk", "confidence", "risk_prob", "target_confidence", "action_confidence"])
def test_nonfinite_rejected_before_whitelist(value, field):
    decision = valid()
    decision[field] = value
    profile = DomainRiskProfile(whitelisted_targets={"safe"})
    assert "INVALID_DECISION" in assert_stopped(decision, profile).triggered_rules


def test_safe_decision_and_explicit_state_supported():
    decision = valid()
    assert DecisionPolicyGate().evaluate_policy(decision).passed
    del decision["context"]
    assert DecisionPolicyGate().evaluate_policy(decision, state={}).passed


@pytest.mark.parametrize("field,value", [("risk_prob", float("nan")), ("risk_prob", 1.0), ("action", "delete")])
def test_typed_decision_cannot_bypass_gate(field, value):
    decision = CompositeStepDecision("safe", .99, "click", .99, 0.0, 0.0)
    assert DecisionPolicyGate().evaluate_policy(decision, state={}).passed
    assert_stopped(replace(decision, **{field: value}), DomainRiskProfile(whitelisted_targets={"safe"}), state={})


@pytest.mark.parametrize("field,value", [("risk_prob", 1.0), ("action_type", "delete"), ("target_confidence", .1)])
def test_conflicting_aliases_rejected(field, value):
    decision = valid()
    decision[field] = value
    assert_stopped(decision)
