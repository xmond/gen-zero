"""Focused regressions for runtime decision and confidence bounds."""

import pytest

from gen_zero.runtime.composite_decision import CompositeDecisionEngine
from gen_zero.runtime.confidence_floor import ConfidenceFloorGate


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan"), float("inf"), float("-inf")])
def test_confidence_floor_rejects_non_unit_interval_confidence(confidence):
    gate = ConfidenceFloorGate()

    with pytest.raises(ValueError, match=r"confidence.*finite.*\[0.0, 1.0\]"):
        gate.evaluate({"action": "keep", "confidence": confidence})


def test_confidence_floor_keeps_explicit_zero_instead_of_deriving_from_probs():
    gate = ConfidenceFloorGate(default_fallback_action="hold")

    verdict = gate.evaluate(
        {"action": "keep", "confidence": 0.0, "probs": {"keep": 0.99}},
        candidates=["keep", "hold"],
    )

    assert verdict.passed is False
    assert verdict.fallback_used is True
    assert verdict.confidence == 0.0
    assert verdict.action == "hold"


@pytest.mark.parametrize("bad_probability", [float("nan"), float("inf"), -0.1, 1.1])
def test_confidence_floor_rejects_invalid_derived_probability(bad_probability):
    gate = ConfidenceFloorGate()

    with pytest.raises(ValueError, match=r"probs.*finite.*\[0.0, 1.0\]"):
        gate.evaluate({"action": "keep", "probs": {"keep": bad_probability}})


def test_confidence_floor_rejects_action_outside_optional_candidates():
    gate = ConfidenceFloorGate()

    with pytest.raises(ValueError, match="Chosen action.*not in candidates"):
        gate.evaluate(
            {"action": "delete", "confidence": 0.9},
            candidates=["keep", "hold"],
        )


def test_confidence_floor_degraded_result_cannot_pass_autonomously():
    gate = ConfidenceFloorGate(default_fallback_action="hold")

    verdict = gate.evaluate(
        {
            "action": "keep",
            "confidence": 0.99,
            "degraded": True,
            "degraded_reason": "untrained_weights_fallback",
        },
        candidates=["keep", "hold"],
    )

    assert verdict.passed is False
    assert verdict.fallback_used is True
    assert verdict.degraded is True
    assert verdict.telemetry["degraded"] is True
    assert "untrained_weights_fallback" in verdict.fallback_reason


class _StaticDecisionClient:
    """Return deterministic responses for the four composite decision queries."""

    def __init__(self, target="target", action="click", risk=0.1):
        self.responses = [
            {"action": target, "confidence": 0.9, "probs": {target: 0.9}},
            {"action": action, "confidence": 0.9, "probs": {action: 0.9}},
            {"probs": {"true": 0.0, "false": 1.0}},
            {"probs": {"true": risk, "false": 1.0 - risk}},
        ]
        self._index = 0

    def decide(self, **_kwargs):
        response = self.responses[self._index]
        self._index += 1
        return response


def _engine(client):
    return CompositeDecisionEngine(client=client)


@pytest.mark.parametrize("target", ["missing", None, ""])
def test_composite_rejects_unknown_target(target):
    with pytest.raises(ValueError, match="target.*not in affordances"):
        _engine(_StaticDecisionClient(target=target)).decide_step(
            state="state", affordances=["target"], action_types=["click"]
        )


@pytest.mark.parametrize("action", ["missing", None, ""])
def test_composite_rejects_unknown_action(action):
    with pytest.raises(ValueError, match="action.*not in action_types"):
        _engine(_StaticDecisionClient(action=action)).decide_step(
            state="state", affordances=["target"], action_types=["click"]
        )


@pytest.mark.parametrize("risk", [-0.01, float("nan"), float("inf"), 1.01])
def test_composite_rejects_invalid_risk_probability(risk):
    with pytest.raises(ValueError, match="risk probability"):
        _engine(_StaticDecisionClient(risk=risk)).decide_step(
            state="state", affordances=["target"], action_types=["click"]
        )


def test_composite_accepts_membership_and_nonnegative_risk():
    decision = _engine(_StaticDecisionClient()).decide_step(
        state="state", affordances=["target"], action_types=["click"]
    )

    assert decision.target == "target"
    assert decision.action == "click"
    assert decision.risk_prob == pytest.approx(0.1)


@pytest.mark.parametrize("boundary", [0.0, 1.0])
def test_confidence_floor_accepts_closed_interval_endpoints(boundary):
    verdict = ConfidenceFloorGate(tau_floor=boundary).evaluate(
        {"action": "keep", "confidence": boundary}, candidates=["keep"]
    )
    assert verdict.passed is True
    assert verdict.confidence == boundary


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), float("inf")])
def test_confidence_floor_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="tau_floor"):
        ConfidenceFloorGate(tau_floor=threshold)


def test_confidence_floor_exposes_fallback_rule_failure():
    def broken_rule(state, candidates):
        raise RuntimeError("rule unavailable")

    verdict = ConfidenceFloorGate(fallback_rule_fn=broken_rule).evaluate(
        {"action": "keep", "confidence": 0.0}, candidates=["keep"]
    )
    assert verdict.passed is False
    assert verdict.action == "ESCALATE_TO_HUMAN"
    assert verdict.telemetry["fallback_rule_error"] == "RuntimeError"
    assert "Fallback rule failed: RuntimeError" in verdict.fallback_reason


@pytest.mark.parametrize("index,field", [(0, "confidence"), (1, "confidence"),
                                          (0, "probs"), (1, "probs"), (2, "probs"), (3, "probs")])
@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_composite_rejects_invalid_probability_metadata(index, field, value):
    client = _StaticDecisionClient()
    if field == "probs":
        client.responses[index][field][next(iter(client.responses[index][field]))] = value
    else:
        client.responses[index][field] = value
    with pytest.raises(ValueError, match="finite number"):
        _engine(client).decide_step(state="state", affordances=["target"], action_types=["click"])


@pytest.mark.parametrize("risk", [0.0, 1.0])
def test_composite_accepts_risk_endpoints(risk):
    decision = _engine(_StaticDecisionClient(risk=risk)).decide_step(
        state="state", affordances=["target"], action_types=["click"]
    )
    assert decision.risk_prob == risk


def test_composite_missing_risk_is_maximal():
    client = _StaticDecisionClient()
    client.responses[3] = {}
    decision = _engine(client).decide_step(
        state="state", affordances=["target"], action_types=["click"]
    )
    assert decision.risk_prob == 1.0
    assert decision.raw_answers["risk_assessed"] is False
