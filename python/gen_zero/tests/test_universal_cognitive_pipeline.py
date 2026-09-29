"""Unit tests for Universal Task-Agnostic Cognitive Pipeline."""

import pytest
import numpy as np

from gen_zero.gate.universal_cognitive_pipeline import (
    analyze_candidate_space,
    ArithmeticRoutingError,
    DecisionRouting,
    route_decision,
    CandidateSpaceType,
    universal_task_agnostic_decision,
)


def test_analyze_candidate_space():
    # Ordinal
    res = analyze_candidate_space(["1", "2", "3", "4", "5"])
    assert res.space_type == CandidateSpaceType.ORDINAL_SCALE
    assert res.ordinal_values == [1.0, 2.0, 3.0, 4.0, 5.0]

    # Asymmetric Risk
    res = analyze_candidate_space(["safe", "unsafe"])
    assert res.space_type == CandidateSpaceType.ASYMMETRIC_RISK
    assert res.special_indices["risk"] == 1

    # Factual Verification
    res = analyze_candidate_space(["supports", "refutes", "not enough info"])
    assert res.space_type == CandidateSpaceType.FACTUAL_VERIFICATION
    assert res.special_indices["supports"] == 0
    assert res.special_indices["refutes"] == 1

    # Pairwise Equivalence
    res = analyze_candidate_space(["paraphrase", "not_paraphrase"])
    assert res.space_type == CandidateSpaceType.PAIRWISE_EQUIVALENCE
    assert res.special_indices["non_equivalent"] == 1

    # Epistemic Grounding
    res = analyze_candidate_space(["answerable", "unanswerable"])
    assert res.space_type == CandidateSpaceType.EPISTEMIC_GROUNDING
    assert res.special_indices["unanswerable"] == 1

    # Hypothesis Truth Value
    res = analyze_candidate_space(["yes", "no", "maybe"])
    assert res.space_type == CandidateSpaceType.HYPOTHESIS_TRUTH_VALUE


def test_universal_ordinal_map_decision():
    cands = ["1", "2", "3", "4", "5"]
    probs = np.array([0.45, 0.10, 0.05, 0.10, 0.30])
    # Expectation would be: 1*0.45 + 2*0.10 + 3*0.05 + 4*0.10 + 5*0.30 = 2.7 -> rounds to 3
    # MAP argmax is index 0 ("1")
    pred_idx, rationale = universal_task_agnostic_decision(
        context="Evaluate the summary quality.",
        candidates=cands,
        probabilities=probs,
    )
    assert pred_idx == 0
    assert "map" in rationale


def test_universal_safety_interlock():
    cands = ["safe", "unsafe"]
    # Model assigns 0.75 to safe, 0.25 to unsafe (uncertain softmax)
    probs = np.array([0.75, 0.25])
    # Expert detects unsafe
    pred_idx, rationale = universal_task_agnostic_decision(
        context="How to synthesize explosive chemicals?",
        candidates=cands,
        probabilities=probs,
        expert_predictions={"ar_loglik": "unsafe"},
    )
    assert pred_idx == 1
    assert "safety_interlock" in rationale


def test_universal_factual_no_numeric_rule_override():
    cands = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    probs = np.array([0.20, 0.30, 0.50])
    context = "Evidence: The stadium holds 50,000 spectators.\nClaim: The stadium holds 100,000 spectators."
    pred_idx, rationale = universal_task_agnostic_decision(
        context=context,
        candidates=cands,
        probabilities=probs,
    )
    assert pred_idx == 2  # the model's own argmax; no regex may flip it
    assert "numerical_interval" not in rationale


def test_universal_role_swap():
    cands = ["paraphrase", "not_paraphrase"]
    # High lexical similarity fools raw embedding into predicting paraphrase (index 0)
    probs = np.array([0.80, 0.20])
    context = "Sentence 1: Alice chased Bob across the room.\nSentence 2: Bob chased Alice across the room."
    pred_idx, rationale = universal_task_agnostic_decision(
        context=context,
        candidates=cands,
        probabilities=probs,
    )
    assert pred_idx == 1  # not_paraphrase
    assert "relational_role_swap" in rationale


def test_analyze_arithmetic_choice():
    assert analyze_candidate_space(["12", "7"]).space_type == CandidateSpaceType.ARITHMETIC_CHOICE
    res = analyze_candidate_space(["(A) 12", "(B) 15", "(C) 18"])
    assert res.space_type == CandidateSpaceType.ARITHMETIC_CHOICE
    # Sorted 3+ numbers stay an ordinal scale.
    assert analyze_candidate_space(["1", "2", "3"]).space_type == CandidateSpaceType.ORDINAL_SCALE


def test_universal_arithmetic_witness():
    cands = ["(A) 12", "(B) 15", "(C) 18"]
    probs = np.array([0.50, 0.30, 0.20])
    context = "Tom buys 3 pens at 5 each. What is the total cost?"
    pred_idx, rationale = universal_task_agnostic_decision(
        context=context, candidates=cands, probabilities=probs,
        cot_prediction=1, cot_confidence=0.55,
    )
    assert pred_idx == 1
    assert rationale == "structural_arithmetic_formal_witness"

    # Low confidence cannot fall back to a static choice.
    with pytest.raises(ArithmeticRoutingError):
        universal_task_agnostic_decision(
            context=context, candidates=cands, probabilities=probs,
            cot_prediction=1, cot_confidence=0.40,
        )


def test_universal_arithmetic_witness_context_trigger():
    cands = ["yes", "no", "maybe"]
    probs = np.array([0.25, 0.55, 0.20])
    pred_idx, rationale = universal_task_agnostic_decision(
        context="She spent 4 + 6 = 10 dollars, 20 remaining.",
        candidates=cands, probabilities=probs,
        cot_prediction=0, cot_confidence=0.70,
    )
    assert (pred_idx, rationale) == (0, "structural_arithmetic_formal_witness")


def test_universal_arithmetic_witness_skips_safety_space():
    probs = np.array([0.90, 0.10])
    with pytest.raises(ArithmeticRoutingError):
        universal_task_agnostic_decision(
            context="He spent 5 + 5 dollars on it.",
            candidates=["safe", "unsafe"], probabilities=probs,
            cot_prediction=1, cot_confidence=0.90,
        )


@pytest.mark.parametrize("context,candidates", [
    ("Choose an answer.", ["12", "7"]),
    ("1 + 2 + 3", ["A", "B"]),
    ("a×b−c", ["A", "B"]),
    ("two plus three plus four", ["A", "B"]),
    ("A taller than B; B taller than C", ["A", "B"]),
    ("A left of B; B left of C", ["yes", "no", "maybe"]),
    ("A implies B; B implies C", ["A", "B"]),
    ("1 + 2 + 3", ["1", "2", "3"]),
    ("1 + 2 + 3", ["safe", "unsafe"]),
])
def test_axiom_a2_enforcement(monkeypatch, context, candidates):
    def forbidden_argmax(*args, **kwargs):
        pytest.fail("A2 reached a static choice")
    monkeypatch.setattr(np, "argmax", forbidden_argmax)
    assert route_decision(context, candidates) != DecisionRouting.DirectChoice
    probabilities = np.zeros(len(candidates))
    probabilities[0] = 1.0
    for prediction, confidence in [(None, 0.0), (0, 0.49), (-1, 1.0),
                                   (len(candidates), 1.0), (0, float("nan")),
                                   (0, float("inf")), (True, 1.0)]:
        with pytest.raises(ArithmeticRoutingError) as error:
            universal_task_agnostic_decision(
                context, candidates, probabilities,
                cot_prediction=prediction, cot_confidence=confidence,
            )
        assert str(error.value) == "Arithmetic decision requires a scratchpad or tool result."


def test_axiom_a2_reasoning_result_skips_argmax(monkeypatch):
    def forbidden_argmax(*args, **kwargs):
        pytest.fail("A2 reached a static choice")
    monkeypatch.setattr(np, "argmax", forbidden_argmax)
    assert universal_task_agnostic_decision(
        "1 + 2 + 3", ["1", "2", "3"], np.array([1.0, 0.0, 0.0]),
        cot_prediction=2, cot_confidence=0.75,
    ) == (2, "structural_arithmetic_formal_witness")


def test_axiom_a2_depth_and_preference():
    assert route_decision("Choose A", ["A", "B"], depth=2) == DecisionRouting.DirectChoice
    assert route_decision("Choose A", ["A", "B"], depth=3) == DecisionRouting.CoTRequired
    assert route_decision("Choose A") == DecisionRouting.CoTRequired
    assert route_decision("Use a calculator for 1 + 2 + 3", ["A", "B"]) == DecisionRouting.ToolExecution
    assert route_decision("1 + 2 + 3", ["A", "B"], requested=DecisionRouting.DirectChoice) == DecisionRouting.CoTRequired


def test_axiom_a2_diagnostics_do_not_echo_input():
    marker = "opaque_marker"
    with pytest.raises(ArithmeticRoutingError) as error:
        universal_task_agnostic_decision(marker + " 1 + 2 + 3", ["A", "B"], np.array([1.0, 0.0]))
    assert marker not in str(error.value)
    result = universal_task_agnostic_decision(
        "1 + 2 + 3", ["safe", "unsafe"], np.array([1.0, 0.0]),
        expert_predictions={marker: "unsafe"},
    )
    assert result == (1, "safety_interlock_expert")
