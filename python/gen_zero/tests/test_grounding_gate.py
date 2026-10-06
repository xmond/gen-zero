"""Unit tests for EpistemicGroundingGate."""

import pytest
from gen_zero.gate.grounding_gate import (
    EpistemicGroundingGate,
    DiscourseFocusRewriter,
    DiscourseFocus,
    analyze_discourse_focus,
    LanguageAgnosticDiscourseAnalyzer,
    GeodesicCurvatureDiscourseDetector,
    extract_passage_and_question,
)


def test_extract_passage_and_question():
    prompt = "Passage: The Eiffel Tower is in Paris.\nQuestion: Where is the Eiffel Tower?"
    p, q = extract_passage_and_question(prompt)
    assert p == "The Eiffel Tower is in Paris."
    assert q == "Where is the Eiffel Tower?"


def test_grounding_gate_hit():
    gate = EpistemicGroundingGate(confidence_gate=0.65)
    passage = "Rollo led the Norse raiders in the 10th century. The Duchy of Normandy began in 911."
    question = "Who led the Norse raiders?"
    ev = gate.evaluate_evidence(passage, question)
    assert ev.entity_cover >= 0.5
    assert not ev.number_conflict
    lbl, reason = gate.decide(0.40, ev)
    assert lbl == "answerable"


def test_grounding_gate_miss_with_number_conflict():
    gate = EpistemicGroundingGate(confidence_gate=0.65)
    passage = "Rollo led the Norse raiders in the 10th century. The Duchy of Normandy began in 911."
    question = "What did Charlemagne build in Paris in 1200?"
    ev = gate.evaluate_evidence(passage, question)
    assert ev.number_conflict
    lbl, reason = gate.decide(0.55, ev)
    assert lbl == "unanswerable"
    assert reason == "anchor_missing_evidence"


# --- Discourse turn tests (curvature, no conjunction lexicon) ---

def _nucleus(focus):
    return [t for t, w in focus.segments if w > 1.0]


def _concession(focus):
    return [t for t, w in focus.segments if w < 1.0]


def test_discourse_focus_no_turn():
    """A restated subject keeps the trajectory straight: neutral focus."""
    passage = "The team climbed the summit. The team reached the summit at dawn."
    focus = analyze_discourse_focus(passage)
    assert not focus.has_turn
    assert focus.nucleus_weight == 1.0
    assert focus.concession_weight == 1.0
    assert focus.segments == [(passage, 1.0)]


def test_discourse_turn_english():
    focus = analyze_discourse_focus("Although the movie was boring, the acting was superb.")
    assert focus.has_turn
    assert focus.nucleus_weight > 1.0 > focus.concession_weight
    assert "acting" in _nucleus(focus)[0]
    assert "boring" in _concession(focus)[0]


def test_discourse_turn_german():
    focus = analyze_discourse_focus(
        "Obwohl die Reise anstrengend war, erreichte die Expedition den Gipfel."
    )
    assert focus.has_turn
    assert "Gipfel" in _nucleus(focus)[0]
    assert "anstrengend" in _concession(focus)[0]


def test_discourse_turn_german_postposed_marker():
    focus = analyze_discourse_focus(
        "Die Reise war anstrengend, jedoch erreichte die Expedition den Gipfel."
    )
    assert focus.has_turn
    assert "Gipfel" in _nucleus(focus)[0]


def test_discourse_turn_chinese():
    focus = analyze_discourse_focus("Although it was late, the rescue team pushed forward.")
    assert focus.has_turn
    assert "pushed forward" in _nucleus(focus)[0]
    assert "was late" in _concession(focus)[0]


def test_discourse_turn_implicit_no_conjunction():
    """Zero conjunctions: the turn comes from trajectory deflection alone."""
    focus = analyze_discourse_focus("The weather was horrific. The team reached the summit.")
    assert focus.has_turn
    assert focus.turn_positions == [1]
    assert "summit" in _nucleus(focus)[0]
    assert "weather" in _concession(focus)[0]


def test_discourse_curvature_formula():
    """kappa(t) = 1 - cos(v_t, v_{t-1}) with v_t = h_t - h_{t-1}, h_0 = origin."""
    import numpy as np
    h = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    kappa = LanguageAgnosticDiscourseAnalyzer.curvature(h)
    # v = (1,0), (-1,1), (0,-1): cos(v2,v1) = -1/sqrt2, cos(v3,v2) = -1/sqrt2
    assert kappa[0] == 0.0
    assert kappa[1] == pytest.approx(1.0 + 1.0 / np.sqrt(2.0))
    assert kappa[2] == pytest.approx(1.0 + 1.0 / np.sqrt(2.0))


def test_discourse_encoder_hook_and_shape_check():
    import numpy as np
    turn = lambda segs: np.array([[1.0, 0.0], [-1.0, 0.0]])
    same = lambda segs: np.array([[1.0, 0.0], [1.0, 0.01]])
    text = "First clause here, second clause there."
    assert LanguageAgnosticDiscourseAnalyzer(encoder=turn).analyze(text).has_turn
    assert not LanguageAgnosticDiscourseAnalyzer(encoder=same).analyze(text).has_turn
    with pytest.raises(ValueError):
        LanguageAgnosticDiscourseAnalyzer(encoder=lambda segs: np.zeros((1, 2))).analyze(text)


def test_discourse_alias_and_wrapper_agree():
    passage = "Despite the rain, the picnic continued."
    f1 = DiscourseFocusRewriter.analyze_discourse_focus(passage)
    f2 = analyze_discourse_focus(passage)
    f3 = GeodesicCurvatureDiscourseDetector().analyze(passage)
    assert f1 == f2 == f3


def test_thousands_separator_is_not_a_clause_boundary():
    focus = analyze_discourse_focus("The town has 3,000 people and 1,200 houses.")
    assert not focus.has_turn


def test_nucleus_weighted_passage_grounds_answer():
    passage = "Although the company reported losses, its stock price rose sharply."
    gate = EpistemicGroundingGate(confidence_gate=0.65)
    focus = analyze_discourse_focus(passage)
    assert focus.has_turn
    parts = []
    for text, weight in focus.segments:
        parts.extend([text] * max(1, int(round(weight))))
    ev = gate.evaluate_evidence(" ".join(parts), "Did the stock price rise sharply?")
    assert ev.co_occurrence_score == 1.0
    assert not ev.number_conflict


# --- Language-agnostic number conflict ---

def test_number_conflict_digits_only_across_scripts():
    gate = EpistemicGroundingGate()
    assert gate.evaluate_evidence("Die Firma wurde 1990 gegründet.", "Wann wurde die Firma 2005 gegründet?").number_conflict
    assert gate.evaluate_evidence("Company was founded in 1990.", "Was the company founded in 2005?").number_conflict
    # Full-width digits fold to the same number.
    assert not gate.evaluate_evidence("Company was founded in 1990.", "Was the company founded in 1990?").number_conflict
    # Thousands grouping does not create a false conflict.
    assert not gate.evaluate_evidence("The fund holds 1,200 shares.", "Does the fund hold 1200 shares?").number_conflict


# --- Entity-predicate co-occurrence tests ---

def test_fragmented_cross_sentence_rejected():
    """When entity and predicate span 3+ sentences with no co-occurrence, reject."""
    passage = (
        "Einstein was born in Ulm. "
        "He later worked at the patent office. "
        "General relativity was completed in 1915."
    )
    gate = EpistemicGroundingGate(confidence_gate=0.65)

    # Q1: "Where was Einstein born?" — co-occurs in sentence 1, should be answerable.
    ev1 = gate.evaluate_evidence(passage, "Where was Einstein born?")
    assert ev1.co_occurrence_score >= 0.8, f"co-occurrence score too low: {ev1.co_occurrence_score}"

    # Q2: "When did Einstein publish general relativity?" — no single sentence has
    # both "Einstein" and "publish general relativity".  Should have low co-occurrence.
    ev2 = gate.evaluate_evidence(passage, "When did Einstein publish general relativity?")
    assert ev2.co_occurrence_score < 0.6, (
        f"Expected fragmented co-occurrence (< 0.6), got {ev2.co_occurrence_score}"
    )
    # With fragmented evidence, gate should lean unanswerable.
    lbl, reason = gate.decide(0.55, ev2)
    assert lbl == "unanswerable", f"Expected unanswerable, got {lbl} ({reason})"


def test_co_occurrence_in_same_sentence_accepted():
    """When entity and predicate co-occur in the same sentence, accept."""
    passage = "Einstein published general relativity in 1915."
    gate = EpistemicGroundingGate(confidence_gate=0.65)
    ev = gate.evaluate_evidence(passage, "When did Einstein publish general relativity?")
    assert ev.co_occurrence_score == 1.0, f"Expected 1.0, got {ev.co_occurrence_score}"
    lbl, reason = gate.decide(0.40, ev)
    assert lbl == "answerable", f"Expected answerable, got {lbl} ({reason})"


def test_squad2_unanswerable_fragmented_entity_predicate():
    """SQuAD 2.0 style: every question word occurs, but never together."""
    passage = (
        "Marie Curie won the Nobel Prize. "
        "Pierre lectured at the Sorbonne in Paris. "
        "The laboratory burned down in 1911."
    )
    gate = EpistemicGroundingGate(confidence_gate=0.65)
    ev = gate.evaluate_evidence(passage, "Where did Marie Curie lecture?")
    assert ev.entity_cover == 1.0  # word overlap alone would call this grounded
    assert ev.co_occurrence_score < 0.8
    # Even a model that leans answerable must not be endorsed.
    lbl, reason = gate.decide(0.40, ev)
    assert lbl == "unanswerable", f"{lbl} ({reason}) score={ev.score}"

    same = "Marie Curie lectured at the Sorbonne in Paris. Pierre won the Nobel Prize."
    ev_ok = gate.evaluate_evidence(same, "Where did Marie Curie lecture?")
    assert ev_ok.co_occurrence_score == 1.0
    assert gate.decide(0.40, ev_ok)[0] == "answerable"


def test_grounding_needs_no_english_stem_table():
    """Prefix stems match inflections in another language without any suffix list."""
    gate = EpistemicGroundingGate()
    passage = "Die Expedition erreichte den Gipfel im Winter."
    ev = gate.evaluate_evidence(passage, "Wann erreichen Expeditionen Gipfel?")
    assert ev.co_occurrence_score == 1.0


def test_source_has_no_language_specific_wordlists():
    import inspect
    import gen_zero.gate.grounding_gate as mod
    src = inspect.getsource(mod).lower()
    for banned in ("however", "although", "nevertheless", "january", "_stop", "_wh ="):
        assert banned not in src, banned


def test_empty_question_fails_closed():
    gate = EpistemicGroundingGate()
    ev = gate.evaluate_evidence("Some passage text.", "")
    assert ev.score == 0.0
    lbl, reason = gate.decide(0.55, ev)
    assert lbl == "unanswerable"
    assert reason == "anchor_missing_evidence"

