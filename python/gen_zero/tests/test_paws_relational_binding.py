"""Tests for signed relational binding on PAWS-style adversarial paraphrase pairs."""

import json
from pathlib import Path

import pytest

from gen_zero.nanocore.choice_head import ActionETFChoiceHead
from gen_zero.nanocore.relational_binding import (
    analyze_context,
    analyze_relational_binding,
    split_sentence_pair,
)

PAWS_PATH = Path(__file__).resolve().parents[3] / "benchmarks" / "data" / "paws.jsonl"


def _prompt(s1: str, s2: str) -> str:
    return f"Sentence 1: {s1}\nSentence 2: {s2}\nDo these two sentences have the exact same meaning?"


def test_split_sentence_pair():
    assert split_sentence_pair(_prompt("A b .", "B a .")) == ("A b .", "B a .")
    assert split_sentence_pair("no pair here") is None


def test_river_tributary_swap_is_flagged():
    r = analyze_relational_binding(
        "The Tabaci River is a tributary of the River Leurda in Romania .",
        "The Leurda River is a tributary of the River Tabaci in Romania .")
    assert r.has_role_swap and ("leurda", "tabaci") in r.swapped_entities
    assert r.swap_penalty > 0


def test_subject_object_swap_across_same_verb():
    r = analyze_relational_binding("Alice hired Bob in Paris .", "Bob hired Alice in Paris .")
    assert r.has_role_swap


def test_hyphen_compound_reversal():
    r = analyze_relational_binding("the anglo-Egyptian Sudan", "the Egyptian-Anglo Sudan")
    assert r.has_role_swap and r.reversed_compounds == ("anglo-Egyptian",)


@pytest.mark.parametrize("s1,s2", [
    # Symmetric coordination keeps the meaning.
    ("Winarsky is a member of the IEEE , Phi Beta Kappa , the ACM and Sigma Xi .",
     "Winarsky is a member of ACM , the IEEE , the Phi Beta Kappa and the Sigma Xi ."),
    ("Kathy and Pete met .", "Pete and Kathy met ."),
    # Fronting one adverbial keeps the meaning.
    ("The family moved to Camp Hill in 1972 .", "In 1972 the family moved to Camp Hill ."),
    # Clause reorder keeps every verb-entity bigram.
    ("John visited Rome and Mary visited Paris .", "Mary visited Paris and John visited Rome ."),
    # Identical sentences.
    ("Bob hired Alice .", "Bob hired Alice ."),
])
def test_paraphrases_are_not_flagged(s1, s2):
    assert not analyze_relational_binding(s1, s2).has_role_swap


def test_signed_report_is_serialisable():
    r = analyze_relational_binding("Alice hired Bob in Paris .", "Bob hired Alice in Paris .")
    json.dumps(r.to_dict())


def _decide(prompt, cands):
    return ActionETFChoiceHead(hidden_dim=32, action_dim=16).decide(prompt, cands)


@pytest.mark.parametrize("cands", [["paraphrase", "not_paraphrase"], ["not_paraphrase", "paraphrase"]])
def test_choice_head_rejects_swapped_roles_in_any_candidate_order(cands):
    res = _decide(_prompt("Madiun is situated on the main road to Yogyakarta and Jakarta .",
                          "Yogyakarta and Jakarta is situated on the main road to Madiun ."), cands)
    assert res.selected_action == "not_paraphrase"
    assert res.binding_report is not None and res.binding_report.has_role_swap
    assert "binding_report" in res.to_dict()


def test_choice_head_leaves_non_paraphrase_tasks_untouched():
    res = _decide(_prompt("Alice hired Bob .", "Bob hired Alice ."), ["yes", "no"])
    assert res.binding_report is None


def test_swap_detector_on_paws_sample():
    """In-sample / benchmark check: verify swap detector precision and recall on PAWS benchmark data."""
    rows = [json.loads(line) for line in PAWS_PATH.read_text().splitlines() if line.strip()]
    assert len(rows) >= 30
    flagged = {r["id"]: analyze_context(r["context"]).has_role_swap for r in rows}
    para = [r["id"] for r in rows if r["ground_truth"] == "paraphrase"]
    not_para = [r["id"] for r in rows if r["ground_truth"] == "not_paraphrase"]
    # False positive rate on true paraphrases must remain negligible (<= 2%)
    assert sum(flagged[i] for i in para) <= max(1, int(0.02 * len(para))), "false swap on a true paraphrase"
    # Substantial true positive recall on adversarial non-paraphrases
    assert sum(flagged[i] for i in not_para) >= (12 if len(rows) == 30 else 50)
    for i in ("paws-0001", "paws-0025", "paws-0027"):
        if i in flagged:
            assert flagged[i]
