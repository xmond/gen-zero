"""Backward compatibility shim.

All reading anchor and answerability logic has been fully internalized into
the core system at `gen_zero.gate.grounding_gate`.
"""

from gen_zero.gate.grounding_gate import (
    AnswerabilityEvidence,
    EpistemicGroundingGate,
    extract_passage_and_question,
)

# Aliases for backward compatibility
split_squad_context = extract_passage_and_question

def answerability_anchor(passage: str, question: str) -> AnswerabilityEvidence:
    gate = EpistemicGroundingGate()
    return gate.evaluate_evidence(passage, question)

def decide_answerable(p_unans: float, ev: AnswerabilityEvidence, conf_gate: float = 0.65):
    gate = EpistemicGroundingGate(confidence_gate=conf_gate)
    return gate.decide(p_unans, ev)
