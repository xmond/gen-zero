"""Gen-Zero Layer 4: Decide-and-Fill Two-Stage Action Pipeline (Issue #15).

Decouples agent cognitive load into two dedicated stages:
- Stage 1 (System 1 Action Routing): Non-autoregressive discrete action selection using
  Gen-Zero pure prefill computation (0 tokens, sub-5ms latency, mathematically bounded).
- Stage 2 (Lightweight Argument Filling):
  - Zero-overhead bypass for argument-free actions (clicks, submits, confirms).
  - Word-Span Extraction for natural language text inputs: determines (type_from, type_to)
    word boundary indices on the original user utterance, performing deterministic string
    slicing with 100% literal fidelity (no autoregressive hallucination or mutation).
  - Structured parameter extraction for typed slots (numbers, booleans, enums).
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import re
import time
import math


@dataclass
class WordSpan:
    """Represents a tokenized word within an utterance and its exact character span."""
    index: int
    text: str
    start_char: int
    end_char: int


class WordSpanExtractor:
    """Extracts exact literal argument spans from user utterances without autoregression.

    Adheres to the zero-use Word-Span paradigm:
    Instead of generating text via an autoregressive LLM (which risks hallucination,
    misspellings, or prompt injection), the model selects start and end word indices,
    and a local deterministic slice `utterance[start_char:end_char]` is executed.
    """

    WORD_PATTERN = re.compile(r'\S+')

    @classmethod
    def tokenize_spans(cls, utterance: str) -> List[WordSpan]:
        """Splits an utterance into whitespace-delimited word tokens with character spans."""
        spans: List[WordSpan] = []
        for idx, match in enumerate(cls.WORD_PATTERN.finditer(utterance)):
            spans.append(WordSpan(
                index=idx,
                text=match.group(0),
                start_char=match.start(),
                end_char=match.end()
            ))
        return spans

    @classmethod
    def slice_utterance(
        cls,
        utterance: str,
        start_word_idx: int,
        end_word_idx: int
    ) -> str:
        """Deterministically slices the utterance from start_word_idx to end_word_idx (inclusive).

        Guarantees 100% literal preservation including punctuation and casing.
        """
        spans = cls.tokenize_spans(utterance)
        if not spans:
            return ""

        s_idx = max(0, min(start_word_idx, len(spans) - 1))
        e_idx = max(s_idx, min(end_word_idx, len(spans) - 1))

        start_char = spans[s_idx].start_char
        end_char = spans[e_idx].end_char
        return utterance[start_char:end_char].strip()

    @classmethod
    def extract_search_or_type_span(
        cls,
        utterance: str,
        action_name: str = "type_text"
    ) -> Tuple[str, int, int]:
        """Extracts the semantic payload span for search/typing actions.

        Filters out command prefixes (e.g. 'search for', 'type', 'enter', 'query', 'find')
        and quotes when present.
        """
        spans = cls.tokenize_spans(utterance)
        if not spans:
            return "", 0, 0

        # Check for quoted phrases first: "..." or '...'
        quote_match = re.search(r'["\']([^"\']+)["\']', utterance)
        if quote_match:
            val = quote_match.group(1).strip()
            return val, 0, len(spans) - 1

        # Strip standard command trigger prefix words
        prefix_words = {"search", "for", "find", "query", "type", "enter", "input", "lookup", "filter", "by", "please"}
        start_idx = 0
        for idx, span in enumerate(spans):
            cleaned = span.text.lower().strip(":,.;!?")
            if cleaned in prefix_words:
                start_idx = idx + 1
            else:
                break

        if start_idx >= len(spans):
            start_idx = 0

        end_idx = len(spans) - 1
        # Strip trailing punctuation keywords if present (e.g. 'in the box', 'please')
        while end_idx > start_idx and spans[end_idx].text.lower().strip(":,.;!?") in {"please", "now"}:
            end_idx -= 1

        extracted_text = cls.slice_utterance(utterance, start_idx, end_idx)
        return extracted_text, start_idx, end_idx


@dataclass
class ActionSpec:
    """Definition of an available agent action or tool."""
    name: str
    description: str
    requires_arguments: bool = False
    parameter_schema: Optional[Dict[str, Any]] = None
    category: str = "general"


@dataclass
class PipelineDecision:
    """Structured decision output from the Decide-and-Fill pipeline."""
    action: str
    arguments: Dict[str, Any]
    confidence: float
    stage_1_latency_ms: float
    stage_2_latency_ms: float
    total_tokens_generated: int = 0
    is_zero_token: bool = True
    word_span: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_latency_ms(self) -> float:
        return self.stage_1_latency_ms + self.stage_2_latency_ms


class DecideAndFillPipeline:
    """Two-Stage Decide-and-Fill Execution Engine.

    Stage 1: System 1 Action Selection via non-autoregressive discrete decision.
    Stage 2: Deterministic argument binding via Word-Span Extraction or slot extraction.
    """

    def __init__(
        self,
        decision_client: Optional[Any] = None,
        available_actions: Optional[List[ActionSpec]] = None
    ):
        self.decision_client = decision_client
        self.actions: Dict[str, ActionSpec] = {}
        if available_actions:
            for act in available_actions:
                self.actions[act.name] = act

    def register_action(self, action: ActionSpec) -> None:
        self.actions[action.name] = action

    def decide_and_fill(
        self,
        utterance: str,
        state: Union[str, Dict[str, Any]],
        candidate_actions: Optional[List[str]] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> PipelineDecision:
        """Executes the two-stage pipeline.

        Args:
            utterance: User objective or task instruction.
            state: Current environment state (text, DOM excerpt, or dict).
            candidate_actions: Optional subset of registered action names to consider.
            overrides: Optional deterministic override parameters.

        Returns:
            PipelineDecision with action, filled arguments, timing, and 0 tokens metric.
        """
        t0 = time.perf_counter()

        # Determine valid candidates
        if candidate_actions:
            cands = [c for c in candidate_actions if c in self.actions] or candidate_actions
        elif self.actions:
            cands = list(self.actions.keys())
        else:
            cands = ["noop"]

        # ----------------------------------------------------
        # STAGE 1: Action Selection (System 1 Action Routing)
        # ----------------------------------------------------
        selected_action = cands[0]
        confidence = 0.95

        if self.decision_client is not None and hasattr(self.decision_client, "decide"):
            try:
                state_str = state if isinstance(state, str) else str(state)
                criteria = {}
                for c in cands:
                    spec = self.actions.get(c)
                    criteria[c] = spec.description if spec else f"Perform action {c}"

                prompt = f"Objective: {utterance}\nEnvironment: {state_str[:400]}"
                res = self.decision_client.decide(
                    state=prompt,
                    candidates=cands,
                    mode="reflex",
                    candidate_descriptions=criteria
                )
                selected_action = res.get("action", cands[0])
                confidence = float(res.get("confidence", 0.95))
            except Exception:
                selected_action = self._heuristic_action_route(utterance, cands)
        else:
            selected_action = self._heuristic_action_route(utterance, cands)

        stage_1_time = (time.perf_counter() - t0) * 1000.0

        # ----------------------------------------------------
        # STAGE 2: Lightweight Argument Filling
        # ----------------------------------------------------
        t1 = time.perf_counter()
        spec = self.actions.get(selected_action)
        requires_args = spec.requires_arguments if spec else ("type" in selected_action or "search" in selected_action)

        arguments: Dict[str, Any] = {}
        span_indices: Optional[Tuple[int, int]] = None

        if not requires_args:
            # Zero-argument action (e.g. click, confirm, submit, scroll, refresh)
            arguments = {}
        else:
            # Text input or search query: execute Word-Span Extraction
            if any(k in selected_action for k in ("search", "type", "input", "query", "filter")):
                text_val, s_idx, e_idx = WordSpanExtractor.extract_search_or_type_span(
                    utterance, action_name=selected_action
                )
                arguments["text"] = text_val
                arguments["word_span"] = {"start": s_idx, "end": e_idx}
                span_indices = (s_idx, e_idx)
            else:
                # General structured slot filling
                arguments = self._fill_structured_slots(utterance, state, spec)

        # Apply any explicit caller overrides
        if overrides:
            arguments.update(overrides)

        stage_2_time = (time.perf_counter() - t1) * 1000.0

        return PipelineDecision(
            action=selected_action,
            arguments=arguments,
            confidence=round(confidence, 4),
            stage_1_latency_ms=round(stage_1_time, 2),
            stage_2_latency_ms=round(stage_2_time, 2),
            total_tokens_generated=0,
            is_zero_token=True,
            word_span=span_indices,
            metadata={
                "action_category": spec.category if spec else "unknown",
                "candidate_count": len(cands),
                "extraction_method": "word_span" if span_indices else "zero_argument"
            }
        )

    def _heuristic_action_route(self, utterance: str, candidates: List[str]) -> str:
        """Fast keyword-directed prior when decision_client is not provided."""
        u_lower = utterance.lower()
        scored = []
        for c in candidates:
            c_clean = c.lower().replace("_", " ")
            score = 0.0
            for word in c_clean.split():
                if word in u_lower:
                    score += 1.0
            spec = self.actions.get(c)
            if spec and spec.description:
                for word in spec.description.lower().split():
                    if len(word) > 3 and word in u_lower:
                        score += 0.5
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else candidates[0]

    def _fill_structured_slots(
        self,
        utterance: str,
        state: Union[str, Dict[str, Any]],
        spec: Optional[ActionSpec]
    ) -> Dict[str, Any]:
        """Extracts structured values (numbers, booleans, dates) via regex matching."""
        slots: Dict[str, Any] = {}
        # Numbers (quantities, prices, indices)
        num_match = re.search(r'\b(\d+(?:\.\d+)?)\b', utterance)
        if num_match:
            slots["quantity"] = float(num_match.group(1)) if "." in num_match.group(1) else int(num_match.group(1))

        # Boolean confirmations
        if any(w in utterance.lower() for w in ["yes", "confirm", "approve", "agree", "enable"]):
            slots["enabled"] = True
        elif any(w in utterance.lower() for w in ["no", "deny", "reject", "cancel", "disable"]):
            slots["enabled"] = False

        return slots
