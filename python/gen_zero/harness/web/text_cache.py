"""Text Parameter Cache and Two-Stage Argument Resolver (Issue #34 & RFC-034).

Supports the two-stage decoupled execution paradigm:
- Stage 1: Fast discrete action decision (<15ms, zero-token).
- Stage 2: Parameter generation for TYPE_TEXT actions, using idempotent TextCache,
  WordSpan literal extraction from goal, or on-demand lightweight model generation.
"""

from typing import Any, Callable, Dict, Optional, Tuple
from gen_zero.harness.action_pipeline import WordSpanExtractor
from .types import WebTargetSlot


class TextCache:
    """Idempotent cache for text parameters across repetitive web interactions."""

    def __init__(self):
        self._cache: Dict[Tuple[str, str, int], str] = {}

    def make_key(self, signature_hash: str, goal: str, slot_id: int) -> Tuple[str, str, int]:
        return (signature_hash, goal.strip(), slot_id)

    def get(self, signature_hash: str, goal: str, slot_id: int) -> Optional[str]:
        key = self.make_key(signature_hash, goal, slot_id)
        return self._cache.get(key)

    def put(self, signature_hash: str, goal: str, slot_id: int, text: str) -> None:
        key = self.make_key(signature_hash, goal, slot_id)
        self._cache[key] = text

    def clear(self) -> None:
        self._cache.clear()

    def resolve_text_parameter(
        self,
        goal: str,
        slot: WebTargetSlot,
        signature_hash: str = "",
        explicit_param: Optional[str] = None,
        generator_fn: Optional[Callable[[str, WebTargetSlot], str]] = None
    ) -> str:
        """Resolves text parameter using hierarchy: explicit -> cache -> WordSpan -> generator."""
        slot_id = slot.slot_id

        # 1. Explicit parameter takes priority
        if explicit_param is not None and explicit_param != "":
            self.put(signature_hash, goal, slot_id, explicit_param)
            return explicit_param

        # 2. Check idempotent cache
        cached = self.get(signature_hash, goal, slot_id)
        if cached is not None:
            return cached

        # 3. Deterministic WordSpan extraction from user goal
        extracted, _, _ = WordSpanExtractor.extract_search_or_type_span(
            goal, action_name=slot.criteria_key
        )
        if extracted and extracted.strip():
            res = extracted.strip()
            self.put(signature_hash, goal, slot_id, res)
            return res

        # 4. Optional lightweight generator callable
        if generator_fn is not None:
            generated = generator_fn(goal, slot)
            if generated:
                self.put(signature_hash, goal, slot_id, generated)
                return generated

        return ""
