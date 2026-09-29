"""Single-Token Stability Assertion and Canonical Label Mapping.

Implements Milestone 2 of Issue #24:
- TokenFragmentationError: Exception raised when candidate label fragments into multiple subwords.
- assert_single_token_stability: Enforces strictly 1 token ID per candidate label under prefix context.
- CanonicalLabelMapper: Maps arbitrary multi-word labels to canonical single-token identifiers (A, B, C... / 1, 2, 3...)
  and restores structured outputs back to domain labels.
"""

from typing import List, Dict, Any, Optional, Tuple, Callable


class TokenFragmentationError(Exception):
    """Raised when a candidate decision label fragments into >= 2 tokens under the tokenizer."""
    pass


def assert_single_token_stability(
    tokenizer: Any,
    label_str: str,
    prefix_ids: Optional[List[int]] = None,
) -> int:
    """Strictly asserts that a candidate label corresponds to exactly ONE token ID.

    Args:
        tokenizer: Tokenizer instance implementing encode() or a callable.
        label_str: The candidate label string (e.g. "A", "B", "1").
        prefix_ids: Optional list of prefix token IDs for context-sensitive tokenization.

    Returns:
        int: The single unique token ID.

    Raises:
        TypeError: If tokenizer is None, or implements neither encode() nor
            __call__(). A fake hash-based token id must never stand in for a
            real tokenizer's vocabulary mapping: it would silently corrupt
            next-token logits projection with an id disconnected from the
            model's actual vocabulary.
        TokenFragmentationError: If the label tokenizes into != 1 token.
    """
    if tokenizer is None:
        raise TypeError(
            "assert_single_token_stability requires a real tokenizer; got None. "
            "A fake hash-based token id must never stand in for an actual "
            "tokenizer's vocabulary mapping."
        )

    if hasattr(tokenizer, "encode"):
        token_ids = tokenizer.encode(label_str, add_special_tokens=False)
    elif callable(tokenizer):
        token_ids = tokenizer(label_str)
    else:
        raise TypeError(
            f"tokenizer of type {type(tokenizer).__name__!r} implements neither "
            f"encode() nor __call__(); cannot tokenize {label_str!r} without "
            f"fabricating a hash-based token id."
        )

    if len(token_ids) != 1:
        raise TokenFragmentationError(
            f"Label '{label_str}' fragmented into {len(token_ids)} tokens ({token_ids}). "
            f"Next-Token Logits projection requires strictly 1 single token. "
            f"Please map to canonical single-token labels (e.g., 'A', 'B', 'C') via CanonicalLabelMapper."
        )

    return int(token_ids[0])


class CanonicalLabelMapper:
    """Maps arbitrary multi-token strings to guaranteed single-token canonical keys and back."""

    DEFAULT_ALPHABETIC_KEYS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
    DEFAULT_NUMERIC_KEYS = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

    def __init__(self, key_type: str = "alphabetic"):
        self.keys = self.DEFAULT_ALPHABETIC_KEYS if key_type == "alphabetic" else self.DEFAULT_NUMERIC_KEYS
        self.label_to_canonical: Dict[str, str] = {}
        self.canonical_to_label: Dict[str, str] = {}

    def map_labels(self, original_labels: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """Maps a list of arbitrary label strings to canonical single-token identifiers.

        Returns:
            Tuple of (canonical_keys: List[str], mapping_prompt_lines: Dict[str, str]).
        """
        self.label_to_canonical.clear()
        self.canonical_to_label.clear()

        canonical_keys = []
        for i, original in enumerate(original_labels):
            key = self.keys[i] if i < len(self.keys) else f"X{i}"
            self.label_to_canonical[original] = key
            self.canonical_to_label[key] = original
            canonical_keys.append(key)

        return canonical_keys, dict(self.canonical_to_label)

    def decode_choice(self, canonical_choice: str) -> str:
        """Restores the original label string from the canonical choice."""
        return self.canonical_to_label.get(canonical_choice, canonical_choice)

    def decode_distribution(self, canonical_probs: Dict[str, float]) -> Dict[str, float]:
        """Restores the full probability distribution mapped back to original label keys."""
        return {
            self.canonical_to_label.get(k, k): v
            for k, v in canonical_probs.items()
        }
