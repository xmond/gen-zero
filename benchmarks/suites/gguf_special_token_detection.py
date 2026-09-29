"""Detect a llama-server tokenizer's special-token wrapping around plain content.

Why this exists. Some embedding servers only detect a BOS-style
PREFIX: it diffs ``tokenize(add_special=True)`` against ``tokenize(add_special=False)`` and keeps
the leading extra tokens, else sets prefix=[]. That is correct for Gemma and Qwen2.5-72B, both of
which prepend BOS. It is silently wrong for a tokenizer that instead appends a SUFFIX (measured on
GTE-Qwen2-7B, cpu_extract_gte7b_13tasks.py: add_special=True on "hello world test" gave
[14990,1879,1273,151643], add_special=False gave [14990,1879,1273]; the extra token 151643 is
appended, not prepended). The prefix-only check would set prefix=[] and drop that token entirely --
fatal under pooling=last, since the pooled vector IS the last token's state (measured cosine
0.41-0.65 against the server's own text path, vs 1.0 once the suffix is restored).

Llama-3.1, Mixtral and DeepSeek-V2 tokenizers cannot be probed from this box (no GPU, no
llama-server here). This module makes no assumption about which pattern any of them use: it
computes the split from two live token sequences and refuses (raises ValueError, never silently
picks prefix-only) when ``plain`` is not a contiguous slice of ``with_special``.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple


def detect_special_tokens(with_special: Sequence[int], plain: Sequence[int]) -> Tuple[List[int], List[int]]:
    """Split ``with_special`` into (prefix, suffix) around the contiguous ``plain`` slice.

    ``with_special`` = tokenize(text, add_special=True), ``plain`` = tokenize(text, add_special=False)
    for the SAME text. Returns (prefix, suffix), either of which may be empty. Every caller must
    apply both: ``prefix + truncate(raw_ids) + suffix``, never prefix alone.

    Raises ValueError if ``plain`` does not appear as a contiguous run inside ``with_special``
    (e.g. the tokenizer rewrites content tokens under add_special, not just wraps them) -- this
    must stop the run, not fall back to a partial guess. When ``plain`` occurs more than once, the
    leftmost occurrence is used (deterministic; a tokenizer that duplicates a probe string this way
    would be unusual enough to warrant its own investigation).
    """
    with_special = list(with_special)
    plain = list(plain)
    if not plain:
        raise ValueError("plain token sequence is empty; cannot locate it inside with_special")
    n = len(plain)
    for start in range(len(with_special) - n + 1):
        if with_special[start:start + n] == plain:
            return with_special[:start], with_special[start + n:]
    raise ValueError(
        "plain is not a contiguous slice of with_special; the tokenizer does not just wrap plain "
        f"content with a fixed prefix/suffix -- inspect both directly. plain={plain} "
        f"with_special={with_special}"
    )
