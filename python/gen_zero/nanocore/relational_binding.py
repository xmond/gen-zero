"""Signed relational-difference binding for adversarial paraphrase pairs (PAWS).

PAWS pairs share about 98% of their words, so a bag-of-words or mean-pooled
representation cannot tell "A tributary of B" from "B tributary of A".
This module tracks *where each entity sits relative to the relation words*
in both sentences and flags an entity role exchange.

A pair is flagged as a role swap when all three checks hold for some entity
pair (a, b):

1. Signed side exchange. Some relation word p is unique in both sentences.
   In sentence 1, a is on one side of p and b on the other. In sentence 2
   the sides are exactly swapped. The signed matrices are
   ``sgn(pos(entity) - pos(p))`` and their difference is anti-symmetric.
2. Relational separator. A shared content word or number sits between a and b.
   Plain coordination ("A and B" -> "B and A") has none, so it stays a
   paraphrase.
3. Bigram binding broke. The (left, right) relation words attached to a and to b
   both change between the sentences. A clause reorder ("J visited R and M
   visited P" -> "M visited P and J visited R") keeps each "<entity> visited"
   binding and is not flagged.

A second, independent signal catches reversed hyphen compounds
("anglo-Egyptian" vs "Egyptian-Anglo").

Limits: adjective swaps ("sparse orchestral" vs "orchestral ... sparse") and
lexical substitutions ("hotels" vs "homes") are not entity role swaps and are
not detected here.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_ARTICLES = frozenset({"the", "a", "an"})
_CONJUNCTIONS = frozenset({"and", "or", "but"})
# Words that cannot carry a relation on their own.
_NON_RELATIONAL = _ARTICLES | _CONJUNCTIONS | frozenset({"also", "as", "that", "which", "who"})
# Capitalised words that are function words when they open a sentence.
_FUNCTION_WORDS = _NON_RELATIONAL | frozenset(
    {"in", "on", "at", "from", "to", "of", "with", "for", "by", "after", "before", "during",
     "he", "she", "it", "they", "we", "i", "his", "her", "their", "its", "this", "these",
     "those", "there", "when", "while", "after", "if", "is", "was", "were", "are",
     "well", "many", "different", "such", "some", "other", "more", "most", "around",
     "approximately", "between", "both", "all"}
)
_SENTENCE_RE = re.compile(
    r"Sentence\s*1\s*:\s*(?P<s1>.*?)\s*\n\s*Sentence\s*2\s*:\s*(?P<s2>.*?)\s*(?:\n|$)",
    re.IGNORECASE | re.DOTALL,
)
_BOUNDARY = "<b>"  # sentence edge or clause-joining conjunction


@dataclass(frozen=True)
class RelationalBindingReport:
    """Outcome of the role-swap check for one sentence pair."""

    has_role_swap: bool
    swapped_entities: Tuple[Tuple[str, str], ...] = ()
    pivots: Tuple[str, ...] = ()
    reversed_compounds: Tuple[str, ...] = ()
    lexical_overlap: float = 0.0
    swap_penalty: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_role_swap": self.has_role_swap,
            "swapped_entities": [list(p) for p in self.swapped_entities],
            "pivots": list(self.pivots),
            "reversed_compounds": list(self.reversed_compounds),
            "lexical_overlap": round(self.lexical_overlap, 4),
            "swap_penalty": self.swap_penalty,
        }


def split_sentence_pair(context: str) -> Optional[Tuple[str, str]]:
    """Extracts (sentence 1, sentence 2) generically from any paired-sentence format."""
    if not context or not context.strip():
        return None
    # 1. Labeled prefixes: Sentence 1/2, Text 1/2, A/B, 1/2, Premise/Hypothesis
    labeled = re.search(
        r"(?:Sentence\s*1|Text\s*1|^[ \t]*A|^[ \t]*1|Premise)\s*:\s*(?P<s1>.*?)\s*\n\s*(?:Sentence\s*2|Text\s*2|^[ \t]*B|^[ \t]*2|Hypothesis)\s*:\s*(?P<s2>.*?)(?:\n|$)",
        context,
        re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    if labeled:
        return labeled.group("s1").strip(), labeled.group("s2").strip()

    # 2. Structural delimiters: |||, [SEP], tabs, or double newlines
    for delim in ("|||", "[SEP]", "\t", "\n\n"):
        if delim in context:
            parts = [p.strip() for p in context.split(delim) if p.strip()]
            if len(parts) == 2:
                return parts[0], parts[1]

    # 3. Single newline split if exactly two non-empty lines
    lines = [line.strip() for line in context.strip().splitlines() if line.strip()]
    if len(lines) == 2:
        return lines[0], lines[1]

    return None


def _tokenize(sentence: str) -> List[str]:
    return [t for t in sentence.split() if t]


def _is_word(tok: str) -> bool:
    return any(ch.isalnum() for ch in tok)


def _is_node_word(tok: str, idx: int) -> bool:
    """A candidate argument-or-pivot token: any content word, letter or digit script.

    No capitalization requirement, so this treats lowercase nouns, German nouns,
    and CJK content words the same as capitalized English proper nouns. Hyphenated
    compounds are excluded here since `_reversed_compounds` handles those
    separately, and a leading function word (e.g. a sentence-opening "The") is
    excluded so it cannot masquerade as an argument.
    """
    if "-" in tok or not _is_word(tok) or not tok.isalnum() or any(ch.isdigit() for ch in tok):
        return False
    low = tok.lower()
    if low in _FUNCTION_WORDS:
        return False
    return True


def _unique_positions(tokens: Sequence[str], pred) -> Dict[str, int]:
    """Maps lowercase token -> position for tokens that satisfy pred and occur exactly once."""
    seen: Dict[str, List[int]] = {}
    for i, t in enumerate(tokens):
        if pred(t, i):
            seen.setdefault(t.lower(), []).append(i)
    return {k: v[0] for k, v in seen.items() if len(v) == 1}


def _binding(tokens: Sequence[str], pos: int, exclude: set, window: int = 6) -> Tuple[str, str]:
    """(left, right) relation words bound to the argument at pos.

    Scans outward and skips articles, punctuation, hyphen compounds and any
    token in `exclude` (the sentence's other candidate argument nodes, so a
    neighbouring argument is never mistaken for the word that governs this
    one). A conjunction or a sentence edge ends the scan, so a clause never
    borrows the verb of its neighbour clause.
    """
    def scan(step: int) -> str:
        j = pos + step
        for _ in range(window):
            if not 0 <= j < len(tokens):
                return _BOUNDARY
            t = tokens[j]
            tl = t.lower()
            if tl in _CONJUNCTIONS:
                return _BOUNDARY
            skip = (not _is_word(t) or tl in _NON_RELATIONAL or "-" in t or tl in exclude)
            if not skip:
                return tl
            j += step
        return _BOUNDARY

    return scan(-1), scan(1)


def _signed_sides(positions: Sequence[int], anchor: int) -> np.ndarray:
    """Signed side (-1 before, +1 after) of each position relative to an anchor position."""
    return np.sign(np.asarray(positions, dtype=np.float64) - float(anchor))


def _is_pure_coordination(toks: Sequence[str], lo: int, hi: int) -> bool:
    span = toks[lo + 1:hi]
    has_comma = any(t in (",", ";", "and", "or") for t in span)
    if has_comma:
        content_words = [
            t for t in span
            if _is_word(t) and t.lower() not in _NON_RELATIONAL and not t[0].isupper()
        ]
        if not content_words:
            return True
    return False


def _separated_by_relation(t1: Sequence[str], t2: Sequence[str], a: str, b: str,
                           p1: Dict[str, int], p2: Dict[str, int], shared: set,
                           exclude: set) -> bool:
    """True when some word common to both sentences sits between a and b.

    That word is the relational separator (a verb, preposition, or "tributary
    of" style phrase) that a and b's positions are read as either side of.
    Plain coordination ("A and B" -> "B and A") has no such separator, since
    "and" is excluded as a non-relational conjunction, so it stays a paraphrase.
    """
    for toks, pos in ((t1, p1), (t2, p2)):
        lo, hi = sorted((pos[a], pos[b]))
        if _is_pure_coordination(toks, lo, hi):
            return False
        for t in toks[lo + 1:hi]:
            tl = t.lower()
            if (tl in shared and _is_word(t) and tl not in _NON_RELATIONAL
                    and tl not in (a, b) and tl not in exclude):
                return True
    return False


def _reversed_compounds(t1: Sequence[str], t2: Sequence[str]) -> List[str]:
    found = []
    parts2 = {tuple(p.lower() for p in t.split("-")) for t in t2 if t.count("-") == 1 and all(t.split("-"))}
    for t in t1:
        if t.count("-") != 1 or not all(t.split("-")):
            continue
        x, y = (p.lower() for p in t.split("-"))
        if x != y and (y, x) in parts2:
            found.append(t)
    return found


def analyze_relational_binding(sentence1: str, sentence2: str,
                               swap_penalty: float = 1.0) -> RelationalBindingReport:
    """Checks whether sentence 2 exchanges entity roles relative to sentence 1."""
    t1, t2 = _tokenize(sentence1), _tokenize(sentence2)
    w1, w2 = {t.lower() for t in t1 if _is_word(t)}, {t.lower() for t in t2 if _is_word(t)}
    overlap = len(w1 & w2) / max(1, len(w1 | w2))

    # Every unique, shared, non-function content word is a candidate node: it
    # can play the argument role (a, b) or the pivot role (p) depending on the
    # triple being tested. No capitalization signal is used, so this treats
    # lowercase nouns, German nouns and non-Latin scripts the same as
    # capitalized English proper nouns.
    nodes1 = _unique_positions(t1, _is_node_word)
    nodes2 = _unique_positions(t2, _is_node_word)
    node_set = sorted(set(nodes1) & set(nodes2))
    shared = w1 & w2

    swapped: List[Tuple[str, str]] = []
    used_pivots: List[str] = []
    if len(node_set) >= 3:
        for p in node_set:
            args = [n for n in node_set if n != p]
            pos1 = [nodes1[e] for e in args]
            pos2 = [nodes2[e] for e in args]
            s1 = _signed_sides(pos1, nodes1[p])
            s2 = _signed_sides(pos2, nodes2[p])
            diff = s2 - s1  # +/-2 where an argument crossed the pivot
            exclude = set(node_set) - {p}  # other nodes never masquerade as a's/b's relation word
            for i in range(len(args)):
                for j in range(i + 1, len(args)):
                    # Opposite crossings: a and b traded sides of p.
                    if diff[i] * diff[j] >= 0 or s1[i] == s1[j]:
                        continue
                    a, b = args[i], args[j]
                    if a in b or b in a:
                        continue
                    if not _separated_by_relation(t1, t2, a, b, nodes1, nodes2, shared, exclude):
                        continue
                    if (_binding(t1, nodes1[a], exclude) == _binding(t2, nodes2[a], exclude)
                            or _binding(t1, nodes1[b], exclude) == _binding(t2, nodes2[b], exclude)):
                        continue
                    if (a, b) not in swapped:
                        swapped.append((a, b))
                    if p not in used_pivots:
                        used_pivots.append(p)

    rev = _reversed_compounds(t1, t2)
    flagged = bool(swapped or rev)
    return RelationalBindingReport(
        has_role_swap=flagged,
        swapped_entities=tuple(swapped),
        pivots=tuple(used_pivots),
        reversed_compounds=tuple(rev),
        lexical_overlap=overlap,
        swap_penalty=float(swap_penalty) if flagged else 0.0,
    )


def analyze_context(context: str, swap_penalty: float = 1.0) -> Optional[RelationalBindingReport]:
    """Runs the binding check on a PAWS-style prompt; None when it holds no sentence pair."""
    pair = split_sentence_pair(context)
    if pair is None:
        return None
    return analyze_relational_binding(pair[0], pair[1], swap_penalty=swap_penalty)
