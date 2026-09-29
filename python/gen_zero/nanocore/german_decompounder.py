"""German compound splitting and a hyperbolic radius floor.

German writes compound nouns as one word (Donaudampfschiff). A subword
tokenizer sees each compound as a rare token, so semantically close
intents drift apart. This module splits a compound into root morphemes with
a small dictionary plus linking-element heuristics, and provides a radius
floor for Poincare-ball points so short or weak signals do not collapse
into the origin (the general centroid).

The splitter is a heuristic, not a morphological analyser. A word that
cannot be covered fully by known roots is returned unsplit.
"""

from __future__ import annotations

import math
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

import numpy as np

__all__ = [
    "DEFAULT_ROOTS",
    "LINKING_ELEMENTS",
    "GermanDecompounder",
    "decompound",
    "project_hyperbolic_floor",
]

# Lowercase (not casefold) keeps the eszett: casefold turns ß into ss.
# Small seed lexicon of common noun/adjective roots. Extend it through the
# `roots` argument; a real deployment should load a full word list.
DEFAULT_ROOTS: FrozenSet[str] = frozenset(
    """
    haus tür fenster schlüssel bund tisch stuhl bett zimmer küche garten
    auto bahn zug schiff fahr rad flug hafen platz straße weg stadt land
    dorf berg tal fluss see meer wasser feuer luft erde sonne mond stern
    buch regal blatt papier stift schule lehrer schüler arzt kranken
    krankenhaus schwester bruder kind mutter vater eltern familie
    hand schuh handschuh kopf fuß bein arm auge ohr nase mund zahn bürste
    kraft werk fabrik arbeit arbeiter zeit tag nacht woche monat jahr
    licht lampe glas flasche milch brot butter käse fleisch wurst obst
    apfel baum saft kuchen tasche geld bank karte kredit konto
    daten bank system netz werk computer maschine lern programm sprache
    wort schatz regel gesetz recht staat regierung volk welt
    kinder spiel zeug sport verein mannschaft ball feld
    donau dampf kapitän mütze lebens mittel versicherung gesellschaft
    unfall wagen wetter bericht vorhersage sicherheit gurt
    """.split()
)

# Fugenelemente, longest first so "ens" wins over "en" and "s".
LINKING_ELEMENTS: Tuple[str, ...] = ("ens", "en", "er", "es", "e", "n", "s", "")

_UMLAUT_BACK = {"ä": "a", "ö": "o", "ü": "u"}
# Links that come with an umlauted plural stem (Buch -> Bücher, Haus -> Häuser).
_UMLAUT_LINKS = frozenset({"er", "e", ""})


def _de_umlaut_last(stem: str) -> Optional[str]:
    """Undo the last umlaut in `stem` (bücher -> bucher), or None if none."""
    for i in range(len(stem) - 1, -1, -1):
        if stem[i] in _UMLAUT_BACK:
            return stem[:i] + _UMLAUT_BACK[stem[i]] + stem[i + 1:]
    return None


class GermanDecompounder:
    """Dictionary-driven compound splitter.

    Splits into the fewest known roots. Ties go to the longer first root.
    Every part must be at least `min_part_len` characters. Non-final parts
    may carry a linking element (Arbeit+s+zeit) or an umlauted plural stem
    (Büch+er+regal). The final part must be a bare root.
    """

    def __init__(
        self,
        roots: Optional[Iterable[str]] = None,
        min_part_len: int = 3,
    ) -> None:
        if min_part_len < 1:
            raise ValueError("min_part_len must be >= 1")
        source = DEFAULT_ROOTS if roots is None else roots
        self.roots: FrozenSet[str] = frozenset(r.lower() for r in source if r)
        self.min_part_len = int(min_part_len)

    def _root_for_chunk(self, chunk: str, final: bool) -> Optional[str]:
        """Return the root that `chunk` spells, or None."""
        if len(chunk) < self.min_part_len:
            return None
        if chunk in self.roots:
            return chunk
        if final:
            return None
        for link in LINKING_ELEMENTS:
            if not link or not chunk.endswith(link):
                continue
            stem = chunk[: -len(link)]
            if len(stem) < self.min_part_len:
                continue
            if stem in self.roots:
                return stem
            if link in _UMLAUT_LINKS:
                plain = _de_umlaut_last(stem)
                if plain is not None and plain in self.roots:
                    return plain
        plain = _de_umlaut_last(chunk)
        if plain is not None and plain in self.roots:
            return plain
        return None

    def split(self, word: str) -> List[str]:
        """Split `word` into lowercase root morphemes.

        Returns `[word.lower()]` when the word is a known root, is too
        short, or cannot be fully covered by known roots.
        """
        if not isinstance(word, str):
            raise TypeError("word must be a str")
        w = word.strip().lower()
        if not w:
            return []
        if w in self.roots or len(w) < 2 * self.min_part_len:
            return [w]

        n = len(w)
        # best[i] = (part_count, -first_part_len, parts) for the suffix w[i:]
        best: List[Optional[Tuple[int, int, Tuple[str, ...]]]] = [None] * (n + 1)
        best[n] = (0, 0, ())
        for i in range(n - 1, -1, -1):
            for j in range(i + self.min_part_len, n + 1):
                nxt = best[j]
                if nxt is None:
                    continue
                root = self._root_for_chunk(w[i:j], final=(j == n))
                if root is None:
                    continue
                cand = (nxt[0] + 1, -(j - i), (root,) + nxt[2])
                if best[i] is None or cand[:2] < best[i][:2]:
                    best[i] = cand
        result = best[0]
        if result is None:
            return [w]
        return list(result[2])

    def is_compound(self, word: str) -> bool:
        return len(self.split(word)) > 1

    def split_text(self, text: str) -> List[str]:
        """Split every whitespace-separated word and flatten the roots."""
        out: List[str] = []
        for token in text.split():
            out.extend(self.split(token))
        return out


_DEFAULT: Optional[GermanDecompounder] = None


def decompound(word: str) -> List[str]:
    """Split `word` with the default lexicon."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = GermanDecompounder()
    return _DEFAULT.split(word)


def project_hyperbolic_floor(
    x: np.ndarray,
    min_radius: float = 0.25,
    max_radius: float = 0.999,
) -> np.ndarray:
    """Clamp Poincare-ball points to the radius band [min_radius, max_radius].

    Points with ||x|| < min_radius move out along their own direction to
    radius `min_radius`. An exact zero vector has no direction, so it goes
    to the first coordinate axis. Points beyond `max_radius` move back
    inside the ball. Accepts shape (d,) or (n, d). Returns float64.
    """
    if not (0.0 < min_radius < max_radius < 1.0):
        raise ValueError("require 0 < min_radius < max_radius < 1")
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim not in (1, 2) or arr.shape[-1] == 0:
        raise ValueError("x must have shape (d,) or (n, d) with d >= 1")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x must be finite")

    rows = arr.reshape(1, -1) if arr.ndim == 1 else arr
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    degenerate = norms < 1e-12
    safe = np.where(degenerate, 1.0, norms)
    dirs = rows / safe
    if np.any(degenerate):
        dirs[degenerate[:, 0]] = 0.0
        dirs[degenerate[:, 0], 0] = 1.0
    radius = np.clip(np.where(degenerate, 0.0, norms), min_radius, max_radius)
    out = dirs * radius
    return out[0] if arr.ndim == 1 else out
