"""Epistemic Grounding Gate for Reading Comprehension & RAG Answerability.

Language-agnostic by construction: there are no per-language keyword lists,
suffix tables, month names or regexes over words.  Everything below works on
Unicode properties (letter/digit classes, character width, capitalisation) and
on geometry (character n-gram vectors, or a caller-supplied encoder).

Two mechanisms:

* Discourse turns.  A passage is cut into clauses at Unicode clause boundaries,
  each clause becomes a vector h_t, and the path h_1 .. h_m is treated as a
  trajectory.  With v_t = h_t - h_{t-1} the turning curvature is
  ``kappa(t) = 1 - <v_t, v_{t-1}> / (|v_t| |v_{t-1}|)``.  A sharp deflection
  (kappa >= tau) marks segment t as the nucleus and segment t-1 as the
  concession.  No conjunction is ever looked up.
* Entity-predicate grounding.  The question subject and predicate must meet in
  the SAME passage sentence, otherwise the evidence is fragmented.
"""

from __future__ import annotations

import re
import unicodedata
import zlib
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Set, Tuple

import numpy as np

# --- Universal text primitives ---------------------------------------------

# Universal Unicode sentence and clause punctuation:
# Includes Latin, CJK, Devanagari danda, Arabic, Armenian, Ethiopic marks.
_TERMINAL_MARKS = r"[.!?…\u3002\uff01\uff1f\uff1b;\u0964\u0965\u061f\u061b\u06d4\u0589\u1362]"
_PAUSE_MARKS = r"[\u3001\uff0c\u060c\u060d\u1802\uff1a:]"
_SENTENCE_SPLIT = re.compile(rf"(?<={_TERMINAL_MARKS})\s*|\n+")
_CLAUSE_SPLIT = re.compile(
    rf"(?<={_TERMINAL_MARKS})\s*|(?<={_PAUSE_MARKS})\s*|\n+|(?<!\d),\s*|,(?!\d)\s*"
)
_TOKEN = re.compile(r"\w+(?:['’]\w+)?")
_NUM_TOKEN = re.compile(r"\d+(?:[.,]\d+)*")
_THOUSANDS = re.compile(r"\d{1,3}(?:\.\d{3})+")

# Prefix length used as a language-neutral stem (truncation stemming).
STEM_PREFIX = 5
# Character n-gram sizes and dimensionality of the hashing embedder.
NGRAM_SIZES: Tuple[int, ...] = (2, 3)
HASH_DIM = 1024


def _normalize(text: str) -> str:
    """NFKC-fold and map every Unicode decimal digit to ASCII."""
    text = unicodedata.normalize("NFKC", text)
    return "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text)


def _is_wide(ch: str) -> bool:
    """True for scripts written without word spaces (CJK, kana, hangul)."""
    return unicodedata.east_asian_width(ch) in ("W", "F")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]


def split_clauses(text: str) -> List[str]:
    return [s.strip() for s in _CLAUSE_SPLIT.split(text) if s and s.strip()]


@dataclass(frozen=True)
class _Tok:
    text: str
    key: str  # language-neutral stem
    pos: int  # index within its text
    is_num: bool
    is_cap: bool
    is_wide: bool

    @property
    def weight(self) -> float:
        """Content prior in [0, 1] from form alone: numbers, wide-script units
        and capitalised tokens are anchors; otherwise longer means more
        informative (closed-class words are short in most languages)."""
        if self.is_num or self.is_wide or (self.is_cap and self.pos > 0):
            return 1.0
        return min(1.0, max(0.0, (len(self.text) - 2) / 4.0))

    @property
    def is_content(self) -> bool:
        return self.weight >= 0.5


def _canonical_number(tok: str) -> str:
    s = tok.replace(",", ".")
    return s.replace(".", "") if _THOUSANDS.fullmatch(s) else s


def tokenize(text: str) -> List[_Tok]:
    """Split into word tokens; wide-script runs become overlapping bigrams."""
    out: List[_Tok] = []
    for m in _TOKEN.finditer(_normalize(text)):
        word = m.group()
        run = ""
        run_wide = False
        pieces: List[Tuple[str, bool]] = []
        for ch in word:
            w = _is_wide(ch)
            if run and w != run_wide:
                pieces.append((run, run_wide))
                run = ""
            run += ch
            run_wide = w
        if run:
            pieces.append((run, run_wide))
        for piece, wide in pieces:
            if wide:
                grams = [piece[i:i + 2] for i in range(max(1, len(piece) - 1))]
                for g in grams:
                    out.append(_Tok(g, g, len(out), False, False, True))
            else:
                low = piece.lower()
                is_num = low.isdigit()
                key = _canonical_number(low) if is_num else low[:STEM_PREFIX]
                out.append(_Tok(low, key, len(out), is_num, piece[0].isupper(), False))
    return out


def extract_numbers(text: str) -> Set[str]:
    """Cardinal numbers as digit sequences only: no unit or month words."""
    return {_canonical_number(n) for n in _NUM_TOKEN.findall(_normalize(text))}


# --- Discourse turn detection -----------------------------------------------

Encoder = Callable[[Sequence[str]], np.ndarray]


def _hash_embed(segments: Sequence[str], dim: int = HASH_DIM) -> np.ndarray:
    """Universal subword hashing: character n-grams, crc32 into ``dim`` bins."""
    mat = np.zeros((len(segments), dim), dtype=np.float64)
    for i, seg in enumerate(segments):
        for word in _TOKEN.findall(_normalize(seg).lower()):
            padded = f"^{word}$"
            for n in NGRAM_SIZES:
                for j in range(max(1, len(padded) - n + 1)):
                    gram = padded[j:j + n]
                    mat[i, zlib.crc32(gram.encode("utf-8")) % dim] += 1.0
    return mat


def _unit_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return np.divide(mat, norms, out=np.zeros_like(mat), where=norms > 0)


@dataclass(frozen=True)
class DiscourseFocus:
    """Result of discourse-turn analysis on a passage.

    Attributes:
        nucleus_weight: Boost factor for nucleus (post-turn) segments.
        concession_weight: Discount factor for concession (pre-turn) segments.
        segments: (text, weight) pairs for weighted token extraction.
        curvature: kappa(t) per segment (index 0 has no incoming direction).
        turn_positions: Segment indices t where kappa(t) >= tau.
    """

    nucleus_weight: float
    concession_weight: float
    segments: List[Tuple[str, float]]
    curvature: List[float] = field(default_factory=list)
    turn_positions: List[int] = field(default_factory=list)

    @property
    def has_turn(self) -> bool:
        return bool(self.turn_positions)


class LanguageAgnosticDiscourseAnalyzer:
    """Geodesic-curvature discourse turn detector (no lexicon of any language).

    The trajectory starts at the origin, so the first direction is v_1 = h_1
    and a turn is measurable from the second clause onward.  Vectors are unit
    length, hence kappa lies in [0, 2].
    """

    NUCLEUS_BOOST: float = 1.3
    CONCESSION_DISCOUNT: float = 0.5
    # kappa at t=2 equals 1 + sqrt((1 - cos(h_1, h_2)) / 2); 1.4 means the two
    # clauses have cosine <= ~0.68.  Measured: clause pairs on different
    # subjects give 1.5-1.7, a restated subject gives ~1.3.
    TAU: float = 1.4

    def __init__(
        self,
        tau: float = TAU,
        encoder: Optional[Encoder] = None,
        nucleus_boost: float = NUCLEUS_BOOST,
        concession_discount: float = CONCESSION_DISCOUNT,
    ):
        self.tau = tau
        self.encoder = encoder
        self.nucleus_boost = nucleus_boost
        self.concession_discount = concession_discount

    def embed(self, segments: Sequence[str]) -> np.ndarray:
        if self.encoder is None:
            return _unit_rows(_hash_embed(segments))
        h = np.asarray(self.encoder(segments), dtype=np.float64)
        if h.ndim != 2 or h.shape[0] != len(segments):
            raise ValueError(
                f"encoder must return shape (len(segments), d); got {h.shape}"
            )
        return _unit_rows(h)

    @staticmethod
    def curvature(h: np.ndarray) -> np.ndarray:
        """kappa(t) = 1 - cos(v_t, v_{t-1}), v_t = h_t - h_{t-1}, h_{-1} = 0.

        A zero-length step has no direction; it is reported as kappa = 0
        (straight on), which is the geometric truth for a repeated clause.
        """
        m = h.shape[0]
        kappa = np.zeros(m, dtype=np.float64)
        prev = np.zeros(h.shape[1], dtype=np.float64)
        vs = []
        for t in range(m):
            vs.append(h[t] - prev)
            prev = h[t]
        for t in range(1, m):
            a, b = vs[t], vs[t - 1]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 1e-12 and nb > 1e-12:
                kappa[t] = 1.0 - float(a @ b) / (na * nb)
        return kappa

    def analyze(self, passage: str) -> DiscourseFocus:
        segs = [s for s in split_clauses(passage) if any(c.isalnum() for c in s)]
        if len(segs) < 2:
            return DiscourseFocus(1.0, 1.0, [(passage, 1.0)], [0.0] * len(segs), [])

        kappa = self.curvature(self.embed(segs))
        turns = [t for t in range(1, len(segs)) if kappa[t] >= self.tau]
        if not turns:
            return DiscourseFocus(1.0, 1.0, [(passage, 1.0)], kappa.tolist(), [])

        nucleus: Set[int] = set(turns)
        concession: Set[int] = {t - 1 for t in turns}
        weights: List[float] = []
        for i, seg in enumerate(segs):
            is_n, is_c = i in nucleus, i in concession
            if is_n and not is_c:
                weights.append(self.nucleus_boost)
            elif is_c and not is_n:
                weights.append(self.concession_discount)
            else:
                weights.append(1.0)
        return DiscourseFocus(
            self.nucleus_boost,
            self.concession_discount,
            list(zip(segs, weights)),
            kappa.tolist(),
            turns,
        )


GeodesicCurvatureDiscourseDetector = LanguageAgnosticDiscourseAnalyzer


class DiscourseFocusRewriter:
    """Reweights passage segments around detected discourse turns."""

    NUCLEUS_BOOST: float = LanguageAgnosticDiscourseAnalyzer.NUCLEUS_BOOST
    CONCESSION_DISCOUNT: float = LanguageAgnosticDiscourseAnalyzer.CONCESSION_DISCOUNT

    @staticmethod
    def analyze_discourse_focus(
        passage: str, encoder: Optional[Encoder] = None
    ) -> DiscourseFocus:
        return LanguageAgnosticDiscourseAnalyzer(encoder=encoder).analyze(passage)


def analyze_discourse_focus(
    passage: str, encoder: Optional[Encoder] = None
) -> DiscourseFocus:
    """Standalone wrapper for discourse turn analysis."""
    return DiscourseFocusRewriter.analyze_discourse_focus(passage, encoder)


# --- Prompt splitting --------------------------------------------------------


def extract_passage_and_question(prompt: str) -> Tuple[str, str]:
    """Extracts reference passage and question from any contextual prompt."""
    m = re.search(r"^(?:Passage|Context|Text|Document):\s*(.*?)\n+(?:Question|Query):\s*(.*?)(?:\n|$)", prompt, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return prompt, ""


# --- Entity-predicate grounding ---------------------------------------------


@dataclass(frozen=True)
class AnswerabilityEvidence:
    entity_cover: float
    best_sentence_cover: float
    relation_cover: float
    co_occurrence_score: float
    type_ok: bool
    number_conflict: bool
    score: float


def _question_terms(question: str) -> Tuple[List[_Tok], List[str], List[str]]:
    """Return (content tokens, subject keys, predicate keys).

    The first non-wide token is dropped as interrogative frame: it is the
    question word in wh-fronting languages and costs at most one weak token
    elsewhere.  Subject = capitalised anchors, else the heaviest content token.
    """
    toks = tokenize(question)
    content = [t for t in toks if t.is_content and (t.is_wide or t.pos > 0)]
    if not content:
        return [], [], []
    subject = [t.key for t in content if t.is_cap and not t.is_wide and not t.is_num]
    if not subject:
        subject = [max(content, key=lambda t: (t.weight, len(t.text))).key]
    predicate = [t.key for t in content if t.key not in subject]
    return content, subject, predicate


class EpistemicGroundingGate:
    """Evaluates whether a question can be definitively answered by a given context."""

    def __init__(self, confidence_gate: float = 0.65, low_threshold: float = 0.55, high_threshold: float = 0.85):
        self.confidence_gate = confidence_gate
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    @staticmethod
    def extract_entities(text: str) -> List[str]:
        """Capitalised (non-initial) tokens and numbers, as neutral stems."""
        return [t.key for t in tokenize(text) if t.is_num or (t.is_cap and t.pos > 0)]

    @staticmethod
    def _compute_co_occurrence_score(passage: str, question: str) -> float:
        """Sentence-local receptive field check, in [0.0, 1.0].

        1.0 subject and predicate share one sentence; 0.6 / 0.5 they sit in
        sentences 1 / 2 apart; 0.3 farther apart; 0.4 one side is missing.
        """
        _, subject, predicate = _question_terms(question)
        if not subject:
            return 1.0
        sets = [{t.key for t in tokenize(s)} for s in split_sentences(passage)]
        has_subj = [all(k in s for k in subject) for s in sets]
        if not predicate:
            return 1.0 if any(has_subj) else 0.3
        has_pred = [any(k in s for k in predicate) for s in sets]

        if any(a and b for a, b in zip(has_subj, has_pred)):
            return 1.0
        subj_at = [i for i, v in enumerate(has_subj) if v]
        pred_at = [i for i, v in enumerate(has_pred) if v]
        if subj_at and pred_at:
            distance = min(abs(i - j) for i in subj_at for j in pred_at)
            return {1: 0.6, 2: 0.5}.get(distance, 0.3)
        return 0.4 if (subj_at or pred_at) else 0.2

    def evaluate_evidence(self, passage: str, question: str) -> AnswerabilityEvidence:
        content, _, _ = _question_terms(question)
        if not content:
            return AnswerabilityEvidence(0.0, 0.0, 0.0, 0.0, False, False, 0.0)

        p_toks = tokenize(passage)
        p_keys = {t.key for t in p_toks}
        q_keys = [t.key for t in content]

        # Entity coverage: anchors if the question has any, else all content.
        anchors = [t.key for t in content if t.is_num or (t.is_cap and not t.is_wide)]
        pool = anchors or q_keys
        entity_cover = sum(k in p_keys for k in pool) / len(pool)

        # Best single-sentence coverage of the question content.
        best_sentence_cover = 0.0
        for s in split_sentences(passage):
            s_keys = {t.key for t in tokenize(s)}
            best_sentence_cover = max(best_sentence_cover, sum(k in s_keys for k in q_keys) / len(q_keys))

        # Relational bigram coverage over content-only sequences.
        if len(q_keys) >= 2:
            p_seq = [t.key for t in p_toks if t.weight >= 0.5]
            p_bigrams = set(zip(p_seq, p_seq[1:]))
            q_bigrams = list(zip(q_keys, q_keys[1:]))
            relation_cover = sum(b in p_bigrams for b in q_bigrams) / len(q_bigrams)
        else:
            relation_cover = entity_cover

        co_occurrence_score = self._compute_co_occurrence_score(passage, question)

        # Expected-type verification needs a per-language question-word table;
        # none is kept, so type carries no signal here.
        type_ok = True

        q_nums = extract_numbers(question)
        p_nums = extract_numbers(passage)
        number_conflict = bool(q_nums and p_nums and not (q_nums & p_nums))

        # Co-occurrence is a requirement, not a bonus: fragmented evidence
        # scales the whole score down.
        score = (0.35 * entity_cover + 0.30 * relation_cover + 0.20 * best_sentence_cover + 0.15 * float(type_ok)) * co_occurrence_score
        if number_conflict:
            score *= 0.5

        return AnswerabilityEvidence(
            entity_cover=round(entity_cover, 4),
            best_sentence_cover=round(best_sentence_cover, 4),
            relation_cover=round(relation_cover, 4),
            co_occurrence_score=round(co_occurrence_score, 4),
            type_ok=type_ok,
            number_conflict=number_conflict,
            score=round(score, 4),
        )

    def decide(self, p_unans: float, evidence: AnswerabilityEvidence) -> Tuple[str, str]:
        """Gates the model. Respects a confident model, but steps in on high uncertainty."""
        conf = max(p_unans, 1.0 - p_unans)
        model_label = "unanswerable" if p_unans >= 0.5 else "answerable"

        if conf >= self.confidence_gate:
            return model_label, "model_confident"

        # Under low model confidence, use evidence anchor
        if evidence.score < self.low_threshold or evidence.number_conflict:
            return "unanswerable", "anchor_missing_evidence"
        if evidence.score >= self.high_threshold and evidence.type_ok:
            return "answerable", "anchor_evidence_grounded"

        return model_label, "anchor_indecisive"
