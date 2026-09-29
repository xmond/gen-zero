"""Semantic candidate scoring for the Rust ``zero`` tool bridge.

The scorer runs the local Zero backbone (the Qwen2.5-0.5B trunk loaded by
``ZeroStandaloneRuntime``) as a causal language model. Qwen2.5-0.5B ties its
input embedding and its output head, so ``final_norm(h) @ embed_tokens.T`` is
the exact next-token distribution of the pretrained model. No head is trained
or invented here.

Decision rule (calibrated likelihood, Holtzman et al. 2021 "domain-conditional PMI"):

    score(c) = log P(c | context + premise) - log P(c | premise)

The premise is the prompt without the context: the frame sentence, plus the
action history during multi-step lookahead. The second term removes the prior
of the candidate string, so a frequent word
such as "delete" does not win only because it is frequent. The frame does not
list the candidates: a listed option set makes a 0.5B model copy the first
option (measured: the winner followed list position in 4 of 6 permutations).

What this is not
    * No manifold. ``ZeroStandaloneRuntime.decide`` needs a manifold fitted on
      calibration data and none ships with the repository, so this module does
      not claim manifold decisions.
    * No keyword table, no hash, no fallback model. If the weights are missing
      the scorer raises and the HTTP layer answers 503; the Rust caller then
      labels its own local fallback.
"""
from __future__ import annotations

import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

ASK_FRAME = "The first action to take is:"
NEXT_FRAME = "The next action to take is:"
ROUTE_FRAME = "The tool to use is:"
# An action frame after a question asks the model the wrong thing, so QA
# contexts get an answer frame instead.
QA_FRAME = "The correct answer is:"
QA_CONTEXT_MARKERS = ("Question:", "Answer with")
QA_CANDIDATE_SETS = (frozenset({"yes", "no"}), frozenset({"true", "false"}))
MAX_CANDIDATES = 64
_BASELINE_CACHE_SIZE = 512


@dataclass(frozen=True)
class CandidateScore:
    name: str
    log_likelihood: float
    baseline_log_likelihood: float
    pmi: float
    probability: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "log_likelihood": self.log_likelihood,
            "baseline_log_likelihood": self.baseline_log_likelihood,
            "pmi": self.pmi,
            "probability": self.probability,
        }


@dataclass(frozen=True)
class ScoreResult:
    candidates: List[CandidateScore]
    entropy: float
    prompt_state: List[float]
    prompt_tokens: int
    forward_ms: float

    @property
    def chosen_index(self) -> int:
        return max(range(len(self.candidates)), key=lambda i: self.candidates[i].probability)


def candidate_text(name: str) -> str:
    """Continuation text for one candidate. Same rule for every language."""
    return " " + name.replace("_", " ").strip()


def normalized_entropy(probs: Sequence[float]) -> float:
    if len(probs) <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in probs if p > 0.0)
    return min(1.0, max(0.0, h / math.log(len(probs))))


def softmax(values: Sequence[float]) -> List[float]:
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def select_ask_frame(context: str, candidates: Sequence[str], history: Sequence[str] = (),
                     explicit: Optional[str] = None) -> tuple:
    """Return ``(frame, source)`` for a semantic_ask call.

    Priority: explicit frame, then QA context, then history, then ASK_FRAME.
    An explicit frame that is blank raises ValueError; it never falls back.
    """
    if explicit is not None:
        if not explicit.strip():
            raise ValueError("frame must be a non-empty string when given")
        return explicit.strip(), "explicit"
    names = frozenset(c.strip().lower() for c in candidates)
    if any(m in context for m in QA_CONTEXT_MARKERS) or names in QA_CANDIDATE_SETS:
        return QA_FRAME, "qa"
    if history:
        return NEXT_FRAME, "history"
    return ASK_FRAME, "default"


def compose_prompt(context: str, frame: str, history: Sequence[str] = ()) -> str:
    parts = [context.strip()] if context and context.strip() else []
    if history:
        parts.append("Actions already taken: " + ", ".join(h.strip() for h in history))
    parts.append(frame)
    return "\n".join(parts)


class SemanticScorer:
    """Calibrated log-likelihood scorer over the Zero backbone. Thread-safe."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.model = runtime.model
        self.tokenizer = runtime.tokenizer
        self.lm_head = self.model.embed_tokens.weight  # tied output head (V, H), bf16 storage
        self.hidden_size = int(runtime.hidden_size)
        self.scorer_id = f"{runtime.encoder_id}/tied-lm-head/pmi"
        self._lock = threading.Lock()
        self._baseline: "OrderedDict[tuple, List[float]]" = OrderedDict()

    # -- low level ----------------------------------------------------------

    def _ids(self, text: str) -> List[int]:
        return self.runtime.token_ids(text)

    def _log_probs(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = hidden.to(torch.float32) @ self.lm_head.to(torch.float32).T
        return torch.log_softmax(logits, dim=-1)

    def _continuation_log_likelihoods(self, prompt: str, texts: Sequence[str]
                                      ) -> tuple[List[float], torch.Tensor, int]:
        """Sum of token log-probs of each continuation after ``prompt``.

        One prompt pass fills the KV cache; all continuations then run as one
        right-padded batch on top of it (same computation as K full sequences
        under causal attention).
        """
        prompt_ids = self._ids(prompt)
        rows = [self.tokenizer.encode(t, add_special_tokens=False).ids for t in texts]
        if any(not r for r in rows):
            raise ValueError("a candidate tokenized to nothing")
        if any(len(prompt_ids) + len(r) > self.runtime.max_length for r in rows):
            raise ValueError("prompt + candidate exceeds max_length; refusing to truncate")
        cand_ids, cand_mask = self.runtime._pad(rows)
        with torch.inference_mode():
            p_ids = torch.tensor([prompt_ids], dtype=torch.long)
            p_mask = torch.ones_like(p_ids)
            hidden, cache = self.model(p_ids, p_mask, return_cache=True)
            last = hidden[0, -1]
            first_lp = self._log_probs(last)
            n = len(rows)
            expanded = [(k.expand(n, -1, -1, -1), v.expand(n, -1, -1, -1)) for k, v in cache]
            hidden_c, _ = self.model(cand_ids, cand_mask, past=expanded, past_mask=p_mask.expand(n, -1))
            cont_lp = self._log_probs(hidden_c)
            scores: List[float] = []
            for i, row in enumerate(rows):
                total = float(first_lp[row[0]])
                for j in range(1, len(row)):
                    total += float(cont_lp[i, j - 1, row[j]])
                scores.append(total)
        if not all(math.isfinite(s) for s in scores):
            raise FloatingPointError("non-finite candidate log-likelihood")
        return scores, last.to(torch.float32), len(prompt_ids)

    def _baseline_scores(self, premise: str, texts: Sequence[str]) -> List[float]:
        key = (premise, tuple(texts))
        cached = self._baseline.get(key)
        if cached is not None:
            self._baseline.move_to_end(key)
            return cached
        scores, _, _ = self._continuation_log_likelihoods(premise, texts)
        self._baseline[key] = scores
        if len(self._baseline) > _BASELINE_CACHE_SIZE:
            self._baseline.popitem(last=False)
        return scores

    # -- public -------------------------------------------------------------

    def score(self, context: str, names: Sequence[str], *, frame: str,
              history: Sequence[str] = (), texts: Optional[Sequence[str]] = None) -> ScoreResult:
        names = list(names)
        if len(names) < 2:
            raise ValueError("need at least two candidates")
        if len(names) > MAX_CANDIDATES:
            raise ValueError(f"at most {MAX_CANDIDATES} candidates")
        if len(set(names)) != len(names):
            raise ValueError("candidates must be distinct")
        if any(not isinstance(n, str) or not n.strip() for n in names):
            raise ValueError("candidates must be non-empty strings")
        if not context or not context.strip():
            raise ValueError("context must be a non-empty string")
        texts = list(texts) if texts is not None else [candidate_text(n) for n in names]
        prompt = compose_prompt(context, frame, history)
        t0 = time.perf_counter()
        with self._lock:
            conditional, state, prompt_tokens = self._continuation_log_likelihoods(prompt, texts)
            # Domain premise = the same prompt minus the context (history kept),
            # so the PMI measures what the context adds, nothing else.
            baseline = self._baseline_scores(compose_prompt("", frame, history), texts)
        forward_ms = (time.perf_counter() - t0) * 1000.0
        pmi = [c - b for c, b in zip(conditional, baseline)]
        probs = softmax(pmi)
        cands = [CandidateScore(n, c, b, p_, pr) for n, c, b, p_, pr
                 in zip(names, conditional, baseline, pmi, probs)]
        unit = state / state.norm().clamp_min(1e-12)
        return ScoreResult(candidates=cands, entropy=normalized_entropy(probs),
                           prompt_state=unit.tolist(), prompt_tokens=prompt_tokens,
                           forward_ms=forward_ms)


_SCORER: Optional[SemanticScorer] = None
_SCORER_LOCK = threading.Lock()


def get_semantic_scorer() -> SemanticScorer:
    """Load the backbone once per process. Raises if the weights are missing."""
    global _SCORER
    if _SCORER is not None:
        return _SCORER
    with _SCORER_LOCK:
        if _SCORER is None:
            from gen_zero.causal.zero_runtime import ZeroStandaloneRuntime

            precision = os.environ.get("GENZERO_SEMANTIC_PRECISION", "fp32")
            artifact = os.environ.get("GENZERO_SEMANTIC_INT8_ARTIFACT")
            threads = os.environ.get("GENZERO_SEMANTIC_THREADS")
            runtime = ZeroStandaloneRuntime(
                precision=precision,
                int8_artifact=Path(artifact) if artifact else None,
                num_threads=int(threads) if threads else None,
            )
            _SCORER = SemanticScorer(runtime)
    return _SCORER
