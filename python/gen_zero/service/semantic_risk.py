"""Multilingual safety-risk classifier for the Rust ``zero`` PolicyGate.

The Rust side used to hard-stop only when the request contained one of seven
English substrings ("rm -rf", "bypass_safety", ...). A Chinese request with the
same meaning passed. This module replaces that table with a model judgement.

Method: in-context learning on the Zero backbone (Qwen2.5-0.5B base, the same
weights as :mod:`gen_zero.service.semantic_scorer`).

* The prompt holds labelled demonstrations from ``risk_data/shots.jsonl``
  (English and Chinese), then ``Request: <text>`` and ``Risk:``.
* The score is the calibrated log-odds of the next token ``" dangerous"`` over
  ``" safe"``: the log-odds after the real request minus the log-odds after an
  empty request (domain-conditional PMI, Holtzman et al. 2021), so the prior of
  the label words cancels.
* A base model follows demonstration order, so the log-odds is averaged over
  ``ORDERS`` fixed permutations of the demonstrations.
* Requests longer than one window are split into overlapping token windows;
  the riskiest window decides. Nothing is truncated.

The demonstration KV caches are built once, so one assessment costs one short
forward pass per order and window.

Measured limits (see ``risk_data/README.md`` for the numbers): this is a 0.5B
base model. Its scores separate dangerous from ordinary requests (held-out
AUC is reported, not assumed) but the two classes overlap. The Rust gate
therefore uses two thresholds: a hard stop only at high risk, and a human
confirmation band below it. No keyword list or pattern exists anywhere in this
path.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

DATA_DIR = Path(__file__).with_name("risk_data")
HEADER = (
    "Each request is rated for safety risk. dangerous = destructive, irreversible, "
    "privilege escalation, or bypassing security. safe = ordinary work.\n\n"
)
LABELS = ("safe", "dangerous")
ORDERS = 3
ORDER_SEED = 0
WINDOW_TOKENS = 192
WINDOW_STRIDE = 128

# Calibrated on risk_data/calibration.jsonl (disjoint from the demonstrations
# and from risk_data/heldout.jsonl) by scripts in risk_data/README.md.
# escalate: the lowest score of any dangerous calibration request, minus a
#   margin, so every dangerous calibration request at least needs confirmation.
# hard_stop: above the highest score of any safe calibration request, so no
#   ordinary calibration request is hard-stopped.
ESCALATE_THRESHOLD = 0.4494
HARD_STOP_THRESHOLD = 0.7620


def load_jsonl(name: str) -> List[Dict[str, Any]]:
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


@dataclass(frozen=True)
class RiskAssessment:
    p_dangerous: float
    log_odds: float
    per_order_log_odds: List[float]
    windows: int
    forward_ms: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "p_dangerous": self.p_dangerous,
            "log_odds": self.log_odds,
            "per_order_log_odds": self.per_order_log_odds,
            "windows": self.windows,
            "forward_ms": round(self.forward_ms, 2),
        }


class SemanticRiskClassifier:
    """Few-shot calibrated risk score over the Zero backbone. Thread-safe via the scorer lock."""

    def __init__(self, scorer: Any, shots: Optional[Sequence[Dict[str, Any]]] = None,
                 orders: int = ORDERS, seed: int = ORDER_SEED) -> None:
        self.scorer = scorer
        self.runtime = scorer.runtime
        self.model = scorer.model
        shots = list(shots if shots is not None else load_jsonl("shots.jsonl"))
        if len({s["label"] for s in shots}) != 2:
            raise ValueError("demonstrations need both labels")
        rng = random.Random(seed)
        self.orders: List[List[Dict[str, Any]]] = []
        for k in range(orders):
            order = list(shots)
            if k:
                rng.shuffle(order)
            self.orders.append(order)
        self.label_ids = [self._single_token(" " + name) for name in LABELS]
        self.request_ids = self.runtime.tokenizer.encode("Request: ", add_special_tokens=False).ids
        self.frame_ids = self.runtime.tokenizer.encode("\nRisk:", add_special_tokens=False).ids
        self.classifier_id = f"{self.runtime.encoder_id}/icl-{len(shots)}shot-x{orders}/pmi"
        self._prefix: List[Tuple[torch.Tensor, list]] = []
        self._baseline: List[float] = []

    def _single_token(self, text: str) -> int:
        ids = self.runtime.tokenizer.encode(text, add_special_tokens=False).ids
        if len(ids) != 1:
            raise ValueError(f"label {text!r} must be one token, got {ids}")
        return ids[0]

    def _demo_text(self, order: Sequence[Dict[str, Any]]) -> str:
        return HEADER + "".join(
            f"Request: {s['text']}\nRisk: {LABELS[int(s['label'])]}\n\n" for s in order)

    def _ensure_prefix(self) -> None:
        if self._prefix:
            return
        with torch.inference_mode():
            for order in self.orders:
                ids = torch.tensor([self.runtime.token_ids(self._demo_text(order))], dtype=torch.long)
                mask = torch.ones_like(ids)
                _, cache = self.model(ids, mask, return_cache=True)
                self._prefix.append((mask, cache))
        self._baseline = [self._raw_log_odds(i, []) for i in range(len(self.orders))]

    def _raw_log_odds(self, order_idx: int, text_ids: Sequence[int]) -> float:
        mask, cache = self._prefix[order_idx]
        tail = self.request_ids + list(text_ids) + self.frame_ids
        if mask.shape[1] + len(tail) > self.runtime.max_length:
            raise ValueError("risk window exceeds max_length")
        ids = torch.tensor([tail], dtype=torch.long)
        with torch.inference_mode():
            hidden, _ = self.model(ids, torch.ones_like(ids), past=cache, past_mask=mask)
            lp = self.scorer._log_probs(hidden[0, -1])
        safe_id, danger_id = self.label_ids
        value = float(lp[danger_id] - lp[safe_id])
        if not math.isfinite(value):
            raise FloatingPointError("non-finite risk log-odds")
        return value

    def _windows(self, text_ids: List[int]) -> List[List[int]]:
        if len(text_ids) <= WINDOW_TOKENS:
            return [text_ids]
        out = []
        for start in range(0, len(text_ids), WINDOW_STRIDE):
            out.append(text_ids[start:start + WINDOW_TOKENS])
            if start + WINDOW_TOKENS >= len(text_ids):
                break
        return out

    def assess(self, text: str) -> RiskAssessment:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        text_ids = self.runtime.tokenizer.encode(text.strip(), add_special_tokens=False).ids
        if not text_ids:
            raise ValueError("text tokenized to nothing")
        windows = self._windows(text_ids)
        t0 = time.perf_counter()
        with self.scorer._lock:
            self._ensure_prefix()
            best: Optional[List[float]] = None
            for window in windows:
                per_order = [self._raw_log_odds(i, window) - self._baseline[i]
                             for i in range(len(self.orders))]
                if best is None or sum(per_order) > sum(best):
                    best = per_order
        assert best is not None
        mean = sum(best) / len(best)
        return RiskAssessment(p_dangerous=_sigmoid(mean), log_odds=mean, per_order_log_odds=best,
                              windows=len(windows), forward_ms=(time.perf_counter() - t0) * 1000.0)


_CLASSIFIER: Optional[SemanticRiskClassifier] = None


def get_risk_classifier() -> SemanticRiskClassifier:
    """Build once per process on top of the shared semantic scorer."""
    global _CLASSIFIER
    if _CLASSIFIER is None:
        from gen_zero.service.semantic_scorer import get_semantic_scorer

        _CLASSIFIER = SemanticRiskClassifier(get_semantic_scorer())
    return _CLASSIFIER
