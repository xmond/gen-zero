"""Contrastive Activation/Decoding (CAD): pure forward inference over verbalizer logits.

Each question is scored twice with the same model:

    cond  = verbalizer logits(prompt with the context)
    prior = verbalizer logits(prompt with the context removed)
    delta = cond - alpha * prior          # what the context added over the model's own prior

``delta`` is turned into yes / no / maybe probabilities by a calibration head. Two heads:

* ``CalibrationHead``: a logistic head over 14 causal-uncertainty features, loaded from a
  JSON file the *user* supplies (no fitted head ships with this repository, and nothing in
  this package can fit one). It is bound to one GGUF file by sha256.
* no head: ``infer_uncalibrated`` applies a temperature softmax to the clipped delta. Its
  output is labelled ``calibrated=False``; alpha, temperature and maybe_bias are untuned
  defaults, so its probabilities are scores, not calibrated confidences.

Clipping (new in this port, not extracted from the research code): ``delta`` is clipped to
``+-delta_clip`` and the head's standardised features to ``+-z_clip``. Both report
``clipped=True`` in the result when they engage, so saturation is never silent.

Pure inference only; no training or optimization routines.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gen_zero.causal.verbalizer_extractor import (
    DEFAULT_LABELS, HFVerbalizerExtractor, LlamaCppVerbalizerExtractor, VerbalizerExtractor, VerbalizerSpec)

LABELS = DEFAULT_LABELS
HEAD_FORMAT = "gen_zero.cad_head.v1"
DEFAULT_ALPHA = 0.5
DEFAULT_DELTA_CLIP = 20.0
DEFAULT_Z_CLIP = 5.0

CAUSAL_FEATURE_NAMES = (
    "diff_yes", "diff_no", "diff_maybe",
    "p_cond_yes", "p_cond_no", "p_cond_maybe",
    "p_prior_yes", "p_prior_no", "p_prior_maybe",
    "H_cond", "H_prior", "delta_H", "KL_cond_prior", "margin_cond",
)
N_CAUSAL_FEATURES = len(CAUSAL_FEATURE_NAMES)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_pair(conditional, unconditional) -> tuple[np.ndarray, np.ndarray]:
    cond = np.asarray(conditional, dtype=np.float64)
    prior = np.asarray(unconditional, dtype=np.float64)
    if cond.shape != prior.shape or cond.shape[-1] != len(LABELS):
        raise ValueError("expected matched yes/no/maybe logits")
    if not np.isfinite(cond).all() or not np.isfinite(prior).all():
        raise ValueError("nonfinite logits")
    return cond, prior


def difference(conditional, unconditional, alpha: float) -> np.ndarray:
    """delta = cond - alpha * prior."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1]")
    cond, prior = _check_pair(conditional, unconditional)
    return cond - alpha * prior


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def extract_causal_features(cond_logits, prior_logits, alpha: float) -> np.ndarray:
    """14-d causal-uncertainty features per row (works on (3,) or (n, 3) input).

    diff(3) | p_cond(3) | p_prior(3) | H_cond | H_prior | delta_H = H_prior - H_cond
    | KL(p_cond || p_prior) | margin_cond = top1 - top2 of p_cond.
    delta_H and KL measure how far the context moved the model from its context-free prior;
    a "maybe" case is one where the context moved it little.
    """
    diff = difference(cond_logits, prior_logits, alpha)
    cond, prior = _check_pair(cond_logits, prior_logits)
    p, q = _softmax(cond), _softmax(prior)
    h_cond = -np.sum(p * np.log(p + 1e-12), axis=-1)
    h_prior = -np.sum(q * np.log(q + 1e-12), axis=-1)
    kl = np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)), axis=-1)
    top2 = np.sort(p, axis=-1)[..., -2:]
    margin = top2[..., 1] - top2[..., 0]
    feats = np.concatenate([diff, p, q, h_cond[..., None], h_prior[..., None],
                            (h_prior - h_cond)[..., None], kl[..., None], margin[..., None]], axis=-1)
    if feats.shape[-1] != N_CAUSAL_FEATURES or not np.isfinite(feats).all():
        raise ValueError("invalid causal features")
    return feats


def _checked_float(value, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid CAD {name}")
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"invalid CAD {name}")
    return value


def _scores_to_probs(scores: np.ndarray, maybe_bias: float, temperature: float) -> np.ndarray:
    scores = scores.copy()
    scores[..., 2] += maybe_bias
    probs = _softmax(scores / temperature)
    if not np.isfinite(probs).all():
        raise ValueError("invalid CAD probabilities")
    return probs


@dataclass(frozen=True)
class CADResult:
    label: str
    probabilities: dict[str, float]
    margin: float                      # top1 - top2 probability
    raw_class_logits: tuple[float, ...]
    unconditional_logits: tuple[float, ...]
    contrastive_logits: tuple[float, ...]
    alpha: float
    calibrated: bool
    clipped: bool
    input_tokens: int

    def to_dict(self) -> dict:
        return {"label": self.label, "probabilities": dict(self.probabilities), "margin": self.margin,
                "raw_class_logits": list(self.raw_class_logits),
                "unconditional_logits": list(self.unconditional_logits),
                "contrastive_logits": list(self.contrastive_logits), "alpha": self.alpha,
                "calibrated": self.calibrated, "clipped": self.clipped,
                "input_tokens": self.input_tokens, "generated_tokens": 0}


def _result(cond, prior, delta, probs, alpha, calibrated, clipped, tokens) -> CADResult:
    order = np.argsort(probs)
    return CADResult(
        label=LABELS[int(order[-1])], probabilities=dict(zip(LABELS, map(float, probs))),
        margin=float(probs[order[-1]] - probs[order[-2]]),
        raw_class_logits=tuple(map(float, cond)), unconditional_logits=tuple(map(float, prior)),
        contrastive_logits=tuple(map(float, delta)), alpha=alpha, calibrated=calibrated,
        clipped=clipped, input_tokens=tokens)


class CalibrationHead:
    """Logistic head over the 14 causal features, bound to one GGUF file. Load-only."""

    def __init__(self, *, alpha, mean, scale, coef, intercept, maybe_bias=0.0, temperature=1.0,
                 z_clip=DEFAULT_Z_CLIP):
        n = N_CAUSAL_FEATURES
        self.alpha = _checked_float(alpha, "alpha", 0.1, 1.0)
        self.maybe_bias = _checked_float(maybe_bias, "maybe_bias", -5.0, 5.0)
        self.temperature = _checked_float(temperature, "temperature", 1e-3, 10.0)
        self.z_clip = _checked_float(z_clip, "z_clip", 1.0, 100.0)
        self.mean, self.scale = np.asarray(mean, dtype=np.float64), np.asarray(scale, dtype=np.float64)
        self.coef, self.intercept = np.asarray(coef, dtype=np.float64), np.asarray(intercept, dtype=np.float64)
        if (self.mean.shape != (n,) or self.scale.shape != (n,) or self.coef.shape != (len(LABELS), n)
                or self.intercept.shape != (len(LABELS),)
                or not all(np.isfinite(x).all() for x in (self.mean, self.scale, self.coef, self.intercept))
                or np.any(self.scale <= 0)):
            raise ValueError("invalid CAD head")

    @classmethod
    def from_json(cls, path, gguf_path) -> "CalibrationHead":
        """Load a user-supplied head and verify it was fitted for exactly this GGUF file."""
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        if config.get("format") != HEAD_FORMAT:
            raise ValueError(f"CAD head format must be {HEAD_FORMAT!r}")
        if config.get("gguf_sha256") != file_sha256(gguf_path):
            raise ValueError("CAD head was fitted on a different GGUF file")
        return cls(alpha=config["alpha"], mean=config["mean"], scale=config["scale"], coef=config["coef"],
                   intercept=config["intercept"], maybe_bias=config.get("maybe_bias", 0.0),
                   temperature=config.get("temperature", 1.0), z_clip=config.get("z_clip", DEFAULT_Z_CLIP))

    def probabilities(self, conditional, unconditional):
        """(causal features, probabilities, clipped) for one row of yes/no/maybe logits."""
        feats = extract_causal_features(conditional, unconditional, self.alpha)
        if feats.shape != (N_CAUSAL_FEATURES,):
            raise ValueError("expected one row of yes/no/maybe logits")
        z = (feats - self.mean) / self.scale
        clipped = bool(np.any(np.abs(z) > self.z_clip))
        z = np.clip(z, -self.z_clip, self.z_clip)
        probs = _scores_to_probs(z @ self.coef.T + self.intercept, self.maybe_bias, self.temperature)
        return feats, probs, clipped


@dataclass(frozen=True)
class PromptTemplate:
    """Conditional prompt = prefix + 'question: Q' + 'context: C' + suffix; the prior drops C."""
    prefix: str = ("Read the context and answer the question. "
                   "Choose exactly one of yes, no, or maybe.\n\nContext and Question:\n")
    context_key: str = "context"
    suffix: str = "\n\nAnswer (yes/no/maybe):"

    def render(self, question: str, context: str) -> str:
        if not question.strip() or not context.strip():
            raise ValueError("question and context must be non-empty")
        return f"{self.prefix}question: {question}\n{self.context_key}: {context}{self.suffix}"

    def render_prior(self, question: str) -> str:
        if not question.strip():
            raise ValueError("question must be non-empty")
        return f"{self.prefix}question: {question}\n{self.context_key}: {self.suffix}"



class CADEngine:
    """Two forward passes (with / without context) and one calibration step per question."""

    def __init__(self, extractor: VerbalizerExtractor, *, template: PromptTemplate = PromptTemplate(),
                 head: CalibrationHead | None = None, alpha: float = DEFAULT_ALPHA,
                 temperature: float = 1.0, maybe_bias: float = 0.0, delta_clip: float = DEFAULT_DELTA_CLIP):
        if len(extractor.groups) != len(LABELS):
            raise ValueError(f"extractor must score exactly {LABELS}")
        self.extractor = extractor
        self.template = template
        self.head = head
        self.alpha = head.alpha if head is not None else _checked_float(alpha, "alpha", 0.0, 1.0)
        self.temperature = _checked_float(temperature, "temperature", 1e-3, 10.0)
        self.maybe_bias = _checked_float(maybe_bias, "maybe_bias", -5.0, 5.0)
        self.delta_clip = _checked_float(delta_clip, "delta_clip", 1.0, 1e4)

    @classmethod
    def from_gguf(cls, gguf_path, *, head_path=None, template: PromptTemplate = PromptTemplate(),
                  n_ctx: int = 2048, n_threads: int | None = None, use_mmap: bool = True,
                  **uncalibrated) -> "CADEngine":
        head = CalibrationHead.from_json(head_path, gguf_path) if head_path else None
        extractor = LlamaCppVerbalizerExtractor(
            gguf_path, spec=VerbalizerSpec(LABELS), prefix=template.prefix, n_ctx=n_ctx,
            n_threads=n_threads, use_mmap=use_mmap)
        return cls(extractor, template=template, head=head, **uncalibrated)

    @classmethod
    def from_hf(cls, model_path, *, tokenizer_path=None, head_path=None, template: PromptTemplate = PromptTemplate(),
                n_ctx: int | None = None, **uncalibrated) -> "CADEngine":
        """Transformers backend (logits_to_keep=1). Paths are local dirs or hub ids; the tokenizer defaults to the model's."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if head_path:
            raise ValueError("CalibrationHead files are bound to a GGUF file by sha256; use from_gguf for heads")
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype="float32")
        extractor = HFVerbalizerExtractor(model, AutoTokenizer.from_pretrained(tokenizer_path or model_path), n_ctx=n_ctx)
        return cls(extractor, template=template, **uncalibrated)

    def raw_logits(self, question: str, context: str):
        """(conditional, prior, total input tokens): the two verbalizer score vectors."""
        cond, n1 = self.extractor.score(self.template.render(question, context))
        prior, n2 = self.extractor.score(self.template.render_prior(question))
        return cond, prior, n1 + n2

    def predict(self, question: str, context: str) -> CADResult:
        """Calibrated prediction. Raises without a head: no silent downgrade to uncalibrated."""
        if self.head is None:
            raise RuntimeError("predict() needs a CalibrationHead; use infer_uncalibrated() for raw CAD scores")
        cond, prior, n = self.raw_logits(question, context)
        feats, probs, clipped = self.head.probabilities(cond, prior)
        return _result(cond, prior, feats[:len(LABELS)], probs, self.head.alpha, True, clipped, n)

    def infer_uncalibrated(self, question: str, context: str) -> CADResult:
        """CAD scores without a fitted head: softmax(clip(delta) [+ maybe_bias] / temperature)."""
        cond, prior, n = self.raw_logits(question, context)
        return self.uncalibrated_from_logits(cond, prior, n)

    def uncalibrated_from_logits(self, cond, prior, tokens: int = 0) -> CADResult:
        delta = difference(cond, prior, self.alpha)
        clipped = bool(np.any(np.abs(delta) > self.delta_clip))
        delta_c = np.clip(delta, -self.delta_clip, self.delta_clip)
        probs = _scores_to_probs(delta_c, self.maybe_bias, self.temperature)
        return _result(np.asarray(cond), np.asarray(prior), delta_c, probs, self.alpha, False, clipped, tokens)

    def classify(self, question: str, context: str) -> CADResult:
        """Production entry: calibrated when a head is loaded, otherwise explicitly uncalibrated.

        Which path ran is carried in ``CADResult.calibrated``; callers must read it.
        """
        return self.predict(question, context) if self.head is not None else self.infer_uncalibrated(question, context)
