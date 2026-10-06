"""Verbalizer logit extraction: read a few candidate-token logits, never the whole vocabulary.

A verbalizer maps each class label (yes / no / maybe) to the single-token spellings the
model could emit as its next token ("yes", " yes", "Yes", " Yes"). Scoring a prompt means
one forward pass and then reading only those ids from the final-position logits, reduced
per label with a logsumexp.

Two backends, same contract ``score(prompt) -> (label_scores, n_tokens)``:

* ``LlamaCppVerbalizerExtractor``: GGUF via llama-cpp-python. With ``logits_all=False``
  llama.cpp computes logits for the last position only (the llama.cpp equivalent of
  ``logits_to_keep=1``), and the row is read through a ctypes pointer as a zero-copy numpy
  view: the ~152k-wide vocabulary row is never copied, only ``len(flat_ids)`` floats are.
* ``HFVerbalizerExtractor``: transformers causal LM, ``logits_to_keep=1`` passed to the
  forward call so the LM head runs on one position instead of the whole sequence.

Pure inference. Nothing here fits, trains, or ships weights. Every failure raises.
"""
from __future__ import annotations

import ctypes
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_LABELS = ("yes", "no", "maybe")
PREFIX_ALIGN = 8
# Upstream observation (research CPU runs, not re-measured by this port's unit tests): the
# llama.cpp CPU matmul is not batch-split invariant, so a prefill split after a token count
# that is not a multiple of 8 moved verbalizer logits by 0.3-0.66, while multiples of 8
# reproduced the one-shot prefill bit for bit. The reusable prefix is therefore cut down to
# a multiple of PREFIX_ALIGN tokens. ``verify_prefix_reuse`` re-checks it on a live model.
SUPPORTED_GGUF_ARCHITECTURES = ("qwen2",)
LLAMA_CPP_PYTHON_REQUIREMENT = "llama-cpp-python>=0.3.36"


class VerbalizerExtractor(Protocol):
    n_ctx: int
    groups: list[list[int]]

    def score(self, prompt: str) -> tuple[np.ndarray, int]: ...


@dataclass(frozen=True)
class VerbalizerSpec:
    labels: tuple[str, ...] = DEFAULT_LABELS

    def spellings(self, label: str) -> tuple[str, ...]:
        return (label, label.title())


def candidate_groups(encode: Callable[[str], Sequence[int]], spec: VerbalizerSpec = VerbalizerSpec()) -> list[list[int]]:
    """One id group per label: every spelling x {no space, leading space} that is ONE token.

    Ids are unique across labels. A label with no single-token spelling raises: scoring
    with a label that can never win would silently skew the contrast.
    """
    if len(set(spec.labels)) != len(spec.labels) or len(spec.labels) < 2:
        raise ValueError("verbalizer needs at least two distinct labels")
    groups: list[list[int]] = []
    seen: set[int] = set()
    for label in spec.labels:
        ids: list[int] = []
        for spelling in spec.spellings(label):
            for prefix in ("", " "):
                sequence = list(encode(prefix + spelling))
                if len(sequence) == 1 and sequence[0] not in seen:
                    ids.append(int(sequence[0]))
                    seen.add(int(sequence[0]))
        if not ids:
            raise ValueError(f"no single-token verbalizer for label {label!r}")
        groups.append(ids)
    return groups


def group_bounds(groups: Sequence[Sequence[int]]) -> np.ndarray:
    return np.cumsum([0] + [len(g) for g in groups])


def reduce_groups(values: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """logsumexp of the candidate logits of each label."""
    scores = np.array([np.logaddexp.reduce(values[a:b]) for a, b in zip(bounds[:-1], bounds[1:])])
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite verbalizer logits")
    return scores


def view_logits(ptr, n_vocab: int) -> np.ndarray:
    """Zero-copy float32 view of a C ``float*`` logits row. Valid only until the next eval."""
    if n_vocab < 1:
        raise ValueError("n_vocab must be positive")
    if not ptr:
        raise RuntimeError("llama.cpp returned a NULL logits pointer (no logits were computed)")
    return np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)), shape=(n_vocab,))


def gather_scores(row: np.ndarray, flat_ids: Sequence[int], bounds: np.ndarray) -> np.ndarray:
    """Copy only the candidate ids out of the row (a few floats), then reduce per label."""
    return reduce_groups(row[list(flat_ids)].astype(np.float64), bounds)


def common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class LlamaCppVerbalizerExtractor:
    """GGUF extractor with a locked, KV-cached instruction prefix.

    ``prefix`` is the fixed instruction text every prompt starts with. It is prefilled once;
    each later prompt reuses the KV rows of its exactly-matching leading tokens.
    """

    def __init__(self, gguf_path, *, spec: VerbalizerSpec = VerbalizerSpec(), prefix: str = "",
                 n_ctx: int = 2048, n_threads: int | None = None, n_batch: int = 512,
                 n_ubatch: int = 512, use_mmap: bool = True):
        import os

        gguf_path = Path(gguf_path)
        if not gguf_path.is_file():
            raise FileNotFoundError(f"GGUF model not found: {gguf_path}")
        if n_ctx < 1:
            raise ValueError("n_ctx must be positive")
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(f"GGUF inference needs {LLAMA_CPP_PYTHON_REQUIREMENT}") from exc
        threads = n_threads or os.cpu_count() or 1
        self.n_ctx = n_ctx
        self.reused_tokens = 0
        # logits_all=False: llama.cpp keeps only the last position's logits.
        self.llm = Llama(model_path=str(gguf_path), n_ctx=n_ctx, n_batch=n_batch, n_ubatch=n_ubatch,
                         n_threads=threads, n_threads_batch=threads, logits_all=False,
                         use_mmap=use_mmap, use_mlock=False, verbose=False)
        self._check_llama_internals()
        arch = self.llm.metadata.get("general.architecture")
        if arch not in SUPPORTED_GGUF_ARCHITECTURES:
            raise ValueError(f"GGUF architecture {arch!r} unsupported: prefix KV reuse needs a plain "
                             f"transformer from {SUPPORTED_GGUF_ARCHITECTURES}")
        self.n_vocab = int(self.llm.n_vocab())
        self.groups = candidate_groups(self._encode, spec)
        self.flat_ids = [i for g in self.groups for i in g]
        self.bounds = group_bounds(self.groups)
        if max(self.flat_ids) >= self.n_vocab:
            raise ValueError("verbalizer id outside the GGUF vocab")
        self.prefix_ids: list[int] = []
        if prefix:
            self._lock_prefix(prefix)

    def _check_llama_internals(self) -> None:
        # Zero-copy logits and KV truncation use llama-cpp-python internals; fail loudly if the
        # installed version lacks them. There is no fallback to llm.scores (a full-vocab copy).
        missing = [name for name in ("_ctx", "n_tokens", "eval", "reset", "tokenize", "metadata", "n_vocab")
                   if not hasattr(self.llm, name)]
        if missing or not hasattr(self.llm._ctx, "get_logits"):
            raise RuntimeError(f"installed llama_cpp lacks {missing or ['_ctx.get_logits']}; "
                               f"install {LLAMA_CPP_PYTHON_REQUIREMENT}")

    def _encode(self, text: str) -> list[int]:
        return list(self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))

    def _lock_prefix(self, prefix: str) -> None:
        ids = self._encode(prefix)
        ids = ids[:len(ids) // PREFIX_ALIGN * PREFIX_ALIGN]
        if not PREFIX_ALIGN <= len(ids) < self.n_ctx:
            raise ValueError(f"instruction prefix keeps {len(ids)} tokens, n_ctx is {self.n_ctx}")
        self.prefix_ids = ids
        self.llm.reset()
        self.llm.eval(ids)

    def score(self, prompt: str, *, reuse_prefix: bool = True) -> tuple[np.ndarray, int]:
        ids = self._encode(prompt)
        if not 1 <= len(ids) <= self.n_ctx:
            raise ValueError(f"input has {len(ids)} tokens, n_ctx is {self.n_ctx}")
        keep = 0
        if reuse_prefix and self.prefix_ids:
            # A BPE merge across the prefix boundary shortens the match, it never corrupts it.
            keep = min(common_prefix_len(ids, self.prefix_ids), len(ids) - 1)
            keep = keep // PREFIX_ALIGN * PREFIX_ALIGN
        self.llm.n_tokens = keep  # Llama.eval() drops KV rows from n_tokens onward
        self.llm.eval(ids[keep:])
        row = view_logits(self.llm._ctx.get_logits(), self.n_vocab)
        scores = gather_scores(row, self.flat_ids, self.bounds)
        self.reused_tokens += keep
        return scores, len(ids)

    def verify_prefix_reuse(self, prompt: str, atol: float = 1e-4) -> float:
        """Max |logit difference| between prefix-reused and full prefill. Raises if above atol."""
        full, _ = self.score(prompt, reuse_prefix=False)
        reused, _ = self.score(prompt, reuse_prefix=True)
        err = float(np.max(np.abs(full - reused)))
        if err > atol:
            raise RuntimeError(f"prefix KV reuse changes verbalizer logits by {err:.3g} > {atol}")
        return err


class HFVerbalizerExtractor:
    """transformers causal LM extractor; the LM head runs on the last position only."""

    def __init__(self, model, tokenizer, *, spec: VerbalizerSpec = VerbalizerSpec(), n_ctx: int | None = None):
        import torch

        self._torch = torch
        self.model = model.eval()
        self.tokenizer = tokenizer
        limit = int(getattr(model.config, "max_position_embeddings", 0))
        self.n_ctx = int(n_ctx or limit)
        if self.n_ctx < 1 or (limit and self.n_ctx > limit):
            raise ValueError(f"n_ctx {self.n_ctx} outside model limit {limit}")
        self.groups = candidate_groups(lambda t: tokenizer.encode(t, add_special_tokens=False), spec)
        self.flat_ids = [i for g in self.groups for i in g]
        self.bounds = group_bounds(self.groups)
        if max(self.flat_ids) >= int(model.config.vocab_size):
            raise ValueError("verbalizer id outside the model vocab")

    def score(self, prompt: str) -> tuple[np.ndarray, int]:
        torch = self._torch
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        n = int(encoded["input_ids"].shape[1])
        if not 1 <= n <= self.n_ctx:
            raise ValueError(f"input has {n} tokens, n_ctx is {self.n_ctx}")
        with torch.inference_mode():
            out = self.model(**encoded, logits_to_keep=1)
        if out.logits.shape[1] != 1:
            raise RuntimeError(f"logits_to_keep=1 ignored by {type(self.model).__name__}: "
                               f"got {out.logits.shape[1]} positions")
        row = out.logits[0, -1].float().numpy()
        return gather_scores(row, self.flat_ids, self.bounds), n

