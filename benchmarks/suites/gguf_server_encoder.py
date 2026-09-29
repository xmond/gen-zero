"""llama-server (GGUF) encoder with the same interface as the HF ``Encoder`` in
benchmark_01png_grand_challenge.py, so the 13-task grand challenge can be
re-run with a large GGUF teacher instead of Qwen2.5-0.5B.

What it does
  * Truncates every text to the SAME protocol as the 0.5B baseline: first
    ``head_tok`` + last ``max_tok - head_tok`` tokens, counted in the TEACHER's
    own tokenizer (server /tokenize, /detokenize). Without this, any gain could
    come from seeing more context instead of from the teacher.
  * Embeds through /v1/embeddings. The server must run with ``--embedding
    --pooling last`` (the same recipe as the extraction scripts), so the feature
    is ONE pooled last-token vector, not the baseline's mean@L12 + last@L18
    concat. That protocol difference is real and must be stated in any report.
  * Fails loudly on non-finite vectors and on collapsed embeddings (distinct
    texts mapped to identical vectors), the failure mode of the 2026-09-24
    1000-char-cut bug.

Encoder latency here is a GPU server round trip; it is NOT comparable with the
0.5B CPU number. Only the per-task head latency is comparable.
"""
from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Sequence

import numpy as np


def _post(url: str, payload: dict, timeout: float) -> object:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _vectors(payload: object, expected: int) -> List[np.ndarray]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or len(data) != expected:
        raise ValueError(f"embedding response has {len(data) if isinstance(data, list) else 'no'} "
                         f"vectors; expected {expected}")
    data = sorted(data, key=lambda item: item.get("index", 0))
    out = []
    for item in data:
        vec = np.asarray(item["embedding"], dtype=np.float32)
        if vec.ndim != 1 or vec.size == 0:
            raise ValueError(f"unexpected embedding shape {vec.shape}")
        out.append(vec)
    return out


class GGUFServerEncoder:
    def __init__(self, base_url: str, max_tok: int, head_tok: int, workers: int = 4,
                 request_batch: int = 8, timeout: float = 600.0) -> None:
        self.base = base_url.rstrip("/")
        self.max_tok, self.head_tok = max_tok, head_tok
        self.workers, self.request_batch, self.timeout = workers, request_batch, timeout
        probe = self._embed(["dimension probe"])
        self.dim = int(probe[0].shape[0])
        self.truncated = 0

    # -- tokenizer-side truncation ------------------------------------------------
    def _truncate(self, text: str) -> tuple:
        """Return (text after head+tail cut, token count actually embedded)."""
        tokens = _post(self.base + "/tokenize", {"content": text, "add_special": False},
                       self.timeout)["tokens"]
        if len(tokens) <= self.max_tok:
            return text, len(tokens)
        kept = tokens[:self.head_tok] + tokens[-(self.max_tok - self.head_tok):]
        self.truncated += 1
        return _post(self.base + "/detokenize", {"tokens": kept}, self.timeout)["content"], len(kept)

    def _embed(self, texts: Sequence[str]) -> List[np.ndarray]:
        last: Exception | None = None
        for _ in range(3):
            try:
                return _vectors(_post(self.base + "/v1/embeddings",
                                      {"model": "gguf-teacher", "input": list(texts)}, self.timeout),
                                len(texts))
            except (OSError, ValueError, KeyError) as exc:
                last = exc
                time.sleep(1.0)
        raise RuntimeError(f"llama-server embedding failed: {last}")

    # -- public interface used by stage_encode -----------------------------------
    def encode(self, texts: List[str]) -> Dict[str, object]:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(self.workers) as pool:
            pairs = list(pool.map(self._truncate, texts))
            cut = [p[0] for p in pairs]
            chunks = [cut[i:i + self.request_batch] for i in range(0, len(cut), self.request_batch)]
            vecs = [v for chunk in pool.map(self._embed, chunks) for v in chunk]
        out = np.stack(vecs).astype(np.float32)
        dt = time.perf_counter() - t0
        if not np.all(np.isfinite(out)):
            raise FloatingPointError("teacher produced non-finite features")
        self._assert_not_collapsed(cut, out)
        return {"X": out, "seconds": dt, "tokens": int(sum(p[1] for p in pairs))}

    @staticmethod
    def _assert_not_collapsed(texts: Sequence[str], X: np.ndarray) -> None:
        # Distinct texts must give distinct vectors. Allow a tiny slack for genuinely
        # equivalent inputs; a real collapse (cut/ctx bug) shows up as a large share.
        n_text = len(set(texts))
        n_vec = len({X[i].tobytes() for i in range(len(X))})
        if n_text > 1 and n_vec < 0.99 * n_text:
            raise RuntimeError(f"collapsed embeddings: {n_text} distinct texts -> {n_vec} distinct vectors")

    def latency_ms(self, texts: List[str]) -> List[float]:
        cut = [self._truncate(t)[0] for t in texts]
        out = []
        for t in cut:
            t0 = time.perf_counter()
            self._embed([t])
            out.append((time.perf_counter() - t0) * 1e3)
        return out
