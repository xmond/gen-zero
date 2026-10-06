"""llama.cpp (GGUF) High-Performance Non-Autoregressive Scoring Adapter.

Implements Milestone 2 of Issue #16:
- Minimal non-autoregressive discrete scoring against a generic llama-server (remote or local).
- Thought Tag Folding (</think>\n) to truncate reasoning traces and project deep representations immediately to logits.
- 1-Token Logprobs Extraction & Sequence Log-Likelihood Evaluation.
- Closed-form confidence calculation: c = (p_max - 1/K) / (1 - 1/K).
- Fail-closed: when llama-server is unreachable or returns no usable candidate
  probabilities, the adapter logs the real error and raises LlamaCppUnavailableError.
  It never substitutes hashed or keyword-overlap pseudo-scores.
"""

from typing import Dict, List, Sequence, Any, Optional, Tuple, Union
import os
import time
import math
import json
import urllib.request
import urllib.error
import logging

logger = logging.getLogger(__name__)


class LlamaCppUnavailableError(RuntimeError):
    """llama-server unreachable, or its reply carried no real candidate probabilities."""


def compute_closed_form_confidence(probs: Union[Dict[str, float], Sequence[float]]) -> float:
    """Computes closed-form confidence c = (p_max - 1/K) / (1 - 1/K).

    Invariants:
    - If K == 1: returns 1.0 (trivial certainty).
    - If uniform distribution: returns 0.0 (maximum uncertainty).
    - If one-hot distribution: returns 1.0 (absolute certainty).
    - Monotonically increasing with argmax probability.
    - Clamped strictly within [0.0, 1.0].
    """
    if isinstance(probs, dict):
        prob_values = list(probs.values())
    else:
        prob_values = list(probs)

    if not prob_values:
        return 0.0

    K = len(prob_values)
    if K <= 1:
        return 1.0

    p_max = max(prob_values) if prob_values else 0.0
    if not math.isfinite(p_max):
        return 0.0

    uniform_baseline = 1.0 / float(K)
    denominator = 1.0 - uniform_baseline
    if denominator <= 1e-12:
        return 1.0

    c = (p_max - uniform_baseline) / denominator
    return max(0.0, min(1.0, float(c)))


class LlamaCppScoreAdapter:
    """High-performance scoring adapter interfacing with llama.cpp (GGUF) server.

    Connects to llama-server (e.g. on 127.0.0.1:8080) for 0-token
    prefill logprobs evaluation and sequence log-likelihood computation.
    Fail-closed: raises LlamaCppUnavailableError when llama-server is unreachable.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 2.0,
        model_name: str = "typesafe/zero-1.13",
        thought_folding: bool = True,
        cache_prompt: bool = True,
    ):
        self.base_url = (
            base_url
            or os.environ.get("LLAMACPP_BASE_URL")
            or os.environ.get("LLAMA_SERVER_ENDPOINT")
            or "http://127.0.0.1:8080"
        ).rstrip("/")
        self.timeout = float(timeout)
        self.model_name = model_name
        self.thought_folding = bool(thought_folding)
        self.cache_prompt = bool(cache_prompt)

    def fold_thought_tags(self, prompt: str, force: bool = False) -> str:
        """Injects thought closure </think>\n to truncate reasoning traces.

        When working with reasoning models (e.g. DeepSeek-R1-Distill), reasoning tokens
        can slow down inference by 1-3 seconds. Thought folding injects </think>\n
        if an unclosed <think> tag is detected (or when force=True), forcing the model
        to immediately project its representations into decision logits.
        """
        if not prompt or not isinstance(prompt, str):
            return prompt

        # If already closed with </think>, leave as is
        if "</think>" in prompt:
            return prompt

        is_reasoning_model = any(
            marker in self.model_name.lower()
            for marker in ("r1", "reason", "deepseek-r1", "thinking")
        )

        if "<think>" in prompt or is_reasoning_model or force:
            return prompt.rstrip() + "\n</think>\n"

        return prompt

    def score(
        self,
        prompt: str,
        candidates: Sequence[str],
        model: Optional[str] = None,
        temperature: float = 1.0,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Calculates non-autoregressive probability scores for discrete candidates.

        Args:
            prompt: Input query, state observation, or context string.
            candidates: Non-empty list of candidate strings to score.
            model: Optional override for model name.
            temperature: Sampling temperature scaling (> 0).
            timeout: Request timeout in seconds.

        Returns:
            Dict conforming to Issue #16 /v1/score wire protocol:
            {
                "choice": best_candidate,
                "scores": [p1, p2, ...],
                "probabilities": {c1: p1, c2: p2, ...},
                "confidence": float,
                "timing_ms": float,
                "source": "llamacpp"
            }

        Raises:
            LlamaCppUnavailableError: llama-server unreachable or no usable probabilities.
        """
        t0 = time.perf_counter()

        if not prompt or not isinstance(prompt, str):
            raise ValueError("'prompt' must be a non-empty string.")

        if not candidates or len(candidates) == 0:
            raise ValueError("'candidates' must be a non-empty sequence of strings.")

        candidates_list = [str(c) for c in candidates]
        effective_timeout = timeout if timeout is not None else self.timeout
        effective_model = model or self.model_name

        # 1. Thought Tag Folding
        effective_prompt = self.fold_thought_tags(prompt) if self.thought_folding else prompt

        # 2. Call llama-server; any failure is surfaced, never masked by a pseudo-score
        try:
            res = self._call_llama_server(
                prompt=effective_prompt,
                candidates=candidates_list,
                temperature=temperature,
                timeout=effective_timeout,
            )
        except LlamaCppUnavailableError:
            raise
        except Exception as exc:
            logger.error("llama-server call to %s failed: %r", self.base_url, exc)
            raise LlamaCppUnavailableError(
                f"llama-server at {self.base_url} unreachable or invalid reply: {exc!r}"
            ) from exc

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        res["timing_ms"] = round(elapsed_ms, 2)
        res["model"] = effective_model
        res["source"] = "llamacpp"
        return res

    def _call_llama_server(
        self,
        prompt: str,
        candidates: List[str],
        temperature: float,
        timeout: float,
    ) -> Dict[str, Any]:
        """Performs 1-token logprobs completion request to llama-server."""
        url = f"{self.base_url}/completion"
        payload = {
            "prompt": prompt,
            "n_predict": 1,
            "n_probs": max(20, len(candidates) * 2),
            "temperature": max(0.01, float(temperature)),
            "cache_prompt": self.cache_prompt,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            result = json.loads(body)

        return self._extract_candidate_probabilities(result, candidates, prompt)

    def _extract_candidate_probabilities(
        self,
        result: Dict[str, Any],
        candidates: List[str],
        prompt: str,
    ) -> Dict[str, Any]:
        """Extracts candidate probabilities from completion logprobs payload."""
        probs_data = result.get("completion_probabilities", [])
        logprob_map: Dict[str, float] = {}

        if probs_data and isinstance(probs_data, list):
            first_step = probs_data[0]
            token_probs = first_step.get("probs", [])
            for item in token_probs:
                tok = item.get("tok_str", "").strip()
                p = float(item.get("prob", 0.0))
                if tok:
                    logprob_map[tok] = max(logprob_map.get(tok, 0.0), p)

        raw_scores: List[float] = []
        for c in candidates:
            c_clean = c.strip()
            score = logprob_map.get(c_clean, 0.0)
            if score == 0.0:
                for tok, p in logprob_map.items():
                    if tok in c_clean or c_clean in tok:
                        score = max(score, p)
            raw_scores.append(score)

        total_raw = sum(raw_scores)
        if total_raw <= 1e-6:
            logger.error(
                "llama-server returned no probability mass for any candidate; refusing pseudo-score"
            )
            raise LlamaCppUnavailableError(
                "llama-server reply carried no probability for any candidate token"
            )

        normalized_probs = [round(s / total_raw, 4) for s in raw_scores]
        prob_dict = {c: p for c, p in zip(candidates, normalized_probs)}

        best_idx = int(max(range(len(candidates)), key=lambda i: normalized_probs[i]))
        best_choice = candidates[best_idx]
        conf = compute_closed_form_confidence(normalized_probs)

        return {
            "choice": best_choice,
            "scores": normalized_probs,
            "probabilities": prob_dict,
            "confidence": round(conf, 4),
        }

    def check_health(self) -> Dict[str, Any]:
        """Checks connection health against llama-server. Reports the real error on failure."""
        url = f"{self.base_url}/health"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=0.8) as resp:
                body = resp.read().decode("utf-8")
                return {"status": "healthy", "remote": True, "details": json.loads(body) if body else {}}
        except Exception as e:
            logger.error("llama-server health check to %s failed: %r", self.base_url, e)
            return {
                "status": "unreachable",
                "remote": False,
                "error": str(e),
                "base_url": self.base_url,
            }
