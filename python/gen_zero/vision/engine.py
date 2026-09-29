"""Multimodal Vision Engine & Shared Vision Prefix KV-Cache for Zero-Decoding.

Implements Milestone 1 and Module 5 of Issue #24:
- Shared Vision Prefix KV-Cache (SharedVisionPrefixCache):
  Encodes image once and freezes KV-cache to serve multiple questions in sub-15ms without re-encoding.
- Zero-Decoding Single-Step Logits Projection:
  Reads Next-Token Logits at assistant prefix without auto-regressive generation.
- Dual-Track GUI Perception Router (AdaptivePerceptionRouter):
  Prefers A11y Tree zero-vision fast-path (~120ms) and falls back to Qwen3.5 Vision Prefill on Canvas/non-standard windows.
- PyTorchVisualDecisionEngine:
  Preserved for backward compatibility and direct candidate scoring.

MultimodalVisionEngine is fail-closed: it never fabricates embeddings, KV-cache state, or
logits. Prefill and scoring require a real ``vision_model`` (see the protocol documented on
MultimodalVisionEngine); without one, both operations raise RuntimeError rather than return
simulated results.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import dataclasses
import hashlib
import logging
import time
import math
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = object
    F = None
    HAS_TORCH = False


class PyTorchVisualDecisionEngine:
    """Non-autoregressive visual candidate scoring engine built on PyTorch."""

    VISION_MODEL_MISSING_ERROR = "Vision model weights are not loaded. Provide a valid checkpoint."

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3.5-9B",
        device: Optional[str] = None,
        feature_dim: int = 4096,
        torch_dtype: Optional[Any] = None
    ):
        self.model_name = model_name_or_path
        self.feature_dim = feature_dim
        
        if HAS_TORCH:
            if device is None:
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            else:
                self.device = str(torch.device(device))
            default_dtype = torch.bfloat16 if torch.device(self.device).type == "cuda" else torch.float32
            self.dtype = torch_dtype or default_dtype
        else:
            self.device = "cpu"
            self.dtype = None

        self.model = None
        self.processor = None
        self.tokenizer = None
        self._is_loaded = False
        self.load_error = None

    @property
    def is_loaded(self) -> bool:
        """True only after real VLM weights are loaded onto the model."""
        return bool(self._is_loaded and self.model is not None)

    @property
    def is_real(self) -> bool:
        """True only after real weights are loaded onto self.model (not a placeholder/mock)."""
        return bool(self._is_loaded and self.model is not None)

    def _normalize_model_path(self, path_or_name: str) -> str:
        """Handles Windows/Linux cross-platform file paths and local HF cache."""
        import os
        import platform
        if not path_or_name:
            return path_or_name
            
        if platform.system() == "Windows" or "\\" in path_or_name or (len(path_or_name) > 1 and path_or_name[1] == ":"):
            norm = os.path.normpath(path_or_name)
            return norm
            
        if os.path.exists(path_or_name):
            return os.path.abspath(path_or_name)
        return path_or_name

    def load_model_if_needed(self):
        """Lazy loads the model onto the target PyTorch device (CUDA on Windows/Linux or CPU)."""
        if self._is_loaded or not HAS_TORCH:
            return

        try:
            import os
            from transformers import AutoProcessor, AutoModelForCausalLM
            
            target_model = self._normalize_model_path(self.model_name)
            
            if torch.device(self.device).type == "cuda" and torch.cuda.is_available():
                if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
                    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

            self.processor = AutoProcessor.from_pretrained(target_model, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                target_model,
                dtype=self.dtype,
                trust_remote_code=True
            ).to(self.device).eval()
            self.tokenizer = getattr(self.processor, "tokenizer", None)
            self._is_loaded = True
        except Exception as e:
            self._is_loaded = False
            self.load_error = str(e)

    def _model_placement(self):
        """Follow loaded parameters, including models moved after initialization."""
        parameter = next(iter(self.model.parameters()), None) if hasattr(self.model, "parameters") else None
        if parameter is not None:
            return parameter.device, parameter.dtype
        return self.device, self.dtype

    def prefill_visual_context(
        self,
        image: Any,
        prompt: str = ""
    ) -> Dict[str, Any]:
        """Runs 1-Pass vision encoding and initial language prefill."""
        t0 = time.perf_counter()

        if not self.is_real:
            raise RuntimeError(self.VISION_MODEL_MISSING_ERROR)

        with torch.inference_mode():
            device, dtype = self._model_placement()
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            inputs = {
                key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
                if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }
            outputs = self.model(
                **inputs,
                use_cache=True,
                output_hidden_states=True
            )
            past_kv = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :].cpu()
            prefill_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "past_key_values": past_kv,
                "last_hidden_state": last_hidden,
                "prefill_ms": round(prefill_ms, 2)
            }

    def score_candidates_direct(
        self,
        prefill_bundle: Dict[str, Any],
        candidates: List[str],
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """Scores multiple candidate actions directly using the prefilled KV cache."""
        t0 = time.perf_counter()

        if not self.is_real:
            raise RuntimeError(self.VISION_MODEL_MISSING_ERROR)

        if not candidates:
            return {"probs": {}, "best_action": None, "scoring_ms": 0.0}

        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be finite and positive.")

        if prefill_bundle.get("past_key_values") is None:
            raise ValueError("A valid visual prefill bundle is required for candidate scoring.")

        with torch.inference_mode():
            past_kv = prefill_bundle["past_key_values"]
            device, _ = self._model_placement()
            cand_logits = {}
            for cand in candidates:
                cand_ids = self.tokenizer.encode(cand, add_special_tokens=False, return_tensors="pt").to(device)
                if cand_ids.numel() == 0:
                    raise ValueError("Candidate must encode to at least one token.")
                out = self.model(cand_ids, past_key_values=past_kv, use_cache=False)
                if not bool(torch.isfinite(out.logits).all()):
                    raise ValueError("Vision model logits must be finite.")
                first_logit = float(out.logits[0, 0, cand_ids[0, 0]].item())
                if not math.isfinite(first_logit):
                    raise ValueError("Vision model logits must be finite.")
                cand_logits[cand] = first_logit

            max_l = max(cand_logits.values())
            exps = {c: math.exp((l - max_l) / max(0.01, temperature)) for c, l in cand_logits.items()}
            total = sum(exps.values())
            probs = {c: round(v / total, 4) for c, v in exps.items()}
            best_action = max(probs, key=probs.get)
            scoring_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "probs": probs,
                "logits": cand_logits,
                "best_action": best_action,
                "scoring_ms": round(scoring_ms, 2)
            }


class PerceptionChannel:
    A11Y_ZERO_VISION = "a11y_zero_vision"
    VISION_MULTIMODAL_FALLBACK = "vision_multimodal_fallback"


@dataclasses.dataclass
class VisionPrefixEntry:
    fingerprint: str
    image_shape: Tuple[int, int]
    prefix_embedding: np.ndarray
    kv_cache: Dict[str, np.ndarray]
    created_at: float = dataclasses.field(default_factory=time.time)
    hit_count: int = 0


class SharedVisionPrefixCache:
    """Caches precomputed vision prefix embeddings and KV-states by image SHA-256 fingerprint."""

    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self._entries: Dict[str, VisionPrefixEntry] = {}

    def get(self, fingerprint: str) -> Optional[VisionPrefixEntry]:
        entry = self._entries.get(fingerprint)
        if entry:
            entry.hit_count += 1
        return entry

    def put(
        self,
        fingerprint: str,
        image_shape: Tuple[int, int],
        prefix_embedding: np.ndarray,
        kv_cache: Optional[Dict[str, np.ndarray]] = None,
    ) -> VisionPrefixEntry:
        if len(self._entries) >= self.capacity:
            lru_key = min(self._entries.keys(), key=lambda k: self._entries[k].hit_count)
            self._entries.pop(lru_key, None)

        entry = VisionPrefixEntry(
            fingerprint=fingerprint,
            image_shape=image_shape,
            prefix_embedding=prefix_embedding,
            kv_cache=kv_cache or {},
        )
        self._entries[fingerprint] = entry
        return entry

    def clear(self) -> None:
        self._entries.clear()

    @property
    def size(self) -> int:
        return len(self._entries)


class MultimodalVisionEngine:
    """Executes single-step zero-decoding multimodal decision prefill with shared vision cache.

    Vision model protocol:
        ``vision_model`` (when provided) must implement:

        - ``encode_image(image_data, image_shape) -> Tuple[np.ndarray, Dict[str, np.ndarray]]``
          Returns ``(prefix_embedding, kv_cache)`` computed by the real vision tower for the
          given image payload.
        - ``score_labels(prefix_entry, question_suffix, labels) -> Sequence[float]``
          Returns one real logit per candidate label, evaluated against the frozen vision
          prefix stored in ``prefix_entry``.

    Without a ``vision_model``, this engine has no checkpoint to run inference against.
    It does not simulate one: ``prefill_image`` and ``score_question_on_prefix`` raise
    ``RuntimeError`` instead of returning random or hash-derived placeholder data.
    """

    VISION_MODEL_MISSING_ERROR = "Vision model weights are not loaded. Provide a valid checkpoint."

    def __init__(
        self,
        embed_dim: int = 4096,
        cache_capacity: int = 64,
        vision_model: Optional[Any] = None,
    ):
        self.embed_dim = embed_dim
        self.cache = SharedVisionPrefixCache(capacity=cache_capacity)
        self.vision_model = vision_model

    def compute_image_fingerprint(self, image_data: Union[bytes, str, np.ndarray]) -> str:
        """Computes deterministic SHA-256 fingerprint for image payload."""
        if isinstance(image_data, bytes):
            raw = image_data
        elif isinstance(image_data, str):
            raw = image_data.encode("utf-8")
        elif isinstance(image_data, np.ndarray):
            raw = image_data.tobytes()
        else:
            raw = str(image_data).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:24]

    def prefill_image(
        self,
        image_data: Union[bytes, str, np.ndarray],
        image_shape: Tuple[int, int] = (1080, 1920),
    ) -> Tuple[VisionPrefixEntry, float]:
        """Encodes image through Qwen3.5 vision tower and freezes KV cache.

        Returns:
            Tuple of (VisionPrefixEntry, encoding_latency_ms).
        """
        t0 = time.perf_counter()

        # A cached prefix is only usable while a real model is mounted.  Check
        # this before looking in the cache so a stale entry can never bypass
        # the fail-closed checkpoint requirement.
        if self.vision_model is None:
            raise RuntimeError(self.VISION_MODEL_MISSING_ERROR)

        fp = self.compute_image_fingerprint(image_data)
        cached = self.cache.get(fp)
        if cached is not None:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return cached, elapsed_ms

        prefix_embedding, kv_cache = self.vision_model.encode_image(image_data, image_shape)

        entry = self.cache.put(fp, image_shape, prefix_embedding, kv_cache)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return entry, elapsed_ms

    def score_question_on_prefix(
        self,
        prefix_entry: VisionPrefixEntry,
        question_suffix: str,
        candidate_labels: List[str],
    ) -> Tuple[str, Dict[str, float], float, float]:
        """Scores candidate tokens at Assistant prefill position against shared vision prefix.

        Returns:
            Tuple of (best_label, label_probabilities, confidence, latency_ms).
        """
        t0 = time.perf_counter()
        if not candidate_labels:
            return "UNKNOWN", {}, 0.0, 0.0

        if self.vision_model is None:
            raise RuntimeError(self.VISION_MODEL_MISSING_ERROR)

        raw_logits = self.vision_model.score_labels(prefix_entry, question_suffix, candidate_labels)
        raw_logits = np.array(list(raw_logits), dtype=np.float64)
        if raw_logits.shape != (len(candidate_labels),):
            raise ValueError(
                "Vision model logits must contain exactly one finite value per candidate label."
            )
        if not np.all(np.isfinite(raw_logits)):
            raise ValueError("Vision model logits must be finite.")
        # Extreme finite logits may subtract to -inf; exp(-inf) correctly gives zero.
        with np.errstate(over="ignore", under="ignore"):
            exp_l = np.exp(raw_logits - np.max(raw_logits))
        probs = exp_l / (np.sum(exp_l) + 1e-8)

        prob_dict = {lbl: float(p) for lbl, p in zip(candidate_labels, probs)}
        best_label = max(candidate_labels, key=lambda l: prob_dict[l])
        p_max = prob_dict[best_label]

        K = len(candidate_labels)
        if K > 1:
            confidence = max(0.0, min(1.0, (p_max - 1.0 / K) / (1.0 - 1.0 / K)))
        else:
            confidence = 1.0

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return best_label, prob_dict, confidence, elapsed_ms

    def batch_evaluate_shared_prefix(
        self,
        image_data: Union[bytes, str, np.ndarray],
        questions: List[Tuple[str, List[str]]],
    ) -> Dict[str, Any]:
        """Prefills vision prefix once, then evaluates N questions reusing the shared KV cache."""
        t_total_start = time.perf_counter()

        prefix_entry, prefill_ms = self.prefill_image(image_data)

        question_results = []
        suffix_latencies = []

        for q_text, labels in questions:
            best_lbl, probs, conf, lat_ms = self.score_question_on_prefix(
                prefix_entry=prefix_entry,
                question_suffix=q_text,
                candidate_labels=labels,
            )
            suffix_latencies.append(lat_ms)
            question_results.append({
                "question": q_text,
                "best_action": best_lbl,
                "probabilities": {k: round(v, 4) for k, v in probs.items()},
                "confidence": round(conf, 4),
                "latency_ms": round(lat_ms, 2),
            })

        total_ms = (time.perf_counter() - t_total_start) * 1000.0
        return {
            "image_fingerprint": prefix_entry.fingerprint,
            "prefill_latency_ms": round(prefill_ms, 2),
            "questions_count": len(questions),
            "results": question_results,
            "total_latency_ms": round(total_ms, 2),
            "mean_question_latency_ms": round(float(np.mean(suffix_latencies)) if suffix_latencies else 0.0, 2),
        }


class AdaptivePerceptionRouter:
    """Routes between A11y Tree zero-vision fast-path and Vision Multimodal fallback."""

    def __init__(self, vision_engine: Optional[MultimodalVisionEngine] = None):
        self.vision_engine = vision_engine or MultimodalVisionEngine()

    def route_environment(self, env_context: Dict[str, Any]) -> str:
        """Determines whether to take A11y zero-vision track or Vision Multimodal fallback."""
        has_a11y = env_context.get("has_a11y_tree", False)
        node_count = env_context.get("a11y_nodes_count", 0)
        is_canvas = env_context.get("is_canvas_or_game", False)
        has_unstructured_screen = env_context.get("unstructured_screen", False)

        if has_a11y and node_count > 0 and not is_canvas and not has_unstructured_screen:
            return PerceptionChannel.A11Y_ZERO_VISION
        return PerceptionChannel.VISION_MULTIMODAL_FALLBACK
