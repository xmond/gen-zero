"""Gen-Zero Gateway Layer: Adaptive Input-Aware Modality Router.

Automatically detects input representation and routes:
1. PURE_NUMERICAL: Dict of floats, numbers, ndarray, or tensor -> Direct State (Zero LLM forward, 0ms overhead).
2. UNSTRUCTURED_TEXT: Raw string, prompt, conversation, tool logs -> 1-Pass LLM Backbone Forward to extract last_hidden_state.
3. MULTIMODAL_HYBRID: Dict containing both textual prompts and numerical sensor/market telemetry -> Feature fusion.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


logger = logging.getLogger(__name__)

VISION_NOT_LOADED_REASON = "vision_engine_not_loaded_heuristic_fallback"


class ModalityType(str, Enum):
    PURE_NUMERICAL = "pure_numerical"
    UNSTRUCTURED_TEXT = "unstructured_text"
    MULTIMODAL_HYBRID = "multimodal_hybrid"
    VISION_IMAGE = "vision_image"
    MULTIMODAL_VISION_TEXT = "multimodal_vision_text"


class IngestedState:
    """Standardized ingested state representation output by AdaptiveModalityRouter."""
    def __init__(
        self,
        raw_input: Any,
        modality: ModalityType,
        normalized_state: Any,
        llm_invoked: bool,
        dense_repr: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        latency_us: float = 0.0
    ):
        self.raw_input = raw_input
        self.modality = modality
        self.normalized_state = normalized_state
        self.llm_invoked = llm_invoked
        self.dense_repr = dense_repr
        self.metadata = metadata or {}
        self.latency_us = latency_us

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality.value,
            "llm_invoked": self.llm_invoked,
            "latency_us": round(self.latency_us, 2),
            "metadata": self.metadata
        }


class AdaptiveModalityRouter:
    """Zero-overhead, microsecond-level Input Modality Classifier and Feature Ingestion Gateway."""

    def __init__(
        self,
        llm_extractor: Optional[Any] = None,
        vision_extractor: Optional[Any] = None,
        feature_dim: int = 1024,
        text_keys: Optional[List[str]] = None,
        image_keys: Optional[List[str]] = None
    ):
        """
        Args:
            llm_extractor: Optional callable or model to extract hidden states from text.
            vision_extractor: Optional callable or model to extract hidden states from visual images.
            feature_dim: Expected dimension of dense representation.
            text_keys: Dict keys indicating unstructured text payloads (e.g. 'prompt', 'text', 'query').
            image_keys: Dict keys indicating visual payloads (e.g. 'image', 'frame', 'visual', 'screenshot').
        """
        self.llm_extractor = llm_extractor
        self.vision_extractor = vision_extractor
        self.feature_dim = feature_dim
        self.text_keys = set(text_keys or [
            "text", "prompt", "query", "instruction", "message", "dialogue", "goal_text"
        ])
        self.image_keys = set(image_keys or [
            "image", "frame", "visual", "screenshot", "camera", "pixel_values"
        ])

    def vision_extractor_ready(self) -> bool:
        """True only when the mounted vision extractor reports real loaded weights.

        A callable without an ``is_loaded`` attribute is treated as not loaded (fail-closed):
        mounting a placeholder must not count as a vision LLM forward.
        """
        if self.vision_extractor is None:
            return False
        flag = getattr(self.vision_extractor, "is_loaded", False)
        # A bound method is always truthy; call it instead of trusting its identity.
        return bool(flag() if callable(flag) else flag)

    def _mark_vision_degraded(self, meta: Dict[str, Any]) -> None:
        logger.warning(
            "Vision extractor not loaded (%s); using CPU deterministic visual projection, llm_invoked=False",
            "none mounted" if self.vision_extractor is None else type(self.vision_extractor).__name__,
        )
        meta["extractor"] = "cpu_deterministic_visual_projection"
        meta["degraded"] = True
        meta["reason"] = VISION_NOT_LOADED_REASON

    def _is_image_object(self, obj: Any) -> bool:
        """Determines if an object represents an image (PIL Image, ndarray frame, or image path)."""
        if hasattr(obj, "size") and hasattr(obj, "mode") and hasattr(obj, "save"):
            # Typical PIL.Image.Image
            return True
        if isinstance(obj, str) and any(obj.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]):
            return True
        # Numpy array with shape [H, W, 3] or [3, H, W]
        if hasattr(obj, "shape") and hasattr(obj, "ndim") and obj.ndim in (2, 3, 4):
            return True
        return False

    def classify_modality(self, raw_input: Any) -> ModalityType:
        """Determines input modality in < 1 microsecond."""
        # 1. Direct Image
        if self._is_image_object(raw_input):
            return ModalityType.VISION_IMAGE

        # 2. Direct String (Prompt vs Image Path)
        if isinstance(raw_input, str):
            if any(raw_input.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]):
                return ModalityType.VISION_IMAGE
            return ModalityType.UNSTRUCTURED_TEXT

        # 3. PyTorch Tensor or list of numbers
        if HAS_TORCH and isinstance(raw_input, torch.Tensor):
            if raw_input.ndim in (3, 4):  # e.g. [C, H, W] or [B, C, H, W]
                return ModalityType.VISION_IMAGE
            return ModalityType.PURE_NUMERICAL

        if isinstance(raw_input, list):
            if all(isinstance(x, (int, float)) for x in raw_input):
                return ModalityType.PURE_NUMERICAL
            if all(isinstance(x, str) for x in raw_input):
                return ModalityType.UNSTRUCTURED_TEXT
            return ModalityType.MULTIMODAL_HYBRID

        # 4. Dictionary
        if isinstance(raw_input, dict):
            has_image = False
            has_text = False
            has_numeric = False
            
            for k, v in raw_input.items():
                if k in self.image_keys or self._is_image_object(v):
                    has_image = True
                elif k in self.text_keys or (isinstance(v, str) and len(v.strip()) > 0 and not any(v.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg"])):
                    has_text = True
                elif isinstance(v, (int, float, bool)):
                    has_numeric = True
                elif HAS_TORCH and isinstance(v, torch.Tensor):
                    if v.ndim in (3, 4):
                        has_image = True
                    else:
                        has_numeric = True
                elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                    has_numeric = True

            if has_image:
                if has_text or has_numeric:
                    return ModalityType.MULTIMODAL_VISION_TEXT
                return ModalityType.VISION_IMAGE
            if has_text and has_numeric:
                return ModalityType.MULTIMODAL_HYBRID
            elif has_text:
                return ModalityType.UNSTRUCTURED_TEXT
            else:
                return ModalityType.PURE_NUMERICAL

        # Default fallback for arbitrary object
        return ModalityType.PURE_NUMERICAL

    def ingest(
        self,
        raw_input: Any,
        force_llm: bool = False
    ) -> IngestedState:
        """Ingests raw input and produces a normalized state representation.
        
        - If pure numerical: bypasses LLM completely (0 GPU memory, 0 latency).
        - If unstructured text: extracts last_hidden_state in 1-pass (or mocks representation if no GPU).
        - If hybrid: fuses numerical telemetry with text embeddings.
        """
        t0 = time.perf_counter()
        
        modality = self.classify_modality(raw_input)
        if force_llm:
            modality = ModalityType.UNSTRUCTURED_TEXT

        llm_invoked = False
        dense_repr = None
        normalized_state = raw_input
        meta: Dict[str, Any] = {"classified_modality": modality.value}

        if modality == ModalityType.PURE_NUMERICAL:
            # 100% bypass LLM - pure mathematical/physical state
            llm_invoked = False
            normalized_state = raw_input
            meta["reason"] = "pure_continuous_or_discrete_state_bypass_llm"

        elif modality == ModalityType.UNSTRUCTURED_TEXT:
            # llm_invoked is only true when a real extractor ran; the hash projection is not an LLM call
            llm_invoked = self.llm_extractor is not None
            text_content = raw_input if isinstance(raw_input, str) else str(raw_input)
            
            if self.llm_extractor is not None:
                # Real LLM forward pass to get last_hidden_state
                dense_repr = self.llm_extractor(text_content)
                meta["extractor"] = "custom_llm_extractor"
            else:
                # Deterministic semantic hash projection (when running in CPU/lightweight test mode)
                dense_repr = self._mock_dense_embedding(text_content)
                meta["extractor"] = "cpu_deterministic_projection"
            
            normalized_state = {
                "text": text_content,
                "embedding": dense_repr,
                "is_semantic": True
            }
            meta["reason"] = "natural_language_semantic_extraction_via_llm"

        elif modality == ModalityType.MULTIMODAL_HYBRID:
            # Hybrid: isolate text fields for LLM, keep numeric fields as telemetry
            llm_invoked = self.llm_extractor is not None
            numeric_part = {}
            text_parts = []
            
            if isinstance(raw_input, dict):
                for k, v in raw_input.items():
                    if k in self.text_keys or isinstance(v, str):
                        text_parts.append(f"{k}: {v}")
                    else:
                        numeric_part[k] = v
            else:
                numeric_part = {"state": raw_input}

            combined_text = "; ".join(text_parts)
            if self.llm_extractor is not None:
                dense_repr = self.llm_extractor(combined_text)
                meta["extractor"] = "custom_llm_extractor"
            else:
                dense_repr = self._mock_dense_embedding(combined_text)
                meta["extractor"] = "cpu_deterministic_projection"

            normalized_state = {
                **numeric_part,
                "text_context": combined_text,
                "text_embedding": dense_repr,
                "is_hybrid": True
            }
            meta["reason"] = "multimodal_fusion_text_and_numerical_telemetry"

        elif modality == ModalityType.VISION_IMAGE:
            # Pure visual input: 1-pass visual encoder (Qwen3.5-0.8B-Vision)
            llm_invoked = self.vision_extractor_ready()
            if llm_invoked:
                dense_repr = self.vision_extractor(raw_input)
                meta["extractor"] = "qwen_visual_prefill_extractor"
                meta["reason"] = "vision_image_1pass_prefill_extraction"
            else:
                dense_repr = self._mock_visual_embedding(raw_input)
                self._mark_vision_degraded(meta)

            normalized_state = {
                "image_raw": raw_input,
                "visual_embedding": dense_repr,
                "is_visual": True
            }

        elif modality == ModalityType.MULTIMODAL_VISION_TEXT:
            # Mixed image + text/telemetry: combined vision-language prefill
            llm_invoked = self.vision_extractor_ready()
            image_val = None
            text_context = ""
            telemetry = {}

            if isinstance(raw_input, dict):
                for k, v in raw_input.items():
                    if k in self.image_keys or self._is_image_object(v):
                        image_val = v
                    elif k in self.text_keys or isinstance(v, str):
                        text_context += f" {k}: {v}"
                    else:
                        telemetry[k] = v
            else:
                image_val = raw_input

            if llm_invoked:
                dense_repr = self.vision_extractor(image_val, text_context.strip())
                meta["extractor"] = "qwen_vlm_joint_prefill_extractor"
                meta["reason"] = "joint_vision_language_multimodal_prefill"
            else:
                dense_repr = self._mock_visual_embedding(image_val)
                self._mark_vision_degraded(meta)

            normalized_state = {
                **telemetry,
                "image_raw": image_val,
                "text_context": text_context.strip(),
                "visual_embedding": dense_repr,
                "is_multimodal_vision": True
            }

        latency_us = (time.perf_counter() - t0) * 1_000_000.0
        return IngestedState(
            raw_input=raw_input,
            modality=modality,
            normalized_state=normalized_state,
            llm_invoked=llm_invoked,
            dense_repr=dense_repr,
            metadata=meta,
            latency_us=latency_us
        )

    def _mock_dense_embedding(self, text: str) -> List[float]:
        """Generates a deterministic normalised vector based on text tokens for zero-LLM tests."""
        import hashlib
        import math
        vec = [0.0] * min(self.feature_dim, 64)
        for i, word in enumerate(text.split()):
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest()[:8], 16)
            idx = (h + i) % len(vec)
            vec[idx] += 1.0 / (1.0 + math.log1p(len(word)))
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [round(x / norm, 6) for x in vec]

    def _mock_visual_embedding(self, image_obj: Any) -> List[float]:
        """Generates deterministic dense representation from visual features for CPU testing."""
        import hashlib
        import math
        # Extract basic perceptual digest
        desc = str(image_obj)
        if hasattr(image_obj, "size"):
            desc += f"_{image_obj.size}"
        if hasattr(image_obj, "shape"):
            desc += f"_{image_obj.shape}"
            
        vec = [0.0] * min(self.feature_dim, 64)
        for i, ch in enumerate(desc):
            idx = (ord(ch) * 17 + i * 31) % len(vec)
            vec[idx] += 1.0 / (1.0 + math.sqrt(i + 1))
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [round(x / norm, 6) for x in vec]


class RouteDecision:
    """Standardized output of the MoV (Mixture of Vectors) geometric router."""
    def __init__(
        self,
        selected_domain: str,
        domain_probs: Dict[str, float],
        confidence: float,
        cosine_scores: Dict[str, float],
        margin: float,
        status: str,
        feature_schema_id: Optional[str] = None,
        latency_us: float = 0.0
    ):
        self.selected_domain = selected_domain
        self.domain_probs = domain_probs
        self.confidence = confidence
        self.cosine_scores = cosine_scores
        self.margin = margin
        self.status = status
        self.feature_schema_id = feature_schema_id
        self.latency_us = latency_us

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_domain": self.selected_domain,
            "confidence": round(self.confidence, 4),
            "margin": round(self.margin, 4),
            "status": self.status,
            "domain_probs": {k: round(v, 4) for k, v in self.domain_probs.items()},
            "cosine_scores": {k: round(v, 4) for k, v in self.cosine_scores.items()},
            "feature_schema_id": self.feature_schema_id,
            "latency_us": round(self.latency_us, 2)
        }


class MoVVectorRouter:
    """Mixture of Vectors (MoV) pure numeric geometric router.
    
    Evaluates normalized directional cosine similarity against calibrated domain prototypes:
    s_j = <z, c_j_hat>
    p_j = softmax((s_j - max(s)) / T)
    
    Gating criteria:
    - p_top >= gamma (confidence threshold)
    - s_top >= rho (absolute prototype proximity)
    - s_top - s_second >= delta (decision margin)
    """
    def __init__(
        self,
        prototypes: Dict[str, Any],
        projection_matrix: Optional[Any] = None,
        temperature: float = 0.15,
        confidence_threshold: float = 0.60,
        min_cosine_threshold: float = 0.10,
        margin_threshold: float = 0.05,
        fallback_domain: str = "fallback",
        version_id: str = "mov-v1.0"
    ):
        import numpy as np

        if not prototypes:
            raise ValueError("MoVVectorRouter requires at least one prototype center")

        self.version_id = version_id
        self.temperature = max(1e-5, float(temperature))
        self.confidence_threshold = float(confidence_threshold)
        self.min_cosine_threshold = float(min_cosine_threshold)
        self.margin_threshold = float(margin_threshold)
        self.fallback_domain = fallback_domain

        # Normalize prototype vectors
        self.domain_names: List[str] = list(prototypes.keys())
        centers = []
        for name in self.domain_names:
            c_raw = np.asarray(prototypes[name], dtype=np.float32).flatten()
            if np.isnan(c_raw).any() or np.isinf(c_raw).any():
                raise ValueError(f"Prototype center for domain '{name}' contains NaN or Inf")
            norm = np.linalg.norm(c_raw)
            if norm < 1e-12:
                raise ValueError(f"Prototype center for domain '{name}' has near-zero norm {norm}")
            centers.append(c_raw / norm)

        self.prototype_dim = len(centers[0])
        for idx, c in enumerate(centers):
            if len(c) != self.prototype_dim:
                raise ValueError(f"Dimension mismatch in prototypes: {len(c)} vs {self.prototype_dim}")

        self.centers = np.stack(centers, axis=0)  # [M, k]

        # Projection matrix W: [d, k]
        if projection_matrix is not None:
            self.W = np.asarray(projection_matrix, dtype=np.float32)
            if self.W.shape[1] != self.prototype_dim:
                raise ValueError(f"Projection matrix columns ({self.W.shape[1]}) must match prototype dimension ({self.prototype_dim})")
            self.input_dim = self.W.shape[0]
        else:
            self.W = None
            self.input_dim = self.prototype_dim

    def route(
        self,
        state_vec: Any,
        *,
        feature_schema_id: Optional[str] = None
    ) -> RouteDecision:
        """Evaluates pure numeric state vector against geometric prototypes in < 50 microseconds."""
        import numpy as np
        t0 = time.perf_counter()

        # 1. Shape and finiteness validation
        try:
            if HAS_TORCH and isinstance(state_vec, torch.Tensor):
                x = state_vec.detach().cpu().numpy().astype(np.float32).flatten()
            else:
                x = np.asarray(state_vec, dtype=np.float32).flatten()
        except (ValueError, TypeError, RuntimeError, Exception):
            return RouteDecision(
                selected_domain=self.fallback_domain,
                domain_probs={d: 1.0 / len(self.domain_names) for d in self.domain_names},
                confidence=0.0,
                cosine_scores={d: 0.0 for d in self.domain_names},
                margin=0.0,
                status="REJECTED",
                feature_schema_id=feature_schema_id,
                latency_us=(time.perf_counter() - t0) * 1_000_000.0
            )

        if len(x) != self.input_dim:
            return RouteDecision(
                selected_domain=self.fallback_domain,
                domain_probs={d: 1.0 / len(self.domain_names) for d in self.domain_names},
                confidence=0.0,
                cosine_scores={d: 0.0 for d in self.domain_names},
                margin=0.0,
                status="REJECTED",
                feature_schema_id=feature_schema_id,
                latency_us=(time.perf_counter() - t0) * 1_000_000.0
            )

        if np.isnan(x).any() or np.isinf(x).any():
            return RouteDecision(
                selected_domain=self.fallback_domain,
                domain_probs={d: 1.0 / len(self.domain_names) for d in self.domain_names},
                confidence=0.0,
                cosine_scores={d: 0.0 for d in self.domain_names},
                margin=0.0,
                status="REJECTED",
                feature_schema_id=feature_schema_id,
                latency_us=(time.perf_counter() - t0) * 1_000_000.0
            )

        # 2. Linear projection if W is defined
        if self.W is not None:
            v = np.dot(x, self.W)
        else:
            v = x

        norm_v = float(np.linalg.norm(v))
        if norm_v < 1e-12:
            return RouteDecision(
                selected_domain=self.fallback_domain,
                domain_probs={d: 1.0 / len(self.domain_names) for d in self.domain_names},
                confidence=0.0,
                cosine_scores={d: 0.0 for d in self.domain_names},
                margin=0.0,
                status="REJECTED",
                feature_schema_id=feature_schema_id,
                latency_us=(time.perf_counter() - t0) * 1_000_000.0
            )

        # 3. Unit normalise
        z = v / norm_v

        # 4. Cosine similarity scores: s_j = z . c_j
        scores = np.dot(self.centers, z)  # [M]
        cosine_dict = {d: float(scores[i]) for i, d in enumerate(self.domain_names)}

        # 5. Softmax with temperature
        s_max = float(np.max(scores))
        exp_s = np.exp((scores - s_max) / self.temperature)
        sum_exp = float(np.sum(exp_s))
        probs = exp_s / (sum_exp if sum_exp > 0 else 1.0)
        prob_dict = {d: float(probs[i]) for i, d in enumerate(self.domain_names)}

        # 6. Rank and evaluate margins
        ranked_indices = np.argsort(-scores)
        top_idx = int(ranked_indices[0])
        s_top = float(scores[top_idx])
        p_top = float(probs[top_idx])
        top_domain = self.domain_names[top_idx]

        if len(self.domain_names) > 1:
            second_idx = int(ranked_indices[1])
            s_second = float(scores[second_idx])
            margin = s_top - s_second
        else:
            margin = 1.0

        # 7. Tri-criteria gating: confidence, minimum proximity, margin
        lat_us = (time.perf_counter() - t0) * 1_000_000.0

        if s_top < self.min_cosine_threshold:
            return RouteDecision(
                selected_domain=self.fallback_domain,
                domain_probs=prob_dict,
                confidence=p_top,
                cosine_scores=cosine_dict,
                margin=margin,
                status="OOD",
                feature_schema_id=feature_schema_id,
                latency_us=lat_us
            )

        if p_top < self.confidence_threshold or margin < self.margin_threshold:
            return RouteDecision(
                selected_domain=self.fallback_domain,
                domain_probs=prob_dict,
                confidence=p_top,
                cosine_scores=cosine_dict,
                margin=margin,
                status="UNCERTAIN",
                feature_schema_id=feature_schema_id,
                latency_us=lat_us
            )

        return RouteDecision(
            selected_domain=top_domain,
            domain_probs=prob_dict,
            confidence=p_top,
            cosine_scores=cosine_dict,
            margin=margin,
            status="ROUTED",
            feature_schema_id=feature_schema_id,
            latency_us=lat_us
        )

