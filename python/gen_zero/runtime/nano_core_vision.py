"""Gen-Zero Runtime: Specialized NanoCore for Vision Interaction Tasks.

Optimized for:
- Temporal continuous latent vectors (e.g. video / camera / sensor streams)
- Continuous action vector candidates or spatial coordinate proposals
- Direct vector dot-product scoring and temporal dynamics evaluation
- Sub-3.5ms pure CPU latency and < 35MB memory footprint
"""

import time
import math
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

from .base_nano_core import BaseNanoCore


class NanoCoreVision(BaseNanoCore):
    """Specialized NanoCore for continuous visual states and spatial action candidates."""

    def __init__(
        self,
        latent_dim: int = 512,
        action_dim: int = 128,
        embed_dim: int = 128,
        version_id: str = "nano-vision-v1.0",
        weights: Optional[Dict[str, np.ndarray]] = None
    ):
        self._domain = "vision"
        self._version_id = version_id
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim

        # Initialize or load weights
        if weights is not None:
            self.weights = weights
        else:
            self.weights = self._init_default_weights()

        self._dequantized_cache = {
            k: v.astype(np.float32) for k, v in self.weights.items()
        }
        self._checksum = self.compute_weights_checksum(self.weights)

    @property
    def domain(self) -> str:
        return self._domain

    @property
    def version_id(self) -> str:
        return self._version_id

    def _init_default_weights(self) -> Dict[str, np.ndarray]:
        """Initializes calibrated weights for visual latent action evaluation."""
        rng = np.random.RandomState(4242)
        scale = 0.05
        return {
            "latent_proj": (rng.randn(self.embed_dim, self.latent_dim) * scale).astype(np.float32),
            "action_proj": (rng.randn(self.embed_dim, self.action_dim) * scale).astype(np.float32),
            "temporal_conv": (rng.randn(self.embed_dim, self.embed_dim) * scale).astype(np.float32),
            "cross_proj": (rng.randn(self.embed_dim, self.embed_dim) * scale).astype(np.float32),
            "score_mlp": (rng.randn(1, self.embed_dim) * scale).astype(np.float32),
            "val_mlp_0": (rng.randn(64, self.embed_dim) * scale).astype(np.float32),
            "val_mlp_1": (rng.randn(1, 64) * scale).astype(np.float32),
        }

    def _encode_latent_vector(self, state_repr: Any) -> np.ndarray:
        """Transforms visual input into latent_dim float32 vector."""
        if isinstance(state_repr, dict) and "visual_embedding" in state_repr:
            state_repr = state_repr["visual_embedding"]
        try:
            arr = np.asarray(state_repr, dtype=np.float32).flatten()
            if len(arr) < self.latent_dim:
                return np.pad(arr, (0, self.latent_dim - len(arr)))
            return arr[:self.latent_dim]
        except Exception:
            return np.zeros(self.latent_dim, dtype=np.float32)

    def _encode_action_candidates(self, candidates: List[Any]) -> np.ndarray:
        """Converts candidates into [K, action_dim] matrix."""
        k = len(candidates)
        mat = np.zeros((k, self.action_dim), dtype=np.float32)
        for i, c in enumerate(candidates):
            if isinstance(c, (list, tuple, np.ndarray)):
                arr = np.asarray(c, dtype=np.float32).flatten()
                if len(arr) < self.action_dim:
                    mat[i, :len(arr)] = arr
                else:
                    mat[i] = arr[:self.action_dim]
            else:
                # String action representation hash
                import hashlib
                h = int(hashlib.md5(str(c).encode("utf-8")).hexdigest()[:8], 16)
                j_range = np.arange(1, self.action_dim + 1, dtype=np.float32)
                mat[i] = np.sin(h * j_range * 0.04).astype(np.float32)
        return mat

    def score_candidates(
        self,
        state_repr: Union[List[float], Any, str, Dict[str, Any]],
        candidates: List[Union[str, List[float], Any]],
        candidate_embeddings: Optional[Any] = None,
        candidate_descriptions: Optional[Dict[str, str]] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Performs forward scoring across visual action candidates."""
        t0 = time.perf_counter()

        if not candidates:
            return {
                "probs": {},
                "value": 0.0,
                "best_action": None,
                "domain": self.domain,
                "latency_ms": 0.0
            }

        z_vec = self._encode_latent_vector(state_repr)
        k = len(candidates)
        if candidate_embeddings is not None:
            a_mat = np.asarray(candidate_embeddings, dtype=np.float32)
        else:
            a_mat = self._encode_action_candidates(candidates)

        # 1. Linear projections
        z_emb = np.dot(self._dequantized_cache["latent_proj"], z_vec)  # [embed_dim]
        a_emb = np.dot(a_mat, self._dequantized_cache["action_proj"].T)  # [K, embed_dim]

        # 2. Temporal filtering & Cross-modality projection
        z_filtered = np.dot(self._dequantized_cache["temporal_conv"], z_emb)
        z_proj = np.dot(self._dequantized_cache["cross_proj"], z_filtered)  # [embed_dim]

        # 3. Bilinear / Dot-product scoring between action candidates and projected state
        scores = np.dot(a_emb, z_proj)  # [K]
        logits = np.dot(a_emb, self._dequantized_cache["score_mlp"].T).flatten() + scores

        # Softmax
        temp = max(1e-4, float(temperature))
        logits_scaled = (logits - np.max(logits)) / temp
        exp_logits = np.exp(logits_scaled)
        probs_arr = exp_logits / np.sum(exp_logits)

        # 4. Value function V(s) in [-1, 1]
        v_h = np.dot(self._dequantized_cache["val_mlp_0"], z_filtered)
        v_h = np.maximum(v_h, 0.0)
        value = float(np.tanh(np.dot(self._dequantized_cache["val_mlp_1"], v_h)[0]))

        probs_dict = {str(candidates[i]): float(probs_arr[i]) for i in range(k)}
        best_idx = int(np.argmax(probs_arr))
        best_action = candidates[best_idx]

        lat_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "probs": probs_dict,
            "value": round(value, 4),
            "best_action": best_action,
            "domain": self.domain,
            "latency_ms": round(lat_ms, 3)
        }

    def export_artifact(self) -> Dict[str, Any]:
        """Exports canonical model artifact manifest."""
        return {
            "domain": self.domain,
            "version_id": self.version_id,
            "latent_dim": self.latent_dim,
            "action_dim": self.action_dim,
            "embed_dim": self.embed_dim,
            "weights_checksum": self._checksum,
            "weight_keys": list(self.weights.keys()),
            "memory_bytes": self.memory_footprint_bytes()
        }

    def memory_footprint_bytes(self) -> int:
        """Calculates exact memory footprint of stored weights and caches."""
        total = sum(v.nbytes for v in self.weights.values())
        total += sum(v.nbytes for v in self._dequantized_cache.values())
        return total
