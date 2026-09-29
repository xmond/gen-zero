"""Gen-Zero Streaming Spatial-Temporal World Model: Rolling KV-Cache with Attention Sinks.

Implements StreamingLLM-style Attention Sinks + Sliding Window for continuous
multi-frame vision/action streams:
1. Retains initial K anchor tokens (Attention Sinks) to stabilize attention baselines.
2. Maintains a sliding window of W most recent frames/tokens.
3. Fixes memory consumption to strictly O(K + W), preventing OOM across infinite streams.
4. Supports both PyTorch Tensor and NumPy array representations.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import math
import time

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


class RollingVisionKVCache:
    """Bounded, constant-memory rolling KV cache with attention sinks for streaming inputs."""

    def __init__(
        self,
        num_sink_tokens: int = 4,
        window_size: int = 8,
        feature_dim: int = 1024,
        device: str = "cpu",
        dtype: Any = None
    ):
        self.num_sink_tokens = max(1, num_sink_tokens)
        self.window_size = max(1, window_size)
        self.feature_dim = feature_dim
        self.device = device
        self.dtype = dtype or (torch.float32 if HAS_TORCH else None)

        self.total_frames_processed = 0
        self.sink_keys: Optional[Any] = None
        self.sink_values: Optional[Any] = None
        self.window_keys: List[Any] = []
        self.window_values: List[Any] = []
        self.timestamps: List[float] = []

    def reset(self) -> None:
        """Flushes the cache to start a new stream."""
        self.total_frames_processed = 0
        self.sink_keys = None
        self.sink_values = None
        self.window_keys.clear()
        self.window_values.clear()
        self.timestamps.clear()

    def append(
        self,
        key: Union[Any, List[float]],
        value: Union[Any, List[float]],
        timestamp: Optional[float] = None
    ) -> None:
        """Appends a new frame/step key-value representation.
        
        Args:
            key: Tensor or array of shape [1, feature_dim] or [seq_len, feature_dim]
            value: Tensor or array of shape [1, feature_dim] or [seq_len, feature_dim]
            timestamp: Optional timestamp of incoming observation.
        """
        ts = timestamp if timestamp is not None else time.time()
        self.total_frames_processed += 1

        # Standardize representation
        k_tensor, v_tensor = self._ensure_tensor_or_array(key, value)

        # 1. Establish initial Attention Sinks if not yet saturated
        if self.sink_keys is None:
            self.sink_keys = k_tensor
            self.sink_values = v_tensor
            self.timestamps.append(ts)
            return

        current_sink_len = self._get_len(self.sink_keys)
        if current_sink_len < self.num_sink_tokens:
            self.sink_keys = self._concat(self.sink_keys, k_tensor)
            self.sink_values = self._concat(self.sink_values, v_tensor)
            self.timestamps.append(ts)
            return

        # 2. Append to rolling window
        self.window_keys.append(k_tensor)
        self.window_values.append(v_tensor)
        self.timestamps.append(ts)

        # 3. Evict oldest frame in window if exceeding window_size (Constant Memory O(W))
        if len(self.window_keys) > self.window_size:
            self.window_keys.pop(0)
            self.window_values.pop(0)
            if len(self.timestamps) > (self.num_sink_tokens + self.window_size):
                # keep sink timestamps + window timestamps
                del self.timestamps[self.num_sink_tokens]

    def get_kv(self) -> Tuple[Any, Any]:
        """Returns the consolidated [Sink + Rolling Window] keys and values."""
        if self.sink_keys is None:
            if HAS_TORCH and self.dtype is not None:
                empty = torch.empty((0, self.feature_dim), device=self.device, dtype=self.dtype)
                return empty, empty
            elif HAS_NUMPY:
                empty = np.empty((0, self.feature_dim), dtype=np.float32)
                return empty, empty
            return [], []

        if not self.window_keys:
            return self.sink_keys, self.sink_values

        # Concatenate sink + all items in window
        window_k = self._concat_list(self.window_keys)
        window_v = self._concat_list(self.window_values)
        
        full_k = self._concat(self.sink_keys, window_k)
        full_v = self._concat(self.sink_values, window_v)
        return full_k, full_v

    def get_latest_state(self) -> Optional[Any]:
        """Returns the most recent frame representation (from window or sink)."""
        if self.window_keys:
            return self.window_keys[-1]
        elif self.sink_keys is not None:
            return self.sink_keys[-1:] if self._get_len(self.sink_keys) > 0 else self.sink_keys
        return None

    def get_temporal_context(self) -> Dict[str, Any]:
        """Summarizes temporal state and cache statistics."""
        k, v = self.get_kv()
        curr_len = self._get_len(k)
        sink_len = self._get_len(self.sink_keys) if self.sink_keys is not None else 0
        window_frames = len(self.window_keys)
        
        # Estimate memory in MB
        if HAS_TORCH and isinstance(k, torch.Tensor):
            bytes_used = (k.element_size() * k.nelement()) + (v.element_size() * v.nelement())
            mb_used = bytes_used / (1024 * 1024)
        elif HAS_NUMPY and isinstance(k, np.ndarray):
            bytes_used = k.nbytes + v.nbytes
            mb_used = bytes_used / (1024 * 1024)
        else:
            mb_used = 0.01

        return {
            "total_frames_processed": self.total_frames_processed,
            "active_tokens": curr_len,
            "sink_tokens": sink_len,
            "window_frames": window_frames,
            "max_capacity": self.num_sink_tokens + self.window_size,
            "cache_memory_mb": round(mb_used, 4),
            "timestamps_span_sec": round(self.timestamps[-1] - self.timestamps[0], 3) if len(self.timestamps) > 1 else 0.0
        }

    # ---- Internal Utility Helpers ----
    def _ensure_tensor_or_array(self, k: Any, v: Any) -> Tuple[Any, Any]:
        if HAS_TORCH and isinstance(k, torch.Tensor):
            k_t = k.to(device=self.device, dtype=self.dtype)
            v_t = v.to(device=self.device, dtype=self.dtype)
            if k_t.ndim == 1:
                k_t = k_t.unsqueeze(0)
            if v_t.ndim == 1:
                v_t = v_t.unsqueeze(0)
            return k_t, v_t
        elif HAS_NUMPY and isinstance(k, np.ndarray):
            k_a = np.atleast_2d(k).astype(np.float32)
            v_a = np.atleast_2d(v).astype(np.float32)
            return k_a, v_a
        elif HAS_TORCH:
            k_t = torch.tensor(k, device=self.device, dtype=self.dtype or torch.float32)
            v_t = torch.tensor(v, device=self.device, dtype=self.dtype or torch.float32)
            if k_t.ndim == 1:
                k_t = k_t.unsqueeze(0)
            if v_t.ndim == 1:
                v_t = v_t.unsqueeze(0)
            return k_t, v_t
        elif HAS_NUMPY:
            return np.atleast_2d(np.array(k, dtype=np.float32)), np.atleast_2d(np.array(v, dtype=np.float32))
        return k, v

    def _get_len(self, x: Any) -> int:
        if x is None:
            return 0
        if HAS_TORCH and isinstance(x, torch.Tensor):
            return x.shape[0]
        if HAS_NUMPY and isinstance(x, np.ndarray):
            return x.shape[0]
        return len(x)

    def _concat(self, a: Any, b: Any) -> Any:
        if HAS_TORCH and isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return torch.cat([a, b], dim=0)
        if HAS_NUMPY and isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return np.concatenate([a, b], axis=0)
        return list(a) + list(b)

    def _concat_list(self, lst: List[Any]) -> Any:
        if not lst:
            return None
        if HAS_TORCH and isinstance(lst[0], torch.Tensor):
            return torch.cat(lst, dim=0)
        if HAS_NUMPY and isinstance(lst[0], np.ndarray):
            return np.concatenate(lst, axis=0)
        out = []
        for x in lst:
            out.extend(x)
        return out
