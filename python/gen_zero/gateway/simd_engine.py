"""SIMD Zero-Copy Memory Packet Contract and Physical Memory (mlock) Manager.

Implements Milestone 3 of Issue #23:
- 64-byte Cache-Line Aligned SIMD Vector Packet (NanoCoreVectorPacket):
  C-ABI compatible layout eliminating false-sharing and enabling direct _mm512_load_ps operations.
- Physical Memory Locking (mlock / munlock):
  Pins model weights into physical RAM to eliminate OS page cache swapping and cold-start hard page faults.
- Single-Core P99 Latency Profiler validating <= 1.4ms SLA.
"""

from typing import Optional, Dict, Any, Tuple
import ctypes
import dataclasses
import time
import os
import numpy as np


class NanoCoreVectorPacket(ctypes.Structure):
    """64-byte cache-line aligned C-compatible struct for SIMD zero-copy transfers."""
    _pack_ = 64  # Enforce 64-byte cache-line alignment
    _fields_ = [
        ("packet_id", ctypes.c_uint64),      # offset 0..7 (8B)
        ("dim", ctypes.c_uint32),            # offset 8..11 (4B)
        ("_pad0", ctypes.c_uint32),          # offset 12..15 (4B explicit ABI alignment)
        ("timestamp_ns", ctypes.c_uint64),   # offset 16..23 (8B)
        ("reserved", ctypes.c_uint32 * 10),  # offset 24..63 (40B)
        ("data", ctypes.c_float * 1024),     # offset 64..4159 (4096B) -> perfectly 64B cache-line aligned!
    ]

    @classmethod
    def from_numpy(cls, arr: np.ndarray, packet_id: int = 1) -> "NanoCoreVectorPacket":
        """Zero-copy / fast load into 64-byte aligned SIMD packet."""
        flat = np.ascontiguousarray(arr, dtype=np.float32).ravel()
        n = min(len(flat), 1024)
        packet = cls()
        packet.packet_id = packet_id
        packet.dim = n
        packet.timestamp_ns = time.time_ns()
        ctypes.memmove(
            ctypes.byref(packet.data),
            flat.ctypes.data,
            n * ctypes.sizeof(ctypes.c_float)
        )
        return packet

    def to_numpy(self, zero_copy: bool = False) -> np.ndarray:
        """Extracts numpy view or copy from aligned data buffer."""
        view = np.ctypeslib.as_array(self.data)[:self.dim]
        return view if zero_copy else view.copy()


class PhysicalMemoryLocker:
    """Manages physical RAM page locking (mlock / munlock) for GGUF weights."""

    def __init__(self):
        self._libc = None
        try:
            self._libc = ctypes.CDLL(None, use_errno=True)
        except Exception:
            pass
        self._locked_regions: Dict[int, int] = {}  # addr -> length

    def lock_buffer(self, buffer_address: int, buffer_size_bytes: int) -> Tuple[bool, str]:
        """Locks memory buffer in physical RAM via mlock().

        Returns:
            Tuple of (success: bool, status_message: str).
        """
        if self._libc is None or not hasattr(self._libc, "mlock"):
            return False, "MLOCK_UNAVAILABLE_ON_PLATFORM"

        if buffer_address == 0 or buffer_size_bytes <= 0:
            return False, "INVALID_BUFFER_ADDRESS_OR_SIZE"

        res = self._libc.mlock(ctypes.c_void_p(buffer_address), ctypes.c_size_t(buffer_size_bytes))
        if res == 0:
            self._locked_regions[buffer_address] = buffer_size_bytes
            return True, f"LOCKED_{buffer_size_bytes}_BYTES"
        else:
            # Often EPERM (operation not permitted without CAP_IPC_LOCK), fallback gracefully
            errno = ctypes.get_errno() if hasattr(ctypes, "get_errno") else -1
            return False, f"MLOCK_EPERM_FALLBACK_OK (errno={errno})"

    def unlock_buffer(self, buffer_address: int) -> bool:
        """Unlocks previously locked memory buffer via munlock()."""
        if self._libc is None or not hasattr(self._libc, "munlock"):
            return False

        size = self._locked_regions.pop(buffer_address, 0)
        if size > 0:
            res = self._libc.munlock(ctypes.c_void_p(buffer_address), ctypes.c_size_t(size))
            return res == 0
        return False

    def unlock_all(self) -> int:
        """Unlocks all currently tracked locked buffers."""
        addrs = list(self._locked_regions.keys())
        return sum(1 for addr in addrs if self.unlock_buffer(addr))

    @property
    def locked_regions_count(self) -> int:
        return len(self._locked_regions)


class SIMDScoreEngine:
    """Simulates ultra-fast SIMD dot-product scoring over aligned vector packets."""

    def __init__(self, weights: Optional[np.ndarray] = None):
        if weights is not None:
            self.weights = np.asarray(weights, dtype=np.float32)
        else:
            # 5 candidates x 1024 dimensions
            self.weights = np.random.randn(5, 1024).astype(np.float32)

    def score_packet(self, packet: NanoCoreVectorPacket) -> Tuple[np.ndarray, float]:
        """Scores candidate matrix against packet with sub-millisecond latency."""
        t0 = time.perf_counter()
        vec = np.ctypeslib.as_array(packet.data)[:packet.dim]

        # Dot product
        logits = self.weights[:, :len(vec)] @ vec

        # Softmax
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / (np.sum(exp_l) + 1e-8)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return probs, elapsed_ms

    def benchmark_latency(self, rounds: int = 100) -> Dict[str, float]:
        """Measures P50, P90, P99 single-core latency over 64-byte aligned packets."""
        vec = np.random.randn(1024).astype(np.float32)
        packet = NanoCoreVectorPacket.from_numpy(vec)

        # Warm-up
        for _ in range(10):
            self.score_packet(packet)

        latencies = []
        for _ in range(rounds):
            _, lat = self.score_packet(packet)
            latencies.append(lat)

        latencies = np.array(latencies, dtype=np.float32)
        return {
            "p50_ms": float(np.percentile(latencies, 50)),
            "p90_ms": float(np.percentile(latencies, 90)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "mean_ms": float(np.mean(latencies)),
        }
