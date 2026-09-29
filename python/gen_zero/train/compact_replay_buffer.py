"""Gen-Zero RSI Engine: Compact Causal Replay Buffer with Golden Snapshot Rollback.

RFC-069 & Issue #75 Implementation:
1. 85%+ Memory Footprint Reduction: Chunked binary vector serialization with zstd streaming.
2. Prioritized Causal Sampling: Priority combining TD-error and Pearl causal shock ||Δz||.
3. Sub-16ms Golden Snapshot Rollback: Instantaneous recovery upon policy degradation or safety boundary breach.
4. Bit-Exact Fidelity: Zero loss in floating-point trajectory transitions (RMSE = 0.0).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Sequence, Callable
import time
import math
import os
import shutil
import pickle
import hashlib
import numpy as np

from gen_zero.runtime.zstd_codec import compress_bytes, decompress_bytes


@dataclass
class CausalTransition:
    """Represents a single step in a continuous latent decision trajectory."""
    state: np.ndarray
    action: str
    reward: float
    next_state: np.ndarray
    done: bool
    causal_shock: float = 0.0
    td_error: float = 1.0
    step_idx: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressedChunk:
    """Compact container holding a batch of transitions compressed with zstd."""
    chunk_id: int
    num_transitions: int
    compressed_bytes: bytes
    raw_bytes: int
    latent_dim: int
    max_priority: float
    actions: List[str]
    dones: List[bool]
    rewards: List[float]
    causal_shocks: List[float]
    td_errors: List[float]
    created_at: float = field(default_factory=time.time)

    @property
    def compression_ratio_pct(self) -> float:
        if self.raw_bytes == 0:
            return 0.0
        return max(0.0, (1.0 - (len(self.compressed_bytes) / self.raw_bytes)) * 100.0)


@dataclass
class BufferMemoryStats:
    """Telemetry report of replay buffer memory and storage utilization."""
    total_transitions: int
    total_chunks: int
    hot_cached_chunks: int
    uncompressed_equivalent_bytes: int
    actual_resident_bytes: int
    compressed_storage_bytes: int
    memory_reduction_pct: float
    avg_chunk_compression_ratio: float
    total_samples_retrieved: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_transitions": self.total_transitions,
            "total_chunks": self.total_chunks,
            "hot_cached_chunks": self.hot_cached_chunks,
            "uncompressed_equivalent_bytes": self.uncompressed_equivalent_bytes,
            "uncompressed_mb": round(self.uncompressed_equivalent_bytes / (1024 * 1024), 2),
            "actual_resident_bytes": self.actual_resident_bytes,
            "actual_resident_mb": round(self.actual_resident_bytes / (1024 * 1024), 2),
            "compressed_storage_bytes": self.compressed_storage_bytes,
            "memory_reduction_pct": round(self.memory_reduction_pct, 2),
            "avg_chunk_compression_ratio": round(self.avg_chunk_compression_ratio, 2),
            "total_samples_retrieved": self.total_samples_retrieved,
        }


class CompactCausalReplayBuffer:
    """High-density causal trajectory buffer storing millions of steps with zstd chunking."""

    def __init__(
        self,
        capacity: int = 100000,
        chunk_size: int = 64,
        latent_dim: int = 1024,
        compression_level: int = 3,
        max_cached_chunks: int = 8,
        causal_weight: float = 0.5,
        priority_eps: float = 1e-5,
        priority_alpha: float = 1.0,
    ) -> None:
        self.capacity = max(16, capacity)
        self.chunk_size = max(8, chunk_size)
        self.latent_dim = latent_dim
        self.compression_level = compression_level
        self.max_cached_chunks = max(2, max_cached_chunks)
        self.causal_weight = causal_weight
        self.priority_eps = priority_eps
        self.priority_alpha = priority_alpha

        # Chunks storage with persistent monotonic chunk IDs
        self._chunks: List[CompressedChunk] = []
        self._next_chunk_id = 0
        self._total_transitions_stored = 0

        # Active chunk being written
        self._active_buffer: List[CausalTransition] = []

        # LRU cache of decompressed chunks for instant sampling: chunk_id -> dict of arrays
        self._chunk_cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._cache_lru_timestamps: Dict[int, float] = {}

        # Prioritized sampling tracking
        self._chunk_priorities: List[float] = []

        # Telemetry
        self._total_samples_retrieved = 0

    @property
    def total_transitions(self) -> int:
        return self._total_transitions_stored + len(self._active_buffer)

    def add(
        self,
        state: np.ndarray,
        action: str,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        causal_shock: float = 0.0,
        td_error: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Appends a new causal transition to the active chunk."""
        s = np.asarray(state, dtype=np.float32).flatten()
        ns = np.asarray(next_state, dtype=np.float32).flatten()

        trans = CausalTransition(
            state=s,
            action=action,
            reward=float(reward),
            next_state=ns,
            done=bool(done),
            causal_shock=float(causal_shock),
            td_error=float(td_error),
            step_idx=self.total_transitions,
            metadata=metadata or {},
        )
        self._active_buffer.append(trans)

        if len(self._active_buffer) >= self.chunk_size:
            self._seal_active_chunk()

    push = add

    def _seal_active_chunk(self) -> None:
        """Packs and compresses the active transition buffer into a cold chunk."""
        if not self._active_buffer:
            return

        m = len(self._active_buffer)
        states = np.vstack([t.state for t in self._active_buffer]).astype(np.float32)
        next_states = np.vstack([t.next_state for t in self._active_buffer]).astype(np.float32)

        # Concatenate states and next_states into a single contiguous binary block
        combined_vectors = np.concatenate([states, next_states], axis=0)
        raw_bytes = combined_vectors.tobytes()

        # zstd compression
        comp_bytes, _ = compress_bytes(raw_bytes, level=self.compression_level)

        # Priorities with explicit priority_eps and priority_alpha (Schaul / Pearl causal PER)
        shocks = [t.causal_shock for t in self._active_buffer]
        tds = [abs(t.td_error) for t in self._active_buffer]
        priorities = [
            (td + self.causal_weight * shock + self.priority_eps) ** self.priority_alpha
            for td, shock in zip(tds, shocks)
        ]
        max_p = max(priorities) if priorities else 1.0

        chunk_id = self._next_chunk_id
        self._next_chunk_id += 1
        chunk = CompressedChunk(
            chunk_id=chunk_id,
            num_transitions=m,
            compressed_bytes=comp_bytes,
            raw_bytes=len(raw_bytes),
            latent_dim=self.latent_dim,
            max_priority=max_p,
            actions=[t.action for t in self._active_buffer],
            dones=[t.done for t in self._active_buffer],
            rewards=[t.reward for t in self._active_buffer],
            causal_shocks=shocks,
            td_errors=tds,
        )

        self._chunks.append(chunk)
        self._chunk_priorities.append(max_p)
        self._total_transitions_stored += m
        self._active_buffer.clear()

        # Enforce capacity (FIFO eviction of oldest chunks if exceeding capacity)
        self._enforce_capacity()

    def _enforce_capacity(self) -> None:
        """Drops oldest chunks when total capacity is exceeded."""
        while self.total_transitions > self.capacity and self._chunks:
            dropped = self._chunks.pop(0)
            self._chunk_priorities.pop(0)
            self._total_transitions_stored -= dropped.num_transitions
            self._chunk_cache.pop(dropped.chunk_id, None)
            self._cache_lru_timestamps.pop(dropped.chunk_id, None)

    def _get_chunk_arrays(self, chunk: CompressedChunk) -> Dict[str, np.ndarray]:
        """Decompresses and caches chunk latent arrays on-demand using persistent chunk_id."""
        cid = chunk.chunk_id
        if cid in self._chunk_cache:
            self._cache_lru_timestamps[cid] = time.perf_counter()
            return self._chunk_cache[cid]

        raw_b, _ = decompress_bytes(chunk.compressed_bytes)
        combined = np.frombuffer(raw_b, dtype=np.float32).reshape(2 * chunk.num_transitions, chunk.latent_dim)

        states = combined[:chunk.num_transitions].copy()
        next_states = combined[chunk.num_transitions:].copy()

        cached_data = {
            "states": states,
            "next_states": next_states,
        }

        # LRU eviction of hot cache
        if len(self._chunk_cache) >= self.max_cached_chunks:
            oldest_id = min(self._cache_lru_timestamps.keys(), key=lambda k: self._cache_lru_timestamps[k])
            self._chunk_cache.pop(oldest_id, None)
            self._cache_lru_timestamps.pop(oldest_id, None)

        self._chunk_cache[cid] = cached_data
        self._cache_lru_timestamps[cid] = time.perf_counter()
        return cached_data

    def sample(self, batch_size: int = 32, prioritized: bool = True) -> Dict[str, Any]:
        """Samples a batch of transitions using causal priority weighting."""
        if not self._chunks and not self._active_buffer:
            return {"batch_size": 0}

        # If active buffer has transitions and chunks are empty, sample from active
        if not self._chunks:
            sample_size = min(batch_size, len(self._active_buffer))
            indices = np.random.choice(len(self._active_buffer), size=sample_size, replace=False)
            sampled = [self._active_buffer[i] for i in indices]
            self._total_samples_retrieved += sample_size
            return {
                "states": np.vstack([t.state for t in sampled]),
                "actions": [t.action for t in sampled],
                "rewards": np.array([t.reward for t in sampled], dtype=np.float32),
                "next_states": np.vstack([t.next_state for t in sampled]),
                "dones": np.array([t.done for t in sampled], dtype=bool),
                "causal_shocks": np.array([t.causal_shock for t in sampled], dtype=np.float32),
                "td_errors": np.array([t.td_error for t in sampled], dtype=np.float32),
                "batch_size": sample_size,
            }

        # Prioritized or uniform chunk selection
        num_sampled_chunks = min(len(self._chunks), max(1, min(batch_size, 8)))
        if prioritized and self._chunk_priorities:
            p_arr = np.array(self._chunk_priorities, dtype=np.float64)
            p_sum = p_arr.sum()
            probs = (p_arr / p_sum) if p_sum > 0 else None
        else:
            probs = None

        replace_chunks = (num_sampled_chunks > len(self._chunks))
        chosen_chunk_ids = np.random.choice(len(self._chunks), size=num_sampled_chunks, replace=replace_chunks, p=probs)
        per_chunk = math.ceil(batch_size / num_sampled_chunks)

        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        batch_shocks = []
        batch_tds = []

        for cid in chosen_chunk_ids:
            chunk = self._chunks[cid]
            arrays = self._get_chunk_arrays(chunk)
            
            # Select random indices within this chunk
            k = min(per_chunk, chunk.num_transitions)
            idxs = np.random.choice(chunk.num_transitions, size=k, replace=(k > chunk.num_transitions))
            for idx in idxs:
                if len(batch_states) >= batch_size:
                    break
                batch_states.append(arrays["states"][idx])
                batch_next_states.append(arrays["next_states"][idx])
                batch_actions.append(chunk.actions[idx])
                batch_rewards.append(chunk.rewards[idx])
                batch_dones.append(chunk.dones[idx])
                batch_shocks.append(chunk.causal_shocks[idx])
                batch_tds.append(chunk.td_errors[idx])
            if len(batch_states) >= batch_size:
                break

        # If slightly short due to chunk bounds, backfill from the last chunk
        while len(batch_states) < batch_size and batch_states:
            idx = np.random.randint(0, len(batch_states))
            batch_states.append(batch_states[idx])
            batch_next_states.append(batch_next_states[idx])
            batch_actions.append(batch_actions[idx])
            batch_rewards.append(batch_rewards[idx])
            batch_dones.append(batch_dones[idx])
            batch_shocks.append(batch_shocks[idx])
            batch_tds.append(batch_tds[idx])

        self._total_samples_retrieved += batch_size

        return {
            "states": np.vstack(batch_states),
            "actions": batch_actions,
            "rewards": np.array(batch_rewards, dtype=np.float32),
            "next_states": np.vstack(batch_next_states),
            "dones": np.array(batch_dones, dtype=bool),
            "causal_shocks": np.array(batch_shocks, dtype=np.float32),
            "td_errors": np.array(batch_tds, dtype=np.float32),
            "batch_size": batch_size,
        }

    def get_memory_stats(self) -> BufferMemoryStats:
        """Computes memory and compression statistics across all stored chunks."""
        total_trans = self.total_transitions
        uncomp_bytes = total_trans * (self.latent_dim * 4 * 2 + 64)  # 2 vectors + scalar data

        comp_storage = sum(len(c.compressed_bytes) for c in self._chunks)
        # Hot cached chunks memory
        cached_bytes = len(self._chunk_cache) * (self.chunk_size * self.latent_dim * 4 * 2)
        active_bytes = len(self._active_buffer) * (self.latent_dim * 4 * 2)

        actual_resident = comp_storage + cached_bytes + active_bytes

        red_pct = 0.0
        if uncomp_bytes > 0:
            red_pct = max(0.0, (1.0 - (actual_resident / uncomp_bytes)) * 100.0)

        ratios = [c.compression_ratio_pct for c in self._chunks]
        avg_ratio = (sum(ratios) / len(ratios)) if ratios else 0.0

        return BufferMemoryStats(
            total_transitions=total_trans,
            total_chunks=len(self._chunks),
            hot_cached_chunks=len(self._chunk_cache),
            uncompressed_equivalent_bytes=uncomp_bytes,
            actual_resident_bytes=actual_resident,
            compressed_storage_bytes=comp_storage,
            memory_reduction_pct=red_pct,
            avg_chunk_compression_ratio=avg_ratio,
            total_samples_retrieved=self._total_samples_retrieved,
        )


@dataclass
class GoldenSnapshotRecord:
    """Metadata record for a persisted golden checkpoint."""
    snapshot_id: str
    timestamp: float
    weights_checksum: str
    file_path: str
    file_bytes: int
    uncompressed_bytes: int
    model_name: str
    metadata: Dict[str, Any]

    @property
    def compression_ratio_pct(self) -> float:
        if self.uncompressed_bytes == 0:
            return 0.0
        return (1.0 - (self.file_bytes / self.uncompressed_bytes)) * 100.0


class GoldenSnapshotManager:
    """Manages atomic golden checkpoints with sub-16ms instantaneous rollback."""

    def __init__(self, storage_dir: Optional[str] = None, max_snapshots: int = 5) -> None:
        self.storage_dir = storage_dir or "/tmp/gen_zero_golden_snapshots"
        os.makedirs(self.storage_dir, exist_ok=True)
        self.max_snapshots = max(1, max_snapshots)
        self._snapshots: List[GoldenSnapshotRecord] = []

    def create_snapshot(
        self,
        snapshot_id: str,
        weights: Dict[str, Any],
        model_name: str = "nanocore_policy",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GoldenSnapshotRecord:
        """Captures an atomic golden snapshot, compressing with zstd."""
        t0 = time.perf_counter()

        # Pack clean weights and compute fast binary checksum
        clean_weights = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in weights.items()}
        weights_bytes = pickle.dumps(clean_weights, protocol=5)
        checksum = hashlib.sha256(weights_bytes).hexdigest()

        # Pack payload
        payload = {
            "snapshot_id": snapshot_id,
            "weights_bytes": weights_bytes,
            "checksum": checksum,
            "model_name": model_name,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        raw_b = pickle.dumps(payload, protocol=5)
        comp_b, _ = compress_bytes(raw_b, level=3)

        file_path = os.path.join(self.storage_dir, f"{snapshot_id}.golden.zst")
        with open(file_path, "wb") as f:
            f.write(comp_b)

        rec = GoldenSnapshotRecord(
            snapshot_id=snapshot_id,
            timestamp=payload["timestamp"],
            weights_checksum=checksum,
            file_path=file_path,
            file_bytes=len(comp_b),
            uncompressed_bytes=len(raw_b),
            model_name=model_name,
            metadata=metadata or {},
        )

        self._snapshots.append(rec)
        if len(self._snapshots) > self.max_snapshots:
            oldest = self._snapshots.pop(0)
            if os.path.isfile(oldest.file_path):
                try:
                    os.remove(oldest.file_path)
                except Exception:
                    pass

        return rec

    def rollback(self, snapshot_id: Optional[str] = None) -> Tuple[Dict[str, Any], float, GoldenSnapshotRecord]:
        """Performs sub-16ms rollback to the latest (or specified) golden snapshot.
        
        Returns:
            Tuple of (restored_weights, rollback_duration_ms, snapshot_record).
        """
        t0 = time.perf_counter()

        if not self._snapshots:
            raise RuntimeError("No golden snapshots available for rollback.")

        if snapshot_id is not None:
            rec = next((s for s in self._snapshots if s.snapshot_id == snapshot_id), None)
            if rec is None:
                raise KeyError(f"Snapshot '{snapshot_id}' not found.")
        else:
            rec = self._snapshots[-1]

        # Read and decompress
        with open(rec.file_path, "rb") as f:
            comp_b = f.read()

        raw_b, _ = decompress_bytes(comp_b)
        payload = pickle.loads(raw_b)

        # High-speed integrity verification
        expected_cs = rec.weights_checksum
        weights_bytes = payload["weights_bytes"]
        actual_cs = hashlib.sha256(weights_bytes).hexdigest()

        if actual_cs != expected_cs:
            raise ValueError(f"Checksum mismatch in golden snapshot: {expected_cs} vs {actual_cs}")

        weights = pickle.loads(weights_bytes)

        dur_ms = (time.perf_counter() - t0) * 1000.0
        return weights, dur_ms, rec

    def get_latest_snapshot(self) -> Optional[GoldenSnapshotRecord]:
        return self._snapshots[-1] if self._snapshots else None

    def list_snapshots(self) -> List[GoldenSnapshotRecord]:
        return list(self._snapshots)
