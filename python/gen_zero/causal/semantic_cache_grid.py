"""Static pre-router semantic cache grid.

A cosine-LSH (random-hyperplane / SimHash) bucket index over pre-computed
semantic anchor vectors (prompt-template or counterfactual-manifold
prototype embeddings). A cache hit lets the caller jump straight to a known
manifold attractor and skip a full model forward pass.

Boundary: this module consumes numeric embedding vectors only. It does not
generate embeddings from text and never inspects prompt text, task labels,
or answer formats -- the embedding step is the caller's responsibility and
is out of scope here. Likewise, the per-anchor attractor `radius` is
supplied by the caller, not derived here: this module has no way to compute
a Lyapunov basin size from an embedding alone. The intended source is a
trajectory-level audit such as `lyapunov_verifier.py`; this grid only
consumes the resulting radius as a cosine-distance threshold.

Correctness contract: LSH buckets only ever narrow the *candidate* set for a
query. The hit/miss decision itself is always the exact cosine distance
between the query and the nearest surviving candidate, compared against that
candidate's own attractor radius. Consequences:
  * False positives are structurally impossible: a hit is only returned when
    the exact geometry says the query is inside the anchor's basin.
  * False negatives are possible in two ways, both safe (degrade to
    CacheMiss instead of a fabricated answer): (a) the true nearest anchor
    never collides into a shared LSH bucket with the query (standard LSH
    recall loss); (b) a farther anchor with a larger radius would have
    accepted the query, but the *nearest* candidate's (smaller) radius
    rejects it first. Overlapping, inconsistent attractor basins are a
    caller-side modeling issue, not something this module resolves.

Memory: the index stores float32 unit vectors, so anchor storage alone is
capacity * dim * 4 bytes. To honor a 10 MB budget at capacity=10_000, keep
dim <= ~200 (dim=128 costs 5.12 MB of vector storage, leaving headroom for
hash tables). This is a hard tradeoff, not a hidden one: callers embedding
into higher dimensions must either shrink capacity or project down first.

Latency: every numpy call carries a fixed dispatch cost (measured at
1.5-5 us/call on this machine) that dominates at these data sizes, not FLOP
count. The query path minimizes numpy call count: bucket lookups are plain
Python list indexing by integer key (cheaper than numpy for O(10) size
data), and only the true linear-algebra steps (the two matmuls) use numpy.
Median latency lands under the 50 us target; tail latency (p99/max) is
subject to CPython/OS scheduling jitter that no pure-Python implementation
can bound per-call. See the test suite for measured numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np

MAX_COSINE_DISTANCE = 2.0


def _validate_vector(v: np.ndarray, name: str, dim: int) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if v.ndim != 1 or v.shape[0] != dim:
        raise ValueError(f"{name}: expected a 1-D vector of dim {dim}, got shape {v.shape}")
    return v


@dataclass(frozen=True)
class CacheHit:
    """A verified hit: exact cosine distance was inside the anchor's own radius."""

    anchor_id: int
    payload: Any
    cosine_distance: float
    attractor_radius: float
    confidence: float


@dataclass(frozen=True)
class CacheMiss:
    """No safe match. `nearest_distance`/`nearest_anchor_id` are diagnostic only."""

    reason: str
    nearest_distance: Optional[float] = None
    nearest_anchor_id: Optional[int] = None


QueryResult = Union[CacheHit, CacheMiss]


class SemanticCacheGrid:
    """Fixed-capacity cosine-LSH index over unit-normalized semantic anchors."""

    def __init__(self, dim: int, capacity: int = 10_000, *,
                 num_tables: int = 6, hash_bits: int = 12, seed: int = 0):
        if dim <= 0:
            raise ValueError("dim must be positive")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if num_tables <= 0 or hash_bits <= 0:
            raise ValueError("num_tables and hash_bits must be positive")

        self.dim = dim
        self.capacity = capacity
        self.num_tables = num_tables
        self.hash_bits = hash_bits

        rng = np.random.default_rng(seed)
        # One random hyperplane normal per hash bit, per table; flattened so
        # hashing a vector is a single matmul instead of a per-table loop.
        # Sign of (vector . plane) is invariant to positive scaling, so the
        # hash can be computed from the *raw* vector -- no normalize-then-hash
        # step is needed on the hot path.
        self._planes = rng.standard_normal((dim, num_tables * hash_bits)).astype(np.float32)
        self._bit_weights = (1 << np.arange(hash_bits, dtype=np.int64))

        self._vectors = np.zeros((capacity, dim), dtype=np.float32)
        self._radii: list = [0.0] * capacity
        self._payloads: list = [None] * capacity
        self._count = 0

        # Per table: a dense Python list of length 2**hash_bits, indexed
        # directly by integer bucket key -> a single packed (start,end) int
        # into a sorted-by-key array of anchor ids for that table (0 means
        # "no anchors in this bucket"; a real range always has end>=1).
        # A dense list + packed int avoids the per-entry overhead of a dict
        # of (bytes key -> tuple), which measured ~3 MB heavier at capacity.
        self._pack_shift = max(1, capacity).bit_length() + 1
        self._pack_mask = (1 << self._pack_shift) - 1
        bucket_count = 1 << hash_bits
        self._table_bounds: list = [[0] * bucket_count for _ in range(num_tables)]
        self._table_sorted_ids: list = [np.empty(0, dtype=np.uint16) for _ in range(num_tables)]
        self._dirty = True

    def __len__(self) -> int:
        return self._count

    @property
    def nbytes(self) -> int:
        """Bytes resident in the numeric index structures (planes, vectors,
        per-table sorted-id arrays). Excludes caller-supplied payload objects
        and the small Python dict/list bookkeeping, whose overhead is
        measured separately (see test suite) via tracemalloc."""
        return int(
            self._planes.nbytes + self._bit_weights.nbytes + self._vectors.nbytes
            + sum(a.nbytes for a in self._table_sorted_ids)
        )

    def _hash_keys(self, vector: np.ndarray) -> np.ndarray:
        """One integer bucket key per table, shape (num_tables,)."""
        projections = vector @ self._planes  # (num_tables * hash_bits,) float32
        bits = (projections >= 0.0).astype(np.int64).reshape(self.num_tables, self.hash_bits)
        return bits @ self._bit_weights

    def insert(self, vector: Any, radius: float, payload: Any = None) -> int:
        """Adds a semantic anchor. Returns its anchor_id. The sorted bucket
        structure is rebuilt lazily on the next query, so a batch of inserts
        costs one rebuild, not one per insert."""
        if self._count >= self.capacity:
            raise OverflowError(f"cache grid is at capacity ({self.capacity} anchors)")
        if not math.isfinite(radius) or not (0.0 < radius <= MAX_COSINE_DISTANCE):
            raise ValueError("radius must be finite and within (0, 2.0] cosine-distance units")

        v = _validate_vector(vector, "vector", self.dim)
        norm = float(np.dot(v, v)) ** 0.5
        if not math.isfinite(norm) or norm < 1e-12:
            raise ValueError("vector: not finite, or zero (no direction)")

        idx = self._count
        self._vectors[idx] = v / norm
        self._radii[idx] = float(radius)
        self._payloads[idx] = payload
        self._count += 1
        self._dirty = True
        return idx

    def _rebuild(self) -> None:
        n = self._count
        id_dtype = np.uint16 if self.capacity <= 65_535 else np.uint32
        shift = self._pack_shift
        # (n, num_tables) int64 keys; transient, freed when this call returns.
        keys_per_anchor = (
            np.array([self._hash_keys(self._vectors[i]) for i in range(n)], dtype=np.int64)
            if n else np.empty((0, self.num_tables), dtype=np.int64)
        )
        for t in range(self.num_tables):
            bounds = self._table_bounds[t]
            for i in range(len(bounds)):
                bounds[i] = 0
            if n == 0:
                self._table_sorted_ids[t] = np.empty(0, dtype=id_dtype)
                continue
            keys_t = keys_per_anchor[:, t]
            order = np.argsort(keys_t, kind="stable")
            sorted_ids = order.astype(id_dtype)
            sorted_keys = keys_t[order]
            start = 0
            for i in range(1, n + 1):
                if i == n or sorted_keys[i] != sorted_keys[start]:
                    bounds[int(sorted_keys[start])] = (start << shift) | i
                    start = i
            self._table_sorted_ids[t] = sorted_ids
        self._dirty = False

    def query(self, vector: Any) -> QueryResult:
        if self._count == 0:
            return CacheMiss(reason="empty_index")
        if self._dirty:
            self._rebuild()

        v = _validate_vector(vector, "vector", self.dim)
        norm_sq = float(np.dot(v, v))
        if not math.isfinite(norm_sq) or norm_sq < 1e-24:
            raise ValueError("vector: not finite, or zero (no direction)")

        keys = self._hash_keys(v)
        shift, mask = self._pack_shift, self._pack_mask
        candidates: list = []
        for t in range(self.num_tables):
            packed = self._table_bounds[t][int(keys[t])]
            if packed:
                start, end = packed >> shift, packed & mask
                candidates.extend(self._table_sorted_ids[t][start:end].tolist())

        if not candidates:
            return CacheMiss(reason="no_bucket_candidates")

        raw = self._vectors[candidates] @ v  # cosine_sim * |v| for each candidate (unit anchors)
        best_local = int(np.argmax(raw))
        best_idx = candidates[best_local]
        norm_v = math.sqrt(norm_sq)
        best_distance = 1.0 - float(raw[best_local]) / norm_v
        radius = self._radii[best_idx]

        if best_distance > radius:
            return CacheMiss(
                reason="outside_attractor_radius",
                nearest_distance=best_distance,
                nearest_anchor_id=best_idx,
            )

        confidence = max(0.0, min(1.0, 1.0 - best_distance / radius))
        return CacheHit(
            anchor_id=best_idx,
            payload=self._payloads[best_idx],
            cosine_distance=best_distance,
            attractor_radius=radius,
            confidence=confidence,
        )
