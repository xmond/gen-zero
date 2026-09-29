"""Gen-Zero CPU Decision Engine: cascaded exact-match and manifold lookup tables.

Provides a CPU-only, dependency-light fast path for decisions that have already been
seen (or geometrically resolved) at compile/calibration time, so the expensive model
forward pass is only reached on a genuine cache miss:

1. ExactMatchLookupTable (L0):
   Normalizes text canonically per Document 14 §3.1 (Unicode NFC canonical normalization
   and whitespace collapsing, without casefold or punctuation stripping) and performs
   an O(1) dict lookup into a compiled category probability distribution with a
   configurable margin gate (min_margin).
2. SimplexPrototypeCodebook (L1a):
   Classifies a continuous vector against a small codebook of unit-norm prototypes
   (by default an equiangular tight frame reused from
   :mod:`gen_zero.nanocore.action_etf_embedding`) via vectorized dot products and a
   temperature-scaled softmax.
3. PairwiseManifoldLookup (L1b):
   Judges semantic equivalence of a pair of vectors by projecting the pair onto a
   symmetric joint representation and comparing its cosine similarity against
   offline-fitted class centroids ("equivalent" vs "not equivalent").
4. CPUDecisionEngine:
   Cascades L0 -> L1 and records which layer resolved the decision plus wall-clock
   latency in microseconds, preserving full evidence chains (l0_tie, l1_abstain).
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gen_zero.nanocore.action_etf_embedding import generate_simplex_etf

_BINARY_MAGIC = b"GZL0"
_BINARY_VERSION = 2
# GZL0 v2 Header: magic(4) | version(4) | num_categories(4) | num_entries(4) | min_margin(float64: 8) | table_sha256(32) = 56 bytes
_BINARY_HEADER_SIZE = 56


def _reject_duplicate_json_keys(ordered_pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    """Strict JSON object parser that rejects duplicate member keys fail-closed."""
    d: Dict[str, Any] = {}
    for k, v in ordered_pairs:
        if k in d:
            raise ValueError(f"corrupt json: duplicate member key {k!r} in object")
        d[k] = v
    return d


def normalize_text(text: str) -> str:
    """Canonical text normalization strictly conforming to Document 14 §3.1.

    Performs Unicode NFC normalization and whitespace collapsing.
    Does NOT casefold, strip punctuation, translate, or reorder tokens,
    strictly preserving case-sensitive units ('5mW' vs '5MW'), boolean operators
    ('!enabled' vs 'enabled'), measurement primes ('5′' vs '5'), signs ('-5' vs '5'),
    and decimals ('1.5mg' vs '15mg').

    Operational Domain:
    Applies to natural language text where inter-token whitespace variance is non-semantic.
    For domains where whitespace carries syntactic or semantic significance (e.g. source
    code tokens, quoted literal comparisons, indentation), callers must use typed AST
    serialization per Document 14 §3.1 ('type_tag | length | payload') rather than raw text.
    """
    nfc = unicodedata.normalize("NFC", text)
    return " ".join(nfc.split())


def _digest_key(normalized: str) -> str:
    """Stable content-addressed key for a normalized string.

    Uses Python's built-in ``hash`` composed with length/codepoint mixing via
    ``zlib.crc32``-free arithmetic is avoided in favor of a simple, dependency-free
    FNV-1a 64-bit hash so exported artifacts are reproducible across processes
    (unlike the randomized ``str.__hash__`` seed).
    """
    h = 0xCBF29CE484222325
    prime = 0x100000001B3
    mask = (1 << 64) - 1
    for byte in normalized.encode("utf-8"):
        h ^= byte
        h = (h * prime) & mask
    return format(h, "016x")


@dataclass
class LookupEntry:
    """A single compiled exact-match entry: normalized key -> category distribution."""
    normalized_text: str
    categories: Tuple[str, ...]
    distribution: np.ndarray  # shape [C], sums to 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "normalized_text": self.normalized_text,
            "categories": list(self.categories),
            # Full float64 precision: rounding here can silently reorder which
            # category is the argmax after a JSON round-trip.
            "distribution": [float(p) for p in self.distribution],
        }


class ExactMatchLookupTable:
    """Text-normalizing, hash-backed L0 lookup table.

    A hit returns a precompiled category probability distribution in O(1) time
    (a Python ``dict`` is itself hash-table backed, so no extra hashing is needed
    on the hot path; the FNV digest is only used for the stable exported key).
    """

    def __init__(self, categories: Sequence[str], min_margin: float = 1e-7) -> None:
        if not categories:
            raise ValueError("ExactMatchLookupTable requires at least one category")
        if len(categories) != len(set(categories)):
            raise ValueError(f"duplicate categories not permitted: {categories}")
        if not (np.isfinite(min_margin) and min_margin >= 0.0):
            raise ValueError(f"min_margin must be a finite non-negative float, got {min_margin}")
        self.categories: Tuple[str, ...] = tuple(categories)
        self.min_margin: float = float(min_margin)
        self._table: Dict[str, LookupEntry] = {}

    def __len__(self) -> int:
        return len(self._table)

    @staticmethod
    def _validate_distribution(dist_raw: Any, num_categories: int) -> np.ndarray:
        if isinstance(dist_raw, (list, tuple)):
            for idx, val in enumerate(dist_raw):
                if not isinstance(val, (int, float, np.integer, np.floating)) or isinstance(val, (bool, np.bool_)):
                    raise ValueError(
                        f"distribution element at index {idx} must be a real number, got {val!r}"
                    )
        dist = np.asarray(dist_raw, dtype=np.float64)
        if dist.shape != (num_categories,):
            raise ValueError(
                f"distribution shape {dist.shape} must match ({num_categories},)"
            )
        if not np.all(np.isfinite(dist)):
            raise ValueError("distribution must contain only finite values")
        if np.any(dist < 0):
            raise ValueError("distribution must not contain negative values")
        total = float(np.sum(dist))
        if not np.isfinite(total) or total <= 0:
            raise ValueError("distribution sum must be finite and positive")
        dist = dist / total
        if not np.all(np.isfinite(dist)) or np.all(dist == 0):
            raise ValueError("distribution normalized vector is degenerate")
        return dist

    @classmethod
    def _validate_serialized_distribution(cls, dist_raw: Any, num_categories: int) -> np.ndarray:
        """Validates an already-serialized distribution without re-normalizing.

        Preserves exact IEEE-754 floating point bits so deserialization round-trips
        never shift decision margins or flip abstention into acceptance.
        """
        if isinstance(dist_raw, (list, tuple)):
            for idx, val in enumerate(dist_raw):
                if not isinstance(val, (int, float, np.integer, np.floating)) or isinstance(val, (bool, np.bool_)):
                    raise ValueError(
                        f"corrupt distribution: element at index {idx} must be a real number, got {val!r}"
                    )
        dist = np.asarray(dist_raw, dtype=np.float64)
        if dist.shape != (num_categories,):
            raise ValueError(
                f"corrupt distribution shape {dist.shape}, expected ({num_categories},)"
            )
        if not np.all(np.isfinite(dist)):
            raise ValueError("corrupt distribution: must contain only finite values")
        if np.any(dist < 0):
            raise ValueError("corrupt distribution: must not contain negative values")
        total = float(np.sum(dist))
        if not np.isclose(total, 1.0, atol=1e-5):
            raise ValueError(f"corrupt distribution: sum {total} deviates from 1.0")
        return dist

    def _distribution_from_label(self, label: str) -> np.ndarray:
        if label not in self.categories:
            raise ValueError(f"label {label!r} not in categories {self.categories}")
        dist = np.zeros(len(self.categories), dtype=np.float64)
        dist[self.categories.index(label)] = 1.0
        return dist

    def insert(
        self,
        text: str,
        label: Optional[str] = None,
        distribution: Optional[Sequence[float]] = None,
        overwrite: bool = False,
    ) -> str:
        """Compiles and stores an entry. Returns the normalized key used.

        Defaults to fail-closed collision prevention (overwrite=False).
        """
        if (label is None) == (distribution is None):
            raise ValueError("provide exactly one of label= or distribution=")

        if distribution is not None:
            dist = self._validate_distribution(distribution, len(self.categories))
        else:
            dist = self._distribution_from_label(label)  # type: ignore[arg-type]

        key = normalize_text(text)
        if key in self._table:
            existing = self._table[key]
            same_argmax = int(np.argmax(existing.distribution)) == int(np.argmax(dist))
            max_diff = float(np.max(np.abs(existing.distribution - dist)))
            if not overwrite:
                if not same_argmax or max_diff > 1e-9:
                    raise ValueError(
                        f"Normalization collision detected: '{text}' normalizes to '{key}', "
                        f"which already exists with a conflicting distribution (max_diff={max_diff:.3e}). "
                        f"Pass overwrite=True to force."
                    )
                # Idempotent re-insertion with identical distribution: preserve existing entry without mutating
                return key

        self._table[key] = LookupEntry(
            normalized_text=key,
            categories=self.categories,
            distribution=dist,
        )
        return key

    def lookup(
        self, text: str, min_margin: Optional[float] = None
    ) -> Tuple[bool, Optional[np.ndarray], Optional[str]]:
        """Returns (hit, distribution, argmax_label).

        If the distribution has a top-2 margin <= effective_margin,
        argmax_label returns None to prevent arbitrary tie-breaking.
        """
        if min_margin is not None:
            if not (np.isfinite(min_margin) and float(min_margin) >= 0.0):
                raise ValueError(f"min_margin must be a finite non-negative float, got {min_margin}")
            effective_margin = float(min_margin)
        else:
            effective_margin = self.min_margin

        key = normalize_text(text)
        entry = self._table.get(key)
        if entry is None:
            return False, None, None

        sorted_probs = np.sort(entry.distribution)
        margin = float(sorted_probs[-1] - sorted_probs[-2]) if len(sorted_probs) > 1 else float(sorted_probs[-1])

        if len(sorted_probs) > 1 and margin <= effective_margin:
            top_label: Optional[str] = None
        else:
            top_idx = int(np.argmax(entry.distribution))
            top_label = self.categories[top_idx]

        # Return a defensive copy so callers can never corrupt the lookup table
        # in place by mutating the array they got back from a lookup.
        return True, entry.distribution.copy(), top_label

    def to_json(self) -> str:
        payload = {
            "categories": list(self.categories),
            "min_margin": self.min_margin,
            "entries": {
                key: {
                    "digest": _digest_key(key),
                    **entry.to_dict(),
                }
                for key, entry in self._table.items()
            },
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "ExactMatchLookupTable":
        try:
            data = json.loads(payload, object_pairs_hook=_reject_duplicate_json_keys)
        except json.JSONDecodeError as ex:
            raise ValueError(f"malformed json: {ex}") from ex

        if not isinstance(data, dict):
            raise ValueError("corrupt json: root payload must be an object")
        if "categories" not in data or "entries" not in data:
            raise ValueError("corrupt json: missing 'categories' or 'entries'")
        if not isinstance(data["categories"], list) or not all(isinstance(c, str) for c in data["categories"]):
            raise ValueError("corrupt json: 'categories' must be a list of strings")

        raw_margin = data.get("min_margin", 1e-7)
        if (
            not isinstance(raw_margin, (int, float))
            or isinstance(raw_margin, bool)
            or not np.isfinite(raw_margin)
            or float(raw_margin) < 0.0
        ):
            raise ValueError(
                f"corrupt json: 'min_margin' must be a finite non-negative number, got {raw_margin!r}"
            )

        table = cls(
            categories=data["categories"],
            min_margin=float(raw_margin),
        )

        raw_entries = data["entries"]
        if not isinstance(raw_entries, dict):
            raise ValueError("corrupt json: 'entries' must be an object")

        for member_key, entry in raw_entries.items():
            if not isinstance(entry, dict):
                raise ValueError("corrupt json: entry must be an object")
            key = entry.get("normalized_text")
            if not isinstance(key, str):
                raise ValueError("corrupt json: entry 'normalized_text' must be a string")
            if key != normalize_text(key):
                raise ValueError(f"corrupt json: key {key!r} is not canonically normalized")
            if key in table._table:
                raise ValueError(f"corrupt json: duplicate key {key!r} detected")
            computed_digest = _digest_key(key)
            if member_key != computed_digest and member_key != key:
                raise ValueError(
                    f"corrupt json: member key mismatch for {key!r} "
                    f"(payload member {member_key!r}, computed {computed_digest!r})"
                )
            if "categories" not in entry or tuple(entry["categories"]) != table.categories:
                raise ValueError(
                    f"corrupt json: entry categories {entry.get('categories')} "
                    f"mismatch table categories {list(table.categories)}"
                )
            dist = cls._validate_serialized_distribution(entry["distribution"], len(table.categories))
            table._table[key] = LookupEntry(
                normalized_text=key,
                categories=table.categories,
                distribution=dist,
            )
        return table

    def to_bytes(self) -> bytes:
        """Packs the table into a compact custom binary format GZL0 v2.

        Layout: magic(4) | version(u32: 2) | num_categories(u32) | num_entries(u32)
                | min_margin(float64: 8) | table_sha256(32)
                | categories block (length-prefixed UTF-8 strings)
                | per entry: key_digest(16 ASCII hex) | klen(u32) | key_bytes | distribution (float64 * C)
        The SHA-256 covers the entire file excluding only the 32-byte checksum field itself
        (i.e. header bytes 0..24 concatenated with the payload).
        """
        num_categories = len(self.categories)
        payload_parts: List[bytes] = []
        for cat in self.categories:
            cat_bytes = cat.encode("utf-8")
            payload_parts.append(struct.pack("<I", len(cat_bytes)))
            payload_parts.append(cat_bytes)
        for key, entry in self._table.items():
            digest = _digest_key(key)
            payload_parts.append(digest.encode("ascii"))
            key_bytes = key.encode("utf-8")
            payload_parts.append(struct.pack("<I", len(key_bytes)))
            payload_parts.append(key_bytes)
            payload_parts.append(entry.distribution.astype("<f8").tobytes())

        payload = b"".join(payload_parts)
        header_prefix = struct.pack(
            "<4sIIId",
            _BINARY_MAGIC,
            _BINARY_VERSION,
            num_categories,
            len(self._table),
            self.min_margin,
        )
        table_sha256 = hashlib.sha256(header_prefix + payload).digest()
        return header_prefix + table_sha256 + payload

    @classmethod
    def from_bytes(cls, blob: bytes) -> "ExactMatchLookupTable":
        """Unpacks a binary blob produced by :meth:`to_bytes` (GZL0 v2).

        Fail-closed: validates full-table cryptographic SHA-256 integrity covering
        header parameters (min_margin, categories, counts) as well as all keys and distributions.
        """
        if len(blob) < _BINARY_HEADER_SIZE:
            raise ValueError(
                f"blob too short ({len(blob)} bytes) for a valid "
                f"ExactMatchLookupTable header (need >= {_BINARY_HEADER_SIZE})"
            )
        magic, version, num_categories, num_entries, min_margin, stored_sha = struct.unpack_from(
            "<4sIIId32s", blob, 0
        )
        if magic != _BINARY_MAGIC:
            raise ValueError("invalid ExactMatchLookupTable binary magic")
        if version != _BINARY_VERSION:
            raise ValueError(f"unsupported binary version {version}")

        actual_sha = hashlib.sha256(blob[:24] + blob[_BINARY_HEADER_SIZE:]).digest()
        if stored_sha != actual_sha:
            raise ValueError("corrupt blob: table sha256 checksum mismatch (data corrupted or tampered)")

        offset = _BINARY_HEADER_SIZE
        categories: List[str] = []
        for _ in range(num_categories):
            if offset + 4 > len(blob):
                raise ValueError("truncated blob: category length header cut off")
            (length,) = struct.unpack_from("<I", blob, offset)
            offset += 4
            if offset + length > len(blob):
                raise ValueError("truncated blob: category bytes cut off")
            categories.append(blob[offset:offset + length].decode("utf-8"))
            offset += length

        table = cls(categories=categories, min_margin=min_margin)
        dist_bytes = 8 * num_categories
        for _ in range(num_entries):
            if offset + 16 > len(blob):
                raise ValueError("truncated blob: entry digest cut off")
            stored_digest = blob[offset:offset + 16].decode("ascii")
            offset += 16
            if offset + 4 > len(blob):
                raise ValueError("truncated blob: entry key length header cut off")
            (klen,) = struct.unpack_from("<I", blob, offset)
            offset += 4
            if offset + klen > len(blob):
                raise ValueError("truncated blob: entry key bytes cut off")
            key = blob[offset:offset + klen].decode("utf-8")
            offset += klen
            if key != normalize_text(key):
                raise ValueError(f"corrupt blob: entry key {key!r} is not canonically normalized")
            if key in table._table:
                raise ValueError(f"corrupt blob: duplicate key {key!r} detected in binary blob")
            expected_digest = _digest_key(key)
            if stored_digest != expected_digest:
                raise ValueError(
                    f"corrupt blob: entry digest mismatch for key {key!r} "
                    f"(stored {stored_digest}, expected {expected_digest})"
                )
            if offset + dist_bytes > len(blob):
                raise ValueError("truncated blob: entry distribution bytes cut off")
            dist_raw = np.frombuffer(blob[offset:offset + dist_bytes], dtype="<f8").copy()
            offset += dist_bytes
            dist = cls._validate_serialized_distribution(dist_raw, len(categories))
            table._table[key] = LookupEntry(
                normalized_text=key,
                categories=table.categories,
                distribution=dist,
            )
        if offset != len(blob):
            raise ValueError(
                f"corrupt blob: {len(blob) - offset} trailing byte(s) after the "
                f"last declared entry"
            )
        return table


class SimplexPrototypeCodebook:
    """Compact prototype / ETF codebook classifier.

    Prototypes are unit-norm vectors in R^dim, one per label. Classification is a
    single vectorized dot product against the codebook followed by a temperature
    softmax -- O(K*dim) per query, O(N*K*dim) for a batch via ``np.einsum``.
    """

    def __init__(
        self,
        labels: Sequence[str],
        dim: int,
        temperature: float = 1.0,
        prototypes: Optional[np.ndarray] = None,
        min_confidence: float = 0.0,
        min_margin: float = 0.0,
    ) -> None:
        if not labels:
            raise ValueError("SimplexPrototypeCodebook requires at least one label")
        if len(labels) != len(set(labels)):
            raise ValueError(f"duplicate labels not permitted: {labels}")
        self.labels: Tuple[str, ...] = tuple(labels)
        self.dim = int(dim)
        if not (np.isfinite(temperature) and float(temperature) > 0.0):
            raise ValueError(f"temperature must be a finite positive float, got {temperature}")
        self.temperature = max(1e-4, float(temperature))
        if not (np.isfinite(min_confidence) and 0.0 <= float(min_confidence) <= 1.0):
            raise ValueError(f"min_confidence must be a finite float in [0.0, 1.0], got {min_confidence}")
        if not (np.isfinite(min_margin) and float(min_margin) >= 0.0):
            raise ValueError(f"min_margin must be a finite non-negative float, got {min_margin}")
        # Abstain gates: a query that clears neither threshold is reported as
        # unresolved (label=None, abstain=True) instead of a forced argmax --
        # the anti-cheat rule is that a fallback score must never masquerade
        # as a confident decision.
        self.min_confidence = float(min_confidence)
        self.min_margin = float(min_margin)

        if prototypes is not None:
            proto = np.asarray(prototypes, dtype=np.float64)
            if proto.shape != (len(self.labels), self.dim):
                raise ValueError(
                    f"prototypes shape {proto.shape} must match "
                    f"({len(self.labels)}, {self.dim})"
                )
            if not np.all(np.isfinite(proto)):
                raise ValueError("prototypes must contain only finite values")
        else:
            proto = generate_simplex_etf(len(self.labels), self.dim)

        norms = np.linalg.norm(proto, axis=1, keepdims=True)
        if np.any(norms < 1e-7) or not np.all(np.isfinite(norms)):
            raise ValueError("prototypes must have non-zero, finite norms")
        self.prototypes = proto / norms  # [K, dim], unit-normalized

    def classify(self, vector: np.ndarray) -> Tuple[Optional[str], np.ndarray, Dict[str, Any]]:
        """Classifies a single [dim] vector. Returns (label, probabilities, metrics).

        ``label`` is ``None`` and ``metrics["abstain"]`` is ``True`` when the
        query cannot be reliably classified: non-finite input, a (near-)zero
        vector with no well-defined direction, an exact tie, or a confidence/margin
        below the configured gates. Callers must treat an abstain as a miss, never
        as a valid decision.
        """
        raw = np.asarray(vector, dtype=np.float64).reshape(-1)
        zero_probs = np.zeros(len(self.labels), dtype=np.float64)

        if len(raw) != self.dim:
            return None, zero_probs, {
                "abstain": True, "reason": "dim_mismatch",
                "expected_dim": self.dim, "actual_dim": len(raw),
                "confidence": 0.0, "margin": 0.0,
            }

        if not np.all(np.isfinite(raw)):
            return None, zero_probs, {
                "abstain": True, "reason": "non_finite",
                "confidence": 0.0, "margin": 0.0,
            }

        norm = float(np.linalg.norm(raw))
        if not np.isfinite(norm):
            return None, zero_probs, {
                "abstain": True, "reason": "overflow_norm",
                "confidence": 0.0, "margin": 0.0,
            }
        if norm < 1e-7:
            return None, zero_probs, {
                "abstain": True, "reason": "zero_norm",
                "confidence": 0.0, "margin": 0.0,
            }

        v = raw / norm
        if not np.all(np.isfinite(v)):
            return None, zero_probs, {
                "abstain": True, "reason": "overflow_norm",
                "confidence": 0.0, "margin": 0.0,
            }

        logits = np.dot(self.prototypes, v) / self.temperature  # [K]
        if not np.all(np.isfinite(logits)):
            return None, zero_probs, {
                "abstain": True, "reason": "non_finite_logits",
                "confidence": 0.0, "margin": 0.0,
            }

        probs = self._softmax(logits)
        top_idx = int(np.argmax(probs))
        sorted_probs = np.sort(probs)
        confidence = float(probs[top_idx])
        margin = float(sorted_probs[-1] - (sorted_probs[-2] if len(probs) > 1 else 0.0))

        effective_min_margin = max(self.min_margin, 1e-6)
        if confidence < self.min_confidence or margin < effective_min_margin or margin <= 1e-12:
            return None, probs, {
                "abstain": True,
                "reason": "tie" if margin <= 1e-12 else ("low_confidence" if confidence < self.min_confidence else "low_margin"),
                "confidence": confidence,
                "margin": margin,
            }

        metrics = {"confidence": confidence, "margin": margin, "abstain": False}
        return self.labels[top_idx], probs, metrics

    def classify_batch(self, vectors: np.ndarray) -> Tuple[List[Optional[str]], np.ndarray]:
        """Vectorized batch classification. vectors: [N, dim]. Returns (labels, probs [N, K]).

        Applies the same per-row abstain rules as :meth:`classify`: a row that
        is non-finite, has (near-)zero norm, or falls below the confidence/
        margin gates gets ``label=None`` at that row instead of a forced
        argmax.
        """
        raw = np.asarray(vectors, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != self.dim:
            raise ValueError(f"classify_batch requires 2D array of shape [N, {self.dim}], got {raw.shape}")
        n = raw.shape[0]
        probs = np.zeros((n, len(self.labels)), dtype=np.float64)
        labels: List[Optional[str]] = [None] * n

        finite_mask = np.all(np.isfinite(raw), axis=-1)
        norms = np.linalg.norm(raw, axis=-1)
        valid_mask = finite_mask & (norms >= 1e-7) & np.isfinite(norms)

        if np.any(valid_mask):
            v = raw[valid_mask] / norms[valid_mask, None]
            valid_v_finite = np.all(np.isfinite(v), axis=-1)
            if np.any(valid_v_finite):
                v_clean = v[valid_v_finite]
                clean_indices = np.flatnonzero(valid_mask)[valid_v_finite]
                logits = np.einsum("nd,kd->nk", v_clean, self.prototypes) / self.temperature
                valid_probs = self._softmax_batch(logits)
                probs[clean_indices] = valid_probs

                top_idx = np.argmax(valid_probs, axis=1)
                sorted_probs = np.sort(valid_probs, axis=1)
                confidence = sorted_probs[:, -1]
                margin = confidence - (sorted_probs[:, -2] if valid_probs.shape[1] > 1 else 0.0)
                effective_min_margin = max(self.min_margin, 1e-6)
                accepted = (confidence >= self.min_confidence) & (margin >= effective_min_margin) & (margin > 1e-12)

                for row, idx, keep in zip(clean_indices, top_idx, accepted):
                    if keep:
                        labels[row] = self.labels[idx]

        return labels, probs

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        m = np.max(logits)
        exp_l = np.exp(logits - m)
        return exp_l / np.sum(exp_l)

    @staticmethod
    def _softmax_batch(logits: np.ndarray) -> np.ndarray:
        m = np.max(logits, axis=1, keepdims=True)
        exp_l = np.exp(logits - m)
        return exp_l / np.sum(exp_l, axis=1, keepdims=True)


class PairwiseManifoldLookup:
    """Pairwise semantic-equivalence lookup via offline-fitted centroid geometry."""

    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self.positive_center: Optional[np.ndarray] = None
        self.negative_center: Optional[np.ndarray] = None

    @staticmethod
    def _unit(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        if not np.all(np.isfinite(norms)):
            raise ValueError("vector norm is non-finite (overflow encountered)")
        norms = np.where(norms < 1e-12, 1.0, norms)
        return vectors / norms

    def pair_features(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Returns the invariant 2D relational feature ``[cosine, divergence_norm]``."""
        u_hat = self._unit(np.asarray(u, dtype=np.float64).reshape(1, -1))[0]
        v_hat = self._unit(np.asarray(v, dtype=np.float64).reshape(1, -1))[0]
        cosine = float(np.dot(u_hat, v_hat))
        divergence = float(np.linalg.norm(u_hat - v_hat))
        return np.array([cosine, divergence], dtype=np.float64)

    def fit_centers(
        self,
        pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
        labels: Sequence[bool],
    ) -> None:
        if len(pairs) != len(labels):
            raise ValueError("pairs and labels must have the same length")
        if not pairs:
            raise ValueError("fit_centers requires at least one calibration pair")

        pos_feats: List[np.ndarray] = []
        neg_feats: List[np.ndarray] = []
        for idx, ((u, v), is_equiv) in enumerate(zip(pairs, labels)):
            u_raw = np.asarray(u, dtype=np.float64).reshape(-1)
            v_raw = np.asarray(v, dtype=np.float64).reshape(-1)
            if len(u_raw) != self.dim or len(v_raw) != self.dim:
                raise ValueError(
                    f"calibration pair at index {idx} has dimensionality mismatch: "
                    f"expected {self.dim}, got len(u)={len(u_raw)}, len(v)={len(v_raw)}"
                )
            if not (np.all(np.isfinite(u_raw)) and np.all(np.isfinite(v_raw))):
                raise ValueError(
                    f"calibration pair at index {idx} contains non-finite values (NaN or Inf)"
                )
            u_norm = float(np.linalg.norm(u_raw))
            v_norm = float(np.linalg.norm(v_raw))
            if not (np.isfinite(u_norm) and np.isfinite(v_norm)):
                raise ValueError(
                    f"calibration pair at index {idx} has non-finite norm (overflow)"
                )
            if u_norm < 1e-7 or v_norm < 1e-7:
                raise ValueError(
                    f"calibration pair at index {idx} contains near-zero or zero-norm vector (norm < 1e-7)"
                )
            feat = self.pair_features(u_raw, v_raw)
            if not np.all(np.isfinite(feat)):
                raise ValueError(f"computed feature at index {idx} contains non-finite values")
            (pos_feats if is_equiv else neg_feats).append(feat)

        if not pos_feats or not neg_feats:
            raise ValueError("fit_centers requires both positive and negative examples")

        pos_center = np.mean(pos_feats, axis=0)
        neg_center = np.mean(neg_feats, axis=0)
        if not (np.all(np.isfinite(pos_center)) and np.all(np.isfinite(neg_center))):
            raise ValueError("computed centers contain non-finite values")
        if np.linalg.norm(pos_center) < 1e-7 or np.linalg.norm(neg_center) < 1e-7:
            raise ValueError("computed center has degenerate norm")
        self.positive_center = pos_center
        self.negative_center = neg_center

    def predict_equivalence(self, u: np.ndarray, v: np.ndarray) -> Tuple[Optional[bool], Dict[str, Any]]:
        """Judges semantic equivalence of the pair. Requires :meth:`fit_centers` first."""
        if self.positive_center is None or self.negative_center is None:
            raise RuntimeError("PairwiseManifoldLookup centers are not fitted; call fit_centers() first")

        u_raw = np.asarray(u, dtype=np.float64).reshape(-1)
        v_raw = np.asarray(v, dtype=np.float64).reshape(-1)
        if len(u_raw) != self.dim or len(v_raw) != self.dim:
            return None, {
                "abstain": True,
                "reason": "dim_mismatch",
                "cosine_similarity": None,
                "divergence_norm": None,
                "margin": 0.0,
            }
        if not (np.all(np.isfinite(u_raw)) and np.all(np.isfinite(v_raw))):
            return None, {
                "abstain": True,
                "reason": "non_finite",
                "cosine_similarity": None,
                "divergence_norm": None,
                "margin": 0.0,
            }

        u_norm = float(np.linalg.norm(u_raw))
        v_norm = float(np.linalg.norm(v_raw))
        if not (np.isfinite(u_norm) and np.isfinite(v_norm)):
            return None, {
                "abstain": True,
                "reason": "overflow_norm",
                "cosine_similarity": None,
                "divergence_norm": None,
                "margin": 0.0,
            }
        if u_norm < 1e-7 or v_norm < 1e-7:
            return None, {
                "abstain": True,
                "reason": "zero_norm",
                "cosine_similarity": 0.0,
                "divergence_norm": 0.0,
                "margin": 0.0,
            }

        feat = self.pair_features(u_raw, v_raw)
        dist_pos = float(np.linalg.norm(feat - self.positive_center))
        dist_neg = float(np.linalg.norm(feat - self.negative_center))
        margin = dist_neg - dist_pos

        if abs(margin) < 1e-7:
            return None, {
                "abstain": True,
                "reason": "tie",
                "cosine_similarity": round(float(feat[0]), 6),
                "divergence_norm": round(float(feat[1]), 6),
                "margin": 0.0,
            }

        is_equivalent = dist_pos < dist_neg
        metrics = {
            "abstain": False,
            "cosine_similarity": round(float(feat[0]), 6),
            "divergence_norm": round(float(feat[1]), 6),
            "distance_to_positive_center": round(dist_pos, 6),
            "distance_to_negative_center": round(dist_neg, 6),
            "margin": round(margin, 6),
        }
        return is_equivalent, metrics


@dataclass
class DecisionResult:
    """Outcome of a single :meth:`CPUDecisionEngine.decide` call."""
    hit_level: str  # "L0_exact_match" | "L1_prototype" | "L1_manifold" | "miss"
    label: Optional[Any]
    confidence: Optional[float]
    elapsed_us: float
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_us / 1000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hit_level": self.hit_level,
            "label": self.label,
            "confidence": self.confidence,
            "elapsed_us": round(self.elapsed_us, 3),
            "elapsed_ms": round(self.elapsed_ms, 5),
            "details": self.details,
        }


class CPUDecisionEngine:
    """Cascading CPU decision engine: L0 exact match, then L1 manifold lookup(s)."""

    def __init__(
        self,
        exact_table: Optional[ExactMatchLookupTable] = None,
        prototype_codebook: Optional[SimplexPrototypeCodebook] = None,
        manifold_lookup: Optional[PairwiseManifoldLookup] = None,
        feature_extractor: Optional[Callable[[str], np.ndarray]] = None,
    ) -> None:
        self.exact_table = exact_table
        self.prototype_codebook = prototype_codebook
        self.manifold_lookup = manifold_lookup
        self.feature_extractor = feature_extractor

    def decide(
        self,
        text: Optional[str] = None,
        vector: Optional[np.ndarray] = None,
        pair: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> DecisionResult:
        t_start = time.perf_counter()

        l0_tie_details: Optional[Dict[str, Any]] = None
        if text is not None and self.exact_table is not None:
            hit, distribution, top_label = self.exact_table.lookup(text)
            if hit:
                confidence = float(np.max(distribution)) if distribution is not None else None
                elapsed_us = (time.perf_counter() - t_start) * 1e6
                if top_label is not None:
                    return DecisionResult(
                        hit_level="L0_exact_match",
                        label=top_label,
                        confidence=confidence,
                        elapsed_us=elapsed_us,
                        details={"distribution": distribution.tolist() if distribution is not None else None},
                    )
                l0_tie_details = {
                    "hit": True,
                    "reason": "insufficient_margin",
                    "distribution": distribution.tolist() if distribution is not None else None,
                }

        l1_abstain_metrics: Optional[Dict[str, Any]] = None
        effective_vector = vector
        if effective_vector is None and text is not None and self.feature_extractor is not None:
            effective_vector = self.feature_extractor(text)

        if effective_vector is not None and self.prototype_codebook is not None:
            label, probs, metrics = self.prototype_codebook.classify(effective_vector)
            if not metrics.get("abstain", False) and label is not None:
                elapsed_us = (time.perf_counter() - t_start) * 1e6
                result_details = dict(metrics)
                if l0_tie_details is not None:
                    result_details["l0_tie"] = l0_tie_details
                return DecisionResult(
                    hit_level="L1_prototype",
                    label=label,
                    confidence=metrics["confidence"],
                    elapsed_us=elapsed_us,
                    details=result_details,
                )
            l1_abstain_metrics = metrics

        l1_manifold_abstain_metrics: Optional[Dict[str, Any]] = None
        if pair is not None and self.manifold_lookup is not None:
            is_equivalent, metrics = self.manifold_lookup.predict_equivalence(*pair)
            if not metrics.get("abstain", False) and is_equivalent is not None:
                elapsed_us = (time.perf_counter() - t_start) * 1e6
                result_details = dict(metrics)
                if l0_tie_details is not None:
                    result_details["l0_tie"] = l0_tie_details
                if l1_abstain_metrics is not None:
                    result_details["l1_abstain"] = l1_abstain_metrics
                return DecisionResult(
                    hit_level="L1_manifold",
                    label=is_equivalent,
                    confidence=None,
                    elapsed_us=elapsed_us,
                    details=result_details,
                )
            l1_manifold_abstain_metrics = metrics

        elapsed_us = (time.perf_counter() - t_start) * 1e6
        details: Dict[str, Any] = {}
        if l0_tie_details is not None:
            details["l0_tie"] = l0_tie_details
        if l1_abstain_metrics is not None:
            details["l1_abstain"] = l1_abstain_metrics
        if l1_manifold_abstain_metrics is not None:
            details["l1_manifold_abstain"] = l1_manifold_abstain_metrics
        return DecisionResult(
            hit_level="miss",
            label=None,
            confidence=None,
            elapsed_us=elapsed_us,
            details=details,
        )
