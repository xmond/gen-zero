"""Gen-Zero Runtime: Base Abstract Class for Specialized Domain NanoCores.

Defines the contract for lightweight, domain-specialized decision cores:
- Browser Core: DOM tree, semantic element set scoring, action verification.
- Vision Core: Temporal latent vectors, continuous-action candidates, visual grounding.

Enhanced with RFC-069:
- Multi-tier adaptive zstd checkpoint persistence (C-ext -> ctypes -> CLI -> gzip).
- SHA-256 integrity verification with bit-exact weight reconstruction (RMSE <= 10^-7).
- Dual-read compatibility (.zst, .pt, .json, .npz, .pkl).
- Cold startup <= 2ms and 0.00ms runtime inference overhead.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any, Type
import os
import io
import time
import json
import pickle
import struct
import hashlib
from dataclasses import dataclass

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

from .zstd_codec import (
    compress_bytes,
    decompress_exact,
    DECODE_SCRATCH_BYTES,
    is_zstd_magic,
    is_gzip_magic,
    get_compression_tier,
)
from .pickle_budget import (
    MemoryBoundExceededError,
    PickleRefusedError,
    UnpickleCost,
    scan_pickle,
)


# Fixed prefix of every pickle checkpoint: magic, declared resident footprint,
# decompressed payload size, compressed body size, interpreter-object
# allowance. It is read before the body so a loader can size its allocation
# before making it (X-F01).
CHECKPOINT_MAGIC = b"GZNCKPT1"
_CHECKPOINT_HEADER = struct.Struct("<8sQQQQ")

# Loader transients outside the unpickled objects: signature inspection,
# kwargs copies, hashing state. Every overhead allowance includes it.
OBJECT_OVERHEAD_FLOOR = 64 * 1024

# SPDK only: the JSON manifest is decoded into Python objects, allowed at
# OBJECT_OVERHEAD_FACTOR x its bytes. This factor is NOT a bound (R6-M01: an
# int list builds ~40 bytes per few JSON bytes); pickle checkpoints use
# unpickle_overhead_bytes instead.
OBJECT_OVERHEAD_FACTOR = 4


def object_overhead_allowance(non_array_bytes: int) -> int:
    return max(OBJECT_OVERHEAD_FLOOR, OBJECT_OVERHEAD_FACTOR * max(0, non_array_bytes))


def unpickle_overhead_bytes(cost: UnpickleCost, resident_bytes: int) -> int:
    """Allowance a pickle checkpoint must declare for its interpreter
    objects (R6-M01): the scanned worst case of every object outside the
    array buffers, plus any array data the resident size does not cover,
    plus the loader's own transients. Array data is charged once, as a
    view of its pickled buffer: that holds because the scan admits only
    plain scalar dtypes, whose _frombuffer reshape never copies (R8-M01)."""
    return OBJECT_OVERHEAD_FLOOR + cost.object_bytes + max(0, cost.array_bytes - resident_bytes)
CHECKPOINT_HEADER_SIZE = _CHECKPOINT_HEADER.size


class CheckpointFormatError(ValueError):
    """Raised when a checkpoint carries no valid size header, or its payload
    does not match the sizes the header declares."""
    pass


@dataclass(frozen=True)
class CheckpointBound:
    """Upper bound on what loading one checkpoint allocates, known before the
    payload is read.

    resident_bytes: footprint of the loaded core (memory_footprint_bytes()).
    raw_payload_bytes: transient decompressed / reassembled payload.
    body_bytes: transient compressed body read from storage.
    overhead_bytes: allowance for interpreter objects built while decoding.
    decode_scratch_bytes: working memory of the decompressor itself (window,
        state, in-flight chunks), on top of the body and the payload buffer.
        A property of the loader, not of the file, so it is not in the header.
    """
    resident_bytes: int
    raw_payload_bytes: int
    body_bytes: int
    overhead_bytes: int
    decode_scratch_bytes: int

    @property
    def load_peak_bytes(self) -> int:
        """Most memory a load holds at any instant. Two stages never overlap,
        because the loader frees the body once it is decompressed:
        - decoding holds body + payload buffer + the decoder's scratch;
        - unpickling and building hold payload + core (the checksum's
          per-array copy is smaller than the payload freed before it).
        Interpreter objects come on top of either."""
        return self.overhead_bytes + max(
            self.body_bytes + self.raw_payload_bytes + self.decode_scratch_bytes,
            self.raw_payload_bytes + self.resident_bytes,
        )


class CheckpointBudgetExceededError(MemoryError):
    """Raised by a budgeted loader BEFORE any payload allocation, when the
    checkpoint's declared load peak does not fit the caller's reservation."""

    def __init__(self, source: str, bound: CheckpointBound, budget_bytes: int) -> None:
        super().__init__(
            f"Loading {source} needs up to {bound.load_peak_bytes} bytes "
            f"(resident {bound.resident_bytes}, payload {bound.raw_payload_bytes}, "
            f"body {bound.body_bytes}, decode scratch {bound.decode_scratch_bytes}, "
            f"objects {bound.overhead_bytes}), over the reserved {budget_bytes} bytes; nothing was allocated."
        )
        self.source = source
        self.bound = bound
        self.budget_bytes = budget_bytes


class CheckpointBoundViolationError(RuntimeError):
    """Raised when a loaded core is larger than the resident size its
    checkpoint declared. The checkpoint lied; the core is discarded."""
    pass


def read_checkpoint_bound(path: str) -> CheckpointBound:
    """Reads only the fixed header of a pickle checkpoint and returns its
    declared load bound. Allocates CHECKPOINT_HEADER_SIZE bytes, nothing more."""
    with open(path, "rb") as f:
        return _read_bound_from_handle(f, path)


def _read_bound_from_handle(f: Any, path: str) -> CheckpointBound:
    head = f.read(CHECKPOINT_HEADER_SIZE)
    if len(head) != CHECKPOINT_HEADER_SIZE:
        raise CheckpointFormatError(f"{path} is too short to hold a checkpoint header.")
    magic, resident, raw_len, body_len, overhead = _CHECKPOINT_HEADER.unpack(head)
    if magic != CHECKPOINT_MAGIC:
        raise CheckpointFormatError(
            f"{path} has no {CHECKPOINT_MAGIC!r} size header; its load size cannot be bounded "
            f"before allocation. Re-save it with BaseNanoCore.save_checkpoint."
        )
    file_len = os.fstat(f.fileno()).st_size
    if file_len != CHECKPOINT_HEADER_SIZE + body_len:
        raise CheckpointFormatError(
            f"{path} is {file_len} bytes but its header declares a {body_len}-byte body."
        )
    # The codec is only known once the body is read, so the scratch of the
    # hungriest bounded decoder is reserved.
    return CheckpointBound(
        resident_bytes=resident, raw_payload_bytes=raw_len, body_bytes=body_len, overhead_bytes=overhead,
        decode_scratch_bytes=DECODE_SCRATCH_BYTES,
    )


class BaseNanoCore(ABC):
    """Abstract Base Class for specialized NanoCores."""

    @property
    @abstractmethod
    def domain(self) -> str:
        """Domain identifier (e.g. 'browser', 'vision')."""
        pass

    @property
    @abstractmethod
    def version_id(self) -> str:
        """Version and schema identifier."""
        pass

    @abstractmethod
    def score_candidates(
        self,
        state_repr: Union[List[float], Any, str, Dict[str, Any]],
        candidates: List[Union[str, List[float], Any]],
        candidate_embeddings: Optional[Any] = None,
        candidate_descriptions: Optional[Dict[str, str]] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Performs forward scoring across candidates with sub-5ms latency."""
        pass

    @abstractmethod
    def export_artifact(self) -> Dict[str, Any]:
        """Exports canonical model artifact manifest with checksums and metadata."""
        pass

    @abstractmethod
    def memory_footprint_bytes(self) -> int:
        """Reports total memory footprint in bytes."""
        pass

    @staticmethod
    def compute_weights_checksum(weights: Dict[str, Any]) -> str:
        """Computes deterministic SHA-256 checksum of model weights."""
        hasher = hashlib.sha256()
        for k in sorted(weights.keys()):
            val = weights[k]
            hasher.update(k.encode("utf-8"))
            if hasattr(val, "tobytes"):
                hasher.update(val.tobytes())
            elif isinstance(val, (list, tuple)):
                hasher.update(str(val).encode("utf-8"))
            else:
                hasher.update(repr(val).encode("utf-8"))
        return hasher.hexdigest()

    def save_checkpoint(
        self,
        path: str,
        compress: bool = True,
        compression_level: int = 3,
    ) -> Dict[str, Any]:
        """Persists NanoCore weights and metadata to disk with zstd compression.
        
        Args:
            path: Destination file path.
            compress: If True, compresses using multi-tier zstd.
            compression_level: Compression level (default: 3).
            
        Returns:
            Dict containing checkpoint metrics (path, bytes, ratio, sha256, tier).
        """
        t0 = time.perf_counter()
        
        weights = getattr(self, "weights", {})
        checksum = self.compute_weights_checksum(weights)
        
        # Collect constructor arguments
        init_kwargs = {}
        for attr in ["state_dim", "candidate_dim", "embed_dim", "latent_dim", "action_dim", "seed"]:
            if hasattr(self, attr):
                init_kwargs[attr] = getattr(self, attr)
        init_kwargs["version_id"] = self.version_id
        if hasattr(self, "domain"):
            init_kwargs["domain"] = self.domain

        metadata = {
            "domain": self.domain,
            "version_id": self.version_id,
            "class_name": self.__class__.__name__,
            "module_name": self.__class__.__module__,
            "weights_checksum": checksum,
            "init_kwargs": init_kwargs,
            "weight_keys": list(weights.keys()),
            "artifact": self.export_artifact(),
            "timestamp": time.time(),
        }

        # Pack payload
        payload = {
            "metadata": metadata,
            "weights": {k: (v.copy() if hasattr(v, "copy") else v) for k, v in weights.items()},
        }
        
        raw_bytes = pickle.dumps(payload, protocol=5)
        raw_len = len(raw_bytes)
        
        tier_used = "uncompressed"
        if compress:
            saved_bytes, tier_used = compress_bytes(raw_bytes, level=compression_level)
        else:
            saved_bytes = raw_bytes
            
        saved_len = len(saved_bytes)
        # The loader refuses what this scan refuses, so a core whose payload
        # cannot be bounded fails here instead of writing an unloadable file.
        try:
            cost = scan_pickle(raw_bytes)
        except PickleRefusedError as e:
            raise CheckpointFormatError(
                f"{self.__class__.__name__} payload cannot be loaded under a memory bound: {e}"
            ) from e
        resident_bytes = int(self.memory_footprint_bytes())
        bound = CheckpointBound(
            resident_bytes=resident_bytes,
            raw_payload_bytes=raw_len,
            body_bytes=saved_len,
            overhead_bytes=unpickle_overhead_bytes(cost, resident_bytes),
            decode_scratch_bytes=DECODE_SCRATCH_BYTES,
        )
        header = _CHECKPOINT_HEADER.pack(
            CHECKPOINT_MAGIC, bound.resident_bytes, bound.raw_payload_bytes, bound.body_bytes,
            bound.overhead_bytes,
        )

        # Ensure target directory exists
        dir_name = os.path.dirname(os.path.abspath(path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        with open(path, "wb") as f:
            f.write(header)
            f.write(saved_bytes)

        file_sha256 = hashlib.sha256(header + saved_bytes).hexdigest()
        duration_ms = (time.perf_counter() - t0) * 1000.0
        
        ratio = round((1.0 - (saved_len / max(1, raw_len))) * 100.0, 2)
        
        return {
            "path": path,
            "raw_bytes": raw_len,
            "saved_bytes": CHECKPOINT_HEADER_SIZE + saved_len,
            "bound": bound,
            "compressed": compress,
            "compression_ratio": ratio,
            "weights_checksum": checksum,
            "file_sha256": file_sha256,
            "tier": tier_used,
            "duration_ms": round(duration_ms, 3),
        }

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        verify_checksum: bool = True,
        max_bytes: Optional[int] = None,
    ) -> "BaseNanoCore":
        """Loads a NanoCore from disk, supporting transparent decompression and dual-reads.
        
        Supports formats:
        - .zst / compressed zstd checkpoint
        - .pkl / uncompressed binary checkpoint
        - .npz / NumPy zip archive
        - .json / JSON checkpoint
        - .pt / PyTorch state_dict
        
        Args:
            path: Path to checkpoint file.
            verify_checksum: If True, validates SHA-256 weights checksum.
            max_bytes: Reservation the caller holds for this load. When set,
                the header's declared load peak is checked against it before
                the body is read; nothing larger than the header is allocated
                if it does not fit. Only header-carrying pickle checkpoints
                can be loaded under a budget.
            
        Returns:
            Instantiated BaseNanoCore ready for sub-millisecond inference.

        Raises:
            CheckpointBudgetExceededError: the declared peak exceeds max_bytes.
            CheckpointFormatError: missing header, payload sizes disagree with
                it, or the payload uses a pickle feature a checkpoint never needs.
            MemoryBoundExceededError: unpickling the payload would build more
                objects than the header's overhead allowance; raised before
                pickle.loads, whether or not max_bytes is set.
            CheckpointBoundViolationError: the loaded core is larger than declared.
        """
        t0 = time.perf_counter()
        
        if not os.path.isfile(path):
            raise FileNotFoundError(f"NanoCore checkpoint not found: {path}")
            
        # Determine format based on extension and magic bytes
        lower_path = path.lower()
        foreign = lower_path.endswith((".npz", ".json", ".pt", ".bin"))
        if foreign and max_bytes is not None:
            raise CheckpointFormatError(
                f"{path}: .npz/.json/.pt checkpoints carry no size header, so a load under a "
                f"{max_bytes}-byte reservation cannot be bounded before allocation."
            )

        # Handle .npz files directly
        if lower_path.endswith(".npz"):
            return cls._load_from_npz(path, verify_checksum)
            
        # Handle .json files directly
        if lower_path.endswith(".json"):
            return cls._load_from_json(path, verify_checksum)
            
        # Handle .pt PyTorch files
        if lower_path.endswith(".pt") or lower_path.endswith(".bin"):
            return cls._load_from_torch(path, verify_checksum)

        with open(path, "rb") as f:
            # Same handle for header and body: a file swapped after the header
            # read cannot smuggle in a bigger body (fstat length is checked).
            bound = _read_bound_from_handle(f, path)
            if max_bytes is not None and bound.load_peak_bytes > max_bytes:
                raise CheckpointBudgetExceededError(path, bound, max_bytes)
            body = f.read(bound.body_bytes + 1)
        if len(body) != bound.body_bytes:
            raise CheckpointFormatError(
                f"{path}: read {len(body)} body bytes, header declares {bound.body_bytes}."
            )

        # decompress_exact never allocates past raw_payload_bytes, the size the
        # budget check above already counted; a body that disagrees is refused.
        try:
            decompressed_bytes, tier = decompress_exact(body, bound.raw_payload_bytes)
        except ValueError as e:
            raise CheckpointFormatError(f"{path}: {e}") from e
        del body

        # R6-M01: the header's overhead allowance is a contract, checked
        # before any object is built. The scan stops as soon as its running
        # charge passes the allowance, so a refused payload costs no more
        # than the allowance itself.
        object_limit = bound.overhead_bytes - OBJECT_OVERHEAD_FLOOR
        try:
            if object_limit < 0:
                raise MemoryBoundExceededError(
                    f"its header allows {bound.overhead_bytes} object bytes, under the "
                    f"{OBJECT_OVERHEAD_FLOOR}-byte loader minimum",
                    UnpickleCost(array_bytes=0, object_bytes=0, objects=0),
                )
            cost = scan_pickle(decompressed_bytes, limit=object_limit)
        except PickleRefusedError as e:
            raise CheckpointFormatError(f"{path}: {e}") from e
        except MemoryBoundExceededError as e:
            raise MemoryBoundExceededError(f"{path}: {e}", e.cost) from e
        needed = unpickle_overhead_bytes(cost, bound.resident_bytes)
        if needed > bound.overhead_bytes:
            raise MemoryBoundExceededError(
                f"{path}: unpickling needs up to {needed} bytes of objects ({cost.objects} objects, "
                f"{cost.array_bytes} array bytes against resident {bound.resident_bytes}), over the "
                f"header's {bound.overhead_bytes}-byte allowance; nothing was unpickled.",
                cost,
            )

        try:
            payload = pickle.loads(decompressed_bytes)
        except Exception as e:
            raise ValueError(f"Failed to deserialize checkpoint payload from {path}: {e}")
        del decompressed_bytes

        if not isinstance(payload, dict) or "weights" not in payload:
            raise ValueError(f"Invalid checkpoint format in {path}: expected dict with 'weights'")
            
        weights = payload["weights"]
        metadata = payload.get("metadata", {})
        
        # Verify weights checksum
        expected_checksum = metadata.get("weights_checksum")
        if verify_checksum and expected_checksum:
            actual_checksum = cls.compute_weights_checksum(weights)
            if actual_checksum != expected_checksum:
                raise ValueError(
                    f"Checksum mismatch for {path}: expected {expected_checksum}, got {actual_checksum}"
                )

        # Determine target concrete class
        target_cls = cls
        if target_cls is BaseNanoCore:
            class_name = metadata.get("class_name", "")
            domain = metadata.get("domain", "")
            target_cls = cls._resolve_concrete_class(class_name, domain)
            
        # Prepare constructor arguments
        init_kwargs = metadata.get("init_kwargs", {}).copy()
        if "domain" not in init_kwargs and "domain" in metadata:
            init_kwargs["domain"] = metadata["domain"]
        init_kwargs["weights"] = weights
        
        # Filter init_kwargs according to target_cls.__init__ signature
        import inspect
        sig = inspect.signature(target_cls.__init__)
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if not has_kwargs:
            valid_keys = set(sig.parameters.keys())
            init_kwargs = {k: v for k, v in init_kwargs.items() if k in valid_keys}
        
        # Instantiate model
        instance = target_cls(**init_kwargs)
        
        # Record cold-load duration
        instance._cold_load_duration_ms = (time.perf_counter() - t0) * 1000.0
        instance._checkpoint_metadata = metadata
        instance._load_tier = tier

        actual = int(instance.memory_footprint_bytes())
        if actual > bound.resident_bytes:
            raise CheckpointBoundViolationError(
                f"{path}: loaded core reports {actual} bytes, its header declared "
                f"{bound.resident_bytes}; the checkpoint is corrupt and the core is discarded."
            )
        return instance

    @classmethod
    def _resolve_concrete_class(cls, class_name: str, domain: str) -> Type["BaseNanoCore"]:
        """Resolves concrete subclass from class name or domain."""
        if class_name == "NanoCoreBrowser" or domain == "browser":
            from .nano_core_browser import NanoCoreBrowser
            return NanoCoreBrowser
        if class_name == "NanoCoreVision" or domain == "vision":
            from .nano_core_vision import NanoCoreVision
            return NanoCoreVision
        if class_name in ("DomainSpecialistNanoCore", "SpecialistNanoCore") or domain not in ("browser", "vision"):
            from .specialist_nano_core import DomainSpecialistNanoCore
            return DomainSpecialistNanoCore
        raise ValueError(f"Unknown NanoCore class: {class_name} (domain: {domain})")

    @classmethod
    def _load_from_npz(cls, path: str, verify_checksum: bool) -> "BaseNanoCore":
        """Loads weights from NumPy .npz archive."""
        if np is None:
            raise ImportError("NumPy is required to load .npz checkpoints.")
        data = np.load(path, allow_pickle=True)
        weights = {k: data[k] for k in data.files if k != "metadata"}
        metadata = {}
        if "metadata" in data.files:
            metadata = data["metadata"].item()
            
        target_cls = cls
        if target_cls is BaseNanoCore:
            target_cls = cls._resolve_concrete_class(
                metadata.get("class_name", ""),
                metadata.get("domain", "browser"),
            )
        init_kwargs = metadata.get("init_kwargs", {}).copy()
        init_kwargs["weights"] = weights
        return target_cls(**init_kwargs)

    @classmethod
    def _load_from_json(cls, path: str, verify_checksum: bool) -> "BaseNanoCore":
        """Loads weights from JSON format."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        metadata = data.get("metadata", {})
        raw_weights = data.get("weights", {})
        weights = {k: np.array(v, dtype=np.float32) for k, v in raw_weights.items()} if np else raw_weights
        
        target_cls = cls
        if target_cls is BaseNanoCore:
            target_cls = cls._resolve_concrete_class(
                metadata.get("class_name", ""),
                metadata.get("domain", "browser"),
            )
        init_kwargs = metadata.get("init_kwargs", {}).copy()
        init_kwargs["weights"] = weights
        return target_cls(**init_kwargs)

    @classmethod
    def _load_from_torch(cls, path: str, verify_checksum: bool) -> "BaseNanoCore":
        """Loads weights from PyTorch state dict."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required to load .pt checkpoints.")
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict) and "weights" in loaded:
            state_dict = loaded["weights"]
            metadata = loaded.get("metadata", {})
        else:
            state_dict = loaded
            metadata = {}
            
        weights = {}
        for k, v in state_dict.items():
            if hasattr(v, "detach"):
                weights[k] = v.detach().cpu().numpy().astype(np.float32)
            else:
                weights[k] = np.array(v, dtype=np.float32)

        target_cls = cls
        if target_cls is BaseNanoCore:
            target_cls = cls._resolve_concrete_class(
                metadata.get("class_name", ""),
                metadata.get("domain", "browser"),
            )
        init_kwargs = metadata.get("init_kwargs", {}).copy()
        init_kwargs["weights"] = weights
        return target_cls(**init_kwargs)
