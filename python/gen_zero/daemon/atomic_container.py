"""Gen-Zero Atomic Zero-Downtime Model Container.

Ensures lock-free / thread-safe hot-reloading of neural network weights and decision models
during continuous 24/7 background self-evolution:
1. Online inference threads read active model pointer with 0ms interruption.
2. Background training threads atomically swap model pointer via mutex write locks.
3. Maintains rollback checkpoints of last known good models.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import threading
import time
import copy


class ServingSnapshot:
    """Immutable serving snapshot coupling neural model and CPU runtime scorer."""
    def __init__(self, model: Any, scorer: Any, version_id: int, version_tag: str, timestamp: float):
        self._model = model
        self._scorer = scorer
        self._version_id = version_id
        self._version_tag = version_tag
        self._timestamp = timestamp

    @property
    def model(self) -> Any:
        return self._model

    @property
    def scorer(self) -> Any:
        return self._scorer

    @property
    def version_id(self) -> int:
        return self._version_id

    @property
    def version_tag(self) -> str:
        return self._version_tag

    @property
    def timestamp(self) -> float:
        return self._timestamp


class AtomicModelContainer:
    """Thread-safe atomic container for online zero-downtime serving snapshot management."""

    def __init__(self, initial_model: Any, initial_scorer: Any = None, max_history: int = 5):
        self._lock = threading.RLock()
        self._version_id = 1
        self._active_snapshot = ServingSnapshot(
            model=initial_model,
            scorer=initial_scorer,
            version_id=1,
            version_tag="v_init",
            timestamp=time.time()
        )
        self._history_checkpoints: List[ServingSnapshot] = []
        self._max_history = max(1, max_history)
        self._total_swaps = 0
        self._last_swap_time = time.time()

    def get_snapshot(self) -> ServingSnapshot:
        """Returns the currently active immutable serving snapshot."""
        with self._lock:
            return self._active_snapshot

    def get_model(self) -> Any:
        """Returns the currently active model instance from the active snapshot."""
        with self._lock:
            return self._active_snapshot.model

    def get_scorer(self) -> Any:
        """Returns the currently active CPU scorer instance from the active snapshot."""
        with self._lock:
            return self._active_snapshot.scorer

    def swap_snapshot(self, new_model: Any, new_scorer: Any, version_tag: Optional[str] = None) -> Dict[str, Any]:
        """Atomically replaces the serving snapshot (model + scorer) in a single operation.
        
        Guarantees that readers never observe an inconsistent state with new model and old scorer or vice versa.
        """
        t0 = time.perf_counter()
        with self._lock:
            prev_snapshot = self._active_snapshot
            self._history_checkpoints.append(prev_snapshot)
            if len(self._history_checkpoints) > self._max_history:
                self._history_checkpoints.pop(0)

            self._version_id += 1
            new_tag = version_tag or f"gen_{self._version_id}"
            self._active_snapshot = ServingSnapshot(
                model=new_model,
                scorer=new_scorer,
                version_id=self._version_id,
                version_tag=new_tag,
                timestamp=time.time()
            )
            self._total_swaps += 1
            self._last_swap_time = time.time()
            swap_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "status": "SWAPPED",
            "previous_version": prev_snapshot.version_id,
            "new_version": self._version_id,
            "total_swaps": self._total_swaps,
            "swap_latency_ms": round(swap_elapsed_ms, 4),
            "version_tag": new_tag
        }

    def set_pending_scorer(self, scorer: Any) -> None:
        """Stages a candidate scorer to be atomically committed upon next swap_model."""
        with self._lock:
            self._pending_scorer = scorer

    def swap_model(self, new_model: Any, version_tag: Optional[str] = None) -> Dict[str, Any]:
        """Backwards-compatible swap that commits pending scorer if staged, or retains active scorer."""
        with self._lock:
            target_scorer = getattr(self, "_pending_scorer", None)
            self._pending_scorer = None
            if target_scorer is None:
                target_scorer = self._active_snapshot.scorer
        return self.swap_snapshot(new_model, target_scorer, version_tag)

    def rollback(self) -> Dict[str, Any]:
        """Rolls back to the previous known good serving snapshot."""
        with self._lock:
            if not self._history_checkpoints:
                return {"status": "ROLLBACK_FAILED", "reason": "No previous checkpoint in history"}

            prev_snap = self._history_checkpoints.pop()
            rolled_back_from = self._version_id
            self._active_snapshot = prev_snap
            self._version_id = prev_snap.version_id

            return {
                "status": "ROLLED_BACK",
                "restored_version": self._version_id,
                "rolled_back_from": rolled_back_from,
                "checkpoint_timestamp": prev_snap.timestamp
            }

    def rollback_model(self) -> Dict[str, Any]:
        """Alias for rollback() to ensure seamless backwards-compatibility."""
        return self.rollback()


    def get_status(self) -> Dict[str, Any]:
        """Returns container health and version telemetry."""
        with self._lock:
            return {
                "active_version": self._version_id,
                "total_swaps": self._total_swaps,
                "history_depth": len(self._history_checkpoints),
                "seconds_since_last_swap": round(time.time() - self._last_swap_time, 2)
            }
