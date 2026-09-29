"""Gen-Zero Gateway Layer: Cloud-Edge GPU Arbiter Bridge.

Handles automatic fallback when local CPU or edge planner experiences high entropy
or low confidence on out-of-distribution (OOD) states:
1. Detects low confidence (< threshold) or high Shannon entropy (> threshold).
2. Asynchronously / Synchronously dispatches unconfident frames to the high-capacity
   Qwen3.5-0.8B GPU model on A100 for authoritative global arbitration.
3. Automatically mines unconfident frames into HardSampleMiner for continuous self-evolution.
"""

from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
import math
import time
import threading
import json
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


def _is_real_gpu_engine(engine: Any) -> bool:
    """Fail-closed check: only a loaded, weight-backed GPU engine counts as real.

    An engine declares itself real via an explicit ``is_real`` attribute
    (bool or no-arg callable), or via the ``PyTorchVisualDecisionEngine``
    contract of ``_is_loaded=True`` with a non-null ``model``. Anything else
    (unloaded placeholders, ``MagicMock`` stand-ins, unknown shapes) is
    treated as untrusted so it never scores production arbitration.
    """
    if engine is None:
        return False
    if hasattr(engine, "is_real"):
        flag = engine.is_real
        return (flag() is True) if callable(flag) else (flag is True)
    if hasattr(engine, "_is_loaded"):
        return getattr(engine, "_is_loaded", False) is True and getattr(engine, "model", None) is not None
    return False


def _sanitize_for_json(data: Any) -> Any:
    """Recursively cleans float('nan'), float('inf'), float('-inf') to None or safe finite values."""
    import math
    if isinstance(data, float):
        if not math.isfinite(data):
            return None
        return data
    elif isinstance(data, dict):
        return {k: _sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [_sanitize_for_json(v) for v in data]
    return data


class ArbiterVerdict:
    """Represents the verdict returned by the GPU Arbiter."""
    def __init__(
        self,
        action: Optional[str],
        confidence: float,
        probs: Dict[str, float],
        is_fallback: bool,
        arbitration_source: str,
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        backend_reachable: bool = False
    ):
        self.action = action
        self.confidence = confidence
        self.probs = probs
        self.is_fallback = is_fallback
        self.arbitration_source = arbitration_source
        self.latency_ms = latency_ms
        self.metadata = metadata or {}
        self.backend_reachable = backend_reachable

    def to_dict(self) -> Dict[str, Any]:
        import math
        safe_conf = self.confidence if (isinstance(self.confidence, (int, float)) and math.isfinite(self.confidence)) else 0.0
        safe_probs = {}
        for k, v in (self.probs or {}).items():
            safe_probs[k] = round(float(v), 4) if (isinstance(v, (int, float)) and math.isfinite(v)) else 0.0
        safe_lat = self.latency_ms if (isinstance(self.latency_ms, (int, float)) and math.isfinite(self.latency_ms)) else 0.0
        return {
            "action": self.action,
            "confidence": round(float(safe_conf), 4),
            "probs": safe_probs,
            "is_fallback": self.is_fallback,
            "arbitration_source": self.arbitration_source,
            "latency_ms": round(float(safe_lat), 2),
            "metadata": _sanitize_for_json(self.metadata),
            "backend_reachable": self.backend_reachable
        }



class CloudGPUArbiterBridge:
    """Manages dynamic fallback and asynchronous dispatch to the remote Qwen GPU Arbiter."""

    def __init__(
        self,
        confidence_threshold: float = 0.40,
        entropy_threshold: float = 0.75,
        remote_endpoint: Optional[str] = None,
        local_gpu_engine: Optional[Any] = None,
        miner: Optional[Any] = None,
        timeout_s: float = 2.0
    ):
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.remote_endpoint = remote_endpoint
        self.local_gpu_engine = local_gpu_engine
        self.miner = miner
        self.timeout_s = float(timeout_s)
        self.mined_history = []
        self._async_tasks = []
        self._arbitration_count = 0
        self._fallback_count = 0
        self._unreachable_count = 0
        self._error_count = 0

    def should_trigger_fallback(self, confidence: float, entropy: Optional[float] = None) -> bool:
        """Determines if the decision is too unconfident to proceed without GPU arbitration."""
        if confidence < self.confidence_threshold:
            return True
        if entropy is not None and entropy > self.entropy_threshold:
            return True
        return False

    def arbitrate(
        self,
        state: Any,
        candidates: List[str],
        local_action: str,
        local_confidence: float,
        local_probs: Dict[str, float],
        mode: str = "async"
    ) -> ArbiterVerdict:
        """Executes arbitration.
        
        Args:
            state: Unconfident state payload (text, image path, or dict).
            candidates: Available action choices.
            local_action: Decision made by local CPU/reflex planner.
            local_confidence: Confidence score from local planner.
            local_probs: Probability distribution from local planner.
            mode: 'async' (non-blocking, keeps local action or ABSTAIN, sends to GPU for training)
                  | 'sync' (blocks to wait for GPU global verdict).
        """
        t0 = time.perf_counter()
        self._fallback_count += 1
        
        transition_record = {
            "state": state,
            "action": local_action,
            "confidence": local_confidence,
            "probs": local_probs,
            "candidates": candidates,
            "timestamp": time.time()
        }
        self.mined_history.append(transition_record)

        # 2. Synchronous mode: wait for authoritative Qwen GPU Arbiter
        if mode == "sync":
            gpu_res = self._call_gpu_model(state, candidates)
            latency = (time.perf_counter() - t0) * 1000.0
            reachable = gpu_res.get("backend_reachable", False) is True
            return ArbiterVerdict(
                action=gpu_res.get("best_action", local_action) if reachable else None,
                confidence=gpu_res.get("confidence", local_confidence) if reachable else 0.0,
                probs=gpu_res.get("probs", local_probs) if reachable else {c: 0.0 for c in candidates},
                is_fallback=True,
                arbitration_source=gpu_res.get("arbitration_source", "fail_closed_unreachable"),
                latency_ms=latency,
                metadata={
                    "local_initial_action": local_action,
                    "mined_as_hard_sample": True,
                    "status": gpu_res.get("status", "ok" if reachable else "failed"),
                },
                backend_reachable=reachable
            )


        # 3. Asynchronous mode: non-blocking dispatch to GPU, local returns conservative action
        can_dispatch = _is_real_gpu_engine(self.local_gpu_engine) or bool(self.remote_endpoint)
        dispatched = False
        if can_dispatch:
            thread = threading.Thread(
                target=self._async_dispatch_and_mine,
                args=(state, candidates, local_action),
                daemon=True
            )
            thread.start()
            self._async_tasks.append(thread)
            dispatched = True
        else:
            logger.warning(
                "Arbiter async dispatch skipped: no real local GPU engine and no remote_endpoint configured; "
                "refusing to fabricate a GPU-arbitrated decision."
            )

        latency = (time.perf_counter() - t0) * 1000.0
        # If candidates has ABSTAIN or STAY, prefer it under extreme ambiguity
        safe_action = "ABSTAIN" if "ABSTAIN" in candidates else local_action
        return ArbiterVerdict(
            action=safe_action,
            confidence=local_confidence,
            probs=local_probs,
            is_fallback=True,
            arbitration_source="edge_safe_abstain_async_gpu_mining",
            latency_ms=latency,
            metadata={"async_dispatched_to_gpu": dispatched, "local_action": local_action},
            backend_reachable=False
        )

    def _async_dispatch_and_mine(self, state: Any, candidates: List[str], local_action: str):
        """Background thread executing GPU forward pass and mining labels."""
        try:
            res = self._call_gpu_model(state, candidates)
            # `_call_gpu_model` returns backend_reachable=False when it failed closed
            # (no real engine, no remote, or remote failed). That is not a real
            # arbitration and must not inflate the success count.
            if res.get("backend_reachable", False) is True:
                self._arbitration_count += 1
            else:
                self._unreachable_count += 1
        except Exception:
            self._error_count += 1
            logger.exception("Async arbiter dispatch failed while mining a hard sample")

    def _call_gpu_model(self, state: Any, candidates: List[str]) -> Dict[str, Any]:
        """Invokes either in-process GPU engine or remote HTTP GPU endpoint.

        Fail-closed dispatch order: a real (weight-loaded) local GPU engine
        first, then the remote HTTP arbiter, and only if both are unavailable
        a marked-unreachable result. A mock/placeholder local engine is never
        invoked for scoring -- it must not shadow the remote channel.
        """
        # A. In-process PyTorch Visual / Language Engine (only if genuinely loaded)
        if self.local_gpu_engine is not None:
            if _is_real_gpu_engine(self.local_gpu_engine):
                prefill = self.local_gpu_engine.prefill_visual_context(state, prompt="Arbitrate best action")
                result = self.local_gpu_engine.score_candidates_direct(prefill, candidates)
                if "confidence" not in result:
                    probs = result.get("probs") or {}
                    result["confidence"] = max(probs.values()) if probs else 0.0
                result["backend_reachable"] = True
                result["arbitration_source"] = "local_gpu_engine"
                result.setdefault("status", "ok")
                return result
            logger.warning(
                "local_gpu_engine=%r is not a real weight-loaded GPU engine (mock/placeholder); "
                "refusing char-hash mock scoring and routing to the remote arbiter instead",
                type(self.local_gpu_engine).__name__,
            )

        # B. Remote HTTP Endpoint (if configured and running)
        if self.remote_endpoint:
            try:
                payload = json.dumps({"state": str(state), "candidates": candidates}).encode("utf-8")
                req = urllib.request.Request(
                    self.remote_endpoint,
                    data=payload,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                # A successful transport is not proof of a valid arbitration verdict.
                if not isinstance(result, dict):
                    raise ValueError("Remote arbiter response must be an object")
                action = result.get("best_action")
                probs = result.get("probs")
                if not isinstance(action, str) or action not in candidates:
                    raise ValueError("Remote arbiter best_action must be a candidate")
                if (
                    not isinstance(probs, dict) or not probs or action not in probs
                    or any(
                        key not in candidates or type(value) not in (int, float)
                        or not math.isfinite(value) or not 0.0 <= value <= 1.0
                        for key, value in probs.items()
                    )
                    or probs[action] <= 0.0
                ):
                    raise ValueError("Remote arbiter probs must contain valid candidate probabilities")
                if result.get("status", "ok") != "ok" or result.get("backend_reachable", True) is not True:
                    raise ValueError("Remote arbiter reported failure")
                confidence = result.get("confidence", probs[action])
                if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                    raise ValueError("Remote arbiter confidence must be a finite probability")
                result["confidence"] = confidence
                result["backend_reachable"] = True
                result["arbitration_source"] = "remote_http_arbiter"
                result.setdefault("status", "ok")
                return result
            except Exception as exc:
                logger.warning(
                    "Remote arbiter endpoint %s unavailable or protocol invalid (%s: %s)",
                    self.remote_endpoint, type(exc).__name__, exc,
                )
                if isinstance(exc, ValueError):
                    return {
                        "best_action": None,
                        "confidence": 0.0,
                        "probs": {c: 0.0 for c in candidates},
                        "backend_reachable": False,
                        "status": "failed",
                        "arbitration_source": "fail_closed_malformed_remote",
                        "error": str(exc),
                    }

        # C. Fail-closed: neither a real local engine nor a reachable remote endpoint.
        logger.error(
            "GPU arbiter fully unreachable (no real local engine, no reachable remote endpoint); "
            "failing closed instead of fabricating a verdict."
        )
        return {
            "best_action": None,
            "confidence": 0.0,
            "probs": {c: 0.0 for c in candidates},
            "backend_reachable": False,
            "status": "failed",
            "arbitration_source": "fail_closed_unreachable",
        }
