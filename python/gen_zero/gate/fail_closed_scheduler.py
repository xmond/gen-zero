"""Dual-Track Decoupled State Machine & Strict Fail-Closed Circuit Breaker.

Implements Module 1 of Issue #28:
- Orthogonal dual-track state space:
  SystemStatus in {SUCCESS, TIMEOUT, OOM, ENGINE_PANIC}
  ModelDecision in {PASS, FAIL, UNCERTAIN}
- Strict Fail-Closed Axiom:
  If SystemStatus != SUCCESS (timeout > 50ms, OOM, panic), 100% force DefenseAction.BLOCK
  with is_fail_closed=True and trigger infrastructure SLA alert.
  Never allow silent fail-open pass-through under engine overload or failure!
- System failure vs Model uncertainty decoupling:
  System failure -> logs SLA incident and blocks.
  Model UNCERTAIN -> routes to Human-in-the-Loop or conservative fallback.
"""

from typing import Dict, List, Any, Optional, Tuple, Callable
import dataclasses
import enum
import time
import traceback


class SystemStatus(str, enum.Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    OOM = "oom"
    ENGINE_PANIC = "engine_panic"


class ModelDecision(str, enum.Enum):
    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"


class DefenseAction(str, enum.Enum):
    PASS = "pass"
    BLOCK = "block"
    UNCERTAIN_ESCALATE = "uncertain_escalate"


@dataclasses.dataclass
class DualTrackVerdict:
    system_status: SystemStatus
    model_decision: Optional[ModelDecision]
    action: DefenseAction
    is_fail_closed: bool
    risk_probability: Optional[float]
    tau_low: float
    tau_high: float
    latency_ms: float
    alert_triggered: bool
    diagnostics: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_status": self.system_status.value,
            "model_decision": self.model_decision.value if self.model_decision else None,
            "action": self.action.value,
            "is_fail_closed": self.is_fail_closed,
            "risk_probability": round(self.risk_probability, 4) if self.risk_probability is not None else None,
            "thresholds": {"tau_low": self.tau_low, "tau_high": self.tau_high},
            "latency_ms": round(self.latency_ms, 2),
            "alert_triggered": self.alert_triggered,
            "diagnostics": self.diagnostics,
        }


class DualTrackScheduler:
    """Decoupled dual-track state machine with strict Fail-Closed safety guarantee."""

    def __init__(
        self,
        timeout_ms: float = 50.0,
        tau_low: float = 0.20,
        tau_high: float = 0.65,
        on_alert_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.timeout_ms = float(timeout_ms)
        self.tau_low = float(tau_low)
        self.tau_high = float(tau_high)
        self.on_alert_callback = on_alert_callback
        self._injected_fault: Optional[str] = None
        self.sla_incident_log: List[Dict[str, Any]] = []

    def inject_fault(self, fault_type: Optional[str]) -> None:
        """Injects a simulated runtime failure ('timeout', 'oom', 'engine_panic', or None to reset)."""
        valid_faults = {None, "timeout", "oom", "engine_panic"}
        if fault_type not in valid_faults:
            raise ValueError(f"Invalid fault_type: {fault_type}. Must be one of {valid_faults}")
        self._injected_fault = fault_type

    def clear_fault(self) -> None:
        """Clears any active fault injection."""
        self._injected_fault = None

    def execute_evaluation(
        self,
        context: Dict[str, Any],
        evaluation_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
    ) -> DualTrackVerdict:
        """Executes full evaluation under strict Fail-Closed supervision."""
        t0 = time.perf_counter()
        system_status = SystemStatus.SUCCESS
        risk_probability: Optional[float] = None
        model_decision: Optional[ModelDecision] = None
        diagnostics: Dict[str, Any] = {}
        alert_triggered = False

        # Step 1: Check for injected fault or execute function with boundary protection
        try:
            if self._injected_fault == "timeout":
                time.sleep(min(0.06, (self.timeout_ms + 5.0) / 1000.0))
                raise TimeoutError(f"Prefill forward execution timed out (> {self.timeout_ms}ms)")
            elif self._injected_fault == "oom":
                raise MemoryError("CUDA out of memory: tried to allocate 2.40 GiB")
            elif self._injected_fault == "engine_panic":
                raise RuntimeError("Engine segmentation fault: worker core corrupted")

            if evaluation_fn is not None:
                risk_probability = float(evaluation_fn(context))
            else:
                # Default baseline evaluation from context
                risk_probability = float(context.get("risk_score", 0.10))

            latency_ms = (time.perf_counter() - t0) * 1000.0
            if latency_ms > self.timeout_ms:
                raise TimeoutError(f"Prefill forward execution latency {latency_ms:.1f}ms exceeded SLA limit {self.timeout_ms}ms")

        except TimeoutError as exc:
            system_status = SystemStatus.TIMEOUT
            diagnostics["error"] = str(exc)
            diagnostics["traceback"] = traceback.format_exc()
        except MemoryError as exc:
            system_status = SystemStatus.OOM
            diagnostics["error"] = str(exc)
            diagnostics["traceback"] = traceback.format_exc()
        except Exception as exc:
            system_status = SystemStatus.ENGINE_PANIC
            diagnostics["error"] = str(exc)
            diagnostics["traceback"] = traceback.format_exc()

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Step 2: Evaluate Strict Fail-Closed Axiom
        if system_status != SystemStatus.SUCCESS:
            # 100% Mandatory Fail-Closed: Never silently pass on engine failure!
            action = DefenseAction.BLOCK
            is_fail_closed = True
            alert_triggered = True
            model_decision = None
            risk_probability = None

            # Record infrastructure SLA incident
            incident_record = {
                "timestamp": time.time(),
                "system_status": system_status.value,
                "latency_ms": round(latency_ms, 2),
                "error": diagnostics.get("error", "Unknown system fault"),
                "context_summary": {k: str(v)[:64] for k, v in context.items()},
            }
            self.sla_incident_log.append(incident_record)

            if self.on_alert_callback is not None:
                try:
                    self.on_alert_callback(incident_record)
                except Exception:
                    pass

            return DualTrackVerdict(
                system_status=system_status,
                model_decision=model_decision,
                action=action,
                is_fail_closed=is_fail_closed,
                risk_probability=risk_probability,
                tau_low=self.tau_low,
                tau_high=self.tau_high,
                latency_ms=latency_ms,
                alert_triggered=alert_triggered,
                diagnostics=diagnostics,
            )

        # Step 3: Track 2: Model Decision Evaluation (System completed successfully)
        assert risk_probability is not None
        risk_probability = float(max(0.0, min(1.0, risk_probability)))

        if risk_probability < self.tau_low:
            model_decision = ModelDecision.PASS
            action = DefenseAction.PASS
        elif risk_probability <= self.tau_high:
            model_decision = ModelDecision.UNCERTAIN
            action = DefenseAction.UNCERTAIN_ESCALATE  # Route to Human-in-the-Loop
            diagnostics["routing"] = "escalated_to_human_review"
        else:
            model_decision = ModelDecision.FAIL
            action = DefenseAction.BLOCK
            diagnostics["routing"] = "risk_intercepted"

        return DualTrackVerdict(
            system_status=SystemStatus.SUCCESS,
            model_decision=model_decision,
            action=action,
            is_fail_closed=False,
            risk_probability=risk_probability,
            tau_low=self.tau_low,
            tau_high=self.tau_high,
            latency_ms=latency_ms,
            alert_triggered=False,
            diagnostics=diagnostics,
        )
