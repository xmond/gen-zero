"""Gen-Zero Layer 5: Frozen Snapshot Benchmark & Honest Telemetry Architecture.

RFC Implementation for Issue #8 (Module 5):
1. Lightweight Frozen Snapshot Benchmark:
   - Evaluates offline static environment snapshots in seconds.
   - Computes Target Accuracy, Action Accuracy, Joint Accuracy, and 10-Bin ECE.
2. Honest Telemetry Metrics:
   - Strictly separates autonomous model decisions from takeovers/bypasses.
   - Explicitly tracks autonomous_rate vs takeover_rate.
   - Forbids counting human confirmations or planner bypasses as autonomous model successes.
"""

from dataclasses import dataclass, field, asdict
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..gate.locked_evaluator import CalibrationEvaluator, CalibrationReport
from ..runtime.composite_decision import CompositeStepDecision, CompositeDecisionEngine
from ..gate.policy_gate import DecisionPolicyGate, PolicyGateVerdict, PolicyVerdictAction


@dataclass
class SnapshotItem:
    id: str
    state: Any
    affordances: List[str]
    goal: str
    expected_target: str
    expected_action: str
    state_feedback: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HonestTelemetryRecord:
    total_steps: int
    autonomous_decisions: int
    bypassed_steps: int
    confirmed_steps: int
    escalated_steps: int
    stopped_steps: int

    @property
    def autonomous_rate(self) -> float:
        return self.autonomous_decisions / max(1, self.total_steps)

    @property
    def takeover_rate(self) -> float:
        takeovers = self.bypassed_steps + self.confirmed_steps + self.escalated_steps + self.stopped_steps
        return takeovers / max(1, self.total_steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "autonomous_decisions": self.autonomous_decisions,
            "bypassed_steps": self.bypassed_steps,
            "confirmed_steps": self.confirmed_steps,
            "escalated_steps": self.escalated_steps,
            "stopped_steps": self.stopped_steps,
            "autonomous_rate": round(self.autonomous_rate, 4),
            "takeover_rate": round(self.takeover_rate, 4)
        }


class HonestTelemetryTracker:
    """Audits decision paths and computes uncompromised autonomy metrics."""

    def __init__(self):
        self.total_steps: int = 0
        self.autonomous_decisions: int = 0
        self.bypassed_steps: int = 0
        self.confirmed_steps: int = 0
        self.escalated_steps: int = 0
        self.stopped_steps: int = 0

    def record_step(
        self,
        gate_verdict: PolicyGateVerdict,
        was_bypassed_by_planner: bool = False
    ) -> None:
        """Records a single step execution path under strict truth-in-metric auditing."""
        self.total_steps += 1

        if was_bypassed_by_planner:
            self.bypassed_steps += 1
        elif gate_verdict.action == PolicyVerdictAction.CONFIRM:
            self.confirmed_steps += 1
        elif gate_verdict.action == PolicyVerdictAction.ESCALATE:
            self.escalated_steps += 1
        elif gate_verdict.action == PolicyVerdictAction.STOP:
            self.stopped_steps += 1
        elif gate_verdict.action == PolicyVerdictAction.PROCEED:
            self.autonomous_decisions += 1
        else:
            self.escalated_steps += 1

    def get_summary(self) -> HonestTelemetryRecord:
        return HonestTelemetryRecord(
            total_steps=self.total_steps,
            autonomous_decisions=self.autonomous_decisions,
            bypassed_steps=self.bypassed_steps,
            confirmed_steps=self.confirmed_steps,
            escalated_steps=self.escalated_steps,
            stopped_steps=self.stopped_steps
        )

    def reset(self) -> None:
        self.total_steps = 0
        self.autonomous_decisions = 0
        self.bypassed_steps = 0
        self.confirmed_steps = 0
        self.escalated_steps = 0
        self.stopped_steps = 0


@dataclass
class SnapshotBenchmarkReport:
    total_snapshots: int
    target_accuracy: float
    action_accuracy: float
    joint_accuracy: float
    calibration: Dict[str, Any]
    telemetry: Dict[str, Any]
    latency_p50_ms: float
    latency_p95_ms: float
    total_duration_sec: float
    # A default CompositeDecisionEngine has no model client and uses a
    # deterministic lexical fallback.  Keep this provenance on the report so
    # its accuracy cannot be mistaken for model evidence.
    is_synthetic: bool = True
    model_provenance: str = "fallback_or_unverified"
    metrics_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_snapshots": self.total_snapshots,
            "target_accuracy": round(self.target_accuracy, 2),
            "action_accuracy": round(self.action_accuracy, 2),
            "joint_accuracy": round(self.joint_accuracy, 2),
            "calibration": self.calibration,
            "telemetry": self.telemetry,
            "latency_p50_ms": round(self.latency_p50_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
            "total_duration_sec": round(self.total_duration_sec, 3),
            "is_synthetic": self.is_synthetic,
            "model_provenance": self.model_provenance,
            "metrics_valid": self.metrics_valid,
        }


class FrozenSnapshotBenchmark:
    """Offline benchmark harness evaluating model decisions against static frozen snapshots."""

    def __init__(
        self,
        snapshots: List[SnapshotItem],
        decision_engine: Optional[CompositeDecisionEngine] = None,
        policy_gate: Optional[DecisionPolicyGate] = None
    ):
        self.snapshots = snapshots
        self.decision_engine = decision_engine or CompositeDecisionEngine()
        self.policy_gate = policy_gate or DecisionPolicyGate()

    def run_benchmark(self) -> SnapshotBenchmarkReport:
        """Runs offline evaluation in seconds, outputting accuracy, ECE, and honest telemetry."""
        t_start = time.perf_counter()
        latencies_ms: List[float] = []
        target_hits = 0
        action_hits = 0
        joint_hits = 0

        calib_predictions: List[Dict[str, Any]] = []
        telemetry = HonestTelemetryTracker()

        for item in self.snapshots:
            t0 = time.perf_counter()
            decision = self.decision_engine.decide_step(
                state=item.state,
                affordances=item.affordances,
                goal=item.goal,
                state_feedback=item.state_feedback
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(lat_ms)

            # Check correctness
            t_correct = (decision.target == item.expected_target)
            a_correct = (decision.action == item.expected_action)
            j_correct = (t_correct and a_correct)

            if t_correct:
                target_hits += 1
            if a_correct:
                action_hits += 1
            if j_correct:
                joint_hits += 1

            # Check policy gate
            gate_res = self.policy_gate.evaluate_policy(decision, state=item.state)
            telemetry.record_step(gate_res)

            # Record calibration prediction for joint decision
            joint_conf = min(decision.target_confidence, decision.action_confidence)
            calib_predictions.append({
                "sample_id": item.id,
                "confidence": joint_conf,
                "predicted_label": (decision.target, decision.action),
                "ground_truth": (item.expected_target, item.expected_action)
            })

        n = len(self.snapshots)
        t_acc = (target_hits / max(1, n)) * 100.0
        a_acc = (action_hits / max(1, n)) * 100.0
        j_acc = (joint_hits / max(1, n)) * 100.0

        # Calibration
        calib_report = CalibrationEvaluator.compute_calibration(calib_predictions)
        calibration = calib_report.to_dict()
        if not calib_predictions:
            # CalibrationEvaluator intentionally returns an EMPTY_PREDICTIONS
            # report for reuse by other callers.  A benchmark must not turn
            # that absence of evidence into a passing safety gate.
            calibration["passed_safety_red_line"] = False
            calibration["verdict"] = "NO_SAMPLES"

        # Latencies
        latencies_ms.sort()
        p50 = latencies_ms[int(len(latencies_ms) * 0.50)] if latencies_ms else 0.0
        p95 = latencies_ms[int(len(latencies_ms) * 0.95)] if latencies_ms else 0.0
        total_dur = time.perf_counter() - t_start

        model_client = getattr(self.decision_engine, "client", None)
        checkpoint_loaded = getattr(model_client, "weights_loaded_from_checkpoint", None) is True
        is_synthetic = model_client is None or not checkpoint_loaded
        model_provenance = "checkpoint" if checkpoint_loaded else "fallback_or_unverified"

        return SnapshotBenchmarkReport(
            total_snapshots=n,
            target_accuracy=t_acc,
            action_accuracy=a_acc,
            joint_accuracy=j_acc,
            calibration=calibration,
            telemetry=telemetry.get_summary().to_dict(),
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            total_duration_sec=total_dur,
            is_synthetic=is_synthetic,
            model_provenance=model_provenance,
            metrics_valid=bool(self.snapshots),
        )
