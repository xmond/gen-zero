"""Gen-Zero Extreme Quantized CPU Runtime Package.

Modules:
- nano_core: NanoGenZeroCore compact decision model.
- quantized_engine: QuantizedCandidateScorer pure CPU INT8 vectorized engine.
"""

from .nano_core import NanoGenZeroCore
from .quantized_engine import QuantizedCandidateScorer, QuantizedTensor
from .base_nano_core import BaseNanoCore
from .nano_core_browser import NanoCoreBrowser
from .nano_core_vision import NanoCoreVision
from .specialist_nano_core import DomainSpecialistNanoCore
from .composite_decision import (
    CompositeStepDecision,
    CompositeDecisionEngine,
    DEFAULT_ACTION_TYPES
)
from .loop_state_machine import (
    StagnationCircuitBreaker,
    CompositeExecutionLoop,
    LoopExecutionStatus,
    LoopExecutionReport,
    compute_perceptual_fingerprint
)
from .candidate_governance import (
    AntiTruncationRanker,
    GovernedCandidatePool,
    ROLE_WEIGHTS
)

from .debouncer import CoalescedDebouncer, DebounceEvent
from .supervisor_observation import FactoryObservation, smart_truncate_text
from .supervisor_assessment import SupervisorAssessment, SemanticAssessmentAdapter
from .supervisor_policy import (
    SupervisorPolicyStateMachine,
    SupervisorAction,
    SupervisorState,
    SteeringGuidance,
)
from .async_supervisor import AsyncSemanticSupervisor, SupervisorTelemetry
from .h1_entropy_gate import (
    EntropyEvaluation,
    MultiReadStatistics,
    H1EntropyGate,
)
from .confidence_floor import (
    DEFAULT_CONFIDENCE_FLOOR,
    DEFAULT_MAX_ALLOWED_DRIFT,
    FloorGateVerdict,
    ConfidenceFloorGate,
    InvarianceCalibrationResult,
    MultilingualInvarianceCalibrator,
)
from .zstd_codec import (
    compress_bytes,
    decompress_bytes,
    is_zstd_magic,
    get_compression_tier,
)

__all__ = [
    "NanoGenZeroCore",
    "QuantizedCandidateScorer",
    "QuantizedTensor",
    "BaseNanoCore",
    "NanoCoreBrowser",
    "NanoCoreVision",
    "DomainSpecialistNanoCore",
    "CompositeStepDecision",
    "CompositeDecisionEngine",
    "DEFAULT_ACTION_TYPES",
    "StagnationCircuitBreaker",
    "CompositeExecutionLoop",
    "LoopExecutionStatus",
    "LoopExecutionReport",
    "compute_perceptual_fingerprint",
    "AntiTruncationRanker",
    "GovernedCandidatePool",
    "ROLE_WEIGHTS",
    "CoalescedDebouncer",
    "DebounceEvent",
    "FactoryObservation",
    "smart_truncate_text",
    "SupervisorAssessment",
    "SemanticAssessmentAdapter",
    "SupervisorPolicyStateMachine",
    "SupervisorAction",
    "SupervisorState",
    "SteeringGuidance",
    "AsyncSemanticSupervisor",
    "SupervisorTelemetry",
    "EntropyEvaluation",
    "MultiReadStatistics",
    "H1EntropyGate",
    "DEFAULT_CONFIDENCE_FLOOR",
    "DEFAULT_MAX_ALLOWED_DRIFT",
    "FloorGateVerdict",
    "ConfidenceFloorGate",
    "InvarianceCalibrationResult",
    "MultilingualInvarianceCalibrator",
    "compress_bytes",
    "decompress_bytes",
    "is_zstd_magic",
    "get_compression_tier",
]


