"""Tri-Party Continuous Verifier Harness Package (Issue #13) & Host Harness Scaffolding (RFC-079)."""

from .evidence_sanitizer import (
    ObjectiveEvidence,
    ObjectiveEvidenceSanitizer,
)
from .goal_verifier import (
    VerifierStatus,
    GoalVerificationVerdict,
    GoalVerifier,
)
from .self_healing import (
    SelfHealingTicket,
    InLoopSelfHealingTrigger,
)
from .context_compactor import (
    CompactionResult,
    SessionContextCompactor,
)
from .tri_party_harness import (
    TriPartyStepResult,
    TriPartyHarness,
)
from .rsi_benchmark import (
    ContinuousVerificationMetrics,
    TriPartyVerificationBenchmark,
)
from .action_pipeline import (
    WordSpan,
    WordSpanExtractor,
    ActionSpec,
    PipelineDecision,
    DecideAndFillPipeline,
)
from .semantic_synthesizer import (
    DOMClosureHandle,
    SynthesizedAction,
    SemanticActionSynthesizer,
)
from .web import (
    BoundingBox,
    DOMElement,
    ModalOverlay,
    WebTargetSlot,
    WebObservation,
    WebOperationType,
    BrowserStepResult,
    HarnessError,
    StaleObservationError,
    OccludedElementError,
    InvalidSlotError,
    HarnessSecurityException,
    PointerPenetrationChecker,
    OffscreenDetector,
    DOMObserver,
    PageSignature,
    ClosureHandlePool,
    TextCache,
    VisualEvent,
    VisualTrajectoryRecorder,
    ZeroBrowserHarness,
)
from .setup import setup_host_harness

__all__ = [
    "ObjectiveEvidence",
    "ObjectiveEvidenceSanitizer",
    "VerifierStatus",
    "GoalVerificationVerdict",
    "GoalVerifier",
    "SelfHealingTicket",
    "InLoopSelfHealingTrigger",
    "CompactionResult",
    "SessionContextCompactor",
    "TriPartyStepResult",
    "TriPartyHarness",
    "ContinuousVerificationMetrics",
    "TriPartyVerificationBenchmark",
    "WordSpan",
    "WordSpanExtractor",
    "ActionSpec",
    "PipelineDecision",
    "DecideAndFillPipeline",
    "DOMClosureHandle",
    "SynthesizedAction",
    "SemanticActionSynthesizer",
    "BoundingBox",
    "DOMElement",
    "ModalOverlay",
    "WebTargetSlot",
    "WebObservation",
    "WebOperationType",
    "BrowserStepResult",
    "HarnessError",
    "StaleObservationError",
    "OccludedElementError",
    "InvalidSlotError",
    "HarnessSecurityException",
    "PointerPenetrationChecker",
    "OffscreenDetector",
    "DOMObserver",
    "PageSignature",
    "ClosureHandlePool",
    "TextCache",
    "VisualEvent",
    "VisualTrajectoryRecorder",
    "ZeroBrowserHarness",
    "setup_host_harness",
]
