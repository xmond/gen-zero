"""Gen-Zero Layer 5: runtime safety, alignment and perturbation gates."""

from .safety_gate import SafetyGate, GateVerdict, PerturbationStabilityGate
from .evaluate_perturbations import PerturbationEvaluator
from .locked_evaluator import (
    LockedTestSet,
    CalibrationEvaluator,
    CalibrationReport,
    CalibrationBin,
    LockedEvaluator,
    compute_wilson_score_interval,
    generate_ascii_calibration_curve
)
from .policy_gate import (
    DecisionPolicyGate,
    PolicyGateVerdict,
    PolicyVerdictAction,
    DomainRiskProfile,
    RiskLevel
)
from .contract_drift_gate import (
    ContractDriftGate,
    ContractGateVerdict,
)
from .diff_parser import (
    DiffHunk,
    FileDiff,
    DualContext,
    DiffHunkParser,
    DualContextCollator,
)
from .review_taxonomy import (
    RiskDimension,
    ChangeProfile,
    ReviewAction,
    OwnerDomain,
    MECHANISM_TAXONOMY,
    NO_MATCH,
    NO_ISSUE,
    map_dimension_to_owner,
)
from .staged_review_gate import (
    ReviewIssue,
    ReviewGateVerdict,
    StagedReviewGate,
)
from .cpsat_formal_solver import (
    CPSATFormalSolver,
    CPSATVerdict,
)
from .constraint_compiler import (
    ConstraintLinearProjectionCompiler,
    CompiledConstraintRule,
    CompilationReport,
    ASTConstraintParser,
    Tristate,
    SchemaMismatchError,
    LossyNumericConversionError,
)
from .differentiable_safety_layer import (
    DifferentiableSafetyLayer,
    SafetyProjectionResult,
    PyTorchDifferentiableSafetyModule,
)
from .alignment_gate import (
    AlignmentLevel,
    AlignmentAction,
    EntityFingerprint,
    AlignmentDiagnosisPacket,
    AlignmentVerdict,
    extract_entity_fingerprint,
    StateAlignmentGate,
)
from .fail_closed_scheduler import (
    SystemStatus,
    ModelDecision,
    DefenseAction,
    DualTrackVerdict,
    DualTrackScheduler,
)

__all__ = [
    "SystemStatus",
    "ModelDecision",
    "DefenseAction",
    "DualTrackVerdict",
    "DualTrackScheduler",
    "AlignmentLevel",
    "AlignmentAction",
    "EntityFingerprint",
    "AlignmentDiagnosisPacket",
    "AlignmentVerdict",
    "extract_entity_fingerprint",
    "StateAlignmentGate",
    "SafetyGate",
    "GateVerdict",
    "PerturbationStabilityGate",
    "PerturbationEvaluator",
    "LockedTestSet",
    "CalibrationEvaluator",
    "CalibrationReport",
    "CalibrationBin",
    "LockedEvaluator",
    "compute_wilson_score_interval",
    "generate_ascii_calibration_curve",
    "DecisionPolicyGate",
    "PolicyGateVerdict",
    "PolicyVerdictAction",
    "DomainRiskProfile",
    "RiskLevel",
    "ContractDriftGate",
    "ContractGateVerdict",
    "DiffHunk",
    "FileDiff",
    "DualContext",
    "DiffHunkParser",
    "DualContextCollator",
    "RiskDimension",
    "ChangeProfile",
    "ReviewAction",
    "OwnerDomain",
    "MECHANISM_TAXONOMY",
    "NO_MATCH",
    "NO_ISSUE",
    "map_dimension_to_owner",
    "ReviewIssue",
    "ReviewGateVerdict",
    "StagedReviewGate",
    "CPSATFormalSolver",
    "CPSATVerdict",
    "ConstraintLinearProjectionCompiler",
    "CompiledConstraintRule",
    "CompilationReport",
    "ASTConstraintParser",
    "Tristate",
    "SchemaMismatchError",
    "DifferentiableSafetyLayer",
    "SafetyProjectionResult",
    "PyTorchDifferentiableSafetyModule",
]

