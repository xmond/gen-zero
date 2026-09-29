"""Gen-Zero Manifold Package: closed-form manifold-regularised linear heads and decision fusion."""

from .master_objective import (
    MasterClosedFormSolver,
    MasterObjectiveError,
    SolveDiagnostics,
    ordinal_expected_decode,
    ordinal_soft_targets,
)
from .graph_laplacian import MultiModelGraphLaplacian
from .candidate_prior import CandidateSemanticPrior
from .gcca_fusion import GCCAMidFusion, RelativeAnchorEncoder
from .adaptive_gating import (
    InstanceAdaptiveRouter,
    extract_reliability_features,
    jensen_shannon_divergence,
    normalized_entropy,
    top1_top2_margin,
)

__all__ = [
    "MasterClosedFormSolver",
    "MasterObjectiveError",
    "SolveDiagnostics",
    "MultiModelGraphLaplacian",
    "CandidateSemanticPrior",
    "GCCAMidFusion",
    "RelativeAnchorEncoder",
    "InstanceAdaptiveRouter",
    "extract_reliability_features",
    "jensen_shannon_divergence",
    "normalized_entropy",
    "top1_top2_margin",
    "ordinal_expected_decode",
    "ordinal_soft_targets",
]
