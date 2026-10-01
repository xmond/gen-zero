"""Gen-Zero Python client SDK for the Rust cognitive decision runtime.

Inference-side components only:
- Policy + Value Dual-Head with Permutation-Equivariant Set-Attention
- Active Abstain safety guardrail & Prefix KV-cache sharing
- Uncertainty-driven A* & Dual-Head PUCT MCTS planners
- Text World Model for black-box environments
- Runtime safety, alignment and perturbation gates

Offline training, replay, distillation and self-play are not part of this
package; they live in gen-zero-research.
"""

__version__ = "0.1.1"
__author__ = "Gen-Zero Authors"

from .config import GenZeroConfig
from .client import GenZero
from .manifold import (
    MasterClosedFormSolver,
    MultiModelGraphLaplacian,
    CandidateSemanticPrior,
    GCCAMidFusion,
    RelativeAnchorEncoder,
    InstanceAdaptiveRouter,
)

__all__ = [
    "GenZero",
    "GenZeroConfig",
    "MasterClosedFormSolver",
    "MultiModelGraphLaplacian",
    "CandidateSemanticPrior",
    "GCCAMidFusion",
    "RelativeAnchorEncoder",
    "InstanceAdaptiveRouter",
]
