"""Gen-Zero: Universal Decision Architecture and Evaluation SDK.

A unified decision architecture integrating:
- Policy + Value Dual-Head with Permutation-Equivariant Set-Attention
- Active Abstain safety guardrail & Prefix KV-cache sharing
- Uncertainty-driven A* & Dual-Head PUCT MCTS planners
- Text World Model for black-box environments
- Online Hard Mining & 1:3 Stability Experience Replay
- Automated Frozen Safety Gate & Meta^n Recursive Self-Improvement
"""

__version__ = "0.1.0"
__author__ = "Google DeepMind pair programming team"

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
