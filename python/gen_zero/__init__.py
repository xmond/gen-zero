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

_OPTIONAL_DEPS = {"torch", "scipy"}


def _optional_dep_missing(exc):
    # Only a missing torch/scipy is tolerated; any other missing module is a real bug.
    return (exc.name or "").split(".")[0] in _OPTIONAL_DEPS


from .config import GenZeroConfig

# torch/scipy are optional: light submodules such as the numpy-only
# causal.nanocore_bridge must stay importable on hosts without them.
try:
    from .client import GenZero
except ModuleNotFoundError as exc:
    if not _optional_dep_missing(exc):
        raise
try:
    from .manifold import (
        MasterClosedFormSolver,
        MultiModelGraphLaplacian,
        CandidateSemanticPrior,
        GCCAMidFusion,
        RelativeAnchorEncoder,
        InstanceAdaptiveRouter,
    )
except ModuleNotFoundError as exc:
    if not _optional_dep_missing(exc):
        raise

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
