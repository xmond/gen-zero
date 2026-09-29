"""Gen-Zero Streaming Spatial-Temporal World Model Package.

Modules:
- rolling_kv_cache: Attention Sinks + Sliding Window for constant O(W) memory.
- latent_dynamics: Residual latent transition model with Pearl causal shock detection.
- imagination_planner: Ultra-fast (<1.5ms) MCTS forward search entirely inside latent space.
- streaming_engine: High-level streaming decision engine.
- neural_dynamics: Checkpoint-loaded residual transition + safety/reward model (fail-closed).
"""

from .rolling_kv_cache import RollingVisionKVCache
from .latent_dynamics import LatentTransitionModel
from .imagination_planner import ImaginationMCTSPlanner
from .streaming_engine import StreamingWorldModelEngine
from .compressed_tree import (
    CompressedState,
    TreeMemoryStats,
    CompressedTreeNode,
    TieredTreeMemoryManager,
    CompressedImaginationMCTS,
)
from .hamiltonian_dynamics import (
    HamiltonianWorldModel,
    HamiltonianStepResult,
    HamiltonianRolloutResult,
    PyTorchHamiltonianNeuralODE,
    PotentialEnergyNetwork,
)
from .text_world_model import GenZeroTextWorldModel
from .neural_dynamics import NeuralDynamicsWorldModel


def __getattr__(name: str):
    if name in (
        "WorldModelNanoCoreOrchestrator",
        "WorldModelOrchestrationResult",
        "NanoCoreClusterStatus",
        "ImaginedStepTelemetry",
    ):
        from gen_zero.nanocore.world_model_orchestrator import (
            WorldModelNanoCoreOrchestrator,
            WorldModelOrchestrationResult,
            NanoCoreClusterStatus,
            ImaginedStepTelemetry,
        )
        mapping = {
            "WorldModelNanoCoreOrchestrator": WorldModelNanoCoreOrchestrator,
            "WorldModelOrchestrationResult": WorldModelOrchestrationResult,
            "NanoCoreClusterStatus": NanoCoreClusterStatus,
            "ImaginedStepTelemetry": ImaginedStepTelemetry,
        }
        return mapping[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "RollingVisionKVCache",
    "LatentTransitionModel",
    "ImaginationMCTSPlanner",
    "StreamingWorldModelEngine",
    "WorldModelNanoCoreOrchestrator",
    "WorldModelOrchestrationResult",
    "NanoCoreClusterStatus",
    "ImaginedStepTelemetry",
    "CompressedState",
    "TreeMemoryStats",
    "CompressedTreeNode",
    "TieredTreeMemoryManager",
    "CompressedImaginationMCTS",
    "HamiltonianWorldModel",
    "HamiltonianStepResult",
    "HamiltonianRolloutResult",
    "PyTorchHamiltonianNeuralODE",
    "PotentialEnergyNetwork",
    "GenZeroTextWorldModel",
    "NeuralDynamicsWorldModel",
]
