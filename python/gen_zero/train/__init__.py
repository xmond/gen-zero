"""Gen-Zero Layer 4: Multi-Task Distillation & Experience Replay."""

from .replay_buffer import StabilityReplayBuffer
from .distiller import GenZeroDistiller
from .dagger_loop import (
    DAggerState,
    DAggerRelabeledSample,
    DAggerExpertRelabeler,
    DAggerCurriculumController,
)

from .compact_replay_buffer import (
    CausalTransition,
    CompressedChunk,
    BufferMemoryStats,
    CompactCausalReplayBuffer,
    GoldenSnapshotRecord,
    GoldenSnapshotManager,
)

__all__ = [
    "StabilityReplayBuffer",
    "GenZeroDistiller",
    "DAggerState",
    "DAggerRelabeledSample",
    "DAggerExpertRelabeler",
    "DAggerCurriculumController",
    "CausalTransition",
    "CompressedChunk",
    "BufferMemoryStats",
    "CompactCausalReplayBuffer",
    "GoldenSnapshotRecord",
    "GoldenSnapshotManager",
]
