"""Gen-Zero Causal Reasoning & Intervention Package."""

from .shuffled_benchmark import (
    ShuffledStateBenchmark,
    ShuffledBenchmarkReport,
    compute_cag_and_cgr,
)
from .router import (
    DecisionRouting,
    EpistemicAssessment,
    EpistemicComplexityRouter,
    FormalCalculatorTool,
    IterativeCoTEngine,
)
from .symplectic_thinking import (
    ExitReason,
    NeuralPotential,
    SymplecticThinker,
    ThinkingResult,
)
from .deq_thinking import (
    AndersonAccelerator,
    DEQSolverState,
    DEQThinkingBlock,
    DEQThinkingFunction,
    DEQThinkingModule,
    _np_solve_fixed_point,
)

from .manifold_anchor_distiller import ManifoldAnchorDistiller
from .nanocore_bridge import NanocoreAnchorBridge
from .latent_bridge import (
    CHART_DIM,
    DEFAULT_ZCA_SIGMA_CEILING,
    ENTRY_BYTES,
    ENTRY_DOMAIN,
    LatentEntryContractV1,
    LatentThinkingEngine,
    LatentThinkingResult,
    ManifoldMetricParams,
    MixedManifoldChartProjector,
    MixedManifoldCoord,
    MixedManifoldPotential,
    fallback_geodesic_sq,
    retract_to_manifold,
)

from .cad_engine import (
    CADEngine,
    CADResult,
    CalibrationHead,
    PromptTemplate,
    extract_causal_features,
)
from .numa_affinity import NumaTopology, WorkerPlacement, plan_placements, read_topology
from .verbalizer_extractor import (
    HFVerbalizerExtractor,
    LlamaCppVerbalizerExtractor,
    VerbalizerSpec,
    candidate_groups,
)

__all__ = [
    "CADEngine",
    "CADResult",
    "CalibrationHead",
    "PromptTemplate",
    "extract_causal_features",
    "NumaTopology",
    "WorkerPlacement",
    "plan_placements",
    "read_topology",
    "HFVerbalizerExtractor",
    "LlamaCppVerbalizerExtractor",
    "VerbalizerSpec",
    "candidate_groups",
    "ManifoldAnchorDistiller",
    "NanocoreAnchorBridge",
    "ShuffledStateBenchmark",
    "ShuffledBenchmarkReport",
    "compute_cag_and_cgr",
    "DecisionRouting",
    "EpistemicAssessment",
    "EpistemicComplexityRouter",
    "FormalCalculatorTool",
    "IterativeCoTEngine",
    "ExitReason",
    "NeuralPotential",
    "SymplecticThinker",
    "ThinkingResult",
    "AndersonAccelerator",
    "DEQSolverState",
    "DEQThinkingBlock",
    "DEQThinkingFunction",
    "DEQThinkingModule",
    "CHART_DIM",
    "DEFAULT_ZCA_SIGMA_CEILING",
    "ENTRY_BYTES",
    "ENTRY_DOMAIN",
    "LatentEntryContractV1",
    "LatentThinkingEngine",
    "LatentThinkingResult",
    "ManifoldMetricParams",
    "MixedManifoldChartProjector",
    "MixedManifoldCoord",
    "MixedManifoldPotential",
    "fallback_geodesic_sq",
    "retract_to_manifold",
]
