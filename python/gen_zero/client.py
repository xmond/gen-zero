"""Gen-Zero Top-Level Public Interface.

Provides clean, intuitive one-line entrypoints for:
- Single-step fast decision (System 1)
- Multi-step lookahead planning (System 2 MCTS / Uncertainty A*)
- Online self-evolution flywheel (RSI)
"""

from typing import Dict, List, Any, Mapping, Optional, Tuple, Callable, Sequence, Union, Set
import copy
import math
import os
import time
import logging
from pathlib import Path
import numpy as np

from .config import GenZeroConfig
from .world_model.hamiltonian_dynamics import HamiltonianWorldModel
from .world_model.neural_dynamics import NeuralDynamicsWorldModel
from .world_model.simulation import (
    PROVENANCE_GRID,
    PROVENANCE_NEURAL,
    PROVENANCE_TEXT,
    SOURCE_HEURISTIC,
    SOURCE_NEURAL,
    StepOutcome,
    fixed_plan_policy,
    greedy_lookahead_policy,
    rank_candidates,
    rollout,
    summarize_candidate,
    to_jsonable,
    validate_horizon,
    wrap_caller_transition,
)
from .model.dual_head import GenZeroDualHeadModel, encode_leaf_tokens

# Whitelisted actions for the symbolic grid world; anything else fails closed instead of
# silently no-op'ing through GenZeroTextWorldModel's `dirs.get(action, (0, 0))` fallback.
_GRID_ACTIONS = frozenset({"north", "south", "east", "west"})


class UnsupportedStateError(ValueError):
    """The state/action shape given to a rollout step_fn doesn't match any known world model.

    Distinct from a plain ValueError so callers (notably decide()'s trailing bonus
    rollout) can catch exactly this case and downgrade it to a 'trajectory_status', while
    any other ValueError -- a broken caller-supplied transition_fn, a non-finite reward, a
    bad horizon -- is a real contract violation and must propagate uncaught (fail closed).
    """


def _validate_grid_state_schema(state: Dict[str, Any]) -> None:
    """Whitelist check for the symbolic grid-world dict schema; unknown shapes fail closed."""
    size = state.get("size")
    body = state.get("body")
    valid_size = isinstance(size, int) and not isinstance(size, bool) and size > 0
    valid_body = (
        isinstance(body, list) and len(body) > 0
        and all(
            isinstance(pt, (list, tuple)) and len(pt) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) for v in pt)
            for pt in body
        )
    )
    if "size" not in state or "body" not in state or not valid_size or not valid_body:
        raise UnsupportedStateError(f"Unsupported dict state schema: {list(state.keys())}")


def _sanitize_for_json(data: Any) -> Any:
    """Recursively cleans float('nan'), float('inf'), float('-inf') to None or safe finite values."""
    import math
    if isinstance(data, float):
        if not math.isfinite(data):
            return None
        return data
    elif isinstance(data, dict):
        return {k: _sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [_sanitize_for_json(v) for v in data]
    return data
from .world_model.text_world_model import GenZeroTextWorldModel
from .gate.action_constraints import FV_NOT_REQUESTED
from .planner.continuous_gflownet import (
    ContinuousGFlowNetAdapter,
    ContinuousManifoldGFlowNetSampler,
)
from .planner.engines import (
    AStarEngine,
    MctsEngine,
    MpcCemEngine,
    ManifoldGFlowNetEngine,
    CfrNashEngine,
    CpSatFormalEngine,
)


class UniversalParadigmRouter:
    """Intelligent Dynamic Strategy Router across 9 Orthogonal Planning Paradigms.

    ``task_hint`` is a caller-declared routing request, not a free-text domain
    description: it must be ``None`` or the exact name of one of
    ``self.all_paradigms``. Earlier revisions matched English/Chinese substrings
    inside ``task_hint`` (and dict-key/type shape of ``state``) against ad-hoc
    keyword lists ("board", "poker", "trap", ...) and called the coincidental
    hit "understanding" the task. A maze state literally containing a waypoint
    named "board_7", or a caller who happened to write "already authorized" in
    a hint, would silently be routed as an adversarial board game or a CP-SAT
    security case. That is pattern matching over arbitrary content dressed up
    as reasoning. Routing now uses only: (1) an exact declared paradigm name in
    ``task_hint``, (2) structural engine preconditions (e.g. A* requires a
    dict state with both "position" and "goal"), and (3) a fixed default
    prior order applied when nothing else applies.
    """

    # Fixed default prior ordering used only when task_hint is None and no
    # structural precondition/exploration/reflex boost fires. Strictly
    # decreasing small base logits so ties in downstream selection never
    # depend on dict/insertion order.
    DEFAULT_PARADIGM_PRIOR: Tuple[str, ...] = (
        "world_model", "mcts", "mpc_cem", "gflownet", "astar",
        "cfr", "bidirectional", "cp_sat", "reflex",
    )

    def __init__(self, latency_budget_ms: float = 20.0):
        self.latency_budget_ms = latency_budget_ms
        self.all_paradigms = [
            "reflex", "mcts", "astar", "bidirectional", "world_model",
            "mpc_cem", "gflownet", "cfr", "cp_sat"
        ]

    def _validate_task_hint(self, task_hint: Optional[str]) -> None:
        if task_hint is not None and task_hint not in self.all_paradigms:
            raise ValueError(
                f"task_hint must be None or an exact paradigm name in {self.all_paradigms}, "
                f"got {task_hint!r}"
            )

    def compute_complexity(
        self,
        state: Any,
        candidates: List[str],
        task_hint: Optional[str] = None,
        policy_entropy: Optional[float] = None
    ) -> float:
        self._validate_task_hint(task_hint)

        score = 0.20
        if len(candidates) <= 2:
            score -= 0.10
        elif len(candidates) >= 5:
            score += 0.15

        if policy_entropy is not None:
            score += (policy_entropy - 0.40) * 0.40

        return max(0.05, min(0.95, score))

    def route_dynamic(
        self,
        state: Any,
        candidates: List[str],
        policy_entropy: Optional[float] = None,
        task_hint: Optional[str] = None,
        prefer_exploration: bool = False
    ) -> Dict[str, Any]:
        self._validate_task_hint(task_hint)

        complexity = self.compute_complexity(
            state=state,
            candidates=candidates,
            task_hint=task_hint,
            policy_entropy=policy_entropy
        )
        if complexity < 0.35:
            k = 1
        elif complexity < 0.70:
            k = 2
        else:
            k = 3

        # Strictly decreasing base logits from the documented default prior:
        # ties in `sorted(..., reverse=True)` never depend on dict/insertion
        # order.
        n = len(self.DEFAULT_PARADIGM_PRIOR)
        logits: Dict[str, float] = {
            name: 1.0 - i * (0.5 / n) for i, name in enumerate(self.DEFAULT_PARADIGM_PRIOR)
        }
        for name in self.all_paradigms:
            logits.setdefault(name, 0.0)

        routing_basis = "default_prior_no_hint"

        if task_hint is not None:
            logits[task_hint] += 6.0
            routing_basis = "declared_hint"

        if prefer_exploration:
            logits["gflownet"] += 6.0
            if routing_basis != "declared_hint":
                routing_basis = "structural_precondition"

        # A* needs an explicit goal to search toward: this is an engine
        # precondition (A* cannot run without one), not a free-text guess.
        if isinstance(state, dict) and "position" in state and "goal" in state:
            logits["astar"] += 5.5
            if routing_basis != "declared_hint":
                routing_basis = "structural_precondition"

        if complexity < 0.35 and len(candidates) <= 2:
            logits["reflex"] += 3.5
            if routing_basis != "declared_hint":
                routing_basis = "structural_precondition"

        sorted_experts = sorted(logits.items(), key=lambda x: x[1], reverse=True)
        top_k = sorted_experts[:k]

        tau = 1.8
        max_score = top_k[0][1]
        exp_vals = [math.exp((s - max_score) / tau) for _, s in top_k]
        sum_exp = sum(exp_vals)
        selected_experts = [(name, round(ev / sum_exp, 4)) for (name, _), ev in zip(top_k, exp_vals)]

        has_cpsat = any(name == "cp_sat" for name, _ in selected_experts)
        return {
            "k": k,
            "complexity_score": round(complexity, 3),
            "selected_experts": selected_experts,
            "all_logits": logits,
            "pipeline_has_cpsat": has_cpsat,
            "routing_basis": routing_basis,
        }

    def route_expert(
        self,
        state: Any,
        candidates: List[str],
        task_hint: Optional[str] = None,
        prefer_exploration: bool = False
    ) -> str:
        dyn = self.route_dynamic(
            state=state,
            candidates=candidates,
            task_hint=task_hint,
            prefer_exploration=prefer_exploration
        )
        return dyn["selected_experts"][0][0]


DecisionMoERouter = UniversalParadigmRouter
from .model.prm import ProcessRewardModel
from .causal.counterfactual_engine import CounterfactualEngine
from .causal.nanocore_bridge import NanocoreAnchorBridge
from .rollout.hard_miner import HardSampleMiner
from .gate.alignment_gate import (
    StateAlignmentGate,
    AlignmentVerdict,
    AlignmentLevel,
    AlignmentAction,
    extract_entity_fingerprint,
)
from .service.co_riding_adapter import (
    CoRidingProbesAdapter,
    CoRidingAlignmentRequest,
    CoRidingAlignmentResponse,
)
from .guard import (
    DualGateGuardrail,
    GuardVerdict,
    GuardAction,
    AgentToolInterlock,
    ToolInterlockVerdict,
)
from .retrieval import ZeroReranker, DecoupledSemanticFind
from .verifier import PassageMultiAspectBattery, TwoTierCitationVerifier
from .dispatch import TypedDispatcher, ProgressiveSkillRouter
from .pipeline import ZeroGenStructureRecoveryEngine
from .gate.fail_closed_scheduler import (
    DualTrackScheduler,
    SystemStatus,
    ModelDecision,
    DefenseAction,
    DualTrackVerdict,
)
from .gateway.in_process_prefill import (
    InProcessPrefillEngine,
    InProcessPrefillResult,
    STANDARD_PROBES,
)
from .calibration.cost_sensitive_roc import (
    CostMatrix,
    CostSensitiveROCOptimizer,
    ROCOptimizationResult,
)
from .dataset.contrastive_perturbation import (
    SemanticPerturbationGenerator,
    ContrastiveSamplePair,
    PerturbationType,
    evaluate_boundary_discrimination,
)
from .router import (
    ZeroRouterMiddleware,
    AdaptiveRouteDecision,
    ModelTier,
    ThinkingEffort,
)
from .compactor import (
    VerbatimContextCompactor,
    CompactorAction,
    CompactionSummary,
    MessageCompactionItem,
)
from .firewall import (
    ParallelNoulFirewall,
    CostOfFailureLevel,
    SecurityRiskVector,
    OneShotPRReviewer,
    PRReviewReport,
)
from .streaming import (
    StreamUtterance,
    StreamActionType,
    StreamIntentDecision,
    StreamIntentGate,
    PendingDraftItem,
    PendingDraftQueue,
    SandwichStreamingPipeline,
)
from .graph import (
    GraphAST,
    GraphMutation,
    GraphOp,
    GraphNode,
    GraphEdge,
)
from .harness.web import (
    ZeroBrowserHarness,
    DOMElement,
    ModalOverlay,
    BoundingBox,
    WebObservation,
    WebOperationType,
    BrowserStepResult,
)
from .protocol.canvas_protocol import (
    CanvasSlotType,
    CanvasSlotSpec,
    SlotResult,
    CanvasDecisionResult,
    CanvasTemplate,
)
from .runtime.h1_entropy_gate import (
    H1EntropyGate,
    EntropyEvaluation,
    MultiReadStatistics,
)
from .nanocore.bidirectional_slot_attention import (
    create_hybrid_slot_mask,
    BidirectionalSlotAttention,
    BidirectionalNanoCore,
)
from .nanocore.choice_head import ActionETFChoiceHead
from .sandbox.tool_registry import ToolRegistry, ToolDefinition, SideEffectLevel
from .sandbox.causal_sandbox import CausalToolSandbox, ExecutionResult, ErrorCategory
from .sandbox.safety_barrier import PRMSafetyBarrier, SafetyVerdict
from .sandbox.self_healing_planner import SelfHealingToolchainPlanner, WorkflowStep, WorkflowExecutionReport
from .multiagent.decentralized_scm import (
    IntentType,
    MultiAgentState,
    AgentIntent,
    DecentralizedSCM,
    AsymmetricBluffDetector,
)
from .multiagent.causal_cfr_engine import (
    CausalCFREngine,
    CausalCFROutcome,
)
from .multiagent.league_arena import (
    LeagueRole,
    AgentSnapshot,
    MatchResult,
    LeagueArena,
)
from .gateway.modality_router import (
    AdaptiveModalityRouter,
    ModalityType,
    IngestedState,
)
from .manifold.adaptive_gating import InstanceAdaptiveRouter
from .gateway.arbiter_bridge import (
    CloudGPUArbiterBridge,
    ArbiterVerdict,
)
from .gateway.llamacpp_adapter import (
    LlamaCppScoreAdapter,
    compute_closed_form_confidence,
)
from .vision.engine import (
    PyTorchVisualDecisionEngine,
    MultimodalVisionEngine,
    SharedVisionPrefixCache,
    PerceptionChannel,
    AdaptivePerceptionRouter,
)
from .vision.discrete_bins_scorer import (
    DiscreteBinsExpectationScorer,
    DiscreteBinsVerdict,
)
from .model.token_stability import (
    assert_single_token_stability,
    TokenFragmentationError,
    CanonicalLabelMapper,
)
from .world_model.streaming_engine import StreamingWorldModelEngine
from .runtime import QuantizedCandidateScorer

# Expert statuses meaning "this expert found no admissible action": any one forces ABSTAIN.
EXPERT_ABSTAIN_STATUSES = frozenset({
    "NO_VALID_CANDIDATES", "NO_CANDIDATES", "NO_LEGAL_ACTIONS", "ROOT_DEAD_END", "ALL_PRUNED", "INCONCLUSIVE",
    # X-M02: a transition/step (or reward_fn) returned a non-finite reward. The empty
    # valid_set already forces has_active_valid_candidates=False downstream, but this
    # entry additionally surfaces an honest top-level status instead of a silent abstain.
    "NON_FINITE_TRANSITION_REWARD",
    # T3-M01: every individual step reward was finite, but the accumulated return
    # (value_sum / Q / expected_value) overflowed to +/-inf across backpropagation.
    # Must fail closed exactly like NON_FINITE_TRANSITION_REWARD, never surface as OK.
    "NON_FINITE_RETURN",
    # Reviewer fix: mpc_cem_engine's discrete plan reports the same non-finite-reward
    # abstain under this name; without it here, a CEM NaN/Inf reward would abstain
    # locally but never force the global fail-closed reject at the fusion step below.
    "NON_FINITE_REWARD",
    # B15: the cp_sat expert found zero candidates passing hard-rule verification;
    # there is no verified-safe fallback action to fall back on.
    "NO_FEASIBLE_CANDIDATES",
})


UNTRAINED_WEIGHTS_SCORER = "untrained_weights_fallback"

# Scorer labels for experts that never read the dual-head weights. A result whose
# scorer is one of these is not affected by a missing checkpoint, so _mark_untrained
# must not relabel it.
NON_NEURAL_SCORERS = frozenset({
    "heuristic_grid_rollout",
    "symbolic_graph_search",
    "symbolic_one_step_lookahead",
    "python_predicate_filter",
})


class _VisionEngineExtractor:
    """Router-facing vision extractor; ``is_loaded`` mirrors the engine's real weight state."""

    def __init__(self, engine: Any):
        self._engine = engine

    @property
    def is_loaded(self) -> bool:
        flag = getattr(self._engine, "is_loaded", False)
        return bool(flag() if callable(flag) else flag)

    def __call__(self, img: Any, prompt: str = "") -> Any:
        return self._engine.prefill_visual_context(img, prompt).get("last_hidden_state")


class GenZero:
    """Universal Decision Engine Client Interface."""

    def __init__(self, config: Optional[GenZeroConfig] = None,
                 hamiltonian_model: Optional[HamiltonianWorldModel] = None,
                 action_effects: Optional[Dict[str, str]] = None):
        self.config = config or GenZeroConfig()
        if hamiltonian_model is not None and not isinstance(hamiltonian_model, HamiltonianWorldModel):
            raise TypeError("hamiltonian_model must be a HamiltonianWorldModel")
        if hamiltonian_model is None and self.config.use_hamiltonian_dynamics:
            hamiltonian_model = HamiltonianWorldModel(latent_dim=self.config.embed_dim)
        if hamiltonian_model is None:
            logging.getLogger(__name__).info("Hamiltonian dynamics world model is not mounted on MCTS")
        else:
            logging.getLogger(__name__).info("Hamiltonian dynamics world model mounted on MCTS; parameters are not trained by GenZero")
        self.hamiltonian_model = hamiltonian_model

        self.neural_dynamics_model: Optional[NeuralDynamicsWorldModel] = None
        if self.config.neural_dynamics_checkpoint is not None:
            self.neural_dynamics_model = NeuralDynamicsWorldModel.from_checkpoint(
                self.config.neural_dynamics_checkpoint
            )
            logging.getLogger(__name__).info(
                "Neural dynamics world model mounted on MCTS from %s", self.config.neural_dynamics_checkpoint
            )
        else:
            logging.getLogger(__name__).warning(
                "Neural dynamics world model is NOT mounted (config.neural_dynamics_checkpoint is None); "
                "vector-state transitions will raise ValueError and MCTS/MPC-CEM planners run without it"
            )
        
        # Layer 1: Model
        self.model = GenZeroDualHeadModel(
            hidden_dim=self.config.hidden_dim,
            embed_dim=self.config.embed_dim,
            num_layers=self.config.num_attention_layers,
            num_heads=self.config.num_attention_heads,
            use_value_head=self.config.use_value_head,
            enable_abstain=self.config.enable_abstain
        )
        # Fail-closed provenance flag: True only after a dual-head state_dict was loaded
        # strictly from disk. A freshly constructed model is random-init and must never be
        # served over the wire as if it were inference (see service/app.py gate).
        self.weights_loaded_from_checkpoint: bool = False
        # Same idea for the INT8 CPU scorer: True only once it holds weights exported from a loaded checkpoint.
        self.scorer_synced_from_checkpoint: bool = False
        self.dual_head_checkpoint_path: Optional[str] = None
        # Fail-closed: no manifold anchor artifact is loaded until the caller explicitly
        # loads one via load_manifold_anchor_artifact(...).
        self.nanocore_anchor_bridge: Optional[NanocoreAnchorBridge] = None
        if self.config.dual_head_checkpoint:
            self.load_dual_head_checkpoint(self.config.dual_head_checkpoint)
        else:
            logging.getLogger(__name__).warning(
                "GenZero dual-head model is RANDOM-INIT (no GENZERO_DUAL_HEAD_CHECKPOINT); "
                "service endpoints will fail closed with 503"
            )

        # Layer 2: 6 Converged Orthogonal Planning Engines (RFC-093)
        self.astar_engine = AStarEngine(lambda_weight=self.config.astar_lambda)
        self.mcts_engine = MctsEngine(
            c_puct=self.config.mcts_cpuct,
            num_simulations=self.config.mcts_simulations,
            max_depth=self.config.mcts_depth,
            hamiltonian_model=self.hamiltonian_model,
            dynamics_model=None if self.hamiltonian_model is not None else self.neural_dynamics_model,
        )
        self.mpc_cem_engine = MpcCemEngine(
            action_dim=4, horizon=5, num_samples=32, latent_model=self.neural_dynamics_model
        )
        self.manifold_gflownet_engine = ManifoldGFlowNetEngine(temperature=1.0)
        self.cfr_nash_engine = CfrNashEngine(iterations=100)
        self.cpsat_formal_engine = CpSatFormalEngine(strict_mode=True, action_effects=action_effects)

        # Unified engine wiring and direct properties
        self.astar_planner = self.astar_engine
        self.mcts_planner = self.mcts_engine
        # GenZeroTextWorldModel returns hand-set, uncalibrated heuristic constants (see its
        # docstring). acknowledge_uncalibrated=True confirms this client never presents
        # predict_consequences() output as calibrated confidence; see audit_action's
        # is_uncalibrated_text handling and PROVENANCE_TEXT tagging below for how it is used.
        self.world_model = GenZeroTextWorldModel(acknowledge_uncalibrated=True)
        self.mpc_cem_planner = self.mpc_cem_engine
        self.gflownet_sampler = self.manifold_gflownet_engine.adapter
        self.continuous_gflownet_sampler = self.manifold_gflownet_engine.adapter
        self.bidirectional_planner = self.astar_engine
        self.cfr_expert = self.cfr_nash_engine
        self.cp_sat_solver = self.cpsat_formal_engine
        self.latent_world_model = None
        self.continuous_mpc_planner = self.mpc_cem_engine
        self.prm_verifier = ProcessRewardModel()
        self.moe_router = UniversalParadigmRouter()
        self.causal_engine = CounterfactualEngine()

        # Layer 3: hard-sample miner (feeds the arbiter bridge's escalation telemetry)
        self.miner = HardSampleMiner(
            entropy_threshold=self.config.exploration_entropy_threshold,
            td_error_threshold=self.config.td_error_threshold,
            history_window=self.config.hard_sample_history_steps
        )

        # Layer 5: Gates
        self.alignment_gate = StateAlignmentGate()
        self.guardrail = DualGateGuardrail()
        self.tool_interlock = AgentToolInterlock(guardrail=self.guardrail)
        self.reranker = ZeroReranker()
        self.semantic_find = DecoupledSemanticFind()
        self.evidence_battery = PassageMultiAspectBattery()
        self.citation_verifier = TwoTierCitationVerifier()
        self.typed_dispatcher = TypedDispatcher()
        self.skill_router = ProgressiveSkillRouter()
        self.structure_recovery = ZeroGenStructureRecoveryEngine()
        self.fail_closed_scheduler = DualTrackScheduler()
        self.in_process_prefill = InProcessPrefillEngine()
        self.roc_optimizer = CostSensitiveROCOptimizer()
        self.perturbation_generator = SemanticPerturbationGenerator()
        self.mcp_router = ZeroRouterMiddleware()
        self.verbatim_compactor = VerbatimContextCompactor()
        self.parallel_firewall = ParallelNoulFirewall()
        self.pr_reviewer = OneShotPRReviewer()
        self.streaming_pipeline = SandwichStreamingPipeline()
        self.graph_ast = self.streaming_pipeline.graph_ast
        self.browser_harness = ZeroBrowserHarness(decision_client=self)
        self.h1_entropy_gate = H1EntropyGate(tau_entropy=0.10, multi_read_k=4)
        self.bidirectional_nanocore = BidirectionalNanoCore(embed_dim=128)
        self.nanocore_choice_head = ActionETFChoiceHead(hidden_dim=128, action_dim=128)
        self.registered_nanocores: Dict[int, Any] = {0: self.nanocore_choice_head}
        # domain_id -> bound space manifest (dict), set only via register_nanocore(...,
        # space_manifest=...). Deliberately NOT an attribute on the core objects
        # themselves: self.nanocore_choice_head (and any other core) can be shared
        # across multiple domains, each potentially bound to a different manifest.
        self.nanocore_space_manifests: Dict[int, Dict[str, Any]] = {}

        # Layer 6: Enterprise Tool Sandbox & Multi-Step Orchestration
        self.tool_registry = ToolRegistry()
        self.sandbox = CausalToolSandbox(registry=self.tool_registry)
        self.safety_barrier = PRMSafetyBarrier(strict_mode=True)
        self.workflow_planner = SelfHealingToolchainPlanner(
            sandbox=self.sandbox,
            safety_barrier=self.safety_barrier
        )

        # Layer 7: Multi-Agent Causal Game Theory & Adversarial Engine (Phase 4)
        self.decentralized_scm = DecentralizedSCM()
        self.bluff_detector = AsymmetricBluffDetector(
            d_scm=self.decentralized_scm,
            belief_tracker=getattr(self.cfr_expert, "belief_tracker", None)
        )
        self.causal_cfr_engine = CausalCFREngine(
            d_scm=self.decentralized_scm,
            tracker=getattr(self.cfr_expert, "belief_tracker", None)
        )
        self.league_arena = LeagueArena(causal_cfr=self.causal_cfr_engine)
        
        # Layer 0: Adaptive Input Modality Gateway & PyTorch Vision Engine
        self.vision_engine = PyTorchVisualDecisionEngine(
            model_name_or_path="Qwen/Qwen3.5-9B",
            feature_dim=self.config.hidden_dim
        )
        self.multimodal_vision = MultimodalVisionEngine(embed_dim=self.config.hidden_dim)
        self.perception_router = AdaptivePerceptionRouter(self.multimodal_vision)
        self.discrete_bins_scorer = DiscreteBinsExpectationScorer()
        self.modality_router = AdaptiveModalityRouter(
            llm_extractor=getattr(self.model, "extract_hidden_state", None),
            vision_extractor=_VisionEngineExtractor(self.vision_engine),
            feature_dim=self.config.hidden_dim
        )
        self.arbiter_bridge = CloudGPUArbiterBridge(
            confidence_threshold=self.config.arbiter_confidence_threshold,
            entropy_threshold=self.config.arbiter_entropy_threshold,
            remote_endpoint=self.config.arbiter_endpoint,
            local_gpu_engine=self.vision_engine,
            miner=self.miner,
            timeout_s=self.config.arbiter_timeout_s
        )
        if not getattr(self.vision_engine, "is_real", False):
            logging.getLogger(__name__).warning(
                "arbiter_bridge.local_gpu_engine (vision_engine) has no real weights loaded yet; "
                "arbitration will skip it and fall back to remote_endpoint=%s, then fail-closed",
                self.config.arbiter_endpoint,
            )
        self.llamacpp_adapter = LlamaCppScoreAdapter(
            timeout=2.0,
            model_name="typesafe/zero-1.13"
        )

        # Instance-Adaptive Gating: per-sample expert reliability weighting +
        # log-linear opinion pool fusion for the Stage-2 planner consensus.
        # Fails closed at construction: enabling it without a fitted artifact
        # is a config error, not a silent fallback to the static consensus.
        self.adaptive_gating_router: Optional[InstanceAdaptiveRouter] = None
        if self.config.enable_adaptive_gating:
            if not self.config.adaptive_gating_artifact:
                raise ValueError(
                    "enable_adaptive_gating=True requires config.adaptive_gating_artifact "
                    "(a path saved by InstanceAdaptiveRouter.save()); refusing to start "
                    "with an unfitted or default router."
                )
            self.adaptive_gating_router = InstanceAdaptiveRouter.load(self.config.adaptive_gating_artifact)
            # Every possible Stage-2 planner name must have a fitted risk model,
            # not just whichever subset happened to be sampled at fit time --
            # otherwise a later decide() call could pick an unfitted expert and
            # would have to choose between crashing mid-decision or silently
            # degrading to the static consensus. Neither is acceptable, so this
            # is checked once, loudly, at construction instead.
            planner_pool = set(self.moe_router.all_paradigms) - {"cp_sat"}
            missing = planner_pool - set(self.adaptive_gating_router.expert_names)
            if missing:
                raise ValueError(
                    f"adaptive_gating_artifact={self.config.adaptive_gating_artifact!r} has no "
                    f"fitted risk model for planner expert(s) {sorted(missing)}; refit "
                    f"InstanceAdaptiveRouter.fit(..., expert_names=...) covering all of "
                    f"{sorted(planner_pool)} before enabling adaptive gating."
                )

        # Direction 1: Streaming Spatial-Temporal World Model & Master Orchestrator (RFC-049 / Issue #51)
        self.streaming_world_model = StreamingWorldModelEngine(
            latent_dim=self.config.hidden_dim,
            action_dim=32,
            num_sink_tokens=4,
            window_size=8,
            max_simulations=64
        )
        from .nanocore.world_model_orchestrator import WorldModelNanoCoreOrchestrator
        self.world_model_orchestrator = WorldModelNanoCoreOrchestrator(
            latent_dim=self.config.hidden_dim,
            action_dim=32,
            num_sink_tokens=4,
            window_size=8
        )

        # Direction 2: Pure CPU INT8 Extreme Scorer (< 3.5ms, zero PyTorch dependency)
        self.cpu_extreme_scorer = QuantizedCandidateScorer(
            state_dim=self.config.hidden_dim,
            candidate_dim=self.config.hidden_dim,
            embed_dim=128
        )
        self._sync_scorer_after_checkpoint() if self.weights_loaded_from_checkpoint else self.sync_model_to_scorer()

        # Atomic serving container: each request pins one model/scorer snapshot.
        from .daemon.atomic_container import AtomicModelContainer
        self.container = AtomicModelContainer(
            initial_model=self.model,
            initial_scorer=getattr(self, "cpu_extreme_scorer", None)
        )

        # Issue #8: Composite Decision, Policy Gate & Candidate Governance
        from .runtime.composite_decision import CompositeDecisionEngine
        from .gate.policy_gate import DecisionPolicyGate
        from .runtime.candidate_governance import AntiTruncationRanker
        from .runtime.loop_state_machine import StagnationCircuitBreaker
        self.composite_engine = CompositeDecisionEngine(client=self)
        self.policy_gate = DecisionPolicyGate()
        self.candidate_ranker = AntiTruncationRanker(max_candidates=15)
        self.stagnation_breaker = StagnationCircuitBreaker()

    def load_dual_head_checkpoint(self, path: str) -> None:
        """Loads a dual-head state_dict strictly; raises on any mismatch (fail-closed).

        Only a successful strict load flips ``weights_loaded_from_checkpoint``.
        """
        import torch

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Dual-head checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError(f"Dual-head checkpoint {path} does not contain a state_dict")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.weights_loaded_from_checkpoint = True
        self.dual_head_checkpoint_path = path
        logging.getLogger(__name__).info("Dual-head weights loaded from checkpoint %s", path)
        # During __init__ the scorer does not exist yet; the constructor syncs it right after creation.
        if getattr(self, "cpu_extreme_scorer", None) is not None:
            self._sync_scorer_after_checkpoint()

    def _sync_scorer_after_checkpoint(self) -> None:
        self.scorer_synced_from_checkpoint = self.sync_model_to_scorer()
        if not self.scorer_synced_from_checkpoint:
            logging.getLogger(__name__).warning(
                "Checkpoint loaded but INT8 CPU scorer sync failed; CPU scorer outputs stay marked %s",
                UNTRAINED_WEIGHTS_SCORER,
            )

    def _mark_untrained(self, res: Dict[str, Any], entrypoint: str, trained: Optional[bool] = None) -> Dict[str, Any]:
        """Fail-loud tag for outputs computed from random-init weights (no checkpoint loaded).

        Keeps any earlier degraded_reason (e.g. a torch exception) and appends ours.
        """
        if self.weights_loaded_from_checkpoint if trained is None else trained:
            return res
        if res.get("scorer") in NON_NEURAL_SCORERS:
            return res
        logging.getLogger(__name__).warning(
            "%s: no dual-head checkpoint loaded; result comes from untrained random-init weights "
            "(degraded=True, scorer=%s)", entrypoint, UNTRAINED_WEIGHTS_SCORER,
        )
        reasons = [r for r in str(res.get("degraded_reason") or "").split(";") if r]
        if UNTRAINED_WEIGHTS_SCORER not in reasons:
            reasons.append(UNTRAINED_WEIGHTS_SCORER)
        res["degraded"] = True
        res["degraded_reason"] = ";".join(reasons)
        res["scorer"] = UNTRAINED_WEIGHTS_SCORER
        res["weights_loaded_from_checkpoint"] = False
        return res

    def sync_model_to_scorer(self, model: Any = None, target_scorer: Any = None) -> bool:
        """Synchronizes learned neural weights from model into scorer without mutating uncommitted active scorers."""
        target_model = model if model is not None else getattr(self, "model", None)
        scorer = target_scorer if target_scorer is not None else getattr(self, "cpu_extreme_scorer", None)
        if hasattr(target_model, "export_to_scorer_weights"):
            try:
                w_dict = target_model.export_to_scorer_weights()
                if w_dict and scorer is not None:
                    scorer.load_from_weight_dict(w_dict)
                    return True
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Model-to-scorer synchronization failed (%s: %s)", type(exc).__name__, exc
                )
        return False

    def prepare_inference_state(self, raw_state: Any) -> Any:
        """Unified modality ingestion gateway ensuring identical state representation
        across evaluation benchmarks and live inference (Probes R8_E02, R8_E03).
        """
        if hasattr(self, "modality_router") and self.modality_router is not None:
            return self.modality_router.ingest(raw_input=raw_state).normalized_state
        from gen_zero.model.dual_head import normalize_state_repr
        return normalize_state_repr(raw_state)

    def _execute_expert_distribution(
        self,
        expert_name: str,
        state: Any,
        candidates: List[str],
        trans_fn: Callable,
        candidate_descriptions: Optional[Dict[str, str]] = None,
        active_snapshot: Optional[Any] = None,
        legal_actions_fn: Optional[Callable[[Any], Sequence[str]]] = None,
        observed_opponent_action: Optional[str] = None,
        complexity_score: Optional[float] = None,
        task_hint: Optional[str] = None,
    ) -> Tuple[Dict[str, float], float, str, Dict[str, Any]]:
        """Executes a single expert and returns (probs, expected_value, best_action, expert_meta).

        legal_actions_fn(state) gives the actions legal in a given state; MCTS applies it
        (plus the hard rules) at every tree node.

        observed_opponent_action/complexity_score/task_hint are per-call request
        context, passed explicitly rather than stashed on self: this method (and
        decide()) can run concurrently against a shared GenZero instance from
        multiple threads/coroutines, and instance attributes would let one
        in-flight request's context bleed into another's expert dispatch.
        """
        import math
        meta = {}
        model = active_snapshot.model if active_snapshot is not None else getattr(self, "model", None)
        scorer = active_snapshot.scorer if active_snapshot is not None else getattr(self, "cpu_extreme_scorer", None)

        # Without a checkpoint the torch model holds random-init weights.
        live_scorer_label = "torch_live_model" if self.weights_loaded_from_checkpoint else UNTRAINED_WEIGHTS_SCORER

        if expert_name in ("reflex", "fast"):
            # Only the reflex expert itself takes this path. A single candidate (given, or
            # left after legality pruning) must never hand an explicitly named expert to
            # reflex: "one candidate" is not "safe by construction". world_model must still
            # validate the state schema and fail closed (UnsupportedStateError), and MCTS
            # must still run its per-node safety checks (T3-M01). Every named expert runs
            # its own branch below, whatever the candidate count.
            # Direct live PyTorch model scoring path if dual-head model is active
            if model is not None and hasattr(model, "scalar") and hasattr(model, "forward"):
                try:
                    import torch
                    if hasattr(model, "eval"):
                        model.eval()
                    c_ids = [
                        encode_leaf_tokens(state, c, candidate_descriptions.get(c, "") if candidate_descriptions else None)
                        for c in candidates
                    ]
                    ex = [{"leaf_tokens": c_ids, "candidate_ids": candidates, "type": "choice"}]
                    with torch.no_grad():
                        logits, valid, vals = model(ex, pad_token=0, return_value=True)
                        if not torch.isnan(logits).any() and not torch.isinf(logits).any():
                            raw_logits = logits[0, :len(candidates)].clone()
                            # Apply candidate valid mask before softmax (Probe E08, R10_E01)
                            if hasattr(valid, "dtype") and valid.dtype == torch.bool:
                                cand_valid = valid[0, :len(candidates)]
                                valid_list = [bool(cand_valid[i].item()) for i in range(len(candidates))]
                                meta["valid_set"] = [c for i, c in enumerate(candidates) if valid_list[i]]
                                # All-invalid candidates boundary protection (Probe R8_E01)
                                if not cand_valid.any():
                                    probs = {c: 0.0 for c in candidates}
                                    meta["scorer"] = live_scorer_label
                                    meta["status"] = "NO_VALID_CANDIDATES"
                                    return probs, 0.0, "ABSTAIN", meta
                                raw_logits[~cand_valid] = float("-inf")

                            p_t = torch.softmax(raw_logits, dim=-1)
                            probs = {c: float(p_t[i].item()) for i, c in enumerate(candidates)}
                            val = float(vals[0].item()) if (vals is not None and not torch.isnan(vals).any() and not torch.isinf(vals).any()) else 0.0
                            best = candidates[int(torch.argmax(p_t).item())]
                            meta["scorer"] = live_scorer_label
                            return probs, val, best, meta
                        raise ValueError("Non-finite reflex logits")
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "Torch reflex scoring failed (%s: %s); degrading to CPU quantized scorer",
                        type(exc).__name__, exc,
                    )
                    meta["degraded"] = True
                    meta["degraded_reason"] = f"torch_reflex_exception:{type(exc).__name__}"

            if len(candidates) == 1:
                return {candidates[0]: 1.0}, 0.0, candidates[0], meta

            # Connect real candidate scorer for reflex path (System 1)
            if scorer is not None:
                score_res = scorer.score_candidates(
                    state_repr=state,
                    candidates=candidates,
                    candidate_descriptions=candidate_descriptions
                )
                probs = score_res["probs"]
                val = score_res["value"]
                best = score_res["best_action"]
                meta["scorer_latency_ms"] = score_res.get("latency_ms", 0.0)
                if self.scorer_synced_from_checkpoint:
                    meta["scorer"] = "cpu_quantized_scorer"
                else:
                    meta["scorer"] = UNTRAINED_WEIGHTS_SCORER
                    meta["degraded"] = True
                    prev = meta.get("degraded_reason")
                    meta["degraded_reason"] = f"{prev};{UNTRAINED_WEIGHTS_SCORER}" if prev else UNTRAINED_WEIGHTS_SCORER
                return probs, val, best, meta


        if expert_name == "cp_sat":
            sat_res = self.cp_sat_solver.verify_and_prune(state, candidates)
            feasible = sat_res["feasible_actions"]
            probs = {c: (1.0 / len(feasible) if c in feasible else 0.0) for c in candidates}
            if feasible:
                best = feasible[0]
                val = 1.0 if sat_res["all_feasible"] else 0.5
            else:
                # No candidate survives hard-rule verification: candidates[0] is NOT
                # a verified-safe fallback, it is an arbitrary pick from a rejected
                # set. Abstain explicitly instead of silently handing back an
                # unverified action dressed up as this expert's choice.
                best = "ABSTAIN"
                val = -10.0
                meta["status"] = "NO_FEASIBLE_CANDIDATES"
            meta["sat_feasible_count"] = len(feasible)
            # verify_and_prune evaluates the hard rules as plain Python predicates; it never
            # invokes OR-Tools, so this is not a CP-SAT solve and must not be labelled one.
            meta["scorer"] = "python_predicate_filter"
            meta["confidence_kind"] = "rule_predicate_filtering"
            return probs, val, best, meta

        if expert_name == "cfr":
            info_set = str(state)
            opp_act = observed_opponent_action
            cfr_res = self.cfr_expert.solve_imperfect_decision(
                info_set_repr=info_set,
                candidates=candidates,
                observed_opponent_action=opp_act
            )
            probs = {c: cfr_res["strategy"].get(c, 0.0) for c in candidates}
            best = cfr_res["best_action"]
            val = 1.0 - cfr_res["exploitability_bound"]
            meta["adaptive_iterations"] = cfr_res.get("adaptive_iterations")
            meta["exploitation_beta"] = cfr_res.get("exploitation_beta")
            meta["opponent_bias_kl"] = cfr_res.get("opponent_bias_kl")
            meta["is_exploiting"] = cfr_res.get("is_exploiting")
            return probs, val, best, meta

        if expert_name == "mpc_cem":
            cem_res = self.mpc_cem_planner.plan_discrete(
                initial_state=state,
                candidate_actions=candidates,
                transition_fn=trans_fn
            )
            best = cem_res["best_action"]
            # Real CEM elite-fitted first-step categorical distribution, not an
            # assumed constant confidence.
            probs = {c: float(cem_res["action_probs"].get(c, 0.0)) for c in candidates}
            meta["adaptive_horizon"] = cem_res.get("adaptive_horizon")
            meta["adaptive_samples"] = cem_res.get("adaptive_samples")
            cem_status = cem_res.get("status")
            if cem_status is not None and cem_status != "OK":
                # Propagate CEM's abstain status so the fusion step's
                # EXPERT_ABSTAIN_STATUSES check can force a global fail-closed
                # reject instead of letting other experts silently outvote a
                # non-finite-reward abstain.
                meta["status"] = cem_status
            return probs, cem_res["expected_return"], best, meta

        if expert_name in ("gflownet", "continuous_gflownet"):
            gfn_res = self.continuous_gflownet_sampler.sample_trajectory(
                initial_state=state,
                candidate_actions=candidates,
                transition_fn=trans_fn
            )
            probs = {c: gfn_res["action_probs"].get(c, 0.0) for c in candidates}
            best = gfn_res["best_action"]
            meta["adaptive_temperature"] = gfn_res.get("adaptive_temperature")
            meta["diversity_entropy"] = gfn_res.get("diversity_entropy")
            meta["mode_coverage"] = gfn_res.get("mode_coverage")
            meta["modes_discovered"] = gfn_res.get("modes_discovered")
            meta["counterfactual_count"] = gfn_res.get("counterfactual_count")
            meta["avg_step_latency_ms"] = gfn_res.get("avg_step_latency_ms")
            meta["permutation_flip_rate"] = gfn_res.get("permutation_flip_rate")
            return probs, gfn_res["total_flow"], best, meta

        if expert_name == "world_model":
            # Dynamic adaptive horizon determined by the model based on state complexity
            h = self.world_model.determine_adaptive_horizon(
                state=state,
                candidates=candidates,
                complexity_score=complexity_score,
                task_hint=task_hint
            )
            best_action = candidates[0]
            best_val = -1e9
            action_scores = {}
            for a in candidates:
                try:
                    roll_res = self.world_model.adaptive_rollout(
                        state=state,
                        action=a,
                        horizon=h,
                        trans_fn=trans_fn
                    )
                    r_total = roll_res["cumulative_return"]
                    action_scores[a] = r_total
                    if r_total > best_val:
                        best_val = r_total
                        best_action = a
                except UnsupportedStateError:
                    # Fail closed: a uniform distribution with candidates[0] as "best" would be
                    # a fabricated score for a state this expert cannot simulate.
                    raise
            exp_s = {a: math.exp(max(-5.0, min(5.0, score))) for a, score in action_scores.items()}
            sum_exp = sum(exp_s.values()) or 1.0
            probs = {a: exp_s[a] / sum_exp for a in candidates}
            meta["adaptive_horizon"] = h
            meta["scorer"] = "heuristic_grid_rollout"
            meta["confidence_kind"] = "uncalibrated_normalized_rollout_score"
            return probs, best_val, best_action, meta

        if expert_name in ("astar", "bidirectional"):
            goal = None
            if isinstance(state, dict):
                goal = state.get("goal") or state.get("food")

            def is_goal(s):
                if goal is not None and isinstance(s, dict):
                    pos = s.get("pos") or (s.get("body")[0] if s.get("body") else None)
                    return pos == goal
                return False

            def get_neighbors(s):
                neighbors = []
                for a in candidates:
                    try:
                        ns, r, done = trans_fn(s, a)
                        # No real per-edge transition-reliability model is available
                        # at this generic call site. p_success=1.0 makes
                        # uncertainty_penalty(1.0) == 0: an honest "no known
                        # differential risk between edges" rather than a fabricated
                        # 0.95/0.05 split keyed off `done`.
                        neighbors.append((ns, a, 1.0))
                    except Exception as exc:
                        raise RuntimeError("Planner transition failed while expanding neighbors") from exc
                return neighbors

            def h_fn(s, *args):
                if isinstance(s, dict) and goal is not None:
                    pos = s.get("pos") or (s.get("body")[0] if s.get("body") else None)
                    if pos and len(pos) == 2 and len(goal) == 2:
                        return float(abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]))
                return 0.0

            if goal is not None and expert_name == "astar":
                plan_res = self.astar_planner.plan(
                    start_state=state,
                    is_goal_fn=is_goal,
                    get_neighbors_fn=get_neighbors,
                    heuristic_fn=h_fn,
                    max_expansions=300
                )
                if plan_res.get("success") and plan_res.get("path"):
                    best_action = plan_res["path"][0]
                    # A* returns a single deterministic best-first path with no
                    # branching distribution over alternatives: report that honestly
                    # (all mass on the one found action) instead of an assumed
                    # 0.85/0.15 confidence split we have no basis for.
                    probs = {c: (1.0 if c == best_action else 0.0) for c in candidates}
                    meta["nodes_expanded"] = plan_res.get("nodes_expanded", 0)
                    meta["plan_length"] = len(plan_res["path"])
                    # Value grounded in the real accumulated path cost (lower cost
                    # is better), not a fixed "1.0 = fully confident" placeholder.
                    real_cost = float(plan_res.get("cost", 0.0))
                    val = -real_cost if math.isfinite(real_cost) else -1e9
                    meta["scorer"] = "symbolic_graph_search"
                    meta["confidence_kind"] = "heuristic_path_cost"
                    return probs, val, best_action, meta

            # Fallback 1-step lookahead scoring with trans_fn
            action_scores = {}
            for a in candidates:
                try:
                    ns, r, term = trans_fn(state, a)
                    h_val = h_fn(ns) if goal else 0.0
                    if term and r < 0:
                        score = -100.0  # Terminal failure penalty
                    else:
                        score = r - 0.5 * h_val
                    action_scores[a] = score
                except Exception as exc:
                    raise RuntimeError("Planner transition failed during lookahead") from exc

            best_action = max(action_scores, key=lambda a: action_scores[a])
            exp_s = {a: math.exp(max(-5.0, min(5.0, score))) for a, score in action_scores.items()}
            sum_exp = sum(exp_s.values()) or 1.0
            probs = {a: exp_s[a] / sum_exp for a in candidates}
            # Not a graph search: no goal, no A* success, or "bidirectional" (which has
            # no planner call here). Label the one-step lookahead as what it is.
            meta["scorer"] = "symbolic_one_step_lookahead"
            meta["confidence_kind"] = "heuristic_lookahead_score"
            return probs, action_scores[best_action], best_action, meta

        # Default PUCT MCTS with dynamic adaptive depth
        adaptive_depth = self.world_model.determine_adaptive_horizon(
            state=state,
            candidates=candidates,
            complexity_score=complexity_score,
            task_hint=task_hint
        )

        # Per-state action mask: the hard rules are re-run on every tree state, over the
        # caller's legal actions for that state when given, else over the request candidates.
        def state_legal_actions(s):
            universe = list(legal_actions_fn(s)) if legal_actions_fn is not None else list(candidates)
            return self.cp_sat_solver.verify_and_prune(s, universe)["feasible_actions"]

        meta["action_mask_source"] = (
            "caller_legal_actions_fn+hard_rules" if legal_actions_fn is not None else "request_candidates+hard_rules"
        )
        root_valid: Dict[str, List[str]] = {}

        def eval_fn(s, acts):
            e_probs, e_val, _, meta_expert = self._execute_expert_distribution(
                expert_name="reflex",
                state=s,
                candidates=acts,
                trans_fn=trans_fn,
                candidate_descriptions=candidate_descriptions,
                active_snapshot=active_snapshot
            )
            if isinstance(meta_expert, dict):
                # Only the root call speaks for the root candidates; child states have their own action sets.
                if s is state and "valid_set" in meta_expert:
                    root_valid["valid_set"] = list(meta_expert["valid_set"])
                if meta_expert.get("degraded"):
                    meta["degraded"] = True
                    meta["degraded_reason"] = meta_expert.get("degraded_reason", "reflex_eval_degraded")
            return e_probs, e_val

        use_dynamics = self.mcts_engine.dynamics_model is not None and isinstance(state, np.ndarray)
        res = self.mcts_engine.plan(
            root_state=state,
            candidate_actions=candidates,
            transition_fn=trans_fn,
            legal_actions_fn=state_legal_actions,
            eval_fn=None if use_dynamics else eval_fn,
            max_depth=adaptive_depth,
            causal_invariant_fn=lambda s, a, ns, r, done: not self.prm_verifier.verify_step(
                s, a, ns, is_done=done, step_reward=r
            ).get("should_prune", False),
        )
        meta["mcts_status"] = res["status"]
        meta["falsified_pruned_count"] = res["falsified_pruned_count"]
        meta["dead_end_count"] = res["dead_end_count"]
        meta["nodes_expanded"] = res["nodes_expanded"]
        meta["max_depth_reached"] = res["max_depth_reached"]
        meta["transition_source"] = res["transition_source"]
        meta["value_source"] = res["value_source"]
        meta["adaptive_depth"] = adaptive_depth
        if res["status"] != "OK":
            # Fail closed: a pruned or unsearched root must never be refilled with the candidates.
            logging.getLogger(__name__).warning(
                "MCTS returned no action (status=%s, falsified=%d); abstaining",
                res["status"], res["falsified_pruned_count"],
            )
            meta["status"] = res["status"]
            meta["valid_set"] = []
            return {c: 0.0 for c in candidates}, 0.0, "ABSTAIN", meta
        visit_dist = {c: res["visit_distribution"].get(c, 0.0) for c in candidates}
        survivors = [c for c in candidates if c in res["visit_distribution"]]
        if "valid_set" in root_valid:
            survivors = [c for c in survivors if c in root_valid["valid_set"]]
        # Root branches MCTS falsified leave the declared valid set, so no other expert can revive them.
        meta["valid_set"] = survivors
        return visit_dist, res["expected_value"], res["best_action"], meta

    def decide(
        self,
        state: Any,
        candidates: List[str],
        mode: str = "auto",
        task_hint: Optional[str] = None,
        policy_entropy: Optional[float] = None,
        transition_fn: Optional[Callable[[Any, str], Tuple[Any, float, bool]]] = None,
        observed_opponent_action: Optional[str] = None,
        candidate_descriptions: Optional[Dict[str, str]] = None,
        return_trajectory: bool = False,
        constraints: Optional[Sequence[Mapping[str, Any]]] = None,
        legal_actions_fn: Optional[Callable[[Any], Sequence[str]]] = None,
    ) -> Dict[str, Any]:
        """Unified MoE Decision Entrypoint routing across 9 planning paradigms.
        
        Supports Dynamic-K compute allocation & Safe Opponent Exploitation:
        - Automatically judges task complexity C(s) in [0, 1]
        - Automatically decides HOW MANY experts to activate (K in [1, 3])
        - Automatically decides WHICH experts to activate based on model gating
        - Runs CP-SAT as safety pre-filter when constraint risks are detected
        - Combines multiple neural planners via confidence-weighted consensus
        - Executes Safe Counterfactual Regret Exploitation against biased opponents
        
        Args:
            state: Current state (text, dict, grid, or trajectory).
            candidates: List of available action identifiers.
            mode: 'auto' (MoE Dynamic Dispatch) | 'reflex' | 'mcts' | 'astar' | 'mpc_cem' | 'gflownet' | 'cfr'.
            task_hint: Optional exact paradigm name to route to, one of
                UniversalParadigmRouter.all_paradigms (e.g. 'mcts', 'astar',
                'mpc_cem', 'gflownet', 'cfr'), or None for automatic routing.
                Not a free-text domain description; see UniversalParadigmRouter's
                docstring for why keyword/substring hints were removed.
            policy_entropy: Optional normalized policy entropy from dual-head model.
            transition_fn: Transition simulator function. Defaults to built-in world model.
            observed_opponent_action: Optional observed action executed by opponent for online belief updating.
            candidate_descriptions: Optional map of candidate IDs to semantic criteria/descriptions.
            return_trajectory: Also roll the chosen action forward for 'adaptive_horizon' steps
                (greedy continuation over the feasible candidates) and return it as 'trajectory'.
                When no action is chosen, 'trajectory' is None and 'trajectory_status' says why.
            constraints: Optional caller action constraints, e.g.
                [{"type": "mutually_exclusive", "actions": ["reboot", "format"]},
                 {"type": "forbid", "actions": ["wipe"]},
                 {"type": "upper_bound", "action": "reboot", "value": 0.3}].
                Compiled by ConstraintLinearProjectionCompiler and applied LAST, after the arbiter:
                disabled actions get probability exactly 0 and the rest is renormalised
                (see gen_zero/gate/action_constraints.py). Malformed specs raise ValueError.
                Unsatisfiable constraints abstain (status CONSTRAINTS_UNSATISFIABLE).
            legal_actions_fn: Optional state -> legal actions callback. Candidates illegal at the
                root are removed before any expert runs, and MCTS calls it at every tree state.
            
        Returns:
            Dict with 'action', 'confidence', 'probs', 'value', 'k_experts', 'experts_activated',
            'expert_weights', 'complexity_score', 'adaptive_params', 'pruned_by_cpsat', 'expert_outputs', 'latency_ms'
            (k_experts / experts_activated / expert_weights / expert_routed describe the experts that
            actually ran; 'initial_routing_plan' holds the router's first choice when it differs, and
            'routing_reassignment' names why an expert was swapped),
            'cpsat_solver_status' (the hard-rule pre-filter's honest label: ortools_unavailable |
            python_predicate_filter; it is never a CP-SAT proof), 'formal_verification' (not_requested | unavailable_missing_dependency | cp_sat_verified |
            cp_sat_refuted | not_applicable_no_action_selected | cp_sat_error:*) and
            'constraint_projection' (None when no constraints),
            plus 'trajectory' / 'trajectory_status' when return_trajectory=True.
        """
        import time
        t0 = time.time()

        # Fail fast on malformed constraints, before any planning work.
        constraint_compiler = None
        if constraints is not None:
            constraint_compiler = self.create_constraint_compiler()
            constraint_compiler.compile_action_constraints(constraints)

        if not candidates:
            res = {
                "action": None, "confidence": 0.0, "value": 0.0, "mode": mode, "latency_ms": 0.0,
                "formal_verification": self._constraint_formal_status(constraint_compiler, None, [], []),
                "constraint_projection": None,
            }
            if return_trajectory:
                res.update(trajectory=None, trajectory_status="NO_CANDIDATES")
            return res

        # Probe R10_P01: Pin one serving snapshot at request start to guarantee 100% version consistency
        active_snap = self.container.get_snapshot() if hasattr(self, "container") and self.container is not None else None

        # 0. Layer 0: Adaptive Input Modality Ingestion Gateway (Single unified pass, Probe R9_I01)
        ingested = self.modality_router.ingest(raw_input=state)
        effective_state = ingested.normalized_state
        modality_info = ingested.to_dict()

        # 1. Dynamic-K MoE Routing
        if mode == "auto":
            dyn = self.moe_router.route_dynamic(
                state=effective_state,
                candidates=candidates,
                policy_entropy=policy_entropy,
                task_hint=task_hint
            )
            k = dyn["k"]
            selected_experts = dyn["selected_experts"]
            complexity = dyn["complexity_score"]
            has_cpsat = dyn["pipeline_has_cpsat"]
        elif mode == "fast":
            k = 1
            selected_experts = [("reflex", 1.0)]
            complexity = 0.20
            has_cpsat = False
        else:
            k = 1
            selected_experts = [(mode, 1.0)]
            complexity = 0.50
            has_cpsat = (mode == "cp_sat")

        if transition_fn is not None:
            trans_fn = transition_fn
            step_fn = wrap_caller_transition(transition_fn)
        else:
            trans_fn = self._adaptive_transition_tuple
            step_fn = self._adaptive_transition

        # 2. Stage 1: Mandatory Hard Safety Boundary Pruning (Unconditional across ALL modes)
        # Model can choose how to plan, but CANNOT choose to bypass hard safety constraints!
        sat_res = self.cp_sat_solver.verify_and_prune(effective_state, candidates)
        feasible = sat_res["feasible_actions"]
        pruned_by_cpsat = len(feasible) < len(candidates)
        if legal_actions_fn is not None:
            root_legal = set(legal_actions_fn(effective_state))
            feasible = [a for a in feasible if a in root_legal]
        effective_candidates = list(feasible)

        if not feasible:
            latency = (time.time() - t0) * 1000.0
            infeasible = {
                "action": "ABSTAIN",
                "confidence": 0.0,
                "probs": {c: 0.0 for c in candidates},
                "value": -10.0,
                # Only the Stage-1 cp_sat pre-filter ran; no planner was executed.
                "k_experts": 1,
                "experts_activated": ["cp_sat"],
                "expert_weights": {"cp_sat": 1.0},
                "initial_routing_plan": [name for name, _ in selected_experts],
                "complexity_score": round(complexity, 3),
                "adaptive_horizon": 0,
                "adaptive_params": {"status": "INFEASIBLE_ABSTAIN", "pruned_count": len(candidates)},
                "status": "INFEASIBLE_ABSTAIN",
                "pruned_by_cpsat": True,
                "formal_verification": self._constraint_formal_status(constraint_compiler, None, candidates, []),
                "cpsat_solver_status": sat_res.get("solver_status"),
                "constraint_projection": None,
                "expert_outputs": {
                    "cp_sat": {
                        "probs": {c: 0.0 for c in candidates},
                        "value": -10.0,
                        "best_action": "ABSTAIN",
                        "status": "INFEASIBLE_ABSTAIN",
                        "pruned": sat_res["pruned_actions"]
                    }
                },
                "mode": mode,
                "expert_routed": "cp_sat",
                "modality": modality_info,
                "latency_ms": round(latency, 2)
            }
            if return_trajectory:
                infeasible.update(trajectory=None, trajectory_status="INFEASIBLE_ABSTAIN")
            return infeasible

        # 3. Stage 2: Execute selected planners and aggregate via weighted consensus
        planner_experts = [e for e in selected_experts if e[0] != "cp_sat"]
        if not planner_experts:
            planner_experts = [("cp_sat", 1.0)]

        # In auto mode, if world_model was selected but the state is a non-grid dict state
        # (and no custom transition_fn is provided), world_model cannot simulate it; route to reflex.
        # When mode == "world_model" (explicit caller choice), this is skipped and it fails closed.
        routing_reassignment: Optional[str] = None
        if mode == "auto" and any(e[0] == "world_model" for e in planner_experts) and transition_fn is None:
            if isinstance(effective_state, dict) and not effective_state.get("is_visual"):
                try:
                    _validate_grid_state_schema(effective_state)
                except UnsupportedStateError:
                    # Merge by name: the router can pick reflex alongside world_model, and
                    # a duplicate entry would run reflex twice and overwrite its output.
                    merged: Dict[str, float] = {}
                    for name, w in planner_experts:
                        target = "reflex" if name == "world_model" else name
                        merged[target] = merged.get(target, 0.0) + w
                    planner_experts = list(merged.items())
                    routing_reassignment = "routed_world_model_reassigned_to_reflex:non_grid_state"

        total_weight = sum(w for _, w in planner_experts) or 1.0
        norm_planners = [(name, w / total_weight) for name, w in planner_experts]

        accum_probs: Dict[str, float] = {c: 0.0 for c in candidates}
        accum_val = 0.0
        expert_outputs = {}

        all_adaptive_meta = {}
        expert_order: List[str] = []
        static_weight_by_name: Dict[str, float] = {}
        raw_value_by_name: Dict[str, float] = {}
        for expert_name, norm_w in norm_planners:
            e_probs, e_val, e_best, e_meta = self._execute_expert_distribution(
                expert_name=expert_name,
                state=effective_state,
                candidates=effective_candidates,
                trans_fn=trans_fn,
                candidate_descriptions=candidate_descriptions,
                active_snapshot=active_snap,
                legal_actions_fn=legal_actions_fn,
                observed_opponent_action=observed_opponent_action,
                complexity_score=complexity,
                task_hint=task_hint,
            )
            all_adaptive_meta[expert_name] = e_meta
            expert_outputs[expert_name] = {
                "probs": e_probs,
                "value": round(e_val, 4),
                "best_action": e_best,
                "weight": round(norm_w, 4),
                "meta": e_meta
            }
            expert_order.append(expert_name)
            static_weight_by_name[expert_name] = norm_w
            raw_value_by_name[expert_name] = e_val

        # Instance-Adaptive Gating: replace the static norm_w weighted mean with
        # per-sample dynamic weights + a log-linear opinion pool. GenZero.__init__
        # already refused to start adaptive gating unless the loaded router covers
        # every paradigm in self.moe_router.all_paradigms, so every expert name
        # that can reach this point must have a fitted risk model. If one somehow
        # doesn't (e.g. all_paradigms was mutated after construction), that is a
        # broken invariant: raise loudly rather than silently reverting to the
        # static consensus, which would hide the corruption.
        fusion_weight_by_name = dict(static_weight_by_name)
        used_adaptive_gating = False
        if self.adaptive_gating_router is not None:
            if expert_order == ["cp_sat"]:
                # cp_sat is a Stage-1 constraint solver deliberately excluded from router fitting;
                # when it is the sole surviving planner, pass it through directly as identity.
                sole_expert = "cp_sat"
                accum_probs = {c: expert_outputs[sole_expert]["probs"].get(c, 0.0) for c in candidates}
                accum_val = raw_value_by_name[sole_expert]
                fusion_weight_by_name[sole_expert] = 1.0
                expert_outputs[sole_expert]["weight"] = 1.0
                used_adaptive_gating = True
            else:
                fitted_names = set(self.adaptive_gating_router.expert_names)
                unfitted = set(expert_order) - fitted_names
                if unfitted:
                    raise RuntimeError(
                        f"adaptive gating is enabled but expert(s) {sorted(unfitted)} have no fitted "
                        "risk model in adaptive_gating_router; this should be impossible after the "
                        "construction-time coverage check in GenZero.__init__, so moe_router.all_paradigms "
                        "must have changed after the router was validated."
                    )
                expert_probs_matrix = np.array(
                    [[expert_outputs[name]["probs"].get(c, 0.0) for c in candidates] for name in expert_order],
                    dtype=np.float64,
                )
                probs_rows = [expert_probs_matrix[m:m + 1] for m in range(len(expert_order))]
                dynamic_w = self.adaptive_gating_router.predict_weights(probs_rows, expert_names=expert_order)
                fused = self.adaptive_gating_router.fuse_predictions(
                    probs_rows,
                    pool_type=self.config.adaptive_gating_pool_type,
                    dynamic_weights=dynamic_w,
                )[0]
                accum_probs = {c: float(fused[i]) for i, c in enumerate(candidates)}
                for i, name in enumerate(expert_order):
                    fusion_weight_by_name[name] = float(dynamic_w[0, i])
                    expert_outputs[name]["weight"] = round(fusion_weight_by_name[name], 4)
                accum_val = sum(
                    fusion_weight_by_name[name] * raw_value_by_name[name] for name in expert_order
                )
                used_adaptive_gating = True
        else:
            for name in expert_order:
                norm_w = static_weight_by_name[name]
                e_probs = expert_outputs[name]["probs"]
                for c in candidates:
                    accum_probs[c] += norm_w * e_probs.get(c, 0.0)
                accum_val += norm_w * raw_value_by_name[name]

        # 4. Final action selection from consensus & declared valid support set (Probe R10_E01)
        expert_valid_sets = [
            set(out["meta"]["valid_set"])
            for out in expert_outputs.values()
            if isinstance(out.get("meta"), dict) and "valid_set" in out["meta"]
        ]
        if expert_valid_sets:
            declared_valid = set.intersection(*expert_valid_sets)
            valid_candidates = [c for c in effective_candidates if c in declared_valid]
        else:
            valid_candidates = [c for c in effective_candidates if accum_probs.get(c, 0.0) > 0.0]

        # X-CLIENT01 fail-closed fix: the hard consensus support set must propagate through
        # every downstream step. Zero the probability mass of every candidate outside
        # ``valid_candidates`` right here, unconditionally, so neither the arbiter fallback
        # nor _apply_action_constraints can later resurrect an excluded candidate by taking
        # argmax over stale accum_probs mass that was never actually cleared.
        valid_set = set(valid_candidates)
        accum_probs = {c: (accum_probs.get(c, 0.0) if c in valid_set else 0.0) for c in candidates}

        has_active_valid_candidates = len(valid_candidates) > 0
        abstain_statuses = [
            out["meta"]["status"]
            for out in expert_outputs.values()
            if isinstance(out.get("meta"), dict) and out["meta"].get("status") in EXPERT_ABSTAIN_STATUSES
        ]
        is_all_invalid = bool(abstain_statuses) or not has_active_valid_candidates

        if not has_active_valid_candidates or is_all_invalid:
            best_action = "ABSTAIN"
            confidence = 0.0
            accum_probs = {c: 0.0 for c in candidates}
        else:
            valid_choices = [c for c in valid_candidates if c in accum_probs and accum_probs[c] > 0.0]
            if valid_choices:
                best_action = max(valid_choices, key=lambda c: accum_probs.get(c, -1.0))
                confidence = accum_probs.get(best_action, 0.0)
            else:
                best_action = "ABSTAIN"
                confidence = 0.0
                accum_probs = {c: 0.0 for c in candidates}

        # 5. Cloud-Edge GPU Arbiter Fallback Check (Uncertainty / Low Confidence)
        arbiter_report = None
        # Never trigger arbiter fallback if all candidates are invalid or action is ABSTAIN (Probes R9_E02, R9_E03, R10_E01)
        if (
            self.config.enable_gpu_arbiter_fallback
            and best_action != "ABSTAIN"
            and has_active_valid_candidates
            and not is_all_invalid
        ):
            from .adaptive_engine import AdaptiveParameterEngine
            ent = AdaptiveParameterEngine.get_normalized_entropy(accum_probs)
            if self.arbiter_bridge.should_trigger_fallback(confidence=confidence, entropy=ent):
                # Pass strictly valid_candidates to Arbiter to maintain invariant valid support set (Probe R10_E01)
                verdict = self.arbiter_bridge.arbitrate(
                    state=effective_state,
                    candidates=valid_candidates,
                    local_action=best_action,
                    local_confidence=confidence,
                    local_probs=accum_probs,
                    mode="sync" if task_hint == "sync_arbiter" else "async"
                )
                # Re-validate arbitrated action against hard constraints & valid support set
                raw_probs = verdict.probs or {}
                valid_raw_probs = {}
                for c in candidates:
                    val = raw_probs.get(c, 0.0)
                    try:
                        f_val = float(val)
                        if not math.isfinite(f_val) or f_val < 0.0:
                            f_val = 0.0
                    except (ValueError, TypeError):
                        f_val = 0.0
                    valid_raw_probs[c] = f_val

                filtered_probs = {c: valid_raw_probs.get(c, 0.0) if c in valid_candidates else 0.0 for c in candidates}
                prob_sum = sum(filtered_probs[c] for c in valid_candidates)
                if prob_sum > 1e-6 and math.isfinite(prob_sum):
                    accum_probs = {c: (filtered_probs[c] / prob_sum) if c in valid_candidates else 0.0 for c in candidates}
                    if verdict.action in valid_candidates and accum_probs.get(verdict.action, 0.0) > 0.0:
                        best_action = verdict.action
                    else:
                        best_action = max(valid_candidates, key=lambda c: accum_probs.get(c, 0.0))
                    confidence = accum_probs.get(best_action, 0.0)
                else:
                    # If arbiter returns zero probabilities, do NOT manufacture uniform distribution over invalid actions!
                    pass

                arbiter_report = verdict.to_dict()
                arbiter_report["backend_reachable"] = getattr(verdict, "backend_reachable", False)


        # 6. Caller constraints: the final, authoritative gate. Runs after the arbiter so nothing
        # downstream can re-select a disabled action.
        constraint_projection = None
        disabled_by_constraints: List[str] = []
        if constraint_compiler is not None:
            accum_probs, best_action, confidence, constraint_projection, disabled_by_constraints = (
                self._apply_action_constraints(
                    constraint_compiler, candidates, accum_probs, best_action, confidence,
                    hard_support_set=valid_candidates,
                )
            )
        formal_verification = self._constraint_formal_status(
            constraint_compiler, best_action, candidates, disabled_by_constraints
        )

        adaptive_horizon = self.world_model.determine_adaptive_horizon(state, candidates, complexity, task_hint)

        # Every "what ran" field below comes from norm_planners, the plan actually executed.
        # selected_experts is only the router's first choice: it can hold cp_sat (which runs
        # as the Stage-1 pre-filter, not as a planner) or a world_model later reassigned.
        executed_k = len(norm_planners)
        adaptive_params = {
            "k_experts": executed_k,
            "adaptive_horizon": adaptive_horizon,
            "complexity_score": round(complexity, 3),
            "arbiter_fallback": arbiter_report,
            "experts_meta": all_adaptive_meta,
            "fusion": {
                "pool_type": self.config.adaptive_gating_pool_type if used_adaptive_gating else "arithmetic",
                "dynamic": used_adaptive_gating,
                "weights": {name: round(fusion_weight_by_name[name], 4) for name in expert_order},
            },
        }

        latency = (time.time() - t0) * 1000.0
        res_dict = {
            "action": best_action,
            "confidence": round(confidence, 4),
            "probs": {c: round(accum_probs[c], 4) for c in candidates},
            "value": round(accum_val, 4),
            "k_experts": executed_k,
            "experts_activated": [name for name, _ in norm_planners],
            "expert_weights": {name: round(w, 4) for name, w in norm_planners},
            "complexity_score": round(complexity, 3),
            "adaptive_horizon": adaptive_horizon,
            "adaptive_params": adaptive_params,
            "pruned_by_cpsat": pruned_by_cpsat,
            "formal_verification": formal_verification,
            "cpsat_solver_status": sat_res.get("solver_status"),
            "constraint_projection": constraint_projection,
            "arbiter_fallback": arbiter_report,
            "expert_outputs": expert_outputs,
            "mode": mode,
            "expert_routed": norm_planners[0][0],
            "modality": modality_info,
            "degraded": any(
                isinstance(m, dict) and m.get("degraded") for m in all_adaptive_meta.values()
            ),
            "latency_ms": round(latency, 2)
        }
        if [name for name, _ in selected_experts] != res_dict["experts_activated"]:
            res_dict["initial_routing_plan"] = [name for name, _ in selected_experts]
        if routing_reassignment is not None:
            res_dict["routing_reassignment"] = routing_reassignment
            adaptive_params["routing_reassignment"] = routing_reassignment
        if abstain_statuses:
            res_dict["status"] = f"{abstain_statuses[0]}_ABSTAIN"
        expert_reasons = sorted({
            m["degraded_reason"] for m in all_adaptive_meta.values()
            if isinstance(m, dict) and m.get("degraded") and m.get("degraded_reason")
        })
        if expert_reasons:
            res_dict["degraded_reason"] = ";".join(expert_reasons)
        # Two separate facts: the instance's checkpoint state, and the scorer the
        # primary expert actually ran. The primary is the top-ranked expert that was
        # executed; in auto mode selected_experts[0] can be cp_sat, which only runs as
        # the pre-filter above and has no expert meta of its own.
        res_dict["weights_loaded_from_checkpoint"] = bool(self.weights_loaded_from_checkpoint)
        primary_expert = norm_planners[0][0]
        primary_meta = all_adaptive_meta.get(primary_expert)
        # Non-neural scorers propagate to top level regardless of checkpoint state;
        # neural scorers (e.g. torch_live_model) remain in expert_outputs unless untrained.
        if isinstance(primary_meta, dict) and primary_meta.get("scorer") in NON_NEURAL_SCORERS:
            res_dict["scorer"] = primary_meta["scorer"]
            res_dict["scorer_expert"] = primary_expert
            if "confidence_kind" in primary_meta:
                res_dict["confidence_kind"] = primary_meta["confidence_kind"]
        if res_dict.get("scorer") in NON_NEURAL_SCORERS:
            # The primary is non-neural, but a fused secondary expert may still have
            # scored with random-init weights. Surface that instead of hiding it
            # behind the primary's label.
            untrained_experts = sorted(
                name for name, m in all_adaptive_meta.items()
                if isinstance(m, dict) and m.get("scorer") == UNTRAINED_WEIGHTS_SCORER
            )
            if untrained_experts:
                logging.getLogger(__name__).warning(
                    "decide: primary scorer=%s, but fused experts %s used untrained random-init "
                    "weights (degraded=True)", res_dict["scorer"], untrained_experts,
                )
                reasons = [r for r in str(res_dict.get("degraded_reason") or "").split(";") if r]
                reason = f"{UNTRAINED_WEIGHTS_SCORER}:{','.join(untrained_experts)}"
                if reason not in reasons:
                    reasons.append(reason)
                res_dict["degraded"] = True
                res_dict["degraded_reason"] = ";".join(reasons)
        self._mark_untrained(res_dict, "decide")
        if return_trajectory:
            if best_action == "ABSTAIN":
                res_dict.update(trajectory=None, trajectory_status="ABSTAIN")
            else:
                try:
                    roll = rollout(
                        effective_state, adaptive_horizon, step_fn,
                        greedy_lookahead_policy(best_action, list(effective_candidates), step_fn),
                    )
                    roll["continuation_policy"] = "greedy_one_step_over_feasible_candidates"
                    res_dict.update(trajectory=roll, trajectory_status="OK")
                except UnsupportedStateError as exc:
                    # Fail visibly, not silently: the decision above (action/probs/confidence)
                    # is already final and correct. Only the bonus continuation rollout doesn't
                    # fit this state/candidate shape, so scope the catch to just that call
                    # instead of forcing the caller to re-run the whole decide() a second time
                    # (which would double CP-SAT solves, arbiter network calls, etc).
                    #
                    # Any OTHER ValueError (a broken caller-supplied transition_fn, a
                    # non-finite reward, a bad horizon) is a real contract violation, not a
                    # state/schema mismatch, and is deliberately left uncaught here so it
                    # fails closed instead of being repackaged as a fake "unsupported state".
                    res_dict.update(
                        trajectory=None,
                        trajectory_status=f"UNSUPPORTED_STATE_FOR_TRAJECTORY: {exc}",
                    )
            res_dict["latency_ms"] = round((time.time() - t0) * 1000.0, 2)
        return _sanitize_for_json(res_dict)

    @staticmethod
    def _constraint_formal_status(compiler: Any, selected: Optional[str], candidates: Sequence[str],
                                  disabled: Sequence[str]) -> str:
        if compiler is None:
            return FV_NOT_REQUESTED
        return compiler.formal_verification_status(selected, list(candidates), list(disabled))

    @staticmethod
    def _apply_action_constraints(
        compiler: Any,
        candidates: List[str],
        accum_probs: Dict[str, float],
        best_action: str,
        confidence: float,
        hard_support_set: Optional[Sequence[str]] = None,
    ) -> Tuple[Dict[str, float], str, float, Dict[str, Any], List[str]]:
        """Project the final distribution onto the caller constraints and re-pick the action.

        ABSTAIN stays ABSTAIN. If the constraints cannot be met the result is ABSTAIN with all
        probabilities 0: an unsatisfiable rule set never falls back to an unconstrained pick.
        The previous pick is kept only if it is still a top-probability candidate.

        ``hard_support_set`` (X-CLIENT01 fail-closed fix), when given, is the consensus-
        restricted candidate set computed upstream (declared-valid intersection across
        experts). This function does not trust the caller to have already zeroed
        ``accum_probs`` outside it: every candidate outside ``hard_support_set`` is forced
        to zero probability *before* the caller-constraint projection runs, and the final
        pick is restricted to that same set. A caller-constraint pass must never be able to
        resurrect a candidate the consensus already excluded, even when that candidate
        still carries stale positive mass in ``accum_probs`` or has no caller constraint
        naming it at all.
        """
        if best_action == "ABSTAIN":
            return accum_probs, best_action, confidence, {"status": "SKIPPED_ABSTAIN"}, []
        candidate_set = set(candidates)
        allowed = (set(hard_support_set) & candidate_set) if hard_support_set is not None else candidate_set
        gated_probs = {c: (accum_probs.get(c, 0.0) if c in allowed else 0.0) for c in candidates}
        proj = compiler.project_probabilities(gated_probs)
        report = proj.to_dict()
        if proj.status != "OK":
            logging.getLogger(__name__).error("Action constraints unsatisfiable (%s); abstaining.", proj.status)
            report["status"] = f"CONSTRAINTS_UNSATISFIABLE:{proj.status}"
            return {c: 0.0 for c in candidates}, "ABSTAIN", 0.0, report, list(proj.disabled)
        allowed_probs = {c: (proj.probs.get(c, 0.0) if c in allowed else 0.0) for c in candidates}
        top = max(allowed_probs.values())
        if top <= 0.0:
            logging.getLogger(__name__).error(
                "Action constraints leave no positive mass inside the hard support set; abstaining."
            )
            report["status"] = "HARD_SUPPORT_SET_EMPTY_AFTER_PROJECTION"
            return {c: 0.0 for c in candidates}, "ABSTAIN", 0.0, report, list(proj.disabled)
        if allowed_probs.get(best_action, 0.0) >= top - 1e-12:
            new_best = best_action
        else:
            new_best = max(allowed, key=lambda c: allowed_probs.get(c, 0.0))
        return allowed_probs, new_best, allowed_probs[new_best], report, list(proj.disabled)

    def _adaptive_transition(self, state: Any, action: Any) -> StepOutcome:
        """One imagined step; continuous latent vectors go to neural dynamics, dict/str to the text model.

        Fail-closed: a vector state without a matching mounted neural model, a bare 2-D
        position (the heuristic model reads no position from a tuple), a dict that is not
        a valid grid-world schema, an action outside the grid whitelist, or any other type
        raises UnsupportedStateError (a ValueError subclass). Neural errors (weights not
        loaded, unknown action) propagate as plain ValueError/other exceptions.
        """
        nd = self.neural_dynamics_model
        if isinstance(state, (np.ndarray, list, tuple)):
            if nd is None:
                raise UnsupportedStateError(
                    "vector state needs a mounted neural dynamics model "
                    "(GenZeroConfig.neural_dynamics_checkpoint); none is mounted"
                )
            try:
                z = np.asarray(state, dtype=np.float32).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise UnsupportedStateError(f"vector state is not numeric: {exc}") from exc
            if z.shape != (nd.state_dim,):
                raise UnsupportedStateError(
                    f"vector state has shape {z.shape}, neural dynamics model expects ({nd.state_dim},)"
                )
            try:
                z_next, r_hat, done = nd.step(z, action)
            except KeyError as exc:
                raise UnsupportedStateError(f"neural dynamics model cannot encode action {action!r}: {exc}") from exc
            return StepOutcome(z_next, float(r_hat), bool(done), float(r_hat), bool(done), SOURCE_NEURAL, PROVENANCE_NEURAL)

        if isinstance(state, dict):
            if state.get("is_visual"):
                provenance = PROVENANCE_TEXT
            else:
                _validate_grid_state_schema(state)
                if not isinstance(action, str):
                    raise UnsupportedStateError(f"discrete world model needs a string action, got {type(action).__name__}")
                if action not in _GRID_ACTIONS:
                    raise UnsupportedStateError(
                        f"Unsupported action {action!r} for discrete grid state; "
                        f"expected one of {sorted(_GRID_ACTIONS)}"
                    )
                provenance = PROVENANCE_GRID
        elif isinstance(state, str):
            if not isinstance(action, str):
                raise UnsupportedStateError(f"discrete world model needs a string action, got {type(action).__name__}")
            provenance = PROVENANCE_TEXT
        else:
            raise UnsupportedStateError(
                f"unsupported state type {type(state).__name__}; expected ndarray/list vector, dict or str"
            )

        p_fail = float(self.world_model.predict_consequences(state, action)["p_fail"])
        next_state, reward, done = self.world_model.virtual_step(state, action)
        # Heuristic model: only the p_fail > 0.5 branch is lethal; it returns reward -5.0.
        hazard = bool(done) and reward < 0
        return StepOutcome(next_state, float(reward), bool(done), 1.0 - p_fail, hazard, SOURCE_HEURISTIC, provenance)

    def _adaptive_transition_tuple(self, state: Any, action: Any) -> Tuple[Any, float, bool]:
        """(next_state, reward, done) view of _adaptive_transition for the planning engines."""
        out = self._adaptive_transition(state, action)
        return out.next_state, out.reward, out.done

    def simulate(self, state: Any, actions: Sequence[Any], horizon: Optional[int] = None) -> Dict[str, Any]:
        """Roll a fixed action sequence forward through the adaptive world model.

        horizon defaults to len(actions); a larger horizon raises instead of inventing actions.
        Stops at the first done step. See world_model/simulation.py for done vs hazard.
        """
        t0 = time.perf_counter()
        if isinstance(actions, (str, bytes)) or not isinstance(actions, Sequence) or len(actions) == 0:
            raise ValueError("simulate needs a non-empty sequence of actions")
        horizon = len(actions) if horizon is None else validate_horizon(horizon)
        if horizon > len(actions):
            raise ValueError(f"horizon {horizon} exceeds the {len(actions)} given actions")
        result = rollout(state, horizon, self._adaptive_transition, fixed_plan_policy(list(actions)))
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        return result

    def what_if(self, state: Any, candidates: Sequence[Any], horizon: int = 5) -> Dict[str, Any]:
        """Counterfactual branches: each candidate as the first move, then greedy continuation over candidates."""
        t0 = time.perf_counter()
        cands = self._validate_candidates(candidates)
        horizon = validate_horizon(horizon)
        outcomes: Dict[str, Dict[str, Any]] = {}
        for c in cands:
            policy = greedy_lookahead_policy(c, cands, self._adaptive_transition)
            outcomes[str(c)] = summarize_candidate(rollout(state, horizon, self._adaptive_transition, policy))
        ranking = rank_candidates(outcomes)
        return {
            "candidate_outcomes": outcomes,
            "best_candidate": ranking[0],
            "safety_ranking": ranking,
            "traps_detected": [c for c in outcomes if not outcomes[c]["is_safe"]],
            "all_candidates_trapped": all(not o["is_safe"] for o in outcomes.values()),
            "horizon": horizon,
            "continuation_policy": "greedy_one_step_over_candidates",
            "provenance": sorted({p for o in outcomes.values() for p in o["provenance"]}),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    def audit_action(
        self,
        state: Any,
        action: Any,
        horizon: int = 5,
        continuation_actions: Optional[Sequence[Any]] = None,
        warn_risk: float = 0.3,
    ) -> Dict[str, Any]:
        """Shadow risk review of an action an outside actor plans to take.

        Rolls ``action`` forward for ``horizon`` steps. Later steps follow a greedy
        one-step policy over ``continuation_actions``; without them the action is repeated.
        risk_score = 1 - min safe_prob along the rollout. Verdict: any hazard ->
        REJECT_LETHAL; else risk_score >= warn_risk -> WARN_HAZARD; else APPROVED.
        """
        t0 = time.perf_counter()
        horizon = validate_horizon(horizon)
        if not 0.0 < warn_risk <= 1.0:
            raise ValueError(f"warn_risk must lie in (0, 1], got {warn_risk}")
        if continuation_actions is None:
            action_set, policy_name = [action], "repeat_audited_action"
        else:
            action_set, policy_name = self._validate_candidates(continuation_actions), "greedy_one_step_over_continuation_actions"
        roll = rollout(state, horizon, self._adaptive_transition,
                       greedy_lookahead_policy(action, action_set, self._adaptive_transition))
        if roll["min_safe_prob"] is None:
            raise RuntimeError("audit_action: rollout produced no safety estimate; refusing to score risk")
        risk = min(1.0, max(0.0, 1.0 - roll["min_safe_prob"]))
        hazard_step = roll["first_hazard_step"]
        is_uncalibrated_text = PROVENANCE_TEXT in roll["provenance"]
        if hazard_step is not None:
            verdict = "REJECT_LETHAL"
            who = "the audited action itself" if hazard_step == 1 else f"the {policy_name} continuation"
            explanation = (f"Hazard at step {hazard_step} of {horizon} ({who}); "
                           f"survives {roll['survival_horizon']} step(s); risk_score={risk:.3f}.")
        elif risk >= warn_risk:
            verdict = "WARN_HAZARD"
            explanation = (f"No hazard in {roll['steps_simulated']} simulated step(s), but lowest safe_prob "
                           f"{roll['min_safe_prob']:.3f} gives risk_score={risk:.3f} >= warn_risk={warn_risk}.")
        elif is_uncalibrated_text:
            # Text heuristic is hand-set keyword matching, never calibrated against observed
            # outcomes (see world_model/text_world_model.py). It cannot earn an APPROVED.
            verdict = "UNVERIFIED_HEURISTIC"
            explanation = (f"No hazard in {roll['steps_simulated']} simulated step(s) of horizon {horizon}; "
                           f"risk_score={risk:.3f} < warn_risk={warn_risk}. This assessment is based on the "
                           f"uncalibrated text_heuristic provenance (hand-set keyword rules, not fit to data): "
                           f"treat as unverified, not a safety guarantee.")
        else:
            verdict = "APPROVED"
            explanation = (f"No hazard in {roll['steps_simulated']} simulated step(s) of horizon {horizon}; "
                           f"risk_score={risk:.3f} < warn_risk={warn_risk}.")
        return {
            "action": to_jsonable(action),
            "is_safe": roll["is_safe"],
            "survival_horizon": roll["survival_horizon"],
            "risk_score": risk,
            "hazard_detected": hazard_step is not None,
            "first_hazard_step": hazard_step,
            "verdict": verdict,
            "explanation": explanation,
            "continuation_policy": policy_name,
            "safe_prob_source": roll["safe_prob_source"],
            "provenance": roll["provenance"],
            "trajectory": roll["trajectory"],
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    @staticmethod
    def _validate_candidates(candidates: Sequence[Any]) -> List[Any]:
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence) or len(candidates) == 0:
            raise ValueError("candidates must be a non-empty sequence of actions")
        names = [str(c) for c in candidates]
        if len(set(names)) != len(names):
            raise ValueError(f"candidates must be unique, got {names}")
        return list(candidates)

    def plan_path(
        self,
        start_state: Any,
        is_goal_fn: Callable[[Any], bool],
        get_neighbors_fn: Callable[[Any], List[Tuple[Any, str, float]]],
        heuristic_fn: Optional[Callable[[Any], float]] = None
    ) -> Dict[str, Any]:
        """System 2: Uncertainty-driven A* Graph Planning."""
        return self.astar_planner.plan(
            start_state=start_state,
            is_goal_fn=is_goal_fn,
            get_neighbors_fn=get_neighbors_fn,
            heuristic_fn=heuristic_fn
        )

    def plan_bidirectional(
        self,
        start_state: Any,
        goal_state: Any,
        get_forward_neighbors_fn: Callable[[Any], List[Tuple[Any, str, float]]],
        get_backward_neighbors_fn: Callable[[Any], List[Tuple[Any, str, float]]],
        heuristic_fn: Optional[Callable[[Any, Any], float]] = None
    ) -> Dict[str, Any]:
        """System 2: Bidirectional Goal-Directed Search (Start <---> Goal)."""
        return self.bidirectional_planner.plan_bidirectional(
            start_state=start_state,
            goal_state=goal_state,
            get_forward_neighbors_fn=get_forward_neighbors_fn,
            get_backward_neighbors_fn=get_backward_neighbors_fn,
            heuristic_fn=heuristic_fn
        )

    def decide_continuous(
        self,
        state: Any,
        action_dim: int = 4,
        bounds: Optional[Tuple[float, float]] = (-1.0, 1.0),
        simplex: bool = False,
        horizon: Optional[int] = None,
        num_samples: Optional[int] = None,
        custom_reward_fn: Optional[Callable[[List[float], List[float]], float]] = None,
        policy_entropy: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> Dict[str, Any]:
        """Continuous Action Decision Entrypoint via Decoder-Free Latent World Model & MPC.
        
        Solves continuous parameter optimization, robotic continuous joint control,
        and continuous multi-asset portfolio weights:
        - Encodes state directly into latent space z_0 without pixel decoding overhead.
        - Optimizes continuous action trajectories a_{0:H-1} via iterative Latent CEM.
        - Supports box constraints [a_min, a_max]^D and probability simplex constraints.
        - Dynamically scales population and horizon based on state volatility and entropy.
        
        Returns:
            Dict with 'mode', 'action', 'best_action', 'best_trajectory', 'expected_return',
            'horizon', 'num_samples', 'iterations', 'latency_ms', and 'status'.
            When status != 'OK', action and best_trajectory are None.
        """
        return self.mpc_cem_engine.plan_continuous(
            state=state,
            action_dim=action_dim,
            bounds=bounds,
            simplex=simplex,
            horizon=horizon,
            num_samples=num_samples,
            custom_reward_fn=custom_reward_fn,
            policy_entropy=policy_entropy,
            volatility=volatility
        )

    def decide_visual(
        self,
        image: Any,
        candidates: List[str],
        prompt: str = "",
        mode: str = "auto",
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """Layer 0/1/2 Direct Non-Autoregressive Visual Decision Entrypoint.
        
        Uses Qwen3.5-9B Vision Backbone with PyTorch:
        1. Shared Visual Prefill: encodes image + prompt once -> KV Cache + Visual Hidden States.
        2. Candidate Direct Readout: evaluates candidates directly from LM head logits without autoregressive token generation.
        3. Permutation-Equivariant Set Attention: candidate scores have a permutation-equivariant
           structure over candidate ordering; this structural property does not eliminate
           semantic label biases or tie-breaking order dependencies.
        4. Planning MoE / CP-SAT Gating: candidate scoring enters the configured planning and constraint filtering pipeline; results are subject to model, input, and constraint bounds and do not constitute an unconditional zero-collision or zero-hazard guarantee in real environments.
        
        Args:
            image: PIL.Image, numpy frame, or image path string.
            candidates: List of action options (e.g. ['LEFT', 'STAY', 'RIGHT'] or ['REGION_1', 'REGION_2', ...]).
            prompt: Optional task instruction or query context.
            mode: 'auto' | 'direct_readout' | 'mcts' | 'mpc'.
            temperature: Temperature for candidate softmax probability distribution.
            
        Returns:
            Dict with 'action', 'confidence', 'probs', 'visual_prefill_ms', 'scoring_ms',
            'modality', 'latency_ms'.
        """
        import time
        t0 = time.perf_counter()
        
        # 1. Shared Prefill via PyTorch Vision Engine
        prefill_bundle = self.vision_engine.prefill_visual_context(image=image, prompt=prompt)
        
        # 2. Fast Direct Candidate Scoring
        scoring_res = self.vision_engine.score_candidates_direct(
            prefill_bundle=prefill_bundle,
            candidates=candidates,
            temperature=temperature
        )
        
        # 3. Formulate State Representation for Downstream Gen-Zero Planners / Gate
        visual_state = {
            "image": image,
            "prompt": prompt,
            "visual_hidden": prefill_bundle.get("last_hidden_state"),
            "direct_probs": scoring_res["probs"],
            "is_visual": True
        }
        
        # 4. Route through GenZero decide (or CP-SAT safety gate)
        decide_res = self.decide(
            state=visual_state,
            candidates=candidates,
            mode=mode if mode != "direct_readout" else "auto"
        )
        
        total_latency = (time.perf_counter() - t0) * 1000.0
        out = {
            # A direct readout must never override the mandatory safety gate.
            "action": scoring_res["best_action"] if (
                mode == "direct_readout"
                and not decide_res.get("pruned_by_cpsat", False)
                and decide_res.get("action") in candidates
                and scoring_res["best_action"] in candidates
            ) else decide_res.get("action", "ABSTAIN"),
            "confidence": decide_res.get("confidence", max(scoring_res["probs"].values()) if scoring_res["probs"] else 1.0),
            "probs": decide_res.get("probs", scoring_res["probs"]),
            "direct_probs": scoring_res["probs"],
            "visual_prefill_ms": prefill_bundle.get("prefill_ms", 0.0),
            "scoring_ms": scoring_res.get("scoring_ms", 0.0),
            "planner_k": decide_res.get("k_experts", 1),
            "pruned_by_cpsat": decide_res.get("pruned_by_cpsat", False),
            "modality": decide_res.get("modality", {}),
            "latency_ms": round(total_latency, 2)
        }
        reasons = [r for r in str(decide_res.get("degraded_reason") or "").split(";") if r]
        out["degraded"] = bool(decide_res.get("degraded")) or bool(reasons)
        if reasons:
            out["degraded_reason"] = ";".join(dict.fromkeys(reasons))
        if "scorer" in decide_res:
            out["scorer"] = decide_res["scorer"]
        out["weights_loaded_from_checkpoint"] = self.weights_loaded_from_checkpoint
        return out

    def decide_streaming(
        self,
        observation: Any,
        candidates: List[str],
        action_priors: Optional[Dict[str, float]] = None,
        context_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Streaming Spatial-Temporal World Model Entrypoint.
        
        Processes streaming multi-frame inputs via:
        1. Rolling KV-Cache with Attention Sinks (O(W) constant memory).
        2. Latent residual dynamics transition & Pearl causal shock detection.
        3. High-speed (<1.5ms) imagination MCTS lookahead inside latent space.
        
        Args:
            observation: Image frame, vector representation, or state object.
            candidates: List of candidate actions.
            action_priors: Optional prior probability map.
            context_prompt: Optional contextual instruction.
            
        Returns:
            Dict containing 'selected_action', 'action_probabilities', 'expected_value',
            'causal_shock_norm', 'imagined_trajectory', 'kv_cache_stats', and 'latency_breakdown_ms'.
        """
        return self.streaming_world_model.step_stream(
            observation=observation,
            candidate_actions=candidates,
            action_priors=action_priors,
            context_prompt=context_prompt
        )

    def reset_stream(self) -> None:
        """Flushes the streaming world model cache for a new episode."""
        self.streaming_world_model.reset_stream()

    def decide_cpu_extreme(
        self,
        state_repr: Any,
        candidates: List[str],
        temperature: float = 1.0,
        candidate_descriptions: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Direction 2: Pure CPU INT8 Extreme Scorer Entrypoint.
        
        Evaluates candidate actions in < 3.5ms on pure CPU using symmetric INT8
        quantization and vectorized GEMV without any PyTorch dependencies.
        
        Args:
            state_repr: Feature vector, dictionary, or state object.
            candidates: List of candidate action strings.
            temperature: Softmax temperature for probability distribution.
            candidate_descriptions: Optional mapping from candidate keys to semantic descriptions.
            
        Returns:
            Dict containing 'best_action', 'confidence', 'probabilities', 'expected_value',
            and 'scoring_latency_ms'.
        """
        # A raw string (or a dict with no real feature/embedding payload) has no
        # semantic representation available on this pure-CPU path: there is no
        # tokenizer/encoder wired in here. Hashing the text into a sine wave
        # produces a vector that LOOKS like an embedding but carries none of the
        # text's actual meaning (any two unrelated strings are just as
        # "close"/"far" as any other pair) — that is exactly the kind of
        # plausible-looking fake signal this scorer must never emit silently.
        # Fail closed: require a real feature vector, or a dict carrying one.
        if isinstance(state_repr, str):
            raise ValueError(
                "decide_cpu_extreme: state_repr is a raw string but no real embedder "
                "is wired into this CPU-extreme path. Pass a real feature vector "
                "(or a dict with a 'features'/'embedding' key) computed by an actual "
                "encoder; a hash-derived pseudo-vector is not a semantic representation."
            )
        elif isinstance(state_repr, dict):
            s_vec = state_repr.get("features") or state_repr.get("embedding")
            if s_vec is None:
                raise ValueError(
                    "decide_cpu_extreme: dict state_repr has no 'features' or "
                    "'embedding' key. A hash-derived pseudo-vector is not a real "
                    "representation; provide real features or embed upstream."
                )
        else:
            s_vec = state_repr

        res = self.cpu_extreme_scorer.score_candidates(
            state_repr=s_vec,
            candidates=candidates,
            candidate_descriptions=candidate_descriptions,
            temperature=temperature
        )
        # INT8 weights are synced from self.model; without a checkpoint they are random-init.
        return self._mark_untrained(res, "decide_cpu_extreme", trained=self.scorer_synced_from_checkpoint)

    # ---- Issue #8: Composite 4-Tuple Decision Bundling & Policy Gate ----
    def decide_step(
        self,
        state: Any,
        affordances: List[str],
        action_types: Optional[List[str]] = None,
        goal: Optional[str] = None,
        risk_criteria: Optional[str] = None,
        state_feedback: Optional[Dict[str, Any]] = None,
        max_affordances: int = 15,
        apply_policy_gate: bool = True,
        profile: Optional[Any] = None,
        return_trajectory: bool = False
    ) -> Dict[str, Any]:
        """Executes composite 4-tuple decision (target, action, done, risk) in a single pass.

        Bundles:
        1. Anti-truncation candidate ranking (protecting bottom elements like Next/Submit/Confirm).
        2. Fast sub-5ms composite inference.
        3. Dual-track policy gate arbitration (hard rules + soft risk + confidence tiers).

        Args:
            state: Current environment state / observation.
            affordances: Candidate interaction targets.
            action_types: Available action verbs.
            goal: Task goal or user instruction.
            risk_criteria: Optional custom risk definition.
            state_feedback: Optional register values (display reading, focused element).
            max_affordances: Maximum candidates to retain without positional bias.
            apply_policy_gate: Whether to run dual-track policy gate check.
            profile: Optional DomainRiskProfile.
            return_trajectory: Also roll the chosen `target` forward through the same
                world-model rollout used by GenZero.decide (greedy continuation over the
                governed candidate pool), so the trajectory's first step is the action
                this call actually returned, not an independently re-decided one. When
                the state/candidates don't fit a supported world model (grid/vector/text),
                'trajectory' is None and 'trajectory_status' names the ValueError rather
                than silently omitting it.

        Returns:
            Dict containing target, action, done_prob, risk_prob, timing_ms,
            candidate_pool metadata, policy_gate verdict, and (when requested)
            'trajectory' / 'trajectory_status'.
        """
        # 1. Candidate governance
        if max_affordances and max_affordances > 0:
            self.candidate_ranker.max_candidates = max_affordances
        pool = self.candidate_ranker.govern_candidates(
            raw_candidates=affordances,
            state_feedback=state_feedback,
            goal=goal
        )

        # 2. Composite inference
        decision = self.composite_engine.decide_step(
            state=state,
            affordances=pool.candidates,
            action_types=action_types,
            goal=goal,
            risk_criteria=risk_criteria,
            state_feedback=pool.state_feedback
        )

        res = decision.to_dict()
        res["candidate_pool"] = pool.to_dict()

        # 3. Dual-track policy gate check
        if apply_policy_gate:
            gate_verdict = self.policy_gate.evaluate_policy(decision, state=state, profile=profile)
            res["policy_gate"] = gate_verdict.to_dict()

        # 4. Optional smooth world-model rollout of the target this call actually chose
        if return_trajectory:
            if not pool.candidates:
                res["trajectory"] = None
                res["trajectory_status"] = "NO_CANDIDATES"
            else:
                try:
                    horizon = self.world_model.determine_adaptive_horizon(state, pool.candidates)
                    policy = greedy_lookahead_policy(decision.target, pool.candidates, self._adaptive_transition)
                    traj = rollout(state, horizon, self._adaptive_transition, policy)
                    traj["continuation_policy"] = "greedy_one_step_over_feasible_candidates"
                    res["trajectory"] = traj
                    res["trajectory_status"] = "OK"
                except UnsupportedStateError as exc:
                    # Fail visibly, not silently: decide_step's core answer above is still
                    # valid, but the bonus rollout doesn't fit this state/candidate shape.
                    # Any other ValueError (non-finite reward, bad horizon) is a real kernel
                    # contract violation and is left uncaught so it fails closed.
                    res["trajectory"] = None
                    res["trajectory_status"] = f"UNSUPPORTED_STATE_FOR_TRAJECTORY: {exc}"

        return res

    # ---- Issue #9: Batch States Inference Protocol ----
    def decide_batch(
        self,
        states: List[Any],
        candidates: List[str],
        mode: str = "reflex",
        task_hint: Optional[str] = None,
        candidate_descriptions: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """Executes parallel / batched decisions across multiple state observations.

        Supports batched neural forward pass when running in reflex/fast mode with PyTorch,
        or concurrent CPU fallback.

        Args:
            states: List of state observations/contexts.
            candidates: List of available action identifiers.
            mode: 'reflex', 'fast', or 'auto'.
            task_hint: Optional exact paradigm name in UniversalParadigmRouter.all_paradigms.
            candidate_descriptions: Optional criteria/descriptions for each candidate.

        Returns:
            List of decision result dicts, aligned 1:1 with input states.
        """
        if not states:
            return []

        if not candidates:
            return [{"action": None, "confidence": 0.0, "value": 0.0, "mode": mode, "latency_ms": 0.0} for _ in states]

        import time
        t0 = time.time()

        # Check if we can run batched PyTorch forward pass
        active_snap = self.container.get_snapshot() if hasattr(self, "container") and self.container is not None else None
        model = active_snap.model if active_snap is not None else getattr(self, "model", None)
        scorer = active_snap.scorer if active_snap is not None else getattr(self, "cpu_extreme_scorer", None)
        degraded_reason: Optional[str] = None

        if mode in ("reflex", "fast") and model is not None and hasattr(model, "scalar") and hasattr(model, "forward"):
            try:
                import torch
                if hasattr(model, "eval"):
                    model.eval()

                examples = []
                safety_results = []
                for s in states:
                    ingested = self.modality_router.ingest(raw_input=s)
                    eff_s = ingested.normalized_state
                    safety_results.append(self.cp_sat_solver.verify_and_prune(eff_s, candidates))
                    c_ids = [
                        encode_leaf_tokens(eff_s, c, candidate_descriptions.get(c, "") if candidate_descriptions else None)
                        for c in candidates
                    ]
                    examples.append({"leaf_tokens": c_ids, "candidate_ids": candidates, "type": "choice"})

                with torch.no_grad():
                    logits, valid, vals = model(examples, pad_token=0, return_value=True)

                results = []
                batch_latency = (time.time() - t0) * 1000.0 / max(1, len(states))
                for i, s in enumerate(states):
                    if not torch.isnan(logits[i]).any() and not torch.isinf(logits[i]).any():
                        raw_l = logits[i, :len(candidates)].clone()
                        feasible = safety_results[i]["feasible_actions"]
                        pruned = len(feasible) < len(candidates)
                        cand_valid = torch.tensor(
                            [c in feasible for c in candidates], dtype=torch.bool, device=raw_l.device
                        )
                        if hasattr(valid, "dtype") and valid.dtype == torch.bool:
                            cand_valid &= valid[i, :len(candidates)].to(device=raw_l.device)
                        if not cand_valid.any():
                            results.append({
                                "action": "ABSTAIN",
                                "pruned_by_cpsat": pruned,
                                "status": "INFEASIBLE_ABSTAIN",
                                "confidence": 0.0,
                                "probs": {c: 0.0 for c in candidates},
                                "value": 0.0,
                                "mode": mode,
                                "latency_ms": round(batch_latency, 2)
                            })
                            continue
                        raw_l[~cand_valid] = float("-inf")
                        p_t = torch.softmax(raw_l, dim=-1)
                        probs = {c: float(p_t[j].item()) for j, c in enumerate(candidates)}
                        best_act = max(candidates, key=lambda c: probs.get(c, -1.0))
                        conf = probs.get(best_act, 0.0)
                        val = float(vals[i].item()) if vals is not None and i < len(vals) else 0.0
                        results.append({
                            "action": best_act,
                            "pruned_by_cpsat": pruned,
                            "confidence": round(conf, 4),
                            "probs": {c: round(probs[c], 4) for c in candidates},
                            "value": round(val, 4),
                            "mode": mode,
                            "latency_ms": round(batch_latency, 2)
                        })
                    else:
                        logging.getLogger(__name__).warning(
                            "Non-finite batch logits; degrading to per-state decide()"
                        )
                        result = self.decide(s, candidates, mode=mode, task_hint=task_hint, candidate_descriptions=candidate_descriptions)
                        result["degraded"] = True
                        reasons = [result.get("degraded_reason"), "batch_nonfinite_logits"]
                        result["degraded_reason"] = ";".join(reason for reason in reasons if reason)
                        results.append(result)
                if not self.weights_loaded_from_checkpoint:
                    for r in results:
                        if r.get("scorer") != UNTRAINED_WEIGHTS_SCORER:
                            self._mark_untrained(r, "decide_batch")
                return results
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Batched torch forward failed (%s: %s); degrading to per-state decide()",
                    type(exc).__name__, exc,
                )
                degraded_reason = f"batch_torch_exception:{type(exc).__name__}"

        # Fallback path: iterate over states
        fallback = [
            self.decide(
                state=s,
                candidates=candidates,
                mode=mode,
                task_hint=task_hint,
                candidate_descriptions=candidate_descriptions
            )
            for s in states
        ]
        if degraded_reason is not None:
            for r in fallback:
                r["degraded"] = True
                reasons = [r.get("degraded_reason"), degraded_reason]
                r["degraded_reason"] = ";".join(dict.fromkeys(reason for reason in reasons if reason))
        return fallback

    def attribute_counterfactual(
        self,
        trajectory: List[Dict[str, Any]],
        final_outcome: str,
        final_score: float,
        value_evaluator: Optional[Callable[[Any], float]] = None
    ) -> Dict[str, Any]:
        """Runs Pearlian Counterfactual Attribution across an episode trajectory.
        
        Disentangles Action Decision Errors from Exogenous Shocks:
        - Calculates Abduced Noise U_t for every transition.
        - Evaluates do(a*) for alternative actions.
        - Measures Individual Treatment Effect (ITE).
        - Pinpoints the root-cause decision step.
        """
        attributions = self.causal_engine.attribute_trajectory_failures(
            trajectory=trajectory,
            final_outcome=final_outcome,
            final_score=final_score,
            value_evaluator=value_evaluator
        )
        root_cause = self.causal_engine.find_root_cause_step(attributions)

        return {
            "attributions": [
                {
                    "step_idx": a.step_idx,
                    "factual_action": a.factual_action,
                    "best_counterfactual_action": a.best_counterfactual_action,
                    "ite": round(a.individual_treatment_effect, 4),
                    "is_decision_culprit": a.is_decision_culprit,
                    "is_exogenous_shock": a.is_exogenous_shock,
                    "explanation": a.causal_explanation
                }
                for a in attributions
            ],
            "root_cause_step": root_cause.step_idx if root_cause else None,
            "root_cause_best_cf_action": root_cause.best_counterfactual_action if root_cause else None,
            "root_cause_ite": round(root_cause.individual_treatment_effect, 4) if root_cause else 0.0,
            "has_decision_culprit": root_cause is not None
        }

    def register_tool(
        self,
        name: str,
        func: Callable[..., Any],
        description: str = "",
        side_effect_level: SideEffectLevel = SideEffectLevel.READ_ONLY,
        required_params: Optional[List[str]] = None,
        rollback_func: Optional[Callable[..., Any]] = None,
        timeout_seconds: float = 5.0,
        is_idempotent: bool = True
    ) -> ToolDefinition:
        """Registers a tool in the enterprise sandbox with safety annotations."""
        return self.tool_registry.register(
            name=name,
            func=func,
            description=description,
            side_effect_level=side_effect_level,
            required_params=required_params,
            rollback_func=rollback_func,
            timeout_seconds=timeout_seconds,
            is_idempotent=is_idempotent
        )

    def execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        auto_retry_exogenous: bool = True
    ) -> ExecutionResult:
        """Executes a single tool call inside the Causal Tool Sandbox with PRM Safety Barrier audit."""
        tool = self.tool_registry.get(tool_name)
        if not tool:
            return ExecutionResult(
                tool_name=tool_name,
                params=params,
                success=False,
                error_category=ErrorCategory.ACTION_DECISION_ERROR,
                error_message=f"Tool '{tool_name}' not found in registry."
            )

        # Mandatory Global Safety Barrier Gate (enforce ALLOWED only)
        audit = self.safety_barrier.audit(tool, params)
        if audit.verdict != SafetyVerdict.ALLOWED:
            raise PermissionError(
                f"Direct tool execution blocked by PRM Safety Barrier: {audit.verdict.name} - {audit.reason}"
            )

        return self.sandbox.execute(
            tool_name=tool_name,
            params=params,
            auto_retry_exogenous=auto_retry_exogenous
        )

    def execute_workflow(
        self,
        steps: List[WorkflowStep]
    ) -> WorkflowExecutionReport:
        """Executes a multi-step toolchain with PRM formal gating and autonomous self-healing."""
        return self.workflow_planner.execute_workflow(steps=steps)

    def analyze_opponent_intent(
        self,
        opponent_id: str,
        state: Dict[str, float],
        ego_action: str,
        opponent_action: str,
        actual_next_state: Dict[str, float],
        opponent_revealed_strength: Optional[float] = None,
        candidate_opponent_actions: Optional[List[str]] = None,
        nash_strategy: Optional[Dict[str, float]] = None
    ) -> AgentIntent:
        """Analyzes opponent strategic intent, separating external shocks from bluffs."""
        return self.bluff_detector.analyze_intent(
            opponent_id=opponent_id,
            state=state,
            ego_action=ego_action,
            opponent_action=opponent_action,
            actual_next_state=actual_next_state,
            candidate_opponent_actions=candidate_opponent_actions,
            opponent_revealed_strength=opponent_revealed_strength,
            nash_strategy=nash_strategy
        )

    def solve_causal_game(
        self,
        info_set: str,
        legal_actions: List[str],
        base_utilities: Optional[Dict[str, float]] = None,
        observed_opponent_action: Optional[str] = None,
        abduced_shock_norm: float = 0.0,
        opponent_revealed_strength: Optional[float] = None
    ) -> CausalCFROutcome:
        """Solves an imperfect information game using Causal-CFR with exogenous shock invariance."""
        return self.causal_cfr_engine.solve_causal_game(
            info_set=info_set,
            legal_actions=legal_actions,
            base_utilities=base_utilities,
            observed_opponent_action=observed_opponent_action,
            abduced_shock_norm=abduced_shock_norm,
            opponent_revealed_strength=opponent_revealed_strength
        )

    def run_league_evolution_cycle(
        self,
        generations: int = 5,
        matches_per_agent: int = 10
    ) -> Dict[str, Any]:
        """Runs multi-generational League co-evolution with Prioritized Fictitious Self-Play."""
        reports = []
        for _ in range(generations):
            rep = self.league_arena.evolve_league_generation(matches_per_agent=matches_per_agent)
            reports.append(rep)
        robustness = self.league_arena.evaluate_historical_robustness()
        return {
            "evolution_history": reports,
            "final_robustness": robustness,
            "main_agent_elo": self.league_arena.main_agent.elo_rating
        }
    def score(
        self,
        prompt: str,
        candidates: Sequence[str],
        model: Optional[str] = None,
        temperature: float = 1.0,
        timeout: Optional[float] = None,
        use_llamacpp: bool = False,
    ) -> Dict[str, Any]:
        """Minimal non-autoregressive candidate scoring entrypoint (RFC Issue #16).

        Evaluates discrete candidate options against the prompt with 0 generated tokens.
        Adheres to the closed-form confidence contract c = (p_max - 1/K) / (1 - 1/K).

        Args:
            prompt: Query, context, or observation string.
            candidates: Non-empty sequence of candidate string identifiers.
            model: Optional model identifier override.
            temperature: Sampling temperature scaling (> 0).
            timeout: Request timeout in seconds.
            use_llamacpp: If True, explicitly routes through LlamaCppScoreAdapter.
                If False, runs reflex scoring through the Gen-Zero dual-head model.

        Returns:
            Dict conforming to Issue #16 /v1/score wire protocol:
            {
                "choice": str,
                "scores": List[float],
                "probabilities": Dict[str, float],
                "confidence": float,
                "timing_ms": float,
                "source": str
            }
        """
        import time
        t0 = time.perf_counter()

        if not prompt or not isinstance(prompt, str):
            raise ValueError("'prompt' must be a non-empty string.")

        if not candidates or len(candidates) == 0:
            raise ValueError("'candidates' must be a non-empty sequence of strings.")

        candidates_list = [str(c) for c in candidates]

        if use_llamacpp or os.environ.get("LLAMACPP_BASE_URL"):
            return self.llamacpp_adapter.score(
                prompt=prompt,
                candidates=candidates_list,
                model=model,
                temperature=temperature,
                timeout=timeout,
            )

        # Fast reflex scoring path via Gen-Zero reflex decision
        decide_res = self.decide(
            state=prompt,
            candidates=candidates_list,
            mode="reflex",
        )

        probs = decide_res.get("probs") or {c: 1.0 / len(candidates_list) for c in candidates_list}
        choice = decide_res.get("action") or candidates_list[0]
        scores = [round(probs.get(c, 0.0), 4) for c in candidates_list]
        conf = compute_closed_form_confidence(scores) if choice in candidates_list and any(scores) else 0.0
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "choice": choice,
            "scores": scores,
            "probabilities": {c: round(probs.get(c, 0.0), 4) for c in candidates_list},
            "confidence": round(conf, 4),
            "timing_ms": round(elapsed_ms, 2),
            "source": "gen_zero_reflex",
            "degraded": decide_res.get("degraded", False),
            **{key: decide_res[key] for key in (
                "degraded_reason", "scorer", "confidence_kind", "weights_loaded_from_checkpoint",
                "status", "pruned_by_cpsat",
            ) if key in decide_res},
        }

    def decide_alignment(
        self,
        entity_a: Union[Dict[str, Any], str, Any],
        entity_b: Union[Dict[str, Any], str, Any],
        context: Optional[Union[Dict[str, Any], str]] = None,
        hard_safety_constraints: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """Evaluates tri-state state entity alignment (DROP/BUFFER/MERGE) with co-riding probes."""
        verdict = self.alignment_gate.evaluate_alignment(
            entity_a=entity_a,
            entity_b=entity_b,
            context=context,
            hard_safety_constraints=hard_safety_constraints,
        )
        return verdict.to_dict()

    def guard_input(
        self,
        prompt: str,
        context: Optional[str] = None,
        policy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluates input prompt through non-autoregressive Input Guardrail."""
        return self.guardrail.guard_input(prompt=prompt, context=context, policy=policy).to_dict()

    def guard_output(
        self,
        response: str,
        input_context: Optional[str] = None,
        policy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluates response through non-autoregressive Output Guardrail."""
        return self.guardrail.guard_output(response=response, input_context=input_context, policy=policy).to_dict()

    def interlock_tool_pre(
        self,
        tool_name: str,
        tool_args: Any,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Inspects agent tool call before execution with CP-SAT safety interlock."""
        return self.tool_interlock.interlock_pre_execution(tool_name=tool_name, tool_args=tool_args, agent_context=context).to_dict()

    def interlock_tool_post(
        self,
        tool_name: str,
        tool_output: Any,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Inspects agent tool output for sensitive token or data leakage."""
        return self.tool_interlock.interlock_post_execution(tool_name=tool_name, tool_output=tool_output, agent_context=context).to_dict()

    def rerank(
        self,
        query: str,
        passages: List[Union[str, Dict[str, Any], Tuple[str, str]]],
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Reranks candidate passages using closed-form probabilities without text generation."""
        return self.reranker.rerank(query=query, passages=passages, top_k=top_k).to_dict()

    def semantic_find_line(
        self,
        query: str,
        document_text: str,
    ) -> Dict[str, Any]:
        """Locates specific line using decoupled Choice (where) and Noul (exists) questions."""
        return self.semantic_find.find(query=query, document_text=document_text).to_dict()

    def verify_citation(
        self,
        cited_quote: str,
        source_document: str,
        claimed_fact: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Verifies quotation using two-tier literal matching and semantic entailment."""
        return self.citation_verifier.verify_citation(
            cited_quote=cited_quote,
            source_document=source_document,
            claimed_fact=claimed_fact,
        ).to_dict()

    def slot_evidence(
        self,
        query: str,
        passages: List[Tuple[str, str]],
    ) -> Dict[str, Any]:
        """Evaluates 4 Noul probes per passage and slots into conflicting/accepted evidence blocks."""
        return self.evidence_battery.process_passages(query=query, passages=passages).to_dict()

    def dispatch_call(
        self,
        func: Callable,
        user_prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extracts strictly-typed function arguments from natural language."""
        return self.typed_dispatcher.dispatch(func=func, user_prompt=user_prompt, context=context).to_dict()

    def recover_structure(
        self,
        raw_text: str,
    ) -> Dict[str, Any]:
        """Reconstructs structured Markdown from raw text with 100% literal fidelity."""
        return self.structure_recovery.recover(raw_text=raw_text).to_dict()

    def evaluate_with_fail_closed(
        self,
        context: Dict[str, Any],
        evaluation_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
    ) -> Dict[str, Any]:
        """Supervises evaluation under strict Fail-Closed defense guarantee."""
        return self.fail_closed_scheduler.execute_evaluation(context=context, evaluation_fn=evaluation_fn).to_dict()

    def in_process_prefill_scan(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Executes zero-RPC in-process prefill evaluation across 8 concurrent probes."""
        return self.in_process_prefill.evaluate_probes(context=context).to_dict()

    def optimize_cost_sensitive_roc(
        self,
        y_true: List[int],
        y_scores: List[float],
        c_fp: float = 50.0,
        c_fn: float = 1.0,
    ) -> Dict[str, Any]:
        """Calibrates three-state decision boundaries under asymmetric mistake cost."""
        opt = CostSensitiveROCOptimizer(CostMatrix(c_fp=c_fp, c_fn=c_fn))
        return opt.fit(y_true=y_true, y_scores=y_scores).to_dict()

    def generate_contrastive_perturbations(
        self,
        samples: List[str],
    ) -> List[Dict[str, Any]]:
        """Generates paired hard negative samples via minimal semantic perturbation."""
        pairs = self.perturbation_generator.generate_paired_dataset(samples=samples)
        return [p.to_dict() for p in pairs]

    def route_mcp(
        self,
        session_id: str,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        is_subagent: bool = False,
        subagent_task: Optional[str] = None,
        task_error: bool = False,
        top_k_tools: int = 4,
    ) -> Dict[str, Any]:
        """Executes MCP Gateway Adaptive Three-Tuple Routing in < 12ms."""
        decision = self.mcp_router.route(
            session_id=session_id,
            prompt=prompt,
            tools=tools,
            is_subagent=is_subagent,
            subagent_task=subagent_task,
            task_error=task_error,
            top_k_tools=top_k_tools,
        )
        return decision.to_dict()

    def compact_context(
        self,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compacts conversation messages using discrete verbatim actions without fact drift."""
        compacted, items, summary = self.verbatim_compactor.compact_session(messages=messages)
        return {
            "compacted_messages": compacted,
            "items": [item.to_dict() for item in items],
            "summary": summary.to_dict(),
        }

    def inspect_firewall(
        self,
        tool_name: str,
        arguments: Union[str, Dict[str, Any]],
        cost_level: Optional[Union[str, CostOfFailureLevel]] = None,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Performs orthogonal 4-Noul security evaluation with Cost-of-Failure gating in < 15ms."""
        level = None
        if cost_level is not None:
            level = CostOfFailureLevel(cost_level) if isinstance(cost_level, str) else cost_level
        risk_vector = self.parallel_firewall.inspect_tool_call(
            tool_name=tool_name,
            arguments=arguments,
            cost_level=level,
            context=context,
        )
        return risk_vector.to_dict()

    def review_diff(
        self,
        diff_text: str,
    ) -> Dict[str, Any]:
        """Executes 10+ typed verification checks over Git Diff in a single pass (< 500ms)."""
        report = self.pr_reviewer.review_diff(diff_text=diff_text)
        return report.to_dict()

    def process_streaming_utterance(
        self,
        utterance_id: int,
        speaker: str,
        text: str,
        timestamp_ms: int = 0,
        parent_node_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Processes real-time ASR speech / log line through the 4-phase Sandwich Streaming Pipeline."""
        u = StreamUtterance(
            utterance_id=utterance_id,
            speaker=speaker,
            timestamp_ms=timestamp_ms,
            text=text,
        )
        return self.streaming_pipeline.process_utterance(utterance=u, explicit_parent_id=parent_node_id)

    def confirm_workflow_draft(self, draft_id: str) -> Optional[str]:
        """Confirms a tentative workflow draft node in the graph."""
        return self.streaming_pipeline.confirm_draft(draft_id=draft_id)

    def dismiss_workflow_draft(self, draft_id: str) -> bool:
        """Dismisses and purges a tentative workflow draft from the graph."""
        return self.streaming_pipeline.dismiss_draft(draft_id=draft_id)

    def render_workflow_graph(self, format: str = "mermaid") -> str:
        """Renders the workflow graph AST as 'mermaid' or 'drawio_xml'."""
        if format.lower() == "drawio_xml":
            return self.streaming_pipeline.export_drawio_xml()
        return self.streaming_pipeline.export_mermaid()

    def create_browser_harness(
        self,
        url: str = "about:blank",
        title: str = "Blank",
        viewport_width: float = 1280.0,
        viewport_height: float = 800.0,
    ) -> ZeroBrowserHarness:
        """Creates a dedicated ZeroBrowserHarness instance with decision client attached."""
        return ZeroBrowserHarness(
            url=url,
            title=title,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            decision_client=self,
        )

    def observe_browser(
        self,
        elements: Optional[Sequence[DOMElement]] = None,
        overlays: Optional[Sequence[ModalOverlay]] = None,
        url: Optional[str] = None,
        title: Optional[str] = None,
        scroll_x: Optional[float] = None,
        scroll_y: Optional[float] = None,
    ) -> WebObservation:
        """Performs DOM observation and penetration hit-testing on the shared browser harness."""
        return self.browser_harness.observe(
            elements=elements,
            overlays=overlays,
            url=url,
            title=title,
            scroll_x=scroll_x,
            scroll_y=scroll_y,
        )

    def step_browser(
        self,
        action: Union[str, int],
        text_param: Optional[str] = None,
        option_param: Optional[str] = None,
        assert_freshness: bool = True,
    ) -> BrowserStepResult:
        """Executes a single discrete browser step on the shared browser harness."""
        return self.browser_harness.step(
            action_choice=action,
            text_param=text_param,
            option_param=option_param,
            assert_freshness=assert_freshness,
        )

    def run_autonomous_browser(
        self,
        goal: str,
        max_steps: int = 15,
        step_callback: Optional[Callable[[int, BrowserStepResult], None]] = None,
    ) -> Dict[str, Any]:
        """Runs an autonomous closed-loop browser session until goal completion or sentinel escalation."""
        return self.browser_harness.run_autonomous(
            goal=goal,
            decision_client=self,
            max_steps=max_steps,
            step_callback=step_callback,
        )

    def decide_canvas(
        self,
        state: str,
        template: Optional[CanvasTemplate] = None,
        affordances: Optional[Sequence[str]] = None,
        action_types: Optional[Sequence[str]] = None,
        risk_criteria: Optional[str] = None,
        enable_adaptive_sampling: bool = True,
    ) -> CanvasDecisionResult:
        """Executes a structured Canvas decision with optional H1 entropy adaptive sampling."""
        t0 = time.perf_counter()

        # 1. Build or use CanvasTemplate
        tmpl = template
        if tmpl is None:
            tmpl = CanvasTemplate.standard_4tuple(
                affordances=affordances or ["target_primary", "target_secondary"],
                action_types=action_types,
                risk_criteria=risk_criteria,
            )

        prompt = tmpl.format_prompt(state)

        # 2. Evaluate slots
        if not enable_adaptive_sampling:
            slots_res: Dict[str, SlotResult] = {}
            max_ent = 0.0
            for slot in tmpl.slots:
                if slot.slot_type == CanvasSlotType.CHOICE:
                    cands = slot.candidates or ["opt_a", "opt_b"]
                    dec = self.decide(prompt, candidates=cands, mode="reflex")
                    probs = dec.get("probs", {c: 1.0 / len(cands) for c in cands})
                    # A kernel ABSTAIN (e.g. all-zero probs from CONSTRAINTS_UNSATISFIABLE
                    # or NO_VALID_CANDIDATES) must surface as ABSTAIN, never get silently
                    # relabelled as a confident pick.
                    if dec.get("action") == "ABSTAIN":
                        chosen = "ABSTAIN"
                        conf = 0.0
                    else:
                        chosen = dec.get("action")
                        conf = float(dec.get("confidence", 0.5))
                elif slot.slot_type == CanvasSlotType.NOUL:
                    cands = ["true", "false"]
                    dec = self.decide(prompt, candidates=cands, mode="reflex")
                    probs = dec.get("probs", {"true": 0.5, "false": 0.5})
                    if dec.get("action") == "ABSTAIN":
                        chosen = "ABSTAIN"
                        conf = 0.0
                    else:
                        conf = float(probs.get("true", 0.5))
                        chosen = conf >= 0.5
                else:  # SCORE
                    cands = ["low", "medium", "high"]
                    dec = self.decide(prompt, candidates=cands, mode="reflex")
                    probs = dec.get("probs", {"low": 0.33, "medium": 0.34, "high": 0.33})
                    if dec.get("action") == "ABSTAIN":
                        chosen = "ABSTAIN"
                        conf = 0.0
                    else:
                        score_val = float(probs.get("low", 0.0) * 0.1 + probs.get("medium", 0.0) * 0.5 + probs.get("high", 0.0) * 0.9)
                        conf = score_val
                        chosen = round(score_val, 4)

                ent = self.h1_entropy_gate.compute_h1_entropy(list(probs.values()))
                if ent > max_ent:
                    max_ent = ent
                slots_res[slot.name] = SlotResult(
                    name=slot.name,
                    slot_type=slot.slot_type,
                    chosen_value=chosen,
                    confidence=conf,
                    probabilities=probs,
                    entropy=ent,
                    mean_confidence=conf,
                    std_error=0.0,
                    multi_read=False,
                    reads_count=1,
                )
            latency = (time.perf_counter() - t0) * 1000.0
            return CanvasDecisionResult(
                slots=slots_res,
                is_multi_read=False,
                max_entropy=max_ent,
                timing_ms=latency,
                prompt_tokens=len(prompt.split()),
            )

        # Adaptive H1 Entropy sampling.
        #
        # `perturbation` is intentionally NOT used to synthesize fake variance: there
        # is no genuine per-read noise source available at this call site (reflex
        # mode is a deterministic eval-mode forward pass, no dropout/sampling).
        # Previously this hashed the candidate label text into a fabricated offset
        # and called the resulting spread a "multi-read" sample; that produced a
        # deterministic function of the label string dressed up as statistical
        # uncertainty (mu +/- sigma "error bars"), not a real repeated measurement.
        # Each K-pass below genuinely re-invokes the decision pipeline; if the
        # underlying model is deterministic the measured std_dev will honestly
        # come out as 0.0 rather than a manufactured non-zero spread.
        def run_forward_pass(perturbation: float) -> Dict[str, Any]:
            slot_outputs = {}
            for slot in tmpl.slots:
                if slot.slot_type == CanvasSlotType.CHOICE:
                    cands = slot.candidates or ["opt_a", "opt_b"]
                    dec = self.decide(prompt, candidates=cands, mode="reflex")
                    probs = dec.get("probs", {c: 1.0 / len(cands) for c in cands})
                    if dec.get("action") == "ABSTAIN":
                        slot_outputs[slot.name] = {"probs": probs, "action": "ABSTAIN", "conf": 0.0}
                    else:
                        slot_outputs[slot.name] = {"probs": probs, "action": max(probs, key=probs.get), "conf": max(probs.values())}
                elif slot.slot_type == CanvasSlotType.NOUL:
                    cands = ["true", "false"]
                    dec = self.decide(prompt, candidates=cands, mode="reflex")
                    probs = dec.get("probs", {"true": 0.5, "false": 0.5})
                    if dec.get("action") == "ABSTAIN":
                        slot_outputs[slot.name] = {"probs": probs, "action": "ABSTAIN", "conf": 0.0}
                    else:
                        p_val = probs.get("true", 0.5)
                        slot_outputs[slot.name] = {"probs": probs, "action": p_val >= 0.5, "conf": p_val}
                else:  # SCORE
                    cands = ["low", "medium", "high"]
                    dec = self.decide(prompt, candidates=cands, mode="reflex")
                    probs = dec.get("probs", {"low": 0.33, "medium": 0.34, "high": 0.33})
                    if dec.get("action") == "ABSTAIN":
                        slot_outputs[slot.name] = {"probs": probs, "action": "ABSTAIN", "conf": 0.0}
                    else:
                        score_val = float(probs.get("low", 0.0) * 0.1 + probs.get("medium", 0.0) * 0.5 + probs.get("high", 0.0) * 0.9)
                        slot_outputs[slot.name] = {"probs": probs, "action": round(score_val, 4), "conf": score_val}
            return slot_outputs

        def extract_probs(out: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
            return {s_name: d["probs"] for s_name, d in out.items()}

        raw_first, is_multi, max_h1, stats_dict = self.h1_entropy_gate.schedule_execution(
            forward_fn=run_forward_pass,
            extract_slot_probs_fn=extract_probs,
        )

        slots_res = {}
        for slot in tmpl.slots:
            first_d = raw_first[slot.name]
            st = stats_dict.get(slot.name)
            slots_res[slot.name] = SlotResult(
                name=slot.name,
                slot_type=slot.slot_type,
                chosen_value=first_d["action"],
                confidence=first_d["conf"],
                probabilities=first_d["probs"],
                entropy=self.h1_entropy_gate.compute_h1_entropy(list(first_d["probs"].values())),
                mean_confidence=st.mean if st else first_d["conf"],
                std_error=st.std_dev if st else 0.0,
                multi_read=is_multi,
                reads_count=st.reads_count if st else 1,
            )

        latency = (time.perf_counter() - t0) * 1000.0
        return CanvasDecisionResult(
            slots=slots_res,
            is_multi_read=is_multi,
            max_entropy=max_h1,
            timing_ms=latency,
            prompt_tokens=len(prompt.split()),
        )




    def route_hdt(
        self,
        text: str,
        domain_taxonomy: Dict[str, List[str]],
        domain_descriptions: Optional[Dict[str, str]] = None,
        leaf_descriptions: Optional[Dict[str, str]] = None,
        decision_client: Optional[Any] = None,
    ) -> Any:
        """Executes two-stage Hierarchical Decision Tree routing (RFC-030)."""
        from .pipeline.hdt_router import HDTRouter
        router = HDTRouter(domain_taxonomy=domain_taxonomy, decision_client=decision_client)
        return router.classify(
            text=text,
            domain_descriptions=domain_descriptions,
            leaf_descriptions=leaf_descriptions,
        )

    def pre_aggregate_events(
        self,
        events: Sequence[Any],
    ) -> Any:
        """Pre-aggregates raw log/event records via in-memory SQLite stage (RFC-030)."""
        from .prefill.pre_aggregator import DeterministicPreAggregator
        aggregator = DeterministicPreAggregator()
        aggregator.ingest_batch(events)
        return aggregator.aggregate()

    def micro_audit_pr(
        self,
        diff_text: str,
    ) -> Any:
        """Evaluates 14 typed orthogonal audit probes against Git diff in sub-500ms (RFC-030)."""
        from .audit.pr_micro_audit import PRMicroAuditMatrix
        matrix = PRMicroAuditMatrix(decision_client=self)
        return matrix.audit_diff(diff_text)

    def evaluate_confidence_floor(
        self,
        decision_result: Dict[str, Any],
        tau_floor: float = 0.70,
        default_fallback_action: str = "ESCALATE_TO_HUMAN",
        fallback_rule_fn: Optional[Callable[[Any, Sequence[str]], Optional[str]]] = None,
        state: Optional[Any] = None,
        candidates: Optional[Sequence[str]] = None,
    ) -> Any:
        """Safeguards autonomous micro-core execution via rigid confidence floor cut-off (RFC-030)."""
        from .runtime.confidence_floor import ConfidenceFloorGate
        gate = ConfidenceFloorGate(
            tau_floor=tau_floor,
            default_fallback_action=default_fallback_action,
            fallback_rule_fn=fallback_rule_fn,
        )
        return gate.evaluate(decision_result=decision_result, state=state, candidates=candidates)

    def calibrate_multilingual_invariance(
        self,
        multilingual_prompts: Dict[str, str],
        candidates: Sequence[str],
        max_allowed_drift: float = 0.025,
        candidate_descriptions: Optional[Dict[str, str]] = None,
        decision_client: Optional[Any] = None,
    ) -> Any:
        """Verifies that decision micro-cores maintain logit invariance across multilingual variants (RFC-030)."""
        from .runtime.confidence_floor import MultilingualInvarianceCalibrator
        calibrator = MultilingualInvarianceCalibrator(
            max_allowed_drift=max_allowed_drift,
            decision_client=decision_client,
        )
        return calibrator.evaluate_invariance(
            multilingual_prompts=multilingual_prompts,
            candidates=candidates,
            candidate_descriptions=candidate_descriptions,
        )


    def imagine_world_model(
        self,
        state: Any,
        candidate_actions: List[str],
        horizon: int = 4,
        enforce_cpsat: bool = True,
        forbidden_actions: Optional[Set[str]] = None,
        safety_evaluator: Optional[Callable[[Any, str], float]] = None,
        critic_evaluator: Optional[Callable[[Any], float]] = None,
        constraints: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Any:
        """Executes counterfactual imagination search coordinating NanoCores via World Model (RFC-049 / Issue #51).

        ``constraints``: forbid / mutually_exclusive action constraints (see decide()); bounds are rejected here.
        """
        return self.world_model_orchestrator.imagine_and_orchestrate(
            state=state,
            candidate_actions=candidate_actions,
            horizon=horizon,
            enforce_cpsat=enforce_cpsat,
            forbidden_actions=forbidden_actions,
            safety_evaluator=safety_evaluator,
            critic_evaluator=critic_evaluator,
            constraints=constraints,
        )

    def encode_lossless_vector(
        self,
        state: Any,
        constraints: Optional[List[str]] = None,
        dim: int = 1024,
    ) -> Any:
        """Encodes state and boolean constraints into lossless invertible vector z (RFC-069)."""
        from .model.invertible_encoder import InvertibleVectorEncoder
        encoder = InvertibleVectorEncoder(dim=dim)
        return encoder.encode(state, constraints=constraints)

    def decode_lossless_vector(
        self,
        vector_output: Any,
    ) -> Dict[str, Any]:
        """Decodes lossless vector z back into exact original state and constraints (RFC-069)."""
        from .model.invertible_encoder import InvertibleVectorEncoder
        dim = len(vector_output.vector) if hasattr(vector_output, "vector") else 1024
        encoder = InvertibleVectorEncoder(dim=dim)
        res = encoder.decode(vector_output)
        sym = res.get("symbolic_state", {})
        if isinstance(sym, dict) and "constraints" in sym:
            constraints = sym.get("constraints", [])
            state = {k: v for k, v in sym.items() if k != "constraints"}
            if "state" in state and len(state) == 1:
                state = state["state"]
            return {"state": state, "constraints": constraints, **res}
        return {"state": sym, "constraints": [], **res}

    def save_nanocore_checkpoint(
        self,
        core: Any,
        path: str,
        compress: bool = True,
        compression_level: int = 3,
    ) -> Dict[str, Any]:
        """Persists NanoCore weights to disk using zstd compact compression (RFC-069)."""
        return core.save_checkpoint(path=path, compress=compress, compression_level=compression_level)

    def load_nanocore_checkpoint(
        self,
        path: str,
        verify_checksum: bool = True,
    ) -> Any:
        """Loads NanoCore checkpoint with transparent zstd decompression and SHA-256 verification (RFC-069)."""
        from .runtime.base_nano_core import BaseNanoCore
        return BaseNanoCore.load_checkpoint(path=path, verify_checksum=verify_checksum)

    def create_fleet_scheduler(
        self,
        max_resident_cores: int = 5,
        max_resident_bytes: int = 1024 * 1024 * 1024,
        storage_dir: Optional[str] = None,
        verify_checksum: bool = True,
        compression_level: int = 3,
    ) -> Any:
        """Instantiates a NanoCoreFleetScheduler for managing multi-microcore fleets with zstd hot-swapping (RFC-069 & Issue #73)."""
        from .nanocore.fleet_scheduler import NanoCoreFleetScheduler, FleetSchedulerConfig
        config = FleetSchedulerConfig(
            max_resident_cores=max_resident_cores,
            max_resident_bytes=max_resident_bytes,
            storage_dir=storage_dir,
            verify_checksum_on_load=verify_checksum,
            compression_level=compression_level,
        )
        return NanoCoreFleetScheduler(config)

    def create_constraint_compiler(
        self,
        latent_dim: int = 1024,
        seed: int = 42,
        hard_timeout_ms: float = 2.0,
    ) -> Any:
        """Instantiates a ConstraintLinearProjectionCompiler for Google OR-Tools CP-SAT (RFC-069 & Issue #76)."""
        from .gate.constraint_compiler import ConstraintLinearProjectionCompiler
        return ConstraintLinearProjectionCompiler(
            latent_dim=latent_dim,
            seed=seed,
            hard_timeout_ms=hard_timeout_ms,
        )

    def load_manifold_anchor_artifact(self, artifact_path: Union[str, Path], *, core_manifest=None) -> NanocoreAnchorBridge:
        """Loads and caches a fitted ManifoldAnchorDistiller artifact as a NanocoreAnchorBridge."""
        self.nanocore_anchor_bridge = NanocoreAnchorBridge(artifact_path, core_manifest=core_manifest)
        return self.nanocore_anchor_bridge

    def project_hidden_to_nanocore_state(self, raw_hidden: np.ndarray) -> List[float]:
        """Projects a raw hidden representation to the 128-d nanocore_state MCP expects."""
        if self.nanocore_anchor_bridge is None:
            raise RuntimeError(
                "no manifold anchor artifact loaded; call load_manifold_anchor_artifact(...) first"
            )
        return self.nanocore_anchor_bridge.project_to_nanocore_state(raw_hidden)

    def generate_nanocore_ask_payload(
        self, raw_hidden: np.ndarray, domain_id: int, candidates: List[str]
    ) -> Dict[str, Any]:
        """Builds the nanocore_ask MCP payload {"verb": "ask", "nanocore_domain", "nanocore_state", "candidates"}."""
        if self.nanocore_anchor_bridge is None:
            raise RuntimeError(
                "no manifold anchor artifact loaded; call load_manifold_anchor_artifact(...) first"
            )
        return self.nanocore_anchor_bridge.generate_mcp_ask_payload(raw_hidden, domain_id, candidates)

    def register_nanocore(
        self, domain_id: int, name: str = "general", core: Any = None, *, space_manifest: Optional[Dict[str, Any]] = None
    ) -> None:
        """Registers a micro-core instance in the client fleet registry.

        ``space_manifest``, when given, binds this domain's core to a specific
        anchor/GCCA/source identity: it must carry the full 7-key manifest
        {source_model, layer, norm, GCCA_map, anchor_basis, core, domain_id},
        ``manifest["core"]`` must equal the actual core's ``core_digest()``,
        and ``manifest["domain_id"]`` must equal ``domain_id``. A core with no
        ``core_digest()`` method cannot be bound. Registration WITHOUT a
        manifest is still legal (e.g. for tests exercising only the registry
        lifecycle), but ``decide_nanocore`` will then refuse to run for that
        domain until a matching manifest is bound, because it has no basis to
        prove the registered core matches whatever artifact the bridge holds.
        """
        if isinstance(domain_id, bool) or not isinstance(domain_id, (int, np.integer)) or not (0 <= domain_id <= 0xFFFF_FFFF):
            raise ValueError(f"domain_id must be a u32 integer, got {domain_id!r}")
        if core is None:
            core = getattr(self, "nanocore_choice_head", None)
        decide_fn = getattr(core, "decide", None)
        if not callable(decide_fn):
            raise TypeError(
                f"nanocore instance for domain {domain_id} must have a callable decide() method, got {type(core)!r}"
            )
        domain_id = int(domain_id)
        if space_manifest is not None:
            required = {"source_model", "layer", "norm", "GCCA_map", "anchor_basis", "core", "domain_id"}
            if not isinstance(space_manifest, dict) or set(space_manifest) != required:
                raise ValueError(f"space_manifest must be a dict with exactly keys {sorted(required)}")
            digest_fn = getattr(core, "core_digest", None)
            if not callable(digest_fn):
                raise ValueError(
                    f"nanocore instance for domain {domain_id} has no core_digest() method; "
                    "cores without a verifiable digest cannot be bound to a space_manifest"
                )
            if space_manifest["core"] != digest_fn():
                raise ValueError(f"space_manifest['core'] does not match the actual core's digest (domain {domain_id})")
            if space_manifest["domain_id"] != domain_id:
                raise ValueError(
                    f"space_manifest['domain_id'] = {space_manifest['domain_id']!r} does not match domain_id={domain_id}"
                )
            self.nanocore_space_manifests[domain_id] = dict(space_manifest)
        else:
            # A registration never inherits another core's provenance.
            self.nanocore_space_manifests.pop(domain_id, None)
        self.registered_nanocores[domain_id] = core

    def unregister_nanocore(self, domain_id: int) -> None:
        """Unregisters a micro-core instance (and any bound space manifest) from the client fleet registry."""
        domain_id = int(domain_id)
        self.registered_nanocores.pop(domain_id, None)
        self.nanocore_space_manifests.pop(domain_id, None)

    def decide_nanocore(
        self, raw_hidden: np.ndarray, domain_id: int, candidates: List[str]
    ) -> Dict[str, Any]:
        """Executes a NanoCore decision for a raw 8192-d representation.

        Projects the raw representation to the 128-d nanocore_state, scores the
        candidates on the Simplex ETF choice head, and runs the H1 entropy gate
        over the resulting distribution. An ambiguous distribution is reported
        as gate_status="gated_ambiguous" and logged; it is never relabelled.
        """
        if (
            isinstance(domain_id, bool)
            or not isinstance(domain_id, (int, np.integer))
            or not (0 <= domain_id <= 0xFFFF_FFFF)
        ):
            raise ValueError(
                f"domain_id must be an integer in [0, 4294967295] (a u32 domain ID), got {domain_id!r}"
            )
        registered = getattr(self, "registered_nanocores", None)
        if not registered or domain_id not in registered:
            raise ValueError(
                f"micro-core unavailable (domain {domain_id}): domain is not registered in fleet"
            )
        # A bare str is iterable and would silently split into characters.
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, (list, tuple)):
            raise ValueError("candidates must be a list or tuple of strings")
        if not (1 <= len(candidates) <= 16):
            raise ValueError(f"candidates must contain 1..16 entries, got {len(candidates)}")
        if not all(isinstance(c, str) and c.strip() for c in candidates):
            raise ValueError("candidates must all be non-blank strings")
        # Duplicates would collapse in the head's name-keyed probability dict.
        if len(set(candidates)) != len(candidates):
            raise ValueError(f"candidates must be unique, got {list(candidates)}")
        candidates = list(candidates)

        core = registered[domain_id]
        bridge = self.nanocore_anchor_bridge
        if bridge is None:
            raise RuntimeError(
                "no manifold anchor artifact loaded; call load_manifold_anchor_artifact(...) first"
            )
        bound_manifest = self.nanocore_space_manifests.get(domain_id)
        if bound_manifest != bridge.space:
            raise ValueError(
                f"registered core space manifest mismatch (domain {domain_id}): the manifest bound via "
                "register_nanocore(..., space_manifest=...) does not match the currently loaded bridge's "
                "space; register this domain with a manifest matching the loaded artifact before deciding"
            )
        state_128 = self.generate_nanocore_ask_payload(raw_hidden, domain_id, candidates)["nanocore_state"]
        decide_fn = getattr(core, "decide", None)
        if callable(decide_fn):
            choice_res = decide_fn(state_128, candidates)
        else:
            raise TypeError(
                f"nanocore instance for domain {domain_id} must have a callable decide() method, got {type(core)!r}"
            )
        entropy_eval = self.h1_entropy_gate.evaluate_distribution(choice_res.action_probabilities)
        gate_status = "passed" if not entropy_eval.is_ambiguous else "gated_ambiguous"
        if entropy_eval.is_ambiguous:
            log = logging.getLogger(__name__)
            if entropy_eval.is_valid:
                log.warning(
                    "nanocore decision gated as ambiguous: domain=%d H1=%.4f > tau=%.4f, chosen=%r",
                    domain_id, entropy_eval.entropy, entropy_eval.threshold, choice_res.selected_action,
                )
            else:
                # is_valid=False means the choice head handed back an empty,
                # all-zero, or non-finite distribution: not genuine real-valued
                # ambiguity but a broken/degenerate head output. Escalate loudly
                # rather than logging it identically to ordinary ambiguity.
                log.error(
                    "nanocore decision gated as ambiguous on a DEGENERATE distribution "
                    "(empty/all-zero/non-finite, H1=%s): domain=%d chosen=%r — this "
                    "indicates a broken choice head, not ordinary ambiguity.",
                    entropy_eval.entropy, domain_id, choice_res.selected_action,
                )

        return {
            "engine": "nanocore",
            "domain": int(domain_id),
            "chosen_action": choice_res.selected_action,
            "confidence": float(choice_res.confidence),
            "action_probabilities": {k: float(v) for k, v in choice_res.action_probabilities.items()},
            "action_logits": {k: float(v) for k, v in choice_res.action_logits.items()},
            "attention_entropy": float(choice_res.attention_entropy),
            "snr": float(choice_res.snr),
            "is_equiangular": bool(choice_res.is_equiangular),
            "nanocore_state": state_128,
            "candidates": list(candidates),
            "gate_status": gate_status,
        }


# Public Client Alias
GenZeroClient = GenZero




