"""World Model & NanoCore Specialized Micro-Cores Orchestration (Issue #51 & RFC-049).

Implements Master Orchestrator:
- WorldModelNanoCoreOrchestrator:
  1. Virtual Sandbox Supply: O(W) constant-memory Attention Sinks rolling KV cache + latent dynamics z_{t+k} = z_{t+k-1} + Delta z.
  2. Virtual Node Dispatch: Dispatches Safety, Action, and Critic micro-cores concurrently across imagined state nodes.
  3. CP-SAT Formal Verification: 0-1 ILP hard constraint pruning over imagined action utilities.
     The ETF action head is reported as telemetry (CONVERGED / DIVERGENT / FAILED); it does not vote.
  4. Pearl Causal Shock Autonomic Adaptation: Measures ||z_real - z_pred|| and dynamically adapts NanoCore weights and horizon depth.

Fail-closed contract:
- No ``safety_evaluator`` means safety is UNVERIFIED: every candidate scores 0.0 and the
  orchestrator returns ``DEFAULT_FALLBACK_SAFE_ACTION``. There is no keyword heuristic.
- Every degradation is logged and listed in ``WorldModelOrchestrationResult.degradations``.
- ``cpsat_verified`` is True only when a CP-SAT solve really ran and returned a solution.
  Any solver fallback (OR-Tools missing, timeout, exception, no feasible action) leaves it
  False and appends ``CPSAT_NOT_VERIFIED:<solver_status>`` to ``degradations``.
- When every candidate is blocked the decision is ``SAFETY_INTERLOCKED``: the result carries
  ``DEFAULT_FALLBACK_SAFE_ACTION`` only because that action is a no-op outside the candidate
  set. If the fallback itself is forbidden, ``SafetyInterlockError`` is raised instead.
"""

from dataclasses import asdict, dataclass, field
import logging
import math
import os
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

from gen_zero.world_model.rolling_kv_cache import RollingVisionKVCache
from gen_zero.world_model.latent_dynamics import LatentTransitionModel, resolve_torch_device
from gen_zero.world_model.imagination_planner import ImaginationMCTSPlanner, LatentMCTSNode
from gen_zero.gate.cpsat_formal_solver import CPSATFormalSolver, CPSATVerdict
from gen_zero.gate.action_constraints import FV_NOT_REQUESTED
from gen_zero.gate.constraint_compiler import ConstraintLinearProjectionCompiler


DEFAULT_CAUSAL_SHOCK_THRESHOLD: float = 0.35  # Epsilon threshold for Pearl Causal Shock
DEFAULT_FALLBACK_SAFE_ACTION: str = "HOLD"
SAFETY_FLOOR: float = 0.50

# Degradation markers surfaced in WorldModelOrchestrationResult.degradations.
UNVERIFIED_SAFETY: str = "UNVERIFIED_SAFETY"
PSEUDO_EMBEDDING: str = "PSEUDO_EMBEDDING"
ACTION_HEAD_FAILED: str = "ACTION_HEAD_FAILED"
CAUSAL_SHOCK_UNAVAILABLE: str = "CAUSAL_SHOCK_UNAVAILABLE"
ALL_CANDIDATES_BLOCKED: str = "ALL_CANDIDATES_BLOCKED"
CPSAT_NOT_VERIFIED: str = "CPSAT_NOT_VERIFIED"

# Decision status values surfaced in WorldModelOrchestrationResult.decision_status.
DECISION_OK: str = "OK"
SAFETY_INTERLOCKED: str = "SAFETY_INTERLOCKED"

# Solver statuses that mean CP-SAT (or its exact single-candidate reduction) really solved the
# 0-1 ILP with OR-Tools present. Everything else is a fallback and is never "verified".
CPSAT_REAL_SOLVE_STATUSES: frozenset = frozenset({
    "OPTIMAL",                    # CPSATFormalSolver, OR-Tools OPTIMAL
    "FEASIBLE",                   # CPSATFormalSolver, OR-Tools FEASIBLE
    "DETERMINISTIC_SAFE_SOLVED",  # CPSATFormalSolver, OR-Tools present, one feasible candidate
    "CP_SAT_OPTIMAL",             # ConstraintLinearProjectionCompiler, OR-Tools solve
    "0_1_ILP_FEASIBLE",           # ConstraintLinearProjectionCompiler, OR-Tools present, one feasible
})


class SafetyInterlockError(RuntimeError):
    """Raised when no action, not even the designated safe fallback, may be released."""

    def __init__(self, message: str, degradations: List[str], cpsat_status: Optional[str] = None, selected_action: str = DEFAULT_FALLBACK_SAFE_ACTION):
        super().__init__(message)
        self.degradations = list(degradations)
        self.cpsat_status = cpsat_status
        self.selected_action = selected_action

logger = logging.getLogger("gen_zero.nanocore.world_model_orchestrator")


@dataclass
class NanoCoreClusterStatus:
    """Status telemetry across orchestrated NanoCores."""
    safety_core: str  # PASS, BLOCKED, MARGINAL, UNVERIFIED_SAFETY
    safety_probability: float
    action_core: str  # CONVERGED, DIVERGENT, FAILED
    critic_value: float
    cpsat_verified: bool  # True only when CP-SAT really solved (status in CPSAT_REAL_SOLVE_STATUSES).
    applied_constraints: List[str] = field(default_factory=list)
    safety_verified: bool = False
    cpsat_status: Optional[str] = None  # e.g. OPTIMAL, CP_SAT_OPTIMAL, ORTOOLS_UNAVAILABLE_FALLBACK, ALL_CANDIDATES_FORBIDDEN_INTERCEPT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "safety_core": self.safety_core,
            "safety_probability": round(self.safety_probability, 4),
            "safety_verified": self.safety_verified,
            "action_core": self.action_core,
            "critic_value": round(self.critic_value, 4),
            "cpsat_verified": self.cpsat_verified,
            "cpsat_status": self.cpsat_status,
            "applied_constraints": self.applied_constraints,
        }


@dataclass
class ImaginedStepTelemetry:
    """Telemetry of an individual lookahead step along the imagined tree."""
    step: int
    action: str
    predicted_risk: float
    predicted_value: float
    pruned: bool = False
    prune_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "predicted_risk": round(self.predicted_risk, 4),
            "predicted_value": round(self.predicted_value, 4),
            "pruned": self.pruned,
            "prune_reason": self.prune_reason,
        }


@dataclass
class WorldModelOrchestrationResult:
    """Comprehensive decision and causal telemetry emitted by the World Model Orchestrator."""
    selected_action: str
    horizon_explored: int
    trap_paths_pruned: int
    confidence: float
    expected_value: float
    imagined_trajectory: List[ImaginedStepTelemetry]
    nanocore_status: NanoCoreClusterStatus
    causal_shock: float
    shock_detected: bool
    adaptive_safety_mode: bool
    effective_weights: Dict[str, float]
    planning_time_ms: float
    decision_status: str = DECISION_OK  # DECISION_OK or SAFETY_INTERLOCKED
    formal_verification: str = FV_NOT_REQUESTED
    constraint_projection: Optional[Dict[str, Any]] = None
    degradations: List[str] = field(default_factory=list)

    @property
    def interlocked(self) -> bool:
        return self.decision_status == SAFETY_INTERLOCKED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_action": self.selected_action,
            "decision_status": self.decision_status,
            "horizon_explored": self.horizon_explored,
            "trap_paths_pruned": self.trap_paths_pruned,
            "confidence": round(self.confidence, 4),
            "expected_value": round(self.expected_value, 4),
            "imagined_trajectory": [step.to_dict() for step in self.imagined_trajectory],
            "nanocore_status": self.nanocore_status.to_dict(),
            # NaN means the shock measurement failed; emit null so the JSON stays standard.
            "causal_shock": round(self.causal_shock, 4) if math.isfinite(self.causal_shock) else None,
            "shock_detected": self.shock_detected,
            "adaptive_safety_mode": self.adaptive_safety_mode,
            "effective_weights": {k: round(v, 4) for k, v in self.effective_weights.items()},
            "planning_time_ms": round(self.planning_time_ms, 2),
            "degradations": list(self.degradations),
            "formal_verification": self.formal_verification,
            "constraint_projection": self.constraint_projection,
        }


class WorldModelNanoCoreOrchestrator:
    """Master Orchestrator: World Model coordinating specialized NanoCore micro-cores."""

    def __init__(
        self,
        latent_dim: int = 1024,
        action_dim: int = 32,
        num_sink_tokens: int = 4,
        window_size: int = 8,
        causal_shock_threshold: float = DEFAULT_CAUSAL_SHOCK_THRESHOLD,
        device: str = "cpu",
        fleet_scheduler: Optional[Any] = None,
    ):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.causal_shock_threshold = causal_shock_threshold
        self.device = resolve_torch_device(device)
        self.fleet_scheduler = fleet_scheduler

        # 1. Rolling Attention Sinks KV Cache O(K+W)
        self.kv_cache = RollingVisionKVCache(
            num_sink_tokens=num_sink_tokens,
            window_size=window_size,
            feature_dim=latent_dim,
            device=self.device
        )

        # 2. Residual Latent Dynamics
        self.transition_model = LatentTransitionModel(
            latent_dim=latent_dim,
            action_dim=action_dim,
            device=self.device
        )

        # 3. CP-SAT Solver & Constraint Compiler (Issue #76)
        cpsat_timeout = float(os.environ.get("GENZERO_CPSAT_TIMEOUT_MS", "50.0"))
        self.cpsat_solver = CPSATFormalSolver(hard_timeout_ms=cpsat_timeout)
        self.constraint_compiler = ConstraintLinearProjectionCompiler(latent_dim=latent_dim, hard_timeout_ms=cpsat_timeout)
        # X-C01: captured by compile_safety_rules, not read live from
        # constraint_compiler.schema_fingerprint at solve time -- see that method.
        self._constraint_schema_fingerprint: Optional[str] = None

        # 4. Simplex ETF Action Manifold Choice Head (Issue #72)
        from .choice_head import ActionETFChoiceHead
        self.etf_choice_head = ActionETFChoiceHead(
            hidden_dim=latent_dim,
            action_dim=action_dim,
            blend_alpha=0.85,
        )

        # NanoCore base weights
        self.base_weights = {
            "safety": 0.35,
            "action": 0.35,
            "critic": 0.30,
        }
        self.active_weights = dict(self.base_weights)

        # State tracking
        self.last_latent: Optional[Any] = None
        self.last_predicted_latent: Optional[Any] = None
        self.last_action: Optional[str] = None
        self.step_count = 0

    def reset(self) -> None:
        """Resets orchestrator session state."""
        self.kv_cache.reset()
        self.last_latent = None
        self.last_predicted_latent = None
        self.last_action = None
        self.step_count = 0
        self.active_weights = dict(self.base_weights)

    def _encode_to_latent(self, state: Any) -> Tuple[Any, bool]:
        """Maps a state into a latent vector.

        Returns ``(latent, is_pseudo)``. ``is_pseudo`` is True when the state had no numeric
        vector and fell through to the hash pseudo-embedding (see ``_pseudo_embed``).

        A raw numeric state (vector, tensor, list, tuple) shorter than ``latent_dim``
        raises ``ValueError`` rather than zero-padding (X-C02, applied at this
        production boundary too, not only inside the constraint compiler): the
        constraint compiler's schema binds specific variables to specific
        coordinates, and a silently-padded 0.0 in a coordinate the caller never
        actually supplied is indistinguishable from a real reading of zero --
        exactly the "y == 0 evaluates True on missing y" failure mode. A longer
        state is still truncated: that only drops information, it never
        fabricates it. ``imagine_and_orchestrate`` callers (including the
        ``imagine_world_model`` MCP tool) already catch ``ValueError`` and report
        it as a clean validation error.
        """
        if hasattr(state, "vector"):
            # InvertibleVectorOutput support (RFC-069)
            vec = state.vector
            if HAS_TORCH:
                if not isinstance(vec, torch.Tensor):
                    t = torch.tensor(vec, device=self.device, dtype=torch.float32).flatten()
                else:
                    t = vec.to(device=self.device, dtype=torch.float32).flatten()
                if len(t) < self.latent_dim:
                    raise ValueError(
                        f"state.vector has {len(t)} dims, fewer than latent_dim={self.latent_dim}; "
                        "a short vector cannot be zero-padded without fabricating readings for "
                        "the missing coordinates (X-C02)."
                    )
                elif len(t) > self.latent_dim:
                    t = t[:self.latent_dim]
                return t, False
            else:
                arr = np.array(vec, dtype=np.float32).flatten()
                if len(arr) < self.latent_dim:
                    raise ValueError(
                        f"state.vector has {len(arr)} dims, fewer than latent_dim={self.latent_dim}; "
                        "a short vector cannot be zero-padded without fabricating readings for "
                        "the missing coordinates (X-C02)."
                    )
                elif len(arr) > self.latent_dim:
                    arr = arr[:self.latent_dim]
                return arr, False
        if isinstance(state, dict):
            if "vector" in state:
                return self._encode_to_latent(state["vector"])
            if "lossless_vector" in state:
                return self._encode_to_latent(state["lossless_vector"])

        if HAS_TORCH and isinstance(state, torch.Tensor):
            t = state.to(device=self.device, dtype=torch.float32).flatten()
            if len(t) < self.latent_dim:
                raise ValueError(
                    f"state tensor has {len(t)} dims, fewer than latent_dim={self.latent_dim}; "
                    "a short vector cannot be zero-padded without fabricating readings for "
                    "the missing coordinates (X-C02)."
                )
            elif len(t) > self.latent_dim:
                t = t[:self.latent_dim]
            return t, False
        elif HAS_NUMPY and isinstance(state, np.ndarray):
            if state.size < self.latent_dim:
                raise ValueError(
                    f"state array has {state.size} elements, fewer than latent_dim={self.latent_dim}; "
                    "a short vector cannot be zero-padded without fabricating readings for "
                    "the missing coordinates (X-C02)."
                )
            if HAS_TORCH:
                return torch.tensor(state, device=self.device, dtype=torch.float32), False
            return state, False
        elif isinstance(state, (list, tuple)):
            arr = list(state)
            if len(arr) < self.latent_dim:
                raise ValueError(
                    f"state has {len(arr)} elements, fewer than latent_dim={self.latent_dim}; "
                    "a short vector cannot be zero-padded without fabricating readings for "
                    "the missing coordinates (X-C02)."
                )
            else:
                arr = arr[:self.latent_dim]
            if HAS_TORCH:
                return torch.tensor(arr, device=self.device, dtype=torch.float32), False
            elif HAS_NUMPY:
                return np.array(arr, dtype=np.float32), False
            return arr, False

        return self._pseudo_embed(state), True

    def _pseudo_embed(self, state: Any) -> Any:
        """UNCALIBRATED pseudo-embedding for states without a numeric vector (e.g. raw text).

        This is ``sin(hash(str(state)) * k)``: no encoder, no training, no semantics. Similar
        texts do NOT map to nearby vectors, and Python salts ``hash`` per process
        (PYTHONHASHSEED), so the same text gives a different vector in another process.
        Do not treat it as a self-supervised world-model latent. Callers must flag it.
        """
        logger.warning(
            "%s: state of type %s has no numeric vector; using uncalibrated hash/sin "
            "pseudo-embedding. Latent dynamics and causal shock on this state carry no meaning.",
            PSEUDO_EMBEDDING, type(state).__name__,
        )
        h = abs(hash(str(state)))
        vals = [(math.sin(h * (i + 1) * 0.1) * 0.5) for i in range(self.latent_dim)]
        if HAS_TORCH:
            return torch.tensor(vals, device=self.device, dtype=torch.float32)
        elif HAS_NUMPY:
            return np.array(vals, dtype=np.float32)
        return vals

    def imagine_and_orchestrate(
        self,
        state: Any,
        candidate_actions: List[str],
        horizon: int = 4,
        enforce_cpsat: bool = True,
        forbidden_actions: Optional[Set[str]] = None,
        safety_evaluator: Optional[Callable[[Any, str], float]] = None,
        critic_evaluator: Optional[Callable[[Any], float]] = None,
        preserve_entropy: bool = True,
        constraints: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> WorldModelOrchestrationResult:
        """Executes counterfactual imagination search, coordinating NanoCores across virtual nodes.

        Without ``safety_evaluator`` safety is UNVERIFIED: all candidates score 0.0, all are
        blocked, and ``DEFAULT_FALLBACK_SAFE_ACTION`` is returned with ``UNVERIFIED_SAFETY``
        in ``degradations``.

        ``constraints`` (forbid / mutually_exclusive dicts, see gen_zero/gate/action_constraints.py)
        are compiled per call and disable actions before the CP-SAT selection; mutex winners are
        the higher-scoring candidates. Probability bounds are rejected (ValueError): this path
        returns one action, not a distribution. Malformed specs raise before any state changes.
        """
        t0 = time.perf_counter()
        action_compiler = None
        if constraints is not None:
            action_compiler = ConstraintLinearProjectionCompiler(latent_dim=self.latent_dim)
            action_compiler.compile_action_constraints(constraints)
            # Dry run so unsupported spec kinds (bounds) fail before step_count / caches mutate.
            action_compiler.resolve_disabled_actions({a: 0.0 for a in candidate_actions})
        self.step_count += 1
        degradations: List[str] = []
        z_curr, is_pseudo = self._encode_to_latent(state)
        if is_pseudo:
            degradations.append(PSEUDO_EMBEDDING)

        # 1. Pearl Causal Shock Detection: ||z_real - z_pred||
        causal_shock = 0.0
        shock_detected = False
        if self.last_predicted_latent is not None:
            causal_shock = self._compute_latent_distance(z_curr, self.last_predicted_latent)
            if not math.isfinite(causal_shock):
                # Fail closed: an unmeasurable shock is treated as a shock.
                logger.error(
                    "%s: causal shock distance is %r; forcing adaptive safety mode.",
                    CAUSAL_SHOCK_UNAVAILABLE, causal_shock,
                )
                degradations.append(CAUSAL_SHOCK_UNAVAILABLE)
                shock_detected = True
            elif causal_shock > self.causal_shock_threshold:
                shock_detected = True

        # 2. Adaptive Reweighting based on Causal Shock
        effective_weights = dict(self.base_weights)
        adaptive_safety_mode = False
        effective_horizon = horizon

        if shock_detected:
            # Exogenous disturbance: boost safety, reduce horizon, enter defensive mode
            adaptive_safety_mode = True
            effective_weights["safety"] = 0.80
            effective_weights["action"] = 0.10
            effective_weights["critic"] = 0.10
            effective_horizon = max(2, horizon // 2)
        else:
            self.active_weights = dict(self.base_weights)

        # 3. Virtual Sandbox Expansion & NanoCore Virtual Node Dispatch
        # In the virtual imagination tree, evaluate candidates and roll forward
        trajectories: List[ImaginedStepTelemetry] = []
        action_scores: Dict[str, float] = {}
        safety_probs: Dict[str, float] = {}
        critic_vals: Dict[str, float] = {}
        trap_paths_pruned = 0

        if safety_evaluator is None:
            logger.warning(
                "%s: no safety_evaluator configured; every candidate scores safety 0.0 and "
                "the orchestrator returns %r.",
                UNVERIFIED_SAFETY, DEFAULT_FALLBACK_SAFE_ACTION,
            )
            degradations.append(UNVERIFIED_SAFETY)

        for act in candidate_actions:
            # Predict virtual next latent state: z' = z + Delta z
            next_z, step_r, _ = self.transition_model.step(z_curr, act, preserve_entropy=preserve_entropy)

            # A. Safety NanoCore on virtual node
            if safety_evaluator is not None:
                p_safe = self._checked_safety_probability(safety_evaluator(next_z, act), act)
            else:
                p_safe = 0.0

            safety_probs[act] = p_safe

            # B. Critic NanoCore on virtual node
            if critic_evaluator is not None:
                v_val = critic_evaluator(next_z)
            else:
                v_val = step_r

            critic_vals[act] = v_val

            # Prune trap path if below safety floor
            is_pruned = False
            prune_reason = None
            if p_safe < SAFETY_FLOOR:
                is_pruned = True
                prune_reason = f"Safety risk {1.0 - p_safe:.2f} exceeded threshold"
                trap_paths_pruned += 1

            # Multi-step lookahead rollout along unpruned branches
            cum_r = v_val
            curr_roll_z = next_z
            for h in range(1, effective_horizon):
                if is_pruned:
                    break
                curr_roll_z, sub_r, _ = self.transition_model.step(curr_roll_z, act, preserve_entropy=preserve_entropy)
                cum_r += (0.95 ** h) * sub_r

            action_scores[act] = -10.0 if is_pruned else cum_r
            trajectories.append(ImaginedStepTelemetry(
                step=1,
                action=act,
                predicted_risk=round(1.0 - p_safe, 4),
                predicted_value=round(cum_r, 4),
                pruned=is_pruned,
                prune_reason=prune_reason,
            ))

        # 4. Action Core: ETF head on the Simplex action manifold. Telemetry only; it does not vote.
        etf_selected: Optional[str] = None
        action_head_failed = False
        if candidate_actions:
            try:
                z_np = z_curr.detach().cpu().numpy() if hasattr(z_curr, "detach") else np.asarray(z_curr)
                etf_selected = self.etf_choice_head.decide(z_np, candidate_actions).selected_action
            except Exception:
                logger.error(
                    "%s: ETF action head raised; action_core marked FAILED.",
                    ACTION_HEAD_FAILED, exc_info=True,
                )
                degradations.append(ACTION_HEAD_FAILED)
                action_head_failed = True

        # 5. CP-SAT Formal Verification
        active_forbidden = set(forbidden_actions or set())
        for a, p in safety_probs.items():
            if p < SAFETY_FLOOR:
                active_forbidden.add(a)
        constraint_projection: Optional[Dict[str, Any]] = None
        constraint_disabled: List[str] = []
        if action_compiler is not None:
            constraint_disabled, mutex_dropped, unmatched = action_compiler.resolve_disabled_actions(
                action_scores, pre_disabled=active_forbidden
            )
            active_forbidden |= set(constraint_disabled)
            constraint_projection = {
                "status": "OK",
                "disabled": list(constraint_disabled),
                "mutex_dropped": mutex_dropped,
                "unmatched_actions": unmatched,
            }

        selected_action = DEFAULT_FALLBACK_SAFE_ACTION
        cpsat_verified = False
        cpsat_constraints: List[str] = []
        cpsat_status: Optional[str] = None
        cpsat_res: Optional[CPSATVerdict] = None

        if enforce_cpsat and candidate_actions:
            if self.constraint_compiler.num_rules > 0:
                # The compiler only knows its own rules; drop safety-pruned and caller-forbidden
                # actions first, or it can select them.
                permitted = {a: u for a, u in action_scores.items() if a not in active_forbidden}
                if not permitted:
                    # Nothing survived. Do not short-circuit to NO_CANDIDATES: run the compiled
                    # rules against the fallback itself, so a rule that forbids HOLD is caught.
                    permitted = {DEFAULT_FALLBACK_SAFE_ACTION: 0.0}
                cpsat_res = self.constraint_compiler.solve_safest_action(
                    candidate_utilities=permitted,
                    z_latent=z_curr,
                    fallback_safe_action=DEFAULT_FALLBACK_SAFE_ACTION,
                    schema_fingerprint=self._constraint_schema_fingerprint,
                )
            else:
                cpsat_res = self.cpsat_solver.solve_safest_optimal_action(
                    candidate_utilities=action_scores,
                    forbidden_actions=active_forbidden,
                    fallback_safe_action=DEFAULT_FALLBACK_SAFE_ACTION
                )
            selected_action = cpsat_res.selected_action
            cpsat_constraints = cpsat_res.applied_constraints
            cpsat_status = cpsat_res.solver_status
            if self.constraint_compiler.num_rules > 0 and DEFAULT_FALLBACK_SAFE_ACTION in permitted:
                if cpsat_res.solver_status == "ALL_FORBIDDEN_FAILSAFE":
                    active_forbidden.add(DEFAULT_FALLBACK_SAFE_ACTION)
        else:
            # Best valid unpruned action
            valid_actions = [a for a in candidate_actions if a not in active_forbidden]
            if valid_actions:
                selected_action = max(valid_actions, key=lambda a: action_scores.get(a, -100.0))

        # Decide all_blocked from the final outcome, after the constraint compiler and CP-SAT ran.
        # The compiled rules can forbid every candidate the safety evaluator let through; the
        # solver then returns the HOLD failsafe and a pre-solve check would still say OK.
        # A candidate was released only if the solver picked one that nothing forbade.
        released_candidate = selected_action in candidate_actions and selected_action not in active_forbidden
        all_blocked = bool(candidate_actions) and not released_candidate

        if cpsat_res is not None:
            # Verified means a real solve over real candidates: no fallback, a safe verdict, a
            # whitelisted status, and a candidate was released. ORTOOLS_UNAVAILABLE_FALLBACK,
            # timeouts, exceptions, intercepts and every all-blocked failsafe stay False.
            cpsat_verified = (
                not all_blocked
                and not cpsat_res.fallback_used
                and cpsat_res.is_safe
                and cpsat_res.solver_status in CPSAT_REAL_SOLVE_STATUSES
            )
            if not cpsat_verified:
                marker = f"{CPSAT_NOT_VERIFIED}:{cpsat_res.solver_status}"
                logger.warning(
                    "%s: CP-SAT did not verify (fallback_used=%s, is_safe=%s); selected %r.",
                    marker, cpsat_res.fallback_used, cpsat_res.is_safe, selected_action,
                )
                degradations.append(marker)

        if candidate_actions and all(a in active_forbidden for a in candidate_actions):
            if ALL_CANDIDATES_BLOCKED not in degradations:
                degradations.append(ALL_CANDIDATES_BLOCKED)

        # Fail closed. A CP-SAT verdict that is not safe (fallback forbidden by a compiled rule,
        # conflicting requirements, unsupported syntax) must never release an action.
        # However, ALL_CANDIDATES_FORBIDDEN_INTERCEPT / ALL_FORBIDDEN_FAILSAFE with a releasable fallback
        # should proceed to the all_blocked handler below to return a typed SAFETY_INTERLOCKED result.
        if cpsat_res is not None and not cpsat_res.is_safe:
            is_releasable_fallback = (
                cpsat_res.solver_status in ("ALL_CANDIDATES_FORBIDDEN_INTERCEPT", "ALL_FORBIDDEN_FAILSAFE")
                and selected_action == DEFAULT_FALLBACK_SAFE_ACTION
                and selected_action not in active_forbidden
            )
            if not is_releasable_fallback:
                raise SafetyInterlockError(
                    f"{SAFETY_INTERLOCKED}: CP-SAT verdict {cpsat_res.solver_status} is not safe; "
                    f"no action may be released.",
                    degradations=degradations,
                    cpsat_status=cpsat_status,
                    selected_action=selected_action,
                )

        # Defensive: a solver must never hand back a forbidden action while a candidate was free.
        # When every candidate was forbidden this falls through to the all_blocked branch below.
        filters_blocked_all = bool(candidate_actions) and all(a in active_forbidden for a in candidate_actions)
        if selected_action in active_forbidden and not filters_blocked_all:
            raise SafetyInterlockError(
                f"{SAFETY_INTERLOCKED}: solver returned forbidden action {selected_action!r}.",
                degradations=degradations,
                cpsat_status=cpsat_status,
            )

        decision_status = DECISION_OK
        if all_blocked:
            degradations.append(ALL_CANDIDATES_BLOCKED)
            decision_status = SAFETY_INTERLOCKED
            if selected_action != DEFAULT_FALLBACK_SAFE_ACTION or selected_action in active_forbidden:
                # The solver picked a blocked candidate, or the fallback itself is forbidden.
                # Either way nothing may be released.
                logger.error(
                    "%s: all %d candidates blocked and fallback %r is not releasable.",
                    SAFETY_INTERLOCKED, len(candidate_actions), selected_action,
                )
                raise SafetyInterlockError(
                    f"{SAFETY_INTERLOCKED}: all {len(candidate_actions)} candidates blocked and "
                    f"fallback {selected_action!r} is not releasable.",
                    degradations=degradations,
                    cpsat_status=cpsat_status,
                )
            logger.warning(
                "%s: all %d candidates blocked; decision is %s with no-op %r.",
                ALL_CANDIDATES_BLOCKED, len(candidate_actions), SAFETY_INTERLOCKED, selected_action,
            )

        formal_verification = FV_NOT_REQUESTED
        if action_compiler is not None:
            formal_verification = action_compiler.formal_verification_status(
                selected_action, candidate_actions, constraint_disabled
            )

        # Predict future latent anchor for the chosen action
        next_pred_z, _, _ = self.transition_model.step(z_curr, selected_action)
        self.last_latent = z_curr
        self.last_predicted_latent = next_pred_z
        self.last_action = selected_action

        # Append to Rolling Attention Sinks KV Cache
        self.kv_cache.append(key=z_curr, value=z_curr, timestamp=time.time())

        # Compile telemetry. An action with no safety score has no verified safety: 0.0.
        safety_verified = safety_evaluator is not None
        safe_prob = safety_probs.get(selected_action, 0.0)
        if not safety_verified:
            safety_status = UNVERIFIED_SAFETY
        else:
            safety_status = "PASS" if safe_prob >= 0.80 else ("MARGINAL" if safe_prob >= SAFETY_FLOOR else "BLOCKED")

        if action_head_failed:
            action_core = "FAILED"
        elif etf_selected is not None and etf_selected == selected_action:
            action_core = "CONVERGED"
        else:
            action_core = "DIVERGENT"

        nanocore_status = NanoCoreClusterStatus(
            safety_core=f"{safety_status} ({safe_prob:.3f})",
            safety_probability=safe_prob,
            action_core=action_core,
            critic_value=critic_vals.get(selected_action, 0.0),
            cpsat_verified=cpsat_verified,
            applied_constraints=cpsat_constraints,
            safety_verified=safety_verified,
            cpsat_status=cpsat_status,
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return WorldModelOrchestrationResult(
            selected_action=selected_action,
            horizon_explored=effective_horizon,
            trap_paths_pruned=trap_paths_pruned,
            confidence=round(safe_prob, 4),
            expected_value=round(action_scores.get(selected_action, 0.0), 4),
            imagined_trajectory=trajectories,
            nanocore_status=nanocore_status,
            causal_shock=round(causal_shock, 4) if math.isfinite(causal_shock) else causal_shock,
            shock_detected=shock_detected,
            adaptive_safety_mode=adaptive_safety_mode,
            effective_weights=effective_weights,
            planning_time_ms=latency_ms,
            degradations=degradations,
            decision_status=decision_status,
            formal_verification=formal_verification,
            constraint_projection=constraint_projection,
        )

    @staticmethod
    def _checked_safety_probability(raw: Any, action: str) -> float:
        """Rejects evaluator output that is not a probability.

        NaN would slip past ``p < SAFETY_FLOOR`` and count as safe, so it must fail loudly.
        """
        try:
            p = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"safety_evaluator returned non-numeric {raw!r} for action {action!r}") from exc
        if not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise ValueError(f"safety_evaluator returned {p!r} for action {action!r}; expected a probability in [0, 1]")
        return p

    def _compute_latent_distance(self, z1: Any, z2: Any) -> float:
        """Computes Euclidean distance ||z1 - z2|| between latent states."""
        def to_numpy(z):
            if HAS_TORCH and isinstance(z, torch.Tensor):
                return z.detach().cpu().numpy().astype(np.float32).flatten()
            elif HAS_NUMPY and isinstance(z, np.ndarray):
                return z.astype(np.float32).flatten()
            return np.array(z, dtype=np.float32).flatten()

        try:
            arr1 = to_numpy(z1)
            arr2 = to_numpy(z2)
        except Exception:
            logger.error("%s: latent conversion failed; returning NaN.", CAUSAL_SHOCK_UNAVAILABLE, exc_info=True)
            return float("nan")
        if len(arr1) != len(arr2):
            logger.error(
                "%s: latent length mismatch %d vs %d; returning NaN.",
                CAUSAL_SHOCK_UNAVAILABLE, len(arr1), len(arr2),
            )
            return float("nan")
        return float(np.linalg.norm(arr1 - arr2))

    def compile_safety_rules(self, rules: Sequence[str]) -> Any:
        """Compiles declarative natural language and AST rules into CP-SAT 0-1 constraints.

        X-C01: captures the resulting schema_fingerprint into an orchestrator-owned
        field, rather than letting ``step()`` read ``constraint_compiler.schema_fingerprint``
        live. Reading it live would make the schema check a tautology (it could
        never fail); capturing it here means a *later*, out-of-band
        ``compile_safety_rules`` / ``compile_rules`` call that shifts the
        variable->coordinate map is caught the next time ``step()`` runs.
        """
        report = self.constraint_compiler.compile_rules(rules)
        self._constraint_schema_fingerprint = report.schema_version
        return report

    step = imagine_and_orchestrate
