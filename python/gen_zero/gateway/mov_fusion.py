"""Mixture of Vectors (MoV) Composite Decision Layer & Fallback Watchdog.

Implements Milestone 4 of Issue #23:
- RMSNorm Geometric Normalization: Eliminates magnitude dominance between micro-cores.
- Gated Vector Field Fusion: Computes domain prototype gating coefficients in latent space.
- Bayesian Confidence-Driven Pooling: Aggregates probabilities weighted by closed-form confidence C_k(s).
- Fallback Watchdog: Intercepts unconfident or negative-value composites and escalates to full-parameter model.
"""

from typing import List, Dict, Any, Mapping, Optional, Sequence, Tuple
import dataclasses
import math
import numpy as np

from gen_zero.nanocore.hierarchical_simplex import HierarchicalRoute, HierarchicalSimplexRouter


# The gateway keeps the scenario vocabulary in one place.  The values are
# metadata, rather than vectors or labels: the vectors are constructed from the
# hierarchical Simplex router below.  This makes the geometry independent of
# process-local random state and keeps the registry useful to callers that only
# need to inspect capabilities.
SCENARIO_DOMAINS: Dict[str, Dict[str, Any]] = {
    "DOM": {
        "id": "general_interaction",
        "display_name": "General interaction",
        "description": "General user and environment interaction decisions.",
        "default_intent": "DOM_default",
        "intents": ("DOM_default",),
        "risk_level": "medium",
        "capabilities": ("interaction", "state_transition"),
    },
    "Code": {
        "id": "software_engineering",
        "display_name": "Code",
        "description": "Software construction, debugging, and code-operation decisions.",
        "default_intent": "Code_default",
        "intents": ("Code_default",),
        "risk_level": "medium",
        "capabilities": ("programming", "debugging", "verification"),
    },
    "Ops": {
        "id": "operations",
        "display_name": "Operations",
        "description": "Operational execution, deployment, and infrastructure decisions.",
        "default_intent": "Ops_default",
        "intents": ("Ops_default",),
        "risk_level": "high",
        "capabilities": ("operations", "deployment", "rollback"),
    },
    "Market": {
        "id": "market",
        "display_name": "Market",
        "description": "Market-state and trading-oriented decision support.",
        "default_intent": "Market_default",
        "intents": ("Market_default",),
        "risk_level": "high",
        "capabilities": ("market_state", "portfolio", "risk_awareness"),
    },
    "Finance": {
        "id": "finance",
        "display_name": "Finance",
        "description": "Financial planning, accounting, and risk decisions.",
        "default_intent": "Finance_default",
        "intents": ("Finance_default",),
        "risk_level": "high",
        "capabilities": ("financial_analysis", "risk_assessment", "compliance"),
    },
    "Safety": {
        "id": "safety",
        "display_name": "Safety",
        "description": "Safety, security, and policy-constrained decisions.",
        "default_intent": "Safety_default",
        "intents": ("Safety_default",),
        "risk_level": "critical",
        "capabilities": ("hazard_screening", "policy", "fail_closed"),
    },
    "CausalReasoning": {
        "id": "causal_reasoning",
        "display_name": "Causal reasoning",
        "description": "Intervention, counterfactual, and causal attribution decisions.",
        "default_intent": "CausalReasoning_default",
        "intents": ("CausalReasoning_default",),
        "risk_level": "high",
        "capabilities": ("intervention", "counterfactual", "attribution"),
    },
    "LogicDecision": {
        "id": "logic_decision",
        "display_name": "Logical decision",
        "description": "Constraint, formal logic, and structured decision problems.",
        "default_intent": "LogicDecision_default",
        "intents": ("LogicDecision_default",),
        "risk_level": "high",
        "capabilities": ("constraints", "formal_logic", "decision_procedure"),
    },
}


def _taxonomy_from_metadata(
    metadata: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Tuple[str, ...]]:
    """Build a router taxonomy from validated scenario metadata."""
    taxonomy: Dict[str, Tuple[str, ...]] = {}
    for domain, info in metadata.items():
        raw_intents = info.get("intents")
        if raw_intents is None:
            default_intent = info.get("default_intent", f"{domain}_default")
            raw_intents = (default_intent,)
        intents = tuple(str(intent) for intent in raw_intents)
        if not intents or any(not intent for intent in intents):
            raise ValueError(f"scenario {domain!r} must define at least one non-empty intent")
        taxonomy[str(domain)] = intents
    if not taxonomy:
        raise ValueError("domain taxonomy must contain at least one scenario")
    return taxonomy


DEFAULT_SCENARIO_TAXONOMY = _taxonomy_from_metadata(SCENARIO_DOMAINS)

__all__ = [
    "SCENARIO_DOMAINS",
    "DEFAULT_SCENARIO_TAXONOMY",
    "rms_norm",
    "MicroCoreOutput",
    "CompositeMoVDecision",
    "MoVDecisionLayer",
]


def rms_norm(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Applies Root Mean Square Normalization (RMSNorm) to enforce unit hypersphere geometry."""
    v = np.asarray(vector, dtype=np.float32)
    rms = np.sqrt(np.mean(v ** 2, axis=-1, keepdims=True) + eps)
    return v / rms


@dataclasses.dataclass
class MicroCoreOutput:
    core_id: str
    domain: str
    action_probabilities: Dict[str, float]
    closed_form_confidence: float
    feature_vector: np.ndarray
    expected_value: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "core_id": self.core_id,
            "domain": self.domain,
            "closed_form_confidence": round(self.closed_form_confidence, 4),
            "expected_value": round(self.expected_value, 4),
            "action_probabilities": {k: round(v, 4) for k, v in self.action_probabilities.items()},
        }


@dataclasses.dataclass
class CompositeMoVDecision:
    best_action: str
    composite_probabilities: Dict[str, float]
    composite_confidence: float
    composite_expected_value: float
    gating_weights: Dict[str, float]
    fallback_escalated: bool
    escalation_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_action": self.best_action,
            "composite_probabilities": {k: round(v, 4) for k, v in self.composite_probabilities.items()},
            "composite_confidence": round(self.composite_confidence, 4),
            "composite_expected_value": round(self.composite_expected_value, 4),
            "gating_weights": {k: round(v, 4) for k, v in self.gating_weights.items()},
            "fallback_escalated": self.fallback_escalated,
            "escalation_reason": self.escalation_reason,
        }


class MoVDecisionLayer:
    """Mixture of Vectors (MoV) Fusion Layer with RMSNorm Geometry and Watchdog."""

    def __init__(
        self,
        dim: int = 1024,
        confidence_floor: float = 0.30,
        expected_value_floor: float = -0.20,
        domain_taxonomy: Optional[Mapping[str, Sequence[str]]] = None,
    ):
        if int(dim) != dim or int(dim) <= 0:
            raise ValueError("dim must be a positive integer")
        self.dim = int(dim)
        self.confidence_floor = confidence_floor
        self.expected_value_floor = expected_value_floor

        # Construct the router first.  Its Helmert ETF is deterministic and
        # provides the canonical anchor direction for every domain prototype.
        # A caller-supplied taxonomy is intentionally independent from the
        # metadata registry, which is useful for small isolated deployments.
        taxonomy: Mapping[str, Sequence[str]]
        if domain_taxonomy is None:
            taxonomy = DEFAULT_SCENARIO_TAXONOMY
        else:
            taxonomy = domain_taxonomy
        normalized_taxonomy: Dict[str, Tuple[str, ...]] = {}
        for domain, intents in taxonomy.items():
            # Accept metadata records as a convenience while retaining the
            # public Mapping[str, Sequence[str]] contract.
            if isinstance(intents, Mapping):
                raw_intents = intents.get("intents")
                if raw_intents is None:
                    raw_intents = (intents.get("default_intent", f"{domain}_default"),)
            else:
                raw_intents = intents
            normalized_taxonomy[str(domain)] = tuple(str(intent) for intent in raw_intents)
        self.simplex_router = HierarchicalSimplexRouter(normalized_taxonomy, dim=self.dim)
        self.domain_taxonomy = self.simplex_router.intents

        self.domain_prototypes: Dict[str, np.ndarray] = {
            domain: rms_norm(self.simplex_router.stage1_anchor(domain))
            for domain in self.simplex_router.scenarios
        }

    def compute_gating_weights(
        self,
        global_state_vector: np.ndarray,
        active_domains: List[str],
    ) -> Dict[str, float]:
        """Compute a posterior over the active domains from stage-1 routing.

        The router scores all registered scenarios.  MoV only needs the
        posterior mass for domains represented by the current micro-cores, so
        those masses are renormalized over the active set.  An unknown domain
        is a taxonomy/configuration error and is rejected explicitly.
        """
        state = np.asarray(global_state_vector, dtype=np.float64)
        if state.shape != (self.dim,):
            raise ValueError(f"expected global state shape ({self.dim},), got {state.shape}")
        if not np.all(np.isfinite(state)):
            raise ValueError("global state contains NaN or inf")
        if not active_domains:
            return {}

        # Preserve first-seen order while preventing duplicate core domains
        # from changing the posterior itself.
        domains = list(dict.fromkeys(str(domain) for domain in active_domains))
        unknown = [domain for domain in domains if domain not in self.domain_prototypes]
        if unknown:
            raise ValueError(f"unknown active domain(s): {unknown}")
        stage1 = self.simplex_router.scenario_probs(state)
        posterior = {
            domain: float(stage1[index])
            for index, domain in enumerate(self.simplex_router.scenarios)
            if domain in domains
        }
        total = sum(posterior.values())
        if total <= 0.0 or not math.isfinite(total):
            raise ValueError("active domain posterior is non-finite or has zero mass")
        return {domain: float(posterior.get(domain, 0.0) / total) for domain in domains}

    def route_hierarchical(self, global_state_vector: np.ndarray) -> HierarchicalRoute:
        """Route a state through the scenario and intent Simplex frames."""
        return self.simplex_router.route(global_state_vector)

    def fuse_hierarchical(
        self,
        global_state: np.ndarray,
        core_outputs: List[MicroCoreOutput],
        candidates: List[str],
    ) -> CompositeMoVDecision:
        """Fuse outputs after validating the same hierarchical route used for gating."""
        # Route first so malformed states fail before any probability pooling.
        self.route_hierarchical(global_state)
        return self.fuse_decisions(global_state, core_outputs, candidates)

    def fuse_decisions(
        self,
        global_state: np.ndarray,
        core_outputs: List[MicroCoreOutput],
        candidates: List[str],
    ) -> CompositeMoVDecision:
        """Fuses multi-core outputs using RMSNorm normalized vector pooling and Bayesian weighting."""
        if not core_outputs or not candidates:
            return CompositeMoVDecision(
                best_action="ABSTAIN",
                composite_probabilities={c: 1.0 / max(1, len(candidates)) for c in candidates},
                composite_confidence=0.0,
                composite_expected_value=0.0,
                gating_weights={},
                fallback_escalated=True,
                escalation_reason="EMPTY_INPUTS",
            )

        # 1. Step 1: RMSNorm input vectors to eliminate magnitude dominance
        normalized_features = []
        for out in core_outputs:
            norm_v = rms_norm(out.feature_vector)
            normalized_features.append(norm_v)

        # 2. Step 2: Compute Gating Weights
        domains = [out.domain for out in core_outputs]
        gating_map = self.compute_gating_weights(global_state, domains)

        # 3. Step 3: Bayesian Confidence-Weighted Log-Probability Pooling
        # w_k = (g_k * C_k(s)) / sum(g_j * C_j(s))
        raw_weights = [
            gating_map.get(out.domain, 1.0 / len(core_outputs)) * max(1e-4, out.closed_form_confidence)
            for out in core_outputs
        ]
        sum_w = sum(raw_weights) + 1e-8
        norm_weights = [w / sum_w for w in raw_weights]

        # Log-probability pooling: log P_comp(a) = sum w_k * log P_k(a)
        log_composite = {c: 0.0 for c in candidates}
        for w, out in zip(norm_weights, core_outputs):
            for c in candidates:
                p_kc = max(1e-6, out.action_probabilities.get(c, 0.0))
                log_composite[c] += w * math.log(p_kc)

        # Exponentiate and normalize
        max_log = max(log_composite.values())
        exp_comp = {c: math.exp(log_composite[c] - max_log) for c in candidates}
        sum_exp = sum(exp_comp.values()) + 1e-8
        composite_probs = {c: exp_comp[c] / sum_exp for c in candidates}

        # Argmax action and closed-form confidence
        best_action = max(candidates, key=lambda c: composite_probs[c])
        p_max = composite_probs[best_action]
        K = len(candidates)
        if K > 1:
            closed_form_conf = max(0.0, min(1.0, (p_max - 1.0 / K) / (1.0 - 1.0 / K)))
        else:
            closed_form_conf = 1.0

        # Weighted composite expected value
        comp_value = sum(w * out.expected_value for w, out in zip(norm_weights, core_outputs))

        # 4. Step 4: Fallback Watchdog Check
        fallback_escalated = False
        escalation_reason = None

        if closed_form_conf < self.confidence_floor:
            fallback_escalated = True
            escalation_reason = f"LOW_COMPOSITE_CONFIDENCE ({closed_form_conf:.3f} < {self.confidence_floor})"
        elif comp_value < self.expected_value_floor:
            fallback_escalated = True
            escalation_reason = f"NEGATIVE_COMPOSITE_VALUE ({comp_value:.3f} < {self.expected_value_floor})"

        return CompositeMoVDecision(
            best_action=best_action,
            composite_probabilities=composite_probs,
            composite_confidence=closed_form_conf,
            composite_expected_value=comp_value,
            gating_weights=gating_map,
            fallback_escalated=fallback_escalated,
            escalation_reason=escalation_reason,
        )
