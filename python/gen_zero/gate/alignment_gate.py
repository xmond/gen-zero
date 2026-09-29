"""State Entity Alignment Gate with Co-Riding Attributive Probes and Asymmetric Defense.

Implements Milestones 2, 3, and 4 of Issue #25:
- Module 2: Zero-Tuned Tri-State Semantic Rubric:
  Action = OUTCOME[min(floor(Score + 0.5), 2)]
  0: DROP (distinct entities)
  1: BUFFER (suspect/variant with local conflict -> isolation buffer queue)
  2: MERGE (confirmed identical -> automatic coalescence)
- Module 3: Entity & State Fingerprinting (extract_entity_fingerprint):
  Extracts schema keys, canonicalized key-value attributes, and computes SHA-256 fingerprint.
- Module 4: Asymmetric Mistake Cost Defense:
  Mistake cost of False Positive (false merge) >> False Negative.
  Borderline / ambiguous states 100% converge into Level 1 buffer zone without false merges.
- CP-SAT Formal Interlock:
  Hard security constraints (is_read_only_source, affects_production_db) trigger CP-SAT
  formal solver veto, enforcing deterministic downgrade of high-risk merges.
- Verifier Feedback Integration (AlignmentDiagnosisPacket):
  Packs conflict fields, differing attributes, and repair strategies for Verifier / Self-Healing loops.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Set
import dataclasses
import enum
import time
import math
import hashlib
import json

from gen_zero.service.co_riding_adapter import (
    CoRidingAlignmentRequest,
    CoRidingAlignmentResponse,
    CoRidingProbesAdapter,
    NoulProbeSpec,
    NoulProbeResult,
    DEFAULT_PROBES,
)
from gen_zero.gate.cpsat_formal_solver import CPSATFormalSolver, CPSATVerdict


class AlignmentLevel(enum.IntEnum):
    """Zero-Tuned Tri-State Levels."""
    DIFFERENT = 0       # Completely disjoint / negative -> DROP
    SUSPECT_BUFFER = 1  # High affinity with local conflict / ambiguous -> BUFFER
    CONFIRMED_SAME = 2  # Conclusively identical -> MERGE


class AlignmentAction(str, enum.Enum):
    DROP = "DROP"
    BUFFER = "BUFFER"
    MERGE = "MERGE"


@dataclasses.dataclass
class EntityFingerprint:
    """Compact structured fingerprint for state entities, variables, or file handles."""
    entity_id: Optional[str]
    entity_type: str
    schema_keys: List[str]
    attributes: Dict[str, Any]
    hash_digest: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "schema_keys": self.schema_keys,
            "attributes": self.attributes,
            "hash_digest": self.hash_digest,
        }


@dataclasses.dataclass
class AlignmentDiagnosisPacket:
    """Structured diagnostic packet dispatched to Issue #13 Verifier for targeted repair."""
    entity_a_id: str
    entity_b_id: str
    conflict_probes: Dict[str, float]
    differing_fields: Dict[str, Tuple[Any, Any]]
    suggested_repair_strategy: str
    target_patch_hint: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_a_id": self.entity_a_id,
            "entity_b_id": self.entity_b_id,
            "conflict_probes": {k: round(v, 4) for k, v in self.conflict_probes.items()},
            "differing_fields": {k: list(v) for k, v in self.differing_fields.items()},
            "suggested_repair_strategy": self.suggested_repair_strategy,
            "target_patch_hint": self.target_patch_hint,
        }


@dataclasses.dataclass
class AlignmentVerdict:
    """Comprehensive verdict returned by StateAlignmentGate."""
    level: AlignmentLevel
    action: AlignmentAction
    macro_score: float
    confidence: float
    probe_verdicts: Dict[str, float]
    conflict_fields: Dict[str, float]
    interlocked_by_cpsat: bool
    cpsat_reason: Optional[str]
    diagnosis_packet: Optional[AlignmentDiagnosisPacket]
    evaluation_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": int(self.level),
            "action": self.action.value,
            "macro_score": round(self.macro_score, 4),
            "confidence": round(self.confidence, 4),
            "probe_verdicts": {k: round(v, 4) for k, v in self.probe_verdicts.items()},
            "conflict_fields": {k: round(v, 4) for k, v in self.conflict_fields.items()},
            "interlocked_by_cpsat": self.interlocked_by_cpsat,
            "cpsat_reason": self.cpsat_reason,
            "diagnosis_packet": self.diagnosis_packet.to_dict() if self.diagnosis_packet else None,
            "evaluation_time_ms": round(self.evaluation_time_ms, 2),
        }


def extract_entity_fingerprint(entity: Union[Dict[str, Any], str, Any]) -> EntityFingerprint:
    """Extracts compact schema keys, canonicalized attributes, and SHA-256 fingerprint."""
    if isinstance(entity, dict):
        attrs = dict(entity)
        ent_id = str(attrs.get("id") or attrs.get("identifier") or attrs.get("name") or attrs.get("key") or "")
        ent_type = str(attrs.get("type") or attrs.get("category") or "generic_entity")
        keys = sorted(list(attrs.keys()))
    elif hasattr(entity, "__dict__"):
        attrs = dict(entity.__dict__)
        ent_id = str(attrs.get("id") or attrs.get("identifier") or attrs.get("name") or attrs.get("key") or "")
        ent_type = entity.__class__.__name__
        keys = sorted(list(attrs.keys()))
    else:
        raw = str(entity)
        attrs = {"raw": raw}
        ent_id = raw[:32]
        ent_type = "string_entity"
        keys = ["raw"]

    # Canonical JSON string for hash digest
    canonical_repr = json.dumps(attrs, sort_keys=True, default=str)
    digest = hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()[:16]

    return EntityFingerprint(
        entity_id=ent_id if ent_id else digest,
        entity_type=ent_type,
        schema_keys=keys,
        attributes=attrs,
        hash_digest=digest,
    )


class StateAlignmentGate:
    """Zero-Tuned Tri-State Alignment Gate with Asymmetric Defense and CP-SAT Safety Interlock."""

    def __init__(
        self,
        adapter: Optional[CoRidingProbesAdapter] = None,
        cpsat_solver: Optional[CPSATFormalSolver] = None,
        asymmetric_fp_penalty_weight: float = 2.5,
    ):
        self.adapter = adapter or CoRidingProbesAdapter()
        self.cpsat_solver = cpsat_solver or CPSATFormalSolver(hard_timeout_ms=2.0)
        self.asymmetric_fp_penalty_weight = asymmetric_fp_penalty_weight

    def evaluate_alignment(
        self,
        entity_a: Union[Dict[str, Any], str, Any],
        entity_b: Union[Dict[str, Any], str, Any],
        context: Optional[Union[Dict[str, Any], str]] = None,
        probes: Optional[List[NoulProbeSpec]] = None,
        hard_safety_constraints: Optional[Dict[str, bool]] = None,
    ) -> AlignmentVerdict:
        """Evaluates entity alignment using co-riding probes, asymmetric defense, and CP-SAT interlock."""
        t0 = time.perf_counter()

        # Step 1: Extract Fingerprints
        fp_a = extract_entity_fingerprint(entity_a)
        fp_b = extract_entity_fingerprint(entity_b)

        # Step 2: Co-Riding Assessment
        request = CoRidingAlignmentRequest(
            entity_a=fp_a.attributes,
            entity_b=fp_b.attributes,
            context=context,
            probes=probes,
        )
        response: CoRidingAlignmentResponse = self.adapter.evaluate_co_riding(request)

        macro_score = response.macro_score
        conflict_fields = response.conflict_fields
        probe_probs = {k: v.probability for k, v in response.probes.items()}

        # Step 3: Natural Rounding Level Determination
        # Baseline level = min(2, max(0, floor(macro_score + 0.5)))
        raw_level_idx = min(2, max(0, int(math.floor(macro_score + 0.5))))

        # Step 4: Asymmetric Mistake Cost Defense
        # Consistency conflicts (excluding hard safety probes handled by CP-SAT)
        consistency_conflicts = [
            k for k in conflict_fields
            if k not in ["is_read_only_source", "affects_production_db"]
        ]
        downgraded_by_asymmetry = False
        if raw_level_idx == 2:
            id_prob = probe_probs.get("same_identifier", 1.0)
            schema_prob = probe_probs.get("same_schema", 1.0)
            sem_prob = probe_probs.get("compatible_semantics", 1.0)

            # Strict identity check for MERGE
            if id_prob < 0.65 or schema_prob < 0.50 or sem_prob < 0.45 or len(consistency_conflicts) > 0:
                raw_level_idx = 1
                downgraded_by_asymmetry = True

        # In borderline ambiguous range [0.40, 0.60] (near DROP vs BUFFER), converge to BUFFER
        if raw_level_idx == 0 and macro_score >= 0.38 and len(fp_a.schema_keys) > 0:
            if fp_a.entity_type == fp_b.entity_type:
                raw_level_idx = 1

        # Step 5: CP-SAT Formal Interlock for Hard Safety Invariants
        interlocked_by_cpsat = False
        cpsat_reason = None

        ro_prob = probe_probs.get("is_read_only_source", 0.0)
        prod_prob = probe_probs.get("affects_production_db", 0.0)

        # Merge is considered unsafe if read-only source or production db is affected
        is_unsafe_merge = (ro_prob > 0.15) or (prod_prob > 0.15)

        if hard_safety_constraints:
            for k, val in hard_safety_constraints.items():
                if val:  # Constraint is active
                    is_unsafe_merge = True

        if is_unsafe_merge and (raw_level_idx == 2 or (macro_score >= 1.2 and probe_probs.get("same_identifier", 0.0) >= 0.8)):
            # Invoke CP-SAT formal solver to solve optimal safe downgrade
            candidate_utils = {
                "MERGE": macro_score,
                "BUFFER": 1.0,
                "DROP": 0.0,
            }
            forbidden = {"MERGE"}
            cpsat_res: CPSATVerdict = self.cpsat_solver.solve_safest_optimal_action(
                candidate_utilities=candidate_utils,
                forbidden_actions=forbidden,
                fallback_safe_action="BUFFER",
            )
            raw_level_idx = 1 if cpsat_res.selected_action == "BUFFER" else 0
            interlocked_by_cpsat = True
            cpsat_reason = (
                f"CP-SAT Formal Interlock vetoed MERGE action: "
                f"is_read_only_source={ro_prob:.2f}, affects_production_db={prod_prob:.2f}."
            )

        level = AlignmentLevel(raw_level_idx)
        if level == AlignmentLevel.DIFFERENT:
            action_enum = AlignmentAction.DROP
        elif level == AlignmentLevel.CONFIRMED_SAME:
            action_enum = AlignmentAction.MERGE
        else:
            action_enum = AlignmentAction.BUFFER

        # Step 6: Verifier Feedback Integration
        # If in Level 1 (Buffer), construct diagnostic packet for self-healing verifier
        diag_packet = None
        if level == AlignmentLevel.SUSPECT_BUFFER:
            diag_packet = self._build_diagnosis_packet(fp_a, fp_b, conflict_fields, probe_probs)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return AlignmentVerdict(
            level=level,
            action=action_enum,
            macro_score=macro_score,
            confidence=response.confidence,
            probe_verdicts=probe_probs,
            conflict_fields=conflict_fields,
            interlocked_by_cpsat=interlocked_by_cpsat,
            cpsat_reason=cpsat_reason,
            diagnosis_packet=diag_packet,
            evaluation_time_ms=elapsed_ms,
        )

    def _build_diagnosis_packet(
        self,
        fp_a: EntityFingerprint,
        fp_b: EntityFingerprint,
        conflict_fields: Dict[str, float],
        probe_probs: Dict[str, float],
    ) -> AlignmentDiagnosisPacket:
        """Constructs actionable diagnostic packet for Issue #13 Self-Healing Verifier."""
        # Find differing key-values between entity_a and entity_b
        differing: Dict[str, Tuple[Any, Any]] = {}
        all_keys = set(fp_a.attributes.keys()).union(set(fp_b.attributes.keys()))

        for k in all_keys:
            val_a = fp_a.attributes.get(k)
            val_b = fp_b.attributes.get(k)
            if val_a != val_b:
                differing[k] = (val_a, val_b)

        # Determine suggested repair strategy
        id_prob = probe_probs.get("same_identifier", 1.0)
        schema_prob = probe_probs.get("same_schema", 1.0)

        if id_prob < 0.5:
            strategy = "DISAMBIGUATE_IDENTIFIER"
        elif schema_prob < 0.6:
            strategy = "RECONCILE_SCHEMA"
        elif "is_read_only_source" in conflict_fields or "affects_production_db" in conflict_fields:
            strategy = "REQUEST_HUMAN_CLARIFICATION"
        else:
            strategy = "FIELD_OVERRIDE"

        hint = {
            "differing_keys_count": len(differing),
            "primary_conflict_probe": max(conflict_fields.keys(), key=lambda k: conflict_fields[k]) if conflict_fields else "none",
        }

        return AlignmentDiagnosisPacket(
            entity_a_id=fp_a.entity_id or "entity_a",
            entity_b_id=fp_b.entity_id or "entity_b",
            conflict_probes=conflict_fields,
            differing_fields=differing,
            suggested_repair_strategy=strategy,
            target_patch_hint=hint,
        )
