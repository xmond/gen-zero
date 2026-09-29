"""Graph Grounding Auditor and Topology Deduplicator.

Phase 3 of the Sandwich Streaming Pipeline:
- GroundingAuditor: Verifies structured steps against original speaker utterance text (zero-hallucination).
- GraphDeduplicator: Evaluates candidate against existing graph nodes to prevent duplicate node creation.
- RoleAttributor: Maps extracted roles to canonical discrete role pool.
"""

from dataclasses import asdict, dataclass
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class StructuredStepCandidate:
    """A structured workflow candidate step extracted from speech."""
    step_id: str
    label: str
    candidate_role: str
    condition: Optional[str] = None
    source_utterance_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphNodeSummary:
    """Summary representation of an existing node in the workflow graph."""
    node_id: str
    label: str
    role: str
    condition: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditDecision:
    """Audit verification verdict output by Zero."""
    grounded: bool
    deduplicated_target: Optional[str]  # Target node_id if duplicate; None for unique new node
    assigned_role: str
    confidence: float
    latency_ms: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GroundingAuditor:
    """Verifies that extracted structured facts strictly originate from speaker utterance."""

    def __init__(self) -> None:
        pass

    def verify_grounding(self, utterance_text: str, candidate: StructuredStepCandidate) -> Tuple[bool, float, str]:
        """Checks if step label and condition keywords exist in source utterance."""
        u_text = utterance_text.lower()
        step_label = candidate.label.lower()

        # Tokenize step label into meaningful keywords (excluding stop characters)
        keywords = [w for w in re.findall(r"[\w\u4e00-\u9fff]{2,}", step_label)]
        if not keywords:
            return True, 0.95, "Short label accepted"

        # Check overlap
        matched = [kw for kw in keywords if kw in u_text]
        grounding_ratio = len(matched) / len(keywords)

        if grounding_ratio >= 0.50:
            return True, round(0.70 + 0.30 * grounding_ratio, 4), f"Grounded ({len(matched)}/{len(keywords)} key terms present)"
        else:
            return False, 0.25, f"Hallucination detected: key terms {keywords} missing from source utterance"


class GraphDeduplicator:
    """Deduplicates structured step against existing graph topology."""

    def __init__(self, similarity_threshold: float = 0.50) -> None:
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def _compute_jaccard(s1: str, s2: str) -> float:
        stopwords = {"首先", "然后", "接着", "我们", "你们", "他们", "还", "需要", "对", "完成", "做", "进行", "的", "与", "和", "在", "把", "请", "以及", "由"}
        clean1 = re.sub(r"[^\w\u4e00-\u9fff]", "", s1.lower())
        clean2 = re.sub(r"[^\w\u4e00-\u9fff]", "", s2.lower())
        for w in stopwords:
            clean1 = clean1.replace(w, "")
            clean2 = clean2.replace(w, "")
        if not clean1 or not clean2:
            return 0.0
        chars1 = set(clean1)
        chars2 = set(clean2)
        c_sim = len(chars1 & chars2) / len(chars1 | chars2)
        bg1 = {clean1[i:i+2] for i in range(len(clean1) - 1)}
        bg2 = {clean2[i:i+2] for i in range(len(clean2) - 1)}
        bg_sim = (len(bg1 & bg2) / len(bg1 | bg2)) if (bg1 and bg2) else c_sim
        return max(c_sim, bg_sim)

    def deduplicate(
        self,
        candidate: StructuredStepCandidate,
        existing_nodes: List[GraphNodeSummary],
    ) -> Tuple[Optional[str], float, str]:
        """Returns (duplicate_node_id, confidence, reason). If unique, duplicate_node_id is None."""
        if not existing_nodes:
            return None, 0.99, "Empty graph; candidate is guaranteed unique"

        best_match_id: Optional[str] = None
        highest_sim = 0.0

        for node in existing_nodes:
            # 1. Exact string match
            if candidate.label.strip() == node.label.strip():
                return node.node_id, 0.99, f"Exact label match with existing node '{node.node_id}'"

            # 2. Semantic token overlap
            sim = self._compute_jaccard(candidate.label, node.label)
            if sim > highest_sim:
                highest_sim = sim
                best_match_id = node.node_id

        if highest_sim >= self.similarity_threshold:
            return best_match_id, round(0.80 + 0.20 * highest_sim, 4), f"Semantic duplicate of node '{best_match_id}' (sim={highest_sim:.2f})"

        return None, round(1.0 - highest_sim, 4), "Unique new node verified"


class RoleAttributor:
    """Maps candidate roles to discrete canonical role lanes."""

    DEFAULT_ROLE_POOL = [
        "客户经理",
        "风控法务",
        "权证专员",
        "审批主管",
        "业务经办",
        "外部客户",
        "系统自动化",
    ]

    def __init__(self, allowable_roles: Optional[List[str]] = None) -> None:
        self.allowable_roles = allowable_roles or self.DEFAULT_ROLE_POOL

    def attribute_role(self, candidate_role: str) -> str:
        if not candidate_role:
            return "业务经办"
        c_role = candidate_role.strip()
        for role in self.allowable_roles:
            if role in c_role or c_role in role:
                return role
        return self.allowable_roles[0]


class PostVerificationAuditor:
    """Combines Grounding, Graph Deduplication, and Role Attribution in < 15ms."""

    def __init__(self, allowable_roles: Optional[List[str]] = None) -> None:
        self.grounding_auditor = GroundingAuditor()
        self.deduplicator = GraphDeduplicator()
        self.role_attributor = RoleAttributor(allowable_roles)

    def audit(
        self,
        candidate: StructuredStepCandidate,
        utterance_text: str,
        existing_nodes: List[GraphNodeSummary],
    ) -> AuditDecision:
        start_time = time.perf_counter()

        grounded, g_conf, g_reason = self.grounding_auditor.verify_grounding(utterance_text, candidate)
        dedup_id, d_conf, d_reason = self.deduplicator.deduplicate(candidate, existing_nodes)
        assigned_role = self.role_attributor.attribute_role(candidate.candidate_role)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        confidence = min(g_conf, d_conf)

        return AuditDecision(
            grounded=grounded,
            deduplicated_target=dedup_id,
            assigned_role=assigned_role,
            confidence=round(confidence, 4),
            latency_ms=round(latency_ms, 3),
            reason=f"Grounding: {g_reason}; Deduplication: {d_reason}",
        )
