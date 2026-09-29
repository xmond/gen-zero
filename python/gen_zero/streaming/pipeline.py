"""Sandwich Streaming Pipeline.

Unifies:
1. Stage 1: Zero StreamIntentGate (<15ms, 80%+ chit-chat suppression).
2. Stage 2: Lightweight Semantic Structuring.
3. Stage 3: Zero PostVerificationAuditor (Evidence Grounding & Graph Deduplication).
4. Stage 4: Deterministic GraphAST Mutation & Layout (Mermaid & Draw.io XML).
"""

import time
from typing import Any, Dict, List, Optional

from gen_zero.graph.ast_engine import GraphAST, GraphMutation, GraphOp
from gen_zero.streaming.intent_gate import (
    PendingDraftItem,
    PendingDraftQueue,
    StreamActionType,
    StreamIntentDecision,
    StreamIntentGate,
    StreamUtterance,
)
from gen_zero.verifier.graph_auditor import (
    AuditDecision,
    GraphNodeSummary,
    PostVerificationAuditor,
    StructuredStepCandidate,
)


class SandwichStreamingPipeline:
    """Four-phase end-to-end streaming speech/log to workflow pipeline."""

    def __init__(
        self,
        intent_gate: Optional[StreamIntentGate] = None,
        post_auditor: Optional[PostVerificationAuditor] = None,
        graph_ast: Optional[GraphAST] = None,
    ) -> None:
        if intent_gate is not None:
            self.intent_gate = intent_gate
            self.draft_queue = intent_gate.draft_queue
        else:
            self.draft_queue = PendingDraftQueue()
            self.intent_gate = StreamIntentGate(draft_queue=self.draft_queue)
        self.post_auditor = post_auditor or PostVerificationAuditor()
        self.graph_ast = graph_ast or GraphAST()
        self.last_node_id: Optional[str] = None

    def _get_existing_node_summaries(self) -> List[GraphNodeSummary]:
        return [
            GraphNodeSummary(
                node_id=n.node_id,
                label=n.label,
                role=n.role_lane,
            )
            for n in self.graph_ast.nodes.values()
            if not n.is_draft
        ]

    def process_utterance(
        self,
        utterance: StreamUtterance,
        explicit_parent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Processes a streaming utterance through the 4-phase sandwich pipeline."""
        start_time = time.perf_counter()

        # Phase 1: Stream Intent Gate (<15ms)
        decision, draft_item = self.intent_gate.gate_utterance(utterance)

        # 1.1 Dropped Chit-chat / Noise (80%+ filtered)
        if decision.channel == "DROP":
            return {
                "status": "DROPPED",
                "utterance_id": utterance.utterance_id,
                "speaker": utterance.speaker,
                "channel": decision.channel,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "node_id": None,
                "latency_ms": round((time.perf_counter() - start_time) * 1000.0, 3),
            }

        # 1.2 Moderate-confidence Tentative Draft (Buffered & Painted Translucent)
        if decision.channel == "PENDING_DRAFT" and draft_item is not None:
            # Mount as translucent draft node in Graph AST
            parent_ids = [explicit_parent_id or self.last_node_id] if (explicit_parent_id or self.last_node_id) else []
            draft_mutation = GraphMutation(
                op=GraphOp.CREATE_NODE,
                target_node_id=draft_item.draft_id,
                label=f"待确认: {draft_item.text[:30]}",
                role_lane="待定",
                depends_on=parent_ids,
                is_draft=True,
            )
            created_draft_id = self.graph_ast.apply_mutation(draft_mutation)
            return {
                "status": "PENDING_DRAFT_QUEUED",
                "utterance_id": utterance.utterance_id,
                "speaker": utterance.speaker,
                "channel": decision.channel,
                "confidence": decision.confidence,
                "draft_id": draft_item.draft_id,
                "node_id": created_draft_id,
                "latency_ms": round((time.perf_counter() - start_time) * 1000.0, 3),
            }

        # Phase 2: Lightweight Semantic Structuring (High Confidence)
        candidate = StructuredStepCandidate(
            step_id=f"step_{utterance.utterance_id}",
            label=utterance.text.strip(),
            candidate_role=self.post_auditor.role_attributor.attribute_role(utterance.speaker),
            source_utterance_id=utterance.utterance_id,
        )

        # Phase 3: Zero Post-Verification Auditor (Grounding & Graph Deduplication)
        existing_summaries = self._get_existing_node_summaries()
        audit = self.post_auditor.audit(
            candidate=candidate,
            utterance_text=utterance.text,
            existing_nodes=existing_summaries,
        )

        # 3.1 Grounding failure (Hallucination rejection)
        if not audit.grounded:
            return {
                "status": "GROUNDING_REJECTED",
                "utterance_id": utterance.utterance_id,
                "confidence": audit.confidence,
                "reason": audit.reason,
                "node_id": None,
                "latency_ms": round((time.perf_counter() - start_time) * 1000.0, 3),
            }

        # 3.2 Graph Topology Deduplication (No duplicate nodes created)
        if audit.deduplicated_target is not None:
            # Update edge/attribute on existing node rather than creating new node
            dup_id = audit.deduplicated_target
            parent_id = explicit_parent_id or self.last_node_id
            if parent_id and parent_id != dup_id:
                self.graph_ast.apply_mutation(
                    GraphMutation(
                        op=GraphOp.ADD_EDGE,
                        target_node_id=dup_id,
                        label="",
                        role_lane=audit.assigned_role,
                        depends_on=[parent_id],
                    )
                )
            self.last_node_id = dup_id
            return {
                "status": "DEDUPLICATED_EXISTING_NODE",
                "utterance_id": utterance.utterance_id,
                "node_id": dup_id,
                "channel": decision.channel,
                "confidence": audit.confidence,
                "reason": audit.reason,
                "latency_ms": round((time.perf_counter() - start_time) * 1000.0, 3),
            }

        # Phase 4: Deterministic Graph AST Mutation
        parent_ids = [explicit_parent_id or self.last_node_id] if (explicit_parent_id or self.last_node_id) else []
        node_id = f"node_{utterance.utterance_id}"
        mutation = GraphMutation(
            op=GraphOp.CREATE_NODE,
            target_node_id=node_id,
            label=candidate.label,
            role_lane=audit.assigned_role,
            depends_on=parent_ids,
            is_draft=False,
        )
        created_id = self.graph_ast.apply_mutation(mutation)
        self.last_node_id = created_id

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "status": "MUTATION_APPLIED",
            "utterance_id": utterance.utterance_id,
            "speaker": utterance.speaker,
            "action": decision.action.value,
            "node_id": created_id,
            "role": audit.assigned_role,
            "confidence": audit.confidence,
            "channel": decision.channel,
            "latency_ms": round(total_latency_ms, 3),
        }

    def confirm_draft(self, draft_id: str) -> Optional[str]:
        """Upgrades a tentative draft to a verified permanent node."""
        draft = self.draft_queue.confirm(draft_id)
        if draft and draft_id in self.graph_ast.nodes:
            node = self.graph_ast.nodes[draft_id]
            node.is_draft = False
            node.label = draft.text
            self.last_node_id = draft_id
            self.graph_ast._recompute_layout()
            return draft_id
        return None

    def dismiss_draft(self, draft_id: str) -> bool:
        """Removes a tentative draft from queue and graph AST."""
        self.draft_queue.dismiss(draft_id)
        res = self.graph_ast.apply_mutation(
            GraphMutation(
                op=GraphOp.DELETE_NODE,
                target_node_id=draft_id,
                label="",
                role_lane="",
                depends_on=[],
            )
        )
        return bool(res)

    def export_mermaid(self) -> str:
        return self.graph_ast.to_mermaid()

    def export_drawio_xml(self) -> str:
        return self.graph_ast.to_drawio_xml()
