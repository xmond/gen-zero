# anti-leakage: allow-mock-tensor
"""Tests for Issue #35: Real-time Sandwich Streaming Decision Pipeline."""

import unittest
import xml.etree.ElementTree as ET
from gen_zero.streaming.intent_gate import (
    StreamUtterance,
    StreamIntentGate,
    StreamActionType,
    PendingDraftQueue,
)
from gen_zero.verifier.graph_auditor import (
    StructuredStepCandidate,
    GraphNodeSummary,
    PostVerificationAuditor,
)
from gen_zero.graph import GraphAST, GraphMutation, GraphOp
from gen_zero.streaming.pipeline import SandwichStreamingPipeline
from gen_zero.client import GenZeroClient


class TestStreamIntentGate(unittest.TestCase):
    """Tests for Stage 1: Ultra-fast Stream Intent Gate (<15ms)."""

    def setUp(self):
        self.draft_queue = PendingDraftQueue()
        self.gate = StreamIntentGate(draft_queue=self.draft_queue)

    def test_chitchat_suppression_rate(self):
        chitchat_samples = [
            "hello there",
            "good morning, can you hear me?",
            "haha indeed that is true",
            "okay great got it",
            "thank you all for attending today",
            "Hello everyone, how are you today?",
            "Yeah cool sounds good",
            "Yep right got it",
            "nice weather today outside",
            "hold on a second let me take a call",
        ]
        dropped_count = 0
        for idx, text in enumerate(chitchat_samples):
            u = StreamUtterance(utterance_id=idx, speaker="Alice", timestamp_ms=idx * 1000, text=text)
            decision, draft = self.gate.gate_utterance(u)
            if decision.channel == "DROP":
                dropped_count += 1
            self.assertLess(decision.latency_ms, 15.0)

        suppression_rate = dropped_count / len(chitchat_samples)
        self.assertGreaterEqual(suppression_rate, 0.80, f"Suppression rate {suppression_rate:.2f} < 0.80")

    def test_action_classification_and_confidence_tiering(self):
        # 1. High-confidence ADD
        u_add = StreamUtterance(1, "account_manager", 1000, "First we need to add customer qualification pre-review")
        d_add, _ = self.gate.gate_utterance(u_add)
        self.assertEqual(d_add.action, StreamActionType.ADD)
        self.assertEqual(d_add.channel, "AUTO_PASS")
        self.assertGreater(d_add.confidence, 0.50)

        # 2. High-confidence MODIFY
        u_mod = StreamUtterance(2, "risk_legal", 2000, "Update prerequisite check to require tax payments for two years")
        d_mod, _ = self.gate.gate_utterance(u_mod)
        self.assertEqual(d_mod.action, StreamActionType.MODIFY)
        self.assertEqual(d_mod.channel, "AUTO_PASS")

        # 3. High-confidence DELETE
        u_del = StreamUtterance(3, "approval_lead", 3000, "Cancel offline paper stamping step and remove entirely")
        d_del, _ = self.gate.gate_utterance(u_del)
        self.assertEqual(d_del.action, StreamActionType.DELETE)
        self.assertEqual(d_del.channel, "AUTO_PASS")

        # 4. Moderate-confidence Hedged -> Pending Draft
        u_hedge = StreamUtterance(4, "operator", 4000, "Maybe we need third-party agency assessment later, not sure yet")
        d_hedge, draft = self.gate.gate_utterance(u_hedge)
        self.assertEqual(d_hedge.channel, "PENDING_DRAFT")
        self.assertTrue(0.30 <= d_hedge.confidence <= 0.50)
        self.assertIsNotNone(draft)
        self.assertEqual(len(self.draft_queue), 1)


class TestPostVerificationAuditor(unittest.TestCase):
    """Tests for Stage 3: Grounding and Graph Topology Deduplication."""

    def setUp(self):
        self.auditor = PostVerificationAuditor()

    def test_grounding_auditor_prevents_hallucinations(self):
        utterance = "We need account manager to verify original business license"
        # Grounded candidate
        cand_good = StructuredStepCandidate("s1", "account manager verify business license", "account_manager")
        audit_good = self.auditor.audit(cand_good, utterance, [])
        self.assertTrue(audit_good.grounded)
        self.assertGreaterEqual(audit_good.confidence, 0.70)

        # Hallucinated candidate with foreign terms
        cand_bad = StructuredStepCandidate("s2", "retrieve blockchain anti-money laundering audit log", "risk_legal")
        audit_bad = self.auditor.audit(cand_bad, utterance, [])
        self.assertFalse(audit_bad.grounded)
        self.assertLess(audit_bad.confidence, 0.35)

    def test_graph_deduplicator_prevents_duplicate_nodes(self):
        existing = [
            GraphNodeSummary(node_id="node_1", label="customer qualification precheck", role="account_manager"),
            GraphNodeSummary(node_id="node_2", label="property mortgage certificate filing", role="warrant_specialist"),
        ]

        # 1. Semantically identical candidate
        dup_candidate = StructuredStepCandidate("s3", "customer qualification precheck", "account_manager")
        audit_dup = self.auditor.audit(dup_candidate, "customer qualification precheck", existing)
        self.assertEqual(audit_dup.deduplicated_target, "node_1")

        # 2. Distinct novel candidate
        new_candidate = StructuredStepCandidate("s4", "sign tripartite fund escrow agreement", "operator")
        audit_new = self.auditor.audit(new_candidate, "sign tripartite fund escrow agreement", existing)
        self.assertIsNone(audit_new.deduplicated_target)


class TestDeterministicGraphAST(unittest.TestCase):
    """Tests for Stage 4: Deterministic Graph AST and Output Formats."""

    def setUp(self):
        self.ast = GraphAST()

    def test_mutations_and_layout(self):
        n1 = self.ast.apply_mutation(GraphMutation(GraphOp.CREATE_NODE, "n1", "submit loan application", "account_manager", []))
        n2 = self.ast.apply_mutation(GraphMutation(GraphOp.CREATE_NODE, "n2", "risk compliance review", "risk_legal", ["n1"]))
        n3 = self.ast.apply_mutation(GraphMutation(GraphOp.CREATE_NODE, "n3", "sign mortgage contract", "warrant_specialist", ["n2"], is_draft=True))

        self.assertEqual(len(self.ast.nodes), 3)
        self.assertEqual(len(self.ast.edges), 2)

        # Check coordinates assigned
        self.assertGreater(self.ast.nodes["n2"].y, self.ast.nodes["n1"].y)
        self.assertGreater(self.ast.nodes["n3"].y, self.ast.nodes["n2"].y)

    def test_mermaid_export(self):
        self.ast.apply_mutation(GraphMutation(GraphOp.CREATE_NODE, "n1", "step_one", "role_a", []))
        self.ast.apply_mutation(GraphMutation(GraphOp.CREATE_NODE, "n2", "step_two", "role_b", ["n1"], is_draft=True))

        mermaid = self.ast.to_mermaid()
        self.assertIn("flowchart TD", mermaid)
        self.assertIn('n1["role_a: step_one"]', mermaid)
        self.assertIn('n2["role_b: step_two [Draft?]"]', mermaid)
        self.assertIn("n1 --> n2", mermaid)

    def test_drawio_xml_export_validity(self):
        self.ast.apply_mutation(GraphMutation(GraphOp.CREATE_NODE, "n1", "verify credit score", "risk_legal", []))
        xml_str = self.ast.to_drawio_xml()

        # Parse XML to verify valid structure
        root = ET.fromstring(xml_str)
        self.assertEqual(root.tag, "mxfile")
        diagram = root.find("diagram")
        self.assertIsNotNone(diagram)
        mx_model = diagram.find("mxGraphModel")
        self.assertIsNotNone(mx_model)
        cells = mx_model.find("root").findall("mxCell")
        # 0, 1 + 1 node cell = 3
        self.assertGreaterEqual(len(cells), 3)


class TestSandwichStreamingPipeline(unittest.TestCase):
    """End-to-End Simulation of Sandwich Streaming Pipeline."""

    def setUp(self):
        self.pipeline = SandwichStreamingPipeline()

    def test_mixed_session_simulation(self):
        session = [
            (1, "Alice", "Good morning everyone, today we discuss mortgage lending workflow."),       # Chit-chat -> DROP
            (2, "Alice", "Sounds good, can you hear me?"),                                              # Chit-chat -> DROP
            (3, "Alice", "First we need to add buyer seller qualification and credit check."),           # High-conf ADD -> node_3
            (4, "Bob", "Next step submit dossier to risk legal department for collateral valuation."),   # High-conf ADD -> node_4 (depends on node_3)
            (5, "Charlie", "Maybe we need guarantee agency letter later, tentative for now."),          # Hedged -> Draft Queued
            (6, "Alice", "First we need to add buyer seller qualification and credit check."),           # Semantic Duplicate -> Dedup to node_3
        ]

        results = []
        for u_id, spk, txt in session:
            u = StreamUtterance(utterance_id=u_id, speaker=spk, timestamp_ms=u_id * 1000, text=txt)
            res = self.pipeline.process_utterance(u)
            results.append(res)

        # 1. First two utterances must be dropped cleanly
        self.assertEqual(results[0]["status"], "DROPPED")
        self.assertEqual(results[1]["status"], "DROPPED")

        # 2. Utterances 3 & 4 must create nodes
        self.assertEqual(results[2]["status"], "MUTATION_APPLIED")
        self.assertEqual(results[3]["status"], "MUTATION_APPLIED")

        # 3. Utterance 5 must queue as pending draft
        self.assertEqual(results[4]["status"], "PENDING_DRAFT_QUEUED")
        draft_id = results[4]["draft_id"]

        # 4. Utterance 6 must be deduplicated back to node_3
        self.assertEqual(results[5]["status"], "DEDUPLICATED_EXISTING_NODE")
        self.assertEqual(results[5]["node_id"], "node_3")

        # Verify draft queue count
        self.assertEqual(len(self.pipeline.draft_queue.list_pending()), 1)

        # 5. Confirm draft manually (Human-in-the-loop)
        confirmed_id = self.pipeline.confirm_draft(draft_id)
        self.assertEqual(confirmed_id, draft_id)
        self.assertEqual(len(self.pipeline.draft_queue.list_pending()), 0)

        # Verify export
        mermaid = self.pipeline.export_mermaid()
        self.assertIn("node_3", mermaid)
        self.assertIn("node_4", mermaid)
        xml = self.pipeline.export_drawio_xml()
        self.assertIn("node_3", xml)


class TestGenZeroClientStreamingIntegration(unittest.TestCase):
    """Tests for Client SDK endpoints."""

    def setUp(self):
        self.client = GenZeroClient()

    def test_client_streaming_and_rendering(self):
        # 1. Process utterance
        res1 = self.client.process_streaming_utterance(101, "Alice", "First step need to add project approval application")
        self.assertEqual(res1["status"], "MUTATION_APPLIED")

        # 2. Process hedged utterance
        res2 = self.client.process_streaming_utterance(102, "Bob", "Maybe legal needs to review compliance memo, not sure yet")
        self.assertEqual(res2["status"], "PENDING_DRAFT_QUEUED")
        draft_id = res2["draft_id"]

        # 3. Confirm draft
        c_res = self.client.confirm_workflow_draft(draft_id)
        self.assertEqual(c_res, draft_id)

        # 4. Render formats
        mermaid = self.client.render_workflow_graph(format="mermaid")
        self.assertIn("flowchart TD", mermaid)

        drawio = self.client.render_workflow_graph(format="drawio_xml")
        self.assertIn("<mxfile", drawio)


if __name__ == "__main__":
    unittest.main()
