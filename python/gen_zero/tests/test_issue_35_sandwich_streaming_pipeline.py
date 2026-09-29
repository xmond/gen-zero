"""Comprehensive Test Suite for Issue #35: Zero Sandwich Streaming Pipeline & Graph AST Specification.

Validates:
1. StreamIntentGate: 80%+ chit-chat suppression, four-way action classification, < 15ms latency.
2. Confidence Tiering & PendingDraftQueue: Auto-pass (>0.50), Draft buffering (0.30~0.50), Drop (<0.30).
3. PostVerificationAuditor: Evidence Grounding and Graph Topology Deduplication (Zero duplicate nodes).
4. Deterministic GraphAST Engine: Sugiyama-style topological layout, Mermaid & Draw.io XML export.
5. End-to-End Pipeline & GenZeroClient integration.
"""

import unittest
import xml.etree.ElementTree as ET

from gen_zero.graph import (
    GraphAST,
    GraphEdge,
    GraphMutation,
    GraphNode,
    GraphOp,
)
from gen_zero.streaming import (
    PendingDraftItem,
    PendingDraftQueue,
    SandwichStreamingPipeline,
    StreamActionType,
    StreamIntentDecision,
    StreamIntentGate,
    StreamUtterance,
)
from gen_zero.verifier import (
    AuditDecision,
    GraphDeduplicator,
    GraphNodeSummary,
    GroundingAuditor,
    PostVerificationAuditor,
    RoleAttributor,
    StructuredStepCandidate,
)
from gen_zero.client import GenZeroClient


class TestStreamIntentGate(unittest.TestCase):
    """Tests for Stage 1: Stream Intent Gate and Confidence Tiering."""

    def setUp(self):
        self.draft_queue = PendingDraftQueue()
        self.gate = StreamIntentGate(draft_queue=self.draft_queue)

    def test_chitchat_suppression_rate(self):
        chitchat_samples = [
            "你好",
            "早上好，听得到吗",
            "哈哈确实是这样",
            "好的好的收到",
            "谢谢大家今天参会",
            "Hello everyone, how are you today?",
            "Yeah cool sounds good",
            "Yep right got it",
            "今天天气挺不错的",
            "稍微等一下我接个电话",
        ]
        dropped_count = 0
        for idx, text in enumerate(chitchat_samples):
            u = StreamUtterance(utterance_id=idx, speaker="张三", timestamp_ms=idx * 1000, text=text)
            decision, draft = self.gate.gate_utterance(u)
            if decision.channel == "DROP":
                dropped_count += 1
            self.assertLess(decision.latency_ms, 15.0)

        suppression_rate = dropped_count / len(chitchat_samples)
        self.assertGreaterEqual(suppression_rate, 0.80, f"Suppression rate {suppression_rate:.2f} < 0.80")

    def test_action_classification_and_confidence_tiering(self):
        # 1. High-confidence ADD
        u_add = StreamUtterance(1, "客户经理", 1000, "首先由风控团队发起客户尽调与资质预审")
        d_add, _ = self.gate.gate_utterance(u_add)
        self.assertEqual(d_add.action, StreamActionType.ADD)
        self.assertEqual(d_add.channel, "AUTO_PASS")
        self.assertGreater(d_add.confidence, 0.50)

        # 2. High-confidence MODIFY
        u_mod = StreamUtterance(2, "风控法务", 2000, "把前置审核条件修改为需要同时满足纳税满两年")
        d_mod, _ = self.gate.gate_utterance(u_mod)
        self.assertEqual(d_mod.action, StreamActionType.MODIFY)
        self.assertEqual(d_mod.channel, "AUTO_PASS")

        # 3. High-confidence DELETE
        u_del = StreamUtterance(3, "主管", 3000, "取消线下纸质材料盖章这一步，全部走线上")
        d_del, _ = self.gate.gate_utterance(u_del)
        self.assertEqual(d_del.action, StreamActionType.DELETE)
        self.assertEqual(d_del.channel, "AUTO_PASS")

        # 4. Moderate-confidence Hedged -> Pending Draft
        u_hedge = StreamUtterance(4, "经办人", 4000, "后续可能需要外部第三方评估机构出具报告，先不确定")
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
        utterance = "我们需要客户经理上门核验营业执照原件"
        # Grounded candidate
        cand_good = StructuredStepCandidate("s1", "客户经理上门核验营业执照", "客户经理")
        audit_good = self.auditor.audit(cand_good, utterance, [])
        self.assertTrue(audit_good.grounded)
        self.assertGreaterEqual(audit_good.confidence, 0.70)

        # Hallucinated candidate with foreign terms
        cand_bad = StructuredStepCandidate("s2", "向中国人民银行征信系统调取区块链反洗钱报告", "风控法务")
        audit_bad = self.auditor.audit(cand_bad, utterance, [])
        self.assertFalse(audit_bad.grounded)
        self.assertLess(audit_bad.confidence, 0.35)

    def test_graph_deduplicator_prevents_duplicate_nodes(self):
        existing = [
            GraphNodeSummary(node_id="node_1", label="客户资质初审", role="客户经理"),
            GraphNodeSummary(node_id="node_2", label="房产抵押权证办理", role="权证专员"),
        ]

        # 1. Semantically identical candidate
        dup_candidate = StructuredStepCandidate("s3", "进行客户资质审查", "客户经理")
        audit_dup = self.auditor.audit(dup_candidate, "进行客户资质审查", existing)
        self.assertEqual(audit_dup.deduplicated_target, "node_1")

        # 2. Distinct novel candidate
        new_candidate = StructuredStepCandidate("s4", "签订三方资金监管协议", "经办人")
        audit_new = self.auditor.audit(new_candidate, "签订三方资金监管协议", existing)
        self.assertIsNone(audit_new.deduplicated_target)


class TestDeterministicGraphAST(unittest.TestCase):
    """Tests for Stage 4: Deterministic Graph AST and Output Formats."""

    def setUp(self):
        self.ast = GraphAST()

    def test_mutations_and_layout(self):
        n1 = self.ast.apply_mutation(GraphMutation(GraphOp.CREATE_NODE, "n1", "提交贷款申请", "客户经理", []))
        n2 = self.ast.apply_mutation(GraphMutation(GraphOp.CREATE_NODE, "n2", "风控合规审查", "风控法务", ["n1"]))
        n3 = self.ast.apply_mutation(GraphMutation(GraphOp.CREATE_NODE, "n3", "签署抵押合同", "权证专员", ["n2"], is_draft=True))

        self.assertEqual(len(self.ast.nodes), 3)
        self.assertEqual(len(self.ast.edges), 2)

        # Check coordinates assigned
        self.assertGreater(self.ast.nodes["n2"].y, self.ast.nodes["n1"].y)
        self.assertGreater(self.ast.nodes["n3"].y, self.ast.nodes["n2"].y)

    def test_mermaid_export(self):
        self.ast.apply_mutation(GraphMutation(GraphOp.CREATE_NODE, "n1", "步骤一", "角色A", []))
        self.ast.apply_mutation(GraphMutation(GraphOp.CREATE_NODE, "n2", "步骤二", "角色B", ["n1"], is_draft=True))

        mermaid = self.ast.to_mermaid()
        self.assertIn("flowchart TD", mermaid)
        self.assertIn("n1[\"角色A: 步骤一\"]", mermaid)
        self.assertIn("n2[\"角色B: 步骤二 [草稿?]\"]", mermaid)
        self.assertIn("n1 --> n2", mermaid)

    def test_drawio_xml_export_validity(self):
        self.ast.apply_mutation(GraphMutation(GraphOp.CREATE_NODE, "n1", "核验征信", "风控", []))
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
            (1, "张经理", "大家早上好，今天主要讨论按揭放款的业务流。"),       # Chit-chat -> DROP
            (2, "张经理", "好的，听得见。"),                                # Chit-chat -> DROP
            (3, "张经理", "首先由客户经理完成买卖双方资质与征信初审。"),           # High-conf ADD -> node_3
            (4, "李风控", "资质审核完成后，转交风控部门进行房屋估值和终审。"),       # High-conf ADD -> node_4 (depends on node_3)
            (5, "王经办", "后续可能需要担保公司介入出具一份保函，但这块先待定。"),   # Hedged -> Draft Queued
            (6, "张经理", "我们还需要对买卖双方的资质和征信做初审。"),             # Semantic Duplicate -> Dedup to node_3
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
        res1 = self.client.process_streaming_utterance(101, "主管", "第一步由业务发起项目立项申请")
        self.assertEqual(res1["status"], "MUTATION_APPLIED")

        # 2. Process hedged utterance
        res2 = self.client.process_streaming_utterance(102, "法务", "可能需要法务部先出具合规备忘录，先不确定")
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
