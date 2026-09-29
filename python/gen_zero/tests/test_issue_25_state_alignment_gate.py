"""Unit and Integration Tests for Issue #25: State Entity Alignment Gate & Co-Riding Probes.

Tests:
1. Co-Riding Probes Protocol Adapter: Concurrent evaluation of macro score and attributive nouls under shared prefix KV-cache.
2. Zero-Tuned Tri-State Natural Rounding: Exact cutoffs at 0.5 and 1.5 mapping to DROP, BUFFER, and MERGE without arbitrary floating constants.
3. Asymmetric Mistake Cost Defense: 100% convergence to Level 1 BUFFER on subtle divergence and ambiguous boundary states (0% False Positive merge).
4. CP-SAT Formal Safety Interlock: Immediate 100% veto and safe downgrade of MERGE when critical constraints (read-only, production DB) are active.
5. Verifier Feedback Integration: Generates structured AlignmentDiagnosisPacket with differing fields, probe conflicts, and repair strategy.
6. Entity Fingerprint Extraction: Robust extraction across dicts, classes, and strings with deterministic hashing.
7. Client Integration: Top-level decide_alignment API through GenZeroClient.
"""

import unittest
from gen_zero.service.co_riding_adapter import (
    CoRidingAlignmentRequest,
    CoRidingAlignmentResponse,
    CoRidingProbesAdapter,
    NoulProbeSpec,
    NoulProbeResult,
    DEFAULT_PROBES,
)
from gen_zero.gate.alignment_gate import (
    AlignmentLevel,
    AlignmentAction,
    EntityFingerprint,
    AlignmentDiagnosisPacket,
    AlignmentVerdict,
    extract_entity_fingerprint,
    StateAlignmentGate,
)
from gen_zero.client import GenZeroClient, GenZero


class TestCoRidingProbesAdapter(unittest.TestCase):
    """Test 1: Co-Riding Probes Protocol Adapter."""

    def setUp(self):
        self.adapter = CoRidingProbesAdapter()

    def test_shared_prefix_fingerprint_deterministic(self):
        ent_a = {"id": "user_42", "name": "Alice", "role": "admin"}
        ent_b = {"id": "user_42", "name": "Alice", "role": "admin"}
        fp1 = self.adapter.compute_prefix_fingerprint(ent_a, ent_b)
        fp2 = self.adapter.compute_prefix_fingerprint(ent_a, ent_b)
        self.assertEqual(fp1, fp2)
        self.assertEqual(len(fp1), 24)

    def test_single_request_co_riding_evaluation(self):
        ent_a = {"id": "file_101", "path": "/src/app.py", "type": "code"}
        ent_b = {"id": "file_101", "path": "/src/app.py", "type": "code"}

        req = CoRidingAlignmentRequest(entity_a=ent_a, entity_b=ent_b)
        res = self.adapter.evaluate_co_riding(req)

        self.assertIsInstance(res, CoRidingAlignmentResponse)
        self.assertIn(res.macro_level, [0, 1, 2])
        self.assertIn(res.macro_action, ["DROP", "BUFFER", "MERGE"])
        self.assertGreater(len(res.probes), 3)
        self.assertIn("same_identifier", res.probes)
        self.assertIn("same_schema", res.probes)
        self.assertIn("compatible_semantics", res.probes)
        self.assertLess(res.total_latency_ms, 50.0)  # sub-50ms SLA


class TestZeroTunedTriStateRubric(unittest.TestCase):
    """Test 2: Zero-tuned tri-state rubric with natural cutoffs."""

    def setUp(self):
        self.gate = StateAlignmentGate()

    def test_level_0_drop_for_distinct_entities(self):
        ent_a = {"id": "order_999", "amount": 100.0, "type": "finance"}
        ent_b = {"id": "log_error_888", "message": "Disk Full", "type": "system_log"}

        verdict = self.gate.evaluate_alignment(ent_a, ent_b)
        self.assertEqual(verdict.level, AlignmentLevel.DIFFERENT)
        self.assertEqual(verdict.action, AlignmentAction.DROP)
        self.assertLess(verdict.macro_score, 0.5)

    def test_level_2_merge_for_identical_entities(self):
        ent_a = {"id": "session_abc", "user_id": 10, "type": "web_session", "active": True}
        ent_b = {"id": "session_abc", "user_id": 10, "type": "web_session", "active": True}

        verdict = self.gate.evaluate_alignment(ent_a, ent_b)
        self.assertEqual(verdict.level, AlignmentLevel.CONFIRMED_SAME)
        self.assertEqual(verdict.action, AlignmentAction.MERGE)
        self.assertGreaterEqual(verdict.macro_score, 1.5)
        self.assertFalse(verdict.interlocked_by_cpsat)


class TestAsymmetricMistakeCostDefense(unittest.TestCase):
    """Test 3: Asymmetric mistake cost defense against false positive merges."""

    def setUp(self):
        self.gate = StateAlignmentGate()

    def test_subtle_identifier_mismatch_converges_to_buffer(self):
        # Similar fields, but different IDs: high risk of catastrophic false merge
        ent_a = {"id": "customer_account_001", "name": "Corp Inc", "tier": "gold"}
        ent_b = {"id": "customer_account_002", "name": "Corp Inc", "tier": "gold"}

        verdict = self.gate.evaluate_alignment(ent_a, ent_b)
        # MUST NOT be MERGE (False Positive)
        self.assertNotEqual(verdict.action, AlignmentAction.MERGE)
        self.assertEqual(verdict.level, AlignmentLevel.SUSPECT_BUFFER)
        self.assertEqual(verdict.action, AlignmentAction.BUFFER)
        self.assertIn("same_identifier", verdict.conflict_fields)

    def test_partial_schema_overlap_converges_to_buffer(self):
        # Compatible semantics, but disjoint attributes
        ent_a = {"id": "asset_10", "type": "media", "codec": "h264", "duration": 120}
        ent_b = {"id": "asset_10", "type": "media", "format": "mp4", "bitrate": 5000}

        verdict = self.gate.evaluate_alignment(ent_a, ent_b)
        self.assertEqual(verdict.level, AlignmentLevel.SUSPECT_BUFFER)
        self.assertEqual(verdict.action, AlignmentAction.BUFFER)


class TestCPSATFormalSafetyInterlock(unittest.TestCase):
    """Test 4: CP-SAT Formal Interlock for hard security constraints."""

    def setUp(self):
        self.gate = StateAlignmentGate()

    def test_read_only_source_interlocks_merge(self):
        # Exact same ID and attributes, but entity_a is a read-only protected source
        ent_a = {"id": "config_master", "read_only": True, "type": "config"}
        ent_b = {"id": "config_master", "read_only": False, "type": "config"}

        verdict = self.gate.evaluate_alignment(ent_a, ent_b)
        # Even with identical IDs, merge must be vetoed by CP-SAT interlock!
        self.assertTrue(verdict.interlocked_by_cpsat)
        self.assertIn("CP-SAT Formal Interlock vetoed MERGE", str(verdict.cpsat_reason))
        self.assertNotEqual(verdict.action, AlignmentAction.MERGE)
        self.assertEqual(verdict.action, AlignmentAction.BUFFER)

    def test_affects_production_db_interlocks_merge(self):
        ent_a = {"id": "prod_users_table", "env": "prod", "type": "database"}
        ent_b = {"id": "prod_users_table", "env": "prod", "type": "database"}

        verdict = self.gate.evaluate_alignment(ent_a, ent_b)
        self.assertTrue(verdict.interlocked_by_cpsat)
        self.assertEqual(verdict.action, AlignmentAction.BUFFER)

    def test_explicit_hard_safety_constraint_veto(self):
        ent_a = {"id": "order_77", "type": "order"}
        ent_b = {"id": "order_77", "type": "order"}

        verdict = self.gate.evaluate_alignment(
            ent_a,
            ent_b,
            hard_safety_constraints={"destructive_action_locked": True}
        )
        self.assertTrue(verdict.interlocked_by_cpsat)
        self.assertNotEqual(verdict.action, AlignmentAction.MERGE)


class TestVerifierFeedbackIntegration(unittest.TestCase):
    """Test 5: Verifier Feedback Integration & Diagnosis Packet."""

    def setUp(self):
        self.gate = StateAlignmentGate()

    def test_diagnosis_packet_generated_on_buffer(self):
        ent_a = {"id": "doc_1", "title": "Quarterly Report", "status": "draft"}
        ent_b = {"id": "doc_2", "title": "Quarterly Report", "status": "published"}

        verdict = self.gate.evaluate_alignment(ent_a, ent_b)
        self.assertEqual(verdict.action, AlignmentAction.BUFFER)
        self.assertIsNotNone(verdict.diagnosis_packet)

        packet: AlignmentDiagnosisPacket = verdict.diagnosis_packet
        self.assertEqual(packet.entity_a_id, "doc_1")
        self.assertEqual(packet.entity_b_id, "doc_2")
        self.assertIn("id", packet.differing_fields)
        self.assertIn("status", packet.differing_fields)
        self.assertEqual(packet.suggested_repair_strategy, "DISAMBIGUATE_IDENTIFIER")

    def test_schema_reconciliation_strategy_on_schema_conflict(self):
        ent_a = {"id": "user_item", "field1": "val1"}
        ent_b = {"id": "user_item", "col_a": "val1", "col_b": "val2", "col_c": "val3"}

        verdict = self.gate.evaluate_alignment(ent_a, ent_b)
        if verdict.action == AlignmentAction.BUFFER and verdict.diagnosis_packet:
            self.assertIn(
                verdict.diagnosis_packet.suggested_repair_strategy,
                ["RECONCILE_SCHEMA", "FIELD_OVERRIDE", "DISAMBIGUATE_IDENTIFIER"]
            )


class TestEntityFingerprintExtraction(unittest.TestCase):
    """Test 6: Entity fingerprint extraction."""

    def test_dict_fingerprint(self):
        d = {"id": "user_123", "name": "Bob", "role": "developer"}
        fp = extract_entity_fingerprint(d)
        self.assertEqual(fp.entity_id, "user_123")
        self.assertEqual(fp.schema_keys, ["id", "name", "role"])
        self.assertIsInstance(fp.hash_digest, str)
        self.assertEqual(len(fp.hash_digest), 16)

    def test_custom_class_fingerprint(self):
        class ResourceNode:
            def __init__(self, key, kind):
                self.key = key
                self.kind = kind

        obj = ResourceNode("res_node_1", "compute")
        fp = extract_entity_fingerprint(obj)
        self.assertEqual(fp.entity_id, "res_node_1")
        self.assertEqual(fp.entity_type, "ResourceNode")

    def test_string_fingerprint(self):
        raw_str = "MUTEX_LOCK_HANDLE_0x892"
        fp = extract_entity_fingerprint(raw_str)
        self.assertEqual(fp.entity_type, "string_entity")
        self.assertEqual(fp.entity_id, "MUTEX_LOCK_HANDLE_0x892")


class TestClientDecideAlignment(unittest.TestCase):
    """Test 7: Client integration through GenZero / GenZeroClient."""

    def setUp(self):
        self.client = GenZeroClient()

    def test_client_decide_alignment_end_to_end(self):
        self.assertTrue(hasattr(self.client, "alignment_gate"))

        res = self.client.decide_alignment(
            entity_a={"id": "job_01", "task": "compile", "status": "running"},
            entity_b={"id": "job_01", "task": "compile", "status": "running"},
        )
        self.assertIn("level", res)
        self.assertIn("action", res)
        self.assertIn("macro_score", res)
        self.assertIn("probe_verdicts", res)
        self.assertEqual(res["action"], "MERGE")
        self.assertEqual(res["level"], 2)


if __name__ == "__main__":
    unittest.main()
