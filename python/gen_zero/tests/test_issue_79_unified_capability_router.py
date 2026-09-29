"""Unit tests for RFC-079 / Issue #79: Unified Heterogeneous Capability Routing,
Diversity-Penalized Beam Search Planning, and Cryptographic Provenance Audit.
"""

import unittest
import os
import tempfile
import json
import numpy as np

from gen_zero.capability.descriptor import CapabilityDescriptor, CapabilityType, RiskLevel
from gen_zero.capability.registry import CapabilityRegistry
from gen_zero.planner.diversity_beam import DiversityBeamPlanner, DiversityBeamPlanResult
from gen_zero.provenance.auditor import DecisionProvenanceAuditor, DecisionProvenanceRecord
from gen_zero.provenance.permission_arbiter import PermissionArbiter, PermissionVerdict
from gen_zero.harness.setup import setup_host_harness


class TestCapabilityDescriptorAndRegistry(unittest.TestCase):
    """Tests for CapabilityDescriptor and thread-safe CapabilityRegistry."""

    def setUp(self):
        self.registry = CapabilityRegistry()
        self.tool1 = CapabilityDescriptor(
            capability_id="tool:db_query",
            capability_type=CapabilityType.TOOL,
            description="Execute read-only SQL queries",
            domain="data",
            cost_weight=1.0,
            risk_level=RiskLevel.READ_ONLY,
            required_permissions=["db:read"],
        )
        self.tool2 = CapabilityDescriptor(
            capability_id="tool:fs_write",
            capability_type=CapabilityType.TOOL,
            description="Write files to disk",
            domain="dev",
            cost_weight=2.0,
            risk_level=RiskLevel.WRITE_SAFE,
            required_permissions=["fs:write"],
        )
        self.tool3 = CapabilityDescriptor(
            capability_id="cli:docker_rm",
            capability_type=CapabilityType.CLI,
            description="Remove container",
            domain="system",
            cost_weight=3.5,
            risk_level=RiskLevel.DESTRUCTIVE,
            required_permissions=["docker:admin"],
        )
        self.skill1 = CapabilityDescriptor(
            capability_id="skill:deep_audit",
            capability_type=CapabilityType.SKILL,
            description="Perform cross-repository security and regression audit",
            domain="verify",
            cost_weight=2.5,
            risk_level=RiskLevel.READ_ONLY,
        )

    def test_descriptor_properties_and_digest(self):
        digest1 = self.tool1.digest()
        self.assertTrue(isinstance(digest1, str) and len(digest1) == 64)
        d_dict = self.tool1.to_dict()
        self.assertEqual(d_dict["capability_id"], "tool:db_query")
        self.assertEqual(d_dict["risk_level"], "read_only")

    def test_registry_registration_and_retrieval(self):
        self.registry.register(self.tool1)
        self.registry.register(self.tool2)
        self.registry.register(self.tool3)
        self.registry.register(self.skill1)

        self.assertEqual(self.registry.count(), 4)
        self.assertEqual(self.registry.get("tool:db_query"), self.tool1)
        self.assertIsNone(self.registry.get("non_existent"))

    def test_registry_filtering(self):
        self.registry.register(self.tool1)
        self.registry.register(self.tool2)
        self.registry.register(self.tool3)
        self.registry.register(self.skill1)

        # Filter by capability type
        tools = self.registry.filter(capability_type=CapabilityType.TOOL)
        self.assertEqual(len(tools), 2)
        self.assertTrue(all(t.capability_type == CapabilityType.TOOL for t in tools))

        # Filter by domain
        dev_caps = self.registry.filter(domain="dev")
        self.assertEqual(len(dev_caps), 1)
        self.assertEqual(dev_caps[0].capability_id, "tool:fs_write")

        # Filter by risk level
        safe_caps = self.registry.filter(max_risk=RiskLevel.READ_ONLY)
        self.assertEqual(len(safe_caps), 2)
        self.assertNotIn(self.tool3, safe_caps)

        # Filter by permissions
        granted = {"db:read", "fs:write"}
        permitted_caps = self.registry.filter(required_permissions=granted)
        self.assertIn(self.tool1, permitted_caps)
        self.assertIn(self.tool2, permitted_caps)
        self.assertNotIn(self.tool3, permitted_caps)

    def test_coarse_domain_gating_stage1(self):
        self.registry.register(self.tool1)
        self.registry.register(self.tool2)
        self.registry.register(self.tool3)
        self.registry.register(self.skill1)

        # Query with dev domain
        gated = self.registry.coarse_filter_stage1(domain="dev")
        ids = [c.capability_id for c in gated]
        self.assertIn("tool:fs_write", ids)

    def test_snapshot_hash_and_manifest(self):
        self.registry.register(self.tool1)
        self.registry.register(self.tool2)
        h1 = self.registry.snapshot_hash()

        manifest = self.registry.export_manifest()
        self.assertEqual(manifest["snapshot_hash"], h1)
        self.assertEqual(len(manifest["capabilities"]), 2)

        # Registering new capability changes hash
        self.registry.register(self.tool3)
        h2 = self.registry.snapshot_hash()
        self.assertNotEqual(h1, h2)

        # Roundtrip into fresh registry
        fresh = CapabilityRegistry()
        fresh.import_manifest(manifest)
        self.assertEqual(fresh.snapshot_hash(), h1)
        self.assertEqual(fresh.count(), 2)

    def test_auto_discover_host_cli(self):
        count = self.registry.auto_discover_host_cli()
        self.assertGreater(count, 0)
        # Check that python3 or git is discovered
        all_ids = [c.capability_id for c in self.registry.list_all()]
        self.assertTrue(any("cli:python3" in cid or "cli:git" in cid for cid in all_ids))


class TestDiversityBeamPlanner(unittest.TestCase):
    """Tests for DiversityBeamPlanner (RFC-079 formulation)."""

    def setUp(self):
        self.candidates = [
            CapabilityDescriptor(
                capability_id="tool:status_check",
                capability_type=CapabilityType.TOOL,
                description="Poll server health status",
                domain="system",
                cost_weight=0.5,
                risk_level=RiskLevel.READ_ONLY,
            ),
            CapabilityDescriptor(
                capability_id="tool:diagnose_log",
                capability_type=CapabilityType.TOOL,
                description="Grep and scan error logs",
                domain="dev",
                cost_weight=1.0,
                risk_level=RiskLevel.READ_ONLY,
            ),
            CapabilityDescriptor(
                capability_id="tool:restart_service",
                capability_type=CapabilityType.TOOL,
                description="Trigger service restart",
                domain="system",
                cost_weight=2.0,
                risk_level=RiskLevel.WRITE_SAFE,
            ),
        ]

    def test_diversity_penalty_avoids_repetition_trap(self):
        """When an action has artificially high base score, standard beam repeats it;
        Diversity beam with lambda > 0 penalizes repetitions and forces diversified path."""
        # Simulated base policy function that assigns high prior to tool:status_check
        def policy_fn(state, action_names):
            return {
                "tool:status_check": 5.0,
                "tool:diagnose_log": 4.2,
                "tool:restart_service": 3.5,
            }

        # 1. Baseline planner without diversity penalty (lambda = 0.0)
        planner_baseline = DiversityBeamPlanner(
            beam_width=3,
            max_depth=3,
            lambda_diversity=0.0,
            mu_cost=0.0,
        )
        res_baseline = planner_baseline.plan(
            initial_state="System alert triggered",
            candidate_capabilities=self.candidates,
            prior_policy_fn=policy_fn,
        )
        # Baseline repeats status_check 3 times
        self.assertEqual(res_baseline.selected_sequence, [
            "tool:status_check",
            "tool:status_check",
            "tool:status_check",
        ])
        self.assertEqual(res_baseline.repetition_count, 2)

        # 2. Diversity planner with lambda = 2.0
        planner_div = DiversityBeamPlanner(
            beam_width=3,
            max_depth=3,
            lambda_diversity=2.0,
            mu_cost=0.0,
        )
        res_div = planner_div.plan(
            initial_state="System alert triggered",
            candidate_capabilities=self.candidates,
            prior_policy_fn=policy_fn,
        )

        # Diversity planner completely eliminates repetitive loops:
        self.assertEqual(res_div.repetition_count, 0)
        unique_actions = set(res_div.selected_sequence)
        self.assertEqual(len(unique_actions), 3)
        self.assertEqual(len(res_div.selected_sequence), 3)

    def test_cost_factor_influences_selection(self):
        """Cost factor mu penalizes high cost actions."""
        def uniform_policy(state, action_names):
            return {a: 3.0 for a in action_names}

        planner = DiversityBeamPlanner(
            beam_width=2,
            max_depth=1,
            lambda_diversity=0.0,
            mu_cost=1.0,  # strong cost penalty
        )
        res = planner.plan(
            initial_state="",
            candidate_capabilities=self.candidates,
            prior_policy_fn=uniform_policy,
        )
        # tool:status_check has lowest cost (0.5), so net score 3.0 - 0.5 = 2.5 > 2.0 > 1.0
        self.assertEqual(res.selected_sequence, ["tool:status_check"])

    def test_empty_candidates_or_zero_horizon(self):
        planner = DiversityBeamPlanner(beam_width=2, max_depth=3)
        res_empty = planner.plan(initial_state="", candidate_capabilities=[])
        self.assertEqual(res_empty.selected_sequence, [])
        self.assertEqual(res_empty.selected_score, 0.0)


class TestDecisionProvenanceAuditor(unittest.TestCase):
    """Tests for DecisionProvenanceAuditor and cryptographic verification."""

    def setUp(self):
        self.auditor = DecisionProvenanceAuditor(audit_secret="test_secret_gen_zero")
        self.candidates = [
            CapabilityDescriptor(
                capability_id="tool:read",
                capability_type=CapabilityType.TOOL,
                description="Read file",
                domain="dev",
                cost_weight=1.0,
                risk_level=RiskLevel.READ_ONLY,
                required_permissions=["fs:read"],
            ),
            CapabilityDescriptor(
                capability_id="tool:write",
                capability_type=CapabilityType.TOOL,
                description="Write file",
                domain="dev",
                cost_weight=1.5,
                risk_level=RiskLevel.WRITE_SAFE,
                required_permissions=["fs:write"],
            ),
        ]
        self.plan = ["tool:read", "tool:write"]
        self.context = "Edit config file safe"
        self.policy = {"min_confidence": 0.85, "max_risk": "write_safe"}

    def test_record_generation_and_valid_verification(self):
        record = self.auditor.generate_provenance(
            active_descriptors=self.candidates,
            policy_thresholds=self.policy,
            input_context=self.context,
            selected_sequence=self.plan,
        )

        self.assertTrue(record.trace_token)
        self.assertEqual(len(record.trace_token), 64)

        # Verification passes on genuine record
        is_valid, msg = self.auditor.verify_provenance(
            record,
            active_descriptors=self.candidates,
            policy_thresholds=self.policy,
            input_context=self.context,
        )
        self.assertTrue(is_valid, f"Expected valid record, got: {msg}")

    def test_tamper_detection_on_sequence(self):
        record = self.auditor.generate_provenance(
            active_descriptors=self.candidates,
            policy_thresholds=self.policy,
            input_context=self.context,
            selected_sequence=self.plan,
        )

        # Tamper chosen sequence: replace "tool:read" with unauthorized action
        tampered_dict = record.to_dict()
        tampered_dict["selected_sequence"] = ["tool:delete_all", "tool:write"]
        tampered_record = DecisionProvenanceRecord.from_dict(tampered_dict)

        is_valid, msg = self.auditor.verify_provenance(tampered_record)
        self.assertFalse(is_valid)
        self.assertIn("inconsistency", msg.lower())

    def test_tamper_detection_on_policy_fingerprint(self):
        record = self.auditor.generate_provenance(
            active_descriptors=self.candidates,
            policy_thresholds=self.policy,
            input_context=self.context,
            selected_sequence=self.plan,
        )

        tampered_dict = record.to_dict()
        tampered_dict["policy_fingerprint"] = "0" * 64
        tampered_record = DecisionProvenanceRecord.from_dict(tampered_dict)

        is_valid, msg = self.auditor.verify_provenance(tampered_record)
        self.assertFalse(is_valid)

    def test_tamper_detection_on_candidates_snapshot(self):
        record = self.auditor.generate_provenance(
            active_descriptors=self.candidates,
            policy_thresholds=self.policy,
            input_context=self.context,
            selected_sequence=self.plan,
        )

        tampered_dict = record.to_dict()
        tampered_dict["candidate_snapshot_hash"] = "1" * 64
        tampered_record = DecisionProvenanceRecord.from_dict(tampered_dict)

        is_valid, msg = self.auditor.verify_provenance(tampered_record)
        self.assertFalse(is_valid)

    def test_serialization_roundtrip(self):
        record = self.auditor.generate_provenance(
            active_descriptors=self.candidates,
            policy_thresholds=self.policy,
            input_context=self.context,
            selected_sequence=self.plan,
        )

        r_dict = record.to_dict()
        loaded = DecisionProvenanceRecord.from_dict(r_dict)
        self.assertEqual(loaded.trace_token, record.trace_token)
        valid, _ = self.auditor.verify_provenance(loaded)
        self.assertTrue(valid)


class TestPermissionArbiter(unittest.TestCase):
    """Tests for Zero-Privilege PermissionArbiter."""

    def setUp(self):
        self.arbiter = PermissionArbiter(
            granted_permissions=["fs:read", "fs:write", "db:read"],
            max_allowed_risk=RiskLevel.WRITE_SAFE,
        )
        self.read_cap = CapabilityDescriptor(
            capability_id="tool:read",
            capability_type=CapabilityType.TOOL,
            description="Read file",
            domain="dev",
            cost_weight=1.0,
            risk_level=RiskLevel.READ_ONLY,
            required_permissions=["fs:read"],
        )
        self.write_cap = CapabilityDescriptor(
            capability_id="tool:write",
            capability_type=CapabilityType.TOOL,
            description="Write file",
            domain="dev",
            cost_weight=1.5,
            risk_level=RiskLevel.WRITE_SAFE,
            required_permissions=["fs:write"],
        )
        self.destr_cap = CapabilityDescriptor(
            capability_id="cli:rm_rf",
            capability_type=CapabilityType.CLI,
            description="Delete recursively",
            domain="dev",
            cost_weight=3.0,
            risk_level=RiskLevel.DESTRUCTIVE,
            required_permissions=["fs:write"],
        )
        self.net_cap = CapabilityDescriptor(
            capability_id="tool:curl",
            capability_type=CapabilityType.TOOL,
            description="Network call",
            domain="web",
            cost_weight=1.0,
            risk_level=RiskLevel.NETWORK,
            required_permissions=["network:outbound"],
        )

    def test_permission_verdict_granted(self):
        ok1, _ = self.arbiter.evaluate_capability(self.read_cap)
        self.assertTrue(ok1)
        ok2, _ = self.arbiter.evaluate_capability(self.write_cap)
        self.assertTrue(ok2)

    def test_risk_level_exceeded_rejection(self):
        # Destructive risk exceeds WRITE_SAFE session ceiling
        ok, reason = self.arbiter.evaluate_capability(self.destr_cap)
        self.assertFalse(ok)
        self.assertIn("exceeds maximum allowed boundary", reason)

    def test_missing_permission_rejection(self):
        # Admin permission not in granted set
        admin_cap = CapabilityDescriptor(
            capability_id="tool:admin_action",
            capability_type=CapabilityType.TOOL,
            description="Perform admin action",
            domain="system",
            cost_weight=1.0,
            risk_level=RiskLevel.WRITE_SAFE,
            required_permissions=["admin:sudo"],
        )
        ok, reason = self.arbiter.evaluate_capability(admin_cap)
        self.assertFalse(ok)
        self.assertIn("missing required host permissions", reason.lower())

    def test_explicit_blocking_rule(self):
        self.arbiter.revoke_capability("tool:write")
        ok, reason = self.arbiter.evaluate_capability(self.write_cap)
        self.assertFalse(ok)
        self.assertIn("explicitly revoked", reason)

    def test_evaluate_plan(self):
        plan = [self.read_cap, self.write_cap]
        verdict = self.arbiter.evaluate_plan(plan)
        self.assertTrue(verdict.is_authorized)
        self.assertEqual(len(verdict.authorized_capabilities), 2)
        self.assertEqual(len(verdict.blocked_capabilities), 0)

        # Plan with unauthorized action
        unsafe_plan = [self.read_cap, self.destr_cap]
        unsafe_verdict = self.arbiter.evaluate_plan(unsafe_plan)
        self.assertFalse(unsafe_verdict.is_authorized)
        self.assertEqual(len(unsafe_verdict.blocked_capabilities), 1)
        self.assertEqual(unsafe_verdict.blocked_capabilities[0], self.destr_cap)


class TestHostHarnessSetup(unittest.TestCase):
    """Tests for setup_host_harness."""

    def test_setup_host_harness_execution(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            res = setup_host_harness(output_path=temp_path)
            self.assertEqual(res["status"], "SUCCESS")
            self.assertGreater(res["total_capabilities"], 0)
            self.assertGreater(res["host_cli_detected"], 0)
            self.assertTrue(res["snapshot_hash"])
            self.assertTrue(os.path.exists(temp_path))

            with open(temp_path, "r", encoding="utf-8") as fp:
                manifest = json.load(fp)
            self.assertEqual(manifest["snapshot_hash"], res["snapshot_hash"])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
