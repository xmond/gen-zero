"""Round 12 Formal Verification Contract Tests.

Validates complete elimination of all counterexample clusters identified in the
Round 12 review report:
- R12-01: S08, S10, S17, Open/Closed Boundary, S12, S13, S07, S14, S15
- R12-02: F06, F07, F08, Lease handoff and actual quota verification
- R12-03: L04, MCTS root valid_set state-isolation
"""

import unittest
import os
import shutil
import tempfile
import math
import numpy as np
from unittest.mock import patch, MagicMock

from gen_zero.gate.constraint_compiler import (
    ConstraintLinearProjectionCompiler,
    CompiledConstraintRule,
    Tristate,
)
from gen_zero.nanocore.fleet_scheduler import (
    NanoCoreFleetScheduler,
    FleetSchedulerConfig,
    FleetQuotaExceededError,
)
from gen_zero.runtime.base_nano_core import BaseNanoCore
from gen_zero.runtime.specialist_nano_core import DomainSpecialistNanoCore
from gen_zero.client import GenZeroClient


class TestRound12Contract(unittest.TestCase):
    """Formal test suite verifying all 14 counterexamples are cleared."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="gen_zero_r12_test_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # =========================================================================
    # R12-01: Constraint Semantics and Fail-Closed (S08, S10, S17, S12, S13, S07, S14, S15)
    # =========================================================================

    def test_r12_01_s08_declared_coordinate_alignment(self):
        """Probe S08: In declared coordinate mapping x=z[0], direct predicate equals projection predicate."""
        compiler = ConstraintLinearProjectionCompiler(latent_dim=32, seed=42)
        compiler.compile_rules([
            "FORBID EXECUTE IF x > 0",
        ])
        # Case 1: x = 1.0 (z[0] = 1.0) -> Predicate is True -> EXECUTE must be forbidden
        z_pos = np.zeros(32, dtype=np.float32)
        z_pos[0] = 1.0
        prop_map = compiler.project_latent_propositions(z_pos, schema_fingerprint=compiler.schema_fingerprint)
        self.assertIs(prop_map["x > 0"], Tristate.TRUE)

        verdict = compiler.solve_safest_action(
            candidate_utilities={"EXECUTE": 1.0, "HOLD": 0.5},
            z_latent=z_pos,
            fallback_safe_action="HOLD",
            schema_fingerprint=compiler.schema_fingerprint,
        )
        self.assertNotEqual(verdict.selected_action, "EXECUTE")
        self.assertEqual(verdict.selected_action, "HOLD")

    def test_r12_01_s10_non_zero_threshold_projection(self):
        """Probe S10: Non-zero threshold x > 10 is correctly handled with affine bias."""
        compiler = ConstraintLinearProjectionCompiler(latent_dim=32, seed=42)
        compiler.compile_rules([
            "FORBID EXECUTE IF x > 10",
        ])
        # x = 11.0 -> True
        z_11 = np.zeros(32, dtype=np.float32)
        z_11[0] = 11.0
        prop_map = compiler.project_latent_propositions(z_11, schema_fingerprint=compiler.schema_fingerprint)
        self.assertIs(prop_map["x > 10"], Tristate.TRUE)

        # x = 9.0 -> False
        z_9 = np.zeros(32, dtype=np.float32)
        z_9[0] = 9.0
        prop_map_9 = compiler.project_latent_propositions(z_9, schema_fingerprint=compiler.schema_fingerprint)
        self.assertIs(prop_map_9["x > 10"], Tristate.FALSE)

    def test_r12_01_strict_open_closed_boundary(self):
        """Verify strict inequality x > 0 at zero vector z=0 is False, while x >= 0 is True."""
        compiler = ConstraintLinearProjectionCompiler(latent_dim=32, seed=42)
        compiler.compile_rules([
            "FORBID ACT_GT IF x > 0",
            "FORBID ACT_GTE IF x >= 0",
        ])
        z_zero = np.zeros(32, dtype=np.float32)
        prop_map = compiler.project_latent_propositions(z_zero, schema_fingerprint=compiler.schema_fingerprint)
        self.assertIs(prop_map["x > 0"], Tristate.FALSE, "x > 0 at z=0 must be False")
        self.assertIs(prop_map["x >= 0"], Tristate.TRUE, "x >= 0 at z=0 must be True")

    def test_r12_01_s17_complementary_conditions_mutex(self):
        """Probe S17: Complementary conditions x > 10 and x <= 10 can NEVER both be True for any finite state."""
        compiler = ConstraintLinearProjectionCompiler(latent_dim=32, seed=42)
        compiler.compile_rules([
            "FORBID ACT_1 IF x > 10",
            "FORBID ACT_2 IF x <= 10",
        ])

        # Test across continuous range including around threshold
        sample_points = [-100.0, -1.0, 0.0, 9.999, 10.0, 10.0001, 15.0, 100.0]
        for val in sample_points:
            z = np.zeros(32, dtype=np.float32)
            z[0] = val
            prop_map = compiler.project_latent_propositions(z, schema_fingerprint=compiler.schema_fingerprint)
            p_gt = prop_map["x > 10"]
            p_le = prop_map["x <= 10"]
            # Both conditions can NEVER be True simultaneously
            self.assertFalse(p_gt is Tristate.TRUE and p_le is Tristate.TRUE, f"Contradiction: both True at x={val}")
            # Exactly one must be True
            self.assertTrue(p_gt is Tristate.TRUE or p_le is Tristate.TRUE, f"Neither True at x={val}")

    def test_r12_01_s12_s13_infinite_and_nan_fail_closed(self):
        """Probes S12 & S13: -inf, +inf, and NaN must Fail-Closed and not authorize unsafe actions."""
        compiler = ConstraintLinearProjectionCompiler(latent_dim=32, seed=42)
        compiler.compile_rules([
            "FORBID EXECUTE IF temp > 10",
            "ALLOW EXECUTE ONLY IF budget >= 10",
        ])

        # S12: temp = -inf must Fail-Closed and forbid EXECUTE
        v_s12 = compiler.solve_safest_action(
            candidate_utilities={"EXECUTE": 1.0, "SAFE_WAIT": 0.5},
            current_metrics={"temp": -math.inf},
        )
        self.assertNotEqual(v_s12.selected_action, "EXECUTE")

        # S13: budget = +inf must Fail-Closed and deny EXECUTE authorization
        v_s13 = compiler.solve_safest_action(
            candidate_utilities={"EXECUTE": 1.0, "SAFE_WAIT": 0.5},
            current_metrics={"budget": math.inf},
        )
        self.assertNotEqual(v_s13.selected_action, "EXECUTE")

        # NaN must also Fail-Closed
        v_nan = compiler.solve_safest_action(
            candidate_utilities={"EXECUTE": 1.0, "SAFE_WAIT": 0.5},
            current_metrics={"temp": math.nan},
        )
        self.assertNotEqual(v_nan.selected_action, "EXECUTE")

    def test_r12_01_s07_conflicting_requirements(self):
        """Probe S07: Simultaneously requiring mutually exclusive actions returns is_safe=False."""
        compiler = ConstraintLinearProjectionCompiler(latent_dim=32, seed=42)
        compiler.compile_rules([
            "REQUIRE A WHEN alert > 0",
            "REQUIRE B WHEN alert > 0",
        ])
        verdict = compiler.solve_safest_action(
            candidate_utilities={"A": 1.0, "B": 0.9, "HOLD": 0.1},
            current_metrics={"alert": 1.0},
        )
        self.assertFalse(verdict.is_safe)
        self.assertEqual(verdict.solver_status, "CONFLICTING_REQUIREMENTS")

    def test_r12_01_s14_fallback_action_validation(self):
        """Probe S14: When candidates are infeasible and fallback does not satisfy REQUIRE, is_safe=False."""
        compiler = ConstraintLinearProjectionCompiler(latent_dim=32, seed=42)
        compiler.compile_rules([
            "REQUIRE STOP WHEN critical > 0",
        ])
        verdict = compiler.solve_safest_action(
            candidate_utilities={"EXECUTE": 1.0},
            current_metrics={"critical": 1.0},
            fallback_safe_action="HOLD",
        )
        self.assertFalse(verdict.is_safe)
        self.assertEqual(verdict.solver_status, "UNSAFE_NO_FEASIBLE_ACTIONS")

    def test_r12_01_s15_colon_action_regex_and_syntax_fail_closed(self):
        """Probe S15: Colons in action names are parsed, and invalid syntax fails closed."""
        compiler = ConstraintLinearProjectionCompiler(latent_dim=32, seed=42)
        compiler.compile_rules([
            "FORBID tool:execute IF danger > 0",
        ])
        v = compiler.solve_safest_action(
            candidate_utilities={"tool:execute": 1.0, "tool:inspect": 0.5},
            current_metrics={"danger": 1.0},
        )
        self.assertEqual(v.selected_action, "tool:inspect")
        self.assertTrue(v.is_safe)

        # Unsupported syntax fails closed
        bad_compiler = ConstraintLinearProjectionCompiler(latent_dim=32, seed=42)
        bad_compiler.compile_rules([
            "UNSUPPORTED GRAMMAR RULE",
        ])
        v_bad = bad_compiler.solve_safest_action(
            candidate_utilities={"tool:execute": 1.0},
            fallback_safe_action="HOLD",
        )
        self.assertFalse(v_bad.is_safe)
        self.assertEqual(v_bad.solver_status, "UNSUPPORTED_SYNTAX_FAIL_CLOSED")

    # =========================================================================
    # R12-02: Fleet Scheduler and Quota Enforcement (F06, F07, F08)
    # =========================================================================

    def test_r12_02_f07_atomic_lease_no_window(self):
        """Probe F07: acquire_lease atomically increments active_leases within lock."""
        config = FleetSchedulerConfig(max_resident_cores=2, storage_dir=self.temp_dir)
        scheduler = NanoCoreFleetScheduler(config)
        core = DomainSpecialistNanoCore(domain="test_f07", state_dim=32, candidate_dim=32, embed_dim=16, seed=1)
        scheduler.register_instance("core_a", core, persist=True)

        lease = scheduler.acquire_lease("core_a")
        self.assertIs(lease.core, core)
        desc = scheduler.get_core_descriptor("core_a")
        self.assertEqual(desc.active_leases, 1)

        # Release lease; descriptors are snapshots, so re-read the ledger view.
        scheduler.release_lease(lease)
        self.assertEqual(scheduler.get_core_descriptor("core_a").active_leases, 0)

    def test_r12_02_f06_cold_load_actual_bytes_quota(self):
        """Probe F06: a checkpoint declaring more than the budget raises
        FleetQuotaExceededError at admission, before any loader runs."""
        config = FleetSchedulerConfig(
            max_resident_cores=5,
            max_resident_bytes=1000,
            storage_dir=self.temp_dir,
        )
        scheduler = NanoCoreFleetScheduler(config)
        core = DomainSpecialistNanoCore(domain="test_f06", state_dim=32, candidate_dim=32, embed_dim=16, seed=2)
        core.memory_footprint_bytes = lambda: 2000
        path = os.path.join(self.temp_dir, "core_f06.zst")
        core.save_checkpoint(path, compress=True)
        scheduler.register_checkpoint("core_f06", path)

        with patch.object(BaseNanoCore, "load_checkpoint") as loader:
            with self.assertRaises(FleetQuotaExceededError):
                scheduler.acquire_core("core_f06")
            loader.assert_not_called()

    def test_r12_02_f08_evict_core_respects_active_leases(self):
        """Probe F08: evict_core(force=False) must never evict cores with active_leases > 0."""
        config = FleetSchedulerConfig(max_resident_cores=2, storage_dir=self.temp_dir)
        scheduler = NanoCoreFleetScheduler(config)
        core = DomainSpecialistNanoCore(domain="test_f08", state_dim=32, candidate_dim=32, embed_dim=16, seed=3)
        scheduler.register_instance("core_f08", core, persist=True)

        with scheduler.lease_core("core_f08"):
            # Attempt to evict active lease without force
            evicted = scheduler.evict_core("core_f08", force=False)
            self.assertFalse(evicted)
            self.assertTrue(scheduler.is_resident("core_f08"))

        # Once lease is released, evict succeeds
        evicted_after = scheduler.evict_core("core_f08", force=False)
        self.assertTrue(evicted_after)
        self.assertFalse(scheduler.is_resident("core_f08"))

    # =========================================================================
    # R12-03: MCTS Root Valid Set Isolation (L04)
    # =========================================================================

    def test_r12_03_l04_mcts_root_valid_set_isolation(self):
        """Probe L04: MCTS search preserves root valid_set and does not intersect with deeper child states.

        T3-M01: a single-candidate MCTS request must actually run the trajectory
        rollout (never shortcut past it), so trans_fn now returns a real
        (next_state, reward, done) triple exactly as the real transition_fn contract
        requires, and legal_actions_fn is wired through so the child state's
        different legal action set (["ENTER"], not ["OPEN"]) is genuinely reachable
        and exercised, matching what this probe claims to verify.
        """
        client = GenZeroClient()

        # Define root state s0 and child state s1
        s0 = "door_closed"
        s1 = "door_opened"

        def mock_actions(s):
            if s == s0:
                return ["OPEN"]
            return ["ENTER"]

        def mock_trans(s, a):
            if s == s0 and a == "OPEN":
                return s1, 0.0, False
            return s, 0.0, True

        # Execute decision with MCTS planner
        probs, val, best_act, meta = client._execute_expert_distribution(
            expert_name="mcts",
            state=s0,
            candidates=["OPEN"],
            trans_fn=mock_trans,
            legal_actions_fn=mock_actions,
        )

        # Meta valid_set at root MUST contain root valid actions ["OPEN"], NOT truncated to empty set!
        self.assertIn("valid_set", meta)
        self.assertEqual(meta["valid_set"], ["OPEN"])
        self.assertEqual(best_act, "OPEN")


if __name__ == "__main__":
    unittest.main()
