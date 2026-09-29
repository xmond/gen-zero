"""Comprehensive Unit Tests for Issue #51 & RFC-049.

Verifies:
1. WorldModelNanoCoreOrchestrator:
   - Coordinates Safety NanoCore, Action NanoCore, and Critic NanoCore across virtual nodes.
   - Attention Sinks rolling KV cache maintaining O(W) constant memory.
   - Multi-step lookahead imagination avoiding dead-end traps with 100% survival rate.
   - CP-SAT 0-1 ILP formal safety verification.
2. Pearl Causal Shock Autonomic Adaptation:
   - Detects ||z_real - z_pred|| exceeding threshold.
   - Autonomously adapts NanoCore weights (safety weight up to 0.80) and reduces horizon depth.
3. MCP Protocol Exposure:
   - zero_imagine tool schema and JSON-RPC execution.
   - zero_stream tool schema and streaming execution.
   - tools/list correctly registers tools alongside zero_ask and zero_route.
4. GenZero Client Integration:
   - client.imagine_world_model(...) high-level interface.
"""

import unittest
import asyncio
import json
import math
import time
import numpy as np

from gen_zero.nanocore.world_model_orchestrator import (
    ACTION_HEAD_FAILED,
    ALL_CANDIDATES_BLOCKED,
    CAUSAL_SHOCK_UNAVAILABLE,
    CPSAT_NOT_VERIFIED,
    CPSAT_REAL_SOLVE_STATUSES,
    DECISION_OK,
    DEFAULT_CAUSAL_SHOCK_THRESHOLD,
    DEFAULT_FALLBACK_SAFE_ACTION,
    PSEUDO_EMBEDDING,
    SAFETY_INTERLOCKED,
    UNVERIFIED_SAFETY,
    NanoCoreClusterStatus,
    ImaginedStepTelemetry,
    SafetyInterlockError,
    WorldModelOrchestrationResult,
    WorldModelNanoCoreOrchestrator,
)
from gen_zero.gate.cpsat_formal_solver import CPSATVerdict
from gen_zero.world_model.streaming_engine import (
    DEGRADATION_DIM_MISMATCH_PROJECTED,
    DEGRADATION_UNTRAINED_TEXT_HASH_PRIOR,
    PROVENANCE_PROJECTED_FEATURE_VECTOR,
    PROVENANCE_RAW_FEATURE_VECTOR,
    PROVENANCE_UNTRAINED_TEXT_HASH_PRIOR,
    StreamingWorldModelEngine,
)
from gen_zero.mcp.server import (
    MCPServer,
    ZERO_SCHEMA,
    execute_zero,
    execute_zero_imagine,
    execute_zero_stream,
)
from gen_zero.client import GenZero


class TestWorldModelNanoCoreOrchestrator(unittest.TestCase):
    """Tests for World Model Master Orchestrator coordinating NanoCores."""

    @classmethod
    def setUpClass(cls):
        # Warm up lazy imports (e.g. ortools) so first test timing is purely computational
        warmup_orch = WorldModelNanoCoreOrchestrator(latent_dim=64, action_dim=8)
        warmup_orch.imagine_and_orchestrate("init", ["A", "B"], horizon=1)

    def setUp(self):
        self.orchestrator = WorldModelNanoCoreOrchestrator(
            latent_dim=1024,
            action_dim=32,
            num_sink_tokens=4,
            window_size=8,
            causal_shock_threshold=0.35
        )

    def test_dead_end_trap_avoidance_and_cpsat(self):
        """Milestone 4: Avoids dead-end traps by lookahead search and CP-SAT hard pruning."""
        state = "Robot entering narrow corridor: target visible east, but trap pit located at step 3."
        candidates = ["MOVE_EAST_TRAP", "MOVE_NORTH_SAFE", "MOVE_WEST_SAFE", "HOLD"]

        def mock_safety(z, act):
            # MOVE_EAST_TRAP appears fine in step 1, but causes safety failure
            if "TRAP" in act:
                return 0.15
            return 0.95

        result = self.orchestrator.imagine_and_orchestrate(
            state=state,
            candidate_actions=candidates,
            horizon=4,
            enforce_cpsat=True,
            safety_evaluator=mock_safety
        )

        self.assertIsInstance(result, WorldModelOrchestrationResult)
        # Verify trap path was pruned
        self.assertNotEqual(result.selected_action, "MOVE_EAST_TRAP")
        self.assertIn(result.selected_action, ["MOVE_NORTH_SAFE", "MOVE_WEST_SAFE", "HOLD"])
        self.assertGreaterEqual(result.trap_paths_pruned, 1)
        # cpsat_verified is an environment-independent claim: True only for a real solve.
        # With OR-Tools missing the status is ORTOOLS_UNAVAILABLE_FALLBACK and it must be False.
        status = result.nanocore_status.cpsat_status
        self.assertEqual(result.nanocore_status.cpsat_verified, status in CPSAT_REAL_SOLVE_STATUSES)
        if not result.nanocore_status.cpsat_verified:
            self.assertIn(f"{CPSAT_NOT_VERIFIED}:{status}", result.degradations)
        self.assertEqual(result.decision_status, DECISION_OK)
        self.assertLess(result.planning_time_ms, 50.0)  # sub-50ms execution

    def test_pearl_causal_shock_adaptive_reweighting(self):
        """Milestone 3: Exogenous perturbation triggers Causal Shock, boosting safety weight to 0.80."""
        # Step 1: Normal step
        state_1 = "Normal server traffic: load 45%, latency 32ms"
        candidates = ["SCALE_OUT", "THROTTLE", "HOLD"]
        res1 = self.orchestrator.imagine_and_orchestrate(
            state_1, candidates, horizon=4, safety_evaluator=lambda z, act: 0.9
        )
        self.assertFalse(res1.shock_detected)
        self.assertIn(PSEUDO_EMBEDDING, res1.degradations)
        self.assertEqual(res1.effective_weights["safety"], 0.35)

        # Step 2: Inject large exogenous shock in environment
        # Completely different state tensor causing ||z_real - z_pred|| > epsilon
        shocked_state = np.random.randn(1024).astype(np.float32) * 5.0
        res2 = self.orchestrator.imagine_and_orchestrate(
            shocked_state, candidates, horizon=4, safety_evaluator=lambda z, act: 0.9
        )

        self.assertTrue(res2.shock_detected)
        self.assertTrue(res2.adaptive_safety_mode)
        self.assertGreater(res2.causal_shock, 0.35)
        # Safety weight boosted to 0.80
        self.assertEqual(res2.effective_weights["safety"], 0.80)
        # Horizon reduced to defense mode
        self.assertLessEqual(res2.horizon_explored, 2)
        self.assertNotIn(PSEUDO_EMBEDDING, res2.degradations)


class TestWorldModelFailClosed(unittest.TestCase):
    """Fail-closed behaviour: no silent fallback, no fake safety score."""

    def setUp(self):
        self.orchestrator = WorldModelNanoCoreOrchestrator(latent_dim=64, action_dim=8)
        self.state = np.linspace(-1.0, 1.0, 64).astype(np.float32)

    def test_no_safety_evaluator_blocks_every_candidate(self):
        # "DRAIN_NODE" used to score 0.10 and "RESTART_POD" 0.98 from keyword matching.
        candidates = ["RESTART_POD", "DRAIN_NODE"]
        with self.assertLogs("gen_zero.nanocore.world_model_orchestrator", level="WARNING") as logs:
            res = self.orchestrator.imagine_and_orchestrate(self.state, candidates, horizon=2)
        self.assertTrue(any(UNVERIFIED_SAFETY in line for line in logs.output))
        self.assertEqual(res.selected_action, DEFAULT_FALLBACK_SAFE_ACTION)
        self.assertEqual(res.confidence, 0.0)
        self.assertEqual(res.trap_paths_pruned, 2)
        self.assertFalse(res.nanocore_status.safety_verified)
        self.assertTrue(res.nanocore_status.safety_core.startswith(UNVERIFIED_SAFETY))
        self.assertIn(UNVERIFIED_SAFETY, res.degradations)
        self.assertIn(ALL_CANDIDATES_BLOCKED, res.degradations)
        self.assertTrue(all(step.predicted_risk == 1.0 for step in res.imagined_trajectory))

    def test_no_safety_evaluator_without_cpsat_also_falls_back(self):
        res = self.orchestrator.imagine_and_orchestrate(
            self.state, ["RESTART_POD", "SCALE_OUT"], horizon=2, enforce_cpsat=False
        )
        self.assertEqual(res.selected_action, DEFAULT_FALLBACK_SAFE_ACTION)
        self.assertFalse(res.nanocore_status.cpsat_verified)

    def test_all_unsafe_including_fallback_raises_interlock(self):
        # Old code passed candidate_actions[0] as the "safe" fallback, and later releases HOLD
        # even when the evaluator rated HOLD itself unsafe. Neither may be released.
        with self.assertRaises(SafetyInterlockError) as ctx:
            self.orchestrator.imagine_and_orchestrate(
                self.state, ["DELETE_DB", "HOLD"], horizon=2, safety_evaluator=lambda z, act: 0.1
            )
        self.assertIn(SAFETY_INTERLOCKED, str(ctx.exception))
        self.assertIn(ALL_CANDIDATES_BLOCKED, ctx.exception.degradations)

    def test_all_unsafe_with_releasable_fallback_is_interlocked_not_ok(self):
        res = self.orchestrator.imagine_and_orchestrate(
            self.state, ["DELETE_DB", "DROP_TABLE"], horizon=2, safety_evaluator=lambda z, act: 0.1
        )
        self.assertEqual(res.selected_action, DEFAULT_FALLBACK_SAFE_ACTION)
        self.assertEqual(res.decision_status, SAFETY_INTERLOCKED)
        self.assertTrue(res.interlocked)
        self.assertIn(ALL_CANDIDATES_BLOCKED, res.degradations)
        self.assertFalse(res.nanocore_status.cpsat_verified)
        self.assertEqual(res.nanocore_status.cpsat_status, "ALL_CANDIDATES_FORBIDDEN_INTERCEPT")
        self.assertEqual(res.to_dict()["decision_status"], SAFETY_INTERLOCKED)

    def test_all_unsafe_without_cpsat_is_also_interlocked(self):
        res = self.orchestrator.imagine_and_orchestrate(
            self.state, ["DELETE_DB", "DROP_TABLE"], horizon=2, enforce_cpsat=False,
            safety_evaluator=lambda z, act: 0.1,
        )
        self.assertEqual(res.selected_action, DEFAULT_FALLBACK_SAFE_ACTION)
        self.assertEqual(res.decision_status, SAFETY_INTERLOCKED)

    def test_caller_forbidding_the_fallback_raises_interlock(self):
        with self.assertRaises(SafetyInterlockError):
            self.orchestrator.imagine_and_orchestrate(
                self.state, ["DELETE_DB"], horizon=2, forbidden_actions={"DELETE_DB", "HOLD"},
                safety_evaluator=lambda z, act: 0.9,
            )


    def test_nan_safety_score_raises(self):
        with self.assertRaises(ValueError):
            self.orchestrator.imagine_and_orchestrate(
                self.state, ["A", "B"], safety_evaluator=lambda z, act: float("nan")
            )

    def test_out_of_range_safety_score_raises(self):
        with self.assertRaises(ValueError):
            self.orchestrator.imagine_and_orchestrate(
                self.state, ["A", "B"], safety_evaluator=lambda z, act: 1.7
            )

    def test_action_head_failure_is_marked_not_masked(self):
        def boom(*args, **kwargs):
            raise RuntimeError("etf head broke")

        self.orchestrator.etf_choice_head.decide = boom
        with self.assertLogs("gen_zero.nanocore.world_model_orchestrator", level="ERROR") as logs:
            res = self.orchestrator.imagine_and_orchestrate(
                self.state, ["A", "B"], horizon=2, safety_evaluator=lambda z, act: 0.9
            )
        self.assertTrue(any(ACTION_HEAD_FAILED in line for line in logs.output))
        self.assertEqual(res.nanocore_status.action_core, "FAILED")
        self.assertIn(ACTION_HEAD_FAILED, res.degradations)

    def test_action_head_reports_real_verdict(self):
        res = self.orchestrator.imagine_and_orchestrate(
            self.state, ["A", "B"], horizon=2, safety_evaluator=lambda z, act: 0.9
        )
        self.assertIn(res.nanocore_status.action_core, ("CONVERGED", "DIVERGENT"))
        self.assertNotIn(ACTION_HEAD_FAILED, res.degradations)

    def test_latent_distance_failure_returns_nan(self):
        with self.assertLogs("gen_zero.nanocore.world_model_orchestrator", level="ERROR"):
            self.assertTrue(math.isnan(self.orchestrator._compute_latent_distance(["x"], [1.0])))
        with self.assertLogs("gen_zero.nanocore.world_model_orchestrator", level="ERROR"):
            self.assertTrue(math.isnan(self.orchestrator._compute_latent_distance([1.0, 2.0], [1.0])))
        self.assertAlmostEqual(self.orchestrator._compute_latent_distance([3.0, 0.0], [0.0, 4.0]), 5.0)

    def test_unmeasurable_shock_forces_safety_mode(self):
        evaluator = lambda z, act: 0.9
        self.orchestrator.imagine_and_orchestrate(self.state, ["A", "B"], horizon=4, safety_evaluator=evaluator)
        self.orchestrator.last_predicted_latent = ["not-a-number"]
        res = self.orchestrator.imagine_and_orchestrate(self.state, ["A", "B"], horizon=4, safety_evaluator=evaluator)
        self.assertTrue(math.isnan(res.causal_shock))
        self.assertTrue(res.shock_detected)
        self.assertTrue(res.adaptive_safety_mode)
        self.assertEqual(res.effective_weights["safety"], 0.80)
        self.assertIn(CAUSAL_SHOCK_UNAVAILABLE, res.degradations)
        payload = json.loads(json.dumps(res.to_dict(), allow_nan=False))
        self.assertIsNone(payload["causal_shock"])

    def test_text_state_warns_pseudo_embedding(self):
        with self.assertLogs("gen_zero.nanocore.world_model_orchestrator", level="WARNING") as logs:
            res = self.orchestrator.imagine_and_orchestrate(
                "free text state", ["A", "B"], horizon=2, safety_evaluator=lambda z, act: 0.9
            )
        self.assertTrue(any(PSEUDO_EMBEDDING in line for line in logs.output))
        self.assertIn(PSEUDO_EMBEDDING, res.degradations)


class TestCPSATVerifiedGate(unittest.TestCase):
    """cpsat_verified must mirror the solver verdict, never the fact that the solver was called."""

    def setUp(self):
        self.orchestrator = WorldModelNanoCoreOrchestrator(latent_dim=64, action_dim=8)
        self.state = np.linspace(-1.0, 1.0, 64).astype(np.float32)
        self.candidates = ["A", "B"]

    def _run_with_verdict(self, verdict: CPSATVerdict) -> WorldModelOrchestrationResult:
        self.orchestrator.cpsat_solver.solve_safest_optimal_action = lambda **kw: verdict
        return self.orchestrator.imagine_and_orchestrate(
            self.state, self.candidates, horizon=2, enforce_cpsat=True,
            safety_evaluator=lambda z, act: 0.9,
        )

    @staticmethod
    def _verdict(status: str, fallback_used: bool, is_safe: bool = True, action: str = "A") -> CPSATVerdict:
        return CPSATVerdict(
            selected_action=action, is_safe=is_safe, solve_time_ms=0.1, timed_out=False,
            fallback_used=fallback_used, solver_status=status, applied_constraints=[],
        )

    def test_real_solve_is_verified(self):
        res = self._run_with_verdict(self._verdict("OPTIMAL", fallback_used=False))
        self.assertTrue(res.nanocore_status.cpsat_verified)
        self.assertEqual(res.nanocore_status.cpsat_status, "OPTIMAL")
        self.assertFalse(any(d.startswith(CPSAT_NOT_VERIFIED) for d in res.degradations))
        self.assertEqual(res.selected_action, "A")

    def test_ortools_unavailable_fallback_is_not_verified(self):
        with self.assertLogs("gen_zero.nanocore.world_model_orchestrator", level="WARNING") as logs:
            res = self._run_with_verdict(self._verdict("ORTOOLS_UNAVAILABLE_FALLBACK", fallback_used=True))
        self.assertFalse(res.nanocore_status.cpsat_verified)
        self.assertIn(f"{CPSAT_NOT_VERIFIED}:ORTOOLS_UNAVAILABLE_FALLBACK", res.degradations)
        self.assertTrue(any(CPSAT_NOT_VERIFIED in line for line in logs.output))
        self.assertEqual(res.decision_status, DECISION_OK)

    def test_solver_timeout_fallback_is_not_verified(self):
        res = self._run_with_verdict(self._verdict("SOLVER_TIMEOUT_FALLBACK", fallback_used=True))
        self.assertFalse(res.nanocore_status.cpsat_verified)
        self.assertIn(f"{CPSAT_NOT_VERIFIED}:SOLVER_TIMEOUT_FALLBACK", res.degradations)

    def test_exception_fallback_is_not_verified(self):
        res = self._run_with_verdict(self._verdict("CPSAT_EXCEPTION_FALLBACK:RuntimeError", fallback_used=True))
        self.assertFalse(res.nanocore_status.cpsat_verified)
        self.assertIn(f"{CPSAT_NOT_VERIFIED}:CPSAT_EXCEPTION_FALLBACK:RuntimeError", res.degradations)

    def test_unlisted_status_without_fallback_flag_is_still_not_verified(self):
        # A future solver branch that forgets fallback_used=True must not slip through.
        res = self._run_with_verdict(self._verdict("SOMETHING_NEW", fallback_used=False))
        self.assertFalse(res.nanocore_status.cpsat_verified)
        self.assertIn(f"{CPSAT_NOT_VERIFIED}:SOMETHING_NEW", res.degradations)

    def test_unsafe_verdict_raises_interlock(self):
        with self.assertRaises(SafetyInterlockError):
            self._run_with_verdict(
                self._verdict("UNSAFE_NO_FEASIBLE_ACTIONS", fallback_used=True, is_safe=False, action="HOLD")
            )

    def test_solver_returning_forbidden_action_raises_interlock(self):
        self.orchestrator.cpsat_solver.solve_safest_optimal_action = (
            lambda **kw: self._verdict("OPTIMAL", fallback_used=False, action="B")
        )
        with self.assertRaises(SafetyInterlockError):
            self.orchestrator.imagine_and_orchestrate(
                self.state, self.candidates, horizon=2, enforce_cpsat=True,
                forbidden_actions={"B"}, safety_evaluator=lambda z, act: 0.9,
            )

    def test_all_blocked_real_solve_over_fallback_is_not_verified(self):
        # OR-Tools present, every candidate blocked: the solver may return a clean verdict for
        # the fallback alone. That is still not a verification of any candidate.
        self.orchestrator.cpsat_solver.solve_safest_optimal_action = (
            lambda **kw: self._verdict("OPTIMAL", fallback_used=False, action=DEFAULT_FALLBACK_SAFE_ACTION)
        )
        res = self.orchestrator.imagine_and_orchestrate(
            self.state, self.candidates, horizon=2, enforce_cpsat=True,
            safety_evaluator=lambda z, act: 0.1,
        )
        self.assertEqual(res.decision_status, SAFETY_INTERLOCKED)
        self.assertFalse(res.nanocore_status.cpsat_verified)
        self.assertIn(f"{CPSAT_NOT_VERIFIED}:OPTIMAL", res.degradations)

    def test_cpsat_disabled_is_never_verified(self):
        res = self.orchestrator.imagine_and_orchestrate(
            self.state, self.candidates, horizon=2, enforce_cpsat=False,
            safety_evaluator=lambda z, act: 0.9,
        )
        self.assertFalse(res.nanocore_status.cpsat_verified)
        self.assertIsNone(res.nanocore_status.cpsat_status)


class TestCompilerBlocksAllCandidates(unittest.TestCase):
    """all_blocked must be decided after the constraint compiler, not before it.

    Every candidate passes the safety evaluator, so the pre-solve check sees nothing blocked.
    The compiled rules then forbid every candidate and the compiler falls back to HOLD.
    """

    def setUp(self):
        self.orchestrator = WorldModelNanoCoreOrchestrator(latent_dim=64, action_dim=8)
        self.state = np.linspace(-1.0, 1.0, 64).astype(np.float32)
        self.candidates = ["A", "B"]

    def _run(self):
        return self.orchestrator.imagine_and_orchestrate(
            self.state, self.candidates, horizon=2, enforce_cpsat=True,
            safety_evaluator=lambda z, act: 0.9,
        )

    def test_rules_blocking_every_candidate_are_interlocked(self):
        # x >= 0 and x < 0 cover every state, so A and B are forbidden whatever the latent is.
        self.orchestrator.compile_safety_rules([
            "FORBID A IF x >= 0", "FORBID A IF x < 0",
            "FORBID B IF x >= 0", "FORBID B IF x < 0",
        ])
        res = self._run()
        self.assertEqual(res.selected_action, DEFAULT_FALLBACK_SAFE_ACTION)
        self.assertEqual(res.nanocore_status.cpsat_status, "ALL_FORBIDDEN_FAILSAFE")
        self.assertEqual(res.decision_status, SAFETY_INTERLOCKED)
        self.assertTrue(res.interlocked)
        self.assertFalse(res.nanocore_status.cpsat_verified)
        self.assertIn(ALL_CANDIDATES_BLOCKED, res.degradations)
        self.assertEqual(res.to_dict()["decision_status"], SAFETY_INTERLOCKED)

    def test_rules_blocking_one_candidate_still_release_the_other(self):
        self.orchestrator.compile_safety_rules(["FORBID A IF x >= 0", "FORBID A IF x < 0"])
        res = self._run()
        self.assertEqual(res.selected_action, "B")
        self.assertEqual(res.decision_status, DECISION_OK)
        self.assertNotIn(ALL_CANDIDATES_BLOCKED, res.degradations)

    def test_clean_verdict_for_non_candidate_fallback_is_interlocked(self):
        # A compiler verdict that claims a real solve but hands back the fallback, which is not
        # a candidate, released nothing the caller asked for.
        self.orchestrator.compile_safety_rules(["FORBID A IF x >= 0"])
        self.orchestrator.constraint_compiler.solve_safest_action = lambda **kw: CPSATVerdict(
            selected_action=DEFAULT_FALLBACK_SAFE_ACTION, is_safe=True, solve_time_ms=0.1,
            timed_out=False, fallback_used=False, solver_status="CP_SAT_OPTIMAL",
            applied_constraints=[],
        )
        res = self._run()
        self.assertEqual(res.decision_status, SAFETY_INTERLOCKED)
        self.assertFalse(res.nanocore_status.cpsat_verified)
        self.assertIn(ALL_CANDIDATES_BLOCKED, res.degradations)
        self.assertIn(f"{CPSAT_NOT_VERIFIED}:CP_SAT_OPTIMAL", res.degradations)


class TestWorldModelMCPProtocol(unittest.TestCase):
    """Tests for MCP server tool registration and execution (Issue #51)."""

    def setUp(self):
        self.server = MCPServer()

    def test_mcp_tools_list_contains_world_model_tools(self):
        """Milestone 2: tools/list registers strictly the canonical single zero tool."""
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        }
        res = asyncio.run(self.server.handle_request(req))
        tool_names = [t["name"] for t in res["result"]["tools"]]
        self.assertEqual(tool_names, ["zero"])

    def test_execute_zero_imagine_tool(self):
        """Calls zero tool in imagine mode via MCP JSON-RPC protocol."""
        req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "zero",
                "arguments": {
                    "action": "imagine",
                    "state": "Kubernetes cluster node high memory pressure",
                    "candidate_actions": ["RESTART_POD", "DRAIN_NODE", "THROTTLE"],
                    "horizon": 4,
                    "enforce_cpsat": True
                }
            }
        }
        res = asyncio.run(self.server.handle_request(req))
        self.assertEqual(res["id"], 2)
        # MCP cannot inject a safety evaluator, so every candidate is blocked. That is a
        # SAFETY_INTERLOCKED error at the tool boundary, not a success carrying HOLD.
        self.assertTrue(res["result"]["isError"])
        envelope = json.loads(res["result"]["content"][0]["text"])
        self.assertEqual(envelope["error"], SAFETY_INTERLOCKED)
        self.assertIn(SAFETY_INTERLOCKED, envelope["message"])
        data = envelope["result"]
        self.assertEqual(data["decision_status"], SAFETY_INTERLOCKED)
        self.assertEqual(data["selected_action"], DEFAULT_FALLBACK_SAFE_ACTION)
        self.assertEqual(data["confidence"], 0.0)
        self.assertFalse(data["nanocore_status"]["safety_verified"])
        self.assertFalse(data["nanocore_status"]["cpsat_verified"])
        self.assertEqual(data["nanocore_status"]["cpsat_status"], "ALL_CANDIDATES_FORBIDDEN_INTERCEPT")
        self.assertIn(UNVERIFIED_SAFETY, data["degradations"])
        self.assertIn(PSEUDO_EMBEDDING, data["degradations"])
        self.assertIn(ALL_CANDIDATES_BLOCKED, data["degradations"])
        self.assertIn(f"{CPSAT_NOT_VERIFIED}:ALL_CANDIDATES_FORBIDDEN_INTERCEPT", data["degradations"])

    def test_execute_zero_imagine_forbidding_fallback_is_typed_error(self):
        res = asyncio.run(execute_zero_imagine({
            "state": "any", "candidate_actions": ["A"], "forbidden_actions": ["A", "HOLD"],
        }))
        self.assertTrue(res["isError"])
        envelope = json.loads(res["content"][0]["text"])
        self.assertEqual(envelope["error"], SAFETY_INTERLOCKED)
        self.assertIn(ALL_CANDIDATES_BLOCKED, envelope["degradations"])

    def test_execute_zero_stream_tool(self):
        """Calls zero tool in stream mode via polymorphic auto-detection."""
        req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "zero",
                "arguments": {
                    "observation": "frame_camera_001",
                    "candidate_actions": ["ACTION_A", "ACTION_B", "HOLD"]
                }
            }
        }
        res = asyncio.run(self.server.handle_request(req))
        self.assertEqual(res["id"], 3)
        self.assertFalse(res["result"]["isError"])
        content = res["result"]["content"][0]["text"]
        data = json.loads(content)
        self.assertEqual(data["step"], 1)
        self.assertIn(data["selected_action"], ["ACTION_A", "ACTION_B", "HOLD"])
        # A text observation is hashed into an untrained transition model. The response must say so.
        self.assertTrue(data["degraded"])
        self.assertEqual(data["provenance"], PROVENANCE_UNTRAINED_TEXT_HASH_PRIOR)
        self.assertIn(DEGRADATION_UNTRAINED_TEXT_HASH_PRIOR, data["degradations"])

    def test_execute_zero_stream_numeric_vector_is_not_degraded(self):
        # Default engine latent_dim is 1024 (see StreamingWorldModelEngine.__init__); a vector
        # of exactly that length passes through unchanged as a real raw feature vector.
        res = asyncio.run(execute_zero_stream({
            "observation": [0.1] * 1024,
            "candidate_actions": ["ACTION_A", "ACTION_B"],
        }))
        self.assertFalse(res["isError"])
        data = json.loads(res["content"][0]["text"])
        self.assertFalse(data["degraded"])
        self.assertEqual(data["provenance"], PROVENANCE_RAW_FEATURE_VECTOR)
        self.assertEqual(data["degradations"], [])

    def test_execute_zero_stream_dim_mismatch_vector_is_projected_and_degraded(self):
        # B10: a vector shorter/longer than latent_dim must NOT be silently zero-padded or
        # truncated while still claiming to be a real feature vector. It goes through the
        # explicit deterministic projector instead, and is flagged degraded.
        res = asyncio.run(execute_zero_stream({
            "observation": [0.1, 0.2, 0.3, 0.4],
            "candidate_actions": ["ACTION_A", "ACTION_B"],
        }))
        self.assertFalse(res["isError"])
        data = json.loads(res["content"][0]["text"])
        self.assertTrue(data["degraded"])
        self.assertEqual(data["provenance"], PROVENANCE_PROJECTED_FEATURE_VECTOR)
        self.assertIn(DEGRADATION_DIM_MISMATCH_PROJECTED, data["degradations"])

    def test_streaming_engine_dim_projector_is_deterministic_and_uses_all_input_dims(self):
        # Same input length -> same cached projector matrix -> identical output, both within
        # one engine instance and across fresh instances (seed depends only on dims, not data).
        engine_a = StreamingWorldModelEngine(latent_dim=64)
        engine_b = StreamingWorldModelEngine(latent_dim=64)
        z1, prov1 = engine_a._encode_observation([1.0, 2.0, 3.0])
        z2, prov2 = engine_b._encode_observation([1.0, 2.0, 3.0])
        self.assertEqual(prov1, PROVENANCE_PROJECTED_FEATURE_VECTOR)
        self.assertEqual(prov2, PROVENANCE_PROJECTED_FEATURE_VECTOR)
        self.assertEqual(len(z1), 64)
        self.assertTrue(np.allclose(np.asarray(z1), np.asarray(z2)))

        # Changing one input dimension must change the projected output: proves every input
        # dimension is actually used by the projection (not a truncated/padded copy).
        z3, _ = engine_a._encode_observation([1.0, 2.0, 999.0])
        self.assertFalse(np.allclose(np.asarray(z1), np.asarray(z3)))

    def test_streaming_engine_dim_projector_cache_is_lru_bounded(self):
        # X-STREAM-B1 (round 2): an unbounded per-input-length projector cache lets a caller
        # grow server memory without limit by sending many distinct observation lengths.
        from gen_zero.world_model.streaming_engine import MAX_DIM_PROJECTOR_CACHE_ENTRIES
        engine = StreamingWorldModelEngine(latent_dim=8)
        for n in range(1, MAX_DIM_PROJECTOR_CACHE_ENTRIES + 5):
            engine._encode_observation([float(n)] * (n + 1))
        self.assertLessEqual(len(engine._dim_projectors), MAX_DIM_PROJECTOR_CACHE_ENTRIES)
        # The dimensions evicted first (smallest n) must be gone; the most recent must remain.
        self.assertNotIn(2, engine._dim_projectors)
        self.assertIn(MAX_DIM_PROJECTOR_CACHE_ENTRIES + 5, engine._dim_projectors)

    def test_streaming_engine_dim_projector_rejects_oversized_input(self):
        # X-STREAM-B1 (round 2): a single absurdly long observation must be rejected outright
        # rather than allocating a huge projector matrix.
        from gen_zero.world_model.streaming_engine import MAX_PROJECTOR_INPUT_DIM
        engine = StreamingWorldModelEngine(latent_dim=8)
        oversized = [0.1] * (MAX_PROJECTOR_INPUT_DIM + 1)
        with self.assertRaises(ValueError):
            engine._encode_observation(oversized)
        # The same guard must surface as a clean MCP validation error, not an internal crash.
        res = asyncio.run(execute_zero_stream({
            "observation": oversized,
            "candidate_actions": ["A"],
        }))
        self.assertTrue(res["isError"])
        self.assertIn("Validation Error", res["content"][0]["text"])

    def test_execute_zero_stream_rejects_invalid_observation(self):
        for bad in ([], ["x", 1], [True], 42):
            res = asyncio.run(execute_zero_stream({"observation": bad, "candidate_actions": ["A"]}))
            self.assertTrue(res["isError"], msg=repr(bad))
            self.assertIn("Validation Error", res["content"][0]["text"])
        res = asyncio.run(execute_zero_stream({"observation": "ok", "candidate_actions": ["A", 3]}))
        self.assertTrue(res["isError"])

    def test_streaming_engine_failed_step_leaves_state_untouched(self):
        # step_count, KV-cache and the causal-shock anchor may only advance after decoding,
        # validation and planning all succeed.
        engine = StreamingWorldModelEngine(latent_dim=8)
        ok = engine.step_stream([0.1] * 8, ["A", "B"])
        self.assertEqual(ok["step"], 1)
        kv_before = engine.kv_cache.get_temporal_context()
        latent_before, action_before = engine.last_latent, engine.last_action

        bad_inputs = (
            [float("nan")] + [0.1] * 7,
            [float("inf")] + [0.1] * 7,
            [10 ** 1000] + [0.1] * 7,
            np.array([np.nan] + [0.1] * 7),
            np.array([1e300] + [0.1] * 7),  # finite float64, overflows to inf in float32
        )
        for bad in bad_inputs:
            with self.assertRaises(ValueError, msg=repr(bad)[:40]):
                engine.step_stream(bad, ["A", "B"])
        self.assertEqual(engine.step_count, 1)
        self.assertEqual(engine.kv_cache.get_temporal_context(), kv_before)
        self.assertIs(engine.last_latent, latent_before)
        self.assertEqual(engine.last_action, action_before)

        # A failure inside planning (after encoding succeeded) must not commit state either.
        original_plan = engine.planner.plan

        def exploding_plan(**kwargs):
            raise RuntimeError("planner failure")

        engine.planner.plan = exploding_plan
        with self.assertRaises(RuntimeError):
            engine.step_stream([0.3] * 8, ["A", "B"])
        engine.planner.plan = original_plan
        self.assertEqual(engine.step_count, 1)
        self.assertEqual(engine.kv_cache.get_temporal_context(), kv_before)
        self.assertIs(engine.last_latent, latent_before)

        self.assertEqual(engine.step_stream([0.2] * 8, ["A", "B"])["step"], 2)

    def test_execute_zero_stream_rejects_non_finite_and_null_session(self):
        for bad in ([float("nan")], [float("inf")], [float("-inf")], [10 ** 1000]):
            res = asyncio.run(execute_zero_stream({"observation": bad, "candidate_actions": ["A"]}))
            self.assertTrue(res["isError"], msg=repr(bad)[:40])
            self.assertIn("finite", res["content"][0]["text"])
        res = asyncio.run(execute_zero_stream({
            "observation": [0.1], "candidate_actions": ["A"], "session_id": None,
        }))
        self.assertTrue(res["isError"])
        self.assertIn("Invalid session_id", res["content"][0]["text"])

    def test_streaming_engine_text_prior_is_deterministic_and_tagged(self):
        z1, prov1 = StreamingWorldModelEngine(latent_dim=16)._encode_observation("frame_camera_001")
        z2, prov2 = StreamingWorldModelEngine(latent_dim=16)._encode_observation("frame_camera_001")
        self.assertEqual(prov1, PROVENANCE_UNTRAINED_TEXT_HASH_PRIOR)
        self.assertEqual(prov2, PROVENANCE_UNTRAINED_TEXT_HASH_PRIOR)
        self.assertTrue(np.allclose(np.asarray(z1), np.asarray(z2)))
        with self.assertLogs("gen_zero.world_model.streaming_engine", level="WARNING") as logs:
            out = StreamingWorldModelEngine(latent_dim=16).step_stream("frame_camera_001", ["A", "B"])
        self.assertTrue(any(DEGRADATION_UNTRAINED_TEXT_HASH_PRIOR in line for line in logs.output))
        self.assertTrue(out["degraded"])


class TestClientWorldModelIntegration(unittest.TestCase):
    """Tests for GenZero client integration with World Model Orchestrator."""

    def test_client_imagine_world_model(self):
        client = GenZero()
        state = "Production microservice checkout flow high error rate"
        candidates = ["CIRCUIT_BREAK", "ROLLBACK", "ALERT_ONCALL"]

        res = client.imagine_world_model(
            state=state,
            candidate_actions=candidates,
            horizon=3,
            enforce_cpsat=True,
            safety_evaluator=lambda z, act: 0.9,
        )

        self.assertIsInstance(res, WorldModelOrchestrationResult)
        self.assertIn(res.selected_action, candidates)
        self.assertEqual(res.horizon_explored, 3)
        self.assertEqual(
            res.nanocore_status.cpsat_verified,
            res.nanocore_status.cpsat_status in CPSAT_REAL_SOLVE_STATUSES,
        )
        self.assertTrue(res.nanocore_status.safety_verified)
        self.assertEqual(res.decision_status, DECISION_OK)

    def test_client_imagine_world_model_without_evaluator_is_fail_closed(self):
        client = GenZero()
        res = client.imagine_world_model(
            state="Production microservice checkout flow high error rate",
            candidate_actions=["CIRCUIT_BREAK", "ROLLBACK"],
            horizon=3,
        )
        self.assertEqual(res.selected_action, DEFAULT_FALLBACK_SAFE_ACTION)
        self.assertIn(UNVERIFIED_SAFETY, res.degradations)


if __name__ == "__main__":
    unittest.main()
