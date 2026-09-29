"""Unit and Integration Tests for Issue #33: MCP Gateway Adaptive Three-Tuple Router.

Tests:
1. Session-Locked Main Model (Session Start evaluation, multi-turn lock, escape hatch).
2. Per-Task Elastic Subagent Tiering (flash_lite, flash, pro).
3. Thinking Budget Pre-Gating (none, low, medium, high with budget mapping).
4. MCP Tool Schema Dynamic Pruning Gate (compressing 35+ tools down to Top 4).
5. Gateway Middleware End-to-End Latency SLA (< 12.0ms).
6. Client-level SDK Integration (client.route_mcp).
"""

import unittest
import time

from gen_zero.router.adaptive_router import (
    ModelTier,
    ThinkingEffort,
    THINKING_BUDGET_MAP,
    AdaptiveRouteDecision,
    SessionLockedRouter,
    SubagentTierRouter,
    ThinkingEffortGate,
    ToolSchemaPruningGate,
    ZeroRouterMiddleware,
)
from gen_zero.client import GenZeroClient


class TestSessionLockedRouter(unittest.TestCase):
    """Test 1: Session-Locked Main Model for Prompt Prefix Caching."""

    def setUp(self):
        self.router = SessionLockedRouter()

    def test_initial_session_assessment(self):
        # Complex architecture task -> PRO
        p_pro = "Design distributed consensus architecture with Byzantine fault tolerance."
        self.assertEqual(self.router.assess_initial_model(p_pro), ModelTier.PRO)

        # Simple file lookup -> FLASH
        p_flash = "view file test.py"
        self.assertEqual(self.router.assess_initial_model(p_flash), ModelTier.FLASH)

        # Standard programming -> STANDARD
        p_std = "Add error logging to the user authentication controller."
        self.assertEqual(self.router.assess_initial_model(p_std), ModelTier.STANDARD)

    def test_multi_turn_session_locking(self):
        session_id = "sess_cache_101"
        prompt_turn1 = "Add pagination to user query endpoint."

        # Turn 1: Evaluated and assigned STANDARD
        model_t1, is_locked_t1, turn1 = self.router.resolve_main_model(session_id, prompt_turn1)
        self.assertEqual(model_t1, ModelTier.STANDARD)
        self.assertFalse(is_locked_t1)
        self.assertEqual(turn1, 1)

        # Turn 2: Even if user sends a short simple message, model MUST remain LOCKED to preserve prefix cache!
        model_t2, is_locked_t2, turn2 = self.router.resolve_main_model(session_id, "ok, do it")
        self.assertEqual(model_t2, ModelTier.STANDARD)
        self.assertTrue(is_locked_t2)
        self.assertEqual(turn2, 2)

        # Turn 3: Even if user asks something complex, model remains LOCKED
        model_t3, is_locked_t3, turn3 = self.router.resolve_main_model(session_id, "now optimize the query")
        self.assertEqual(model_t3, ModelTier.STANDARD)
        self.assertTrue(is_locked_t3)
        self.assertEqual(turn3, 3)

    def test_explicit_user_escape_hatch(self):
        session_id = "sess_escape_202"
        # Turn 1: locked to FLASH
        m1, _, _ = self.router.resolve_main_model(session_id, "view file readme.md")
        self.assertEqual(m1, ModelTier.FLASH)

        # Turn 2: Explicit command "switch to pro" MUST unlock and escalate
        m2, is_locked2, _ = self.router.resolve_main_model(session_id, "Please switch to pro model now.")
        self.assertEqual(m2, ModelTier.PRO)
        self.assertFalse(is_locked2)

    def test_consecutive_error_escalation(self):
        session_id = "sess_err_303"
        # Turn 1: standard
        self.router.resolve_main_model(session_id, "Fix login bug")

        # 2 errors -> still locked
        self.router.resolve_main_model(session_id, "Try again", task_error=True)
        self.router.resolve_main_model(session_id, "Try again 2", task_error=True)
        m, is_locked, _ = self.router.resolve_main_model(session_id, "Try again 3", task_error=True)

        # 3rd consecutive error triggers auto-escalation to PRO
        self.assertEqual(m, ModelTier.PRO)


class TestSubagentTierRouter(unittest.TestCase):
    """Test 2: Per-Task Elastic Subagent Tiering."""

    def setUp(self):
        self.router = SubagentTierRouter()

    def test_subagent_tiering(self):
        # 1. Flash-Lite: read-only search & lookup
        task_search = "grep for function def handle_request in repo"
        self.assertEqual(self.router.evaluate_subagent_tier(task_search), ModelTier.FLASH_LITE)

        # 2. Flash: single function or unit test
        task_code = "write unit test for string formatter"
        self.assertEqual(self.router.evaluate_subagent_tier(task_code), ModelTier.FLASH)

        # 3. Pro: race conditions, memory leaks, formal verification
        task_deep = "debug race condition in lockless queue under memory leak pressure"
        self.assertEqual(self.router.evaluate_subagent_tier(task_deep), ModelTier.PRO)


class TestThinkingEffortGate(unittest.TestCase):
    """Test 3: Thinking Effort & Budget Pre-Gating."""

    def setUp(self):
        self.gate = ThinkingEffortGate()

    def test_effort_and_budget_mapping(self):
        # NONE (0 budget)
        effort_none, budget_none = self.gate.evaluate_effort("run git status")
        self.assertEqual(effort_none, ThinkingEffort.NONE)
        self.assertEqual(budget_none, 0)

        # LOW (1024 budget)
        effort_low, budget_low = self.gate.evaluate_effort("check if port is open")
        self.assertEqual(effort_low, ThinkingEffort.LOW)
        self.assertEqual(budget_low, 1024)

        # MEDIUM (4096 budget)
        effort_med, budget_med = self.gate.evaluate_effort("implement a LRU cache class with thread safety")
        self.assertEqual(effort_med, ThinkingEffort.MEDIUM)
        self.assertEqual(budget_med, 4096)

        # HIGH (16384 budget)
        effort_high, budget_high = self.gate.evaluate_effort("why does this fail with deadlock, find the root cause")
        self.assertEqual(effort_high, ThinkingEffort.HIGH)
        self.assertEqual(budget_high, 16384)


class TestToolSchemaPruningGate(unittest.TestCase):
    """Test 4: MCP Tool Schema Dynamic Pruning Gate."""

    def setUp(self):
        self.pruner = ToolSchemaPruningGate()
        # Synthetic large catalog with 35 tools
        self.large_catalog = [
            {"name": f"unrelated_tool_{i}", "description": f"Perform database operations on table {i}"}
            for i in range(30)
        ]
        # Add target relevant tools
        self.large_catalog.extend([
            {"name": "grep_search", "description": "Search regex pattern across repository files"},
            {"name": "view_file", "description": "Read file lines and view content"},
            {"name": "list_dir", "description": "List files in directory"},
            {"name": "find_by_name", "description": "Locate files matching glob pattern"},
            {"name": "git_log", "description": "View commit history"},
        ])

    def test_pruning_compresses_catalog(self):
        self.assertEqual(len(self.large_catalog), 35)

        pruned = self.pruner.prune_tools(
            task_goal="grep search for exception handler in python files",
            tools=self.large_catalog,
            top_k=4,
        )

        self.assertEqual(len(pruned), 4)
        pruned_names = [t["name"] for t in pruned]
        self.assertIn("grep_search", pruned_names)
        # Verify significant compression ratio (> 80%)
        compression_ratio = 1.0 - (len(pruned) / len(self.large_catalog))
        self.assertGreater(compression_ratio, 0.80)


class TestZeroRouterMiddleware(unittest.TestCase):
    """Test 5: Middleware Integration and Latency SLA (< 12.0ms)."""

    def setUp(self):
        self.middleware = ZeroRouterMiddleware()
        self.sample_tools = [
            {"name": "run_command", "description": "Execute bash command"},
            {"name": "view_file", "description": "Read file"},
            {"name": "edit_file", "description": "Modify file contents"},
            {"name": "git_status", "description": "Check git repository status"},
        ]

    def test_middleware_route_decision(self):
        decision: AdaptiveRouteDecision = self.middleware.route(
            session_id="mid_session_001",
            prompt="run git status to check modified files",
            tools=self.sample_tools,
            top_k_tools=2,
        )

        self.assertEqual(decision.session_id, "mid_session_001")
        self.assertIn(decision.locked_main_model, [ModelTier.FLASH, ModelTier.STANDARD, ModelTier.PRO])
        self.assertEqual(decision.thinking_effort, ThinkingEffort.NONE)
        self.assertEqual(decision.thinking_budget, 0)
        self.assertEqual(len(decision.pruned_tool_schemas), 2)
        self.assertIn("git_status", decision.pruned_tool_names)
        self.assertLess(decision.latency_ms, 12.0)  # Sub-12ms SLA

    def test_latency_microbenchmark_sub_12ms(self):
        trials = 50
        latencies = []
        for i in range(trials):
            t0 = time.perf_counter()
            self.middleware.route(
                session_id=f"bench_sess_{i % 5}",
                prompt="debug race condition in multi-threaded queue and refactor",
                tools=self.sample_tools,
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)

        p95 = sorted(latencies)[int(0.95 * len(latencies))]
        self.assertLess(p95, 12.0)


class TestClientIntegration(unittest.TestCase):
    """Test 6: Client-level SDK Integration."""

    def setUp(self):
        self.client = GenZeroClient()

    def test_client_route_mcp(self):
        res = self.client.route_mcp(
            session_id="client_sess_1",
            prompt="Implement binary search algorithm in Python",
            tools=[{"name": "edit_file", "description": "Edit code"}],
        )

        self.assertIn("locked_main_model", res)
        self.assertIn("thinking_effort", res)
        self.assertIn("thinking_budget", res)
        self.assertIn("pruned_tool_names", res)
        self.assertLess(res["latency_ms"], 12.0)


if __name__ == "__main__":
    unittest.main()
