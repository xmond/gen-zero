"""Gen-Zero Autonomous Causal Tool Sandbox & Execution Environment.

Provides:
1. Strongly typed ToolRegistry with side-effect levels (READ_ONLY, MUTATING_REVERSIBLE, MUTATING_IRREVERSIBLE).
2. SCM-driven CausalSandbox: Disentangles exogenous network/server shocks (U_t) from agent logic errors (A_t).
3. PRM Formal Safety Barrier: CP-SAT hard invariant gating with zero catastrophic mutations.
4. MCTS Self-Healing Toolchain Planner: Metacognitive reflection & smart alternative tool fallback.
"""

from .tool_registry import ToolRegistry, ToolDefinition, SideEffectLevel
from .causal_sandbox import CausalToolSandbox, ExecutionResult, ErrorCategory
from .safety_barrier import PRMSafetyBarrier, SafetyVerdict
from .self_healing_planner import SelfHealingToolchainPlanner

__all__ = [
    "ToolRegistry",
    "ToolDefinition",
    "SideEffectLevel",
    "CausalToolSandbox",
    "ExecutionResult",
    "ErrorCategory",
    "PRMSafetyBarrier",
    "SafetyVerdict",
    "SelfHealingToolchainPlanner"
]
