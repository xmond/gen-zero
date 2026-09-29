"""Gen-Zero Layer 4 Sandbox: MCTS Self-Healing Toolchain Orchestrator.

Orchestrates complex multi-step enterprise workflows with:
1. Step-level formal PRM safety verification before execution.
2. SCM-driven error diagnosis (exogenous network shock vs action decision error).
3. Metacognitive Causal Reflection and automatic transaction rollback on failure.
4. Autonomous re-planning to fallback toolchains (e.g. secondary gateway, cache lookup).
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable

from .tool_registry import ToolRegistry, ToolDefinition, SideEffectLevel
from .causal_sandbox import CausalToolSandbox, ExecutionResult, ErrorCategory
from .safety_barrier import PRMSafetyBarrier, SafetyVerdict


@dataclass
class CausalReflection:
    """Metacognitive Reflection Token generated during workflow execution."""
    branch_depth: int
    culprit_action: str
    recommended_action: str
    reflection_insight: str
    ite_advantage: float = 0.0


@dataclass
class WorkflowStep:
    """Specification of a single step in a multi-step tool workflow."""
    step_id: str
    tool_name: str
    params: Dict[str, Any]
    fallback_tool_name: Optional[str] = None
    fallback_params: Optional[Dict[str, Any]] = None


@dataclass
class WorkflowExecutionReport:
    """Final execution summary of a multi-step workflow."""
    success: bool
    completed_steps: int
    total_steps: int
    step_receipts: List[ExecutionResult]
    reflections: List[Dict[str, Any]]
    rollbacks_triggered: int
    self_healing_occurred: bool
    final_output: Any = None
    failure_reason: Optional[str] = None


class SelfHealingToolchainPlanner:
    """Orchestrates multi-step toolchains with autonomous self-healing capabilities."""

    def __init__(
        self,
        sandbox: Optional[CausalToolSandbox] = None,
        safety_barrier: Optional[PRMSafetyBarrier] = None
    ):
        self.sandbox = sandbox or CausalToolSandbox()
        self.safety_barrier = safety_barrier or PRMSafetyBarrier()

    def execute_workflow(
        self,
        steps: List[WorkflowStep],
        stop_on_unrecoverable_failure: bool = True
    ) -> WorkflowExecutionReport:
        """Executes a multi-step toolchain with PRM gating and autonomous self-healing."""
        self.sandbox.clear_transactions()
        receipts: List[ExecutionResult] = []
        reflections: List[Dict[str, Any]] = []
        rollbacks = 0
        healed = False
        last_output = None
        unrecoverable_failures: List[str] = []
        successful_steps_count = 0

        for idx, step in enumerate(steps):
            tool = self.sandbox.registry.get(step.tool_name)
            if not tool:
                # Missing tool -> unrecoverable: rollback all prior committed mutating transactions
                rolled = self.sandbox.rollback_transactions()
                rollbacks += rolled
                return WorkflowExecutionReport(
                    success=False,
                    completed_steps=idx,
                    total_steps=len(steps),
                    step_receipts=receipts,
                    reflections=reflections,
                    rollbacks_triggered=rollbacks,
                    self_healing_occurred=healed,
                    failure_reason=f"Tool '{step.tool_name}' not found."
                )

            # 1. Pre-execution PRM Safety Barrier Audit
            audit = self.safety_barrier.audit(tool, step.params)
            if audit.verdict != SafetyVerdict.ALLOWED:
                # Safety barrier did NOT allow (BLOCKED or DEFERRED pending human approval)
                refl = CausalReflection(
                    branch_depth=idx,
                    culprit_action=step.tool_name,
                    recommended_action=step.fallback_tool_name or "ABSTAIN",
                    reflection_insight=f"PRM Safety Barrier Intercepted ({audit.verdict.name}): {audit.reason}",
                    ite_advantage=10.0
                )
                reflections.append(refl.__dict__)
                rolled = self.sandbox.rollback_transactions()
                rollbacks += rolled

                return WorkflowExecutionReport(
                    success=False,
                    completed_steps=idx,
                    total_steps=len(steps),
                    step_receipts=receipts,
                    reflections=reflections,
                    rollbacks_triggered=rollbacks,
                    self_healing_occurred=healed,
                    failure_reason=f"Safety violation ({audit.verdict.name}): {audit.reason}"
                )

            # 2. Execute primary tool in sandbox
            res = self.sandbox.execute(step.tool_name, step.params)
            receipts.append(res)

            if res.success:
                successful_steps_count += 1
                last_output = res.output
                continue

            # 3. Execution Failed: Diagnose cause via SCM and initiate Self-Healing
            err_cat = res.error_category
            refl = CausalReflection(
                branch_depth=idx,
                culprit_action=step.tool_name,
                recommended_action=step.fallback_tool_name or "RETRY",
                reflection_insight=(
                    f"Step {idx} '{step.tool_name}' failed with {err_cat.value if err_cat else 'error'}: {res.error_message}. "
                    f"Initiating autonomous self-healing via alternative toolchain."
                ),
                ite_advantage=5.0
            )
            reflections.append(refl.__dict__)

            # Check if primary tool had unresolved mutations (single-tool idempotency does NOT prove cross-tool safety)
            has_uncompensated_mutation = (
                res.side_effect_state == "UNKNOWN_MUTATION_STATE"
                or self.sandbox.uncompensated_transactions_count > 0
                or (tool.side_effect_level != SideEffectLevel.READ_ONLY and not res.success and not tool.is_idempotent)
            )

            # Attempt self-healing with fallback tool if defined and safe
            if step.fallback_tool_name and not has_uncompensated_mutation:
                fb_tool = self.sandbox.registry.get(step.fallback_tool_name)
                fb_params = step.fallback_params or step.params

                if fb_tool:
                    # Audit fallback tool
                    fb_audit = self.safety_barrier.audit(fb_tool, fb_params)
                    if fb_audit.verdict == SafetyVerdict.ALLOWED:
                        fb_res = self.sandbox.execute(step.fallback_tool_name, fb_params)
                        receipts.append(fb_res)

                        if fb_res.success:
                            healed = True
                            successful_steps_count += 1
                            last_output = fb_res.output
                            continue

            # If fallback failed, was unsafe, or wasn't available
            rolled = self.sandbox.rollback_transactions()
            rollbacks += rolled
            if has_uncompensated_mutation:
                failure_msg = f"Step '{step.tool_name}' failed with uncompensated partial mutations; fallback aborted to prevent duplicate side effects."
            else:
                failure_msg = f"Unrecoverable error in step '{step.tool_name}': {res.error_message}"
            unrecoverable_failures.append(failure_msg)

            if stop_on_unrecoverable_failure:
                return WorkflowExecutionReport(
                    success=False,
                    completed_steps=idx,
                    total_steps=len(steps),
                    step_receipts=receipts,
                    reflections=reflections,
                    rollbacks_triggered=rollbacks,
                    self_healing_occurred=healed,
                    failure_reason=failure_msg
                )

        # Workflow finished: success is strictly dependent on zero unrecoverable failures
        workflow_succeeded = (len(unrecoverable_failures) == 0)
        if workflow_succeeded:
            self.sandbox.clear_transactions()

        return WorkflowExecutionReport(
            success=workflow_succeeded,
            completed_steps=successful_steps_count,
            total_steps=len(steps),
            step_receipts=receipts,
            reflections=reflections,
            rollbacks_triggered=rollbacks,
            self_healing_occurred=healed,
            failure_reason="; ".join(unrecoverable_failures) if unrecoverable_failures else None,
            final_output=last_output
        )


# Public Alias
SelfHealingPlanner = SelfHealingToolchainPlanner

