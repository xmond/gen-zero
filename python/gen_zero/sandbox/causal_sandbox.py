"""Gen-Zero Layer 2 Sandbox: Causal Tool Sandbox with Exogenous Disturbance Disentanglement.

Implements Judea Pearl's SCM for Tool Execution:
    Observation := f_Tool(Tool_Name, Params) + U_Env
where:
    U_Env: Exogenous network drop, API 503/504, rate limit 429, upstream DB deadlock.
    f_Tool: Deterministic tool logic given valid parameters.

Features:
1. SCM classification: Accurately separates external network faults (U_t) from agent logic errors (A_t).
2. Automated exponential backoff for exogenous disturbances.
3. Transactional rollback stack for MUTATING_REVERSIBLE side effects.
4. Preserves clean policy gradients during self-evolution.
"""

import time
import copy
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable

from .tool_registry import ToolRegistry, ToolDefinition, SideEffectLevel


class ErrorCategory(str, Enum):
    """Causal classification of tool execution errors."""
    EXOGENOUS_SHOCK = "exogenous_shock"           # HTTP 5xx, Timeout, 429, Network drop (U_t != 0)
    ACTION_DECISION_ERROR = "action_decision_error" # Schema error, wrong param, 400 Bad Request (A_t)
    SAFETY_VIOLATION = "safety_violation"         # Intercepted by PRM / CP-SAT hard firewall


@dataclass
class ExecutionResult:
    """Execution receipt returned by CausalToolSandbox."""
    tool_name: str
    params: Dict[str, Any]
    success: bool
    output: Any = None
    error_category: Optional[ErrorCategory] = None
    error_message: Optional[str] = None
    exogenous_noise_detected: bool = False
    retries_attempted: int = 0
    rollback_performed: bool = False
    side_effect_state: str = "CONFIRMED_NONE"  # "CONFIRMED_NONE", "MUTATING_COMMITTED", "UNKNOWN_MUTATION_STATE"
    latency_ms: float = 0.0


class CausalToolSandbox:
    """Isolated, transaction-aware tool execution sandbox with causal error attribution."""

    # Common patterns indicating exogenous environment noise
    EXOGENOUS_KEYWORDS = (
        "500", "502", "503", "504", "429", "timeout", "timed out",
        "connection reset", "connection refused", "econnrefused",
        "network unreachable", "temporary failure", "deadlock", "lock wait"
    )

    def __init__(self, registry: Optional[ToolRegistry] = None):
        self.registry = registry or ToolRegistry()
        self._transaction_stack: List[Tuple[ToolDefinition, Dict[str, Any], Any]] = []
        self._uncompensated_failed_transactions: List[Tuple[ToolDefinition, Dict[str, Any], Any, str]] = []

    def classify_error(self, err: Exception) -> ErrorCategory:
        """Classifies an execution exception into Exogenous Shock vs Action Decision Error."""
        msg = str(err).lower()
        err_type = type(err).__name__.lower()

        # Check for socket/network/timeout or 5xx/429 status codes
        for kw in self.EXOGENOUS_KEYWORDS:
            if kw in msg or kw in err_type:
                return ErrorCategory.EXOGENOUS_SHOCK

        if isinstance(err, (TimeoutError, ConnectionError)):
            return ErrorCategory.EXOGENOUS_SHOCK

        # Check for schema/param/type errors indicating agent decision bugs
        if isinstance(err, (ValueError, TypeError, KeyError, AssertionError)):
            return ErrorCategory.ACTION_DECISION_ERROR

        return ErrorCategory.ACTION_DECISION_ERROR

    def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        auto_retry_exogenous: bool = True
    ) -> ExecutionResult:
        """Executes a tool call inside the causal sandbox with SCM noise handling."""
        t0 = time.perf_counter()
        tool = self.registry.get(tool_name)

        if not tool:
            lat = (time.perf_counter() - t0) * 1000.0
            return ExecutionResult(
                tool_name=tool_name,
                params=params,
                success=False,
                error_category=ErrorCategory.ACTION_DECISION_ERROR,
                error_message=f"Tool '{tool_name}' not found in registry.",
                latency_ms=round(lat, 2)
            )

        # 1. Pre-execution schema validation
        is_valid, validation_err = self.registry.validate_call(tool_name, params)
        if not is_valid:
            lat = (time.perf_counter() - t0) * 1000.0
            return ExecutionResult(
                tool_name=tool_name,
                params=params,
                success=False,
                error_category=ErrorCategory.ACTION_DECISION_ERROR,
                error_message=validation_err,
                latency_ms=round(lat, 2)
            )

        # 2. Execution loop with SCM exogenous retry backoff (Strictly respecting tool.is_idempotent)
        retries = 0
        can_retry = auto_retry_exogenous and tool.is_idempotent
        max_retries = tool.max_retries_on_exogenous if can_retry else 0
        backoff_sec = 0.05

        while True:
            try:
                output = tool.func(**params)
                lat = (time.perf_counter() - t0) * 1000.0

                # If mutating reversible, record to transaction stack for potential rollback
                if tool.side_effect_level == SideEffectLevel.MUTATING_REVERSIBLE:
                    self._transaction_stack.append((tool, params, output))

                return ExecutionResult(
                    tool_name=tool_name,
                    params=params,
                    success=True,
                    output=output,
                    exogenous_noise_detected=(retries > 0),
                    retries_attempted=retries,
                    side_effect_state="MUTATING_COMMITTED" if tool.side_effect_level != SideEffectLevel.READ_ONLY else "CONFIRMED_NONE",
                    latency_ms=round(lat, 2)
                )

            except Exception as e:
                err_cat = self.classify_error(e)

                # Safe retry: ONLY retry exogenous errors IF the tool is explicitly marked idempotent!
                if err_cat == ErrorCategory.EXOGENOUS_SHOCK and retries < max_retries:
                    retries += 1
                    time.sleep(backoff_sec)
                    backoff_sec *= 2.0
                    continue

                lat = (time.perf_counter() - t0) * 1000.0
                is_mutating = (tool.side_effect_level != SideEffectLevel.READ_ONLY)
                if is_mutating:
                    # Partial side effects are unknown: record uncompensated failure
                    self._uncompensated_failed_transactions.append((tool, params, None, f"EXCEPTION_DURING_MUTATION: {e}"))
                return ExecutionResult(
                    tool_name=tool_name,
                    params=params,
                    success=False,
                    error_category=err_cat,
                    error_message=str(e),
                    exogenous_noise_detected=(err_cat == ErrorCategory.EXOGENOUS_SHOCK),
                    retries_attempted=retries,
                    side_effect_state="UNKNOWN_MUTATION_STATE" if is_mutating else "CONFIRMED_NONE",
                    latency_ms=round(lat, 2)
                )

    def rollback_transactions(self) -> int:
        """Rolls back all executed MUTATING_REVERSIBLE actions in LIFO order.
        
        Returns the number of verified successful rollbacks.
        Failed rollbacks are preserved in _uncompensated_failed_transactions for auditable recovery.
        """
        rolled_back_count = 0
        while self._transaction_stack:
            tool, params, output = self._transaction_stack.pop()
            if tool.rollback_func:
                try:
                    import inspect
                    call_kwargs = params
                    try:
                        sig = inspect.signature(tool.rollback_func)
                        try:
                            sig.bind(params=params, original_output=output)
                            call_kwargs = {"params": params, "original_output": output}
                        except TypeError:
                            sig.bind(**params)
                            call_kwargs = params
                    except Exception:
                        call_kwargs = params

                    # Execute EXACTLY ONCE: internal exceptions are captured as failures and NEVER retried
                    tool.rollback_func(**call_kwargs)
                    rolled_back_count += 1
                except Exception as e:
                    # Retain failed compensation record for auditing and manual/retry recovery
                    self._uncompensated_failed_transactions.append((tool, params, output, str(e)))
            else:
                # No rollback function registered: retain in uncompensated transactions list
                self._uncompensated_failed_transactions.append((tool, params, output, "NO_ROLLBACK_FUNCTION_REGISTERED"))
        return rolled_back_count


    def clear_transactions(self):
        """Clears the transaction stack upon workflow successful completion."""
        self._transaction_stack.clear()

    @property
    def pending_transactions_count(self) -> int:
        return len(self._transaction_stack)

    @property
    def uncompensated_transactions_count(self) -> int:
        return len(self._uncompensated_failed_transactions)

    @property
    def uncompensated_records(self) -> List[Tuple[ToolDefinition, Dict[str, Any], Any, str]]:
        return list(self._uncompensated_failed_transactions)

    @property
    def uncompensated_failed_transactions(self) -> List[Tuple[ToolDefinition, Dict[str, Any], Any, str]]:
        return self._uncompensated_failed_transactions

