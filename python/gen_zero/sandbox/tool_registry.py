"""Gen-Zero Layer 1 Sandbox: Strongly Typed Tool Registry & Side-Effect Categorization."""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Set, Tuple


class SideEffectLevel(IntEnum):
    """Side effect severity level of a tool action."""
    READ_ONLY = 0              # Safe queries, status checks, reads (zero environmental side effect)
    MUTATING_REVERSIBLE = 1    # DB update, staging file write (supports rollback callback)
    MUTATING_IRREVERSIBLE = 2  # Wire transfer, prod release, email send, hard delete


@dataclass
class ToolDefinition:
    """Strongly typed tool specification with execution and rollback bindings."""
    name: str
    description: str
    func: Callable[..., Any]
    side_effect_level: SideEffectLevel = SideEffectLevel.READ_ONLY
    required_params: List[str] = field(default_factory=list)
    rollback_func: Optional[Callable[..., Any]] = None
    timeout_seconds: float = 5.0
    max_retries_on_exogenous: int = 3
    is_idempotent: bool = True
    tags: List[str] = field(default_factory=list)


class ToolRegistry:
    """Central repository for enterprise tools with schema and safety contracts."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        func: Callable[..., Any],
        description: str = "",
        side_effect_level: SideEffectLevel = SideEffectLevel.READ_ONLY,
        required_params: Optional[List[str]] = None,
        rollback_func: Optional[Callable[..., Any]] = None,
        timeout_seconds: float = 5.0,
        is_idempotent: bool = True,
        tags: Optional[List[str]] = None
    ) -> ToolDefinition:
        """Registers a tool with explicit safety and execution contracts."""
        tool = ToolDefinition(
            name=name,
            description=description,
            func=func,
            side_effect_level=side_effect_level,
            required_params=required_params or [],
            rollback_func=rollback_func,
            timeout_seconds=timeout_seconds,
            is_idempotent=is_idempotent,
            tags=tags or []
        )
        self._tools[name] = tool
        return tool

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Retrieves tool definition by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        """Returns all registered tool definitions."""
        return list(self._tools.values())

    def validate_call(self, name: str, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validates tool existence and schema compliance prior to execution."""
        tool = self.get(name)
        if not tool:
            return False, f"Tool '{name}' is not registered in ToolRegistry."

        missing = [p for p in tool.required_params if p not in params]
        if missing:
            return False, f"Missing required parameters for tool '{name}': {missing}"

        return True, None
