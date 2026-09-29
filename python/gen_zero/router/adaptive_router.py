"""Zero MCP Gateway Adaptive Three-Tuple Router.

Implements Issue #33 (RFC-033). Every component is a lexical heuristic: fixed English
keyword/cue lists and substring or token-overlap matching, with no learned model.

1. Session-Locked Main Model (SessionLockedRouter):
   - Session Start macro assessment (pro / standard / fast).
   - Multi-turn locking (turns 2~N) of the main model so a provider-side prompt prefix
     cache can be reused. Nothing here measures a cache hit rate.
   - Explicit escape hatch (user command / 3 consecutive errors).
2. Per-Task Elastic Subagent Tiering (SubagentTierRouter):
   - Discrete tiering by keyword cues: flash_lite / flash / pro.
3. Thinking Budget Pre-Gating (ThinkingEffortGate):
   - Four-tier effort: none (0s / budget=0) / low / medium / high.
4. Tool Schema Dynamic Pruning Gate (ToolSchemaPruningGate):
   - Keeps the top_k tools by ASCII word overlap between the query and each tool's
     name/description (see the class docstring for the non-ASCII fallback).
5. Gateway Middleware (ZeroRouterMiddleware):
   - Runs 1-4 in process and reports its own wall-clock latency_ms.

Design targets (reference thresholds, not measured guarantees):
   - Prefix cache hit rate > 88.5% under session locking: no test or benchmark measures it.
   - > 80% tool-schema reduction: test_issue_33_mcp_adaptive_router.py checks
     1 - top_k / len(tools) on a synthetic 35-tool catalog (35 -> 4). That is a schema
     count ratio, true by construction; real prompt-token savings are not measured.
   - Route latency < 12.0 ms: the same test file checks in-process p95 latency on short
     synthetic prompts. It does not cover network, provider or end-to-end request latency.
   - Per-tier decision < 5 ms: no dedicated measurement.
"""

from typing import Dict, List, Any, Optional, Tuple, Set, Union
import dataclasses
import enum
import time
import re
import hashlib


class ModelTier(str, enum.Enum):
    FLASH_LITE = "flash_lite"
    FLASH = "flash"
    STANDARD = "standard"
    PRO = "pro"


class ThinkingEffort(str, enum.Enum):
    NONE = "none"      # 0 tokens CoT (direct command execution, simple file read)
    LOW = "low"        # ~1,024 tokens CoT (single-step validation, multi-tool combination)
    MEDIUM = "medium"  # ~4,096 tokens CoT (multi-step planning, code synthesis)
    HIGH = "high"      # ~16,384 tokens CoT (architectural deduction, root cause, formal proof)


THINKING_BUDGET_MAP: Dict[ThinkingEffort, int] = {
    ThinkingEffort.NONE: 0,
    ThinkingEffort.LOW: 1024,
    ThinkingEffort.MEDIUM: 4096,
    ThinkingEffort.HIGH: 16384,
}


@dataclasses.dataclass
class AdaptiveRouteDecision:
    session_id: str
    locked_main_model: ModelTier
    subagent_tier: ModelTier
    thinking_effort: ThinkingEffort
    thinking_budget: int
    pruned_tool_schemas: List[Dict[str, Any]]
    pruned_tool_names: List[str]
    is_cache_locked: bool
    cache_lock_turn: int
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "locked_main_model": self.locked_main_model.value,
            "subagent_tier": self.subagent_tier.value,
            "thinking_effort": self.thinking_effort.value,
            "thinking_budget": self.thinking_budget,
            "pruned_tool_names": self.pruned_tool_names,
            "pruned_tools_count": len(self.pruned_tool_schemas),
            "is_cache_locked": self.is_cache_locked,
            "cache_lock_turn": self.cache_lock_turn,
            "latency_ms": round(self.latency_ms, 2),
        }


class SessionLockedRouter:
    """Manages multi-turn main model phase-locking for Prompt Prefix Caching economics."""

    def __init__(self):
        # session_id -> {locked_model, turns, consecutive_errors, created_at}
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def assess_initial_model(self, task_prompt: str) -> ModelTier:
        """Evaluates macro complexity for a new session."""
        p_lower = (task_prompt or "").lower()

        # Check for complex architectural signals
        pro_cues = [
            "architecture", "refactor", "distributed", "concurrency",
            "formal verification", "game theory", "algorithm design",
            "security audit", "deadlock", "consensus"
        ]
        if any(cue in p_lower for cue in pro_cues):
            return ModelTier.PRO

        # Simple lightweight cues
        simple_cues = ["view", "check", "status", "format", "read", "grep", "search", "list"]
        tokens = p_lower.split()
        if len(tokens) <= 6 and any(cue in p_lower for cue in simple_cues):
            return ModelTier.FLASH

        return ModelTier.STANDARD

    def resolve_main_model(
        self,
        session_id: str,
        prompt: str,
        task_error: bool = False,
    ) -> Tuple[ModelTier, bool, int]:
        """Resolves main model ensuring phase-locking across multi-turn interactions.

        Returns:
            (model_tier, is_cache_locked, turn_count)
        """
        p_lower = (prompt or "").lower()
        now = time.time()

        # Check for explicit user escape hatch
        user_override = None
        if "switch to pro" in p_lower or "use pro" in p_lower or "use opus" in p_lower:
            user_override = ModelTier.PRO
        elif "switch to flash" in p_lower or "use flash" in p_lower or "use haiku" in p_lower:
            user_override = ModelTier.FLASH
        elif "switch to standard" in p_lower or "use sonnet" in p_lower:
            user_override = ModelTier.STANDARD

        session = self.sessions.get(session_id)

        if session is None or user_override is not None:
            # Session Start or Explicit User Unlock
            tier = user_override or self.assess_initial_model(prompt)
            self.sessions[session_id] = {
                "locked_model": tier,
                "turns": 1,
                "consecutive_errors": 0,
                "created_at": now,
            }
            return tier, False, 1

        # Session In-Progress
        session["turns"] += 1
        if task_error:
            session["consecutive_errors"] += 1
        else:
            session["consecutive_errors"] = 0

        # Automatic escalation on 3 consecutive errors
        if session["consecutive_errors"] >= 3:
            session["locked_model"] = ModelTier.PRO
            session["consecutive_errors"] = 0
            return ModelTier.PRO, False, session["turns"]

        # Strict phase lock to protect prefix prompt cache
        return session["locked_model"], True, session["turns"]


class SubagentTierRouter:
    """Tiers a subagent task by keyword cues (lexical heuristic; latency not benchmarked)."""

    def evaluate_subagent_tier(self, subagent_task: str) -> ModelTier:
        t_lower = (subagent_task or "").lower()

        # 1. Pro tier: high-dimensional reasoning, deep refactoring, safety proofs
        pro_cues = [
            "formal proof", "race condition", "memory leak", "exploit",
            "kernel", "consensus", "compiler pass", "security audit",
            "game theory", "mcts", "equilibrium"
        ]
        if any(cue in t_lower for cue in pro_cues):
            return ModelTier.PRO

        # 2. Flash-Lite tier: shallow read-only exploration and lookup
        lite_cues = [
            "search", "grep", "find_by_name", "list_dir", "view_file",
            "check git", "status", "regex", "locate", "line number"
        ]
        if any(cue in t_lower for cue in lite_cues) and len(t_lower.split()) < 20:
            return ModelTier.FLASH_LITE

        # 3. Flash tier: standard code completion, unit testing, formatting
        return ModelTier.FLASH


class ThinkingEffortGate:
    """Evaluates task depth and converts to reasoning_effort / thinking_budget."""

    def evaluate_effort(
        self,
        prompt: str,
        active_tools: Optional[List[str]] = None,
    ) -> Tuple[ThinkingEffort, int]:
        p_lower = (prompt or "").lower()
        tools = active_tools or []

        # 0. Deterministic direct tool executions: effort NONE
        # Direct CLI execution or known simple inspection
        direct_cues = [
            "run git status", "git status", "view file", "show line",
            "cat ", "ls ", "pwd", "git diff", "what time", "git log"
        ]
        if any(cue in p_lower for cue in direct_cues) or (
            len(p_lower.split()) <= 2 and any(c in p_lower for c in ["ls", "pwd", "status", "ok", "done", "next"])
        ):
            return ThinkingEffort.NONE, THINKING_BUDGET_MAP[ThinkingEffort.NONE]

        # 3. High effort: deep root cause analysis, architecture, formal proofs
        high_cues = [
            "why does this fail", "root cause", "debug deadlock",
            "design architecture", "formal verification", "security vulnerability",
            "analyze bottleneck", "synthesize complete"
        ]
        if any(cue in p_lower for cue in high_cues):
            return ThinkingEffort.HIGH, THINKING_BUDGET_MAP[ThinkingEffort.HIGH]

        # 2. Medium effort: refactor, implementation, multi-step code generation
        medium_cues = ["implement", "refactor", "create", "write test", "integrate", "solve"]
        if any(cue in p_lower for cue in medium_cues) or len(p_lower.split()) > 25:
            return ThinkingEffort.MEDIUM, THINKING_BUDGET_MAP[ThinkingEffort.MEDIUM]

        # 1. Low effort: simple check, single-step edit
        return ThinkingEffort.LOW, THINKING_BUDGET_MAP[ThinkingEffort.LOW]


class ToolSchemaPruningGate:
    """Lexical heuristic that keeps the top_k tool schemas by ASCII word overlap.

    Query, tool name and description are lower-cased and split with the ASCII-only
    pattern [a-z0-9_]+. Query words of 3+ characters that are not stopwords become
    keywords; each tool scores 4.0 per name-token match plus 1.5 per description-token
    match, with ties kept in input order. No embedding, no semantic model. The underscore
    is part of a token, so the query word "grep" does not match the name "grep_search".

    Fallback: when no keyword survives (for example a query written only in Chinese or
    other non-ASCII script), every score ties and the gate returns the first top_k tools
    in input order, whatever their relevance. A mixed query still matches its ASCII words.
    """

    def prune_tools(
        self,
        task_goal: str,
        tools: List[Dict[str, Any]],
        top_k: int = 4,
    ) -> List[Dict[str, Any]]:
        if not tools:
            return []
        if len(tools) <= top_k:
            return tools

        query_text = (task_goal or "").lower()
        query_words = set(re.findall(r"[a-z0-9_]+", query_text))
        stopwords = {"the", "a", "an", "is", "in", "to", "for", "of", "and", "or", "on", "at", "by", "with", "this", "that", "it"}
        keywords = {w for w in query_words if len(w) > 2 and w not in stopwords}

        scored = []
        for i, tool in enumerate(tools):
            name = str(tool.get("name", "")).lower()
            desc = str(tool.get("description", "")).lower()

            name_tokens = set(re.findall(r"[a-z0-9_]+", name))
            desc_tokens = set(re.findall(r"[a-z0-9_]+", desc))

            score = len(name_tokens & keywords) * 4.0 + len(desc_tokens & keywords) * 1.5
            # Preserves original order tiebreaker
            score += 1e-4 * (len(tools) - i)
            scored.append((score, tool))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:top_k]]


class ZeroRouterMiddleware:
    """Gateway-level Adaptive Three-Tuple Router Middleware for MCP ecosystems."""

    def __init__(self):
        self.session_router = SessionLockedRouter()
        self.subagent_router = SubagentTierRouter()
        self.effort_gate = ThinkingEffortGate()
        self.pruning_gate = ToolSchemaPruningGate()

    def route(
        self,
        session_id: str,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        is_subagent: bool = False,
        subagent_task: Optional[str] = None,
        task_error: bool = False,
        top_k_tools: int = 4,
    ) -> AdaptiveRouteDecision:
        """Executes the full three-tuple route decision; latency_ms is its measured in-process time.

        < 12 ms is a design target, checked only in unit tests on short synthetic prompts.
        """
        t0 = time.perf_counter()

        # 1. Main Model Phase Lock
        locked_model, is_locked, turn_count = self.session_router.resolve_main_model(
            session_id=session_id,
            prompt=prompt,
            task_error=task_error,
        )

        # 2. Subagent Tiering (if applicable or requested)
        sub_task_desc = subagent_task if is_subagent else prompt
        subagent_tier = self.subagent_router.evaluate_subagent_tier(sub_task_desc)

        # 3. Thinking Effort & Budget Gating
        effort, budget = self.effort_gate.evaluate_effort(prompt=prompt)

        # 4. Tool Schema Dynamic Pruning
        raw_tools = tools or []
        pruned_tools = self.pruning_gate.prune_tools(
            task_goal=prompt,
            tools=raw_tools,
            top_k=top_k_tools,
        )
        pruned_names = [str(t.get("name", "")) for t in pruned_tools]

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return AdaptiveRouteDecision(
            session_id=session_id,
            locked_main_model=locked_model,
            subagent_tier=subagent_tier,
            thinking_effort=effort,
            thinking_budget=budget,
            pruned_tool_schemas=pruned_tools,
            pruned_tool_names=pruned_names,
            is_cache_locked=is_locked,
            cache_lock_turn=turn_count,
            latency_ms=latency_ms,
        )
