#!/usr/bin/env python3
"""Gen-Zero Canonical Model Context Protocol (MCP) Decision Server.

Features:
- Sub-5ms startup with zero heavy dependencies (no PyTorch, NumPy, or transformers imports).
- Single polymorphic `zero_ask` tool wrapping `typesafe/zero-1.13` decision microservice.
- Authentication token read from the GENZERO_API_KEY environment variable only.
- Fully compliant with the official Model Context Protocol JSON-RPC 2.0 stdio specification.
"""

import sys
import os
import json
import asyncio
import logging
import time
import re
import math
from typing import Dict, Any, Optional, List, Union, Tuple

logger = logging.getLogger("gen_zero.mcp.server")

# Protocol constants
MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "gen-zero-mcp"
SERVER_VERSION = "1.13.0"
DEFAULT_ENDPOINT = "http://127.0.0.1:8999"

# execute_zero_route (B06): ASCII-regex keyword overlap is a lexical baseline, never semantic
# routing. Every route-mode response carries routing_method so callers cannot mistake one for
# the other; a non-ASCII-only query gets degraded=True instead of a silent input-order fallback.
ROUTING_METHOD_LEXICAL_BASELINE = "lexical_baseline"
DEGRADATION_LEXICAL_NO_ASCII_KEYWORDS = "LEXICAL_NO_ASCII_KEYWORDS"

# execute_zero_stream session lifecycle (A07): MCP tool calls are stateless JSON-RPC requests,
# so without an explicit session_id each call would construct a fresh StreamingWorldModelEngine
# and silently discard its KV-cache / causal-shock history. MCPServer keys live engines by
# session_id and evicts them by idle TTL and by an LRU cap once GENZERO_MAX_STREAM_SESSIONS is
# reached, so long-running SSE deployments (one shared MCPServer, many concurrent clients) can't
# leak memory across abandoned sessions.
DEFAULT_STREAM_SESSION_TTL_SECONDS = float(os.environ.get("GENZERO_STREAM_SESSION_TTL_SECONDS", "1800"))
DEFAULT_MAX_STREAM_SESSIONS = int(os.environ.get("GENZERO_MAX_STREAM_SESSIONS", "256"))


def resolve_api_token() -> str:
    """Resolves the authorization token from GENZERO_API_KEY only.

    No credential file, no baked-in default: an empty token is sent as-is and
    the decision service answers 401, which is the fail-closed outcome.
    """
    return os.environ.get("GENZERO_API_KEY", "").strip()


def resolve_endpoint() -> str:
    """Resolves the microservice HTTP endpoint."""
    return os.environ.get("GENZERO_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")


ZERO_SCHEMA: Dict[str, Any] = {
    "name": "zero",
    "description": (
        "Universal Gen-Zero Polymorphic Decision, Planning & Cognitive Primitive (0-Token Pure-Prefill). "
        "Single-entrypoint non-autoregressive gateway for all cognitive tasks: "
        "1. Decision Micro-Cores ('ask'): Evaluate typed Noul gates, Choice routing, or Score ratings on a state. "
        "2. Tool Pruning ('route'): Prune 100+ tool catalog down to top-K for a task goal in < 5ms "
        "using an ASCII-keyword lexical baseline (routing_method='lexical_baseline'), NOT semantic "
        "matching; non-ASCII-only queries (e.g. Chinese) come back with degraded=true. "
        "3. Counterfactual Imagination ('imagine'): Unroll lookahead tree in latent space with CP-SAT safety verification. "
        "4. Spatiotemporal Stream ('stream'): Roll continuous observation frames with Attention Sinks constant memory. "
        "5. Semantic Grep ('grep'): Search text/files using natural language propositions and boolean logic ((A AND B) OR NOT C). "
        "6. Lossless Context Compactor ('compact'): Prune verbose logs and drop ephemeral probes with 0.0% fact mutation rate. "
        "Supports explicit 'action' ('ask'|'route'|'imagine'|'stream'|'grep'|'compact') or auto-infers mode from provided arguments."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["ask", "route", "imagine", "stream", "grep", "compact"],
                "description": "Optional explicit action mode. If omitted, zero automatically deduces intent from provided arguments."
            },
            # Mode: ask / decision
            "state": {
                "type": "string",
                "description": "State description or observation context (used for 'ask' and 'imagine' modes)."
            },
            "questions": {
                "type": "object",
                "description": "Dictionary of named decision questions (Noul, Choice, Score) for 'ask' mode.",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["noul", "choice", "score"],
                            "description": "Decision primitive type: 'noul' (probability gate 0~1), 'choice' (discrete candidate selection), or 'score' (scaled rating 0~2)."
                        },
                        "instructions": {
                            "type": "string",
                            "description": "Clear natural language prompt or question for the decision engine."
                        },
                        "criteria": {
                            "description": "Evaluation criteria defining outcomes.",
                            "anyOf": [
                                {"type": "object", "additionalProperties": {"type": "string"}},
                                {"type": "array", "items": {"type": "string"}}
                            ]
                        }
                    },
                    "required": ["type", "instructions", "criteria"]
                }
            },
            # Mode: route
            "task_goal": {
                "type": "string",
                "description": "User instruction or task goal for tool pruning router (used for 'route' mode)."
            },
            "tools": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Candidate tools catalog to prune (used for 'route' mode)."
            },
            "top_k": {
                "type": "integer",
                "default": 5,
                "description": "Maximum number of pruned tools to return (used for 'route' mode)."
            },
            "context": {
                "type": "string",
                "description": "Optional execution context snippet for tool routing."
            },
            # Mode: imagine & stream
            "candidate_actions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Candidate action identifiers (used for 'imagine' and 'stream' modes)."
            },
            "horizon": {
                "type": "integer",
                "default": 4,
                "description": "Virtual lookahead tree depth (used for 'imagine' mode)."
            },
            "enforce_cpsat": {
                "type": "boolean",
                "default": True,
                "description": "Whether to enforce 0-1 ILP CP-SAT safety interlock (used for 'imagine' mode)."
            },
            "forbidden_actions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Prohibited actions to hard-block (used for 'imagine' mode)."
            },
            "constraints": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["forbid", "mutually_exclusive"]},
                        "actions": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["type", "actions"],
                    "additionalProperties": False
                },
                "description": (
                    "Action constraints (used for 'imagine' mode ONLY; other modes reject it). "
                    "E.g. [{\"type\": \"mutually_exclusive\", \"actions\": [\"reboot\", \"format\"]}]. "
                    "'forbid' blocks the listed actions; 'mutually_exclusive' keeps only the higher-scoring "
                    "member of the group. Result carries 'formal_verification': "
                    "'unavailable_missing_dependency' when OR-Tools is not installed (no proof was run)."
                )
            },
            "observation": {
                "description": "Incoming frame, latent vector, or observation string (used for 'stream' mode). A string or object is NOT encoded: it becomes an untrained hash prior and the response carries degraded=true, provenance='untrained_text_hash_prior'. A numeric vector whose length does not match the engine's latent_dim is routed through an untrained deterministic linear projector (degraded=true, provenance='projected_untrained_linear_projection') rather than truncated or zero-padded."
            },
            "context_prompt": {
                "type": "string",
                "description": "Optional textual prompt for stream step."
            },
            "session_id": {
                "type": "string",
                "description": (
                    "Optional episode identifier (used for 'stream' mode). Calls that share a "
                    "session_id reuse the same StreamingWorldModelEngine instance server-side, so "
                    "its rolling KV-cache and causal-shock history carry over between calls. "
                    "Without session_id each call gets a fresh, history-less engine (KV-cache "
                    "starts empty and causal_shock_norm is always 0.0)."
                )
            },
            # Mode: grep
            "query": {
                "type": "string",
                "description": "Natural language semantic pattern or boolean query (used for 'grep' mode)."
            },
            "expr": {
                "type": "string",
                "description": "Explicit propositional boolean logic expression (used for 'grep' mode)."
            },
            "or_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of semantic patterns to match with OR logic (used for 'grep' mode)."
            },
            "and_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of semantic patterns to match with AND logic (used for 'grep' mode)."
            },
            "not_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of semantic patterns to exclude with AND NOT logic (used for 'grep' mode)."
            },
            "lines": {
                "type": "array",
                "items": {"type": "string"},
                "description": "In-memory text lines to semantically filter (used for 'grep' mode)."
            },
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "File or directory paths to search (used for 'grep' mode)."
            },
            "recursive": {
                "type": "boolean",
                "default": False,
                "description": "Recursively search directory paths (used for 'grep' mode)."
            },
            "level": {
                "type": "string",
                "enum": ["loose", "balanced", "strict"],
                "default": "balanced",
                "description": "Semantic confidence level: loose (P>=0.5), balanced (P>=0.7), strict (P>=0.85)."
            },
            "threshold": {
                "type": "number",
                "description": "Custom probability threshold in [0.0, 1.0] (used for 'grep' mode)."
            },
            "max_matches": {
                "type": "integer",
                "default": 50,
                "description": "Maximum matching lines to return (used for 'grep' mode)."
            },
            # Mode: compact
            "messages": {
                "type": "array",
                "items": {"type": "object"},
                "description": "List of conversation message objects to losslessly compact (used for 'compact' mode)."
            },
            "text": {
                "type": "string",
                "description": "Raw string, log, or verbose output to losslessly truncate head/tail (used for 'compact' mode)."
            },
            "head_lines": {
                "type": "integer",
                "default": 5,
                "description": "Number of leading lines to retain verbatim (default: 5)."
            },
            "tail_lines": {
                "type": "integer",
                "default": 5,
                "description": "Number of trailing lines to retain verbatim (default: 5)."
            },
            "truncate_line_threshold": {
                "type": "integer",
                "default": 15,
                "description": "Minimum line count to trigger head/tail truncation (default: 15)."
            }
        }
    }
}


def _reject_unsupported_constraints(arguments: Any, mode: str) -> Optional[Dict[str, Any]]:
    """'constraints' is only honoured by 'imagine'. Elsewhere it would be silently dropped
    (e.g. the remote /v1/decisions endpoint has no such field), so refuse instead."""
    if isinstance(arguments, dict) and "constraints" in arguments:
        return {
            "isError": True,
            "content": [{
                "type": "text",
                "text": (
                    f"Validation Error: 'constraints' is not supported in '{mode}' mode and would be "
                    "silently ignored. Use the 'imagine' mode (state + candidate_actions + constraints)."
                )
            }]
        }
    return None


async def execute_zero_ask(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Validates arguments and dispatches the request to the local Gen-Zero decision endpoint."""
    rejected = _reject_unsupported_constraints(arguments, 'ask')
    if rejected is not None:
        return rejected
    if not isinstance(arguments, dict):
        return {
            "isError": True,
            "content": [{"type": "text", "text": "Error: Arguments must be a JSON object with 'state' and 'questions'."}]
        }

    state = arguments.get("state")
    questions = arguments.get("questions")

    if state is None:
        return {
            "isError": True,
            "content": [{"type": "text", "text": "Validation Error: 'state' field is required."}]
        }

    if not isinstance(questions, dict) or not questions:
        return {
            "isError": True,
            "content": [{"type": "text", "text": "Validation Error: 'questions' must be a non-empty dictionary of question specs."}]
        }

    # Validate question structures locally to avoid remote HTTP 422 errors
    for q_name, q_spec in questions.items():
        if not isinstance(q_spec, dict):
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Validation Error in question '{q_name}': must be a dictionary."}]
            }
        q_type = str(q_spec.get("type", "")).lower()
        if q_type not in ("noul", "choice", "score"):
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Validation Error in question '{q_name}': 'type' must be one of 'noul', 'choice', or 'score', got '{q_type}'."}]
            }
        if "criteria" not in q_spec or q_spec["criteria"] is None:
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Validation Error in question '{q_name}': 'criteria' field is required."}]
            }

    endpoint = resolve_endpoint()
    api_token = resolve_api_token()
    url = f"{endpoint}/v1/decisions"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "typesafe/zero-1.13",
        "state": str(state) if not isinstance(state, str) else state,
        "questions": questions
    }

    try:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                status_code = response.status_code
                resp_data = response.json()
        except ImportError:
            # Zero-dependency standard library fallback
            import urllib.request
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            loop = asyncio.get_event_loop()
            def sync_post():
                with urllib.request.urlopen(req, timeout=15.0) as resp:
                    return resp.status, json.loads(resp.read().decode("utf-8"))
            status_code, resp_data = await loop.run_in_executor(None, sync_post)

        if status_code == 200:
            return {
                "isError": False,
                "content": [{"type": "text", "text": json.dumps(resp_data, indent=2, ensure_ascii=False)}]
            }
        else:
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Gen-Zero Server Error (HTTP {status_code}):\n{json.dumps(resp_data, indent=2)}"}]
            }
    except Exception as e:
        return {
            "isError": True,
            "content": [{
                "type": "text",
                "text": (
                    f"Connection Error: Could not reach Gen-Zero decision microservice at {url}.\n"
                    f"Details: {str(e)}\n"
                    f"Please ensure the local microservice is running: `systemctl --user status genzero-server.service`"
                )
            }]
        }


async def execute_zero_route(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Fast sub-millisecond MCP tool pruning router (Issue #15).

    Evaluates 100+ tool candidates against task_goal and context in < 5ms,
    returning top-K relevant tool schemas.

    Scoring method is ``ROUTING_METHOD_LEXICAL_BASELINE``: ASCII-word regex extraction plus
    keyword-set overlap counting. There is no embedding model and no semantic understanding
    here -- it never claims to be semantic routing, and the result payload always carries
    ``routing_method`` so callers cannot mistake it for one. A task_goal/context with no ASCII
    words (e.g. pure Chinese input) yields an empty keyword set: nothing can outscore anything
    else, so ranking degenerates to input order. That is flagged explicitly via ``degraded`` /
    ``degradation_reason`` rather than silently returned as if it were a real ranking.
    """
    rejected = _reject_unsupported_constraints(arguments, 'route')
    if rejected is not None:
        return rejected
    t0 = time.perf_counter()
    if not isinstance(arguments, dict):
        return {
            "isError": True,
            "content": [{"type": "text", "text": "Error: Arguments must be a JSON object with 'task_goal' and 'tools'."}]
        }

    task_goal = arguments.get("task_goal")
    tools = arguments.get("tools")
    top_k = arguments.get("top_k", 5)
    context = arguments.get("context", "")

    if not task_goal or not isinstance(task_goal, str):
        return {
            "isError": True,
            "content": [{"type": "text", "text": "Validation Error: 'task_goal' must be a non-empty string."}]
        }

    if not isinstance(tools, list) or len(tools) == 0:
        return {
            "isError": True,
            "content": [{"type": "text", "text": "Validation Error: 'tools' must be a non-empty list of tool schemas."}]
        }

    try:
        top_k = int(top_k)
        if top_k <= 0:
            top_k = 5
    except (TypeError, ValueError):
        top_k = 5

    query_text = f"{task_goal} {context}".lower()
    query_words = set(re.findall(r'[a-zA-Z0-9_\-\.]+', query_text))
    stopwords = {"the", "a", "an", "is", "in", "to", "for", "of", "and", "or", "on", "at", "by", "with", "this", "that", "it"}
    query_keywords = {w for w in query_words if len(w) > 2 and w not in stopwords}

    # ASCII-word regex extraction found nothing to match on -- this is the lexical baseline's
    # blind spot (e.g. task_goal/context is entirely Chinese or other non-ASCII script). Without
    # keyword overlap, every tool scores 0 and "ranking" degenerates to input order, which looks
    # like a normal low-confidence result unless we say otherwise. Flag it instead of returning
    # it silently.
    non_ascii_degraded = bool(query_text.strip()) and not query_keywords
    if non_ascii_degraded:
        logger.warning(
            "%s: task_goal/context yielded zero ASCII keywords; %s cannot score this input and "
            "ranking fell back to input order, not relevance.",
            DEGRADATION_LEXICAL_NO_ASCII_KEYWORDS, ROUTING_METHOD_LEXICAL_BASELINE,
        )

    scored_tools: List[Tuple[float, Dict[str, Any]]] = []
    for t_idx, tool in enumerate(tools):
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name", "")).lower()
        desc = str(tool.get("description", "")).lower()
        params = tool.get("inputSchema", {}).get("properties", {})
        param_names = [str(k).lower() for k in params.keys()] if isinstance(params, dict) else []

        score = 0.0
        # Direct name match / keyword overlap
        name_tokens = set(re.findall(r'[a-zA-Z0-9]+', name))
        score += len(name_tokens & query_keywords) * 3.0

        # Description keyword overlap
        desc_tokens = set(re.findall(r'[a-zA-Z0-9]+', desc))
        score += len(desc_tokens & query_keywords) * 1.5

        # Parameters overlap
        score += len(set(param_names) & query_keywords) * 1.0

        # Small tiebreaker based on original order
        score += 1e-4 * (len(tools) - t_idx)

        scored_tools.append((score, tool))

    scored_tools.sort(key=lambda x: x[0], reverse=True)
    pruned = [t for _, t in scored_tools[:top_k]]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    result_payload = {
        "task_goal": task_goal,
        "total_input_tools": len(tools),
        "top_k": top_k,
        "pruned_tool_names": [t.get("name") for t in pruned],
        "pruned_tools": pruned,
        "routing_method": ROUTING_METHOD_LEXICAL_BASELINE,
        "degraded": non_ascii_degraded,
        "degradation_reason": DEGRADATION_LEXICAL_NO_ASCII_KEYWORDS if non_ascii_degraded else None,
        "timing_ms": round(elapsed_ms, 2)
    }

    return {
        "isError": False,
        "content": [
            {
                "type": "text",
                "text": json.dumps(result_payload, ensure_ascii=False)
            }
        ]
    }


async def execute_zero_imagine(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """MCP Handler for zero_imagine (Issue #51 & RFC-049)."""
    t0 = time.perf_counter()
    if not isinstance(arguments, dict):
        return {
            "isError": True,
            "content": [{"type": "text", "text": "Error: Arguments must be a JSON object with 'state' and 'candidate_actions'."}]
        }

    state = arguments.get("state")
    candidate_actions = arguments.get("candidate_actions")
    horizon = int(arguments.get("horizon", 4))
    enforce_cpsat = bool(arguments.get("enforce_cpsat", True))
    forbidden_list = arguments.get("forbidden_actions") or []
    forbidden_set = set(forbidden_list)
    constraints = arguments.get("constraints")

    if state is None:
        return {
            "isError": True,
            "content": [{"type": "text", "text": "Validation Error: 'state' field is required."}]
        }

    if not isinstance(candidate_actions, list) or not candidate_actions:
        return {
            "isError": True,
            "content": [{"type": "text", "text": "Validation Error: 'candidate_actions' must be a non-empty list of strings."}]
        }

    from gen_zero.nanocore.world_model_orchestrator import (
        DEFAULT_FALLBACK_SAFE_ACTION,
        SAFETY_INTERLOCKED,
        SafetyInterlockError,
        WorldModelNanoCoreOrchestrator,
    )
    orchestrator = WorldModelNanoCoreOrchestrator()
    try:
        res = orchestrator.imagine_and_orchestrate(
            state=state,
            candidate_actions=candidate_actions,
            horizon=horizon,
            enforce_cpsat=enforce_cpsat,
            forbidden_actions=forbidden_set,
            constraints=constraints,
        )
    except ValueError as exc:
        # Malformed 'constraints' is a caller error: fail closed, never run unconstrained.
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"Validation Error in 'constraints': {exc}"}]
        }
    except SafetyInterlockError as exc:
        # Fail closed: nothing is releasable, not even the no-op fallback.
        return {
            "isError": True,
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "error": SAFETY_INTERLOCKED,
                    "message": str(exc),
                    "degradations": exc.degradations,
                    "cpsat_status": exc.cpsat_status,
                }, indent=2, ensure_ascii=False),
            }],
        }

    payload = res.to_dict()
    if res.interlocked:
        # Every candidate was blocked. The payload still carries the no-op fallback and the
        # full telemetry, but the call is an error: no candidate action was approved.
        return {
            "isError": True,
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "error": SAFETY_INTERLOCKED,
                    "message": (
                        f"{SAFETY_INTERLOCKED}: all {len(candidate_actions)} candidate actions were "
                        f"blocked; no candidate is approved. Fallback no-op is {res.selected_action!r}."
                    ),
                    "result": payload,
                }, indent=2, ensure_ascii=False),
            }],
        }

    return {
        "isError": False,
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, indent=2, ensure_ascii=False)
            }
        ]
    }


def _describe_bad_number(v: Any) -> str:
    # repr() of a huge int can itself raise (sys.set_int_max_str_digits) or flood the message.
    if isinstance(v, int) and not isinstance(v, bool):
        return f"int with {v.bit_length()} bits (overflows float)"
    return repr(v)[:64]


def _validate_zero_stream_arguments(arguments: Any) -> Optional[Dict[str, Any]]:
    """Validates 'stream' mode arguments without creating or touching any engine/session.

    Returns an MCP error response dict if the arguments are malformed, or ``None`` if they are
    well-formed. Factored out of ``execute_zero_stream`` so ``MCPServer.handle_request`` can run
    this check BEFORE calling ``_acquire_stream_engine`` (A07 gate 2): an illegal request must
    never create a new session or trigger an LRU eviction of an unrelated, legitimate session.
    """
    rejected = _reject_unsupported_constraints(arguments, 'stream')
    if rejected is not None:
        return rejected
    if not isinstance(arguments, dict):
        return {
            "isError": True,
            "content": [{"type": "text", "text": "Error: Arguments must be a JSON object with 'observation' and 'candidate_actions'."}]
        }

    # An explicit "session_id": null (or a blank/non-string id) is a caller bug, not an opt-out
    # of continuity: only an ABSENT key means "stateless call". Checked here, before any session
    # is acquired, so a broken caller is never silently served by an anonymous engine.
    if "session_id" in arguments:
        sid = arguments["session_id"]
        if sid is None or not isinstance(sid, str) or not sid.strip():
            return {
                "isError": True,
                "content": [{"type": "text", "text": (
                    f"Validation Error: Invalid session_id: {sid!r}. "
                    "Must be a non-empty string or omitted."
                )}]
            }

    observation = arguments.get("observation")
    candidate_actions = arguments.get("candidate_actions")

    if observation is None:
        return {
            "isError": True,
            "content": [{"type": "text", "text": "Validation Error: 'observation' field is required."}]
        }

    if not isinstance(candidate_actions, list) or not candidate_actions:
        return {
            "isError": True,
            "content": [{"type": "text", "text": "Validation Error: 'candidate_actions' must be a non-empty list of strings."}]
        }

    if not all(isinstance(a, str) for a in candidate_actions):
        return {
            "isError": True,
            "content": [{"type": "text", "text": "Validation Error: 'candidate_actions' must be a non-empty list of strings."}]
        }

    # Over JSON-RPC an observation is a str, a list of numbers, or a JSON object. Only a
    # numeric list is a real feature vector; str/object become an untrained hash prior and
    # the response says so. Anything else is a typed error.
    if isinstance(observation, list):
        if not observation or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in observation):
            return {
                "isError": True,
                "content": [{"type": "text", "text": "Validation Error: a list 'observation' must be a non-empty list of numbers (feature vector)."}]
            }
        # Lazy import: this module must not pull in torch/numpy (via streaming_engine) just to
        # validate a request. Reject an oversized vector here, BEFORE _acquire_stream_engine
        # runs, so a flood of oversized observations under fresh session_ids cannot burn through
        # the session LRU and evict legitimate sessions before the engine ever rejects them.
        from gen_zero.world_model.streaming_engine import MAX_PROJECTOR_INPUT_DIM
        if len(observation) > MAX_PROJECTOR_INPUT_DIM:
            return {
                "isError": True,
                "content": [{"type": "text", "text": (
                    f"Validation Error: 'observation' length {len(observation)} exceeds the "
                    f"maximum supported dimension {MAX_PROJECTOR_INPUT_DIM}."
                )}]
            }
        # Python's json module accepts NaN/Infinity literals and arbitrary-size ints, so a
        # non-finite or float-overflowing value can reach us over JSON-RPC. Reject it here,
        # before any session is touched, instead of letting it poison a session's latent state.
        # The streaming engine operates in float32, so any value exceeding float32 limits
        # (max ~3.4028235e38) will overflow to Inf during tensor/array conversion; enforce
        # float32 finiteness here so no session allocation or LRU eviction happens.
        MAX_FLOAT32 = 3.4028234663852886e38
        for idx, v in enumerate(observation):
            try:
                fv = float(v)
                finite = math.isfinite(fv) and abs(fv) <= MAX_FLOAT32
            except (OverflowError, TypeError, ValueError):
                finite = False
            if not finite:
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": (
                        f"Validation Error: 'observation'[{idx}] must be a finite float32 number; "
                        f"got {_describe_bad_number(v)}."
                    )}]
                }
    elif not isinstance(observation, (str, dict)):
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"Validation Error: 'observation' must be a string, a numeric list, or an object; got {type(observation).__name__}."}]
        }

    return None


async def execute_zero_stream(arguments: Dict[str, Any], engine: Optional[Any] = None) -> Dict[str, Any]:
    """MCP Handler for zero_stream (Issue #51 & RFC-049).

    ``engine`` is the session-bound ``StreamingWorldModelEngine`` resolved by
    ``MCPServer._acquire_stream_engine`` from the request's ``session_id`` (A07). When called
    directly without one (e.g. from tests, or a caller that never passes session_id), a fresh
    ephemeral engine is constructed here exactly as before -- no history carries over between
    calls in that case, which is now an explicit, documented choice rather than the only
    possible behavior.
    """
    validation_error = _validate_zero_stream_arguments(arguments)
    if validation_error is not None:
        return validation_error

    observation = arguments.get("observation")
    candidate_actions = arguments.get("candidate_actions")
    context_prompt = arguments.get("context_prompt")

    if engine is None:
        from gen_zero.world_model.streaming_engine import StreamingWorldModelEngine
        engine = StreamingWorldModelEngine()
    try:
        res = engine.step_stream(
            observation=observation,
            candidate_actions=candidate_actions,
            context_prompt=context_prompt,
        )
    except ValueError as exc:
        # E.g. the dimension-projector cache's max-input-dimension guard. Caller error: fail
        # closed with a clear message, never let a malformed observation crash the process.
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"Validation Error: {exc}"}]
        }
    if "degraded" not in res or "provenance" not in res:
        # The engine contract requires provenance. A missing tag is a bug, not a pass.
        return {
            "isError": True,
            "content": [{"type": "text", "text": "Internal Error: streaming engine returned no provenance/degraded tag."}]
        }

    return {
        "isError": False,
        "content": [
            {
                "type": "text",
                "text": json.dumps(res, indent=2, ensure_ascii=False)
            }
        ]
    }


async def execute_zero_grep(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """MCP Handler for zero_grep (Issue #9 & RFC-009).

    Evaluates semantic propositions or complex boolean logic expressions over
    memory text lines or file contents using calibrated Noul confidence thresholds.
    """
    rejected = _reject_unsupported_constraints(arguments, 'grep')
    if rejected is not None:
        return rejected
    t0 = time.perf_counter()
    if not isinstance(arguments, dict):
        return {
            "isError": True,
            "content": [{"type": "text", "text": "Error: Arguments must be a JSON object."}]
        }

    query = arguments.get("query")
    expr = arguments.get("expr")
    or_patterns = arguments.get("or_patterns") or []
    and_patterns = arguments.get("and_patterns") or []
    not_patterns = arguments.get("not_patterns") or []
    lines = arguments.get("lines")
    paths = arguments.get("paths")
    recursive = bool(arguments.get("recursive", False))
    level = str(arguments.get("level", "balanced")).lower()
    custom_threshold = arguments.get("threshold")
    max_matches = int(arguments.get("max_matches", 50))

    if not query and not expr and not or_patterns and not and_patterns and not not_patterns:
        return {
            "isError": True,
            "content": [{"type": "text", "text": "Validation Error: Must provide 'query', 'expr', or 'or_patterns'/'and_patterns'."}]
        }

    if lines is None and paths is None:
        return {
            "isError": True,
            "content": [{"type": "text", "text": "Validation Error: Must provide either 'lines' (list of strings) or 'paths' (list of file paths)."}]
        }

    threshold = 0.70
    if level == "loose":
        threshold = 0.50
    elif level == "strict":
        threshold = 0.85
    elif level == "balanced":
        threshold = 0.70

    if custom_threshold is not None:
        try:
            threshold = float(custom_threshold)
            threshold = max(0.0, min(1.0, threshold))
        except (ValueError, TypeError):
            pass

    from gen_zero.scripts.gen_grep import build_composite_expression, query_batch_decisions
    try:
        ast, patterns = build_composite_expression(
            positional_pattern=query,
            or_patterns=or_patterns if isinstance(or_patterns, list) else [],
            and_patterns=and_patterns if isinstance(and_patterns, list) else [],
            not_patterns=not_patterns if isinstance(not_patterns, list) else [],
            explicit_expr=expr
        )
    except Exception as e:
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"Expression Syntax Error: {str(e)}"}]
        }

    items_to_scan: List[Tuple[str, int, str]] = []

    if isinstance(lines, list):
        for idx, line in enumerate(lines):
            items_to_scan.append(("(memory)", idx + 1, str(line).rstrip("\r\n")))

    if isinstance(paths, list):
        for path_entry in paths:
            if not isinstance(path_entry, str):
                continue
            path_obj = os.path.abspath(os.path.expanduser(path_entry))
            if os.path.isfile(path_obj):
                try:
                    with open(path_obj, "r", encoding="utf-8", errors="replace") as f:
                        for l_idx, line in enumerate(f):
                            items_to_scan.append((path_entry, l_idx + 1, line.rstrip("\r\n")))
                except Exception:
                    continue
            elif os.path.isdir(path_obj):
                if recursive:
                    for root, _, files in os.walk(path_obj):
                        for file_name in sorted(files):
                            if file_name.startswith("."):
                                continue
                            full_fpath = os.path.join(root, file_name)
                            try:
                                with open(full_fpath, "r", encoding="utf-8", errors="replace") as f:
                                    for l_idx, line in enumerate(f):
                                        items_to_scan.append((full_fpath, l_idx + 1, line.rstrip("\r\n")))
                            except Exception:
                                continue
                else:
                    for entry in sorted(os.listdir(path_obj)):
                        full_fpath = os.path.join(path_obj, entry)
                        if os.path.isfile(full_fpath) and not entry.startswith("."):
                            try:
                                with open(full_fpath, "r", encoding="utf-8", errors="replace") as f:
                                    for l_idx, line in enumerate(f):
                                        items_to_scan.append((full_fpath, l_idx + 1, line.rstrip("\r\n")))
                            except Exception:
                                continue

    if not items_to_scan:
        result_payload = {
            "query_expression": str(ast),
            "level": level,
            "threshold": threshold,
            "total_lines_scanned": 0,
            "total_matches": 0,
            "matches": [],
            "timing_ms": round((time.perf_counter() - t0) * 1000.0, 2)
        }
        return {
            "isError": False,
            "content": [{"type": "text", "text": json.dumps(result_payload, indent=2, ensure_ascii=False)}]
        }

    endpoint = resolve_endpoint()
    token = resolve_api_token()
    batch_size = 30
    matched_results: List[Dict[str, Any]] = []

    loop = asyncio.get_event_loop()

    for b_start in range(0, len(items_to_scan), batch_size):
        b_items = items_to_scan[b_start : b_start + batch_size]
        b_texts = [item[2] for item in b_items]

        try:
            def _sync_query():
                return query_batch_decisions(
                    endpoint=endpoint,
                    token=token,
                    states=b_texts,
                    patterns=patterns,
                    model="typesafe/zero-1.13"
                )
            batch_probs, batch_unresolved = await loop.run_in_executor(None, _sync_query)
        except Exception as e:
            return {
                "isError": True,
                "content": [{
                    "type": "text",
                    "text": (
                        f"Connection Error: Could not reach Gen-Zero decision microservice at {endpoint}.\n"
                        f"Details: {str(e)}\n"
                        f"Please ensure the local microservice is running: `systemctl --user status genzero-server.service`"
                    )
                }]
            }

        unresolved_by_row: Dict[int, List[str]] = {}
        for row_idx, pat, reason in batch_unresolved:
            unresolved_by_row.setdefault(row_idx, []).append(f"{pat!r}: {reason}")

        for offset, row_probs in enumerate(batch_probs):
            src, line_no, content = b_items[offset]
            if offset in unresolved_by_row:
                # Fail-closed: never let a kernel abstain silently become probability 0.0
                # (a confident non-match). Surface it as a match needing human review instead.
                matched_results.append({
                    "source": src,
                    "line_number": line_no,
                    "content": content,
                    "confidence": None,
                    "status": "UNRESOLVED",
                    "unresolved_reasons": unresolved_by_row[offset],
                })
                if len(matched_results) >= max_matches:
                    break
                continue
            score = ast.eval(row_probs)
            if score >= threshold:
                matched_results.append({
                    "source": src,
                    "line_number": line_no,
                    "content": content,
                    "confidence": round(float(score), 4)
                })
                if len(matched_results) >= max_matches:
                    break
        if len(matched_results) >= max_matches:
            break

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    result_payload = {
        "query_expression": str(ast),
        "level": level,
        "threshold": threshold,
        "total_lines_scanned": len(items_to_scan),
        "total_matches": len(matched_results),
        "matches": matched_results,
        "timing_ms": round(elapsed_ms, 2)
    }

    return {
        "isError": False,
        "content": [{"type": "text", "text": json.dumps(result_payload, indent=2, ensure_ascii=False)}]
    }


async def execute_zero_compact(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Lossless verbatim context compactor (0-token, 0.0% fact mutation rate).

    Evaluates conversation messages or verbose log text and applies three-way deterministic pruning:
    1. KEEP_VERBATIM: Exact bytes preserved for user intents, errors, constraints, and recent turns.
    2. TRUNCATE_OUTPUT: Keeps head and tail lines of verbose logs, stubs out noisy middle with lossless fingerprint.
    3. DROP: Removes resolved, ephemeral exploratory probe outputs (pwd, whoami, repeated checks).
    4. RESTORE: Inverse operation recovering 100% bit-exact verbatim bytes from lossless fingerprint registry.
    """
    rejected = _reject_unsupported_constraints(arguments, 'compact')
    if rejected is not None:
        return rejected
    if not isinstance(arguments, dict):
        return {
            "isError": True,
            "content": [{"type": "text", "text": "Validation Error: 'arguments' must be a JSON dictionary."}]
        }

    t0 = time.perf_counter()
    action = str(arguments.get("action", "")).lower()
    head_lines = int(arguments.get("head_lines", 5))
    tail_lines = int(arguments.get("tail_lines", 5))
    truncate_line_threshold = int(arguments.get("truncate_line_threshold", 15))

    from ..compactor.verbatim_compactor import VerbatimContextCompactor

    compactor = VerbatimContextCompactor(
        head_lines=head_lines,
        tail_lines=tail_lines,
        truncate_line_threshold=truncate_line_threshold,
    )

    # --------------------------------------------------------------------------
    # Restoration / Decompression Branch
    # --------------------------------------------------------------------------
    is_restore = action in ("restore", "decompress") or arguments.get("restore", False) or ("restore_messages" in arguments) or ("restore_text" in arguments)
    if is_restore:
        lossless_registry = arguments.get("lossless_registry")
        restore_msgs = arguments.get("restore_messages") or arguments.get("messages")
        if restore_msgs is not None:
            if isinstance(restore_msgs, str):
                try:
                    restore_msgs = json.loads(restore_msgs)
                except Exception:
                    pass
            if isinstance(restore_msgs, list):
                restored_msgs, stats = compactor.restore_session(restore_msgs, lossless_registry)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                result = {
                    "action": "restore",
                    "restored_messages": restored_msgs,
                    "recovery_stats": stats,
                    "is_bit_exact": stats.get("all_verified", False),
                    "fact_mutation_rate": stats.get("fact_mutation_rate", 0.0),
                    "timing_ms": round(elapsed_ms, 2),
                }
                return {
                    "isError": False,
                    "content": [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False)}]
                }

        restore_txt = arguments.get("restore_text") or arguments.get("text") or arguments.get("content") or arguments.get("log")
        if restore_txt is not None:
            restored_str, stats = compactor.restore_truncated_text(str(restore_txt), lossless_registry)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            result = {
                "action": "restore",
                "restored_text": restored_str,
                "recovery_stats": stats,
                "is_bit_exact": stats.get("all_verified", False),
                "fact_mutation_rate": stats.get("fact_mutation_rate", 0.0),
                "timing_ms": round(elapsed_ms, 2),
            }
            return {
                "isError": False,
                "content": [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False)}]
            }

    # --------------------------------------------------------------------------
    # Compaction Branch
    # --------------------------------------------------------------------------
    messages = arguments.get("messages")
    if messages is not None:
        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except Exception:
                pass
        if isinstance(messages, list):
            compacted_msgs, items, summary = compactor.compact_session(messages)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            comparison = {
                "time": {
                    "compaction_execution_ms": round(elapsed_ms, 2),
                    "estimated_transfer_before_ms": summary.estimated_network_latency_before_ms,
                    "estimated_transfer_after_ms": summary.estimated_network_latency_after_ms,
                    "speedup": f"{summary.network_latency_speedup}x",
                },
                "space": {
                    "original_bytes": summary.original_bytes,
                    "compacted_bytes": summary.compacted_bytes,
                    "bytes_saved": (summary.original_bytes or 0) - (summary.compacted_bytes or 0),
                    "space_savings_percent": f"{summary.space_savings_pct}%",
                    "token_compression_ratio": f"{round(summary.compression_ratio * 100, 1)}%",
                },
                "fidelity": {
                    "fact_mutation_rate": summary.fact_mutation_rate,
                    "registered_fingerprints": len(compactor.lossless_registry),
                }
            }
            result = {
                "compacted_messages": compacted_msgs,
                "summary": summary.to_dict(),
                "items": [item.to_dict() for item in items],
                "lossless_registry": compactor.lossless_registry,
                "comparison": comparison,
                "timing_ms": round(elapsed_ms, 2),
            }
            return {
                "isError": False,
                "content": [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False)}]
            }

    raw_text = arguments.get("text") or arguments.get("content") or arguments.get("log")
    if raw_text is not None:
        raw_str = str(raw_text)
        orig_bytes = len(raw_str.encode("utf-8"))
        orig_tokens = compactor.estimate_tokens(raw_str)
        compacted_str = compactor.truncate_text(raw_str, head=head_lines, tail=tail_lines)
        compacted_bytes = len(compacted_str.encode("utf-8"))
        compacted_tokens = compactor.estimate_tokens(compacted_str)
        restored_str, restore_stats = compactor.restore_truncated_text(compacted_str)
        roundtrip_ok = bool(restore_stats.get("all_verified")) and restored_str == raw_str
        fact_mutation_rate = 0.0 if roundtrip_ok else None
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        space_savings = max(0.0, 1.0 - (compacted_bytes / max(1, orig_bytes))) * 100.0
        net_before_ms = round(5.0 + (orig_bytes / 12500.0), 2)
        net_after_ms = round(0.4 + (compacted_bytes / 12500.0), 2)
        speedup = round(net_before_ms / max(0.01, net_after_ms), 2)

        comparison = {
            "time": {
                "compaction_execution_ms": round(elapsed_ms, 2),
                "estimated_transfer_before_ms": net_before_ms,
                "estimated_transfer_after_ms": net_after_ms,
                "speedup": f"{speedup}x",
            },
            "space": {
                "original_bytes": orig_bytes,
                "compacted_bytes": compacted_bytes,
                "bytes_saved": orig_bytes - compacted_bytes,
                "space_savings_percent": f"{round(space_savings, 2)}%",
                "token_compression_ratio": f"{round((1.0 - compacted_tokens / max(orig_tokens, 1)) * 100, 1)}%",
            },
            "fidelity": {
                # Measured by restoring the compacted text and comparing it to the input;
                # None when the round trip did not verify (never a hard-coded 0.0).
                "fact_mutation_rate": fact_mutation_rate,
                "restore_bit_exact": roundtrip_ok,
                "registered_fingerprints": len(compactor.lossless_registry),
            }
        }
        result = {
            "compacted_text": compacted_str,
            "original_tokens": orig_tokens,
            "compacted_tokens": compacted_tokens,
            "tokens_saved": orig_tokens - compacted_tokens,
            "compression_ratio": round(1.0 - (compacted_tokens / max(orig_tokens, 1)), 4),
            "fact_mutation_rate": fact_mutation_rate,
            "lossless_registry": compactor.lossless_registry,
            "comparison": comparison,
            "timing_ms": round(elapsed_ms, 2),
        }
        return {
            "isError": False,
            "content": [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False)}]
        }

    return {
        "isError": True,
        "content": [{
            "type": "text",
            "text": "Validation Error: 'compact' mode requires either 'messages' (list of message dicts) or 'text' / 'content' / 'log' string."
        }]
    }


def resolve_zero_stream_mode(arguments: Any) -> bool:
    """True iff ``execute_zero(arguments)`` will dispatch to stream mode.

    Mirrors the explicit-action and implicit-detection precedence inside ``execute_zero``
    (route takes priority over stream when both 'tools'+'task_goal' and 'observation' are
    present). Factored out so ``MCPServer.handle_request`` can decide whether to acquire a
    session-bound streaming engine (A07) before dispatch, without duplicating or drifting out
    of sync with the real dispatch logic below.
    """
    if not isinstance(arguments, dict):
        return False
    explicit_action = arguments.get("action")
    if explicit_action:
        return str(explicit_action).strip().lower() in ("stream", "step")
    if "tools" in arguments and "task_goal" in arguments:
        return False
    return "observation" in arguments


async def execute_zero(arguments: Dict[str, Any], stream_engine: Optional[Any] = None) -> Dict[str, Any]:
    """Universal Gen-Zero Polymorphic Entrypoint (Scheme B: Adaptive Implicit Polymorphism).

    Dispatches dynamically to one of Gen-Zero's specialized micro-services:
    - 'route': Tool candidate pruning (arguments: task_goal, tools)
    - 'imagine': World model counterfactual tree lookahead (arguments: candidate_actions, horizon / state)
    - 'stream': Spatial-temporal rolling cache streaming (arguments: observation, candidate_actions)
    - 'grep': Semantic grep & boolean filter (arguments: query/expr/or_patterns/lines/paths)
    - 'ask': Non-autoregressive decision questions (arguments: questions, state)
    - 'compact': Lossless verbatim context compactor (arguments: messages / text)

    Supports explicit `action` override ('ask', 'route', 'imagine', 'stream', 'grep', 'compact')
    or automatically deduces intent from the presence of distinctive arguments.
    """
    if not isinstance(arguments, dict):
        return {
            "isError": True,
            "content": [{"type": "text", "text": "Validation Error: 'arguments' must be a JSON dictionary."}]
        }

    explicit_action = arguments.get("action")
    if explicit_action:
        act = str(explicit_action).strip().lower()
        if act in ("ask", "decide"):
            return await execute_zero_ask(arguments)
        elif act in ("route", "prune"):
            return await execute_zero_route(arguments)
        elif act in ("imagine", "simulate"):
            return await execute_zero_imagine(arguments)
        elif act in ("stream", "step"):
            return await execute_zero_stream(arguments, engine=stream_engine)
        elif act in ("grep", "search", "filter"):
            return await execute_zero_grep(arguments)
        elif act in ("compact", "compress", "prune_context", "restore", "decompress"):
            return await execute_zero_compact(arguments)
        else:
            return {
                "isError": True,
                "content": [{
                    "type": "text",
                    "text": f"Validation Error: Unknown action '{explicit_action}'. Allowed values: 'ask', 'route', 'imagine', 'stream', 'grep', 'compact', 'restore'."
                }]
            }

    # Implicit polymorphic auto-detection
    # 1. Route: tools and task_goal provided
    if "tools" in arguments and "task_goal" in arguments:
        return await execute_zero_route(arguments)

    # 2. Stream: observation provided
    if "observation" in arguments:
        return await execute_zero_stream(arguments, engine=stream_engine)

    # 3. Imagine: candidate_actions provided with horizon or state without questions
    if "candidate_actions" in arguments and ("horizon" in arguments or ("state" in arguments and "questions" not in arguments)):
        return await execute_zero_imagine(arguments)

    # 4. Grep: expr, or_patterns, and_patterns, not_patterns, lines, paths, or query
    if any(k in arguments for k in ("expr", "or_patterns", "and_patterns", "not_patterns")) or ("lines" in arguments or "paths" in arguments):
        return await execute_zero_grep(arguments)

    # 5. Compact: messages or (text/content/log without state/questions)
    if "messages" in arguments:
        return await execute_zero_compact(arguments)

    # 6. Ask / Decide: questions provided, or state with questions
    if "questions" in arguments:
        return await execute_zero_ask(arguments)

    # If query is provided without lines/paths/questions, default to grep
    if "query" in arguments:
        return await execute_zero_grep(arguments)

    # If state is provided without candidate_actions, default to ask
    if "state" in arguments:
        return await execute_zero_ask(arguments)

    # If text/log is provided without other keys, default to compact
    if any(k in arguments for k in ("text", "log")):
        return await execute_zero_compact(arguments)

    return {
        "isError": True,
        "content": [{
            "type": "text",
            "text": (
                "Ambiguous invocation of 'zero': Could not infer mode from arguments. "
                "Please specify 'action' explicitly ('ask', 'route', 'imagine', 'stream', 'grep', 'compact') "
                "or supply distinguishing parameters:\n"
                "- 'ask': 'questions' and 'state'\n"
                "- 'route': 'task_goal' and 'tools'\n"
                "- 'imagine': 'candidate_actions' and 'state'\n"
                "- 'stream': 'observation' and 'candidate_actions'\n"
                "- 'grep': 'query'/'expr' and 'lines'/'paths'\n"
                "- 'compact': 'messages' or 'text'/'log'"
            )
        }]
    }


class MCPServer:
    """Model Context Protocol stdio JSON-RPC server handler.

    Owns the session_id -> StreamingWorldModelEngine cache for 'stream' mode (A07). A single
    MCPServer instance is shared across every concurrent client in SSE transport mode (see
    sse_transport.py), so this cache -- not the stateless execute_zero_stream() call -- is what
    lets a streaming episode's KV-cache and causal-shock history survive between JSON-RPC calls.
    """

    def __init__(
        self,
        stream_session_ttl_s: float = DEFAULT_STREAM_SESSION_TTL_SECONDS,
        max_stream_sessions: int = DEFAULT_MAX_STREAM_SESSIONS,
    ):
        # Canonical single polymorphic tool
        self.tools = {
            "zero": execute_zero,
        }
        self._stream_sessions: Dict[str, Dict[str, Any]] = {}
        self._stream_session_lock = asyncio.Lock()
        self._stream_session_ttl_s = float(stream_session_ttl_s)
        self._max_stream_sessions = int(max_stream_sessions)

    async def _acquire_stream_engine(self, session_id: Optional[str]) -> Any:
        """Returns the StreamingWorldModelEngine for ``session_id``, creating one if needed.

        Without a session_id (``None``), returns a fresh ephemeral engine that is never cached:
        the caller explicitly opted out of history continuity, so this is not a degradation,
        just a stateless call (logged at debug level so it's still visible if someone expected
        continuity by mistake).

        A session_id that IS provided but is not a non-empty string is a caller error, not a
        signal to degrade: raises ValueError instead of silently falling back to an anonymous
        ephemeral engine, which would hide a broken caller (e.g. one passing an int or "")
        behind output that looks like a normal stateless call.
        """
        from gen_zero.world_model.streaming_engine import StreamingWorldModelEngine

        if session_id is None:
            logger.debug(
                "zero_stream called without session_id: using a fresh ephemeral engine; "
                "KV-cache and causal-shock history will not carry over to the next call."
            )
            return StreamingWorldModelEngine()

        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError(
                f"session_id must be a non-empty string when provided; got {session_id!r} "
                f"(type={type(session_id).__name__})."
            )

        now = time.time()
        async with self._stream_session_lock:
            self._evict_expired_stream_sessions_locked(now)
            entry = self._stream_sessions.get(session_id)
            if entry is None:
                if len(self._stream_sessions) >= self._max_stream_sessions:
                    self._evict_lru_stream_session_locked()
                entry = {"engine": StreamingWorldModelEngine(), "last_used": now}
                self._stream_sessions[session_id] = entry
                logger.info(
                    "zero_stream: created new engine for session_id=%r (active_sessions=%d)",
                    session_id, len(self._stream_sessions),
                )
            else:
                entry["last_used"] = now
            return entry["engine"]

    def _evict_expired_stream_sessions_locked(self, now: float) -> None:
        """Drops engines idle for longer than the TTL. Caller must hold _stream_session_lock."""
        expired = [
            sid for sid, entry in self._stream_sessions.items()
            if now - entry["last_used"] > self._stream_session_ttl_s
        ]
        for sid in expired:
            del self._stream_sessions[sid]
        if expired:
            logger.info("zero_stream: evicted %d idle session(s) past TTL=%.0fs", len(expired), self._stream_session_ttl_s)

    def _evict_lru_stream_session_locked(self) -> None:
        """Drops the least-recently-used engine to stay under the session cap. Caller must hold the lock."""
        if not self._stream_sessions:
            return
        oldest_sid = min(self._stream_sessions, key=lambda sid: self._stream_sessions[sid]["last_used"])
        del self._stream_sessions[oldest_sid]
        logger.warning(
            "zero_stream: session cap (%d) reached; evicted LRU session_id=%r",
            self._max_stream_sessions, oldest_sid,
        )

    async def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        msg_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        # Notifications (no id, no response required)
        if msg_id is None:
            if method == "notifications/initialized":
                pass
            return None

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": SERVER_VERSION
                    }
                }
            }

        elif method == "ping":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {}
            }

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": [
                        ZERO_SCHEMA,
                    ]
                }
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            handler = self.tools.get(tool_name)
            if not handler:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32601,
                        "message": f"Tool '{tool_name}' not found."
                    }
                }
            if tool_name == "zero" and resolve_zero_stream_mode(tool_args):
                # A07 gate 2: validate the request BEFORE touching any session state. A malformed
                # request (bad observation/candidate_actions) must never create a new session or
                # evict an unrelated, legitimate session's engine via the LRU cap.
                validation_error = _validate_zero_stream_arguments(tool_args)
                if validation_error is not None:
                    tool_result = validation_error
                else:
                    session_id = tool_args.get("session_id") if isinstance(tool_args, dict) else None
                    try:
                        engine = await self._acquire_stream_engine(session_id)
                    except ValueError as exc:
                        # A07 gate 1: a session_id of the wrong type/empty is a caller error,
                        # surfaced explicitly -- never silently downgraded to an ephemeral engine.
                        tool_result = {
                            "isError": True,
                            "content": [{"type": "text", "text": f"Validation Error: {exc}"}],
                        }
                    else:
                        tool_result = await execute_zero(tool_args, stream_engine=engine)
            else:
                tool_result = await handler(tool_args)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": tool_result
            }

        else:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Method '{method}' not recognized."
                }
            }


async def run_server():
    """Runs the asynchronous MCP stdio JSON-RPC loop."""
    server = MCPServer()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    loop = asyncio.get_event_loop()
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line_bytes = await reader.readline()
        if not line_bytes:
            break
        line_str = line_bytes.decode("utf-8").strip()
        if not line_str:
            continue

        try:
            req = json.loads(line_str)
            resp = await server.handle_request(req)
            if resp is not None:
                out_str = json.dumps(resp, ensure_ascii=False) + "\n"
                sys.stdout.write(out_str)
                sys.stdout.flush()
        except json.JSONDecodeError:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error: Invalid JSON"}
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Gen-Zero Canonical Model Context Protocol (MCP) Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default=os.environ.get("GENZERO_MCP_TRANSPORT", "stdio"),
        help="Transport layer protocol: 'stdio' (default) or 'sse' (Server-Sent Events network server)."
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("GENZERO_MCP_HOST", "0.0.0.0"),
        help="Host address to bind for SSE transport (default: 0.0.0.0)."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("GENZERO_MCP_PORT", "8999")),
        help="Port to bind for SSE transport (default: 8999)."
    )
    parser.add_argument(
        "--token",
        "--auth-token",
        dest="auth_token",
        default=os.environ.get("GENZERO_MCP_AUTH_TOKEN", None),
        help="Optional Bearer authentication token for SSE transport."
    )

    args = parser.parse_args()

    if args.transport == "sse":
        try:
            from .sse_transport import run_sse_server
        except ImportError:
            from gen_zero.mcp.sse_transport import run_sse_server
        server = MCPServer()
        run_sse_server(mcp_server=server, host=args.host, port=args.port, auth_token=args.auth_token)
    else:
        try:
            asyncio.run(run_server())
        except (KeyboardInterrupt, BrokenPipeError):
            pass


if __name__ == "__main__":
    main()
