"""Gen-Zero Universal MCP Decision & Cognitive Extension Server.

Provides a unified Model Context Protocol (MCP) stdio interface exposing
the single polymorphic `zero` tool to developer coding agents (Codex, Antigravity, Cursor, pi).
"""

from .server import (
    run_server,
    main,
    MCPServer,
    ZERO_SCHEMA,
    execute_zero,
    execute_zero_ask,
    execute_zero_route,
    execute_zero_imagine,
    execute_zero_stream,
    execute_zero_grep,
    execute_zero_compact,
)
from .sse_transport import (
    create_sse_app,
    run_sse_server,
    SSESessionManager,
    SSESession,
)

__all__ = [
    "run_server",
    "main",
    "MCPServer",
    "ZERO_SCHEMA",
    "execute_zero",
    "execute_zero_ask",
    "execute_zero_route",
    "execute_zero_imagine",
    "execute_zero_stream",
    "execute_zero_grep",
    "execute_zero_compact",
    "create_sse_app",
    "run_sse_server",
    "SSESessionManager",
    "SSESession",
]
