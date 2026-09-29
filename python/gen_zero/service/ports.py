"""Default ports, kept apart from the heavy service modules so the CLI can read them.

8999 is the MCP SSE port (Rust ``gen-zero mcp`` and ``gen_zero.mcp``). The
semantic scorer must not share it: the Rust bridge would then call the MCP
server (or whatever else holds 8999) and get 404 instead of scores.
"""

DEFAULT_SEMANTIC_PORT = 8995
