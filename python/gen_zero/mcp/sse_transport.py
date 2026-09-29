"""MCP Server-Sent Events (SSE) Transport Handler (2024-11-05 Specification).

Implements Issue #48:
- GET /sse: Establishes persistent text/event-stream channel and announces endpoint URI.
- POST /messages?session_id=<uuid>: Ingests JSON-RPC 2.0 requests and dispatches to MCPServer.
- Async session queue management with zero-leak teardown on client disconnection.
- Optional Bearer token authorization middleware.
"""

from typing import Dict, Any, Optional, AsyncGenerator
import asyncio
import json
import time
import uuid
import logging

try:
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.requests import Request
    from starlette.responses import Response, StreamingResponse, JSONResponse
    from starlette.middleware import Middleware
    from starlette.middleware.cors import CORSMiddleware
    import uvicorn
    HAS_STARLETTE = True
except ImportError:
    HAS_STARLETTE = False


logger = logging.getLogger("gen_zero.mcp.sse")


class SSESession:
    """Represents an active client SSE connection and its message queue."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.queue: asyncio.Queue = asyncio.Queue()
        self.created_at = time.time()
        self.last_active = time.time()
        self.is_connected = True

    def touch(self):
        self.last_active = time.time()

    async def send_message(self, message: Dict[str, Any]):
        """Pushes a JSON-RPC message into the SSE event stream queue."""
        self.touch()
        await self.queue.put(message)

    async def stream_events(self) -> AsyncGenerator[str, None]:
        """Yields formatted SSE data packets for the connected client."""
        # Initial event announcing the endpoint per MCP 2024-11-05 specification
        endpoint_uri = f"/messages?session_id={self.session_id}"
        yield f"event: endpoint\r\ndata: {endpoint_uri}\r\n\r\n"

        try:
            while self.is_connected:
                # Wait for next JSON-RPC response or notification
                msg = await self.queue.get()
                if msg is None:  # Sentinel to close stream
                    break
                data_str = json.dumps(msg, ensure_ascii=False)
                yield f"event: message\r\ndata: {data_str}\r\n\r\n"
        except asyncio.CancelledError:
            pass
        finally:
            self.is_connected = False


class SSESessionManager:
    """Manages active SSE client sessions and their lifecycle."""

    def __init__(self):
        self.sessions: Dict[str, SSESession] = {}
        self._lock = asyncio.Lock()

    async def create_session(self) -> SSESession:
        """Registers a new SSE session."""
        session_id = str(uuid.uuid4())
        session = SSESession(session_id)
        async with self._lock:
            self.sessions[session_id] = session
        logger.info(f"Created SSE session: {session_id} (active: {len(self.sessions)})")
        return session

    async def get_session(self, session_id: str) -> Optional[SSESession]:
        """Retrieves an active session by session_id."""
        async with self._lock:
            session = self.sessions.get(session_id)
            if session:
                session.touch()
            return session

    async def remove_session(self, session_id: str):
        """Cleans up and removes an SSE session."""
        async with self._lock:
            session = self.sessions.pop(session_id, None)
            if session:
                session.is_connected = False
                await session.queue.put(None)  # Wake up streaming loop
        logger.info(f"Removed SSE session: {session_id} (active: {len(self.sessions)})")


def create_sse_app(
    mcp_server,
    session_manager: Optional[SSESessionManager] = None,
    auth_token: Optional[str] = None,
) -> Any:
    """Creates a Starlette ASGI application implementing MCP SSE transport."""
    if not HAS_STARLETTE:
        raise ImportError(
            "Starlette and Uvicorn are required for MCP SSE transport. "
            "Install with `pip install starlette uvicorn`."
        )

    mgr = session_manager or SSESessionManager()

    def check_auth(request: Request) -> bool:
        if not auth_token:
            return True
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            return token == auth_token
        return False

    async def sse_endpoint(request: Request) -> Response:
        """GET /sse: Establishes SSE stream connection."""
        if not check_auth(request):
            return Response("Unauthorized", status_code=401)

        session = await mgr.create_session()
        single_event = request.query_params.get("single_event") == "true"

        async def event_generator():
            try:
                async for chunk in session.stream_events():
                    yield chunk.encode("utf-8")
                    if single_event:
                        break
            finally:
                await mgr.remove_session(session.session_id)

        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(event_generator(), headers=headers)

    async def messages_endpoint(request: Request) -> Response:
        """POST /messages?session_id=<uuid>: Receives JSON-RPC requests."""
        if not check_auth(request):
            return Response("Unauthorized", status_code=401)

        session_id = request.query_params.get("session_id")
        if not session_id:
            return JSONResponse({"error": "Missing 'session_id' parameter"}, status_code=400)

        session = await mgr.get_session(session_id)
        if not session:
            return JSONResponse({"error": "Session not found or expired"}, status_code=404)

        try:
            body = await request.json()
        except Exception as e:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {str(e)}"}
            }
            await session.send_message(err_resp)
            return Response("Accepted", status_code=202)

        # Handle request via underlying MCPServer
        try:
            resp = await mcp_server.handle_request(body)
            if resp is not None:
                await session.send_message(resp)
        except Exception as e:
            err_resp = {
                "jsonrpc": "2.0",
                "id": body.get("id") if isinstance(body, dict) else None,
                "error": {"code": -32603, "message": f"Internal server error: {str(e)}"}
            }
            await session.send_message(err_resp)

        return Response("Accepted", status_code=202)

    async def health_endpoint(request: Request) -> Response:
        """GET /health: Healthcheck."""
        return JSONResponse({
            "status": "healthy",
            "transport": "sse",
            "active_sessions": len(mgr.sessions),
        })

    routes = [
        Route("/sse", sse_endpoint, methods=["GET"]),
        Route("/messages", messages_endpoint, methods=["POST"]),
        Route("/health", health_endpoint, methods=["GET"]),
    ]

    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )
    ]

    app = Starlette(routes=routes, middleware=middleware)
    app.state.session_manager = mgr
    return app


def run_sse_server(
    mcp_server,
    host: str = "0.0.0.0",
    port: int = 8999,
    auth_token: Optional[str] = None,
    log_level: str = "info",
):
    """Runs Uvicorn server hosting MCP SSE transport."""
    app = create_sse_app(mcp_server, auth_token=auth_token)
    print(f"[gen-zero-mcp] Starting SSE server on http://{host}:{port}/sse")
    uvicorn.run(app, host=host, port=port, log_level=log_level)
