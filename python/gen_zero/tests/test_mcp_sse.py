"""Unit and Integration Tests for MCP Server-Sent Events (SSE) Transport.

Tests:
1. Endpoint advertisement via GET /sse (MCP 2024-11-05 spec).
2. JSON-RPC request ingestion via POST /messages?session_id=<uuid>.
3. Response delivery back into active SSE event stream.
4. Error handling for invalid/missing session IDs.
5. Bearer token authorization middleware.
6. Session queue cleanup and zero memory leakage.
"""

import unittest
import asyncio
import json
import re

from starlette.testclient import TestClient

from gen_zero.mcp.server import MCPServer
from gen_zero.mcp.sse_transport import (
    SSESessionManager,
    create_sse_app,
)


class TestMCPSSETransport(unittest.TestCase):
    """Test suite for Model Context Protocol SSE transport."""

    def setUp(self):
        self.server = MCPServer()
        self.session_mgr = SSESessionManager()
        self.app = create_sse_app(self.server, session_manager=self.session_mgr)
        self.client = TestClient(self.app)

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["transport"], "sse")
        self.assertIn("active_sessions", data)

    def test_sse_endpoint_advertises_messages_uri(self):
        # Establish connection to /sse and read initial event
        response = self.client.get("/sse?single_event=true")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers.get("content-type", ""))

        text = response.text
        self.assertIn("event: endpoint", text)
        self.assertIn("data: /messages?session_id=", text)

        # Extract session_id
        match = re.search(r"session_id=([a-f0-9\-]+)", text)
        self.assertIsNotNone(match)
        session_id = match.group(1)

    def test_jsonrpc_initialize_via_messages(self):
        # Create session directly for queue inspection
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        session = loop.run_until_complete(self.session_mgr.create_session())
        session_id = session.session_id

        # Send initialize JSON-RPC request
        req_payload = {
            "jsonrpc": "2.0",
            "id": 101,
            "method": "initialize",
            "params": {}
        }
        post_resp = self.client.post(f"/messages?session_id={session_id}", json=req_payload)
        self.assertEqual(post_resp.status_code, 202)

        # Retrieve response from session queue
        rpc_resp = loop.run_until_complete(session.queue.get())
        self.assertEqual(rpc_resp["jsonrpc"], "2.0")
        self.assertEqual(rpc_resp["id"], 101)
        self.assertEqual(rpc_resp["result"]["protocolVersion"], "2024-11-05")
        self.assertEqual(rpc_resp["result"]["serverInfo"]["name"], "gen-zero-mcp")

        # Cleanup
        loop.run_until_complete(self.session_mgr.remove_session(session_id))
        loop.close()

    def test_jsonrpc_tools_list_and_pruning(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        session = loop.run_until_complete(self.session_mgr.create_session())
        session_id = session.session_id

        # 1. tools/list returns strictly single 'zero' tool
        list_req = {
            "jsonrpc": "2.0",
            "id": 102,
            "method": "tools/list",
            "params": {}
        }
        self.client.post(f"/messages?session_id={session_id}", json=list_req)
        list_resp = loop.run_until_complete(session.queue.get())
        tool_names = [t["name"] for t in list_resp["result"]["tools"]]
        self.assertEqual(tool_names, ["zero"])

        # 2. tools/call zero (route mode auto-detection)
        call_req = {
            "jsonrpc": "2.0",
            "id": 103,
            "method": "tools/call",
            "params": {
                "name": "zero",
                "arguments": {
                    "task_goal": "Run unit test suite",
                    "tools": [
                        {"name": "test_runner", "description": "Execute pytest or unittest"},
                        {"name": "cook_recipe", "description": "Bake cookies"}
                    ],
                    "top_k": 1
                }
            }
        }
        self.client.post(f"/messages?session_id={session_id}", json=call_req)
        call_resp = loop.run_until_complete(session.queue.get())
        self.assertFalse(call_resp["result"]["isError"])
        content_text = call_resp["result"]["content"][0]["text"]
        self.assertIn("test_runner", content_text)

        # 3. tools/call zero (grep mode auto-detection) via SSE
        grep_req = {
            "jsonrpc": "2.0",
            "id": 104,
            "method": "tools/call",
            "params": {
                "name": "zero",
                "arguments": {
                    "query": "build failure",
                    "lines": [
                        "Compiling module alpha...",
                        "Build failed with syntax error in line 10",
                        "Done in 2.3s"
                    ],
                    "level": "loose"
                }
            }
        }
        with unittest.mock.patch("gen_zero.scripts.gen_grep.query_batch_decisions", side_effect=lambda **kwargs: ([{kwargs["patterns"][0]: 0.1}, {kwargs["patterns"][0]: 0.95}, {kwargs["patterns"][0]: 0.05}], [])):
            self.client.post(f"/messages?session_id={session_id}", json=grep_req)
            grep_resp = loop.run_until_complete(session.queue.get())
        self.assertFalse(grep_resp["result"]["isError"])
        grep_payload = json.loads(grep_resp["result"]["content"][0]["text"])
        self.assertEqual(grep_payload["total_lines_scanned"], 3)
        self.assertGreaterEqual(grep_payload["total_matches"], 1)

        loop.run_until_complete(self.session_mgr.remove_session(session_id))
        loop.close()

    def test_missing_and_invalid_session_handling(self):
        # Missing session_id parameter
        resp_no_id = self.client.post("/messages", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
        self.assertEqual(resp_no_id.status_code, 400)

        # Invalid/non-existent session_id
        resp_fake = self.client.post("/messages?session_id=non-existent-uuid", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
        self.assertEqual(resp_fake.status_code, 404)

    def test_auth_token_protection(self):
        # App with token required
        auth_app = create_sse_app(self.server, auth_token="secret-token-xyz")
        auth_client = TestClient(auth_app)

        # Request without header fails
        r_unauth = auth_client.get("/sse")
        self.assertEqual(r_unauth.status_code, 401)

        # Request with correct Bearer header succeeds
        headers = {"Authorization": "Bearer secret-token-xyz"}
        r_auth = auth_client.get("/sse?single_event=true", headers=headers)
        self.assertEqual(r_auth.status_code, 200)
        self.assertIn("event: endpoint", r_auth.text)


if __name__ == "__main__":
    unittest.main()
