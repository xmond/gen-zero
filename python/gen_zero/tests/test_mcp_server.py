import unittest
import asyncio
import json
import os
from unittest.mock import patch, MagicMock

from gen_zero.mcp.server import (
    MCPServer,
    execute_zero,
    execute_zero_ask,
    execute_zero_route,
    execute_zero_imagine,
    execute_zero_stream,
    execute_zero_grep,
    resolve_api_token,
    resolve_endpoint,
    ZERO_SCHEMA,
)


class TestMCPServer(unittest.IsolatedAsyncioTestCase):
    """Rigorous verification suite for Gen-Zero Universal MCP Decision Server with Single Tool 'zero'."""

    async def asyncSetUp(self):
        self.server = MCPServer()

    async def test_01_initialize_and_ping(self):
        """Verify initialize and ping JSON-RPC methods conform to MCP 2024-11-05 spec."""
        init_req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        init_resp = await self.server.handle_request(init_req)
        self.assertEqual(init_resp["id"], 1)
        self.assertEqual(init_resp["result"]["protocolVersion"], "2024-11-05")
        self.assertEqual(init_resp["result"]["serverInfo"]["name"], "gen-zero-mcp")

        ping_req = {"jsonrpc": "2.0", "id": 2, "method": "ping", "params": {}}
        ping_resp = await self.server.handle_request(ping_req)
        self.assertEqual(ping_resp["id"], 2)
        self.assertEqual(ping_resp["result"], {})

    async def test_02_tools_list_publishes_strictly_single_zero_tool(self):
        """Verify tools/list publishes strictly ONE tool named 'zero' with polymorphic schema."""
        list_req = {"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}}
        list_resp = await self.server.handle_request(list_req)
        tools = list_resp["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        self.assertEqual(tool_names, ["zero"])

        # Verify ZERO_SCHEMA properties
        zero_tool = tools[0]
        self.assertEqual(zero_tool["name"], "zero")
        props = zero_tool["inputSchema"]["properties"]
        self.assertIn("action", props)
        self.assertIn("state", props)
        self.assertIn("questions", props)
        self.assertIn("task_goal", props)
        self.assertIn("tools", props)
        self.assertIn("candidate_actions", props)
        self.assertIn("horizon", props)
        self.assertIn("observation", props)
        self.assertIn("query", props)
        self.assertIn("lines", props)

    async def test_03_zero_ask_local_validation_guards(self):
        """Verify execute_zero_ask catches missing fields locally to prevent remote HTTP 422 errors."""
        # 1. Missing state
        res1 = await execute_zero_ask({"questions": {"q1": {"type": "choice"}}})
        self.assertTrue(res1["isError"])
        self.assertIn("'state' field is required", res1["content"][0]["text"])

        # 2. Empty questions
        res2 = await execute_zero_ask({"state": "test", "questions": {}})
        self.assertTrue(res2["isError"])
        self.assertIn("non-empty dictionary", res2["content"][0]["text"])

        # 3. Invalid question type
        res3 = await execute_zero_ask({
            "state": "test",
            "questions": {
                "q1": {"type": "invalid_kind", "instructions": "test", "criteria": ["a", "b"]}
            }
        })
        self.assertTrue(res3["isError"])
        self.assertIn("must be one of 'noul', 'choice', or 'score'", res3["content"][0]["text"])

        # 4. Missing criteria in question
        res4 = await execute_zero_ask({
            "state": "test",
            "questions": {
                "q1": {"type": "choice", "instructions": "select one"}
            }
        })
        self.assertTrue(res4["isError"])
        self.assertIn("'criteria' field is required", res4["content"][0]["text"])

    async def test_04_unknown_tool_returns_error(self):
        """Verify calling an unregistered tool returns JSON-RPC -32601."""
        req = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "non_existent_tool", "arguments": {}}
        }
        resp = await self.server.handle_request(req)
        self.assertIn("error", resp)
        self.assertEqual(resp["error"]["code"], -32601)

    async def test_05_unreachable_endpoint_graceful_error(self):
        """Verify unreachable service returns clear diagnostic message with service management hint."""
        with patch.dict(os.environ, {"GENZERO_ENDPOINT": "http://127.0.0.1:19999"}):
            res = await execute_zero_ask({
                "state": "Testing down service",
                "questions": {
                    "check": {"type": "noul", "instructions": "is ok?", "criteria": {"true": "yes", "false": "no"}}
                }
            })
            self.assertTrue(res["isError"])
            text = res["content"][0]["text"]
            self.assertIn("Could not reach Gen-Zero decision microservice", text)
            self.assertIn("systemctl --user status genzero-server.service", text)

    @unittest.skipUnless(
        os.environ.get("GENZERO_API_KEY"),
        "live integration test: needs GENZERO_API_KEY (no baked-in default token exists any more)",
    )
    async def test_06_live_end_to_end_decision_via_zero(self):
        """Live integration test invoking 'zero' tool in ask mode against microservice."""
        # 1. Test with implicit polymorphic detection (state + questions)
        req = {
            "jsonrpc": "2.0",
            "id": 100,
            "method": "tools/call",
            "params": {
                "name": "zero",
                "arguments": {
                    "state": "User requested database migration execution.",
                    "questions": {
                        "is_safe": {
                            "type": "noul",
                            "instructions": "Is migration safe to run online?",
                            "criteria": {"true": "Non-blocking schema update", "false": "Locks large tables"}
                        },
                        "action_plan": {
                            "type": "choice",
                            "instructions": "Migration execution strategy",
                            "criteria": {
                                "run_async": "Execute background migration with concurrency control",
                                "run_sync": "Run blocking migration during maintenance window"
                            }
                        },
                        "impact_rating": {
                            "type": "score",
                            "instructions": "Rate potential service impact",
                            "criteria": ["Zero impact", "Minor latency spike", "High service risk"]
                        }
                    }
                }
            }
        }
        resp = await self.server.handle_request(req)
        self.assertNotIn("error", resp)
        res = resp["result"]
        self.assertFalse(res.get("isError", False))

        parsed_payload = json.loads(res["content"][0]["text"])
        self.assertEqual(parsed_payload["model"], "typesafe/zero-1.13")
        answers = parsed_payload["answers"]

        # Verify all 3 primitives responded correctly
        self.assertIn("is_safe", answers)
        self.assertEqual(answers["is_safe"]["type"], "noul")
        self.assertIsInstance(answers["is_safe"]["noul"], float)

        self.assertIn("action_plan", answers)
        self.assertEqual(answers["action_plan"]["type"], "choice")
        self.assertIn("choice", answers["action_plan"])
        self.assertIn(answers["action_plan"]["choice"], ["run_async", "run_sync"])

        self.assertIn("impact_rating", answers)
        self.assertEqual(answers["impact_rating"]["type"], "score")
        self.assertIsInstance(answers["impact_rating"]["score"], (int, float))

    async def test_07_zero_grep_polymorphic_execution(self):
        """Verify 'zero' tool polymorphic execution in grep mode."""
        req = {
            "jsonrpc": "2.0",
            "id": 105,
            "method": "tools/call",
            "params": {
                "name": "zero",
                "arguments": {
                    "query": "database error",
                    "lines": [
                        "2026-09-20 [ERROR] Database connection lost due to socket timeout",
                        "2026-09-20 [INFO] User login successful",
                        "2026-09-20 [WARN] High memory usage detected"
                    ],
                    "level": "balanced"
                }
            }
        }
        with patch("gen_zero.scripts.gen_grep.query_batch_decisions") as mock_q:
            mock_q.return_value = (
                [
                    {"database error": 0.95},
                    {"database error": 0.05},
                    {"database error": 0.10}
                ],
                [],
            )
            resp = await self.server.handle_request(req)
        self.assertNotIn("error", resp)
        self.assertFalse(resp["result"]["isError"])
        payload = json.loads(resp["result"]["content"][0]["text"])
        self.assertEqual(payload["total_lines_scanned"], 3)
        self.assertGreaterEqual(payload["total_matches"], 1)
        self.assertEqual(payload["matches"][0]["line_number"], 1)
        self.assertIn("Database connection lost", payload["matches"][0]["content"])
        self.assertGreaterEqual(payload["matches"][0]["confidence"], 0.70)

    async def test_07b_zero_grep_unresolved_line_flagged_not_zero_confidence(self):
        """B0928 blocker 3: a kernel abstain on a grep line must surface as an explicit
        UNRESOLVED match, never as a silently-scored non-match (confidence 0.0)."""
        req = {
            "jsonrpc": "2.0",
            "id": 105,
            "method": "tools/call",
            "params": {
                "name": "zero",
                "arguments": {
                    "query": "database error",
                    "lines": [
                        "2026-09-20 [ERROR] Database connection lost due to socket timeout",
                        "2026-09-20 [INFO] User login successful",
                    ],
                    "level": "balanced"
                }
            }
        }
        with patch("gen_zero.scripts.gen_grep.query_batch_decisions") as mock_q:
            mock_q.return_value = (
                [{"database error": None}, {"database error": 0.05}],
                [(0, "database error", "INFEASIBLE_ABSTAIN")],
            )
            resp = await self.server.handle_request(req)
        self.assertNotIn("error", resp)
        self.assertFalse(resp["result"]["isError"])
        payload = json.loads(resp["result"]["content"][0]["text"])
        unresolved = [m for m in payload["matches"] if m.get("status") == "UNRESOLVED"]
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0]["line_number"], 1)
        self.assertIsNone(unresolved[0]["confidence"])
        self.assertIn("INFEASIBLE_ABSTAIN", str(unresolved[0]["unresolved_reasons"]))

    async def test_08_zero_route_polymorphic_execution(self):
        """Verify 'zero' tool polymorphic execution in route mode."""
        req = {
            "jsonrpc": "2.0",
            "id": 106,
            "method": "tools/call",
            "params": {
                "name": "zero",
                "arguments": {
                    "action": "route",
                    "task_goal": "Deploy kubernetes pod cluster",
                    "tools": [
                        {"name": "kubectl_apply", "description": "Apply k8s deployment YAML"},
                        {"name": "send_email", "description": "Send email notification"}
                    ],
                    "top_k": 1
                }
            }
        }
        resp = await self.server.handle_request(req)
        self.assertNotIn("error", resp)
        self.assertFalse(resp["result"]["isError"])
        payload = json.loads(resp["result"]["content"][0]["text"])
        self.assertEqual(payload["pruned_tool_names"], ["kubectl_apply"])
        self.assertEqual(payload["routing_method"], "lexical_baseline")
        self.assertFalse(payload["degraded"])

    async def test_08b_zero_route_non_ascii_query_is_degraded_not_silent(self):
        """B06: a non-ASCII task_goal extracts zero ASCII keywords; the lexical baseline
        must say so instead of silently returning an input-order 'ranking'."""
        res = await execute_zero_route({
            "task_goal": "\u0440\u0430\u0437\u0432\u0435\u0440\u0442\u044b\u0432\u0430\u043d\u0438\u0435 \u043a\u043b\u0430\u0441\u0442\u0435\u0440\u0430 \u043a\u043e\u043d\u0442\u0435\u0439\u043d\u0435\u0440\u043e\u0432",
            "tools": [
                {"name": "kubectl_apply", "description": "Apply k8s deployment YAML"},
                {"name": "send_email", "description": "Send email notification"}
            ],
            "top_k": 1
        })
        self.assertFalse(res["isError"])
        payload = json.loads(res["content"][0]["text"])
        self.assertEqual(payload["routing_method"], "lexical_baseline")
        self.assertTrue(payload["degraded"])
        self.assertEqual(payload["degradation_reason"], "LEXICAL_NO_ASCII_KEYWORDS")

    async def test_09_zero_imagine_and_stream_execution(self):
        """Verify 'zero' tool polymorphic execution in imagine and stream modes."""
        # Imagine mode
        imagine_req = {
            "jsonrpc": "2.0",
            "id": 107,
            "method": "tools/call",
            "params": {
                "name": "zero",
                "arguments": {
                    "action": "imagine",
                    "state": "Memory pressure warning",
                    "candidate_actions": ["RESTART", "WAIT"],
                    "horizon": 2
                }
            }
        }
        imagine_resp = await self.server.handle_request(imagine_req)
        # No safety evaluator can be injected over MCP, so imagine is fail-closed: a typed
        # SAFETY_INTERLOCKED error whose payload still carries the full telemetry.
        self.assertTrue(imagine_resp["result"]["isError"])
        envelope = json.loads(imagine_resp["result"]["content"][0]["text"])
        self.assertEqual(envelope["error"], "SAFETY_INTERLOCKED")
        self.assertEqual(envelope["result"]["decision_status"], "SAFETY_INTERLOCKED")
        self.assertIn("selected_action", envelope["result"])

        # Stream mode
        stream_req = {
            "jsonrpc": "2.0",
            "id": 108,
            "method": "tools/call",
            "params": {
                "name": "zero",
                "arguments": {
                    "action": "stream",
                    "observation": "telemetry_frame_1",
                    "candidate_actions": ["ACT_1", "ACT_2"]
                }
            }
        }
        stream_resp = await self.server.handle_request(stream_req)
        self.assertFalse(stream_resp["result"]["isError"])
        stream_payload = json.loads(stream_resp["result"]["content"][0]["text"])
        self.assertEqual(stream_payload["step"], 1)
        self.assertTrue(stream_payload["degraded"])
        self.assertEqual(stream_payload["provenance"], "untrained_text_hash_prior")

    async def test_09b_zero_stream_session_id_reuses_engine_across_calls(self):
        """A07: two stream calls sharing session_id must reuse the same engine, so the rolling
        KV-cache and causal-shock history carry over instead of resetting every call."""
        def stream_req(msg_id, session_id):
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "method": "tools/call",
                "params": {
                    "name": "zero",
                    "arguments": {
                        "action": "stream",
                        "observation": [0.1] * 1024,
                        "candidate_actions": ["ACT_1", "ACT_2"],
                        "session_id": "episode-42",
                    }
                }
            }

        resp1 = await self.server.handle_request(stream_req(201, "episode-42"))
        payload1 = json.loads(resp1["result"]["content"][0]["text"])
        self.assertEqual(payload1["step"], 1)
        self.assertEqual(payload1["kv_cache_stats"]["total_frames_processed"], 1)
        self.assertEqual(payload1["causal_shock_norm"], 0.0)  # no prior step to compare against yet

        resp2 = await self.server.handle_request(stream_req(202, "episode-42"))
        payload2 = json.loads(resp2["result"]["content"][0]["text"])
        # step and total_frames_processed only advance past 1 if the SAME engine (and its
        # KV-cache) served both calls; a fresh engine per call would show step=1 again.
        self.assertEqual(payload2["step"], 2)
        self.assertEqual(payload2["kv_cache_stats"]["total_frames_processed"], 2)

        self.assertIn("episode-42", self.server._stream_sessions)

    async def test_09c_zero_stream_without_session_id_gets_fresh_engine_each_call(self):
        """Without session_id, each call is documented as stateless: step resets to 1 every
        time, and nothing is cached server-side."""
        def stream_req(msg_id):
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "method": "tools/call",
                "params": {
                    "name": "zero",
                    "arguments": {
                        "action": "stream",
                        "observation": [0.1] * 1024,
                        "candidate_actions": ["ACT_1", "ACT_2"],
                    }
                }
            }

        resp1 = await self.server.handle_request(stream_req(203))
        resp2 = await self.server.handle_request(stream_req(204))
        payload1 = json.loads(resp1["result"]["content"][0]["text"])
        payload2 = json.loads(resp2["result"]["content"][0]["text"])
        self.assertEqual(payload1["step"], 1)
        self.assertEqual(payload2["step"], 1)
        self.assertEqual(self.server._stream_sessions, {})

    async def test_09d_zero_stream_session_lru_eviction(self):
        """A07: once the session cap is reached, the least-recently-used session is evicted."""
        server = MCPServer(max_stream_sessions=2)

        async def call(session_id):
            return await server.handle_request({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "zero",
                    "arguments": {
                        "action": "stream",
                        "observation": [0.1] * 1024,
                        "candidate_actions": ["ACT_1"],
                        "session_id": session_id,
                    }
                }
            })

        await call("s1")
        await call("s2")
        self.assertEqual(set(server._stream_sessions.keys()), {"s1", "s2"})
        await call("s3")
        # s1 was least-recently-used (touched before s2); s2 and s3 remain.
        self.assertEqual(set(server._stream_sessions.keys()), {"s2", "s3"})

    async def test_09e_zero_stream_non_string_session_id_is_rejected_not_silently_ephemeral(self):
        """Round-2 blocker 1: a session_id of the wrong type must return an explicit validation
        error, never silently fall back to an anonymous ephemeral engine."""
        for bad_session_id in (42, 3.14, [], {}, True, ""):
            resp = await self.server.handle_request({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "zero",
                    "arguments": {
                        "action": "stream",
                        "observation": [0.1] * 1024,
                        "candidate_actions": ["ACT_1"],
                        "session_id": bad_session_id,
                    }
                }
            })
            self.assertTrue(resp["result"]["isError"], msg=repr(bad_session_id))
            self.assertIn("session_id", resp["result"]["content"][0]["text"])
            self.assertEqual(self.server._stream_sessions, {})

    async def test_09f_zero_stream_illegal_request_never_touches_session_state(self):
        """Round-2 blocker 2: an illegal request (bad observation/candidate_actions) must be
        rejected before any session engine is acquired or created, so it can never trigger an
        LRU eviction of an unrelated, legitimate session."""
        server = MCPServer(max_stream_sessions=1)

        async def call(session_id, observation, candidate_actions):
            return await server.handle_request({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "zero",
                    "arguments": {
                        "action": "stream",
                        "observation": observation,
                        "candidate_actions": candidate_actions,
                        "session_id": session_id,
                    }
                }
            })

        good = await call("legit-session", [0.1] * 1024, ["ACT_1"])
        self.assertFalse(good["result"]["isError"])
        self.assertIn("legit-session", server._stream_sessions)

        # An illegal request against a DIFFERENT session_id must not evict "legit-session"
        # from the size-1 cache: if validation ran after session acquisition, it would.
        bad = await call("attacker-session", "not-a-list-and-not-numbers-ok-but-bad-actions", [])
        self.assertTrue(bad["result"]["isError"])
        self.assertIn("legit-session", server._stream_sessions)
        self.assertNotIn("attacker-session", server._stream_sessions)

        # Same, but the illegal part is an oversized observation vector: the dimension guard
        # must run during pre-check too, not only after step_stream() has already reached the
        # engine -- otherwise an oversized-observation flood under fresh session_ids would burn
        # through the LRU cap and evict "legit-session" before ever failing.
        from gen_zero.world_model.streaming_engine import MAX_PROJECTOR_INPUT_DIM
        oversized_bad = await call(
            "attacker-session-2", [0.1] * (MAX_PROJECTOR_INPUT_DIM + 1), ["ACT_1"]
        )
        self.assertTrue(oversized_bad["result"]["isError"])
        self.assertIn("legit-session", server._stream_sessions)
        self.assertNotIn("attacker-session-2", server._stream_sessions)

    async def test_09g_zero_stream_explicit_null_or_blank_session_id_is_rejected(self):
        """An explicit "session_id": null (or whitespace) must be a validation error, never
        treated as an omitted key and served by an anonymous ephemeral engine."""
        server = MCPServer(max_stream_sessions=1)
        good = await server.handle_request({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "zero", "arguments": {
                "action": "stream", "observation": [0.1] * 1024,
                "candidate_actions": ["ACT_1"], "session_id": "legit-session",
            }},
        })
        self.assertFalse(good["result"]["isError"])
        legit_engine = server._stream_sessions["legit-session"]["engine"]

        acquired = []
        original_acquire = server._acquire_stream_engine

        async def spy_acquire(session_id):
            acquired.append(session_id)
            return await original_acquire(session_id)

        server._acquire_stream_engine = spy_acquire
        for bad_session_id in (None, "   ", "\t\n"):
            resp = await server.handle_request({
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "zero", "arguments": {
                    "action": "stream", "observation": [0.1] * 1024,
                    "candidate_actions": ["ACT_1"], "session_id": bad_session_id,
                }},
            })
            self.assertTrue(resp["result"]["isError"], msg=repr(bad_session_id))
            self.assertIn("Invalid session_id", resp["result"]["content"][0]["text"])
        self.assertEqual(acquired, [], "no engine may be acquired for an invalid session_id")
        self.assertEqual(list(server._stream_sessions.keys()), ["legit-session"])
        self.assertEqual(legit_engine.step_count, 1)

        # Direct call path (no MCPServer) must reject explicit null too.
        direct = await execute_zero({
            "action": "stream", "observation": [0.1] * 1024,
            "candidate_actions": ["ACT_1"], "session_id": None,
        })
        self.assertTrue(direct["isError"])
        self.assertIn("Invalid session_id", direct["content"][0]["text"])

    async def test_09h_zero_stream_non_finite_observation_rejected_before_session(self):
        """NaN, +/-Inf and float-overflowing ints in 'observation' must be rejected in the
        pre-check: no engine acquired, no new session, no eviction, no step/KV pollution of
        the existing session with the same session_id."""
        server = MCPServer(max_stream_sessions=1)

        async def call(session_id, observation):
            return await server.handle_request({
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "zero", "arguments": {
                    "action": "stream", "observation": observation,
                    "candidate_actions": ["ACT_1"], "session_id": session_id,
                }},
            })

        good = await call("legit-session", [0.1] * 1024)
        self.assertFalse(good["result"]["isError"])
        legit_engine = server._stream_sessions["legit-session"]["engine"]
        kv_before = legit_engine.kv_cache.get_temporal_context()
        last_latent_before = legit_engine.last_latent

        acquired = []
        original_acquire = server._acquire_stream_engine

        async def spy_acquire(session_id):
            acquired.append(session_id)
            return await original_acquire(session_id)

        server._acquire_stream_engine = spy_acquire
        for bad_value in (float("nan"), float("inf"), float("-inf"), 1e300, -1e300, 10 ** 100, 10 ** 1000, -(10 ** 1000)):
            for sid in ("legit-session", "attacker-session"):
                obs = [0.1] * 1024
                obs[7] = bad_value
                resp = await call(sid, obs)
                label = f"{sid}/{type(bad_value).__name__}"
                self.assertTrue(resp["result"]["isError"], msg=label)
                self.assertIn("Validation Error", resp["result"]["content"][0]["text"], msg=label)
                self.assertIn("finite", resp["result"]["content"][0]["text"], msg=label)
        self.assertEqual(acquired, [])
        self.assertEqual(list(server._stream_sessions.keys()), ["legit-session"])
        self.assertEqual(legit_engine.step_count, 1)
        self.assertEqual(legit_engine.kv_cache.get_temporal_context(), kv_before)
        self.assertIs(legit_engine.last_latent, last_latent_before)

        # Same values via a real JSON-RPC wire decode: Python's json accepts NaN/Infinity.
        import json as _json
        wire = _json.loads('{"o": [NaN, Infinity, -Infinity]}')["o"]
        for bad in wire:
            resp = await call("legit-session", [bad] + [0.1] * 1023)
            self.assertTrue(resp["result"]["isError"])
        self.assertEqual(legit_engine.step_count, 1)

        # The session still works afterwards and continues at step 2.
        after = await call("legit-session", [0.2] * 1024)
        self.assertFalse(after["result"]["isError"])
        self.assertEqual(legit_engine.step_count, 2)

    async def test_10_zero_ambiguous_and_invalid_action(self):
        """Verify unknown actions and ambiguous arguments return descriptive error guidance."""
        # Unknown action
        res_unknown = await execute_zero({"action": "fly_to_moon"})
        self.assertTrue(res_unknown["isError"])
        self.assertIn("Unknown action 'fly_to_moon'", res_unknown["content"][0]["text"])

        # Ambiguous invocation without any distinguishing arguments
        res_ambig = await execute_zero({})
        self.assertTrue(res_ambig["isError"])
        self.assertIn("Ambiguous invocation of 'zero'", res_ambig["content"][0]["text"])

        # Non-dict arguments
        res_nondict = await execute_zero("invalid")
        self.assertTrue(res_nondict["isError"])
        self.assertIn("must be a JSON dictionary", res_nondict["content"][0]["text"])

    async def test_11_zero_compact_messages_and_text(self):
        """Verify 'zero' tool polymorphic execution in compact mode for messages and text."""
        # 1. Message list compaction (explicit action)
        messages = [
            {"role": "user", "content": "Deploy the application and run checks"},
            {"role": "tool", "name": "pwd", "content": "/workspace"},
            {"role": "tool", "name": "pytest", "content": "\n".join([f"line {i}" for i in range(50)])},
            {"role": "tool", "name": "compiler", "content": "Error: compilation failed on line 42"},
            {"role": "assistant", "content": "Analyzing compiler failure"}
        ]
        req_msgs = {
            "jsonrpc": "2.0",
            "id": 109,
            "method": "tools/call",
            "params": {
                "name": "zero",
                "arguments": {
                    "action": "compact",
                    "messages": messages
                }
            }
        }
        resp_msgs = await self.server.handle_request(req_msgs)
        self.assertFalse(resp_msgs["result"]["isError"])
        payload_msgs = json.loads(resp_msgs["result"]["content"][0]["text"])
        summary = payload_msgs["summary"]
        self.assertEqual(summary["dropped_count"], 1)  # 'pwd' probe dropped
        self.assertEqual(summary["truncated_count"], 1)  # 50 lines truncated
        self.assertIsNone(summary["fact_mutation_rate"])
        self.assertGreater(summary["compression_ratio"], 0.0)

        # 2. Text log compaction (auto-detected from 'text' argument)
        long_log = "\n".join([f"log event {i}: memory usage OK" for i in range(40)])
        req_text = {
            "jsonrpc": "2.0",
            "id": 110,
            "method": "tools/call",
            "params": {
                "name": "zero",
                "arguments": {
                    "text": long_log,
                    "head_lines": 3,
                    "tail_lines": 3
                }
            }
        }
        resp_text = await self.server.handle_request(req_text)
        self.assertFalse(resp_text["result"]["isError"])
        payload_text = json.loads(resp_text["result"]["content"][0]["text"])
        self.assertIn("truncated", payload_text["compacted_text"])
        self.assertEqual(payload_text["fact_mutation_rate"], 0.0)
        self.assertGreater(payload_text["tokens_saved"], 0)


if __name__ == "__main__":
    unittest.main()
