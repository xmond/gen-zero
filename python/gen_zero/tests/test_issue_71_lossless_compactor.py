"""Unit tests for Issue #71: Lossless Invertible Compactor & MCP Context Retrospective Decompression.

Verifies:
1. VerbatimContextCompactor truncation creates lossless zstd and InvertibleVectorEncoder fingerprints.
2. restore_truncated_text restores 100% bit-exact original text (0.0% fact mutation rate).
3. compact_session and restore_session preserve non-truncated messages and perfectly reconstitute truncated turns.
4. MCP execute_zero_compact works bidirectionally (compact and restore actions).
5. Polymorphic execute_zero dispatches to restore seamlessly.
6. Checksum validation detects and prevents corrupted payload restoration.
"""

import asyncio
import base64
import hashlib
import json
import unittest

from gen_zero.compactor.verbatim_compactor import (
    CompactorAction,
    CompactionSummary,
    MessageCompactionItem,
    VerbatimContextCompactor,
)
from gen_zero.mcp.server import execute_zero, execute_zero_compact


class TestIssue71LosslessCompactor(unittest.TestCase):
    """Test suite for Issue #71 lossless compactor integration."""

    def setUp(self):
        self.compactor = VerbatimContextCompactor(
            head_lines=5,
            tail_lines=5,
            truncate_line_threshold=15,
            enable_lossless_fingerprint=True,
        )

    def test_01_truncate_text_attaches_lossless_fingerprint(self):
        lines = [f"Log record {i:04d}: kernel subsystem operation OK" for i in range(100)]
        content = "\n".join(lines)

        truncated = self.compactor.truncate_text(content)
        self.assertIn("[... truncated 90 lines verbatim ...]", truncated)
        self.assertIn("[lossless_fingerprint:", truncated)

        # Check registry
        self.assertEqual(len(self.compactor.lossless_registry), 1)
        fp = list(self.compactor.lossless_registry.keys())[0]
        entry = self.compactor.lossless_registry[fp]
        self.assertEqual(entry["fingerprint"], fp)
        self.assertEqual(entry["truncated_lines"], 90)
        self.assertGreater(entry["orig_bytes"], 0)
        self.assertGreater(entry["compressed_bytes"], 0)
        self.assertLess(entry["compressed_bytes"], entry["orig_bytes"])

    def test_02_restore_truncated_text_is_100_percent_bit_exact(self):
        lines = [f"Line {i:03d}: variable_alpha_{i} = calculate_metric({i * 17})" for i in range(120)]
        original_content = "\n".join(lines) + "\n"

        truncated = self.compactor.truncate_text(original_content)
        self.assertNotEqual(truncated, original_content)

        restored, stats = self.compactor.restore_truncated_text(truncated)
        self.assertEqual(stats["restored_segments"], 1)
        self.assertTrue(stats["all_verified"])
        self.assertEqual(stats["fact_mutation_rate"], 0.0)
        self.assertEqual(restored, original_content, "Restored text must match original byte-for-byte!")

    def test_03_compact_session_and_restore_session(self):
        log_content = "\n".join([f"Step {i}: Cargo building crate_{i}..." for i in range(80)])
        messages = [
            {"id": "msg-0", "role": "system", "content": "You are an autonomous compiler agent."},
            {"id": "msg-1", "role": "user", "content": "Compile the workspace."},
            {"id": "msg-2", "role": "tool", "name": "pwd", "content": "/workspace/project"},
            {"id": "msg-3", "role": "tool", "name": "cargo", "content": log_content},
            {"id": "msg-4", "role": "assistant", "content": "Build succeeded with 0 errors."},
            {"id": "msg-5", "role": "user", "content": "Confirm binary outputs."},
            {"id": "msg-6", "role": "assistant", "content": "Binary is at target/release/app."},
        ]

        compacted, items, summary = self.compactor.compact_session(messages)
        self.assertEqual(summary.dropped_count, 1)  # pwd dropped
        self.assertEqual(summary.truncated_count, 1)  # cargo log truncated
        self.assertIsNone(summary.fact_mutation_rate)
        self.assertGreater(summary.space_savings_pct, 50.0)
        self.assertGreater(summary.network_latency_speedup, 1.0)
        self.assertEqual(len(summary.lossless_fingerprints), 1)

        # Reconstruct session
        restored_msgs, restore_stats = self.compactor.restore_session(compacted)
        self.assertEqual(restore_stats["restored_messages"], 1)
        self.assertTrue(restore_stats["all_verified"])
        self.assertEqual(restore_stats["fact_mutation_rate"], 0.0)

        # Verify the cargo output in restored_msgs is bit-exact to original
        cargo_msg = [m for m in restored_msgs if m.get("name") == "cargo"][0]
        self.assertEqual(cargo_msg["content"], log_content)

    def test_04_mcp_execute_zero_compact_and_restore(self):
        async def run_mcp_test():
            raw_text = "\n".join([f"Trace {i}: database row index {i*101} scanned" for i in range(150)])

            # 1. Compact
            compact_res = await execute_zero_compact({"text": raw_text, "head_lines": 5, "tail_lines": 5})
            self.assertFalse(compact_res.get("isError"))
            compact_data = json.loads(compact_res["content"][0]["text"])

            self.assertIn("compacted_text", compact_data)
            self.assertIn("lossless_registry", compact_data)
            self.assertIn("comparison", compact_data)
            comp_metrics = compact_data["comparison"]
            self.assertIn("time", comp_metrics)
            self.assertIn("space", comp_metrics)
            self.assertIn("fidelity", comp_metrics)
            self.assertEqual(comp_metrics["fidelity"]["fact_mutation_rate"], 0.0)

            # 2. Restore
            restore_res = await execute_zero_compact({
                "action": "restore",
                "text": compact_data["compacted_text"],
                "lossless_registry": compact_data["lossless_registry"],
            })
            self.assertFalse(restore_res.get("isError"))
            restore_data = json.loads(restore_res["content"][0]["text"])
            self.assertTrue(restore_data["is_bit_exact"])
            self.assertEqual(restore_data["restored_text"], raw_text)

            # 3. Polymorphic router test
            poly_compact = await execute_zero({"action": "compact", "text": raw_text})
            self.assertFalse(poly_compact.get("isError"))
            poly_restore = await execute_zero({
                "action": "restore",
                "text": compact_data["compacted_text"],
                "lossless_registry": compact_data["lossless_registry"],
            })
            self.assertFalse(poly_restore.get("isError"))
            poly_restore_data = json.loads(poly_restore["content"][0]["text"])
            self.assertEqual(poly_restore_data["restored_text"], raw_text)

        asyncio.run(run_mcp_test())

    def test_05_tampered_payload_integrity_rejection(self):
        lines = [f"Security audit record {i}" for i in range(60)]
        content = "\n".join(lines)
        truncated = self.compactor.truncate_text(content)

        # Corrupt the payload in the registry
        fp = list(self.compactor.lossless_registry.keys())[0]
        corrupted_registry = dict(self.compactor.lossless_registry)
        entry = dict(corrupted_registry[fp])
        entry["sha256"] = "deadbeef" * 8  # Alter expected checksum
        corrupted_registry[fp] = entry

        # Attempt restoration with corrupted registry
        restored, stats = self.compactor.restore_truncated_text(truncated, corrupted_registry)
        self.assertFalse(stats["all_verified"])
        self.assertIn(fp, stats["failed_fingerprints"])
        # Should NOT silently substitute corrupted text
        self.assertNotEqual(restored, content)


if __name__ == "__main__":
    unittest.main()
