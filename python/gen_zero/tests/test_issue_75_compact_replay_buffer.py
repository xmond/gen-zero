"""Unit tests for Issue #75: Compact Causal Replay Buffer and Golden Snapshot Rollback."""

import os
import shutil
import tempfile
import unittest
import numpy as np

from gen_zero.train.compact_replay_buffer import (
    CausalTransition,
    CompressedChunk,
    BufferMemoryStats,
    CompactCausalReplayBuffer,
    GoldenSnapshotRecord,
    GoldenSnapshotManager,
)
from gen_zero.client import GenZeroClient


class TestCompactCausalReplayBuffer(unittest.TestCase):
    """Verifies chunked zstd causal replay buffer compression, sampling, and fidelity."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.latent_dim = 128
        self.buffer = CompactCausalReplayBuffer(
            capacity=200,
            chunk_size=16,
            latent_dim=self.latent_dim,
            compression_level=3,
            max_cached_chunks=4,
            causal_weight=0.5,
        )

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_chunk_compression_and_exact_decompression(self):
        """Verify that compressed chunks decompress to bit-exact identical arrays (RMSE = 0.0)."""
        rng = np.random.RandomState(42)
        states = []
        next_states = []

        for i in range(16):
            s = np.zeros(self.latent_dim, dtype=np.float32)
            s[:24] = rng.randn(24).astype(np.float32)
            ns = s.copy()
            ns[:24] += 0.05 * rng.randn(24).astype(np.float32)
            states.append(s)
            next_states.append(ns)
            self.buffer.push(
                state=s,
                action=f"ACT_{i % 4}",
                reward=float(i * 0.1),
                next_state=ns,
                done=(i == 15),
                causal_shock=float(np.linalg.norm(ns - s)),
                td_error=1.5,
            )

        # 16 transitions should have triggered exactly 1 chunk compression
        self.assertEqual(len(self.buffer._chunks), 1)
        chunk = self.buffer._chunks[0]
        self.assertEqual(chunk.num_transitions, 16)
        self.assertGreater(chunk.compression_ratio_pct, 50.0)

        # Retrieve and verify bit-exact fidelity
        decomp_dict = self.buffer._get_chunk_arrays(chunk)
        decomp_states = decomp_dict["states"]
        decomp_next = decomp_dict["next_states"]
        for i in range(16):
            np.testing.assert_array_equal(decomp_states[i], states[i])
            np.testing.assert_array_equal(decomp_next[i], next_states[i])

    def test_chunk_cache_consistency_under_eviction(self):
        """Verify that LRU chunk cache never serves stale vectors when capacity eviction shifts chunk list."""
        # Create a small buffer: capacity 64, chunk_size 16 -> max 4 chunks
        buf = CompactCausalReplayBuffer(
            capacity=64,
            chunk_size=16,
            latent_dim=self.latent_dim,
            max_cached_chunks=8,
        )

        # Push 4 chunks (64 transitions) with tagged actions and matching states
        for i in range(64):
            tag = float(i)
            s = np.zeros(self.latent_dim, dtype=np.float32)
            s[0] = tag
            ns = s.copy()
            ns[0] = tag + 0.5
            buf.add(
                state=s,
                action=f"ACT_{int(tag)}",
                reward=tag,
                next_state=ns,
                done=False,
            )

        # Warm up cache across all chunks
        batch1 = buf.sample(batch_size=32, prioritized=False)
        for s, a, r in zip(batch1["states"], batch1["actions"], batch1["rewards"]):
            expected_tag = float(a.split("_")[1])
            self.assertEqual(s[0], expected_tag)
            self.assertEqual(r, expected_tag)

        # Now push 4 MORE chunks (64 more transitions, tags 64..127) to trigger complete eviction of chunks 0..3
        for i in range(64, 128):
            tag = float(i)
            s = np.zeros(self.latent_dim, dtype=np.float32)
            s[0] = tag
            ns = s.copy()
            ns[0] = tag + 0.5
            buf.add(
                state=s,
                action=f"ACT_{int(tag)}",
                reward=tag,
                next_state=ns,
                done=False,
            )

        # Chunks in buffer should now only have tags >= 64
        self.assertLessEqual(buf.total_transitions, 64)

        # Sample and verify absolute alignment between state vector s[0] and action tag
        batch2 = buf.sample(batch_size=32, prioritized=False)
        for s, a, r in zip(batch2["states"], batch2["actions"], batch2["rewards"]):
            expected_tag = float(a.split("_")[1])
            self.assertGreaterEqual(expected_tag, 64.0)
            self.assertEqual(s[0], expected_tag, f"Desync detected: state has tag {s[0]}, but action is {a}!")
            self.assertEqual(r, expected_tag)

    def test_buffer_push_and_capacity_eviction(self):
        """Verify capacity wrapping and eviction when exceeding capacity."""
        rng = np.random.RandomState(101)
        # Push 240 transitions (capacity is 200, chunk size 16)
        for i in range(240):
            s = np.zeros(self.latent_dim, dtype=np.float32)
            s[:16] = rng.randn(16).astype(np.float32)
            ns = s.copy()
            ns[:16] += 0.01
            self.buffer.push(
                state=s,
                action="STEP",
                reward=1.0,
                next_state=ns,
                done=False,
            )

        # Total transitions should be capped at capacity
        self.assertLessEqual(self.buffer.total_transitions, 200)
        # Number of chunks should be bounded
        max_chunks = 200 // 16
        self.assertLessEqual(len(self.buffer._chunks), max_chunks)

        stats = self.buffer.get_memory_stats()
        self.assertGreater(stats.total_transitions, 0)
        self.assertGreater(stats.avg_chunk_compression_ratio, 50.0)

    def test_prioritized_causal_sampling(self):
        """Verify prioritized causal sampling and uniform sampling batches."""
        rng = np.random.RandomState(202)
        for i in range(48):
            s = rng.randn(self.latent_dim).astype(np.float32)
            ns = s + 0.1
            shock = 10.0 if i == 25 else 0.01
            td = 50.0 if i == 25 else 0.1
            self.buffer.push(
                state=s,
                action=f"ACT_{i}",
                reward=float(i),
                next_state=ns,
                done=False,
                causal_shock=shock,
                td_error=td,
            )

        # Sample uniform
        batch_uniform = self.buffer.sample(batch_size=16, prioritized=False)
        self.assertEqual(batch_uniform["states"].shape, (16, self.latent_dim))
        self.assertEqual(batch_uniform["next_states"].shape, (16, self.latent_dim))
        self.assertEqual(len(batch_uniform["actions"]), 16)
        self.assertEqual(len(batch_uniform["rewards"]), 16)

        # Sample prioritized (high shock transition should be sampled frequently)
        sampled_actions = []
        for _ in range(20):
            batch_prio = self.buffer.sample(batch_size=8, prioritized=True)
            sampled_actions.extend(batch_prio["actions"])

        # ACT_25 should appear in the prioritized draws
        self.assertIn("ACT_25", sampled_actions)

    def test_golden_snapshot_rollback_and_bit_exact(self):
        """Verify atomic snapshot creation, sub-16ms rollback, and bit-exact recovery."""
        mgr = GoldenSnapshotManager(storage_dir=self.temp_dir, max_snapshots=3)
        rng = np.random.RandomState(303)

        # Create original model weights
        weights_orig = {
            "w_choice": rng.randn(64, 128).astype(np.float32),
            "b_choice": rng.randn(64).astype(np.float32),
            "w_sat": rng.randn(32, 128).astype(np.float32),
        }

        # Snapshot
        record = mgr.create_snapshot("golden_v1", weights_orig, model_name="safety_core")
        self.assertTrue(os.path.isfile(record.file_path))
        self.assertGreater(record.uncompressed_bytes, record.file_bytes)

        # Corrupt weights to simulate policy drift
        weights_corrupted = {
            "w_choice": weights_orig["w_choice"] + 99.0,
            "b_choice": weights_orig["b_choice"] - 50.0,
            "w_sat": np.zeros_like(weights_orig["w_sat"]),
        }
        self.assertFalse(np.allclose(weights_corrupted["w_choice"], weights_orig["w_choice"]))

        # Execute instantaneous rollback
        restored_weights, rollback_dur_ms, restored_record = mgr.rollback("golden_v1")

        # Verify rollback speed: strictly sub-16ms
        self.assertLess(rollback_dur_ms, 16.0)

        # Verify bit-exact restoration (RMSE = 0.0)
        for key in weights_orig:
            diff = np.max(np.abs(restored_weights[key] - weights_orig[key]))
            self.assertEqual(diff, 0.0)

        self.assertEqual(restored_record.snapshot_id, "golden_v1")
        self.assertEqual(restored_record.weights_checksum, record.weights_checksum)

    def test_client_integration(self):
        """Verify GenZeroClient convenience factory methods for Issue #75."""
        client = GenZeroClient()
        buffer = client.create_compact_replay_buffer(
            capacity=1000,
            chunk_size=32,
            latent_dim=64,
        )
        self.assertIsInstance(buffer, CompactCausalReplayBuffer)
        self.assertEqual(buffer.capacity, 1000)

        snapshot_mgr = client.create_golden_snapshot_manager(
            storage_dir=self.temp_dir,
            max_snapshots=4,
        )
        self.assertIsInstance(snapshot_mgr, GoldenSnapshotManager)
        self.assertEqual(snapshot_mgr.max_snapshots, 4)


if __name__ == "__main__":
    unittest.main()
