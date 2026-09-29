"""Tests for the Triton stream-compaction kernel and its CPU fallback.

These tests run on a CPU-only box (no triton, CPU-only torch is fine); the
triton-specific path is guarded by ``unittest.skipUnless(HAS_TRITON, ...)`` so
that it only executes where triton + CUDA are actually present. The CPU path
is always exercised and is the one used by the planner wiring test.
"""

from __future__ import annotations

import unittest

import numpy as np

from gen_zero.kernels import (
    compact_active,
    compact_active_mask,
    available_backends,
    HAS_TRITON,
)


def _random_mask(n: int, rng: np.random.Generator, p: float = 0.5) -> np.ndarray:
    return rng.random(n) < p


class TestCompactActive(unittest.TestCase):
    """Core correctness of :func:`compact_active` over many sizes/patterns."""

    def test_various_sizes_random(self):
        rng = np.random.default_rng(0)
        sizes = [1, 2, 7, 1000, 1 << 14]
        for n in sizes:
            for p in (0.0, 0.1, 0.5, 0.9, 1.0):
                with self.subTest(n=n, p=p):
                    mask = rng.random(n) < p
                    idx, count = compact_active(mask)
                    self.assertEqual(idx.dtype, np.int64)
                    np.testing.assert_array_equal(idx, np.flatnonzero(mask))
                    self.assertEqual(count, len(idx))
                    self.assertEqual(count, int(mask.sum()))

    def test_all_active(self):
        for n in (1, 5, 100):
            with self.subTest(n=n):
                mask = np.ones(n, dtype=bool)
                idx, count = compact_active(mask)
                np.testing.assert_array_equal(idx, np.arange(n))
                self.assertEqual(count, n)

    def test_all_inactive(self):
        for n in (1, 5, 100):
            with self.subTest(n=n):
                mask = np.zeros(n, dtype=bool)
                idx, count = compact_active(mask)
                self.assertEqual(idx.shape, (0,))
                self.assertEqual(count, 0)

    def test_alternating(self):
        for n in (1, 2, 7, 100):
            with self.subTest(n=n):
                mask = np.fromiter(
                    ((i % 2 == 0) for i in range(n)), dtype=bool, count=n
                )
                idx, count = compact_active(mask)
                np.testing.assert_array_equal(idx, np.arange(0, n, 2))
                self.assertEqual(count, len(np.arange(0, n, 2)))

    def test_single_active_start_middle_end(self):
        for n in (1, 5, 10):
            for pos in (0, n // 2, n - 1):
                with self.subTest(n=n, pos=pos):
                    mask = np.zeros(n, dtype=bool)
                    mask[pos] = True
                    idx, count = compact_active(mask)
                    np.testing.assert_array_equal(idx, np.array([pos], dtype=np.int64))
                    self.assertEqual(count, 1)

    def test_n_one_both_values(self):
        for val in (True, False):
            with self.subTest(val=val):
                mask = np.array([val], dtype=bool)
                idx, count = compact_active(mask)
                expected = np.flatnonzero(mask)
                np.testing.assert_array_equal(idx, expected)
                self.assertEqual(count, int(expected.size))

    def test_batch_capacity_shape(self):
        rng = np.random.default_rng(42)
        B, capacity = 4, 8
        N = B * capacity
        mask = rng.random(N) < 0.5
        idx, count = compact_active(mask)
        np.testing.assert_array_equal(idx, np.flatnonzero(mask))
        self.assertEqual(count, int(mask.sum()))


class TestIdempotency(unittest.TestCase):
    """Compacting an already-compact set is stable."""

    def test_all_true_compacts_to_arange(self):
        n = 1000
        mask = np.ones(n, dtype=bool)
        idx, count = compact_active(mask)
        # Re-compacting the indices as a mask (all True over [0, count)) yields
        # np.arange(count).
        re_idx, re_count = compact_active(np.ones(count, dtype=bool))
        np.testing.assert_array_equal(re_idx, np.arange(count))
        self.assertEqual(re_count, count)

    def test_recompact_output_indices(self):
        rng = np.random.default_rng(1)
        mask = rng.random(2000) < 0.3
        idx, count = compact_active(mask)
        # The indices themselves are a strictly increasing int64 array; building
        # a mask back from them and re-compacting must reproduce them exactly.
        back = np.zeros(int(idx.max()) + 1 if count > 0 else 0, dtype=bool)
        if count > 0:
            back[idx] = True
        idx2, count2 = compact_active(back)
        np.testing.assert_array_equal(idx2, idx)
        self.assertEqual(count2, count)


class TestBackendSelection(unittest.TestCase):
    """Backend availability and explicit backend selection."""

    def test_available_backends_has_cpu(self):
        backends = available_backends()
        self.assertIn("cpu", backends)
        self.assertEqual(backends >= {"cpu"}, True)

    def test_triton_in_backends_iff_has_triton(self):
        backends = available_backends()
        self.assertEqual("triton" in backends, HAS_TRITON)

    def test_cpu_backend_numpy(self):
        rng = np.random.default_rng(3)
        mask = rng.random(500) < 0.5
        idx, count = compact_active(mask, backend="cpu")
        np.testing.assert_array_equal(idx, np.flatnonzero(mask))
        self.assertEqual(count, int(mask.sum()))

    def test_cpu_backend_torch_if_available(self):
        try:
            import torch
        except Exception:
            self.skipTest("torch not available")
        rng = np.random.default_rng(4)
        mask_np = rng.random(500) < 0.5
        mask_t = torch.as_tensor(mask_np)
        idx, count = compact_active(mask_t, backend="cpu")
        self.assertIsInstance(idx, np.ndarray)
        np.testing.assert_array_equal(idx, np.flatnonzero(mask_np))
        self.assertEqual(count, int(mask_np.sum()))

    def test_triton_backend_raises_when_unavailable(self):
        if HAS_TRITON:
            self.skipTest("triton is available; skip the unavailability test")
        rng = np.random.default_rng(5)
        mask = rng.random(100) < 0.5
        with self.assertRaises(ValueError):
            compact_active(mask, backend="triton")

    def test_unknown_backend_raises(self):
        mask = np.ones(4, dtype=bool)
        with self.assertRaises(ValueError):
            compact_active(mask, backend="cuda")  # noqa: arbitrary value

    def test_compact_active_mask_alias(self):
        rng = np.random.default_rng(6)
        mask = rng.random(300) < 0.5
        only = compact_active_mask(mask, backend="cpu")
        idx, _ = compact_active(mask, backend="cpu")
        np.testing.assert_array_equal(only, idx)


@unittest.skipUnless(HAS_TRITON, "triton not available")
class TestTritonPath(unittest.TestCase):
    """Exercises the genuine ``@triton.jit`` path. Skipped without triton+CUDA."""

    def test_triton_path_matches_cpu(self):
        import torch
        rng = np.random.default_rng(7)
        for n in (1, 2, 7, 1024, 1025, 1 << 14):
            with self.subTest(n=n):
                mask_np = rng.random(n) < 0.5
                mask_t = torch.as_tensor(mask_np, device="cuda")
                idx_t, count = compact_active(mask_t, backend="triton")
                self.assertIsInstance(idx_t, torch.Tensor)
                self.assertEqual(idx_t.dtype, torch.int64)
                got = idx_t.cpu().numpy()
                np.testing.assert_array_equal(got, np.flatnonzero(mask_np))
                self.assertEqual(count, int(mask_np.sum()))


class TestPlannerWiring(unittest.TestCase):
    """``use_compact=True`` must be behavior-preserving vs the default."""

    def test_use_compact_identical_to_default(self):
        from gen_zero.causal.vectorized_latent_mcts import BatchLatentMctsPlanner

        rng = np.random.default_rng(123)
        B, dim, k = 8, 16, 6
        emb = rng.normal(size=(k, dim))
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        states = rng.normal(size=(B, dim))
        candidates = [f"c{i}" for i in range(k)]

        planner_default = BatchLatentMctsPlanner(
            num_simulations=20, max_depth=4, use_compact=False
        )
        planner_compact = BatchLatentMctsPlanner(
            num_simulations=20, max_depth=4, use_compact=True
        )

        best_d, policy_d, info_d = planner_default.plan(states, candidates, emb)
        best_c, policy_c, info_c = planner_compact.plan(states, candidates, emb)

        np.testing.assert_array_equal(best_c, best_d)
        np.testing.assert_array_equal(policy_c, policy_d)
        np.testing.assert_array_equal(info_c["nodes"], info_d["nodes"])

    def test_use_compact_helper(self):
        from gen_zero.causal.vectorized_latent_mcts import BatchLatentMctsPlanner

        planner = BatchLatentMctsPlanner(num_simulations=5, max_depth=2,
                                         use_compact=True)
        mask = np.array([False, True, True, False, True], dtype=bool)
        idx = planner._compact_indices(mask)
        np.testing.assert_array_equal(idx, np.flatnonzero(mask))

        planner_off = BatchLatentMctsPlanner(num_simulations=5, max_depth=2,
                                            use_compact=False)
        idx_off = planner_off._compact_indices(mask)
        np.testing.assert_array_equal(idx_off, np.flatnonzero(mask))


if __name__ == "__main__":
    unittest.main()
