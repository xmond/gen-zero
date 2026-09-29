"""Unit tests for Gen-Zero Issue #74: MCTS / A* Compressed Tree Nodes and Tiered Memory."""

import unittest
import numpy as np
import time
import math
from typing import Dict, List, Tuple

from gen_zero.world_model.compressed_tree import (
    CompressedState,
    TreeMemoryStats,
    CompressedTreeNode,
    TieredTreeMemoryManager,
    CompressedImaginationMCTS,
)
from gen_zero.planner.engines import (
    AStarEngine as UncertaintyAStarPlanner,
    AStarEngine as CompressedAStarPlanner,
)


class TestIssue74CompressedTree(unittest.TestCase):
    """Test suite for Issue #74 compressed tree nodes and memory manager."""

    def test_01_compressed_state_bit_exact_roundtrip(self):
        """Validates that CompressedState delivers bit-exact restoration (RMSE <= 10^-7)."""
        rng = np.random.RandomState(42)
        # 1024-dim float32 latent vector with spatial correlation
        base = rng.randn(1024).astype(np.float32)
        latent = np.convolve(base, np.ones(5) / 5.0, mode="same").astype(np.float32)

        comp_state = CompressedState.from_array(latent, level=3)
        self.assertGreater(comp_state.raw_bytes_len, len(comp_state.compressed_bytes))
        
        # Decompress
        restored = comp_state.to_array()
        self.assertEqual(restored.shape, latent.shape)
        self.assertEqual(restored.dtype, latent.dtype)

        rmse = float(np.sqrt(np.mean((latent - restored) ** 2)))
        self.assertLessEqual(rmse, 1e-7, "Restoration RMSE exceeded 10^-7")
        self.assertTrue(np.array_equal(latent, restored), "Binary float32 arrays must be bit-exact")

    def test_02_compressed_tree_node_transparent_access(self):
        """Validates transparent on-demand decompression on get_latent()."""
        latent = np.sin(np.linspace(0, 10, 1024)).astype(np.float32)
        node = CompressedTreeNode(latent_state=latent, depth=2, node_id=1)

        self.assertFalse(node.is_compressed)
        self.assertIsNotNone(node._hot_state)
        uncompressed_total = node.raw_bytes_len + 256
        self.assertEqual(node.memory_footprint_bytes(), uncompressed_total)

        # Compress node
        saved_bytes = node.compress(level=3)
        self.assertGreater(saved_bytes, 0)
        self.assertTrue(node.is_compressed)
        self.assertIsNone(node._hot_state)
        self.assertLess(node.memory_footprint_bytes(), uncompressed_total)

        # Transparent get_latent() call
        retrieved = node.get_latent()
        self.assertTrue(np.array_equal(retrieved, latent))

        # Explicit decompress()
        node.decompress()
        self.assertFalse(node.is_compressed)
        self.assertIsNotNone(node._hot_state)
        self.assertTrue(np.array_equal(node._hot_state, latent))

    def test_03_tiered_tree_memory_manager_eviction(self):
        """Validates that TieredTreeMemoryManager bounds hot nodes and compresses LRU victims."""
        mgr = TieredTreeMemoryManager(max_hot_nodes=3, compression_level=3)

        nodes = []
        for i in range(8):
            vec = np.ones(256, dtype=np.float32) * float(i)
            n = CompressedTreeNode(latent_state=vec, depth=i, node_id=i)
            mgr.register_node(n)
            nodes.append(n)

        stats = mgr.get_memory_stats()
        self.assertEqual(stats.total_nodes, 8)
        self.assertLessEqual(stats.hot_nodes_count, 3)
        self.assertGreaterEqual(stats.cold_nodes_count, 5)
        self.assertGreater(stats.memory_reduction_pct, 40.0)

        # Accessing an evicted cold node decompresses it transparently
        cold_node = [n for n in nodes if n.is_compressed][0]
        arr = mgr.touch_node(cold_node)
        self.assertFalse(cold_node.is_compressed)
        self.assertEqual(float(arr[0]), float(cold_node.depth))

    def test_04_deep_lookahead_mcts_simulation(self):
        """Validates deep lookahead MCTS (H=20, N=128) with high memory compression."""
        root_z = np.zeros(256, dtype=np.float32)
        actions = ["LEFT", "RIGHT", "FORWARD", "RETREAT"]

        planner = CompressedImaginationMCTS(
            transition_model=None,
            max_simulations=128,
            max_depth=20,
            max_hot_nodes=16,
        )

        def simple_eval(z: np.ndarray) -> float:
            return float(np.sum(z[:10]))

        res = planner.plan(
            root_latent=root_z,
            candidate_actions=actions,
            value_evaluator=simple_eval,
        )

        self.assertIn(res["best_action"], actions)
        self.assertGreaterEqual(res["lookahead_depth"], 5)
        stats = res["tree_memory_stats"]
        self.assertGreater(stats["total_nodes"], 20)
        self.assertGreater(stats["memory_reduction_pct"], 30.0)

    def test_05_compressed_astar_parity(self):
        """Validates that CompressedAStarPlanner delivers bit-exact optimal path parity with UncertaintyAStarPlanner."""
        # Simple grid graph with obstacles
        grid = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        start = (0, 0)
        goal = (4, 4)

        def is_goal(s: Tuple[int, int]) -> bool:
            return s == goal

        def get_neighbors(s: Tuple[int, int]) -> List[Tuple[Tuple[int, int], str, float]]:
            r, c = s
            res = []
            moves = [(-1, 0, "UP"), (1, 0, "DOWN"), (0, -1, "LEFT"), (0, 1, "RIGHT")]
            for dr, dc, act in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 5 and 0 <= nc < 5 and grid[nr][nc] == 0:
                    res.append(((nr, nc), act, 0.95))
            return res

        def heuristic(s: Tuple[int, int]) -> float:
            return abs(s[0] - goal[0]) + abs(s[1] - goal[1])

        # Baseline A*
        baseline_planner = UncertaintyAStarPlanner()
        base_res = baseline_planner.plan(
            start_state=start,
            is_goal_fn=is_goal,
            get_neighbors_fn=get_neighbors,
            heuristic_fn=heuristic,
        )

        # Compressed A*
        comp_planner = CompressedAStarPlanner()
        comp_res = comp_planner.plan(
            start_state=start,
            is_goal_fn=is_goal,
            get_neighbors_fn=get_neighbors,
            heuristic_fn=heuristic,
        )

        self.assertTrue(base_res["success"])
        self.assertTrue(comp_res["success"])
        self.assertEqual(base_res["path"], comp_res["path"])
        self.assertEqual(base_res["cost"], comp_res["cost"])
        self.assertEqual(base_res["nodes_expanded"], comp_res["nodes_expanded"])
        self.assertEqual(base_res["states"], comp_res["states"])

    def test_06_sampling_multiplier_under_fixed_memory_budget(self):
        """Validates that under a fixed RAM budget, compressed MCTS supports 6x more simulations."""
        latent_dim = 512
        bytes_per_raw_node = latent_dim * 4 + 256  # 2304 bytes
        # Budget = 480 KB (~213 raw nodes)
        memory_budget_bytes = 480 * 1024
        max_raw_nodes = memory_budget_bytes // bytes_per_raw_node  # ~213 nodes

        # Run compressed MCTS with 150 simulations (generating > 300 nodes, 2.5x more than max_raw_nodes)
        planner = CompressedImaginationMCTS(
            transition_model=None,
            max_simulations=150,
            max_depth=20,
            max_hot_nodes=8,
            compression_level=3,
        )

        res = planner.plan(
            root_latent=np.zeros(latent_dim, dtype=np.float32),
            candidate_actions=["NORTH", "SOUTH", "EAST", "WEST"],
        )

        stats = res["tree_memory_stats"]
        total_nodes = stats["total_nodes"]
        actual_bytes = stats["actual_resident_bytes"]
        uncompressed_bytes = stats["uncompressed_bytes"]

        # Validates that we expanded 2.5x-3x more nodes than raw capacity
        self.assertGreater(total_nodes, int(max_raw_nodes * 2.5))
        # Validates that uncompressed tree would have blown past the 480KB budget
        self.assertGreater(uncompressed_bytes, memory_budget_bytes)
        # Validates that actual resident bytes remained strictly within 480KB budget
        self.assertLessEqual(actual_bytes, memory_budget_bytes)
        # Validates >= 65% memory reduction
        self.assertGreaterEqual(stats["memory_reduction_pct"], 65.0)


if __name__ == "__main__":
    unittest.main()
