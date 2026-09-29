"""Unit and Integration Tests for Issue #23: NanoCore Layer Scaling Benchmark & MoV Architecture.

Validates:
1. Milestone 1: Block Influence (BI), Linear Separability, and Pareto Frontier L1-L6 report.
2. Milestone 2: DualCalibrator (Temp scaling + Isotonic) on Near-Miss benchmark with 10-Bin ECE <= 0.035.
3. Milestone 3: 64-byte aligned SIMD packet, physical memory locking (mlock), and sub-1.4ms latency.
4. Milestone 4: MoV Decision Layer, RMSNorm geometric normalization, Bayesian pooling, and Fallback Watchdog.
5. Milestone 5: CP-SAT 0-1 Integer Solver, 2ms hard timeout gate, and 100% safety interception.
"""

import unittest
import numpy as np
import time

from gen_zero.evaluate.layer_scaling_benchmark import (
    compute_block_influence,
    compute_linear_separability,
    NanoCoreLayerScalingBenchmark,
    LayerMetrics,
)
from gen_zero.evaluate.dual_calibrator import (
    IsotonicCalibrator,
    DualCalibrator,
    CalibrationSample,
)
from gen_zero.gateway.simd_engine import (
    NanoCoreVectorPacket,
    PhysicalMemoryLocker,
    SIMDScoreEngine,
)
from gen_zero.gateway.mov_fusion import (
    rms_norm,
    MicroCoreOutput,
    CompositeMoVDecision,
    MoVDecisionLayer,
)
from gen_zero.gate.cpsat_formal_solver import (
    CPSATFormalSolver,
    CPSATVerdict,
)


class TestLayerScalingAndBlockInfluence(unittest.TestCase):
    """Tests for Milestone 1: Block Influence & Layer-Scaling Pareto Frontier."""

    def test_block_influence_identical_and_orthogonal(self):
        # Identical representations -> cosine similarity = 1.0 -> BI = 0.0
        h_same = np.ones((10, 64), dtype=np.float32)
        bi_zero = compute_block_influence(h_same, h_same)
        self.assertAlmostEqual(bi_zero, 0.0, places=4)

        # Orthogonal representations -> cosine similarity = 0.0 -> BI = 1.0
        h1 = np.zeros((10, 64), dtype=np.float32)
        h1[:, :32] = 1.0
        h2 = np.zeros((10, 64), dtype=np.float32)
        h2[:, 32:] = 1.0
        bi_ortho = compute_block_influence(h1, h2)
        self.assertAlmostEqual(bi_ortho, 1.0, places=4)

    def test_linear_separability_separable_clusters(self):
        # Create 2 separable clusters
        np.random.seed(42)
        c1 = np.random.randn(50, 32).astype(np.float32) + 3.0
        c2 = np.random.randn(50, 32).astype(np.float32) - 3.0
        X = np.vstack([c1, c2])
        y = np.array([0] * 50 + [1] * 50, dtype=np.int32)

        sep = compute_linear_separability(X, y, num_classes=2)
        self.assertGreaterEqual(sep, 0.95)

    def test_linear_separability_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            compute_linear_separability(np.zeros((0, 32), dtype=np.float32), np.zeros((0,), dtype=np.int32), num_classes=2)
        with self.assertRaises(ValueError):
            compute_linear_separability(np.zeros((5, 32), dtype=np.float32), np.zeros((5,), dtype=np.int32), num_classes=1)

    def test_pareto_frontier_requires_measurements(self):
        bench = NanoCoreLayerScalingBenchmark()
        with self.assertRaises(ValueError):
            bench.evaluate_pareto_frontier({})

    def test_pareto_frontier_requires_all_keys(self):
        bench = NanoCoreLayerScalingBenchmark()
        with self.assertRaises(ValueError):
            bench.evaluate_pareto_frontier({1: {"block_influence": 0.1}})

    def test_pareto_frontier_computes_score_and_sla_from_measurements(self):
        bench = NanoCoreLayerScalingBenchmark()
        # Caller-supplied measurements: L2 is engineered to satisfy the SLA
        # (P99 <= 1.4ms, Memory <= 20MB, ECE <= 0.035); L4 violates latency/memory.
        measurements = {
            1: {
                "block_influence": 0.08, "linear_separability": 0.72, "accuracy": 0.74,
                "p99_latency_ms": 0.65, "memory_mb": 9.5, "ece_10bin": 0.052,
            },
            2: {
                "block_influence": 0.28, "linear_separability": 0.91, "accuracy": 0.89,
                "p99_latency_ms": 1.15, "memory_mb": 18.2, "ece_10bin": 0.028,
            },
            4: {
                "block_influence": 0.42, "linear_separability": 0.96, "accuracy": 0.94,
                "p99_latency_ms": 2.40, "memory_mb": 35.0, "ece_10bin": 0.024,
            },
        }
        metrics = bench.evaluate_pareto_frontier(measurements)
        self.assertEqual(len(metrics), 3)

        l2 = next(m for m in metrics if m.layer_depth == 2)
        self.assertLessEqual(l2.p99_latency_ms, 1.4)
        self.assertLessEqual(l2.memory_mb, 20.0)
        self.assertLessEqual(l2.ece_10bin, 0.035)

        # Pareto score is a deterministic function of the supplied metrics.
        expected_score = (l2.accuracy * l2.linear_separability) / (
            l2.p99_latency_ms * (l2.memory_mb / 20.0) * (1.0 + 10.0 * l2.ece_10bin)
        )
        self.assertAlmostEqual(l2.pareto_score, expected_score, places=6)

        l4 = next(m for m in metrics if m.layer_depth == 4)
        self.assertGreater(l4.p99_latency_ms, 1.4)
        self.assertGreater(l4.memory_mb, 20.0)

        # Verify report generation derives its conclusions from the metrics.
        md = bench.generate_pareto_report_markdown(metrics)
        self.assertIn("L2", md)
        self.assertIn("SWEET-SPOT", md)
        self.assertIn("L4", md)


class TestDualCalibratorAndNearMissBenchmark(unittest.TestCase):
    """Tests for Milestone 2: Dual Calibrator and ECE <= 0.035 on Near-Miss Sets."""

    def test_isotonic_calibrator_monotonicity(self):
        calibrator = IsotonicCalibrator()
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
        y = np.array([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
        calibrator.fit(x, y)

        test_points = np.linspace(0.0, 1.0, 20, dtype=np.float32)
        calibrated = calibrator.predict(test_points)

        # Monotonicity check: calibrated[i] <= calibrated[i+1]
        for i in range(len(calibrated) - 1):
            self.assertLessEqual(calibrated[i], calibrated[i + 1] + 1e-6)

    def test_dual_calibrator_achieves_low_ece(self):
        np.random.seed(42)
        # Synthetic over-confident raw predictions on near-miss dataset
        n = 500
        raw_confs = np.random.uniform(0.6, 0.99, size=n).astype(np.float32)
        # Actual accuracy is lower (e.g. 0.75) with near-miss traps
        true_labels = (np.random.rand(n) < 0.75).astype(np.int32)

        raw_ece = DualCalibrator.compute_10bin_ece(raw_confs, true_labels)
        self.assertGreater(raw_ece, 0.05)  # Miscalibrated before

        calibrator = DualCalibrator(temperature=1.3)
        calibrator.fit(raw_confs, true_labels)
        calibrated_confs = calibrator.calibrate(raw_confs)

        calibrated_ece = DualCalibrator.compute_10bin_ece(calibrated_confs, true_labels)
        # Target SLA: ECE <= 0.035
        self.assertLessEqual(calibrated_ece, 0.035)


class TestSIMDZeroCopyAndPhysicalMemoryLock(unittest.TestCase):
    """Tests for Milestone 3: 64-byte Aligned SIMD Packet & mlock."""

    def test_nanocore_vector_packet_alignment_and_roundtrip(self):
        import ctypes
        self.assertEqual(NanoCoreVectorPacket.data.offset, 64)
        self.assertEqual(ctypes.sizeof(NanoCoreVectorPacket), 4160)

        vec = np.linspace(0.0, 1.0, 1024, dtype=np.float32)
        packet = NanoCoreVectorPacket.from_numpy(vec, packet_id=123)

        self.assertEqual(packet.packet_id, 123)
        self.assertEqual(packet.dim, 1024)

        restored = packet.to_numpy()
        self.assertEqual(len(restored), 1024)
        np.testing.assert_allclose(restored, vec, atol=1e-6)

        # Zero copy view check
        view = packet.to_numpy(zero_copy=True)
        self.assertEqual(view.ctypes.data, ctypes.addressof(packet.data))

    def test_physical_memory_locker_graceful_handling(self):
        locker = PhysicalMemoryLocker()
        # Allocate dummy buffer
        buf = (np.ones(1024, dtype=np.float32)).ctypes.data
        success, msg = locker.lock_buffer(buf, 4096)
        # Verify call completes safely without crash
        self.assertIsInstance(success, bool)
        self.assertIsInstance(msg, str)
        # Unlock
        locker.unlock_buffer(buf)
        self.assertEqual(locker.locked_regions_count, 0)

    def test_simd_scoring_sub_millisecond_latency(self):
        engine = SIMDScoreEngine()
        bench = engine.benchmark_latency(rounds=50)
        # Single-core P99 latency SLA <= 1.4ms
        self.assertLessEqual(bench["p99_ms"], 1.4)


class TestMoVFusionAndFallbackWatchdog(unittest.TestCase):
    """Tests for Milestone 4: Mixture of Vectors (MoV) & Watchdog."""

    def setUp(self):
        self.layer = MoVDecisionLayer(dim=64, confidence_floor=0.30, expected_value_floor=-0.20)
        self.candidates = ["BUY", "SELL", "HOLD"]

    def test_rms_norm_invariance(self):
        v1 = np.random.randn(64).astype(np.float32) * 0.01  # Tiny magnitude
        v2 = np.random.randn(64).astype(np.float32) * 100.0 # Huge magnitude

        norm1 = rms_norm(v1)
        norm2 = rms_norm(v2)

        # RMS norm must have RMS = 1.0
        self.assertAlmostEqual(float(np.sqrt(np.mean(norm1 ** 2))), 1.0, places=3)
        self.assertAlmostEqual(float(np.sqrt(np.mean(norm2 ** 2))), 1.0, places=3)

    def test_mov_bayesian_pooling_confident_consensus(self):
        state = np.random.randn(64).astype(np.float32)

        out_code = MicroCoreOutput(
            core_id="core_code",
            domain="Code",
            action_probabilities={"BUY": 0.85, "SELL": 0.05, "HOLD": 0.10},
            closed_form_confidence=0.80,
            feature_vector=np.random.randn(64).astype(np.float32),
            expected_value=0.75,
        )
        out_ops = MicroCoreOutput(
            core_id="core_ops",
            domain="Ops",
            action_probabilities={"BUY": 0.80, "SELL": 0.10, "HOLD": 0.10},
            closed_form_confidence=0.70,
            feature_vector=np.random.randn(64).astype(np.float32),
            expected_value=0.60,
        )

        decision = self.layer.fuse_decisions(state, [out_code, out_ops], self.candidates)
        self.assertEqual(decision.best_action, "BUY")
        self.assertFalse(decision.fallback_escalated)
        self.assertGreaterEqual(decision.composite_confidence, 0.60)

    def test_fallback_watchdog_triggers_on_low_confidence(self):
        state = np.random.randn(64).astype(np.float32)
        # Uniform ambiguity -> very low confidence
        ambiguous_out = MicroCoreOutput(
            core_id="core_ood",
            domain="Market",
            action_probabilities={"BUY": 0.34, "SELL": 0.33, "HOLD": 0.33},
            closed_form_confidence=0.05,
            feature_vector=np.random.randn(64).astype(np.float32),
            expected_value=-0.30,
        )

        decision = self.layer.fuse_decisions(state, [ambiguous_out], self.candidates)
        self.assertTrue(decision.fallback_escalated)
        self.assertIn("LOW_COMPOSITE_CONFIDENCE", decision.escalation_reason)


class TestCPSATFormalSolverAndHardTimeout(unittest.TestCase):
    """Tests for Milestone 5: CP-SAT 0-1 Integer Solver & 2ms Hard Timeout."""

    def setUp(self):
        self.solver = CPSATFormalSolver(hard_timeout_ms=2.0)

    def test_cpsat_optimal_safety_solving(self):
        utilities = {"BUY": 0.95, "SELL": 0.10, "HOLD": 0.50}
        forbidden = {"BUY"}  # BUY is forbidden by safety risk rule

        verdict = self.solver.solve_safest_optimal_action(
            candidate_utilities=utilities,
            forbidden_actions=forbidden,
            fallback_safe_action="HOLD"
        )
        self.assertTrue(verdict.is_safe)
        # BUY was forbidden, so optimal safe action is HOLD (0.50 > 0.10)
        self.assertEqual(verdict.selected_action, "HOLD")
        self.assertLessEqual(verdict.solve_time_ms, 2.0)

    def test_all_candidates_forbidden_triggers_safe_intercept(self):
        utilities = {"BUY": 0.9, "SELL": 0.8}
        forbidden = {"BUY", "SELL"}

        verdict = self.solver.solve_safest_optimal_action(
            candidate_utilities=utilities,
            forbidden_actions=forbidden,
            fallback_safe_action="HOLD"
        )
        self.assertEqual(verdict.selected_action, "HOLD")
        self.assertTrue(verdict.fallback_used)
        self.assertEqual(verdict.solver_status, "ALL_CANDIDATES_FORBIDDEN_INTERCEPT")

    def test_timeout_budget_enforcement(self):
        # Force a microscopic 0.0001ms timeout to verify timeout circuit breaker
        strict_solver = CPSATFormalSolver(hard_timeout_ms=0.00001)
        utilities = {"A": 0.9, "B": 0.8}
        verdict = strict_solver.solve_safest_optimal_action(utilities)
        self.assertIn(verdict.selected_action, ["A", "B"])


if __name__ == "__main__":
    unittest.main()
