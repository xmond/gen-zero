"""Unit tests for Latent Manifold Bridge: 228-byte contract, H^4 x S^3 x R^8 metric, and contact flow."""

import hashlib
import math
import struct
import unittest

import numpy as np

from gen_zero.causal.latent_bridge import (
    CHART_DIM,
    DEFAULT_ZCA_SIGMA_CEILING,
    ENTRY_BYTES,
    ENTRY_DOMAIN,
    HYPERBOLIC_MARGIN,
    LatentEntryContractV1,
    ManifoldMetricParams,
    MixedManifoldChartProjector,
    MixedManifoldCoord,
    MixedManifoldPotential,
    LatentThinkingEngine,
    LatentThinkingResult,
    fallback_geodesic_sq,
    retract_to_manifold,
)
from gen_zero.causal.symplectic_thinking import ExitReason


class TestLatentEntryContract(unittest.TestCase):
    def test_01_canonical_layout_228_bytes(self):
        q0 = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
        p0 = np.zeros(16, dtype=np.float32)
        h0_digest = bytes([0x11] * 32)
        proj_digest = bytes([0x22] * 32)
        sigma = 15.0
        tenant = bytes([0x33] * 32)

        contract = LatentEntryContractV1(
            q0=q0,
            p0=p0,
            h0_digest=h0_digest,
            projection_program_digest=proj_digest,
            zca_sigma_max=sigma,
            tenant_workspace_key=tenant,
        )

        b = contract.canonical_bytes()
        self.assertEqual(len(b), ENTRY_BYTES)

        # Deserialization roundtrip
        recovered = LatentEntryContractV1.from_canonical_bytes(b)
        np.testing.assert_allclose(recovered.q0, q0, atol=1e-7)
        np.testing.assert_allclose(recovered.p0, p0, atol=1e-7)
        self.assertEqual(recovered.h0_digest, h0_digest)
        self.assertEqual(recovered.projection_program_digest, proj_digest)
        self.assertAlmostEqual(recovered.zca_sigma_max, sigma, places=5)
        self.assertEqual(recovered.tenant_workspace_key, tenant)

    def test_02_gen2_test_fixture_digest(self):
        """Cross-repo bit-exact match against gen2 latent_flow_entry_tests.rs."""
        fq = np.array([i * 0.25 - 1.0 for i in range(16)], dtype=np.float32)
        fp = np.array([-float(i) * 0.125 for i in range(16)], dtype=np.float32)
        fp[0] = struct.unpack(">f", bytes([0x80, 0x00, 0x00, 0x00]))[0]  # IEEE 754 -0.0

        contract = LatentEntryContractV1(
            q0=fq,
            p0=fp,
            h0_digest=bytes([1] * 32),
            projection_program_digest=bytes([2] * 32),
            zca_sigma_max=12.5,
            tenant_workspace_key=bytes([3] * 32),
        )
        raw = contract.canonical_bytes()
        self.assertEqual(len(raw), 228)
        self.assertEqual(raw[:4], bytes([0xBF, 0x80, 0x00, 0x00]))  # -1.0 in f32 be
        self.assertEqual(raw[64:68], bytes([0x80, 0x00, 0x00, 0x00]))  # -0.0 in f32 be
        self.assertEqual(raw[192:196], bytes([0x41, 0x48, 0x00, 0x00]))  # 12.5 in f32 be

        # Check domain separated digest
        d = contract.digest()
        self.assertEqual(len(d), 32)
        expected_digest_prefix = "f3f7c85f"
        self.assertTrue(
            d.hex().startswith(expected_digest_prefix),
            f"Digest hex {d.hex()} does not start with expected prefix {expected_digest_prefix}",
        )

    def test_03_g6_validation_and_immutable_ceiling(self):
        c = LatentEntryContractV1(
            q0=np.zeros(16, dtype=np.float32),
            p0=np.zeros(16, dtype=np.float32),
            h0_digest=bytes(32),
            projection_program_digest=bytes(32),
            zca_sigma_max=10.0,
            tenant_workspace_key=bytes(32),
        )
        self.assertTrue(c.validate())

        # Exceeds ceiling
        c.zca_sigma_max = 35.0
        self.assertFalse(c.validate(ceiling=30.0))

        # Caller cannot relax ceiling past DEFAULT_ZCA_SIGMA_CEILING (Finding F7)
        self.assertFalse(c.validate(ceiling=50.0))

        # Negative
        c.zca_sigma_max = -0.5
        self.assertFalse(c.validate())

        # NaN
        c.zca_sigma_max = float("nan")
        self.assertFalse(c.validate())


class TestMixedManifoldMetric(unittest.TestCase):
    def test_04_metric_properties(self):
        params = ManifoldMetricParams()
        x = np.array([0.1, -0.2, 0.3, -0.1, 0.5, 0.5, 0.5, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        y = np.array([-0.2, 0.1, -0.1, 0.2, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        # Self-distance is zero
        d_xx = fallback_geodesic_sq(x, x, params)
        self.assertAlmostEqual(d_xx, 0.0, places=6)

        # Symmetry
        d_xy = fallback_geodesic_sq(x, y, params)
        d_yx = fallback_geodesic_sq(y, x, params)
        self.assertAlmostEqual(d_xy, d_yx, places=6)
        self.assertGreater(d_xy, 0.0)

    def test_05_analytic_gradient_matches_finite_differences(self):
        """Verifies potential gradient matches numerical central differences."""
        centres = np.array([
            [0.1, -0.1, 0.2, -0.2, 0.5, 0.5, 0.5, 0.5, 1.0, 0.0, -1.0, 2.0, 0.5, -0.5, 1.5, -1.5],
            [-0.2, 0.2, -0.1, 0.1, 0.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 2.0, -2.0, 0.5],
        ])
        pot = MixedManifoldPotential(anchors=centres, log_w=np.array([0.2, -0.1]), beta=3.0)

        q = np.array([0.15, -0.05, 0.1, -0.1, 0.6, 0.0, 0.8, 0.0, 0.2, 0.5, -0.3, 1.2, -0.8, 0.1, 1.0, -0.5])

        analytic_grad = pot.grad(q)
        eps = 1e-6
        numerical_grad = np.zeros(16)
        for i in range(16):
            e = np.zeros(16)
            e[i] = eps
            v_plus = pot.energy(q + e)
            v_minus = pot.energy(q - e)
            numerical_grad[i] = (v_plus - v_minus) / (2.0 * eps)

        rel_error = np.linalg.norm(analytic_grad - numerical_grad) / (np.linalg.norm(numerical_grad) + 1e-12)
        self.assertLess(rel_error, 1e-4, f"Relative gradient error {rel_error} exceeds 1e-4")

    def test_06_curvature_general_retraction(self):
        """Finding F7: verify retraction scales with c_H."""
        z = np.ones(16) * 5.0
        q_c1 = retract_to_manifold(z, c_h=1.0)
        q_c4 = retract_to_manifold(z, c_h=4.0)

        coord_c1 = MixedManifoldCoord(q_c1)
        coord_c4 = MixedManifoldCoord(q_c4)

        self.assertTrue(coord_c1.is_valid(c_h=1.0))
        self.assertTrue(coord_c4.is_valid(c_h=4.0))
        self.assertLess(float(np.linalg.norm(coord_c4.hyperbolic)), 0.5)


class TestChartProjectorAndEngine(unittest.TestCase):
    def test_07_projector_retraction_and_contract(self):
        rng = np.random.default_rng(42)
        n, d = 64, 4096
        h_samples = rng.normal(size=(n, d))
        projector = MixedManifoldChartProjector(feature_dim=d, ceiling=30.0).fit(h_samples)

        self.assertLessEqual(projector.zca_sigma_max, DEFAULT_ZCA_SIGMA_CEILING)
        self.assertEqual(len(projector.projection_program_digest), 32)

        # Test single projection
        h = rng.normal(size=d)
        q = projector.project(h)
        self.assertEqual(len(q), 16)

        # Hyperbolic norm < 1.0
        norm_h = float(np.linalg.norm(q[0:4]))
        self.assertLess(norm_h, 1.0)

        # Spherical norm == 1.0
        norm_s = float(np.linalg.norm(q[4:8]))
        self.assertAlmostEqual(norm_s, 1.0, places=5)

        # Contract build
        contract = projector.build_contract(h, tenant_key=b"0" * 32)
        self.assertTrue(contract.validate())
        self.assertEqual(len(contract.canonical_bytes()), 228)

    def test_08_safe_exit_and_coevolution_rollback(self):
        """Findings F2 & F4: verify rollback and no weight corruption on Lyapunov violation or NaN."""
        rng = np.random.default_rng(17)
        proj = MixedManifoldChartProjector(feature_dim=16).fit(rng.normal(size=(64, 16)))
        anchors = np.stack([proj.project(x) for x in rng.normal(size=(3, 16))])
        pot = MixedManifoldPotential(anchors)
        eng = LatentThinkingEngine(pot, proj, dt=0.05, gamma=1.0)

        before_weights = pot.log_w.copy()
        res = eng.run(rng.normal(size=16), bytes(32), coevolve=True)

        if res.exit_reason == ExitReason.LyapunovViolation:
            # Finding F2: weight must NOT be updated on Lyapunov violation!
            np.testing.assert_array_equal(pot.log_w, before_weights)

        # Finding F4: NaN coevolution fail-closed test
        pot.coevolve(np.array([np.nan, 1.0, 0.0]), eta=0.1)
        self.assertTrue(np.all(np.isfinite(pot.weights)))

        # Finding F4: repeated feedback underflow bounds
        for _ in range(1000):
            pot.coevolve(np.array([1.0, 0.0, 0.0]), eta=0.1)
        self.assertTrue(np.all(np.isfinite(pot.weights)))
        self.assertGreater(float(np.min(pot.weights)), 1e-15)

    def test_09_latent_thinking_trajectory_and_latency(self):
        rng = np.random.default_rng(123)
        n, d, k = 64, 4096, 4
        h_samples = rng.normal(size=(n, d))
        projector = MixedManifoldChartProjector(feature_dim=d, ceiling=30.0).fit(h_samples)

        class_h = rng.normal(size=(k, d))
        pot = MixedManifoldPotential.from_class_features(projector, class_h, beta=4.0)
        engine = LatentThinkingEngine(potential=pot, projector=projector, dt=0.05, gamma=1.0, max_steps=16)

        # Warmup
        test_h = class_h[0] + rng.normal(scale=0.05, size=d)
        for _ in range(5):
            engine.run(test_h, tenant_key=b"1" * 32, coevolve=False)

        res = engine.run(test_h, tenant_key=b"1" * 32, coevolve=True)

        # 0-token invariant
        self.assertEqual(res.tokens_emitted, 0)

        # Latency gate: trajectory < 35ms
        self.assertLess(res.segment_l_ms, 35.0)
        self.assertLess(res.segment_t_ms + res.segment_l_ms + res.segment_r_ms, 35.0)

    def test_10_coevolution_simplex_and_finite_eta_validation(self):
        """Finding F4: ensure coevolve validates eta and simplex membership."""
        anchors = np.zeros((2, 16))
        anchors[:, 4] = 1.0
        pot = MixedManifoldPotential(anchors)
        initial_log_w = pot.log_w.copy()

        # NaN eta -> must not change weights
        pot.coevolve(np.array([0.5, 0.5]), eta=float("nan"))
        np.testing.assert_array_equal(pot.log_w, initial_log_w)

        # Negative r -> must not change weights
        pot.coevolve(np.array([-1.0, 2.0]), eta=0.1)
        np.testing.assert_array_equal(pot.log_w, initial_log_w)

        # Unnormalized r -> must not change weights
        pot.coevolve(np.array([1.0, 1.0]), eta=0.1)
        np.testing.assert_array_equal(pot.log_w, initial_log_w)

    def test_11_unconditional_lyapunov_violation_rollback_and_gate(self):
        """Finding F2: unconditional regression test for LyapunovViolation rollback and writeback gate."""
        anchors = np.zeros((2, 16))
        anchors[:, 4] = 1.0

        class DiscontinuousPotential(MixedManifoldPotential):
            def __init__(self, anchors):
                super().__init__(anchors)
                self.calls = 0

            def energy_and_grad(self, q):
                self.calls += 1
                # On step 1, inject a sudden positive jump in energy (violating dH/dt <= 0)
                energy = 0.0 if self.calls == 1 else 100.0
                return energy, np.zeros(16)

        pot = DiscontinuousPotential(anchors)
        eng = LatentThinkingEngine(pot, dt=0.05, gamma=1.0)
        q0 = anchors[0].copy()
        p0 = np.zeros(16)
        p0[8] = 1.0  # momentum in Euclidean block

        q_ret, p_ret, steps, reason, trace = eng.rollout(q0, p0)
        self.assertEqual(reason, ExitReason.LyapunovViolation)
        self.assertEqual(steps, 0)
        np.testing.assert_array_equal(q_ret, q0)
        np.testing.assert_array_equal(p_ret, p0)

    def test_12_ambient_spherical_gradient_boundary_consistency(self):
        """Finding F3: verify ambient spherical gradient matches finite difference at boundary clamp."""
        anchors = np.zeros((1, 16))
        anchors[0, 4] = 1.0
        pot = MixedManifoldPotential(anchors)

        q = np.zeros(16)
        q[4] = 1.1  # off-manifold test point
        eps = 1e-6
        e0, g0 = pot.energy_and_grad(q)

        q_plus = q.copy()
        q_plus[4] += eps
        e_plus, _ = pot.energy_and_grad(q_plus)

        q_minus = q.copy()
        q_minus[4] -= eps
        e_minus, _ = pot.energy_and_grad(q_minus)

        fd = (e_plus - e_minus) / (2.0 * eps)
        self.assertAlmostEqual(g0[4], fd, delta=1e-5)
        self.assertEqual(g0[4], 0.0)

    def test_13_cross_language_simd_metric_tolerance(self):
        """Finding F6: verify exact Rust simd_metric.rs approximations satisfy |d_rust - d_exact| <= 3e-4."""
        # Exact reproduction of Rust functions from crates/foundation/manifold/src/product/simd_metric.rs
        def rust_fast_arcosh_delta_approx(delta: float) -> float:
            d = max(0.0, float(np.float32(delta)))
            if d < 1e-4:
                s = math.sqrt(2.0 * d)
                return float(np.float32(s * (1.0 - d * 0.08333333)))
            radicand = d * (2.0 + d)
            u = 1.0 + d + math.sqrt(radicand)
            return float(np.float32(math.log(u)))

        def rust_fast_arccos_approx(x: float) -> float:
            x_clamped = min(max(float(np.float32(x)), -1.0), 1.0)
            negate = x_clamped < 0.0
            x_abs = abs(x_clamped)
            p = (-0.0187293 * x_abs + 0.0742610) * x_abs - 0.2121144
            p = p * x_abs + 1.5707288
            ret = math.sqrt(max(0.0, 1.0 - x_abs)) * p
            if negate:
                ret = math.pi - ret
            return float(np.float32(ret))

        def rust_fallback_geodesic_sq(x: np.ndarray, y: np.ndarray, params: ManifoldMetricParams) -> float:
            c_h = max(params.c_hyperbolic, 1e-6)
            c_s = max(params.c_spherical, 1e-6)
            eps = min(max(params.anti_collapse_eps, 1e-5), 0.1)

            diff_h_sq = float(np.sum((x[0:4] - y[0:4]) ** 2))
            norm_x_h = float(np.sum(x[0:4] ** 2))
            norm_y_h = float(np.sum(y[0:4] ** 2))
            denom_x = max(1.0 - c_h * norm_x_h, eps)
            denom_y = max(1.0 - c_h * norm_y_h, eps)
            delta = (2.0 * c_h * diff_h_sq) / (denom_x * denom_y)
            d_h = rust_fast_arcosh_delta_approx(delta) / math.sqrt(c_h)

            cos_s = min(max(float(np.dot(x[4:8], y[4:8])), -1.0), 1.0)
            d_s = rust_fast_arccos_approx(cos_s) / math.sqrt(c_s)

            diff_e_sq = float(np.sum((x[8:16] - y[8:16]) ** 2))
            return params.w_hyperbolic * (d_h * d_h) + params.w_spherical * (d_s * d_s) + params.w_euclidean * diff_e_sq

        params = ManifoldMetricParams(c_hyperbolic=1.0, c_spherical=1.0)

        # 1. Hyperbolic domain sweep (including Taylor transition at delta=1e-4)
        for r in [0.005, 0.05, 0.1, 0.3, 0.5, 0.7, 0.85]:
            x = np.zeros(16)
            y = np.zeros(16)
            x[4], y[4] = 1.0, 1.0
            x[0] = r
            d_exact = math.sqrt(fallback_geodesic_sq(x, y, params))
            d_rust = math.sqrt(rust_fallback_geodesic_sq(x, y, params))
            self.assertLessEqual(abs(d_rust - d_exact), 3e-4)

        # 2. Spherical domain sweep (across Remez polynomial range)
        for angle in np.linspace(0.01, math.pi - 0.01, 10):
            x = np.zeros(16)
            y = np.zeros(16)
            x[4] = 1.0
            y[4] = math.cos(angle)
            y[5] = math.sin(angle)
            d_exact = math.sqrt(fallback_geodesic_sq(x, y, params))
            d_rust = math.sqrt(rust_fallback_geodesic_sq(x, y, params))
            self.assertLessEqual(abs(d_rust - d_exact), 3e-4)

        # 3. Mixed product manifold points
        rng = np.random.default_rng(42)
        for _ in range(20):
            x = rng.normal(size=16)
            y = rng.normal(size=16)
            # Project to manifold
            x = retract_to_manifold(x, 1.0)
            y = retract_to_manifold(y, 1.0)
            d_exact = math.sqrt(fallback_geodesic_sq(x, y, params))
            d_rust = math.sqrt(rust_fallback_geodesic_sq(x, y, params))
            self.assertLessEqual(abs(d_rust - d_exact), 3e-4)


if __name__ == "__main__":
    unittest.main()
