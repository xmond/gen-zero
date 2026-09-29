"""Tests for the instance-adaptive gating / log-linear opinion pool module.

Covers: reliability-feature validity, dynamic suppression of an unreliable
expert on high-entropy/high-disagreement samples, log-linear vs arithmetic
pooling behavior, hard support-set / abstain semantics, and save/load.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from gen_zero.manifold import (
    InstanceAdaptiveRouter,
    extract_reliability_features,
    jensen_shannon_divergence,
    normalized_entropy,
    top1_top2_margin,
)


def _one_hot_noisy(labels: np.ndarray, k: int, confidence: float) -> np.ndarray:
    """Rows are ``confidence`` on the true label, uniform over the rest."""
    n = labels.shape[0]
    off_mass = (1.0 - confidence) / (k - 1) if k > 1 else 0.0
    probs = np.full((n, k), off_mass, dtype=np.float64)
    probs[np.arange(n), labels] = confidence
    return probs


# ---------------------------------------------------------------------------
# Reliability feature extraction
# ---------------------------------------------------------------------------

class TestReliabilityFeatures:
    def test_feature_matrix_shape_and_finite(self):
        rng = np.random.default_rng(1)
        n, k = 50, 4
        probs = rng.dirichlet(np.ones(k), size=n)
        mean_probs = rng.dirichlet(np.ones(k), size=n)
        feats = extract_reliability_features(probs, mean_probs)
        assert feats.shape == (n, 3)
        assert np.all(np.isfinite(feats))

    def test_feature_matrix_with_distance_column(self):
        rng = np.random.default_rng(2)
        n, k = 30, 3
        probs = rng.dirichlet(np.ones(k), size=n)
        mean_probs = rng.dirichlet(np.ones(k), size=n)
        distance = rng.random(n)
        feats = extract_reliability_features(probs, mean_probs, distance=distance)
        assert feats.shape == (n, 4)
        assert np.all(np.isfinite(feats))
        assert np.allclose(feats[:, 3], distance)

    def test_no_nan_or_inf_on_degenerate_rows(self):
        # exact one-hot, exact uniform, and K=1 must not blow up log/entropy math.
        one_hot = np.array([[1.0, 0.0, 0.0]])
        uniform = np.array([[1 / 3, 1 / 3, 1 / 3]])
        zero_row = np.array([[0.0, 0.0, 0.0]])
        single_class = np.array([[1.0]])
        for probs in (one_hot, uniform, zero_row):
            mean_probs = np.array([[1 / 3, 1 / 3, 1 / 3]])
            feats = extract_reliability_features(probs, mean_probs)
            assert np.all(np.isfinite(feats)), probs
        feats_k1 = extract_reliability_features(single_class, single_class)
        assert np.all(np.isfinite(feats_k1))
        assert feats_k1[0, 0] == 0.0  # entropy of a K=1 distribution is defined as 0

    def test_normalized_entropy_bounds(self):
        one_hot = normalized_entropy(np.array([[1.0, 0.0]]))
        uniform2 = normalized_entropy(np.array([[0.5, 0.5]]))
        uniform4 = normalized_entropy(np.array([[0.25] * 4]))
        assert np.isclose(one_hot[0], 0.0, atol=1e-9)
        assert np.isclose(uniform2[0], 1.0)
        assert np.isclose(uniform4[0], 1.0)

    def test_top1_top2_margin_range(self):
        margin2 = top1_top2_margin(np.array([[0.9, 0.1], [0.5, 0.5]]))
        assert np.isclose(margin2[0], 0.8)
        assert np.isclose(margin2[1], 0.0)
        margin3 = top1_top2_margin(np.array([[1 / 3, 1 / 3, 1 / 3]]))
        assert np.isclose(margin3[0], 0.0)

    def test_js_divergence_identity_and_bounds(self):
        p = np.array([[0.7, 0.3], [0.1, 0.9]])
        q = np.array([[0.7, 0.3], [0.9, 0.1]])
        js = jensen_shannon_divergence(p, q)
        assert np.isclose(js[0], 0.0, atol=1e-9)  # identical rows -> zero divergence
        assert js[1] > 0.5  # near-opposite rows -> high divergence
        assert np.all(js >= 0.0) and np.all(js <= 1.0 + 1e-9)


# ---------------------------------------------------------------------------
# Dynamic suppression of unreliable experts
# ---------------------------------------------------------------------------

class TestDynamicSuppression:
    def _make_unreliable_when_uncertain(self, rng, n, k):
        """Expert A is confidently correct on 'easy' rows and near-uniform +
        wrong on 'hard' rows; expert B is a steady, moderately reliable
        baseline. The ridge must learn A's risk correlates with A's own
        entropy/margin/JS features -- nothing here hardcodes that rule."""
        y_true = rng.integers(0, k, size=n)
        is_hard = rng.random(n) < 0.5
        wrong_label = (y_true + 1) % k

        expert_a = _one_hot_noisy(y_true, k, confidence=0.85)
        expert_a[is_hard] = _one_hot_noisy(wrong_label[is_hard], k, confidence=1.0 / k + 0.05)

        expert_b = _one_hot_noisy(y_true, k, confidence=0.6)
        return y_true, is_hard, expert_a, expert_b

    def test_high_entropy_expert_is_suppressed(self):
        rng = np.random.default_rng(42)
        n, k = 400, 3
        y_true, is_hard, expert_a, expert_b = self._make_unreliable_when_uncertain(rng, n, k)

        router = InstanceAdaptiveRouter(ridge_lambda=1.0, n_folds=5, random_state=0)
        router.fit([expert_a, expert_b], y_true, tau=1.0, epsilon=0.1, expert_names=["A", "B"])
        weights = router.predict_weights([expert_a, expert_b])

        weight_a_hard = weights[is_hard, 0].mean()
        weight_a_easy = weights[~is_hard, 0].mean()
        assert weight_a_hard < weight_a_easy, (weight_a_hard, weight_a_easy)
        # suppression must be substantial, not a rounding artifact
        assert weight_a_easy - weight_a_hard > 0.1

    def test_epsilon_shrinkage_bounds_weight_away_from_zero(self):
        rng = np.random.default_rng(7)
        n, k = 200, 2
        y_true, is_hard, expert_a, expert_b = self._make_unreliable_when_uncertain(rng, n, k)

        epsilon = 0.3
        router = InstanceAdaptiveRouter(random_state=1)
        router.fit(
            [expert_a, expert_b], y_true,
            global_weights=[0.5, 0.5], tau=0.5, epsilon=epsilon, expert_names=["A", "B"],
        )
        weights = router.predict_weights([expert_a, expert_b])
        # shrinkage toward the uniform (0.5, 0.5) prior guarantees a floor
        min_possible = epsilon * 0.5
        assert np.all(weights[:, 0] >= min_possible - 1e-9)

    def test_predict_weights_by_name_matches_reordered_position(self):
        # Production callers (e.g. dynamic-K MoE routing) may pass experts in
        # a different order call to call; expert_names must key each row to
        # its own risk model regardless of position.
        rng = np.random.default_rng(21)
        n, k = 50, 3
        y_true = rng.integers(0, k, size=n)
        expert_a = rng.dirichlet(np.ones(k), size=n)
        expert_b = rng.dirichlet(np.ones(k), size=n)
        router = InstanceAdaptiveRouter()
        router.fit([expert_a, expert_b], y_true, expert_names=["A", "B"])

        w_fit_order = router.predict_weights([expert_a, expert_b], expert_names=["A", "B"])
        w_swapped = router.predict_weights([expert_b, expert_a], expert_names=["B", "A"])
        assert np.allclose(w_fit_order[:, 0], w_swapped[:, 1])
        assert np.allclose(w_fit_order[:, 1], w_swapped[:, 0])

    def test_predict_weights_unknown_name_raises(self):
        rng = np.random.default_rng(22)
        n, k = 20, 2
        y_true = rng.integers(0, k, size=n)
        expert_a = rng.dirichlet(np.ones(k), size=n)
        expert_b = rng.dirichlet(np.ones(k), size=n)
        router = InstanceAdaptiveRouter()
        router.fit([expert_a, expert_b], y_true, expert_names=["A", "B"])
        with pytest.raises(ValueError):
            router.predict_weights([expert_a], expert_names=["not_fitted"])

    def test_predict_weights_named_subset_renormalizes_prior(self):
        # Selecting a single expert by name must still return valid,
        # normalized weights (subset softmax over just the active experts).
        rng = np.random.default_rng(23)
        n, k = 20, 2
        y_true = rng.integers(0, k, size=n)
        expert_a = rng.dirichlet(np.ones(k), size=n)
        expert_b = rng.dirichlet(np.ones(k), size=n)
        router = InstanceAdaptiveRouter()
        router.fit([expert_a, expert_b], y_true, expert_names=["A", "B"])
        w_subset = router.predict_weights([expert_a], expert_names=["A"])
        assert w_subset.shape == (n, 1)
        assert np.allclose(w_subset[:, 0], 1.0)

    def test_weights_sum_to_one_per_sample(self):
        rng = np.random.default_rng(3)
        n, k = 100, 3
        y_true = rng.integers(0, k, size=n)
        probs = [rng.dirichlet(np.ones(k), size=n) for _ in range(4)]
        router = InstanceAdaptiveRouter()
        router.fit(probs, y_true)
        weights = router.predict_weights(probs)
        assert weights.shape == (n, 4)
        assert np.allclose(weights.sum(axis=1), 1.0)
        assert np.all(weights >= 0.0)


# ---------------------------------------------------------------------------
# Fusion / opinion pool behavior
# ---------------------------------------------------------------------------

class TestFusion:
    def test_arithmetic_matches_manual_weighted_mean(self):
        p1 = np.array([[0.7, 0.3], [0.2, 0.8]])
        p2 = np.array([[0.4, 0.6], [0.9, 0.1]])
        w = np.array([[0.5, 0.5], [0.25, 0.75]])
        router = InstanceAdaptiveRouter()
        fused = router.fuse_predictions([p1, p2], pool_type="arithmetic", dynamic_weights=w)
        expected = w[:, 0:1] * p1 + w[:, 1:2] * p2
        assert np.allclose(fused, expected)

    def test_log_linear_sharper_than_arithmetic_on_agreement(self):
        # both experts agree class 0 is likely; log-linear should be at least
        # as confident in the majority class as the arithmetic mean.
        p1 = np.array([[0.7, 0.3]])
        p2 = np.array([[0.75, 0.25]])
        w = np.array([[0.5, 0.5]])
        router = InstanceAdaptiveRouter()
        fused_log = router.fuse_predictions([p1, p2], pool_type="log_linear", dynamic_weights=w)
        fused_arith = router.fuse_predictions([p1, p2], pool_type="arithmetic", dynamic_weights=w)
        assert fused_log[0, 0] >= fused_arith[0, 0] - 1e-9

    def test_log_linear_lets_a_confident_veto_dominate(self):
        # expert 2 is near-certain class 1; log-linear should move much
        # further toward class 1 than a plain arithmetic average would.
        p1 = np.array([[0.5, 0.5]])
        p2 = np.array([[0.001, 0.999]])
        w = np.array([[0.5, 0.5]])
        router = InstanceAdaptiveRouter()
        fused_log = router.fuse_predictions([p1, p2], pool_type="log_linear", dynamic_weights=w)
        fused_arith = router.fuse_predictions([p1, p2], pool_type="arithmetic", dynamic_weights=w)
        assert fused_arith[0, 1] == pytest.approx(0.7495, abs=1e-3)
        assert fused_log[0, 1] > fused_arith[0, 1]
        assert fused_log[0, 1] > 0.9

    def test_single_expert_fusion_is_near_identity(self):
        p1 = np.array([[0.3, 0.7], [0.9, 0.1]])
        w = np.ones((2, 1))
        router = InstanceAdaptiveRouter()
        fused_log = router.fuse_predictions([p1], pool_type="log_linear", dynamic_weights=w)
        fused_arith = router.fuse_predictions([p1], pool_type="arithmetic", dynamic_weights=w)
        assert np.allclose(fused_log, p1, atol=1e-3)
        assert np.allclose(fused_arith, p1)

    def test_fused_outputs_are_valid_distributions(self):
        rng = np.random.default_rng(9)
        n, k, m = 60, 5, 3
        probs = [rng.dirichlet(np.ones(k), size=n) for _ in range(m)]
        w = rng.dirichlet(np.ones(m), size=n)
        router = InstanceAdaptiveRouter()
        for pool_type in ("arithmetic", "log_linear"):
            fused = router.fuse_predictions(probs, pool_type=pool_type, dynamic_weights=w)
            assert fused.shape == (n, k)
            assert np.all(np.isfinite(fused))
            assert np.all(fused >= 0.0)
            assert np.allclose(fused.sum(axis=1), 1.0)

    def test_hard_support_set_excludes_zero_mass_classes(self):
        # every expert declares class 2 infeasible (exactly 0.0); log-linear's
        # delta-smoothing must not resurrect it.
        p1 = np.array([[0.6, 0.4, 0.0]])
        p2 = np.array([[0.5, 0.5, 0.0]])
        w = np.array([[0.5, 0.5]])
        router = InstanceAdaptiveRouter()
        fused = router.fuse_predictions([p1, p2], pool_type="log_linear", dynamic_weights=w)
        assert fused[0, 2] == 0.0
        assert np.isclose(fused.sum(), 1.0)

    def test_all_zero_row_stays_zero_no_nan(self):
        # all experts abstain (all-zero row): fused output must stay all-zero
        # (downstream treats this as ABSTAIN), never NaN from a 0/0 divide.
        p1 = np.array([[0.0, 0.0]])
        p2 = np.array([[0.0, 0.0]])
        w = np.array([[0.5, 0.5]])
        router = InstanceAdaptiveRouter()
        for pool_type in ("arithmetic", "log_linear"):
            fused = router.fuse_predictions([p1, p2], pool_type=pool_type, dynamic_weights=w)
            assert np.all(np.isfinite(fused))
            assert np.allclose(fused, 0.0)

    def test_unknown_pool_type_raises(self):
        p1 = np.array([[0.5, 0.5]])
        router = InstanceAdaptiveRouter()
        with pytest.raises(ValueError):
            router.fuse_predictions([p1], pool_type="bogus", dynamic_weights=np.ones((1, 1)))

    def test_fuse_predictions_uses_fitted_router_when_weights_omitted(self):
        rng = np.random.default_rng(11)
        n, k = 100, 3
        y_true = rng.integers(0, k, size=n)
        probs = [rng.dirichlet(np.ones(k), size=n) for _ in range(2)]
        router = InstanceAdaptiveRouter()
        router.fit(probs, y_true)
        fused = router.fuse_predictions(probs, pool_type="log_linear")
        assert fused.shape == (n, k)
        assert np.allclose(fused.sum(axis=1), 1.0)

    def test_fuse_predictions_without_fit_or_weights_raises(self):
        p1 = np.array([[0.5, 0.5]])
        router = InstanceAdaptiveRouter()
        with pytest.raises(RuntimeError):
            router.fuse_predictions([p1], pool_type="log_linear")


# ---------------------------------------------------------------------------
# Validation / fail-closed behavior
# ---------------------------------------------------------------------------

class TestValidationAndPersistence:
    def test_predict_before_fit_raises(self):
        router = InstanceAdaptiveRouter()
        with pytest.raises(RuntimeError):
            router.predict_weights([np.array([[0.5, 0.5]])])

    @pytest.mark.parametrize("bad_tau", [0.0, -1.0])
    def test_invalid_tau_raises(self, bad_tau):
        router = InstanceAdaptiveRouter()
        with pytest.raises(ValueError):
            router.fit([np.array([[0.5, 0.5]])], np.array([0]), tau=bad_tau)

    @pytest.mark.parametrize("bad_epsilon", [-0.1, 1.1])
    def test_invalid_epsilon_raises(self, bad_epsilon):
        router = InstanceAdaptiveRouter()
        with pytest.raises(ValueError):
            router.fit([np.array([[0.5, 0.5]])], np.array([0]), epsilon=bad_epsilon)

    def test_mismatched_expert_shapes_raise(self):
        router = InstanceAdaptiveRouter()
        with pytest.raises(ValueError):
            router.fit(
                [np.array([[0.5, 0.5]]), np.array([[0.3, 0.3, 0.4]])],
                np.array([0]),
            )

    def test_out_of_range_label_raises(self):
        router = InstanceAdaptiveRouter()
        with pytest.raises(ValueError):
            router.fit([np.array([[0.5, 0.5]])], np.array([5]))

    def test_save_load_round_trip(self):
        rng = np.random.default_rng(5)
        n, k = 80, 3
        y_true = rng.integers(0, k, size=n)
        probs = [rng.dirichlet(np.ones(k), size=n) for _ in range(2)]
        router = InstanceAdaptiveRouter(ridge_lambda=2.0, random_state=3)
        router.fit(probs, y_true, expert_names=["m0", "m1"], tau=0.7, epsilon=0.15)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "router.npz")
            router.save(path)
            reloaded = InstanceAdaptiveRouter.load(path)

        assert reloaded.expert_names == ["m0", "m1"]
        w_before = router.predict_weights(probs)
        w_after = reloaded.predict_weights(probs)
        assert np.allclose(w_before, w_after)
        fused_before = router.fuse_predictions(probs, pool_type="log_linear")
        fused_after = reloaded.fuse_predictions(probs, pool_type="log_linear")
        assert np.allclose(fused_before, fused_after)


# ---------------------------------------------------------------------------
# Real production wiring: GenZero.decide() Stage-2 consensus fusion
# ---------------------------------------------------------------------------

class TestClientWiring:
    """Proves the module is mounted on the real decide() call path in
    client.py, not an unreferenced island: enabling it changes decide()'s
    actual fused output, and the numbers match an independent, standalone
    InstanceAdaptiveRouter computation over the same inputs. Also proves the
    integration fails closed instead of silently degrading: enabling
    adaptive gating without a router that covers every Stage-2 planner
    paradigm is refused at construction, before any decide() call can ever
    reach an unfitted expert."""

    # Must match UniversalParadigmRouter.all_paradigms minus "cp_sat" in
    # client.py -- the full Stage-2 planner pool the router must cover.
    FULL_PLANNER_POOL = (
        "reflex", "mcts", "astar", "bidirectional", "world_model",
        "mpc_cem", "gflownet", "cfr",
    )

    def _fit_full_coverage_router(self, tmp_path, expert_names=FULL_PLANNER_POOL):
        rng = np.random.default_rng(0)
        n, k = 80, 2
        y_true = rng.integers(0, k, size=n)
        oof_probs = [
            _one_hot_noisy(y_true, k, confidence=0.55 + 0.02 * i)
            for i in range(len(expert_names))
        ]
        router = InstanceAdaptiveRouter()
        router.fit(oof_probs, y_true, expert_names=list(expert_names))
        path = tmp_path / "router.npz"
        router.save(str(path))
        return router, str(path)

    def _make_client(self, artifact_path, pool_type="log_linear"):
        from gen_zero.client import GenZero
        from gen_zero.config import GenZeroConfig

        cfg = GenZeroConfig(
            hidden_dim=32, embed_dim=4, mcts_simulations=8,
            enable_gpu_arbiter_fallback=False,
            enable_adaptive_gating=True,
            adaptive_gating_artifact=artifact_path,
            adaptive_gating_pool_type=pool_type,
        )
        return GenZero(cfg)

    def test_construction_fails_closed_without_artifact(self):
        from gen_zero.client import GenZero
        from gen_zero.config import GenZeroConfig

        with pytest.raises(ValueError):
            GenZero(GenZeroConfig(hidden_dim=32, embed_dim=4, enable_adaptive_gating=True))

    def test_construction_fails_closed_on_partial_planner_coverage(self, tmp_path):
        # A router fitted for only 2 of the 8 real Stage-2 planner paradigms
        # must never be accepted: a later decide() call could pick any of the
        # other 6 and there would be no fitted risk model for it. This must
        # be caught at construction, not discovered mid-decision.
        _, artifact_path = self._fit_full_coverage_router(tmp_path, expert_names=("mcts", "astar"))
        with pytest.raises(ValueError, match="no fitted risk model"):
            self._make_client(artifact_path)

    def test_decide_routes_through_fitted_router_and_matches_standalone_computation(self, tmp_path):
        router, artifact_path = self._fit_full_coverage_router(tmp_path)
        gz = self._make_client(artifact_path)
        assert gz.adaptive_gating_router is not None

        gz.moe_router.route_dynamic = lambda **kwargs: {
            "k": 2,
            "selected_experts": [("mcts", 0.5), ("astar", 0.5)],
            "complexity_score": 0.5,
            "pipeline_has_cpsat": False,
        }

        expert_probs = {"mcts": {"A": 0.9, "B": 0.1}, "astar": {"A": 0.85, "B": 0.15}}

        def fake_expert(**kwargs):
            name = kwargs["expert_name"]
            candidates = kwargs["candidates"]
            probs = {c: expert_probs[name].get(c, 0.0) for c in candidates}
            return probs, 0.5, "A", {"valid_set": list(candidates), "status": "OK"}

        gz._execute_expert_distribution = fake_expert

        res = gz.decide(
            {"x": 0}, ["A", "B"], mode="auto",
            transition_fn=lambda s, a: (s, 0.0, False),
            constraints=[],
        )

        assert res["adaptive_params"]["fusion"]["dynamic"] is True
        assert res["adaptive_params"]["fusion"]["pool_type"] == "log_linear"

        rows = [np.array([[0.9, 0.1]]), np.array([[0.85, 0.15]])]
        expected_w = router.predict_weights(rows, expert_names=["mcts", "astar"])
        expected_fused = router.fuse_predictions(rows, pool_type="log_linear", dynamic_weights=expected_w)

        assert res["probs"]["A"] == pytest.approx(round(float(expected_fused[0, 0]), 4), abs=1e-4)
        assert res["probs"]["B"] == pytest.approx(round(float(expected_fused[0, 1]), 4), abs=1e-4)

    def test_decide_disabled_adaptive_gating_keeps_static_consensus(self, tmp_path):
        # Sanity check on the untouched default path: with adaptive gating off
        # entirely, decide() must still use the original static norm_w mean.
        from gen_zero.client import GenZero
        from gen_zero.config import GenZeroConfig

        gz = GenZero(GenZeroConfig(
            hidden_dim=32, embed_dim=4, mcts_simulations=8,
            enable_gpu_arbiter_fallback=False,
        ))
        assert gz.adaptive_gating_router is None

        gz.moe_router.route_dynamic = lambda **kwargs: {
            "k": 1,
            "selected_experts": [("reflex", 1.0)],
            "complexity_score": 0.2,
            "pipeline_has_cpsat": False,
        }

        def fake_expert(**kwargs):
            candidates = kwargs["candidates"]
            probs = {c: 0.0 for c in candidates}
            probs["A"] = 0.7
            probs["B"] = 0.3
            return probs, 0.4, "A", {"valid_set": list(candidates), "status": "OK"}

        gz._execute_expert_distribution = fake_expert

        res = gz.decide(
            {"x": 0}, ["A", "B"], mode="auto",
            transition_fn=lambda s, a: (s, 0.0, False),
            constraints=[],
        )
        assert res["adaptive_params"]["fusion"]["dynamic"] is False
        assert res["probs"]["A"] == pytest.approx(0.7)
        assert res["probs"]["B"] == pytest.approx(0.3)

    def test_decide_raises_loudly_if_an_unfitted_expert_ever_reaches_fusion(self, tmp_path):
        # Defense in depth: even though construction guarantees full coverage
        # against moe_router.all_paradigms at that moment, prove the decide()-time
        # guard itself refuses to silently fall back if an unfitted name ever
        # reaches the fusion step (e.g. all_paradigms mutated after construction).
        _, artifact_path = self._fit_full_coverage_router(tmp_path)
        gz = self._make_client(artifact_path)

        gz.moe_router.route_dynamic = lambda **kwargs: {
            "k": 1,
            "selected_experts": [("never_fitted_paradigm", 1.0)],
            "complexity_score": 0.2,
            "pipeline_has_cpsat": False,
        }

        def fake_expert(**kwargs):
            candidates = kwargs["candidates"]
            return {c: 1.0 / len(candidates) for c in candidates}, 0.0, "A", {
                "valid_set": list(candidates), "status": "OK",
            }

        gz._execute_expert_distribution = fake_expert

        with pytest.raises(RuntimeError, match="no fitted risk model"):
            gz.decide(
                {"x": 0}, ["A", "B"], mode="auto",
                transition_fn=lambda s, a: (s, 0.0, False),
                constraints=[],
            )
