"""Unit tests for the Wasserstein-OT MoE router and orthogonal tangent pool
(doc `05-heterogeneous-causal-moe-and-future-evolution.md`, Sprint 1 + 2).

No mocks, no hardcoded outputs, no label leakage, no GPU: every assertion is
checked against an independently computed reference (closed-form diagonal
Bures-Wasserstein distance, brute-force cross-basis inner products, direct
eigenvalue computation) or a mathematical invariant (convexity, continuity,
orthogonality, spectral radius). Pure single-core numpy throughout.
"""
import numpy as np
import pytest

from gen_zero.causal.wasserstein_moe_router import (
    ExpertManifold,
    HeterogeneousCausalMoEPipeline,
    MultiTangentManifoldPool,
    WassersteinOptimalTransportRouter,
    bures_wasserstein_sq_distance,
)


# --------------------------------------------------------------------------
# Bures-Wasserstein distance correctness
# --------------------------------------------------------------------------


def test_bures_wasserstein_matches_diagonal_closed_form():
    """For diagonal covariances the Bures-Wasserstein distance has a trivial
    closed form: ``sum (sqrt(c1_i) - sqrt(c2_i))^2`` on top of the mean term,
    since commuting (diagonal) covariances' matrix square roots also commute."""
    rng = np.random.default_rng(0)
    dim = 6
    m1, m2 = rng.normal(size=dim), rng.normal(size=dim)
    c1_diag = rng.uniform(0.1, 3.0, size=dim)
    c2_diag = rng.uniform(0.1, 3.0, size=dim)
    c1, c2 = np.diag(c1_diag), np.diag(c2_diag)

    reference = np.sum((m1 - m2) ** 2) + np.sum((np.sqrt(c1_diag) - np.sqrt(c2_diag)) ** 2)
    got = bures_wasserstein_sq_distance(m1, c1, m2, c2)
    assert got == pytest.approx(reference, abs=1e-8)


def test_bures_wasserstein_zero_covariance_reduces_to_euclidean():
    rng = np.random.default_rng(1)
    dim = 5
    m1, m2 = rng.normal(size=dim), rng.normal(size=dim)
    zero = np.zeros((dim, dim))
    got = bures_wasserstein_sq_distance(m1, zero, m2, zero)
    assert got == pytest.approx(np.sum((m1 - m2) ** 2), abs=1e-10)


def test_bures_wasserstein_identical_distributions_is_zero():
    rng = np.random.default_rng(2)
    dim = 8
    m = rng.normal(size=dim)
    a = rng.normal(size=(dim, dim))
    cov = a @ a.T + 0.1 * np.eye(dim)
    got = bures_wasserstein_sq_distance(m, cov, m, cov)
    assert got == pytest.approx(0.0, abs=1e-6)


def test_bures_wasserstein_rejects_indefinite_covariance():
    dim = 4
    bad_cov = -np.eye(dim)
    with pytest.raises(ValueError):
        bures_wasserstein_sq_distance(np.zeros(dim), bad_cov, np.zeros(dim), np.eye(dim))


# --------------------------------------------------------------------------
# Router: convex-combination property
# --------------------------------------------------------------------------


def _random_expert_manifold(rng, name, dim):
    a = rng.normal(size=(dim, dim)) * 0.3
    cov = a @ a.T + 0.05 * np.eye(dim)
    return ExpertManifold(name=name, mean=rng.normal(size=dim) * 2.0, covariance=cov)


def _three_expert_router(rng, dim=16, temperature=0.5):
    experts = [_random_expert_manifold(rng, name, dim) for name in ("E0_grid", "E1_codebook", "E2_zero")]
    return WassersteinOptimalTransportRouter(experts, temperature=temperature)


def test_router_weights_are_convex_combination():
    rng = np.random.default_rng(10)
    router = _three_expert_router(rng)
    for _ in range(20):
        x = rng.normal(size=router.dim) * 3.0
        weights = router.route(x)
        values = np.array(list(weights.values()))
        assert set(weights) == set(router.names)
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0)
        assert values.sum() == pytest.approx(1.0, abs=1e-8)


def test_router_weights_nearly_one_hot_near_an_attractor_at_low_temperature():
    rng = np.random.default_rng(11)
    router = _three_expert_router(rng, temperature=0.01)
    target = router.experts[1]
    weights = router.route(target.mean, target.covariance)
    assert weights[target.name] > 0.99
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-8)


def test_router_fuse_matches_manual_weighted_sum():
    rng = np.random.default_rng(12)
    router = _three_expert_router(rng)
    x = rng.normal(size=router.dim)
    weights = router.route(x)
    vectors = {name: rng.normal(size=router.dim) for name in router.names}
    fused = router.fuse(weights, vectors)
    reference = sum(weights[name] * vectors[name] for name in router.names)
    assert np.allclose(fused, reference, atol=1e-10)


def test_router_fuse_rejects_mismatched_expert_keys():
    rng = np.random.default_rng(13)
    router = _three_expert_router(rng)
    weights = router.route(rng.normal(size=router.dim))
    incomplete_vectors = {router.names[0]: np.zeros(router.dim)}
    with pytest.raises(ValueError):
        router.fuse(weights, incomplete_vectors)


def test_router_rejects_fewer_than_two_experts():
    rng = np.random.default_rng(14)
    with pytest.raises(ValueError):
        WassersteinOptimalTransportRouter([_random_expert_manifold(rng, "solo", 8)])


def test_router_rejects_mismatched_expert_dims():
    rng = np.random.default_rng(15)
    e1 = _random_expert_manifold(rng, "a", 8)
    e2 = _random_expert_manifold(rng, "b", 16)
    with pytest.raises(ValueError):
        WassersteinOptimalTransportRouter([e1, e2])


# --------------------------------------------------------------------------
# Router: boundary continuity (no step-function jumps)
# --------------------------------------------------------------------------


def test_router_weights_are_lipschitz_continuous_across_the_decision_boundary():
    """Walk a straight line through the exact midpoint between two experts'
    attractors (the classic hard-threshold flip point) and check every
    per-step weight change is bounded proportionally to the step size --
    the signature of a smooth interpolation, not a discrete cascade flip."""
    rng = np.random.default_rng(20)
    dim = 12
    e_a = _random_expert_manifold(rng, "A", dim)
    e_b = _random_expert_manifold(rng, "B", dim)
    router = WassersteinOptimalTransportRouter([e_a, e_b], temperature=1.0)

    midpoint = 0.5 * (e_a.mean + e_b.mean)
    direction = (e_b.mean - e_a.mean)
    direction = direction / np.linalg.norm(direction)

    n_steps = 400
    span = 6.0
    ts = np.linspace(-span / 2, span / 2, n_steps)
    step_size = ts[1] - ts[0]

    prev_weights = None
    max_jump = 0.0
    for t in ts:
        x = midpoint + t * direction
        w = router.route(x)
        wa = w["A"]
        if prev_weights is not None:
            max_jump = max(max_jump, abs(wa - prev_weights))
        prev_weights = wa

    # A Lipschitz-smooth softmin over a quadratic distance cannot jump more
    # than a small multiple of the step size; a hard cascade would jump by
    # ~1.0 in a single step exactly at t=0. Threshold is generous (10x the
    # step size) purely to absorb float noise, not to hide a real jump.
    assert max_jump < 10.0 * step_size


def test_router_weights_vary_continuously_under_small_perturbation():
    rng = np.random.default_rng(21)
    router = _three_expert_router(rng, dim=10, temperature=0.7)
    x0 = rng.normal(size=router.dim) * 2.0
    w0 = np.array(list(router.route(x0).values()))

    for eps in (1e-6, 1e-4, 1e-2):
        direction = rng.normal(size=router.dim)
        direction /= np.linalg.norm(direction)
        w1 = np.array(list(router.route(x0 + eps * direction).values()))
        delta = np.max(np.abs(w1 - w0))
        # Weight change should shrink roughly linearly with eps; it must
        # never be an O(1) jump for an infinitesimal perturbation.
        assert delta < 5.0 * eps + 1e-8


# --------------------------------------------------------------------------
# MultiTangentManifoldPool: orthogonality
# --------------------------------------------------------------------------


def test_tangent_pool_bases_are_mutually_orthogonal():
    pool = MultiTangentManifoldPool(tangent_dim=16, seed=42)
    cross = pool.orthogonality_matrix()
    n = pool.num_charts
    for i in range(n):
        for j in range(n):
            if i == j:
                assert cross[i, j] == pytest.approx(1.0, abs=1e-8)
            else:
                assert cross[i, j] < 1e-8


def test_tangent_pool_basis_columns_are_orthonormal_within_chart():
    pool = MultiTangentManifoldPool(tangent_dim=12, seed=7)
    for name in pool.CHART_NAMES:
        basis = pool.chart(name).basis
        gram = basis.T @ basis
        assert np.allclose(gram, np.eye(basis.shape[1]), atol=1e-8)


def test_tangent_pool_project_lift_round_trip_within_chart():
    pool = MultiTangentManifoldPool(tangent_dim=16, seed=3)
    rng = np.random.default_rng(99)
    x_ambient = rng.normal(size=pool.ambient_dim)
    for name in pool.CHART_NAMES:
        chart = pool.chart(name)
        local = chart.project(x_ambient)
        lifted = chart.lift(local)
        # lift(project(x)) is the orthogonal projection of x onto this
        # chart's subspace: re-projecting it must return the same local coords.
        assert np.allclose(chart.project(lifted), local, atol=1e-8)


# --------------------------------------------------------------------------
# MultiTangentManifoldPool: Lyapunov contractivity
# --------------------------------------------------------------------------


def test_tangent_pool_all_charts_are_spectrally_contractive():
    pool = MultiTangentManifoldPool(tangent_dim=20, seed=123, target_rho=0.9)
    radii = pool.spectral_radii()
    assert set(radii) == set(pool.CHART_NAMES)
    for name, rho in radii.items():
        assert rho < 1.0
        # Independently recompute rho from the stored A to catch any
        # mismatch between the cached value and the actual matrix.
        a = pool.chart(name).a
        recomputed = float(np.max(np.abs(np.linalg.eigvals(a))))
        assert recomputed == pytest.approx(rho, abs=1e-6)
        assert recomputed < 1.0


def test_tangent_pool_fixed_point_matches_iterated_recurrence():
    pool = MultiTangentManifoldPool(tangent_dim=8, seed=5, target_rho=0.85)
    rng = np.random.default_rng(6)
    x_ambient = rng.normal(size=pool.ambient_dim)
    fixed_points = pool.evolve_all(x_ambient)

    for name in pool.CHART_NAMES:
        chart = pool.chart(name)
        drive = chart.b + chart.project(x_ambient)
        h = np.zeros(chart.tangent_dim)
        for _ in range(2000):  # contractive recurrence converges geometrically
            h = chart.a @ h + drive
        assert np.allclose(h, fixed_points[name], atol=1e-4)


def test_tangent_pool_rejects_non_positive_target_rho():
    with pytest.raises(ValueError):
        MultiTangentManifoldPool(target_rho=1.0)
    with pytest.raises(ValueError):
        MultiTangentManifoldPool(target_rho=0.0)


# --------------------------------------------------------------------------
# Pipeline: end-to-end integration
# --------------------------------------------------------------------------


def test_pipeline_end_to_end_produces_convex_weights_and_finite_ambient_state():
    rng = np.random.default_rng(200)
    tangent_dim = 16
    pool = MultiTangentManifoldPool(tangent_dim=tangent_dim, seed=1)
    router = WassersteinOptimalTransportRouter(
        [_random_expert_manifold(rng, name, tangent_dim) for name in ("E0_grid", "E1_codebook", "E2_zero")],
        temperature=0.6,
    )
    pipeline = HeterogeneousCausalMoEPipeline(router, pool)

    router_features = rng.normal(size=tangent_dim)
    expert_states = {name: rng.normal(size=tangent_dim) for name in router.names}
    tangent_input = rng.normal(size=pool.ambient_dim)

    result = pipeline.step(router_features, expert_states, tangent_input)

    weight_values = np.array(list(result.weights.values()))
    assert weight_values.sum() == pytest.approx(1.0, abs=1e-8)
    assert np.all(weight_values >= 0.0)
    assert result.fused_expert_state.shape == (tangent_dim,)
    assert np.all(np.isfinite(result.fused_expert_state))
    assert set(result.tangent_fixed_points) == set(pool.CHART_NAMES)
    for h in result.tangent_fixed_points.values():
        assert h.shape == (tangent_dim,)
        assert np.all(np.isfinite(h))
    assert result.reconstructed_ambient.shape == (pool.ambient_dim,)
    assert np.all(np.isfinite(result.reconstructed_ambient))


def test_pipeline_reconstruction_decomposes_additively_across_orthogonal_charts():
    """Because the four charts are mutually orthogonal, driving only one
    chart's fixed point and zeroing the rest must reconstruct into exactly
    that chart's own orthogonal subspace (no cross-talk leakage)."""
    pool = MultiTangentManifoldPool(tangent_dim=10, seed=55)
    rng = np.random.default_rng(56)
    only_chart = pool.CHART_NAMES[2]
    h = rng.normal(size=pool.tangent_dim)
    local_states = {name: np.zeros(pool.tangent_dim) for name in pool.CHART_NAMES}
    local_states[only_chart] = h

    ambient = pool.reconstruct_ambient(local_states)
    other_chart = pool.CHART_NAMES[0]
    projection_onto_other = pool.chart(other_chart).project(ambient)
    assert np.allclose(projection_onto_other, 0.0, atol=1e-8)


# --------------------------------------------------------------------------
# Numerical stability / CPU-only compatibility
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scale",
    [1e-6, 1.0, 1e3, 1e6],
)
def test_router_numerically_stable_across_input_magnitudes(scale):
    rng = np.random.default_rng(300)
    router = _three_expert_router(rng, dim=10)
    x = rng.normal(size=router.dim) * scale
    weights = router.route(x)
    values = np.array(list(weights.values()))
    assert np.all(np.isfinite(values))
    assert values.sum() == pytest.approx(1.0, abs=1e-6)


def test_router_handles_zero_vector_input():
    rng = np.random.default_rng(301)
    router = _three_expert_router(rng, dim=10)
    weights = router.route(np.zeros(router.dim))
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-8)


def test_module_has_no_gpu_or_torch_dependency():
    """This MoE gateway must run on a single offline CPU core: no torch,
    no CUDA, no distributed backend anywhere in its source."""
    import gen_zero.causal.wasserstein_moe_router as mod

    source = open(mod.__file__, "r", encoding="utf-8").read().lower()
    for forbidden in ("import torch", "cuda", "cupy", "tensorflow"):
        assert forbidden not in source


def test_pool_construction_is_deterministic_given_a_seed():
    pool_a = MultiTangentManifoldPool(tangent_dim=8, seed=77)
    pool_b = MultiTangentManifoldPool(tangent_dim=8, seed=77)
    for name in pool_a.CHART_NAMES:
        assert np.array_equal(pool_a.chart(name).basis, pool_b.chart(name).basis)
        assert np.array_equal(pool_a.chart(name).a, pool_b.chart(name).a)
