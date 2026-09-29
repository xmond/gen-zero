"""Executable analytic systems and counterexamples, not mocked verification."""
import numpy as np
import pytest
from scipy.linalg import expm

from gen_zero.causal.lyapunov_verifier import (
    jacobian, verify_trajectory, verify_conditioned_attractors,
)
from gen_zero.causal.symplectic_thinking import NeuralPotential, SymplecticThinker


def audit(step, x=(1., 0.), **kwargs):
    options = dict(steps=8, manifold_distance=lambda x, c: 0.,
                   max_manifold_distance=0., min_sigma=1e-8, max_sigma=1.01,
                   max_condition_number=100., energy_atol=1e-12)
    options.update(kwargs)
    return verify_trajectory(step, lambda x, c: x @ x / 2,
                             x, [0.], **options)


def oscillator(coupling=None):
    # Nonidentity mass; H=(q-c)^T K(q-c)/2 + p^T M^-1 p/2.
    mass = np.diag([2., 3.])
    stiffness = np.array([[2., .3], [.3, 1.]])
    a = np.block([[np.zeros((2, 2)), np.linalg.inv(mass)],
                  [-stiffness, -1.2 * np.eye(2)]])
    flow = expm(.1 * a)
    b = np.eye(2) if coupling is None else coupling
    target = lambda c: np.r_[b @ c, np.zeros(2)]
    step = lambda x, c: target(c) + flow @ (x - target(c))
    field = lambda x, c: a @ (x - target(c))
    return step, field, target, a, stiffness, mass


def attractors(step, field, target, **kwargs):
    options = dict(steps=600, residual_tol=1e-8, reference_tol=1e-8,
                   min_response=.2)
    options.update(kwargs)
    return verify_conditioned_attractors(step, field,
        [[2., -1., .4, .7], [-2., 3., -.4, -.7]],
        [[0., 0.], [1., 0.], [0., 1.], [-.7, .3]], target, **options)


def test_cumulative_product_order_and_nonlinear_derivative():
    def step(x, c):
        return np.array([.8 * x[0] + .1 * x[1] ** 2, .7 * x[1] + .05 * x[0] ** 2])
    report = audit(step, x=(.3, .7), max_sigma=2.)
    product = np.eye(2)
    for k, x in enumerate(report.states[:-1], 1):
        product = np.array([[.8, .2*x[1]], [.1*x[0], .7]]) @ product
        np.testing.assert_allclose(report.cumulative_jacobians[k], product, atol=1e-10)
    def whole(x):
        for _ in range(8):
            x = step(x, None)
        return x
    np.testing.assert_allclose(report.cumulative_jacobians[-1], jacobian(whole, [.3, .7]), atol=1e-9)


def test_volume_preserving_stretch_is_rejected_even_on_zero_energy_orbit():
    a = np.diag([2., .5])
    r = audit(lambda x, c: a @ x, x=(0., 0.))
    assert r.energy_nonincreasing
    assert not r.distortion_within_bounds
    np.testing.assert_allclose(r.sigma_max, 2. ** np.arange(9), rtol=1e-10)
    np.testing.assert_allclose(r.sigma_min, .5 ** np.arange(9), rtol=1e-10)
    np.testing.assert_allclose(np.linalg.det(r.cumulative_jacobians), 1.)


def test_dissipative_exact_flow_energy_metric_and_reference_manifold():
    step, _, target, a, k, m = oscillator()
    metric = np.block([[k, np.zeros((2, 2))], [np.zeros((2, 2)), np.linalg.inv(m)]])
    c = np.array([.5, -.3])
    # Full phase space is the intended invariant manifold in this test.
    r = verify_trajectory(step, lambda x, c: (x-target(c)) @ metric @ (x-target(c))/2,
        [1., 2., .3, -.4], c, steps=20, metric=metric,
        manifold_distance=lambda x, c: 0., max_manifold_distance=0.,
        min_sigma=.03, max_sigma=1.0000001, max_condition_number=40.)
    assert r.passed
    np.testing.assert_allclose(r.cumulative_jacobians[-1], expm(2*a), atol=1e-9)
    assert np.all(r.energy_increments <= 0)


def test_contraction_does_not_hide_drift():
    r = audit(lambda x, c: .5*x + np.array([0., .1]),
              manifold_distance=lambda x, c: abs(x[1]), max_manifold_distance=.01)
    assert r.distortion_within_bounds
    assert not r.manifold_within_bound
    assert not r.passed


def test_nontrivial_invariant_manifold():
    r = audit(lambda x, c: .8*x, manifold_distance=lambda x, c: abs(x[1]))
    assert r.passed
    np.testing.assert_allclose(r.sigma_max, .8 ** np.arange(9), atol=1e-10)


def test_discrete_damping_does_not_guarantee_energy_descent():
    thinker = SymplecticThinker(NeuralPotential(dim=1, hidden_dim=1, scale=0.),
                                dt=1., gamma=.01)
    def step(x, c):
        q, p, _ = thinker.step(x[:1], x[1:])
        return np.r_[q, p]
    r = audit(step, x=(0., 1.), steps=1, max_sigma=100.)
    assert r.energy_increments[0] > 0
    assert not r.energy_nonincreasing


def test_conditioned_attractors_follow_all_condition_directions():
    step, field, target, *_ = oscillator()
    r = attractors(step, field, target)
    assert r.passed
    np.testing.assert_allclose(r.sensitivity_sigma_min, 1., atol=1e-8)


@pytest.mark.parametrize('coupling', [np.zeros((2, 2)), np.diag([1., 0.])])
def test_fixed_well_and_ignored_condition_dimension_fail(coupling):
    step, field, target, *_ = oscillator(coupling)
    r = attractors(step, field, target)
    assert np.max(r.reference_errors) < 1e-8
    assert not r.passed
    assert np.max(r.sensitivity_sigma_min) < 1e-8


def test_wrong_manifold_and_insufficient_integration_fail():
    step, field, target, *_ = oscillator()
    assert not attractors(step, field, lambda c: target(c) + .1).passed
    assert not attractors(step, field, target, steps=1).passed


def test_continuous_residual_rejects_spurious_discrete_fixed_point():
    step, field, target, *_ = oscillator()
    r = attractors(step, lambda x, c: field(x, c) + 1., target)
    assert not r.passed
    assert np.min(r.equilibrium_residuals) > 1.


def test_unstable_fixed_point_is_not_an_attractor():
    r = verify_conditioned_attractors(lambda x, c: c + 2*(x-c),
        lambda x, c: x-c, [[0.]], [[0.], [1e-10]], lambda c: c,
        steps=1, residual_tol=1e-8, reference_tol=1e-8, min_response=.1)
    assert not r.passed
    assert np.all(r.spectral_radii > 1)


@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_nonfinite_callbacks_fail_closed(bad):
    with pytest.raises(ValueError):
        audit(lambda x, c: np.full_like(x, bad))
    with pytest.raises(ValueError):
        audit(lambda x, c: .5*x, manifold_distance=lambda x, c: bad)


@pytest.mark.parametrize('options', [dict(steps=0), dict(steps=1.5),
    dict(metric=[[1., 2.], [0., 1.]]), dict(metric=[[1., 0.], [0., -1.]]),
    dict(energy_atol=-1.), dict(max_sigma=np.nan), dict(min_sigma=0.),
    dict(manifold_distance=lambda x, c: -1.)])
def test_invalid_contracts_raise(options):
    with pytest.raises(ValueError):
        audit(lambda x, c: .5*x, **options)


def test_singular_map_reports_collapse():
    r = audit(lambda x, c: np.zeros_like(x))
    assert not r.distortion_within_bounds
    assert np.isinf(r.condition_numbers[-1])


def test_existing_unconditioned_thinker_fails_condition_response():
    thinker = SymplecticThinker(NeuralPotential(dim=1, hidden_dim=2), dt=.1, gamma=1.)
    def step(x, c):
        q, p, _ = thinker.step(x[:1], x[1:])
        return np.r_[q, p]
    def field(x, c):
        return np.r_[x[1:], -thinker.potential.grad(x[:1]) - x[1:]]
    r = verify_conditioned_attractors(step, field, [[1., .2]], [[0.], [1.]],
        lambda c: np.r_[c, 0.], steps=600, residual_tol=1e-8,
        reference_tol=1e-8, min_response=.1)
    assert not r.passed
    np.testing.assert_array_equal(r.sensitivity_sigma_min, 0.)
    np.testing.assert_array_equal(r.endpoints[0], r.endpoints[1])
