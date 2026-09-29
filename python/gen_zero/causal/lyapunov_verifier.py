"""Finite-horizon numerical audits of deterministic, differentiable dynamics.

Callbacks use phase vectors x=(q,p) and numeric conditioning vectors c. No text,
labels or task-specific answers are inspected. Central differences are numerical
estimates, not certified derivative bounds. A sampled trajectory cannot prove a
global Lyapunov theorem or absence of drift between integration nodes.

Full dense Jacobians/SVD cost O(n^2) memory / O(K n^3) time. The optional constant
SPD metric changes coordinates via G=L.T@L; singular values are of L J L^-1.
Dissipation necessarily shrinks volume: admissible distortion must be supplied
as quantitative bounds, not interpreted as exact volume preservation.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

Map = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _array(value, name, shape=None):
    a = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(a)) or (shape is not None and a.shape != shape):
        raise ValueError(f"{name}: invalid shape or nonfinite values")
    return a


def _vector(value, name):
    a = _array(value, name)
    if a.ndim != 1 or not a.size:
        raise ValueError(f"{name}: expected nonempty vector")
    return a.copy()


def _positive(value, name, zero=False):
    if not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name}: invalid bound")


def jacobian(function, x, *, epsilon=1e-5):
    """Scale-aware central differences; invalid evaluations raise, never pass."""
    _positive(epsilon, "epsilon")
    x = _vector(x, "x")
    y = _vector(function(x.copy()), "function output")
    j = np.empty((y.size, x.size))
    for i in range(x.size):
        h = epsilon * max(1., abs(x[i]))
        plus, minus = x.copy(), x.copy()
        plus[i] += h
        minus[i] -= h
        j[:, i] = (_array(function(plus), "output", y.shape)
                   - _array(function(minus), "output", y.shape)) / (2 * h)
    return _array(j, "Jacobian")


@dataclass(frozen=True)
class TrajectoryReport:
    states: np.ndarray
    energies: np.ndarray
    energy_increments: np.ndarray
    cumulative_jacobians: np.ndarray
    sigma_max: np.ndarray
    sigma_min: np.ndarray
    condition_numbers: np.ndarray
    manifold_distances: np.ndarray
    energy_nonincreasing: bool
    distortion_within_bounds: bool
    manifold_within_bound: bool

    @property
    def passed(self):
        return (self.energy_nonincreasing and self.distortion_within_bounds
                and self.manifold_within_bound)


def verify_trajectory(step: Map, energy, x0, c, *, steps: int,
                      manifold_distance, max_manifold_distance: float,
                      min_sigma: float, max_sigma: float,
                      max_condition_number: float, metric=None,
                      energy_atol=0., energy_rtol=0., epsilon=1e-5):
    """Audit every prefix J_k=D F(x_{k-1}) ... D F(x_0), including J_0=I.

    manifold_distance(x,c) must independently measure distance to the intended
    manifold; this cannot be inferred from energy or singular values. Bounds
    apply at all sampled nodes, in the supplied constant metric. Tolerances on
    energy are explicit; defaults require literal nonincrease. No early exit.
    """
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
    for name, value in (("min_sigma", min_sigma), ("max_sigma", max_sigma),
                        ("max_condition_number", max_condition_number)):
        _positive(value, name)
    if min_sigma > 1 or max_sigma < 1 or max_condition_number < 1:
        raise ValueError("bounds must include the identity prefix")
    for name, value in (("energy_atol", energy_atol), ("energy_rtol", energy_rtol),
                        ("max_manifold_distance", max_manifold_distance)):
        _positive(value, name, zero=True)
    x, c = _vector(x0, "x0"), _vector(c, "c")
    n = x.size
    g = np.eye(n) if metric is None else _array(metric, "metric", (n, n))
    if not np.allclose(g, g.T, rtol=0, atol=1e-12):
        raise ValueError("metric must be symmetric")
    try:
        l = np.linalg.cholesky(g).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("metric must be positive definite") from exc
    inverse_l = np.linalg.solve(l, np.eye(n))
    states, energies, distances, products, spectra = [], [], [], [], []
    product = np.eye(n)
    for k in range(steps + 1):
        states.append(x.copy())
        energies.append(float(_array(energy(x.copy(), c.copy()), "energy", ())))
        distance = float(_array(manifold_distance(x.copy(), c.copy()), "distance", ()))
        if distance < 0:
            raise ValueError("manifold distance must be nonnegative")
        distances.append(distance)
        products.append(product.copy())
        spectra.append(np.linalg.svd(_array(l @ product @ inverse_l, "product"),
                                    compute_uv=False))
        if k < steps:
            local = jacobian(lambda z: step(z, c.copy()), x, epsilon=epsilon)
            if local.shape != (n, n):
                raise ValueError("step must preserve phase dimension")
            product = _array(local @ product, "cumulative Jacobian")
            x = _array(step(x.copy(), c.copy()), "step", (n,)).copy()
    energies, distances, spectra = map(np.asarray, (energies, distances, spectra))
    hi, lo = spectra[:, 0], spectra[:, -1]
    condition = np.divide(hi, lo, out=np.full_like(hi, np.inf), where=lo > 0)
    delta = np.diff(energies)
    return TrajectoryReport(np.array(states), energies, delta, np.array(products),
                            hi, lo, condition, distances,
                            bool(np.all(delta <= energy_atol + energy_rtol * np.abs(energies[:-1]))),
                            bool(np.all((lo >= min_sigma) & (hi <= max_sigma)
                                        & (condition <= max_condition_number))),
                            bool(np.all(distances <= max_manifold_distance)))


@dataclass(frozen=True)
class AttractorReport:
    endpoints: np.ndarray
    equilibrium_residuals: np.ndarray
    fixed_point_residuals: np.ndarray
    spectral_radii: np.ndarray
    reference_errors: np.ndarray
    sensitivity_sigma_min: np.ndarray
    pairwise_gains: np.ndarray
    passed: bool


def verify_conditioned_attractors(step: Map, vector_field: Map, initial_states,
                                  conditions, reference: Callable, *, steps: int,
                                  residual_tol: float, reference_tol: float,
                                  min_response: float, stability_margin=1e-4,
                                  epsilon=1e-5):
    """Check sampled equilibria from every initial state at every condition.

    Require continuous AND discrete equilibrium, local discrete stability,
    independent reference agreement, pairwise response and full column rank of
    dx*/dc=-(df/dx)^-1 df/dc. This rejects fixed wells and ignored condition
    directions. Injectivity is a caller-requested contract: many real problems
    legitimately share solutions. Finite samples do not prove global injectivity,
    basin coverage, semantic correctness, or continuous-time attraction.
    """
    xs, cs = _array(initial_states, "initial_states"), _array(conditions, "conditions")
    if xs.ndim != 2 or cs.ndim != 2 or min(xs.shape) < 1 or cs.shape[0] < 2 or cs.shape[1] < 1:
        raise ValueError("require initial states and at least two condition vectors")
    if np.unique(cs, axis=0).shape[0] != cs.shape[0]:
        raise ValueError("conditions must be distinct")
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
    for name, value in (("residual_tol", residual_tol), ("reference_tol", reference_tol),
                        ("min_response", min_response), ("stability_margin", stability_margin)):
        _positive(value, name)
    if stability_margin >= 1:
        raise ValueError("stability_margin must be less than one")
    endpoints, residuals, fixed, radii, errors, sensitivities = [], [], [], [], [], []
    for c in cs:
        target = _array(reference(c.copy()), "reference", (xs.shape[1],))
        group = []
        for start in xs:
            x = start.copy()
            for _ in range(steps):
                x = _array(step(x.copy(), c.copy()), "step", start.shape).copy()
            group.append(x)
            residuals.append(np.linalg.norm(_array(vector_field(x.copy(), c.copy()), "field", x.shape)))
            fixed.append(np.linalg.norm(_array(step(x.copy(), c.copy()), "step", x.shape) - x))
            a = jacobian(lambda z: step(z, c.copy()), x, epsilon=epsilon)
            radii.append(np.max(np.abs(np.linalg.eigvals(a))))
            errors.append(np.linalg.norm(x - target))
            fx = jacobian(lambda z: vector_field(z, c.copy()), x, epsilon=epsilon)
            fc = jacobian(lambda v: vector_field(x.copy(), v), c, epsilon=epsilon)
            # Singular equilibrium derivative means sensitivity is not certified.
            try:
                response = np.linalg.solve(fx, fc)
                smin = (np.linalg.svd(response, compute_uv=False)[-1]
                        if cs.shape[1] <= xs.shape[1] else 0.)
            except np.linalg.LinAlgError:
                smin = 0.
            sensitivities.append(smin)
        endpoints.append(group)
    endpoints = np.asarray(endpoints)
    gains = [np.linalg.norm(a - b) / np.linalg.norm(cs[i] - cs[j])
             for i in range(len(cs)) for j in range(i)
             for a in endpoints[i] for b in endpoints[j]]
    residuals, fixed, radii, errors, sensitivities, gains = map(
        np.asarray, (residuals, fixed, radii, errors, sensitivities, gains))
    passed = bool(np.all(residuals <= residual_tol) and np.all(fixed <= residual_tol)
                  and np.all(radii <= 1 - stability_margin)
                  and np.all(errors <= reference_tol)
                  and np.all(sensitivities >= min_response) and np.all(gains >= min_response))
    return AttractorReport(endpoints, residuals, fixed, radii, errors, sensitivities, gains, passed)
