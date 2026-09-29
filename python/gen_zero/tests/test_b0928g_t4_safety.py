import numpy as np
import pytest

from gen_zero.capability.descriptor import CapabilityDescriptor, CapabilityType
from gen_zero.provenance.permission_arbiter import PermissionArbiter
from gen_zero.provenance.auditor import DecisionProvenanceAuditor
from gen_zero.gate.differentiable_safety_layer import (
    DifferentiableSafetyLayer, InfeasibleConstraintError, project_simplex_with_bounds,
)
from gen_zero.planner.engines.cpsat_formal_engine import CpSatFormalEngine
from gen_zero.gate.cpsat_formal_solver import CPSATFormalSolver


def test_empty_plan_not_authorized():
    verdict = PermissionArbiter().evaluate_plan([])
    assert not verdict.is_authorized and verdict.violations


def test_simplex_vjp_and_zero_upstream():
    layer = DifferentiableSafetyLayer(3)
    layer.forward(np.array([0.2, 0.3, 0.5]))
    assert np.allclose(layer.backward(np.ones(3)), 0, atol=1e-8)
    assert np.array_equal(layer.backward(np.zeros(3)), np.zeros(3))


def test_redundant_interior_constraint_preserves_finite_difference_gradient():
    matrix = np.array([[1.0, 0.0]])
    rhs = np.array([1.0])
    x0 = np.array([0.9999, 0.0001])
    layer = DifferentiableSafetyLayer(2, matrix, rhs)
    result = layer.forward(x0, np.zeros(2))
    analytic = layer.backward(np.array([1.0, 0.0]))
    epsilon = 1e-6
    finite_difference = np.empty(2)
    for i in range(2):
        delta = np.eye(2)[i] * epsilon
        plus = layer.forward(x0 + delta, np.zeros(2)).projected_distribution[0]
        minus = layer.forward(x0 - delta, np.zeros(2)).projected_distribution[0]
        finite_difference[i] = (plus - minus) / (2 * epsilon)
    assert result.active_constraints_count == 0
    assert np.linalg.norm(analytic) > 0
    assert np.max(np.abs(analytic - finite_difference)) < 1e-4
    assert np.allclose(analytic, [0.5, -0.5], atol=1e-4)


def test_constraint_pair_and_dimensions_rejected():
    matrix = np.array([[1.0, 0.0]])
    rhs = np.array([1.0])
    for kwargs in ({'constraint_matrix': matrix}, {'constraint_rhs': rhs}):
        with pytest.raises(ValueError, match='must both be provided or both be None'):
            DifferentiableSafetyLayer(2, **kwargs)
    for bad_matrix, bad_rhs in ((np.ones((1, 3)), rhs), (matrix, np.ones(2))):
        with pytest.raises(ValueError, match='dimensions must match'):
            DifferentiableSafetyLayer(2, bad_matrix, bad_rhs)
        with pytest.raises(ValueError, match='dimensions must match'):
            DifferentiableSafetyLayer(2).update_constraints(bad_matrix, bad_rhs)


def test_tiny_mu_utility_gradient_fails_closed():
    torch = pytest.importorskip('torch')
    from gen_zero.gate.differentiable_safety_layer import PyTorchDifferentiableSafetyModule
    module = PyTorchDifferentiableSafetyModule(DifferentiableSafetyLayer(2, mu=1e-310))
    x = torch.tensor([0.5, 0.5], dtype=torch.float64, requires_grad=True)
    utility = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    projected = module(x, utility)
    assert torch.isfinite(projected).all()
    with pytest.raises(ValueError, match='Non-finite utility gradient'):
        projected.backward(torch.tensor([1.0, 0.0], dtype=torch.float64))


def test_infeasible_bounds_rejected():
    with pytest.raises(InfeasibleConstraintError):
        project_simplex_with_bounds(np.array([0.3, 0.7]), np.array([0.2, 0.2]))


def test_effect_based_action_gate():
    engine = CpSatFormalEngine(action_effects={'launch_job': 'EXECUTE', '读取': 'READ_ONLY'})
    state = {'text': 'unauthorized', 'action_effects': {'launch_job': 'EXECUTE', '读取': 'READ_ONLY'}}
    result = engine.verify_and_prune(state, ['launch_job', '执行', '读取'])
    assert result['feasible_actions'] == ['读取']
    assert not CpSatFormalEngine().verify_and_prune(state, ['读取'])['feasible_actions']


def test_intermediate_overflow_and_exact_hard_mask():
    layer = DifferentiableSafetyLayer(2, mu=1e-300)
    with pytest.raises(ValueError, match='Non-finite intermediate'):
        layer.forward(np.array([0.5, 0.5]), np.array([1e300, 0.0]))
    layer = DifferentiableSafetyLayer(2, np.array([[1.0, 0.0]]), np.array([1.0 - 1e-6]), tolerance=1e-3)
    result = layer.forward(np.array([0.9, 0.1]))
    assert result.projected_distribution[0] == 0


def test_canonical_types_and_structured_fields():
    auditor = DecisionProvenanceAuditor('secret')
    digest = auditor.compute_context_digest
    assert digest('x') != digest({'__str__': 'x'})
    assert digest(np.array([1], dtype=np.uint8)) != digest({'__ndarray__': {'dtype': '|u1', 'shape': [1], 'bytes': '01'}})
    a = np.array([(1,)], dtype=[('first', 'i4')])
    b = np.array([(1,)], dtype=[('other', 'i4')])
    assert digest(a) != digest(b)


def test_descriptor_bytearray_frozen():
    original = bytearray(b'abc')
    desc = CapabilityDescriptor('tool:x', CapabilityType.TOOL, 'x', metadata={'raw': original})
    before = desc.digest()
    original[0] = 0
    assert desc.metadata['raw'] == b'abc'
    assert desc.digest() == before


def test_review_security_regressions():
    auditor = DecisionProvenanceAuditor('secret')
    assert auditor.compute_context_digest(np.array(1)) != auditor.compute_context_digest(np.array([1]))
    base = dict(capability_id='tool:x', capability_type=CapabilityType.TOOL, description='x')
    assert CapabilityDescriptor(**base, metadata={'raw': b'a'}).digest() != CapabilityDescriptor(
        **base, metadata={'raw': {'__bytes_hex__': '61'}}).digest()
    source = np.array([1, 2])
    descriptor = CapabilityDescriptor(**base, metadata={'values': source})
    before = descriptor.digest()
    source[0] = 99
    assert descriptor.digest() == before and descriptor.metadata['values'] == (1, 2)
    verdict = CPSATFormalSolver().solve_safest_optimal_action({'a': 1.0}, forbidden_actions={'a'})
    assert not verdict.is_safe and verdict.fallback_used


def test_torch_batch_and_utility_vjp():
    torch = pytest.importorskip('torch')
    from gen_zero.gate.differentiable_safety_layer import PyTorchDifferentiableSafetyModule
    layer = PyTorchDifferentiableSafetyModule(DifferentiableSafetyLayer(3))
    x = torch.tensor([[0.2, 0.3, 0.5], [1.2, -0.1, -0.1]], dtype=torch.float64, requires_grad=True)
    u = torch.zeros_like(x, requires_grad=True)
    output = layer(x, u)
    output.backward(torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=torch.float64))
    assert np.allclose(x.grad[0].numpy(), [2/3, -1/3, -1/3], atol=1e-8)
    assert np.allclose(u.grad.numpy(), x.grad.numpy(), atol=1e-8)
    eps = 1e-6
    probe = np.array([0.2, 0.3, 0.5])
    weights = np.array([1., 0., 0.])
    finite_difference = []
    for j in range(3):
        delta = np.zeros(3)
        delta[j] = eps
        plus = DifferentiableSafetyLayer(3).forward(probe, delta).projected_distribution
        minus = DifferentiableSafetyLayer(3).forward(probe, -delta).projected_distribution
        finite_difference.append(weights @ (plus - minus) / (2 * eps))
    assert np.allclose(u.grad[0].numpy(), finite_difference, atol=1e-7)
    tiny = DifferentiableSafetyLayer(3)
    tiny.forward(np.array([1e-9, 0.4, 0.6 - 1e-9]))
    assert np.allclose(tiny.backward(np.array([1., 0., 0.])), [2/3, -1/3, -1/3], atol=1e-7)
