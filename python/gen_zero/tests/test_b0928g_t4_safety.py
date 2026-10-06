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




def test_infeasible_bounds_rejected():
    with pytest.raises(InfeasibleConstraintError):
        project_simplex_with_bounds(np.array([0.3, 0.7]), np.array([0.2, 0.2]))


def test_effect_based_action_gate():
    engine = CpSatFormalEngine(action_effects={'launch_job': 'EXECUTE', 'read_data': 'READ_ONLY'})
    state = {'text': 'unauthorized', 'action_effects': {'launch_job': 'EXECUTE', 'read_data': 'READ_ONLY'}}
    result = engine.verify_and_prune(state, ['launch_job', 'exec_job', 'read_data'])
    assert result['feasible_actions'] == ['read_data']
    assert not CpSatFormalEngine().verify_and_prune(state, ['read_data'])['feasible_actions']


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


