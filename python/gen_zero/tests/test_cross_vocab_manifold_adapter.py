import numpy as np
import pytest

from gen_zero.causal.cross_vocab_manifold_adapter import CrossVocabManifoldAdapter


def test_bidirectional_projection_and_geometry():
    rng = np.random.default_rng(3)
    qt, _ = np.linalg.qr(rng.normal(size=(12, 4)))
    qs, _ = np.linalg.qr(rng.normal(size=(9, 4)))
    adapter = CrossVocabManifoldAdapter(qt, np.ones(12), qs, np.ones(9))
    z = rng.normal(size=(20, 4))
    teacher, student = z @ qt.T + 1, z @ qs.T + 1
    np.testing.assert_allclose(adapter.project(teacher), z, atol=1e-12)
    np.testing.assert_allclose(adapter.map_hidden(teacher), student, atol=1e-12)
    np.testing.assert_allclose(adapter.map_hidden(student, source='student', destination='teacher'), teacher, atol=1e-12)
    assert np.allclose(adapter.projection_energy_ratio(teacher), 1)
    assert adapter.orthogonality_error() < 1e-12
    objective = adapter.alignment_objective(teacher, student)
    assert objective['geodesic_cosine'] == pytest.approx(1)
    assert objective['bures_wasserstein_distance'] == pytest.approx(0, abs=1e-6)
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    np.testing.assert_allclose(adapter.geodesic_cosine(z, z[::-1]),
                               adapter.geodesic_cosine(z @ rotation, z[::-1] @ rotation))
    assert adapter.bures_wasserstein_distance(z, z[::-1]) == pytest.approx(
        adapter.bures_wasserstein_distance(z @ rotation, z[::-1] @ rotation), abs=1e-6)


def test_rejects_uncalibrated_geometry():
    with pytest.raises(ValueError, match='orthonormal'):
        CrossVocabManifoldAdapter(np.ones((5, 2)), np.zeros(5), np.eye(2), np.zeros(2))
    adapter = CrossVocabManifoldAdapter(np.eye(2), np.zeros(2), np.eye(2), np.zeros(2))
    with pytest.raises(ValueError, match='zero'):
        adapter.projection_energy_ratio(np.zeros(2))


def test_qwen_dimension_projection_uses_supplied_basis():
    teacher_basis = np.eye(4096, 64)
    student_basis = np.eye(896, 64)
    adapter = CrossVocabManifoldAdapter(teacher_basis, np.zeros(4096),
                                        student_basis, np.zeros(896))
    hidden = np.zeros((2, 4096))
    hidden[0, 0] = 3
    hidden[1, 63] = -2
    assert adapter.project(hidden).shape == (2, 64)
    np.testing.assert_array_equal(adapter.project(hidden), hidden[:, :64])
    np.testing.assert_array_equal(adapter.map_hidden(hidden), hidden[:, :896])
