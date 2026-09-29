"""Algebraic and numerical contracts for the MultiNLI Helmert probe."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from gen_zero.model.nli_probe import HelmertNLIProbe, NLI_CLASSES


def softmax(scores):
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


def test_helmert_is_orthogonal_projection_onto_contrast_plane():
    h = HelmertNLIProbe().helmert
    projector = np.eye(3) - np.ones((3, 3)) / 3
    assert_allclose(h.T @ h, np.eye(2), atol=1e-15)
    assert_allclose(h @ h.T, projector, atol=1e-15)
    assert_allclose(projector @ projector, projector, atol=1e-15)
    assert_allclose(h.sum(axis=0), 0, atol=1e-15)


def test_simplex_geometry_and_antipodal_sum():
    v = HelmertNLIProbe().frame
    assert_allclose(v @ v.T, 1.5 * np.eye(3) - 0.5, atol=1e-15)
    assert_allclose(v.T @ v, 1.5 * np.eye(2), atol=1e-15)
    for i in range(3):
        others = np.delete(v, i, axis=0)
        assert_allclose(v[i], -others.sum(axis=0), atol=1e-15)
        assert_allclose(np.degrees(np.arccos(others @ v[i])), 120, atol=1e-12)


def test_class_order_and_each_class_wins_its_vertex():
    probe = HelmertNLIProbe()
    assert probe.classes == NLI_CLASSES == ("entailment", "neutral", "contradiction")
    result = probe.project(np.eye(3), np.zeros(3))
    assert_allclose(result.coordinates, np.sqrt(2 / 3) * probe.frame, atol=1e-15)
    assert_allclose(result.predicted_indices, np.arange(3))


def test_projection_preserves_centered_scores_and_all_log_odds():
    rng = np.random.default_rng(41)
    contextual = rng.normal(size=(2, 7, 3))
    prior = rng.normal(size=(2, 7, 3))
    temperature = 0.7
    result = HelmertNLIProbe(temperature).project(contextual, prior)
    residual = contextual - prior
    centered = residual - residual.mean(axis=-1, keepdims=True)
    assert result.coordinates.shape == (2, 7, 2)
    assert result.predicted_indices.shape == (2, 7)
    assert_allclose(result.logits, centered, atol=2e-15)
    assert_allclose(result.probabilities, softmax(residual / temperature), atol=1e-15)
    assert_allclose(np.linalg.norm(result.coordinates, axis=-1),
                    np.linalg.norm(centered, axis=-1), atol=2e-15)
    for i in range(3):
        for j in range(3):
            assert_allclose(np.log(result.probabilities[..., i] / result.probabilities[..., j]),
                            (residual[..., i] - residual[..., j]) / temperature, atol=3e-15)


def test_shared_vocabulary_bias_cancels_and_corrects_raw_prediction():
    signal = np.array([2.0, 0.0, -1.0])
    bias = np.array([-8.0, 15.0, 6.0])
    assert np.argmax(signal + bias) == 1
    probe = HelmertNLIProbe()
    corrected = probe.project(signal + bias, bias)
    assert corrected.predicted_indices.item() == 0
    assert_allclose(corrected.probabilities, softmax(signal), atol=1e-15)
    extra_bias = np.array([30.0, -40.0, 10.0])
    assert_allclose(probe.project(signal + bias + extra_bias, bias + extra_bias).probabilities,
                    corrected.probabilities, atol=1e-15)


def test_log_probabilities_equal_logits_and_probability_ratio():
    contextual = np.array([1.0, -2.0, 3.0])
    prior = np.array([3.0, 0.0, 2.0])
    p, p0 = softmax(contextual), softmax(prior)
    probe = HelmertNLIProbe()
    result = probe.project(np.log(p), np.log(p0))
    expected = p / p0
    expected /= expected.sum()
    assert_allclose(result.probabilities, expected, atol=1e-15)
    assert_allclose(result.probabilities, probe.project(contextual, prior).probabilities, atol=1e-15)


def test_null_context_is_uniform_and_common_offsets_are_irrelevant():
    probe = HelmertNLIProbe()
    prior = np.array([17.0, -23.0, 9.0])
    result = probe.project(prior, prior)
    assert_allclose(result.coordinates, 0, atol=0)
    assert_allclose(result.probabilities, np.full(3, 1 / 3))
    assert result.predicted_indices.shape == ()
    assert result.predicted_indices.item() == 0  # Documented tie convention.
    scores = np.array([2.0, 0.0, -1.0])
    assert_allclose(probe.project(scores + 700, prior - 800).probabilities,
                    probe.project(scores, prior).probabilities, atol=1e-14)


def test_shared_prior_matches_per_example_prior_and_empty_batch():
    probe = HelmertNLIProbe()
    scores = np.arange(18).reshape(2, 3, 3)
    prior = np.array([1, -1, 2])
    assert_allclose(probe.project(scores, prior).probabilities,
                    probe.project(scores, np.broadcast_to(prior, scores.shape)).probabilities)
    assert probe.project(np.empty((0, 3)), prior).probabilities.shape == (0, 3)


@pytest.mark.parametrize("temperature", [1.0, 1e-300, 1e300])
def test_large_logits_and_extreme_temperatures_are_stable(temperature):
    result = HelmertNLIProbe(temperature).project([10000, 0, -10000], [0, 0, 0])
    assert np.all(np.isfinite(result.probabilities))
    assert_allclose(result.probabilities.sum(), 1)
    assert result.predicted_indices.item() == 0


@pytest.mark.parametrize("temperature", [0, -1, np.inf, -np.inf, np.nan])
def test_reject_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        HelmertNLIProbe(temperature)


@pytest.mark.parametrize("bad", [0, [], [1, 2], [1, 2, 3, 4], [np.nan, 0, 1],
                                  [0, np.inf, 1], [1j, 0, 0]])
@pytest.mark.parametrize("argument", ["context", "prior"])
def test_reject_invalid_scores(bad, argument):
    with pytest.raises(ValueError):
        HelmertNLIProbe().project(bad if argument == "context" else [0, 0, 0],
                                  bad if argument == "prior" else [0, 0, 0])


def test_reject_accidental_prior_broadcasting_and_overflow():
    probe = HelmertNLIProbe()
    with pytest.raises(ValueError, match="match"):
        probe.project(np.zeros((2, 3)), np.zeros((1, 3)))
    with pytest.raises(ValueError, match="floating-point range"):
        probe.project([1e308, 0, 0], [-1e308, 0, 0])


def test_no_input_mutation_or_shared_frame_state():
    probe = HelmertNLIProbe()
    scores = np.array([1.0, 2.0, 3.0])
    prior = np.array([3.0, 2.0, 1.0])
    probe.project(scores, prior)
    assert_allclose(scores, [1, 2, 3])
    assert_allclose(prior, [3, 2, 1])
    probe.frame[:] = 0
    assert_allclose(np.linalg.norm(probe.frame, axis=-1), 1)


from gen_zero.model.nli_probe import ContrastiveManifoldProbe


def relational_fixture():
    # Explicit representation contract: coordinates encode quantity, predicate,
    # and unknown attributes. These are synthetic vectors, not encoder evidence.
    p = np.array([[1., 0., 0.], [0., 1., 0.],
                  [1., 0., 0.], [0., 1., 0.],
                  [1., 0., 0.], [0., 1., 0.]])
    h = np.array([[1., 0., 0.], [0., 1., 0.],
                  [0., 0., 1.], [0., 0., 1.],
                  [-1., 0., 0.], [0., -1., 0.]])
    return p, h, np.array([0, 0, 1, 1, 2, 2])


def test_complement_is_isometric_and_orthogonal_to_symmetric_space():
    d = np.array([1., 2., 3.])
    a = ContrastiveManifoldProbe.orthogonal_complement(d)
    assert_allclose(a @ np.tile([2., -1., 4.], 2), 0, atol=1e-14)
    assert_allclose(a @ a, d @ d)
    assert_allclose(a[:3], -a[3:])


def test_facets_preserve_order_and_absolute_difference_is_swap_even():
    p, h = np.array([1., 0.]), np.array([0., 1.])
    features = ContrastiveManifoldProbe.relational_features(p, h)
    assert_allclose(features[:8], np.r_[p, h, abs(p-h), p*h])
    reverse = ContrastiveManifoldProbe.relational_features(h, p)
    assert_allclose(features[4:], reverse[4:])
    assert not np.array_equal(features[:4], reverse[:4])


def test_numeric_and_semantic_refutations_from_vectors_and_three_way_separation():
    p, h, y = relational_fixture()
    probe = ContrastiveManifoldProbe(ridge=1e-8).fit(p, h, y)
    # New vectors near training manifolds, with no text or numeric parser.
    result = probe.project(p * .98, h * .98)
    assert_allclose(result.predicted_indices, y)
    assert_allclose(probe.frame @ probe.frame.T, 1.5 * np.eye(3) - .5, atol=1e-15)
    assert_allclose(probe.project(p, h).coordinates, probe.frame[y], atol=1e-7)
    assert np.all(result.logits[np.arange(6), y] >
                  np.max(np.where(np.eye(3)[y], -np.inf, result.logits), axis=1))


def test_exclusive_predicates_swap_preserves_maximal_vertex_margin():
    p, h, y = relational_fixture()
    # Contradiction is symmetric; entailment in general is not.
    p2 = np.concatenate((p, h[4:]))
    h2 = np.concatenate((h, p[4:]))
    y2 = np.r_[y, [2, 2]]
    probe = ContrastiveManifoldProbe(ridge=1e-9).fit(p2, h2, y2)
    result = probe.project(p2, h2)
    margins = result.logits[4:, 2] - result.logits[4:, :2].max(axis=1)
    # At a unit triangle vertex, squared distance to either other vertex is 3.
    assert_allclose(margins, 3, atol=1e-7)
    assert_allclose(probe.project(h[:2], p[:2]).predicted_indices, [0, 0])


def test_loo_excludes_held_out_label_and_reports_missing_classes():
    p, h, y = relational_fixture()
    result, missing = ContrastiveManifoldProbe.leave_one_out(p, h, y)
    altered = y.copy()
    altered[0] = 2
    changed, _ = ContrastiveManifoldProbe.leave_one_out(p, h, altered)
    assert_allclose(result.logits[0], changed.logits[0])
    assert missing == [[]] * 6
    single, missing = ContrastiveManifoldProbe.leave_one_out(p[:1], h[:1], y[:1])
    assert missing == [[0, 1, 2]]
    assert_allclose(single.probabilities, [[1/3]*3])


@pytest.mark.parametrize('bad', [np.nan, np.inf, -1, 0])
def test_contrastive_invalid_ridge(bad):
    with pytest.raises(ValueError, match='ridge'):
        ContrastiveManifoldProbe(ridge=bad)


def test_contrastive_validation_and_batch_shapes():
    probe = ContrastiveManifoldProbe()
    p, h, y = relational_fixture()
    with pytest.raises(ValueError, match='fit'):
        probe.project(p, h)
    for a, b in [(p, h[:2]), ([np.nan], [0]), ([1j], [0]), ([], [])]:
        with pytest.raises(ValueError):
            probe.relational_features(a, b)
    with pytest.raises(ValueError, match='integer'):
        probe.fit(p, h, y.astype(float))
    probe.fit(p, h, y)
    assert probe.project(p.reshape(2, 3, 3), h.reshape(2, 3, 3)).logits.shape == (2, 3, 3)
    assert probe.project(p[:0], h[:0]).logits.shape == (0, 3)
    assert np.isfinite(probe.relational_features([1e308], [-1e308])).all()


def test_held_out_contrastive_examples_separate_without_own_labels():
    p, h, y = relational_fixture()
    # Independent perturbed exemplars ensure LOO has support for each manifold.
    p = np.concatenate([p, p * .97, p * .94])
    h = np.concatenate([h, h * .97, h * .94])
    labels = np.tile(y, 3)
    result, _ = ContrastiveManifoldProbe.leave_one_out(p, h, labels, ridge=.001)
    assert_allclose(result.predicted_indices, labels)


def test_directional_entailment_can_be_neutral_after_swapping():
    # A more specific representation entails a subset; reverse adds information.
    p = np.array([[1., 1.], [1., 0.], [1., 0.]])
    h = np.array([[1., 0.], [1., 1.], [-1., 0.]])
    probe = ContrastiveManifoldProbe(ridge=1e-8).fit(p, h, np.arange(3))
    assert probe.project(p[0], h[0]).predicted_indices == 0
    assert probe.project(h[0], p[0]).predicted_indices == 1


def test_manifold_logits_are_negative_squared_separation_distances():
    p, h, y = relational_fixture()
    probe = ContrastiveManifoldProbe(ridge=.1).fit(p, h, y)
    result = probe.project(p, h)
    distances = ((result.coordinates[:, None] - probe.frame) ** 2).sum(axis=-1)
    assert_allclose(result.logits, -distances + distances.mean(axis=-1, keepdims=True))
    assert_allclose(result.predicted_indices, distances.argmin(axis=-1))
    assert_allclose(result.probabilities.sum(axis=-1), 1)
