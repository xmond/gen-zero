"""Tests for the bidirectional counterfactual constraint verifier.

Every assertion is grounded in pure real-vector geometry: there is no text,
word list, regex, or dataset-specific parser anywhere in the verifier, so the
inputs are synthetic representation vectors that encode quantity, sign, and
orthogonality directly.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from gen_zero.causal.counterfactual_constraint_verifier import (
    NLI_CLASSES,
    BidirectionalVerdict,
    CounterfactualConstraintVerifier,
)

ENT, NEU, CON = range(3)  # entailment, neutral, contradiction


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


# --------------------------------------------------------------------------
# Helmert simplex geometry
# --------------------------------------------------------------------------


def test_helmert_frame_is_orthonormal_equilateral_triangle():
    v = CounterfactualConstraintVerifier().frame
    assert_allclose(v @ v.T, 1.5 * np.eye(3) - 0.5, atol=1e-15)  # unit, 120-degree
    assert_allclose(v.sum(axis=0), 0, atol=1e-15)               # zero-mean vertices
    assert v.shape == (3, 2)
    assert NLI_CLASSES == ("entailment", "neutral", "contradiction")


def test_frame_returned_independent_copy():
    probe = CounterfactualConstraintVerifier()
    f = probe.frame
    f[:] = 0
    assert np.linalg.norm(probe.frame, axis=-1).tolist() == [1.0, 1.0, 1.0]


# --------------------------------------------------------------------------
# Bidirectional consistency: bounds and identities
# --------------------------------------------------------------------------


def test_forward_support_is_cosine_and_bounded():
    v = CounterfactualConstraintVerifier()
    rng = np.random.default_rng(7)
    for _ in range(20):
        p = rng.normal(size=5)
        h = rng.normal(size=5)
        s = v.forward_entailment_support(p, h)
        assert -1.0 - 1e-12 <= s <= 1.0 + 1e-12
        assert_allclose(s, np.dot(p, h) / (np.linalg.norm(p) * np.linalg.norm(h)), atol=1e-12)


@pytest.mark.parametrize("angle_deg,cos", [(0.0, 1.0), (60.0, 0.5), (90.0, 0.0), (135.0, -0.7071), (180.0, -1.0)])
def test_bidirectional_components_partition_the_angle(angle_deg, cos):
    v = CounterfactualConstraintVerifier()
    p = np.array([1.0, 0.0])
    h = np.array([np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))])
    s = v.forward_entailment_support(p, h)
    n = v.backward_necessity(p, h)
    r = v.counterfactual_residual(p, h)
    o = v.orthogonal_mass(p, h)
    # Necessity is the positive part, residual the negative part, both in [0,1].
    assert_allclose(n, max(0.0, s), atol=1e-12)
    assert_allclose(r, max(0.0, -s), atol=1e-12)
    assert 0.0 <= n <= 1.0 and 0.0 <= r <= 1.0
    assert_allclose(o, np.sin(np.radians(angle_deg)), atol=1e-9)
    assert_allclose(o * o + s * s, 1.0, atol=1e-12)
    # P ==> H requires R_contra == 0: at angle 0 (full entailment) residual is zero.
    if angle_deg == 0.0:
        assert r == 0.0


def test_necessity_and_residual_are_mutually_exclusive():
    v = CounterfactualConstraintVerifier()
    rng = np.random.default_rng(11)
    for _ in range(20):
        p = rng.normal(size=4)
        h = rng.normal(size=4)
        n = v.backward_necessity(p, h)
        r = v.counterfactual_residual(p, h)
        # Exactly one of (necessity, residual) is positive; both never fire together.
        assert (n == 0.0) != (r == 0.0) or (n == 0.0 and r == 0.0)


# --------------------------------------------------------------------------
# Three-way clean separation: each class lands at its own simplex vertex
# --------------------------------------------------------------------------


def test_three_way_clean_classification_and_wide_margin():
    v = CounterfactualConstraintVerifier()
    rng = np.random.default_rng(3)
    # Clean classes in a 16-D space so nothing is axis-aligned by coincidence.
    basis = np.linalg.qr(rng.normal(size=(16, 4)))[0][:, :4]
    p_dir, e_dir, n_dir, c_dir = basis.T
    premise = p_dir
    pairs = [
        (premise, premise + 0.0 * e_dir, ENT),                       # entailment: H == P
        (premise, 2.0 * n_dir, NEU),                                 # neutral: H orthogonal mass
        (premise, -premise, CON),                                    # contradiction: H = -P
    ]
    verdicts = [v.verify(p, h) for p, h, _ in pairs]
    for (p, h, y), ver in zip(pairs, verdicts):
        assert ver.predicted_index == y, (y, ver.predicted_index, ver.probabilities)
        assert ver.margin > 0.2  # clean vertex margin
        assert np.argmax(ver.probabilities) == y
    # The three clean coordinates sit at 120-degree-separated simplex directions.
    coords = np.array([ver.coordinates for ver in verdicts])
    cos = np.array([
        [np.dot(coords[i], coords[j]) / (np.linalg.norm(coords[i]) * np.linalg.norm(coords[j]))
         for j in range(3)] for i in range(3)
    ])
    off = cos[~np.eye(3, dtype=bool)]
    assert_allclose(off, -0.5, atol=1e-9)  # pairwise 120 degrees


def test_orthogonal_mass_rescues_ambiguous_cosine_neutral():
    """A forward-only cosine line is fragile here; the bidirectional axis widens it.

    cos(P, H) = 0.49 sits 0.01 from a typical 0.5 entailment threshold, so a
    one-direction line classifies it neutral with a ~0.01 margin. The
    bidirectional verifier uses the orthogonal-mass axis (O ~ 0.87), placing the
    pair firmly at the neutral vertex with a ~0.34 probability margin.
    """
    v = CounterfactualConstraintVerifier()
    p = np.array([1.0, 0.0])
    h = np.array([0.49, np.sqrt(1.0 - 0.49 ** 2)])  # unit vector, cos = 0.49
    ver = v.verify(p, h)
    assert ver.forward_support == pytest.approx(0.49, abs=1e-12)
    assert 0.0 < ver.forward_support < 0.5            # fragile on a cosine line
    assert ver.predicted_index == NEU
    assert ver.margin > 0.3                          # robust on the simplex
    # The forward-only margin (distance to the 0.5 threshold) is tiny by contrast.
    forward_only_margin = abs(0.49 - 0.5)
    assert forward_only_margin < 0.05
    assert ver.margin > 6 * forward_only_margin


# --------------------------------------------------------------------------
# Adversarial counterfactual repulsion operator (VitaminC fine flips)
# --------------------------------------------------------------------------


def _negation_flip_pair():
    """VitaminC-style negation flip drowned by lexical bulk.

    Premise asserts a fact (dim 3 = +0.5); hypothesis negates it (dim 3 = -0.5)
    while sharing the lexical bulk (dims 1-2). Forward cosine stays high (0.778),
    so a forward-only probe wrongly calls entailment. The counterfactual
    reference pair holds the shared bulk alone (no fact), so it carries
    information the premise itself does not.
    """
    premise = np.array([1.0, 1.0, 0.5])
    hyp_contra = np.array([1.0, 1.0, -0.5])
    cf_premise = np.array([1.0, 1.0, 0.0])
    cf_hypothesis = np.array([1.0, 1.0, 0.0])
    return premise, hyp_contra, cf_premise, cf_hypothesis


def test_repulsion_excludes_shared_bulk_and_is_isometric():
    v = CounterfactualConstraintVerifier()
    delta = np.array([2.0, 2.0, 1.0])
    bulk = np.array([1.0, 1.0, 0.0])
    delta_perp, penalty = v._exclude_along(delta, bulk)
    # The excluded residual is orthogonal to the bulk.
    assert_allclose(np.dot(delta_perp, bulk), 0.0, atol=1e-12)
    # Exclusion is isometric: ||delta||^2 == ||perp||^2 + ||parallel||^2.
    parallel = delta - delta_perp
    assert_allclose(np.linalg.norm(delta) ** 2,
                    np.linalg.norm(delta_perp) ** 2 + np.linalg.norm(parallel) ** 2, atol=1e-12)
    assert 0.0 <= penalty <= 1.0
    # cos^2(delta, bulk): delta has a large bulk component -> penalty substantial.
    expected = (np.dot(delta, bulk) / (np.linalg.norm(delta) * np.linalg.norm(bulk))) ** 2
    assert_allclose(penalty, expected, atol=1e-12)


def test_repulsion_uninformative_reference_adds_no_evidence():
    """A reference with cf_premise == premise cannot expose any flip.

    The bulk is cf_premise itself, and delta_perp is orthogonal to the bulk,
    so with cf_premise == premise delta_perp is orthogonal to the premise and
    negation_alignment is exactly 0. This is forced by rotation equivariance:
    with cf_premise == cf_hypothesis == premise, the only rotation-invariant
    data is the Gram matrix of (premise, hypothesis), the same data the
    forward-only check sees. An earlier version "flipped" this case to
    contradiction only because ``p0 - p0.mean()`` treated the all-ones axis
    as the bulk, which is coordinate-dependent.
    """
    v = CounterfactualConstraintVerifier()
    for premise, hypothesis in [
        (np.array([1.0, 1.0, 0.5]), np.array([1.0, 1.0, -0.5])),
        (np.array([1.0, 1.0, 3.0]), np.array([1.0, 1.0, 2.0])),
    ]:
        forward = v.verify(premise, hypothesis)
        with_ref = v.verify(premise, hypothesis, cf_premise=premise, cf_hypothesis=premise)
        assert with_ref.negation_alignment == 0.0
        assert with_ref.repulsion_penalty > 0.0  # the flip is masked by the bulk
        assert_allclose(with_ref.probabilities, forward.probabilities, atol=1e-12)
        assert with_ref.predicted_index == forward.predicted_index == ENT


def test_repulsion_flips_wrong_entailment_to_correct_contradiction():
    """VitaminC-style numeric flip (3 -> 2) against a bulk-only reference pair.

    delta = (h - h0) - (p - p0) = [0, 0, -1] is orthogonal to the bulk
    p0 = [1, 1, 0], so it survives exclusion intact and
    negation_alignment = 3 / sqrt(11) ~ 0.905 exactly.
    """
    v = CounterfactualConstraintVerifier()
    premise = np.array([1.0, 1.0, 3.0])
    hypothesis = np.array([1.0, 1.0, 2.0])          # number changed
    cf_p = np.array([1.0, 1.0, 0.0])
    cf_h = np.array([1.0, 1.0, 0.0])               # shared bulk, no number

    wrong = v.verify(premise, hypothesis)
    assert wrong.predicted_index == ENT            # forward cosine ~0.985 masks the flip
    right = v.verify(premise, hypothesis, cf_premise=cf_p, cf_hypothesis=cf_h)
    assert right.negation_alignment == pytest.approx(3.0 / np.sqrt(11.0), abs=1e-12)
    assert right.predicted_index == CON           # corrected label
    assert right.probabilities[CON] > wrong.probabilities[CON] + 0.2
    assert right.probabilities[ENT] < wrong.probabilities[ENT]
    assert right.margin > 0.05


def test_repulsion_widens_margin_for_negation_flip():
    """Negation flip against a bulk-only reference: evidence moves off entailment.

    delta = [0, 0, -1], so negation_alignment = 0.5 / 1.5 = 1/3 exactly. That is
    enough to move the call off entailment (to neutral) but not to
    contradiction: the flipped component is small next to the premise norm.
    """
    v = CounterfactualConstraintVerifier()
    premise, hyp_contra, cf_p, cf_h = _negation_flip_pair()
    wrong = v.verify(premise, hyp_contra)
    assert wrong.forward_support == pytest.approx(1.75 / 2.25, abs=1e-12)  # 0.778
    assert wrong.predicted_index == ENT
    right = v.verify(premise, hyp_contra, cf_premise=cf_p, cf_hypothesis=cf_h)
    assert right.negation_alignment == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert right.predicted_index != ENT
    assert right.probabilities[ENT] < wrong.probabilities[ENT]
    assert right.probabilities[CON] > wrong.probabilities[CON]


def test_repulsion_is_silent_when_reference_matches_the_pair():
    """Control: no flip relative to the reference means no negation evidence."""
    v = CounterfactualConstraintVerifier()
    premise = np.array([1.0, 1.0, 0.5])
    forward = v.verify(premise, premise)
    with_ref = v.verify(premise, premise, cf_premise=[1.0, 1.0, 0.0], cf_hypothesis=[1.0, 1.0, 0.0])
    assert with_ref.negation_alignment == 0.0
    assert_allclose(with_ref.probabilities, forward.probabilities, atol=1e-12)
    assert with_ref.predicted_index == ENT


def test_repulsion_zero_when_pair_already_separated():
    """If the perturbation is already orthogonal to the bulk, no flip is masked."""
    v = CounterfactualConstraintVerifier()
    p = np.array([1.0, 0.0])
    h = np.array([1.0, 1.0])             # entail-ish
    cf_p = np.array([1.0, 0.0])
    cf_h = np.array([1.0, 0.0])           # reference identical to premise
    _, penalty, _ = v.repulsion(p, h, cf_p, cf_h)
    # delta = (h - cf_h) - (p - cf_p) = [0,1] - [0,0] = [0,1]; bulk ~ [1,0]; orthogonal.
    assert penalty == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# Language-agnosticism and zero rule matching
# --------------------------------------------------------------------------


def test_dimension_invariance_and_high_dim_random():
    """Geometry is identical up to an orthogonal change of basis and works in high-D."""
    v = CounterfactualConstraintVerifier()
    rng = np.random.default_rng(23)
    Q = np.linalg.qr(rng.normal(size=(64, 64)))[0]  # random orthogonal basis
    p = Q[:, 0]
    # Entailment, neutral, contradiction along independent orthogonal directions.
    h_ent = Q[:, 0]
    h_neu = Q[:, 0] + 3.0 * Q[:, 1]
    h_con = -Q[:, 0]
    for h, y in [(h_ent, ENT), (h_neu, NEU), (h_con, CON)]:
        ver = v.verify(p, h)
        assert ver.predicted_index == y
        assert ver.margin > 0.2
    # Rotation of the whole problem by an orthogonal matrix preserves the decision.
    R = np.linalg.qr(rng.normal(size=(64, 64)))[0]
    ver_r = v.verify(R @ p, R @ h_con)
    assert ver_r.predicted_index == CON
    assert_allclose(ver_r.forward_support, v.verify(p, h_con).forward_support, atol=1e-10)


def test_no_input_mutation():
    v = CounterfactualConstraintVerifier()
    p = np.array([1.0, 2.0, 3.0])
    h = np.array([0.5, -1.0, 2.0])
    cf_p = np.array([1.0, 2.0, 3.0])
    cf_h = np.array([1.0, 2.0, 3.0])
    pc, hc, cpc, chc = p.copy(), h.copy(), cf_p.copy(), cf_h.copy()
    v.verify(p, h, cf_p, cf_h)
    assert_allclose(p, pc) and assert_allclose(h, hc)
    assert_allclose(cf_p, cpc) and assert_allclose(cf_h, chc)


# --------------------------------------------------------------------------
# Strict fail-closed validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [None, 3, [], [[1, 2], [3, 4]], [1j, 0.0, 1.0],
                                  [np.nan, 0.0, 1.0], [0.0, np.inf, 1.0],
                                  [1.0, 2.0, 3.0, 4.0]])  # wrong/extra dims when paired
def test_verify_rejects_invalid_premise_or_mismatch(bad):
    v = CounterfactualConstraintVerifier()
    good = np.array([1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        v.verify(bad, good)
    with pytest.raises(ValueError):
        v.verify(good, bad)


def test_verify_rejects_zero_norm_and_shape_mismatch():
    v = CounterfactualConstraintVerifier()
    with pytest.raises(ValueError):
        v.verify([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        v.verify([1.0, 0.0], [1.0, 0.0, 0.0])
    # Zero-norm hypothesis.
    with pytest.raises(ValueError):
        v.verify([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def test_repulsion_rejects_mismatched_counterfactual_shapes():
    v = CounterfactualConstraintVerifier()
    p = np.array([1.0, 0.0, 0.0])
    h = np.array([0.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        v.verify(p, h, cf_premise=[1.0, 0.0], cf_hypothesis=[0.0, 1.0])


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_reject_invalid_repulsion_weight(bad):
    with pytest.raises(ValueError, match="repulsion_weight"):
        CounterfactualConstraintVerifier(repulsion_weight=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_reject_invalid_temperature(bad):
    with pytest.raises(ValueError, match="temperature"):
        CounterfactualConstraintVerifier(temperature=bad)


def test_verdict_is_frozen_and_serialisable_fields():
    v = CounterfactualConstraintVerifier()
    ver = v.verify([1.0, 0.0, 0.0], [0.5, 0.5, 0.0])
    assert isinstance(ver, BidirectionalVerdict)
    with pytest.raises(Exception):
        ver.forward_support = 9.0  # frozen dataclass
    assert ver.probabilities.shape == (3,)
    assert_allclose(ver.probabilities.sum(), 1.0, atol=1e-12)
    assert ver.coordinates.shape == (2,)
    assert ver.logits.shape == (3,)
    assert 0.0 <= ver.margin <= 1.0


def test_repulsion_path_is_rotation_equivariant():
    """The counterfactual bulk exclusion must not depend on the coordinate axes.

    A per-coordinate operation such as ``p0 - p0.mean()`` changes under an
    orthogonal change of basis; the projection onto p0's complement does not.
    Generic (non-axis-aligned) vectors make any axis dependence visible.
    """
    v = CounterfactualConstraintVerifier()
    rng = np.random.default_rng(101)
    d = 16
    Q = np.linalg.qr(rng.normal(size=(d, d)))[0]
    for _ in range(50):
        p, h, p0, h0 = rng.normal(size=(4, d)) + 0.7  # nonzero mean on purpose
        a = v.verify(p, h, cf_premise=p0, cf_hypothesis=h0)
        b = v.verify(Q @ p, Q @ h, cf_premise=Q @ p0, cf_hypothesis=Q @ h0)
        assert_allclose(b.repulsion_penalty, a.repulsion_penalty, atol=1e-10)
        assert_allclose(b.negation_alignment, a.negation_alignment, atol=1e-10)
        assert_allclose(b.probabilities, a.probabilities, atol=1e-10)
        assert b.predicted_index == a.predicted_index


def test_norm_overflow_fails_closed():
    """Finite elements whose norm overflows float64 must raise, never pass through."""
    v = CounterfactualConstraintVerifier()
    with pytest.raises(ValueError, match="finite norm"):
        v.verify([1e308, 1e308], [1.0, 0.0])
    with pytest.raises(ValueError, match="finite norm"):
        v.verify([1.0, 0.0], [1e308, 1e308])
    with pytest.raises(ValueError, match="finite norm"):
        v.verify([1.0, 0.0], [0.0, 1.0], cf_premise=[1e308, 1e308], cf_hypothesis=[1.0, 0.0])
    # Each pair has a finite norm, but the counterfactual delta overflows.
    with pytest.raises(ValueError, match="finite norm"):
        v.verify([1e154, 0.0], [-1e154, 0.0], cf_premise=[-1e154, 0.0], cf_hypothesis=[1e154, 0.0])


def test_repulsion_penalty_stays_finite_for_large_finite_vectors():
    """cos^2 is taken from unit vectors; squaring the raw dot product gave nan here."""
    _, penalty = CounterfactualConstraintVerifier._exclude_along([1e150, 1e150], [1e150, 0.0])
    assert np.isfinite(penalty)
    assert_allclose(penalty, 0.5, atol=1e-12)


@pytest.mark.parametrize("kwargs", [
    {"cf_premise": [1.0, 0.0]},
    {"cf_hypothesis": [1.0, 0.0]},
])
def test_verify_rejects_half_a_counterfactual_pair(kwargs):
    v = CounterfactualConstraintVerifier()
    with pytest.raises(ValueError, match="both be provided"):
        v.verify([1.0, 0.0], [0.0, 1.0], **kwargs)
