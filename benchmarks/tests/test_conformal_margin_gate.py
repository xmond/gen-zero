"""Conformal margin abstain gate (benchmarks/suites/conformal_margin_gate.py).

Covers: |m| < theta threshold cut (strict, boundary decides), sign mapping, configurable theta, the
m == 0 tie going to BLOCK, fail-closed behaviour on high-risk traffic, loud failure on bad input, and the
monotone "the gate never releases more unsafe rows than the plain sign rule" property.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "suites"))

import conformal_margin_gate as cmg  # noqa: E402
from conformal_margin_gate import ConformalMarginGate, Verdict  # noqa: E402


# ------------------------------------------------------------------ threshold + sign

def test_default_theta_is_one():
    assert ConformalMarginGate().theta == 1.0


@pytest.mark.parametrize("m,expected", [
    (2.5, Verdict.BLOCK),      # confident unsafe
    (-2.5, Verdict.ALLOW),     # confident safe
    (0.99, Verdict.ABSTAIN),   # inside the band, positive side
    (-0.99, Verdict.ABSTAIN),  # inside the band, negative side
    (0.0, Verdict.ABSTAIN),
])
def test_default_band(m, expected):
    assert ConformalMarginGate().decide(m).verdict is expected


def test_boundary_is_strict_so_theta_itself_decides():
    g = ConformalMarginGate(theta=1.0)
    assert g.decide(1.0).verdict is Verdict.BLOCK
    assert g.decide(-1.0).verdict is Verdict.ALLOW
    assert g.decide(np.nextafter(1.0, 0.0)).verdict is Verdict.ABSTAIN
    assert g.decide(-np.nextafter(1.0, 0.0)).verdict is Verdict.ABSTAIN


def test_theta_is_configurable():
    wide, narrow = ConformalMarginGate(theta=3.0), ConformalMarginGate(theta=0.25)
    assert wide.decide(2.0).verdict is Verdict.ABSTAIN
    assert narrow.decide(2.0).verdict is Verdict.BLOCK
    assert narrow.decide(0.3).verdict is Verdict.BLOCK
    assert narrow.decide(0.2).verdict is Verdict.ABSTAIN


def test_abstain_is_tier2_escalation_and_never_served():
    d = ConformalMarginGate().decide(0.4)
    assert d.verdict is Verdict.ABSTAIN
    assert d.escalate is True and d.released is False
    assert d.label == "TIER2_ESCALATE"
    assert ConformalMarginGate().decide(-5.0).label == "ALLOW"
    assert ConformalMarginGate().decide(5.0).label == "BLOCK"


def test_theta_zero_disables_abstention_and_zero_margin_blocks():
    g = ConformalMarginGate(theta=0.0)
    assert g.decide(0.0).verdict is Verdict.BLOCK          # tie is fail-closed
    assert g.decide(1e-9).verdict is Verdict.BLOCK
    assert g.decide(-1e-9).verdict is Verdict.ALLOW
    assert not any(g.decide(m).verdict is Verdict.ABSTAIN for m in np.linspace(-3, 3, 61))


# ------------------------------------------------------------------ fail-closed

def test_high_risk_uncertain_band_blocks_but_still_escalates():
    g = ConformalMarginGate()
    for m in (-0.9, -0.1, 0.0, 0.1, 0.9):
        d = g.decide(m, high_risk=True)
        assert d.verdict is Verdict.BLOCK, m
        assert d.escalate is True and d.released is False
        assert d.reason == "HIGH_RISK_FAIL_CLOSED"


def test_high_risk_confident_safe_is_still_allowed():
    assert ConformalMarginGate().decide(-1.5, high_risk=True).verdict is Verdict.ALLOW


def test_high_risk_never_allows_inside_the_band_for_any_theta():
    for theta in (0.0, 0.5, 1.0, 2.0, 10.0):
        g = ConformalMarginGate(theta=theta)
        for m in np.linspace(-12, 12, 481):
            d = g.decide(float(m), high_risk=True)
            if abs(m) < theta or m >= 0:
                assert d.verdict is Verdict.BLOCK


def test_is_high_risk_uses_exactly_crime_violence_weapons():
    assert cmg.HIGH_RISK_CATEGORIES == frozenset(
        {"Criminal Planning/Confessions", "Violence", "Guns and Illegal Weapons"})
    assert cmg.is_high_risk(("Sexual", "Violence"))
    assert cmg.is_high_risk(["Guns and Illegal Weapons"])
    assert not cmg.is_high_risk(("Needs Caution", "Profanity"))
    assert not cmg.is_high_risk(())


def test_low_confidence_abstains_and_missing_confidence_is_an_error():
    g = ConformalMarginGate(theta=1.0, min_confidence=0.6)
    d = g.decide(5.0, confidence=0.4)
    assert d.verdict is Verdict.ABSTAIN and d.reason == "LOW_CONFIDENCE"
    assert g.decide(5.0, confidence=0.9).verdict is Verdict.BLOCK
    # A configured confidence floor must never be skipped silently.
    with pytest.raises(ValueError, match="confidence"):
        g.decide(5.0)
    # High risk + low confidence must not turn into a release.
    assert g.decide(-5.0, confidence=0.1, high_risk=True).verdict is Verdict.BLOCK


# ------------------------------------------------------------------ loud failure

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_margin_raises(bad):
    with pytest.raises(ValueError, match="margin"):
        ConformalMarginGate().decide(bad)


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_bad_theta_raises(bad):
    with pytest.raises(ValueError, match="theta"):
        ConformalMarginGate(theta=bad)


@pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan")])
def test_confidence_out_of_range_raises(bad):
    with pytest.raises(ValueError, match="confidence"):
        ConformalMarginGate().decide(1.0, confidence=bad)


@pytest.mark.parametrize("bad", [-0.1, 1.1, -1.0, float("nan"), float("inf")])
def test_min_confidence_out_of_bounds_rejected(bad):
    with pytest.raises(ValueError, match="min_confidence"):
        ConformalMarginGate(min_confidence=bad)


# ------------------------------------------------------------------ batch + monotonicity

def test_batch_matches_scalar_and_requires_aligned_flags():
    g = ConformalMarginGate()
    m = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    hr = np.array([False, False, True, True, False])
    got = g.decide_batch(m, high_risk=hr)
    want = [g.decide(float(a), high_risk=bool(b)) for a, b in zip(m, hr)]
    assert got == want
    assert [d.verdict for d in got] == [Verdict.ALLOW, Verdict.ABSTAIN, Verdict.BLOCK, Verdict.BLOCK, Verdict.BLOCK]
    with pytest.raises(ValueError, match="high_risk"):
        g.decide_batch(m, high_risk=hr[:3])


def test_gate_never_releases_more_unsafe_rows_than_the_plain_sign_rule():
    rng = np.random.default_rng(7)
    m = rng.normal(0.0, 2.0, 2000)
    gold_unsafe = (m + rng.normal(0.0, 1.5, 2000)) > 0          # noisy labels: some unsafe rows have m < 0
    plain_release = m < 0                                        # sign rule, tie -> BLOCK
    for theta in (0.0, 0.5, 1.0, 2.0, 5.0):
        released = np.array([d.released for d in ConformalMarginGate(theta).decide_batch(m)])
        assert (released & gold_unsafe).sum() <= (plain_release & gold_unsafe).sum()
        assert not (released & ~plain_release).any()             # gate never releases what the sign rule blocked
    # leak count is monotone non-increasing in theta
    leaks = [int((np.array([d.released for d in ConformalMarginGate(t).decide_batch(m)]) & gold_unsafe).sum())
             for t in (0.0, 0.5, 1.0, 2.0, 5.0)]
    assert leaks == sorted(leaks, reverse=True)
