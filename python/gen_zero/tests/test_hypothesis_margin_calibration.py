import numpy as np
import pytest

from gen_zero.gate.universal_cognitive_pipeline import universal_task_agnostic_decision

CANDS = ["yes", "no", "maybe"]


def _decide(p_yes, p_no, p_maybe):
    idx, why = universal_task_agnostic_decision("ctx", CANDS, np.array([p_yes, p_no, p_maybe]))
    return CANDS[idx], why


def test_borderline_yes_falls_back_to_no():
    assert _decide(0.5247, 0.4194, 0.0559) == ("no", "hypothesis_affirmative_margin_fallback_no")


def test_clear_yes_is_kept():
    assert _decide(0.7554, 0.2177, 0.0269)[0] == "yes"


def test_margin_boundary_is_inclusive():
    assert _decide(0.55, 0.40, 0.05)[0] == "yes"
    assert _decide(0.54, 0.41, 0.05)[0] == "no"


def test_no_and_maybe_untouched():
    assert _decide(0.2, 0.7, 0.1)[0] == "no"
    assert _decide(0.3, 0.3, 0.4)[0] == "maybe"
