import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "suites"))

from gguf_special_token_detection import detect_special_tokens  # noqa: E402


def test_prefix_only_bos():
    # Gemma/Qwen2.5-72B style: a single BOS token prepended.
    plain = [14990, 1879, 1273]
    with_special = [1] + plain
    prefix, suffix = detect_special_tokens(with_special, plain)
    assert prefix == [1]
    assert suffix == []


def test_suffix_only_eos():
    # GTE-Qwen2-7B style: a single trailing special token appended, no prefix.
    plain = [14990, 1879, 1273]
    with_special = plain + [151643]
    prefix, suffix = detect_special_tokens(with_special, plain)
    assert prefix == []
    assert suffix == [151643]


def test_prefix_and_suffix():
    plain = [10, 20, 30]
    with_special = [1, 2] + plain + [999]
    prefix, suffix = detect_special_tokens(with_special, plain)
    assert prefix == [1, 2]
    assert suffix == [999]


def test_identical_no_special_tokens():
    plain = [5, 6, 7]
    prefix, suffix = detect_special_tokens(plain, plain)
    assert prefix == []
    assert suffix == []


def test_multi_token_prefix_and_suffix():
    plain = [7]
    with_special = [1, 2, 3, 7, 8, 9]
    prefix, suffix = detect_special_tokens(with_special, plain)
    assert prefix == [1, 2, 3]
    assert suffix == [8, 9]


def test_leftmost_occurrence_used_when_plain_repeats():
    plain = [4, 4]
    with_special = [4, 4, 4, 4]
    prefix, suffix = detect_special_tokens(with_special, plain)
    assert prefix == []
    assert suffix == [4, 4]


def test_raises_on_empty_plain():
    with pytest.raises(ValueError, match="empty"):
        detect_special_tokens([1, 2, 3], [])


def test_raises_when_plain_not_contiguous_slice():
    # A tokenizer that rewrites content under add_special, not just wraps it.
    with_special = [1, 99, 2, 3]
    plain = [1, 2, 3]
    with pytest.raises(ValueError, match="not a contiguous slice"):
        detect_special_tokens(with_special, plain)


def test_raises_when_plain_longer_than_with_special():
    with pytest.raises(ValueError, match="not a contiguous slice"):
        detect_special_tokens([1, 2], [1, 2, 3])
