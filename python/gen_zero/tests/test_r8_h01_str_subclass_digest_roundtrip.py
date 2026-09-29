"""Regression test for ChatGPT-6-Pro round-8 audit finding R8-H01.

R8-H01: np.str_ (a str subclass) passes the `isinstance(k, str)` Mapping-key
guard added for R7-H01, but _freeze() stored the key object unmodified
instead of normalizing it to the builtin str. A JSON roundtrip
(to_dict -> json.dumps -> json.loads -> from_dict) always yields plain str
keys, since JSON has no str-subclass concept, so a descriptor built with an
np.str_ key silently changed its key's runtime type across the roundtrip:
equal by value, but digest() (which tags values by
`type(value).__name__` in _typed()) disagreed before vs. after.

Exact reviewer repro:
    before = CapabilityDescriptor(..., metadata={np.str_("x"): "v"})
    after = CapabilityDescriptor.from_dict(json.loads(json.dumps(before.to_dict())))
    assert before == after            # passed
    assert before.digest() == after.digest()  # FAILED

Fix: _freeze()'s Mapping branch now normalizes every accepted key with
str(k) before storing it, so the frozen key domain is always the builtin
str -- identical to what a JSON roundtrip produces -- and detects the
resulting collision if a normalized key already exists (e.g. np.str_("x")
and "x" both supplied as keys of the same mapping).
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from gen_zero.capability.descriptor import CapabilityDescriptor, CapabilityType

BASE = dict(capability_id="tool:test", capability_type=CapabilityType.TOOL, description="test")


def test_r8_h01_exact_probe_is_stable_across_json_roundtrip():
    before = CapabilityDescriptor(
        **BASE,
        metadata={np.str_("x"): "v"},
    )
    after = CapabilityDescriptor.from_dict(
        json.loads(json.dumps(before.to_dict()))
    )
    assert before == after
    assert before.digest() == after.digest()


def test_np_str_key_normalizes_to_builtin_str_immediately_at_construction():
    """Even without any JSON roundtrip, the frozen key must already be a
    builtin str, not the np.str_ instance, so in-memory digest is stable
    too."""
    d = CapabilityDescriptor(**BASE, metadata={np.str_("x"): "v"})
    (key,) = d.metadata.keys()
    assert type(key) is str


def test_np_str_key_and_builtin_str_key_produce_identical_descriptor():
    a = CapabilityDescriptor(**BASE, metadata={np.str_("x"): "v"})
    b = CapabilityDescriptor(**BASE, metadata={"x": "v"})
    assert a == b
    assert a.digest() == b.digest()


class _IdentityKeyedStr(str):
    """A str subclass compared/hashed by identity, not content.

    np.str_ inherits str's content-based __eq__/__hash__, so
    {np.str_("x"): "a", "x": "b"} collapses to one entry before a
    CapabilityDescriptor is ever constructed -- a real dict literal cannot
    hand _freeze() two str-subclass keys that normalize to the same
    string. This class simulates the only way that scenario reaches
    _freeze(): a Mapping whose distinct keys stringify identically.
    """

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


def test_np_str_key_colliding_with_builtin_str_key_is_rejected():
    colliding_keys = {_IdentityKeyedStr("x"): "a", "x": "b"}
    assert len(colliding_keys) == 2  # sanity: dict kept both, no premature collapse
    with pytest.raises(ValueError, match="collision"):
        CapabilityDescriptor(**BASE, metadata=colliding_keys)


def test_two_distinct_keys_colliding_with_each_other_is_rejected():
    colliding_keys = {_IdentityKeyedStr("dup"): "a", _IdentityKeyedStr("dup"): "b"}
    assert len(colliding_keys) == 2  # sanity: dict kept both, no premature collapse
    with pytest.raises(ValueError, match="collision"):
        CapabilityDescriptor(**BASE, metadata=colliding_keys)


def test_nested_np_str_key_roundtrips_losslessly():
    before = CapabilityDescriptor(
        **BASE,
        metadata={"outer": {np.str_("inner"): "v"}},
        input_schema={"properties": {np.str_("field"): {"type": "string"}}},
    )
    after = CapabilityDescriptor.from_dict(
        json.loads(json.dumps(before.to_dict()))
    )
    assert before == after
    assert before.digest() == after.digest()


def test_np_str_key_rejected_when_not_a_string_at_all_still_works():
    """Sanity check that the R7-H01 non-str guard still fires; np.str_
    normalization must not accidentally widen what counts as a string key."""
    with pytest.raises(TypeError, match="Mapping keys must be strings"):
        CapabilityDescriptor(**BASE, metadata={1: "v"})


_SUBPROCESS_SCRIPT = """
import sys
sys.path.insert(0, {python_dir!r})
import numpy as np
from gen_zero.capability.descriptor import CapabilityDescriptor, CapabilityType

d = CapabilityDescriptor(
    capability_id="tool:x",
    capability_type=CapabilityType.TOOL,
    description="x",
    metadata={{"nested": {{np.str_("one"): "v", "two": "w"}}}},
)
print(d.digest())
"""


def test_digest_stable_across_pythonhashseed_with_np_str_keys():
    python_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script = _SUBPROCESS_SCRIPT.format(python_dir=python_dir)
    digests = set()
    for seed in ("0", "1", "42", "7", "1337", "9999"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        digests.add(result.stdout.strip())
    assert len(digests) == 1, f"digest varied across PYTHONHASHSEED: {digests}"
