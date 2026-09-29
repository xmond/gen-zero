"""Regression test for ChatGPT-6-Pro round-6 audit finding R6-H01.

R6-H01 (original): {"nested": {1: "v"}}.digest() == {"nested": {"1": "v"}}.digest(),
because _typed()'s Mapping branch stringified keys with str(k), erasing the
type distinction between the int key 1 and the str key "1".

R6-H01 is superseded by R7-H01: allowing non-str Mapping keys to coexist and
disambiguating them inside digest() was never enough, because json.dumps()
still collapses 1 and "1" into the same key on any real JSON roundtrip
(CapabilityDescriptor.to_dict() -> json.dumps -> json.loads -> from_dict).
The fix moved from the mitigation of round 6 (preserve type info at digest
time) to the fail-closed prevention of round 7 (reject non-str keys before
they ever enter a descriptor). Every scenario below that used to construct a
descriptor with a non-str key now asserts TypeError instead. See
test_r7_h01_descriptor_json_roundtrip_string_keys.py for the roundtrip proof.
"""

import os
import subprocess
import sys

import pytest

from gen_zero.capability.descriptor import CapabilityDescriptor, CapabilityType

BASE = dict(capability_id="tool:x", capability_type=CapabilityType.TOOL, description="x")


def test_int_key_is_rejected_str_key_is_accepted():
    with pytest.raises(TypeError, match="Mapping keys must be strings"):
        CapabilityDescriptor(**BASE, metadata={"nested": {1: "v"}})
    d_str = CapabilityDescriptor(**BASE, metadata={"nested": {"1": "v"}})
    assert d_str.digest()


def test_bool_key_is_rejected():
    # bool is a subclass of int; it must still be rejected like any other non-str key.
    with pytest.raises(TypeError, match="Mapping keys must be strings"):
        CapabilityDescriptor(**BASE, metadata={"nested": {True: "v"}})


def test_float_key_is_rejected():
    with pytest.raises(TypeError, match="Mapping keys must be strings"):
        CapabilityDescriptor(**BASE, metadata={"nested": {1.0: "v"}})


def test_mixed_type_keys_are_rejected():
    with pytest.raises(TypeError, match="Mapping keys must be strings"):
        CapabilityDescriptor(**BASE, metadata={"nested": {1: "a", "1": "b", 1.0: "c"}})


def test_tuple_key_is_rejected():
    with pytest.raises(TypeError, match="Mapping keys must be strings"):
        CapabilityDescriptor(**BASE, metadata={"nested": {(1, 2): "v"}})


_SUBPROCESS_SCRIPT = """
import sys
sys.path.insert(0, {python_dir!r})
from gen_zero.capability.descriptor import CapabilityDescriptor, CapabilityType

d = CapabilityDescriptor(
    capability_id="tool:x",
    capability_type=CapabilityType.TOOL,
    description="x",
    metadata={{"nested": {{"1": "v", "one": "w", "2.5": "x", "tuple": (1, 2)}}}},
)
print(d.digest())
"""


def test_digest_stable_across_pythonhashseed_with_str_only_keys():
    """R6-H01's PYTHONHASHSEED stability guard, kept with str-only keys since
    non-str keys are now rejected outright rather than needing digest-level
    disambiguation."""
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
