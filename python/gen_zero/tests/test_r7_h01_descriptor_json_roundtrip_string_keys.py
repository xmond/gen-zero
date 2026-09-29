"""Regression test for ChatGPT-6-Pro round-7 audit finding R7-H01.

R7-H01: json.dumps() forces every non-str Mapping key to a string. Even
though R6-H01 made digest() distinguish an int key 1 from a str key "1"
in-memory, a real JSON roundtrip (to_dict -> json.dumps -> json.loads ->
from_dict) still collapses {1: "integer", "1": "string"} into {"1": "string"},
silently dropping the "integer" value and changing the digest.

Fix: reject non-str Mapping keys at construction time (in _freeze, which
__post_init__ runs on input_schema/output_schema/metadata, recursively for
every nested Mapping). This makes the collision structurally impossible
instead of merely detectable after the fact.
"""

import json

import pytest

from gen_zero.capability.descriptor import CapabilityDescriptor, CapabilityType

BASE = dict(capability_id="tool:test", capability_type=CapabilityType.TOOL, description="test")


def test_r7_h01_exact_probe_raises_before_json_can_corrupt_it():
    with pytest.raises(TypeError, match="Mapping keys must be strings"):
        CapabilityDescriptor(
            **BASE,
            metadata={
                "nested": {
                    1: "integer",
                    "1": "string",
                }
            },
        )


def test_top_level_int_key_rejected():
    with pytest.raises(TypeError, match="Mapping keys must be strings"):
        CapabilityDescriptor(**BASE, metadata={1: "v"})


def test_bool_key_rejected():
    with pytest.raises(TypeError, match="Mapping keys must be strings"):
        CapabilityDescriptor(**BASE, metadata={True: "v"})


def test_float_key_rejected():
    with pytest.raises(TypeError, match="Mapping keys must be strings"):
        CapabilityDescriptor(**BASE, metadata={1.5: "v"})


def test_tuple_key_rejected():
    with pytest.raises(TypeError, match="Mapping keys must be strings"):
        CapabilityDescriptor(**BASE, metadata={(1, 2): "v"})


def test_none_key_rejected():
    with pytest.raises(TypeError, match="Mapping keys must be strings"):
        CapabilityDescriptor(**BASE, metadata={None: "v"})


def test_deeply_nested_non_str_key_rejected():
    with pytest.raises(TypeError, match="Mapping keys must be strings"):
        CapabilityDescriptor(
            **BASE,
            metadata={"a": {"b": {"c": {2: "too deep to hide"}}}},
        )


def test_non_str_key_in_input_schema_rejected():
    with pytest.raises(TypeError, match="Mapping keys must be strings"):
        CapabilityDescriptor(**BASE, input_schema={"properties": {1: "v"}})


def test_non_str_key_in_output_schema_rejected():
    with pytest.raises(TypeError, match="Mapping keys must be strings"):
        CapabilityDescriptor(**BASE, output_schema={"properties": {1: "v"}})


def test_from_dict_rejects_non_str_key_without_any_json_step():
    """from_dict must re-run the same guard; a caller can hand it a raw dict
    that never went through json.dumps."""
    with pytest.raises(TypeError, match="Mapping keys must be strings"):
        CapabilityDescriptor.from_dict(
            {
                "capability_id": "tool:test",
                "capability_type": "tool",
                "description": "test",
                "metadata": {1: "v"},
            }
        )


def test_legal_str_keyed_mapping_roundtrips_losslessly_through_json():
    before = CapabilityDescriptor(
        **BASE,
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        metadata={
            "nested": {"1": "string-one", "one": "spelled-out"},
            "bytes_val": b"\x00\x01\xff",
            "set_val": {(1, 2), "a", 3},
            "tuple_val": (1, "two", 3.0, True, None),
            "bool_val": False,
            "float_val": 2.5,
            "none_val": None,
        },
    )
    after = CapabilityDescriptor.from_dict(json.loads(json.dumps(before.to_dict())))
    assert after.metadata == before.metadata
    assert after.to_dict() == before.to_dict()
    assert after.digest() == before.digest()


def test_marker_shaped_user_mapping_still_roundtrips():
    """A user mapping that happens to look like our internal bytes marker
    must be escaped/unescaped correctly, unaffected by the key-type guard
    since '__bytes_hex__' is already a str key."""
    before = CapabilityDescriptor(**BASE, metadata={"raw": {"__bytes_hex__": "61"}})
    after = CapabilityDescriptor.from_dict(json.loads(json.dumps(before.to_dict())))
    assert after.metadata == before.metadata
    assert after.digest() == before.digest()
