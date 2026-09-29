"""Regression test for round-9 audit finding R9-H01.

R9-H01: R8-H01 only normalized metadata mapping keys. Other legal str-subclass
positions still survived construction unchanged, and _typed() tagged them by
type(value).__name__ ("str_", or a user subclass name). A JSON round-trip
yields builtin str everywhere, so before == after held but digest() drifted.

Fix: __post_init__ pins capability_id/description/required_permissions to
builtin str, _freeze() canonicalizes every str value and recurses into
frozenset, and _typed() tags every str as "str".
"""

import json

import numpy as np
import pytest

from gen_zero.capability.descriptor import CapabilityDescriptor, CapabilityType


class _Tag(str):
    """Plain user str subclass, with a __str__ override to prove content is kept."""

    def __str__(self):
        return "OVERRIDDEN"


def _kwargs(**overrides):
    base = dict(capability_id="tool:test", capability_type=CapabilityType.TOOL, description="test")
    base.update(overrides)
    return base


CASES = {
    "top_level_capability_id": _kwargs(capability_id=np.str_("tool:np")),
    "top_level_description": _kwargs(description=np.str_("np description")),
    "required_permissions_element": _kwargs(required_permissions=[np.str_("fs.read"), "net"]),
    "metadata_mapping_key": _kwargs(metadata={np.str_("x"): "v"}),
    "metadata_nested_plain_value": _kwargs(metadata={"a": {"b": [np.str_("deep"), (np.str_("t"),)]}}),
    "metadata_str_subclass_value": _kwargs(metadata={"k": _Tag("sub")}),
    "metadata_frozenset_element": _kwargs(metadata={"s": frozenset({np.str_("e1"), "e2"})}),
}


def _roundtrip(d):
    return CapabilityDescriptor.from_dict(json.loads(json.dumps(d.to_dict())))


@pytest.mark.parametrize("case", sorted(CASES))
def test_r9_h01_json_roundtrip_keeps_equality_and_digest(case):
    before = CapabilityDescriptor(**CASES[case])
    after = _roundtrip(before)
    assert before == after
    assert before.digest() == after.digest()


@pytest.mark.parametrize("case", sorted(CASES))
def test_r9_h01_subclass_digest_matches_builtin_str_descriptor(case):
    """Digest must equal that of the same descriptor built from builtin str only."""
    before = CapabilityDescriptor(**CASES[case])
    assert before.digest() == _roundtrip(_roundtrip(before)).digest()


def test_top_level_strings_are_builtin_str_at_construction():
    d = CapabilityDescriptor(
        **_kwargs(
            capability_id=np.str_("tool:np"),
            description=_Tag("desc"),
            required_permissions=(np.str_("p"),),
        )
    )
    assert type(d.capability_id) is str
    assert type(d.description) is str and d.description == "desc"
    assert all(type(p) is str for p in d.required_permissions)


def test_str_subclass_value_keeps_content_not_overridden_str():
    d = CapabilityDescriptor(**_kwargs(metadata={"k": _Tag("sub")}))
    assert type(d.metadata["k"]) is str
    assert d.metadata["k"] == "sub"


def test_frozenset_elements_are_canonicalized():
    d = CapabilityDescriptor(**_kwargs(metadata={"s": frozenset({np.str_("e1")})}))
    (elem,) = d.metadata["s"]
    assert type(elem) is str


def test_subclass_and_builtin_inputs_produce_identical_digest():
    a = CapabilityDescriptor(
        **_kwargs(
            capability_id=np.str_("tool:x"),
            description=np.str_("d"),
            required_permissions=[np.str_("p")],
            metadata={np.str_("k"): _Tag("v"), "s": frozenset({np.str_("e")})},
        )
    )
    b = CapabilityDescriptor(
        **_kwargs(
            capability_id="tool:x",
            description="d",
            required_permissions=["p"],
            metadata={"k": "v", "s": frozenset({"e"})},
        )
    )
    assert a == b
    assert a.digest() == b.digest()


def test_non_str_top_level_fields_are_forced_to_str():
    before = CapabilityDescriptor(**_kwargs(capability_id=123, description=4.5))
    assert type(before.capability_id) is str and before.capability_id == "123"
    after = _roundtrip(before)
    assert before == after
    assert before.digest() == after.digest()
