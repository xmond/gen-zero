"""Regression tests for WebGPT round-4 findings T3-H01 / T3-D01.

T3-H01: CapabilityDescriptor.from_dict(d.to_dict()).digest() != d.digest()
whenever metadata/input_schema/output_schema contained bytes or a set, because
_thaw()'s JSON-safe encoding was never inverted by from_dict() (bytes markers
stayed as plain mappings; frozenset markers collapsed into plain sequences).

T3-D01: digest() was not stable across PYTHONHASHSEED values for descriptors
containing a set/frozenset, because _typed() encoded sets by raw iteration
order instead of a canonical sort.
"""

import os
import subprocess
import sys

from gen_zero.capability.descriptor import CapabilityDescriptor, CapabilityType

BASE = dict(capability_id="tool:x", capability_type=CapabilityType.TOOL, description="x")


def test_bytes_roundtrip_preserves_digest():
    d1 = CapabilityDescriptor(**BASE, metadata={"raw": b"a"})
    d2 = CapabilityDescriptor.from_dict(d1.to_dict())
    assert d2.metadata["raw"] == b"a"
    assert d1.digest() == d2.digest()


def test_nested_bytes_roundtrip_preserves_digest():
    d1 = CapabilityDescriptor(
        **BASE,
        input_schema={"blob": {"inner": b"\x00\xff\x10"}},
        metadata={"items": [b"x", {"deep": b"y"}]},
    )
    d2 = CapabilityDescriptor.from_dict(d1.to_dict())
    assert d2.input_schema["blob"]["inner"] == b"\x00\xff\x10"
    assert d1.digest() == d2.digest()


def test_set_roundtrip_preserves_digest():
    d1 = CapabilityDescriptor(**BASE, metadata={"tags": {"alpha", "beta", "gamma"}})
    d2 = CapabilityDescriptor.from_dict(d1.to_dict())
    assert isinstance(d2.metadata["tags"], frozenset)
    assert d2.metadata["tags"] == frozenset({"alpha", "beta", "gamma"})
    assert d1.digest() == d2.digest()


def test_nested_set_roundtrip_preserves_digest():
    d1 = CapabilityDescriptor(**BASE, metadata={"outer": {"inner_set": {1, 2, 3}}})
    d2 = CapabilityDescriptor.from_dict(d1.to_dict())
    assert d2.metadata["outer"]["inner_set"] == frozenset({1, 2, 3})
    assert d1.digest() == d2.digest()


def test_double_roundtrip_is_idempotent():
    d1 = CapabilityDescriptor(**BASE, metadata={"raw": b"a", "tags": {"x", "y", "z"}})
    d2 = CapabilityDescriptor.from_dict(d1.to_dict())
    d3 = CapabilityDescriptor.from_dict(d2.to_dict())
    assert d1.digest() == d2.digest() == d3.digest()


def test_marker_shaped_user_mapping_roundtrips_unescaped():
    """A genuine user value that happens to look like our internal wire marker
    (e.g. {"__bytes_hex__": ...}) must roundtrip as itself, not get misread as
    an actual bytes/set marker."""
    for collider in (
        {"__bytes_hex__": "61"},
        {"__set__": [1]},
        {"__map__": "not-a-marker"},
    ):
        d1 = CapabilityDescriptor(**BASE, metadata={"raw": collider})
        d2 = CapabilityDescriptor.from_dict(d1.to_dict())
        # The exact defect this guards: a marker-shaped user mapping must not
        # get misread as an actual bytes/set marker on the way back in.
        assert not isinstance(d2.metadata["raw"], (bytes, frozenset)), collider
        # Compare against d1's own (already-frozen) state, not the raw literal:
        # __post_init__ freezes lists to tuples at construction time regardless
        # of any roundtrip, so the literal itself is not the right baseline.
        assert d2.metadata == d1.metadata, collider
        assert d1.digest() == d2.digest(), collider


def test_set_of_tuples_roundtrips_without_crashing():
    d1 = CapabilityDescriptor(**BASE, metadata={"pairs": {(1, 2), (3, 4)}})
    d2 = CapabilityDescriptor.from_dict(d1.to_dict())
    assert d2.metadata["pairs"] == frozenset({(1, 2), (3, 4)})
    assert d1.digest() == d2.digest()


def test_set_digest_independent_of_insertion_order():
    d1 = CapabilityDescriptor(**BASE, metadata={"tags": frozenset(["alpha", "beta", "gamma", "delta"])})
    d2 = CapabilityDescriptor(**BASE, metadata={"tags": frozenset(["delta", "gamma", "beta", "alpha"])})
    assert d1.digest() == d2.digest()


_SUBPROCESS_SCRIPT = """
import sys
sys.path.insert(0, {python_dir!r})
from gen_zero.capability.descriptor import CapabilityDescriptor, CapabilityType

d = CapabilityDescriptor(
    capability_id="tool:x",
    capability_type=CapabilityType.TOOL,
    description="x",
    metadata={{"tags": {{"apple", "banana", "cherry", "date", "elderberry", "fig"}}}},
)
print(d.digest())
"""


def test_digest_stable_across_pythonhashseed():
    python_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script = _SUBPROCESS_SCRIPT.format(python_dir=python_dir)
    digests = set()
    for seed in ("0", "1", "42"):
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
