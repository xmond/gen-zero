"""R6-M01: unpickling checkpoint metadata must stay inside the load bound.

The counterexample: a 64 KB array plus metadata holding list(range(100000)).
The old allowance was 4x the non-array pickle bytes, a 2,476,822-byte load
bound, while pickle.loads alone peaked at 4,115,187 traced bytes. An int costs
2-5 pickle bytes but ~40 traced bytes, an empty set ~235, so no byte factor
is a bound. The loader now scans the opcode stream first and refuses, before
any object is built, a payload whose worst case exceeds the header allowance.

These tests pin tracemalloc peaks, not just exceptions.
"""

import gc
import gzip
import os
import pickle
import struct
import tracemalloc

import numpy as np
import pytest

from gen_zero.runtime import pickle_budget
from gen_zero.runtime.base_nano_core import (
    CHECKPOINT_HEADER_SIZE,
    CHECKPOINT_MAGIC,
    OBJECT_OVERHEAD_FLOOR,
    BaseNanoCore,
    CheckpointFormatError,
    MemoryBoundExceededError,
    _CHECKPOINT_HEADER,
    object_overhead_allowance,
    read_checkpoint_bound,
)
from gen_zero.runtime.pickle_budget import PickleRefusedError, scan_pickle

MB = 1024 * 1024


class _MetaCore(BaseNanoCore):
    """One 65,536-byte float32 array, like the counterexample, and an
    artifact the test chooses."""

    domain = "r6m01"
    version_id = "r6m01-v1"
    artifact = None

    def __init__(self, weights=None, seed=0, **kwargs):
        self.seed = seed
        self.weights = weights if weights is not None else {"w": np.zeros(16384, dtype=np.float32)}

    def score_candidates(self, *args, **kwargs):
        return {}

    def export_artifact(self):
        return {"probe": self.artifact}

    def memory_footprint_bytes(self):
        return sum(int(v.nbytes) for v in self.weights.values())


def _core_with(artifact):
    core = _MetaCore()
    core.artifact = artifact
    return core


def _peak_of(fn):
    gc.collect()
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        try:
            out = fn()
            raised = None
        except Exception as e:  # noqa: BLE001 - the caller asserts the type
            out, raised = None, e
        peak = tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()
    return out, raised, peak


def _rewrite_header(path, **fields):
    """Rewrites the size header in place and keeps the body: a checkpoint
    whose header lies about its allowance."""
    data = open(path, "rb").read()
    magic, resident, raw_len, body_len, overhead = _CHECKPOINT_HEADER.unpack(data[:CHECKPOINT_HEADER_SIZE])
    vals = dict(resident=resident, raw_len=raw_len, body_len=body_len, overhead=overhead)
    vals.update(fields)
    head = _CHECKPOINT_HEADER.pack(magic, vals["resident"], vals["raw_len"], vals["body_len"], vals["overhead"])
    with open(path, "wb") as f:
        f.write(head + data[CHECKPOINT_HEADER_SIZE:])
    return read_checkpoint_bound(path)


def _write_checkpoint(path, raw, resident=65536, overhead=OBJECT_OVERHEAD_FLOOR + MB):
    """A checkpoint around an arbitrary pickle stream, with a generous header."""
    body = gzip.compress(raw)
    head = _CHECKPOINT_HEADER.pack(CHECKPOINT_MAGIC, resident, len(raw), len(body), overhead)
    with open(path, "wb") as f:
        f.write(head + body)
    return read_checkpoint_bound(path)


# -- the counterexample ------------------------------------------------------

def test_r6m01_counterexample_old_allowance_did_not_cover_unpickle():
    # Documents the defect: the old 4x factor against a real unpickle.
    core = _core_with(list(range(100000)))
    raw = pickle.dumps({"metadata": {"artifact": core.export_artifact()}, "weights": core.weights}, protocol=5)
    old_allowance = object_overhead_allowance(len(raw) - 65536)
    _, err, peak = _peak_of(lambda: pickle.loads(raw))
    assert err is None
    assert peak > old_allowance + 65536, (peak, old_allowance)
    cost = scan_pickle(raw)
    assert peak <= cost.array_bytes + cost.object_bytes, (peak, cost)


def test_r6m01_counterexample_honest_load_within_bound(tmp_path):
    path = str(tmp_path / "r6m01.ckpt")
    _core_with(list(range(100000))).save_checkpoint(path)
    bound = read_checkpoint_bound(path)
    assert bound.overhead_bytes > 4_115_187, bound  # the reported unpickle peak

    for _ in range(2):
        core, err, peak = _peak_of(lambda: _MetaCore.load_checkpoint(path, max_bytes=bound.load_peak_bytes))
        assert err is None, err
        assert core._checkpoint_metadata["artifact"]["probe"] == list(range(100000))
        assert peak <= bound.load_peak_bytes, (peak, bound.load_peak_bytes)


@pytest.mark.parametrize("max_bytes", [None, "declared"])
def test_r6m01_counterexample_old_header_refused_before_unpickle(tmp_path, max_bytes):
    path = str(tmp_path / "old.ckpt")
    _core_with(list(range(100000))).save_checkpoint(path)
    honest = read_checkpoint_bound(path)
    old_overhead = object_overhead_allowance(honest.raw_payload_bytes - 65536)
    forged = _rewrite_header(path, overhead=old_overhead)
    assert forged.overhead_bytes < honest.overhead_bytes
    budget = forged.load_peak_bytes if max_bytes == "declared" else None

    _, err, peak = _peak_of(lambda: _MetaCore.load_checkpoint(path, max_bytes=budget))
    assert isinstance(err, MemoryBoundExceededError), err
    assert isinstance(err, MemoryError)
    assert "nothing was unpickled" in str(err)
    # Nothing near the ~4 MB of int objects was built; the load stayed inside
    # the reservation the forged header asked for.
    assert peak <= forged.load_peak_bytes, (peak, forged.load_peak_bytes)
    assert peak < forged.body_bytes + forged.raw_payload_bytes + forged.decode_scratch_bytes + old_overhead


def test_r6m01_array_bytes_over_resident_refused(tmp_path):
    path = str(tmp_path / "resident.ckpt")
    _core_with([1, 2, 3]).save_checkpoint(path)
    _rewrite_header(path, resident=1000)
    with pytest.raises(MemoryBoundExceededError, match="65536 array bytes against resident 1000"):
        _MetaCore.load_checkpoint(path)


def test_r6m01_header_under_loader_minimum_refused(tmp_path):
    path = str(tmp_path / "tiny.ckpt")
    _core_with(None).save_checkpoint(path)
    _rewrite_header(path, overhead=OBJECT_OVERHEAD_FLOOR - 1)
    with pytest.raises(MemoryBoundExceededError, match="loader minimum"):
        _MetaCore.load_checkpoint(path)


# -- the scan's charge against real unpickle peaks ---------------------------

SHAPES = {
    "int": lambda n: list(range(1000, 1000 + n)),
    "bigint": lambda n: [2**62 + i for i in range(n)],
    "float": lambda n: [i + 0.5 for i in range(n)],
    "str": lambda n: [str(i) for i in range(n)],
    "unicode": lambda n: ["\U0001F600" * 3 + str(i) for i in range(n)],
    "bytes": lambda n: [b"ab%d" % i for i in range(n)],
    "empty_list": lambda n: [[] for _ in range(n)],
    "empty_dict": lambda n: [{} for _ in range(n)],
    "empty_set": lambda n: [set() for _ in range(n)],
    "tuple1": lambda n: [(i + 1000,) for i in range(n)],
    "nested": lambda n: [[[]] for _ in range(n)],
    "dict_int": lambda n: {i + 1000: None for i in range(n)},
    "dict_str": lambda n: {str(i): i for i in range(n)},
    "set_int": lambda n: set(range(1000, 1000 + n)),
    "frozenset_int": lambda n: frozenset(range(1000, 1000 + n)),
    "shared_refs": lambda n: (lambda x: [x] * n)([1, 2]),
}


@pytest.mark.parametrize("n", [1, 1000, 43691, 87382])  # dict/set resize edges
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_r6m01_scan_charge_covers_unpickle_peak(shape, n):
    raw = pickle.dumps(SHAPES[shape](n), protocol=5)
    cost = scan_pickle(raw)
    _, err, peak = _peak_of(lambda: pickle.loads(raw))
    assert err is None, err
    assert peak <= cost.object_bytes, (shape, n, peak, cost)


def test_r6m01_single_giant_frame_is_read_in_place():
    # Re-frame a 400 KB stream as one FRAME: loads from memory must not copy it.
    inner = pickle.dumps(list(range(1000, 101000)), protocol=5)
    body = _strip_frames(inner)
    assert len(body) < len(inner) - 2 - 9  # the stream had frames to strip
    raw = b"\x80\x05\x95" + struct.pack("<Q", len(body)) + body
    cost = scan_pickle(raw)
    out, err, peak = _peak_of(lambda: pickle.loads(raw))
    assert err is None, err
    assert out == list(range(1000, 101000))
    assert peak <= cost.object_bytes, (peak, cost)


def _strip_frames(data):
    """Opcode stream after PROTO with every FRAME opcode and its 8-byte
    length removed."""
    import pickletools
    out = []
    ops = list(pickletools.genops(data))
    for i, (op, _, pos) in enumerate(ops):
        nxt = ops[i + 1][2] if i + 1 < len(ops) else len(data)
        if op.name in ("PROTO", "FRAME"):
            continue
        out.append(data[pos:nxt])
    return b"".join(out)


@pytest.mark.parametrize("shape", ["str", "empty_set", "dict_str"])
def test_r6m01_honest_heavy_metadata_load_within_bound(tmp_path, shape):
    path = str(tmp_path / f"{shape}.ckpt")
    _core_with(SHAPES[shape](30000)).save_checkpoint(path)
    bound = read_checkpoint_bound(path)
    core, err, peak = _peak_of(lambda: _MetaCore.load_checkpoint(path, max_bytes=bound.load_peak_bytes))
    assert err is None, err
    assert peak <= bound.load_peak_bytes, (shape, peak, bound.load_peak_bytes)


def test_r6m01_scan_limit_stops_early():
    raw = pickle.dumps([set() for _ in range(50000)], protocol=5)
    full = scan_pickle(raw)
    limit = full.object_bytes // 10
    _, err, peak = _peak_of(lambda: scan_pickle(raw, limit=limit))
    assert isinstance(err, MemoryBoundExceededError), err
    assert err.cost.objects < full.objects // 5, (err.cost, full)
    # The scan's own bookkeeping stays under the limit it enforces.
    assert peak <= limit, (peak, limit)


def test_r6m01_object_count_cap():
    raw = pickle.dumps([None] * (pickle_budget.MAX_UNPICKLE_OBJECTS + 1), protocol=5)
    with pytest.raises(PickleRefusedError, match=f"more than {pickle_budget.MAX_UNPICKLE_OBJECTS} objects"):
        scan_pickle(raw)


# -- refused pickle features --------------------------------------------------

def _stack_global(module, name):
    def s(text):
        b = text.encode()
        return b"\x8c" + bytes([len(b)]) + b
    return s(module) + s(name) + b"\x93"


BYTEARRAY_BOMB = (
    b"\x80\x05" + _stack_global("builtins", "bytearray")
    + b"J" + struct.pack("<i", 512 * MB) + b"\x85R."
)


@pytest.mark.parametrize("raw, reason", [
    (BYTEARRAY_BOMB, "builtins.bytearray is not allowed"),
    (b"\x80\x05" + _stack_global("os", "system") + b"\x8c\x02id\x85R.", "os.system is not allowed"),
    (pickle.dumps(np.zeros(4, dtype=np.float32), protocol=2), "opcode b'c' is not allowed"),
    (pickle.dumps(np.zeros((2, 2), dtype=np.float32)[:, ::-1], protocol=5), "_reconstruct is not allowed"),
    (pickle.dumps(np.float64(1.5), protocol=5), "scalar is not allowed"),
    (b"\x80\x05Nr\xff\xff\xff\xff.", "opcode b'r' is not allowed"),  # LONG_BINPUT index 2**32-1
    (b"\x80\x05]h\x05.", "memo index 5 was never stored"),
    (b"\x80\x05N\x97.", "is not allowed"),  # NEXT_BUFFER
    (b"\x80\x05N.junk", "bytes after STOP"),
    (b"\x80\x05N", "without STOP"),
    (b"\x80\x05\x8c\x05abc", "truncated"),
    (b"\x80\x05}K\x01a.", "APPEND applies only to a list"),
    (b"\x80\x05" + _stack_global("numpy", "dtype") + b"\x8c\x40" + b"f4," * 21 + b"f" + b"\x85R.",
     "short type code"),
    (b"\x80\x05" + _stack_global("numpy", "dtype") + b"\x8c\x02f4\x85R]b.", "flat tuple of scalars"),
    (b"\x80\x05]]b.", "BUILD applies only to a dtype"),
])
def test_r6m01_unboundable_features_refused(raw, reason):
    with pytest.raises(PickleRefusedError, match=reason.replace("(", r"\(").replace(".", r"\.")):
        scan_pickle(raw)


def test_r6m01_allocation_bomb_checkpoint_refused_before_allocation(tmp_path):
    path = str(tmp_path / "bomb.ckpt")
    _write_checkpoint(path, BYTEARRAY_BOMB)
    _, err, peak = _peak_of(lambda: BaseNanoCore.load_checkpoint(path, max_bytes=16 * MB))
    assert isinstance(err, CheckpointFormatError), err
    assert "builtins.bytearray is not allowed" in str(err)
    assert peak < 256 * 1024, peak


def test_r6m01_numpy_scalar_metadata_refused_at_save(tmp_path):
    path = str(tmp_path / "scalar.ckpt")
    with pytest.raises(CheckpointFormatError, match="scalar is not allowed"):
        _core_with(np.float64(0.5)).save_checkpoint(path)
    assert not os.path.exists(path)


def test_r6m01_real_cores_scan_clean(tmp_path):
    from gen_zero.runtime.nano_core_browser import NanoCoreBrowser
    from gen_zero.runtime.nano_core_vision import NanoCoreVision
    from gen_zero.runtime.specialist_nano_core import DomainSpecialistNanoCore

    for core in (NanoCoreBrowser(), NanoCoreVision(), DomainSpecialistNanoCore(domain="r6m01")):
        path = str(tmp_path / f"{core.domain}.ckpt")
        core.save_checkpoint(path)
        bound = read_checkpoint_bound(path)
        array_bytes = sum(int(v.nbytes) for v in core.weights.values() if hasattr(v, "nbytes"))
        loaded, err, peak = _peak_of(lambda: BaseNanoCore.load_checkpoint(path, max_bytes=bound.load_peak_bytes))
        assert err is None, err
        assert peak <= bound.load_peak_bytes, (core.domain, peak, bound.load_peak_bytes)
        assert bound.overhead_bytes < OBJECT_OVERHEAD_FLOOR + 256 * 1024, bound
        for k, v in core.weights.items():
            assert np.array_equal(loaded.weights[k], v), k
        assert array_bytes <= bound.resident_bytes
