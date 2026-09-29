"""R7-M01 and R7-M02: the unpickle scan's own bounds.

R7-M01: the scan charged a str 160 + 2n for n UTF-8 bytes. PEP 393 stores
"a" * 1_000_000 + "\\U00010000" at 4 bytes per char, and the decoder holds
the old and the widened buffer at once: pickle.loads peaks at 5n, and at
7n with a 1->2->4 widen and a surrogate on the way. The charge is now 8n.

R7-M02: every EMPTY_TUPLE made the scan build a fresh tag with no charge
and no limit check: scan_pickle on tuple([()] * 100000) peaked at 15.2 MB
while the load it guards peaked at 1.7 MB. Stateless values now share one
tag, and every stack, mark and memo push is checked before it happens.

These tests pin tracemalloc peaks, not just exceptions.
"""

import gc
import pickle
import tracemalloc

import numpy as np
import pytest

from gen_zero.runtime import pickle_budget
from gen_zero.runtime.base_nano_core import (
    CHECKPOINT_HEADER_SIZE,
    BaseNanoCore,
    MemoryBoundExceededError,
    _CHECKPOINT_HEADER,
    read_checkpoint_bound,
)
from gen_zero.runtime.pickle_budget import (
    STACK_SLOT_BYTES,
    VARSIZE_HEADER_BYTES,
    PickleRefusedError,
    scan_pickle,
)

MB = 1024 * 1024
N = 1_000_000


class _MetaCore(BaseNanoCore):
    domain = "r7m01"
    version_id = "r7m01-v1"
    artifact = None

    def __init__(self, weights=None, seed=0, **kwargs):
        self.seed = seed
        self.weights = weights if weights is not None else {"w": np.zeros(16384, dtype=np.float32)}

    def score_candidates(self, *args, **kwargs):
        return {}

    def export_artifact(self):
        return self.artifact

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


def _rewrite_overhead(path, overhead):
    data = open(path, "rb").read()
    magic, resident, raw_len, body_len, _ = _CHECKPOINT_HEADER.unpack(data[:CHECKPOINT_HEADER_SIZE])
    head = _CHECKPOINT_HEADER.pack(magic, resident, raw_len, body_len, overhead)
    with open(path, "wb") as f:
        f.write(head + data[CHECKPOINT_HEADER_SIZE:])
    return read_checkpoint_bound(path)


# -- R7-M01: str charge against the PEP 393 decode peak ----------------------

COUNTEREXAMPLE_TEXT = "a" * N + "\U00010000"

# Each is a worst case found by measurement on CPython 3.11.16.
WIDE_STRS = {
    "counterexample": COUNTEREXAMPLE_TEXT,  # 1 -> 4 widen: 5n
    "nonbmp_first": "\U00010000" + "a" * N,
    "widen_1_2_4": "a" * N + "Ā" + "\U00010000",  # 6n
    "surrogate_then_4": "a" * N + "\ud800" + "\U00010000",  # 7n
    "widen_2_surrogate_4": "a" * N + "Ā" + "\ud800" + "\U00010000",  # 7n
    "surrogate_first": "\ud800" + "a" * N + "\U00010000",  # 7n
}


def test_r7m01_counterexample_old_charge_did_not_cover_decode():
    # Documents the defect: the old 160 + 2n charge against a real load.
    raw = pickle.dumps({"artifact": {"text": COUNTEREXAMPLE_TEXT}}, protocol=5)
    n = len(COUNTEREXAMPLE_TEXT.encode("utf-8"))
    out, err, peak = _peak_of(lambda: pickle.loads(raw))
    assert err is None, err
    assert out["artifact"]["text"] == COUNTEREXAMPLE_TEXT
    assert peak > VARSIZE_HEADER_BYTES + 2 * n, peak
    # Even a 4n + header charge, PEP 393's steady-state size, is too low.
    assert peak > VARSIZE_HEADER_BYTES + 4 * n, peak


@pytest.mark.parametrize("name", sorted(WIDE_STRS))
def test_r7m01_str_charge_covers_decode_peak(name):
    text = WIDE_STRS[name]
    raw = pickle.dumps({"artifact": {"text": text}}, protocol=5)
    cost = scan_pickle(raw)
    out, err, peak = _peak_of(lambda: pickle.loads(raw))
    assert err is None, err
    assert out["artifact"]["text"] == text
    assert peak <= cost.object_bytes, (name, peak, cost)


def test_r7m01_counterexample_checkpoint_load_within_bound(tmp_path):
    path = str(tmp_path / "r7m01.ckpt")
    _core_with({"text": COUNTEREXAMPLE_TEXT}).save_checkpoint(path)
    bound = read_checkpoint_bound(path)
    core, err, peak = _peak_of(lambda: _MetaCore.load_checkpoint(path, max_bytes=bound.load_peak_bytes))
    assert err is None, err
    assert core._checkpoint_metadata["artifact"]["text"] == COUNTEREXAMPLE_TEXT
    assert peak <= bound.load_peak_bytes, (peak, bound.load_peak_bytes)


def test_r7m01_header_from_old_charge_refused_before_unpickle(tmp_path):
    # A header written under the old 2n str charge reserves 6n too little.
    path = str(tmp_path / "old.ckpt")
    _core_with({"text": COUNTEREXAMPLE_TEXT}).save_checkpoint(path)
    honest = read_checkpoint_bound(path)
    n = len(COUNTEREXAMPLE_TEXT.encode("utf-8"))
    forged = _rewrite_overhead(path, honest.overhead_bytes - 6 * n)
    _, err, peak = _peak_of(lambda: _MetaCore.load_checkpoint(path, max_bytes=forged.load_peak_bytes))
    assert isinstance(err, MemoryBoundExceededError), err
    assert "nothing was unpickled" in str(err)
    assert peak <= forged.load_peak_bytes, (peak, forged.load_peak_bytes)


# -- R7-M02: the scan's own memory -------------------------------------------

def test_r7m02_counterexample_scan_stays_low():
    raw = pickle.dumps({"artifact": tuple([()] * 100000)}, protocol=5)
    cost, err, scan_peak = _peak_of(lambda: scan_pickle(raw))
    assert err is None, err
    # Was 15,206,223 bytes. What is left is the 8-byte stack slot per item.
    assert scan_peak <= 2 * MB, scan_peak
    assert scan_peak <= cost.object_bytes, (scan_peak, cost)
    _, err, load_peak = _peak_of(lambda: pickle.loads(raw))
    assert err is None, err
    assert load_peak <= cost.object_bytes, (load_peak, cost)


def test_r7m02_counterexample_checkpoint_load_within_bound(tmp_path):
    path = str(tmp_path / "r7m02.ckpt")
    _core_with(tuple([()] * 100000)).save_checkpoint(path)
    bound = read_checkpoint_bound(path)
    core, err, peak = _peak_of(lambda: _MetaCore.load_checkpoint(path, max_bytes=bound.load_peak_bytes))
    assert err is None, err
    assert core._checkpoint_metadata["artifact"] == tuple([()] * 100000)
    assert peak <= bound.load_peak_bytes, (peak, bound.load_peak_bytes)


FLOOD = 1_000_000
LIMIT = 256 * 1024

# Opcode floods that push without creating an object. Each would grow the
# scan's stack, mark stack or memo by FLOOD entries (8+ MB) if unchecked.
FLOODS = {
    "empty_tuple": b"\x80\x05(" + b")" * FLOOD,
    "binget": b"\x80\x05N\x94" + b"h\x00" * FLOOD,
    "dup": b"\x80\x05N" + b"2" * FLOOD,
    "mark": b"\x80\x05" + b")" * 300 + b"(" * FLOOD,  # marks past the small-int cache
    "memoize": b"\x80\x05N" + b"\x94" * FLOOD,
}


@pytest.mark.parametrize("name", sorted(FLOODS))
def test_r7m02_push_flood_stopped_at_limit(name):
    raw = FLOODS[name]
    _, err, peak = _peak_of(lambda: scan_pickle(raw, limit=LIMIT))
    assert isinstance(err, MemoryBoundExceededError), err
    assert peak <= LIMIT, (name, peak)


def test_r7m02_stack_depth_capped_without_limit():
    # The save-time scan has no limit; the depth cap still bounds it.
    raw = b"\x80\x05(" + b")" * (pickle_budget.MAX_UNPICKLE_OBJECTS + 1)
    _, err, peak = _peak_of(lambda: scan_pickle(raw))
    assert isinstance(err, PickleRefusedError), err
    assert "stack deeper than" in str(err)
    assert peak <= STACK_SLOT_BYTES * pickle_budget.MAX_UNPICKLE_OBJECTS, peak


def test_r7m02_memo_capped_without_limit():
    raw = b"\x80\x05N" + b"\x94" * (pickle_budget.MAX_UNPICKLE_OBJECTS + 1)
    with pytest.raises(PickleRefusedError, match="memo entries"):
        scan_pickle(raw)


def test_r7m02_tail_charges_checked_against_limit():
    # A bytes object's data is charged only at the end of the scan, once
    # the scan knows no array wraps it. That tail must meet the limit too.
    raw = pickle.dumps({"blob": b"x" * 100000}, protocol=5)
    full = scan_pickle(raw)
    with pytest.raises(MemoryBoundExceededError):
        scan_pickle(raw, limit=full.object_bytes - 1)
    assert scan_pickle(raw, limit=full.object_bytes) == full


def test_r7m02_shared_tags_never_mutated():
    arr = np.arange(64, dtype=np.float32)
    raw = pickle.dumps({"weights": {"w": arr}, "meta": [(), [], {}, "long" * 40]}, protocol=5)
    cost = scan_pickle(raw)
    assert cost.array_bytes == arr.nbytes
    shared = list(pickle_budget._SHARED_TAGS.values())
    shared += [pickle_budget._EMPTY_TUPLE_TAG, pickle_budget._LARGE_TUPLE_TAG]
    for tag in shared:
        assert tag.wrapped is False and tag.value is None, tag.kind
    assert pickle_budget._EMPTY_TUPLE_TAG.items == ()
