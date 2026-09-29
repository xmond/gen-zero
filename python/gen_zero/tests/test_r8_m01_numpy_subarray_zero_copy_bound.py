"""R8-M01: a subarray dtype made _frombuffer copy the whole array.

The scan charged every _frombuffer array as a view of its buffer. It
accepted any short dtype code, so numpy.dtype("(1024,)u1") passed: frombuffer
then yields a (1024, 1024) array, and reshape((1 << 20,), order="F") must
copy it. A 1 MiB checkpoint declared a 2.18 MB load peak and peaked at
6.3 MB under tracemalloc (raw, gzip and zstd bodies alike).

The scan now admits only the plain scalar codes numpy's own dtype.__reduce__
writes (b1, i/u 1-8, f2/f4/f8, c8/c16), confirms each with numpy (no subdtype,
no fields, no objects), refuses a BUILD state that carries a subarray or a
field list, and admits only "C" and "F" orders. For those dtypes frombuffer
is 1-D and contiguous, so its reshape is always a view.

These tests pin tracemalloc peaks and np.shares_memory, not just exceptions.
"""

import gc
import gzip
import pickle
import tracemalloc

import numpy as np
import pytest

from gen_zero.runtime import base_nano_core
from gen_zero.runtime.base_nano_core import (
    CHECKPOINT_MAGIC,
    OBJECT_OVERHEAD_FLOOR,
    BaseNanoCore,
    CheckpointFormatError,
    _CHECKPOINT_HEADER,
    read_checkpoint_bound,
)
from gen_zero.runtime.pickle_budget import PickleRefusedError, scan_pickle
from gen_zero.runtime.zstd_codec import compress_bytes, get_compression_tier, is_gzip_magic, is_zstd_magic

try:
    from numpy._core.numeric import _frombuffer
except ImportError:  # numpy 1.x
    from numpy.core.numeric import _frombuffer

N = 1 << 20
SCALAR_TYPES = (np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                np.uint32, np.uint64, np.float16, np.float32, np.float64, np.complex64, np.complex128)


class _Dtype:
    """Pickles as numpy.dtype(code, False, True), with an optional BUILD state."""

    def __init__(self, code, state=None):
        self.code, self.state = code, state

    def __reduce__(self):
        args = (self.code, False, True)
        return (np.dtype, args) if self.state is None else (np.dtype, args, self.state)


class _Array:
    """Pickles as numpy's _frombuffer(buf, dtype, shape, order)."""

    def __init__(self, buf, dtype, shape, order):
        self.args = (buf, dtype, shape, order)

    def __reduce__(self):
        return (_frombuffer, self.args)


def _payload(array):
    return pickle.dumps({"weights": {"w": array}, "metadata": {}}, protocol=5)


REVIEWER_RAW = _payload(_Array(bytearray(N), _Dtype("(1024,)u1"), (N,), "F"))


class _Core(BaseNanoCore):
    domain = "r8m01"
    version_id = "r8m01-v1"

    def __init__(self, weights=None, seed=0, **kwargs):
        self.seed = seed
        self.weights = weights if weights is not None else {}

    def score_candidates(self, *args, **kwargs):
        return {}

    def export_artifact(self):
        return None

    def memory_footprint_bytes(self):
        return sum(int(v.nbytes) for v in self.weights.values())


def _peak_of(fn):
    gc.collect()
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        try:
            out, raised = fn(), None
        except Exception as e:  # noqa: BLE001 - the caller asserts the type
            out, raised = None, e
        peak = tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()
    return out, raised, peak


def _body(raw, codec):
    if codec == "raw":
        return raw
    if codec == "gzip":
        return gzip.compress(raw)
    body, tier = compress_bytes(raw)
    assert is_zstd_magic(body), tier
    return body


# -- the reviewer's counterexample -------------------------------------------

def test_r8m01_counterexample_reshape_copies_in_numpy():
    """The numpy fact behind the finding: an F reshape of a subarray-dtype
    frombuffer array is a full copy of the buffer."""
    buf = bytearray(N)
    a = np.frombuffer(buf, dtype=np.dtype("(1024,)u1"))
    assert a.shape == (1024, 1024)
    b = a.reshape((N,), order="F")
    assert not np.shares_memory(a, b)


def test_r8m01_counterexample_refused_by_scan():
    with pytest.raises(PickleRefusedError, match="subarray or compound dtype not permitted"):
        scan_pickle(REVIEWER_RAW)


@pytest.mark.parametrize("codec", ["raw", "gzip", "zstd"])
def test_r8m01_counterexample_checkpoint_refused_before_allocation(tmp_path, codec):
    """Before the fix this file loaded under max_bytes=load_peak_bytes
    (2,179,915 declared) and peaked at 6.3 MB. Its header charges 1 MiB of
    array data as resident, the view the old scan assumed."""
    if codec == "zstd" and get_compression_tier() == "gzip_fallback":
        pytest.skip("no zstd tier on this host")
    body = _body(REVIEWER_RAW, codec)
    path = str(tmp_path / f"r8m01_{codec}.zst")
    head = _CHECKPOINT_HEADER.pack(CHECKPOINT_MAGIC, N, len(REVIEWER_RAW), len(body), OBJECT_OVERHEAD_FLOOR + 32768)
    with open(path, "wb") as f:
        f.write(head + body)
    bound = read_checkpoint_bound(path)
    core, err, peak = _peak_of(lambda: BaseNanoCore.load_checkpoint(path, max_bytes=bound.load_peak_bytes))
    assert core is None
    assert isinstance(err, CheckpointFormatError), err
    assert "subarray or compound dtype not permitted" in str(err)
    assert peak <= bound.load_peak_bytes, (codec, peak, bound.load_peak_bytes)


# -- every other way to a non-scalar dtype ------------------------------------

@pytest.mark.parametrize("dtype", [
    _Dtype("(1024,)u1"),   # subarray, the reviewer's code
    _Dtype("(2,2)f8"),
    _Dtype("2i4"),         # subarray without parentheses
    _Dtype("i4,f4"),       # structured
    _Dtype("V8"),          # void: the carrier numpy uses for compound dtypes
    _Dtype("O8"),          # object references
    _Dtype("U4"),
    _Dtype("S4"),
    _Dtype("M8"),
    _Dtype("f16"),         # longdouble
    _Dtype("<f4"),         # byte order belongs in the BUILD state
    _Dtype("f4 "),
    _Dtype("u1", (3, "|", None, "f0", None, 1, 1, 0)),  # field names via BUILD
    _Dtype("u1", (3, "|", None, None, "f0", 1, 1, 0)),  # fields via BUILD
    _Dtype("u1", (3, "|", 1, None, None, 1, 1, 0)),     # subdescr via BUILD
    _Dtype("u1", (3, "|")),                              # short state
], ids=lambda d: f"{d.code}-{'state' if d.state else 'code'}")
def test_r8m01_non_scalar_dtypes_refused(dtype):
    raw = _payload(_Array(bytearray(64), dtype, (64,), "C"))
    with pytest.raises(PickleRefusedError, match="dtype"):
        scan_pickle(raw)


@pytest.mark.parametrize("order", ["A", "K", "c", "f"])
def test_r8m01_orders_other_than_c_and_f_refused(order):
    raw = _payload(_Array(bytearray(64), _Dtype("u1"), (8, 8), order))
    with pytest.raises(PickleRefusedError, match="'C' or 'F'"):
        scan_pickle(raw)


# -- what is admitted is a view -----------------------------------------------

@pytest.mark.parametrize("scalar", SCALAR_TYPES, ids=lambda t: t.__name__)
@pytest.mark.parametrize("byteorder", ["<", ">"])
def test_r8m01_real_arrays_scan_as_views(scalar, byteorder):
    """numpy's own pickles of every admitted scalar type, both byte orders,
    C and F layouts: the scan accepts them, and the unpickled array shares
    memory with the pickled buffer (its base) and owns no data."""
    dt = np.dtype(scalar).newbyteorder(byteorder)
    for arr in (np.arange(24).astype(dt).reshape(4, 6), np.asfortranarray(np.arange(24).astype(dt).reshape(4, 6)),
                np.arange(24).astype(dt).reshape(2, 3, 4)[..., ::1]):
        raw = pickle.dumps({"weights": {"w": arr}, "metadata": {}}, protocol=5)
        cost = scan_pickle(raw)
        assert cost.array_bytes == arr.nbytes
        out = pickle.loads(raw)["weights"]["w"]
        np.testing.assert_array_equal(out, arr)
        assert out.dtype == dt
        assert not out.flags.owndata
        assert np.shares_memory(out, np.frombuffer(out.base, np.uint8))


@pytest.mark.parametrize("code", ["b1", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8",
                                  "f2", "f4", "f8", "c8", "c16"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("buf_type", [bytes, bytearray])
def test_r8m01_admitted_codes_reshape_is_a_view(code, order, buf_type):
    """Every code the scan admits, both orders, both buffer kinds, several
    shapes: _frombuffer returns a view of the very buffer it was given."""
    dt = np.dtype(code)
    assert dt.subdtype is None and dt.names is None and not dt.hasobject
    count = 24
    buf = buf_type(dt.itemsize * count)
    for shape in ((count,), (4, 6), (6, 4), (2, 3, 4), (1, count, 1), (-1, 3)):
        raw = _payload(_Array(buf, _Dtype(code), shape, order))
        assert scan_pickle(raw).array_bytes == len(buf)
        out = _frombuffer(buf, dt, shape, order)
        assert np.shares_memory(out, np.frombuffer(buf, np.uint8)), (code, order, shape)
        assert not out.flags.owndata


# -- honest checkpoints, full load under the declared bound -------------------

def _gzip_only(data, level=3):
    return gzip.compress(data, compresslevel=min(9, max(1, level))), "gzip_fallback"


@pytest.mark.parametrize("codec", ["raw", "gzip", "zstd"])
def test_r8m01_honest_checkpoint_loads_within_bound(tmp_path, monkeypatch, codec):
    if codec == "zstd" and get_compression_tier() == "gzip_fallback":
        pytest.skip("no zstd tier on this host")
    if codec == "gzip":
        monkeypatch.setattr(base_nano_core, "compress_bytes", _gzip_only)
    rng = np.random.default_rng(8)
    weights = {
        "f_order": np.asfortranarray(rng.standard_normal((512, 256)).astype(np.float32)),
        "big_endian": rng.integers(0, 1 << 30, 65536).astype(">i8"),
        "u8": rng.integers(0, 255, (1024, 1024), dtype=np.uint8),
        "mask": rng.random((256, 64)) > 0.5,
        "c": (rng.standard_normal(4096) + 1j).astype(np.complex64),
    }
    path = str(tmp_path / f"honest_{codec}.zst")
    _Core(weights=weights).save_checkpoint(path, compress=codec != "raw")
    body = open(path, "rb").read()[_CHECKPOINT_HEADER.size:]
    if codec == "gzip":
        assert is_gzip_magic(body)
    elif codec == "zstd":
        assert is_zstd_magic(body)
    bound = read_checkpoint_bound(path)
    core, err, peak = _peak_of(lambda: _Core.load_checkpoint(path, max_bytes=bound.load_peak_bytes))
    assert err is None, err
    assert peak <= bound.load_peak_bytes, (codec, peak, bound.load_peak_bytes)
    for name, arr in weights.items():
        np.testing.assert_array_equal(core.weights[name], arr)
        assert core.weights[name].dtype == arr.dtype


# -- BUILD may not retype a dtype an array already uses -----------------------

def _s(text):
    b = text.encode()
    return b"\x8c" + bytes([len(b)]) + b


def _dtype_state(flags):
    return b"(K\x03" + _s("|") + b"NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK" + bytes([flags]) + b"t"


def _rebuild_after_use_stream(flags):
    """Hand-written stream: dtype("u1") built honestly and memoized, used by
    _frombuffer, then fetched from the memo and BUILT again. Before this fix
    the scan admitted it, and flags 63 left a live uint8 array whose dtype
    claims NPY_ITEM_HASOBJECT over raw bytes."""
    return (b"\x80\x05" + _s("numpy") + _s("dtype") + b"\x93" + _s("u1") + b"\x89\x88\x87R\x94"
            + _dtype_state(0) + b"b0"
            + _s("numpy._core.numeric") + _s("_frombuffer") + b"\x93"
            + b"(\x96" + (64).to_bytes(8, "little") + b"\x41" * 64 + b"h\x00K\x40\x85" + _s("C") + b"tR"
            + b"h\x00" + _dtype_state(flags) + b"b0.")


@pytest.mark.parametrize("flags", [0, 1, 63])
def test_r8m01_second_build_on_used_dtype_refused(flags):
    with pytest.raises(PickleRefusedError, match="already built or used"):
        scan_pickle(_rebuild_after_use_stream(flags))


def test_r8m01_honest_stream_shape_still_admitted():
    """The same stream without the second BUILD is what numpy writes."""
    raw = _rebuild_after_use_stream(0)
    raw = raw[: raw.rindex(b"h\x00")] + b"."
    assert scan_pickle(raw).array_bytes == 64
    out = pickle.loads(raw)
    assert out.dtype == np.uint8 and not out.dtype.hasobject and bytes(out) == b"\x41" * 64


def test_r8m01_object_flag_before_use_fails_without_allocation():
    """A first BUILD may still set the object flag, but then frombuffer
    refuses the dtype with ValueError before it allocates anything."""
    raw = _payload(_Array(bytearray(N), _Dtype("u1", (3, "|", None, None, None, -1, -1, 63)), (N,), "C"))
    scan_pickle(raw)
    out, err, peak = _peak_of(lambda: pickle.loads(raw))
    assert isinstance(err, ValueError) and "OBJECT" in str(err), err
    assert peak < 2 * N + 65536, peak
