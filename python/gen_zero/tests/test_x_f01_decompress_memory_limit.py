"""X-F01: decompress_exact must refuse an oversized stream before it inflates.

zlib's decompress(data, max_length=0) means "no limit", so a declared size of
0 used to inflate a 2 MB gzip payload in full (tracemalloc peak ~7.7 MB) and
only then raise. These tests pin the memory peak, not just the exception.
"""

import gzip
import os
import struct
import tracemalloc
import zlib

import pytest

from gen_zero.runtime import zstd_codec
from gen_zero.runtime.base_nano_core import (
    CHECKPOINT_HEADER_SIZE,
    CHECKPOINT_MAGIC,
    BaseNanoCore,
    CheckpointBudgetExceededError,
    CheckpointFormatError,
    _CHECKPOINT_HEADER,
    read_checkpoint_bound,
)
from gen_zero.runtime.zstd_codec import decompress_exact
from gen_zero.runtime.nano_core_browser import NanoCoreBrowser

PEAK_LIMIT = 256 * 1024
MB = 1024 * 1024


def _forge_isize(data: bytes, size: int) -> bytes:
    """Rewrites the gzip trailer so the cheap pre-check passes and the
    in-stream bound is what has to stop the inflation."""
    return data[:-4] + struct.pack("<I", size & 0xFFFFFFFF)


def _peak_of(fn):
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        try:
            fn()
            raised = None
        except Exception as e:  # noqa: BLE001 - the caller asserts the type
            raised = e
        peak = tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()
    return raised, peak


@pytest.fixture(scope="module", params=[2 * MB, 5 * MB])
def big_gzip(request):
    return gzip.compress(b"A" * request.param), request.param


def test_x_f01_expected_zero_refused_without_inflating(big_gzip):
    data, _ = big_gzip
    err, peak = _peak_of(lambda: decompress_exact(data, 0))
    assert isinstance(err, ValueError), err
    assert peak < PEAK_LIMIT, peak


def test_x_f01_forged_trailer_expected_zero_stops_in_stream(big_gzip):
    data, _ = big_gzip
    forged = _forge_isize(data, 0)
    err, peak = _peak_of(lambda: decompress_exact(forged, 0))
    assert isinstance(err, ValueError), err
    assert "more than the declared 0 bytes" in str(err)
    assert peak < PEAK_LIMIT, peak


def test_x_f01_forged_trailer_small_expected_stops_in_stream(big_gzip):
    data, _ = big_gzip
    forged = _forge_isize(data, 1000)
    err, peak = _peak_of(lambda: decompress_exact(forged, 1000))
    assert isinstance(err, ValueError), err
    assert "more than the declared 1000 bytes" in str(err)
    assert peak < PEAK_LIMIT, peak


def test_x_f01_never_passes_zero_max_length(monkeypatch):
    lengths = []
    real = zlib.decompressobj

    class Spy:
        def __init__(self, *a):
            self._d = real(*a)

        def decompress(self, data, max_length=0):
            lengths.append(max_length)
            return self._d.decompress(data, max_length)

        def __getattr__(self, name):
            return getattr(self._d, name)

    monkeypatch.setattr(zlib, "decompressobj", Spy)
    forged = _forge_isize(gzip.compress(b"B" * MB), 0)
    with pytest.raises(ValueError):
        decompress_exact(forged, 0)
    empty, _ = decompress_exact(gzip.compress(b""), 0)
    assert bytes(empty) == b""
    ok, _ = decompress_exact(gzip.compress(b"C" * 300_000), 300_000)
    assert bytes(ok) == b"C" * 300_000
    assert lengths and min(lengths) >= 1, lengths


@pytest.mark.parametrize("size", [0, 1, 65535, 65536, 65537, 2 * MB])
@pytest.mark.parametrize("kind", ["repeat", "random"])
def test_x_f01_exact_roundtrip(size, kind):
    raw = b"Z" * size if kind == "repeat" else os.urandom(size)
    out, tier = decompress_exact(gzip.compress(raw), size)
    assert tier == "gzip"
    assert bytes(out) == raw


def test_x_f01_valid_decode_peak_is_payload_plus_small_overhead():
    raw = b"Q" * (2 * MB)
    data = gzip.compress(raw)
    holder = {}
    err, peak = _peak_of(lambda: holder.setdefault("out", decompress_exact(data, len(raw))))
    assert err is None
    assert bytes(holder["out"][0]) == raw
    assert peak < len(raw) + PEAK_LIMIT, peak


def test_x_f01_declared_larger_than_stream_is_refused():
    raw = os.urandom(100_000)  # incompressible, so the 1032:1 pre-check passes
    forged = _forge_isize(gzip.compress(raw), 150_000)
    with pytest.raises(ValueError, match="incorrect length check|decodes to 100000 bytes, declared 150000"):
        decompress_exact(forged, 150_000)


def test_x_f01_trailer_mismatch_refused_before_allocation():
    data = gzip.compress(b"S" * MB)
    err, peak = _peak_of(lambda: decompress_exact(data, MB - 1))
    assert isinstance(err, ValueError) and "trailer declares" in str(err)
    assert peak < PEAK_LIMIT, peak


def test_x_f01_absurd_declared_size_refused_before_allocation():
    data = _forge_isize(gzip.compress(b"T" * 1000), 10**12)
    err, peak = _peak_of(lambda: decompress_exact(data, 10**12))
    assert isinstance(err, ValueError) and "1032:1" in str(err)
    assert peak < PEAK_LIMIT, peak


def test_x_f01_trailing_truncated_and_corrupt_refused():
    raw = b"U" * 10_000
    data = gzip.compress(raw)
    with pytest.raises(ValueError, match="trailing data"):
        # A second member ends in the same ISIZE, so only the trailing check can catch it.
        decompress_exact(data + data, len(raw))
    truncated = _forge_isize(data[: len(data) // 2], len(raw))
    with pytest.raises(ValueError):
        decompress_exact(truncated, len(raw))
    corrupt = bytearray(data)
    corrupt[12] ^= 0xFF
    with pytest.raises(ValueError):
        decompress_exact(bytes(corrupt), len(raw))


@pytest.mark.skipif(
    zstd_codec.get_compression_tier() not in ("zstandard_c_ext", "libzstd_ctypes"),
    reason="needs an in-process zstd tier",
)
def test_x_f01_zstd_expected_zero_refused_without_inflating():
    data, _ = zstd_codec.compress_bytes(b"V" * (2 * MB))
    err, peak = _peak_of(lambda: decompress_exact(data, 0))
    assert isinstance(err, ValueError), err
    assert peak < PEAK_LIMIT, peak


def test_x_f01_load_checkpoint_lying_header_refused_with_small_peak(tmp_path):
    body = _forge_isize(gzip.compress(b"W" * (2 * MB)), 0)
    header = _CHECKPOINT_HEADER.pack(CHECKPOINT_MAGIC, 0, 0, len(body), 0)
    path = tmp_path / "lying.ckpt"
    path.write_bytes(header + body)
    err, peak = _peak_of(lambda: BaseNanoCore.load_checkpoint(str(path), max_bytes=10 * MB))
    assert isinstance(err, CheckpointFormatError), err
    assert "more than the declared 0 bytes" in str(err)
    # The body itself (a few KB) is read; nothing near the 2 MB payload is.
    assert peak < PEAK_LIMIT, peak


def test_x_f01_truncated_stream_drain_path_is_capped(monkeypatch):
    # Input can run out while inflate still owes output (a pending match
    # copy). The drain path must use the same cap as the main loop; flush()
    # would not, since its length is only an initial buffer size. A 1-byte
    # step makes those truncation points easy to hit.
    monkeypatch.setattr(zstd_codec, "_GZIP_IN_STEP", 1)
    monkeypatch.setattr(zstd_codec, "_GZIP_OUT_STEP", 1)
    real = zlib.decompressobj
    drains = []

    class Spy:
        def __init__(self, *a):
            self._d = real(*a)

        def decompress(self, data, max_length=0):
            out = self._d.decompress(data, max_length)
            if len(data) == 0:
                drains.append((max_length, len(out)))
            return out

        def __getattr__(self, name):
            return getattr(self._d, name)

    monkeypatch.setattr(zlib, "decompressobj", Spy)
    raw = b"A" * 3000
    full = gzip.compress(raw)
    for cut in range(11, len(full)):
        with pytest.raises(ValueError):
            zstd_codec._gzip_inflate_bounded(full[:cut], len(raw))
    assert any(n for _, n in drains), "drain path never produced output"
    assert all(1 <= cap and n <= cap for cap, n in drains), drains


@pytest.mark.skipif(
    zstd_codec.get_compression_tier() not in ("zstandard_c_ext", "libzstd_ctypes"),
    reason="needs an in-process zstd tier",
)
def test_x_f01_zstd_absurd_declared_size_refused_before_allocation():
    data, _ = zstd_codec.compress_bytes(b"Y" * 1000)
    err, peak = _peak_of(lambda: decompress_exact(data, 10**12))
    assert isinstance(err, ValueError) and "32768:1" in str(err)
    assert peak < PEAK_LIMIT, peak


@pytest.mark.skipif(
    zstd_codec.get_compression_tier() not in ("zstandard_c_ext", "libzstd_ctypes"),
    reason="needs an in-process zstd tier",
)
def test_x_f01_zstd_ratio_bound_holds_for_extreme_input():
    raw = bytes(64 * MB)
    data, _ = zstd_codec.compress_bytes(raw, level=19)
    assert len(raw) <= zstd_codec._ZSTD_MAX_RATIO * len(data)
    out, _ = decompress_exact(data, len(raw))
    assert bytes(out) == raw


def test_x_f01_gzip_checkpoint_save_load_roundtrip(tmp_path, monkeypatch):
    # The production loader on a gzip body: exercises the memoryview return
    # through pickle.loads in load_checkpoint.
    import gzip as _gzip
    from gen_zero.runtime import base_nano_core

    monkeypatch.setattr(
        base_nano_core, "compress_bytes",
        lambda data, level=3: (_gzip.compress(data, compresslevel=6), "gzip_fallback"),
    )
    core = NanoCoreBrowser(state_dim=64, candidate_dim=64, embed_dim=16)
    path = tmp_path / "core.ckpt"
    core.save_checkpoint(str(path), compress=True)
    assert path.read_bytes()[len(CHECKPOINT_MAGIC) + 32:][:2] == b"\x1f\x8b"
    loaded = BaseNanoCore.load_checkpoint(str(path), verify_checksum=True, max_bytes=512 * MB)
    assert loaded._load_tier == "gzip"
    assert core.weights.keys() == loaded.weights.keys()
    for k, v in core.weights.items():
        assert (loaded.weights[k] == v).all(), k


# R5-M01: the load reservation must cover the decompressor's own working
# memory (zlib window and state, in-flight chunks and input tails, the zstd
# context), not just the body and the payload buffer. WebGPT measured a
# 238,439-byte gzip decode stage against a 196,853-byte reservation.

class _FixtureCore(BaseNanoCore):
    """A core whose resident size is exactly its array bytes, like the
    WebGPT fixture (one 65,536-byte float32 array)."""

    domain = "r5m01"
    version_id = "r5m01-v1"

    def __init__(self, weights=None, seed=0, **kwargs):
        import numpy as np

        self.seed = seed
        self.weights = weights if weights is not None else {"w": np.zeros(16384, dtype=np.float32)}

    def score_candidates(self, *args, **kwargs):
        return {}

    def export_artifact(self):
        return {}

    def memory_footprint_bytes(self):
        return sum(int(v.nbytes) for v in self.weights.values())


def _fixture_weights(nbytes, kind):
    import numpy as np

    if kind == "zeros":
        return {"w": np.zeros(nbytes // 4, dtype=np.float32)}
    return {"w": np.frombuffer(os.urandom(nbytes), dtype=np.float32).copy()}


def _use_gzip(monkeypatch):
    from gen_zero.runtime import base_nano_core

    monkeypatch.setattr(
        base_nano_core, "compress_bytes",
        lambda data, level=3: (gzip.compress(data, compresslevel=6), "gzip_fallback"),
    )


def _decode_stage_peak(path, bound):
    """Traced peak of the loader's decode stage alone: read the body, then
    decompress it, exactly as load_checkpoint does."""
    holder = {}

    def stage():
        with open(path, "rb") as f:
            f.seek(CHECKPOINT_HEADER_SIZE)
            body = f.read(bound.body_bytes + 1)
        holder["out"] = decompress_exact(body, bound.raw_payload_bytes)

    err, peak = _peak_of(stage)
    assert err is None, err
    return peak, holder["out"][1]


def _zstd_context_bytes():
    """Native size of the context the bounded zstd path creates; tracemalloc
    cannot see it because libzstd allocates with C malloc."""
    lib = zstd_codec._c_libzstd
    dctx = lib.ZSTD_createDCtx()
    try:
        return lib.ZSTD_sizeof_DCtx(dctx)
    finally:
        lib.ZSTD_freeDCtx(dctx)


@pytest.mark.parametrize("size", [0, 1000, 65536, 262144, 4 * MB])
@pytest.mark.parametrize("kind", ["repeat", "random"])
def test_r5m01_gzip_decode_scratch_within_bound(size, kind):
    raw = b"R" * size if kind == "repeat" else os.urandom(size)
    data = gzip.compress(raw)
    holder = {}
    err, peak = _peak_of(lambda: holder.setdefault("out", decompress_exact(data, size)))
    assert err is None, err
    assert bytes(holder["out"][0]) == raw
    scratch = peak - size
    assert scratch <= zstd_codec.GZIP_DECODE_SCRATCH_BYTES, (scratch, zstd_codec.GZIP_DECODE_SCRATCH_BYTES)


def test_r5m01_webgpt_counterexample_gzip(tmp_path, monkeypatch):
    _use_gzip(monkeypatch)
    path = str(tmp_path / "webgpt.ckpt")
    core = _FixtureCore()
    assert core.memory_footprint_bytes() == 65536
    core.save_checkpoint(path)
    bound = read_checkpoint_bound(path)
    assert bound.resident_bytes == 65536
    assert bound.decode_scratch_bytes >= zstd_codec.GZIP_DECODE_SCRATCH_BYTES

    for _ in range(3):
        decode_peak, tier = _decode_stage_peak(path, bound)
        assert tier == "gzip"
        # The decode stage fits body + payload + scratch alone, without
        # borrowing from the interpreter-object allowance.
        assert decode_peak <= bound.body_bytes + bound.raw_payload_bytes + bound.decode_scratch_bytes, (
            decode_peak, bound)
        assert decode_peak <= bound.load_peak_bytes

        holder = {}
        err, load_peak = _peak_of(
            lambda: holder.setdefault("core", _FixtureCore.load_checkpoint(path, max_bytes=bound.load_peak_bytes))
        )
        assert err is None, err
        assert holder["core"]._load_tier == "gzip"
        assert load_peak <= bound.load_peak_bytes, (load_peak, bound.load_peak_bytes)

    with pytest.raises(CheckpointBudgetExceededError, match="decode scratch"):
        _FixtureCore.load_checkpoint(path, max_bytes=bound.load_peak_bytes - 1)


@pytest.mark.parametrize("nbytes", [65536, 4 * MB])
@pytest.mark.parametrize("kind", ["zeros", "random"])
@pytest.mark.parametrize("codec", ["gzip", "zstd"])
def test_r5m01_full_load_peak_within_reservation(tmp_path, monkeypatch, codec, kind, nbytes):
    if codec == "gzip":
        _use_gzip(monkeypatch)
        native = 0
    else:
        if zstd_codec.get_compression_tier() != "libzstd_ctypes":
            pytest.skip("native context size is only measurable on the libzstd_ctypes tier")
        native = _zstd_context_bytes()
        assert native <= zstd_codec.ZSTD_DECODE_SCRATCH_BYTES
    path = str(tmp_path / f"{codec}.ckpt")
    _FixtureCore(_fixture_weights(nbytes, kind)).save_checkpoint(path)
    bound = read_checkpoint_bound(path)

    # The native zstd context only lives inside decompress_exact, so it is
    # charged to the decode stage; it is freed before unpickling starts.
    decode_peak, tier = _decode_stage_peak(path, bound)
    assert tier == ("gzip" if codec == "gzip" else "libzstd_ctypes")
    assert decode_peak + native <= bound.body_bytes + bound.raw_payload_bytes + bound.decode_scratch_bytes
    assert decode_peak + native <= bound.load_peak_bytes

    holder = {}
    err, load_peak = _peak_of(
        lambda: holder.setdefault("core", _FixtureCore.load_checkpoint(path, max_bytes=bound.load_peak_bytes))
    )
    assert err is None, err
    assert load_peak <= bound.load_peak_bytes, (load_peak, bound.load_peak_bytes)
    assert holder["core"].memory_footprint_bytes() == nbytes


@pytest.mark.skipif(
    zstd_codec.get_compression_tier() != "libzstd_ctypes",
    reason="exercises the libzstd_ctypes context check",
)
def test_r5m01_zstd_context_over_scratch_is_refused_before_output(monkeypatch):
    real = zstd_codec._c_libzstd
    created, freed = [], []

    class Spy:
        def ZSTD_createDCtx(self):
            d = real.ZSTD_createDCtx()
            created.append(d)
            return d

        def ZSTD_freeDCtx(self, d):
            freed.append(d)
            return real.ZSTD_freeDCtx(d)

        def __getattr__(self, name):
            return getattr(real, name)

    monkeypatch.setattr(zstd_codec, "_c_libzstd", Spy())
    monkeypatch.setattr(zstd_codec, "ZSTD_DECODE_SCRATCH_BYTES", 1024)
    data, _ = zstd_codec.compress_bytes(b"X" * (2 * MB))
    err, peak = _peak_of(lambda: decompress_exact(data, 2 * MB))
    assert isinstance(err, MemoryError) and "decode scratch" in str(err), err
    assert peak < PEAK_LIMIT, peak  # the 2 MiB output buffer was never made
    assert created and freed == created
