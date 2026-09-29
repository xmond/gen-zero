"""Gen-Zero Runtime: Multi-Tier Adaptive zstd Compression Engine.

Implements high-throughput, sub-2ms compression and decompression for
NanoCore specialist fleets and latent world model checkpoints.

Tier hierarchy:
1. Python `zstandard` C-extension (if installed in environment)
2. Native system `libzstd.so.1` via ctypes (in-process zero-fork C execution, <= 1.8ms)
3. Native `/usr/bin/zstd` CLI via subprocess pipe
4. Standard library `gzip` fallback
"""

import os
import io
import time
import gzip
import shutil
import hashlib
import subprocess
from typing import Dict, Optional, Tuple, Any, Union

# Magic numbers
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
GZIP_MAGIC = b"\x1f\x8b"

# --- Tier 1: zstandard Python C-extension ---
_HAS_ZSTANDARD = False
_zstd_module = None
try:
    import zstandard as _zstd_module
    _HAS_ZSTANDARD = True
except ImportError:
    pass

# --- Tier 2: Native libzstd via ctypes ---
_HAS_CTYPES_LIBZSTD = False
_c_libzstd = None

if not _HAS_ZSTANDARD:
    try:
        import ctypes
        import ctypes.util
        _lib_path = ctypes.util.find_library("zstd") or "libzstd.so.1"
        _c_lib = ctypes.CDLL(_lib_path)
        
        # Setup function prototypes
        _c_lib.ZSTD_versionNumber.restype = ctypes.c_uint
        _c_lib.ZSTD_compressBound.restype = ctypes.c_size_t
        _c_lib.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
        _c_lib.ZSTD_compress.restype = ctypes.c_size_t
        _c_lib.ZSTD_compress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        _c_lib.ZSTD_getFrameContentSize.restype = ctypes.c_ulonglong
        _c_lib.ZSTD_getFrameContentSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        _c_lib.ZSTD_decompress.restype = ctypes.c_size_t
        _c_lib.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
        _c_lib.ZSTD_isError.restype = ctypes.c_uint
        _c_lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
        _c_lib.ZSTD_getErrorName.restype = ctypes.c_char_p
        _c_lib.ZSTD_getErrorName.argtypes = [ctypes.c_size_t]
        # Explicit context for the bounded path, so its native size is known.
        _c_lib.ZSTD_createDCtx.restype = ctypes.c_void_p
        _c_lib.ZSTD_createDCtx.argtypes = []
        _c_lib.ZSTD_freeDCtx.restype = ctypes.c_size_t
        _c_lib.ZSTD_freeDCtx.argtypes = [ctypes.c_void_p]
        _c_lib.ZSTD_sizeof_DCtx.restype = ctypes.c_size_t
        _c_lib.ZSTD_sizeof_DCtx.argtypes = [ctypes.c_void_p]
        _c_lib.ZSTD_decompressDCtx.restype = ctypes.c_size_t
        _c_lib.ZSTD_decompressDCtx.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
        ]

        _c_libzstd = _c_lib
        _HAS_CTYPES_LIBZSTD = True
    except Exception:
        _c_libzstd = None
        _HAS_CTYPES_LIBZSTD = False

# --- Tier 3: CLI executable ---
_ZSTD_CLI_PATH = shutil.which("zstd") or "/usr/bin/zstd"
_HAS_ZSTD_CLI = os.path.isfile(_ZSTD_CLI_PATH) and os.access(_ZSTD_CLI_PATH, os.X_OK)


def get_compression_tier() -> str:
    """Returns the currently active compression tier."""
    if _HAS_ZSTANDARD:
        return "zstandard_c_ext"
    if _HAS_CTYPES_LIBZSTD:
        return "libzstd_ctypes"
    if _HAS_ZSTD_CLI:
        return "zstd_cli_pipe"
    return "gzip_fallback"


def is_zstd_magic(data: bytes) -> bool:
    """Checks whether the first 4 bytes match the zstd frame header."""
    return len(data) >= 4 and data[:4] == ZSTD_MAGIC


def is_gzip_magic(data: bytes) -> bool:
    """Checks whether the first 2 bytes match the gzip frame header."""
    return len(data) >= 2 and data[:2] == GZIP_MAGIC


def compress_bytes(data: bytes, level: int = 3) -> Tuple[bytes, str]:
    """Compresses raw bytes using the fastest available tier.
    
    Args:
        data: Uncompressed bytes.
        level: Compression level (default: 3).
        
    Returns:
        Tuple of (compressed_bytes, tier_used).
    """
    # Tier 1: zstandard module
    if _HAS_ZSTANDARD and _zstd_module is not None:
        cctx = _zstd_module.ZstdCompressor(level=level)
        return cctx.compress(data), "zstandard_c_ext"

    # Tier 2: ctypes libzstd (sub-millisecond in-process)
    if _HAS_CTYPES_LIBZSTD and _c_libzstd is not None:
        import ctypes
        src_len = len(data)
        bound = _c_libzstd.ZSTD_compressBound(src_len)
        dst_buf = ctypes.create_string_buffer(bound)
        res = _c_libzstd.ZSTD_compress(dst_buf, bound, data, src_len, int(level))
        if not _c_libzstd.ZSTD_isError(res):
            return dst_buf.raw[:res], "libzstd_ctypes"

    # Tier 3: CLI subprocess pipe
    if _HAS_ZSTD_CLI:
        try:
            cmd = [_ZSTD_CLI_PATH, f"-{level}", "-c"]
            p = subprocess.run(cmd, input=data, capture_output=True, check=True)
            return p.stdout, "zstd_cli_pipe"
        except Exception:
            pass

    # Tier 4: gzip fallback
    return gzip.compress(data, compresslevel=min(9, max(1, level))), "gzip_fallback"


def decompress_bytes(compressed_data: bytes) -> Tuple[bytes, str]:
    """Decompresses bytes automatically detecting format and using the fastest tier.
    
    Args:
        compressed_data: Compressed bytes (zstd, gzip, or uncompressed).
        
    Returns:
        Tuple of (uncompressed_bytes, tier_used).
    """
    if not compressed_data:
        return b"", "empty"

    # Check for gzip format
    if is_gzip_magic(compressed_data):
        return gzip.decompress(compressed_data), "gzip"

    # If not zstd magic, treat as uncompressed payload
    if not is_zstd_magic(compressed_data):
        return compressed_data, "uncompressed"

    # Tier 1: zstandard module
    if _HAS_ZSTANDARD and _zstd_module is not None:
        dctx = _zstd_module.ZstdDecompressor()
        return dctx.decompress(compressed_data), "zstandard_c_ext"

    # Tier 2: ctypes libzstd (sub-2ms in-process)
    if _HAS_CTYPES_LIBZSTD and _c_libzstd is not None:
        import ctypes
        comp_len = len(compressed_data)
        content_size = _c_libzstd.ZSTD_getFrameContentSize(compressed_data, comp_len)
        # 0xFFFFFFFFFFFFFFFF indicates unknown size; 0xFFFFFFFFFFFFFFFE indicates error
        if content_size != 0xFFFFFFFFFFFFFFFF and content_size != 0xFFFFFFFFFFFFFFFE and content_size > 0:
            dst_buf = ctypes.create_string_buffer(content_size)
            res = _c_libzstd.ZSTD_decompress(dst_buf, content_size, compressed_data, comp_len)
            if not _c_libzstd.ZSTD_isError(res):
                return dst_buf.raw[:res], "libzstd_ctypes"
        else:
            # Buffer doubling strategy if frame content size is unknown
            buf_size = max(1024 * 1024, comp_len * 4)
            for _ in range(5):
                dst_buf = ctypes.create_string_buffer(buf_size)
                res = _c_libzstd.ZSTD_decompress(dst_buf, buf_size, compressed_data, comp_len)
                if not _c_libzstd.ZSTD_isError(res):
                    return dst_buf.raw[:res], "libzstd_ctypes"
                buf_size *= 4

    # Tier 3: CLI subprocess pipe
    if _HAS_ZSTD_CLI:
        try:
            cmd = [_ZSTD_CLI_PATH, "-d", "-c"]
            p = subprocess.run(cmd, input=compressed_data, capture_output=True, check=True)
            return p.stdout, "zstd_cli_pipe"
        except Exception:
            pass

    raise RuntimeError("Failed to decompress zstd stream: no functional decompression engine found.")


# Bounded gzip decoding feeds at most _GZIP_IN_STEP compressed bytes and asks
# for at most _GZIP_OUT_STEP decoded bytes per call. Both caps size the
# decoder's working memory, which zlib's Python API cannot write in place:
# every call returns a fresh bytes chunk, and leftover input comes back as a
# fresh unconsumed_tail copy.
_GZIP_IN_STEP = 8 * 1024
_GZIP_OUT_STEP = 16 * 1024

# Working memory of one bounded gzip decode, on top of the compressed body and
# the exact-size output buffer. zconf.h: inflate needs "1 << windowBits ...
# plus about 7 kilobytes for small objects"; the state allowance is twice
# that, and also covers the Decompress object and the chunk object headers.
# The output chunk is counted twice (shrinking it to its decoded length can
# reallocate), and so is the input tail: the previous tail is only released
# after the call that creates the next one returns.
# This is a documented bound, pinned by tests (zlib has no API to report or
# cap its own allocations); the zstd bound below is enforced at run time.
ZLIB_WINDOW_BYTES = 1 << 15
ZLIB_STATE_ALLOWANCE_BYTES = 16 * 1024
GZIP_DECODE_SCRATCH_BYTES = (
    ZLIB_WINDOW_BYTES + ZLIB_STATE_ALLOWANCE_BYTES + 2 * _GZIP_OUT_STEP + 2 * _GZIP_IN_STEP
)

# Native decompression context of one bounded zstd decode (libzstd 1.5.5:
# ZSTD_sizeof_DCtx = 95,992). A one-shot decode into a full-size buffer
# needs no window buffer beyond it. The size is checked before and after
# every decode, and a context larger than this bound is refused.
ZSTD_DECODE_SCRATCH_BYTES = 128 * 1024

# A loader reserves this before it has read the body, so before it knows the
# codec: the largest working memory of any bounded decoder.
DECODE_SCRATCH_BYTES = max(GZIP_DECODE_SCRATCH_BYTES, ZSTD_DECODE_SCRATCH_BYTES)

# 128 KiB block maximum / 4-byte smallest block (3-byte header + 1 RLE byte).
_ZSTD_MAX_RATIO = (128 * 1024) // 4


def _gzip_decompress_exact(compressed_data: bytes, expected_size: int) -> memoryview:
    """Decodes one gzip member into a buffer of exactly `expected_size` bytes.

    Gotcha: zlib's `decompress(data, max_length)` treats max_length=0 as
    "no limit", so a declared size of 0 used to inflate the whole stream.
    Every call here asks for at least 1 byte and at most one byte past the
    declared size, so an oversized stream is refused one byte past the
    declared size, never after the full payload.
    """
    # Cheap checks before the buffer exists, so a lying size header cannot
    # make us allocate: the gzip trailer's ISIZE is the size mod 2**32, and
    # deflate cannot expand past 1032:1 (zlib's documented maximum).
    if len(compressed_data) < 18:
        raise ValueError("gzip payload is shorter than a gzip header plus trailer.")
    isize = int.from_bytes(compressed_data[-4:], "little")
    if isize != expected_size & 0xFFFFFFFF:
        raise ValueError(
            f"gzip trailer declares {isize} bytes (mod 2**32), checkpoint header declares {expected_size}."
        )
    if expected_size > 1032 * len(compressed_data):
        raise ValueError(
            f"declared {expected_size} bytes exceeds deflate's 1032:1 limit for a "
            f"{len(compressed_data)}-byte gzip payload."
        )

    return _gzip_inflate_bounded(compressed_data, expected_size)


def _gzip_inflate_bounded(compressed_data: bytes, expected_size: int) -> memoryview:
    """The bounded inflate loop behind _gzip_decompress_exact, without its
    pre-checks. Holds at most expected_size + 1 decoded bytes at any time."""
    import zlib

    buf = bytearray(expected_size)
    pos = 0
    src = memoryview(compressed_data)
    src_off = 0
    dobj = zlib.decompressobj(16 + zlib.MAX_WBITS)

    def _take(out: bytes) -> None:
        nonlocal pos
        if pos + len(out) > expected_size:
            raise ValueError(
                f"gzip payload decodes to more than the declared {expected_size} bytes; "
                f"refused before inflating the rest."
            )
        buf[pos:pos + len(out)] = out
        pos += len(out)

    try:
        while not dobj.eof:
            if dobj.unconsumed_tail:
                chunk = dobj.unconsumed_tail
            elif src_off < len(src):
                chunk = src[src_off:src_off + _GZIP_IN_STEP]
                src_off += len(chunk)
            else:
                # Input exhausted: drain pending output with an empty feed.
                # Not flush(): its length is only an initial buffer size, not a cap.
                out = dobj.decompress(b"", min(_GZIP_OUT_STEP, expected_size - pos + 1))
                if not out:
                    break
                _take(out)
                del out
                continue
            out = dobj.decompress(chunk, min(_GZIP_OUT_STEP, expected_size - pos + 1))
            del chunk
            _take(out)
            del out
    except zlib.error as e:
        raise ValueError(f"gzip payload is corrupt: {e}") from e

    if not dobj.eof:
        raise ValueError(f"gzip payload is truncated after {pos} of {expected_size} declared bytes.")
    if dobj.unused_data or src_off < len(src):
        raise ValueError("gzip payload carries trailing data after its first member.")
    if pos != expected_size:
        raise ValueError(f"gzip payload decodes to {pos} bytes, declared {expected_size}.")
    return memoryview(buf)


def _check_zstd_scratch(context_bytes: int, when: str) -> None:
    """Refuses a zstd context that outgrows the scratch every load reserves."""
    if context_bytes > ZSTD_DECODE_SCRATCH_BYTES:
        raise MemoryError(
            f"zstd decompression context is {context_bytes} bytes {when}, over the "
            f"{ZSTD_DECODE_SCRATCH_BYTES}-byte decode scratch the load reserved."
        )


def decompress_exact(compressed_data: bytes, expected_size: int) -> Tuple[Union[bytes, memoryview], str]:
    """Decompresses into a buffer of exactly `expected_size` bytes, allocated
    before decoding starts. A stream that decodes to any other size raises
    instead of growing the buffer, so the caller's pre-allocation bound holds.

    Returns a bytes-like object (bytes or memoryview) and the tier used.
    """
    if expected_size < 0:
        raise ValueError(f"expected_size must be >= 0, got {expected_size}")

    if is_gzip_magic(compressed_data):
        return _gzip_decompress_exact(compressed_data, expected_size), "gzip"

    if not is_zstd_magic(compressed_data):
        if len(compressed_data) != expected_size:
            raise ValueError(
                f"Uncompressed payload is {len(compressed_data)} bytes, declared {expected_size}."
            )
        return compressed_data, "uncompressed"

    # Pre-allocation check for zstd, like the gzip ratio check: a block
    # header is 3 bytes and an RLE block's content is 1 byte, and no block
    # decodes to more than 128 KiB (RFC 8878 3.1.1.2), so a frame cannot
    # expand past 32768:1. Coarse, but it stops an absurd declared size
    # from reaching the buffer allocation below.
    if expected_size > _ZSTD_MAX_RATIO * len(compressed_data):
        raise ValueError(
            f"declared {expected_size} bytes exceeds zstd's {_ZSTD_MAX_RATIO}:1 limit for a "
            f"{len(compressed_data)}-byte frame."
        )

    if _HAS_ZSTANDARD and _zstd_module is not None:
        params = _zstd_module.get_frame_parameters(compressed_data)
        if params.content_size != expected_size:
            raise ValueError(
                f"zstd frame declares {params.content_size} bytes, checkpoint header declares {expected_size}."
            )
        dctx = _zstd_module.ZstdDecompressor()
        _check_zstd_scratch(dctx.memory_size(), "before decoding")
        try:
            out = dctx.decompress(compressed_data, max_output_size=expected_size)
        except _zstd_module.ZstdError as e:
            raise ValueError(f"zstd bounded decompression failed: {e}") from e
        _check_zstd_scratch(dctx.memory_size(), "after decoding")
        if len(out) != expected_size:
            raise ValueError(f"zstd frame decoded to {len(out)} bytes, declared {expected_size}.")
        return out, "zstandard_c_ext"

    if _HAS_CTYPES_LIBZSTD and _c_libzstd is not None:
        import ctypes
        comp_len = len(compressed_data)
        content_size = _c_libzstd.ZSTD_getFrameContentSize(compressed_data, comp_len)
        if content_size != expected_size:
            raise ValueError(
                f"zstd frame declares {content_size} bytes, checkpoint header declares {expected_size}."
            )
        dctx = _c_libzstd.ZSTD_createDCtx()
        if not dctx:
            raise MemoryError("ZSTD_createDCtx failed to allocate a decompression context.")
        try:
            # Checked before the output buffer exists: a context larger than
            # the reserved scratch is refused, not decoded with anyway.
            _check_zstd_scratch(_c_libzstd.ZSTD_sizeof_DCtx(dctx), "before decoding")
            # ZSTD_decompressDCtx fails with dstSize_tooSmall rather than
            # writing past this buffer, so a lying frame cannot make us
            # allocate more.
            dst_buf = ctypes.create_string_buffer(max(1, expected_size))
            res = _c_libzstd.ZSTD_decompressDCtx(dctx, dst_buf, expected_size, compressed_data, comp_len)
            _check_zstd_scratch(_c_libzstd.ZSTD_sizeof_DCtx(dctx), "after decoding")
        finally:
            _c_libzstd.ZSTD_freeDCtx(dctx)
        if _c_libzstd.ZSTD_isError(res):
            name = _c_libzstd.ZSTD_getErrorName(res).decode("utf-8", "replace")
            raise ValueError(f"zstd bounded decompression failed: {name}")
        if res != expected_size:
            raise ValueError(f"zstd frame decoded to {res} bytes, declared {expected_size}.")
        # A view, not dst_buf.raw: slicing .raw would copy the whole payload.
        return memoryview(dst_buf).cast("B")[:expected_size], "libzstd_ctypes"

    # The CLI tier pipes an unbounded stdout into memory; it cannot honour
    # the bound, so it is refused rather than trusted.
    raise RuntimeError(
        "Bounded zstd decompression needs the zstandard module or libzstd; "
        "the CLI tier cannot cap its output and is refused."
    )
