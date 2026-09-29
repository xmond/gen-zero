#!/usr/bin/env python3
"""Physical phase-transition-layer slicer for GGUF checkpoints.

Dense-model semantics finish transitioning by roughly the first two-thirds of the stack (see
docs/zero/31-dense-fleet-manifold-anchor-t4-sys-design.md); the remaining layers mostly refine the
vocabulary decode. This tool truncates a GGUF file to its first K transformer blocks so the slice
fits entirely in a single A100's VRAM, with no CPU-GPU paging.

Kept: token_embd.weight, blk.0.* .. blk.{K-1}.*, output_norm.weight, and any other non-block tensor.
Dropped: blk.K.* .. blk.{L-1}.*, and output.weight (the vocabulary lm_head, the single largest
non-block tensor and the one this tool exists to cut).

Uses only the stdlib, mirrors the header-parsing approach already verified in
inspect_gguf_layer_bytes.py, but never decodes metadata array contents (a tokenizer vocab array can
hold hundreds of thousands of entries) -- unmodified metadata entries are copied as raw bytes so
non-UTF-8-safe content round-trips exactly.

Usage: python slice_gguf_layers.py --model-path <in.gguf> --max-layers <K> --output-path <out.gguf>
"""
import argparse
import dataclasses
import os
import re
import struct
import sys
from pathlib import Path

SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
GGUF_MAGIC = b"GGUF"
BLOCK_TENSOR_RE = re.compile(r"^blk\.(\d+)\.")


def _rd(f, fmt):
    n = struct.calcsize(fmt)
    buf = f.read(n)
    if len(buf) != n:
        raise ValueError(f"truncated GGUF file: expected {n} bytes, got {len(buf)}")
    return struct.unpack(fmt, buf)[0]


def _rd_str(f):
    n = _rd(f, "<Q")
    buf = f.read(n)
    if len(buf) != n:
        raise ValueError(f"truncated GGUF string: expected {n} bytes, got {len(buf)}")
    return buf.decode("utf-8")


def _skip_value(f, t):
    """Advance past one metadata value without materializing array contents."""
    if t in SCALAR:
        n = struct.calcsize(SCALAR[t])
        if len(f.read(n)) != n:
            raise ValueError("truncated GGUF scalar value")
    elif t == 8:
        _rd_str(f)
    elif t == 9:
        elem_t = _rd(f, "<I")
        count = _rd(f, "<Q")
        for _ in range(count):
            _skip_value(f, elem_t)
    else:
        raise ValueError(f"unsupported GGUF metadata value type: {t}")


@dataclasses.dataclass
class KV:
    key: str
    type: int
    raw: bytes  # full entry bytes: key-length-prefixed key + type + value, as they appear on disk
    value_start_in_raw: int  # offset of the value (post key+type) within `raw`, for scalar in-place patch


@dataclasses.dataclass
class TensorInfo:
    name: str
    dims: list
    dtype: int
    offset: int  # original offset, relative to original data section start


@dataclasses.dataclass
class Header:
    version: int
    kvs: list
    tensors: list
    header_end: int
    align: int
    arch: str
    block_count_key: str
    n_layer: int


def parse_header(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF file (bad magic {magic!r})")
        version = _rd(f, "<I")
        if version not in (2, 3):
            raise ValueError(f"{path}: unsupported GGUF version {version}, expected 2 or 3")
        n_tensors = _rd(f, "<Q")
        n_kv = _rd(f, "<Q")

        kvs = []
        for _ in range(n_kv):
            entry_start = f.tell()
            key = _rd_str(f)
            t = _rd(f, "<I")
            value_start = f.tell()
            _skip_value(f, t)
            entry_end = f.tell()
            f.seek(entry_start)
            raw = f.read(entry_end - entry_start)
            kvs.append(KV(key=key, type=t, raw=raw, value_start_in_raw=value_start - entry_start))

        tensors = []
        for _ in range(n_tensors):
            name = _rd_str(f)
            n_dims = _rd(f, "<I")
            dims = [_rd(f, "<Q") for _ in range(n_dims)]
            dtype = _rd(f, "<I")
            offset = _rd(f, "<Q")
            tensors.append(TensorInfo(name=name, dims=dims, dtype=dtype, offset=offset))

        header_end = f.tell()

    by_key = {kv.key: kv for kv in kvs}

    def scalar_value(kv):
        if kv.type not in SCALAR:
            raise ValueError(f"metadata key {kv.key!r} is not a scalar (type {kv.type}); cannot patch")
        return struct.unpack(SCALAR[kv.type], kv.raw[kv.value_start_in_raw:])[0]

    if "general.architecture" not in by_key:
        raise ValueError(f"{path}: missing required metadata key general.architecture")
    arch_kv = by_key["general.architecture"]
    if arch_kv.type != 8:
        raise ValueError("general.architecture is not a string")
    arch = arch_kv.raw[arch_kv.value_start_in_raw + 8:].decode("utf-8")

    block_count_key = f"{arch}.block_count"
    if block_count_key not in by_key:
        raise ValueError(f"{path}: missing required metadata key {block_count_key}")
    n_layer = scalar_value(by_key[block_count_key])

    align = 32
    if "general.alignment" in by_key:
        align = scalar_value(by_key["general.alignment"])
    if not (align > 0 and (align & (align - 1)) == 0):
        raise ValueError(f"{path}: general.alignment must be a positive power of two, got {align}")

    return Header(version=version, kvs=kvs, tensors=tensors, header_end=header_end, align=align,
                  arch=arch, block_count_key=block_count_key, n_layer=n_layer)


def _tensor_sizes(header, file_size):
    """Byte size of every original tensor, computed from consecutive offsets (last one: to EOF).
    Also guards against a corrupt/hostile header: an offset already past EOF, or an offset order
    that implies a negative tensor size."""
    data_start = ((header.header_end + header.align - 1) // header.align) * header.align
    order = sorted(range(len(header.tensors)), key=lambda i: header.tensors[i].offset)
    sizes = {}
    for j, i in enumerate(order):
        t = header.tensors[i]
        if data_start + t.offset > file_size:
            raise ValueError(
                f"tensor {t.name!r} offset {t.offset} (data section starts at {data_start}) "
                f"runs past end of file ({file_size} bytes)"
            )
        if j + 1 < len(order):
            end = header.tensors[order[j + 1]].offset
        else:
            end = file_size - data_start
        sizes[t.name] = end - t.offset
        if sizes[t.name] < 0:
            raise ValueError(
                f"tensor {t.name!r} has negative implied size {sizes[t.name]} (offset {t.offset}); "
                f"tensor offsets are corrupt or out of order"
            )
    return sizes, data_start


def _keep_tensor(name, max_layers):
    m = BLOCK_TENSOR_RE.match(name)
    if m:
        return int(m.group(1)) < max_layers
    if name == "output.weight":
        return False
    return True


def _check_layer_tensor_consistency(header, path):
    """The source model's own metadata and tensor list must agree before any --max-layers
    slicing happens: block_count claiming more layers than tensors exist for is a corrupt or
    hand-edited file, not something to silently truncate further."""
    block_indices = [int(m.group(1)) for name in (t.name for t in header.tensors)
                      for m in [BLOCK_TENSOR_RE.match(name)] if m]
    if set(block_indices) != set(range(header.n_layer)):
        raise ValueError(f"{path}: block indices do not match block_count declared layer count")
    actual_max_block = max(block_indices)
    if actual_max_block < header.n_layer - 1:
        raise ValueError(
            f"{path}: {header.block_count_key} claims {header.n_layer} layers, but tensors only "
            f"go up to block index {actual_max_block} (need at least {header.n_layer - 1})"
        )


def _check_not_same_file(model_path, output_path):
    """Refuse to let output clobber input, by path identity or by inode (hardlink)."""
    resolved_model = model_path.resolve()
    resolved_output = output_path.resolve()
    if resolved_model == resolved_output:
        raise ValueError(f"refusing to write output over the input file: {resolved_output}")
    if output_path.exists():
        model_stat = model_path.stat()
        output_stat = output_path.stat()
        if (output_stat.st_dev, output_stat.st_ino) == (model_stat.st_dev, model_stat.st_ino):
            raise ValueError(f"refusing to write output over the input file: {resolved_output}")


def slice_gguf(model_path, max_layers, output_path):
    model_path = Path(model_path)
    output_path = Path(output_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"model path does not exist: {model_path}")
    if max_layers < 1:
        raise ValueError(f"--max-layers must be >= 1, got {max_layers}")
    _check_not_same_file(model_path, output_path)

    header = parse_header(model_path)
    _check_layer_tensor_consistency(header, model_path)
    if max_layers > header.n_layer:
        raise ValueError(
            f"--max-layers={max_layers} exceeds actual layer count {header.n_layer} "
            f"({header.block_count_key} in {model_path})"
        )

    file_size = model_path.stat().st_size
    sizes, data_start_orig = _tensor_sizes(header, file_size)

    kept = [t for t in header.tensors if _keep_tensor(t.name, max_layers)]
    if not kept:
        raise ValueError("slicing produced zero tensors; refusing to write an empty model")

    new_offsets = {}
    cursor = 0
    for t in kept:
        aligned = ((cursor + header.align - 1) // header.align) * header.align
        new_offsets[t.name] = aligned
        cursor = aligned + sizes[t.name]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    try:
        with open(model_path, "rb") as fin, open(tmp_path, "wb") as fout:
            fout.write(GGUF_MAGIC)
            fout.write(struct.pack("<I", header.version))
            fout.write(struct.pack("<Q", len(kept)))
            fout.write(struct.pack("<Q", len(header.kvs)))

            for kv in header.kvs:
                if kv.key == header.block_count_key:
                    packed = struct.pack(SCALAR[kv.type], max_layers)
                    fout.write(kv.raw[:kv.value_start_in_raw])
                    fout.write(packed)
                else:
                    fout.write(kv.raw)

            for t in kept:
                fout.write(struct.pack("<Q", len(t.name.encode("utf-8"))))
                fout.write(t.name.encode("utf-8"))
                fout.write(struct.pack("<I", len(t.dims)))
                for d in t.dims:
                    fout.write(struct.pack("<Q", d))
                fout.write(struct.pack("<I", t.dtype))
                fout.write(struct.pack("<Q", new_offsets[t.name]))

            header_end_new = fout.tell()
            data_start_new = ((header_end_new + header.align - 1) // header.align) * header.align
            fout.write(b"\x00" * (data_start_new - header_end_new))

            pos = 0
            for t in kept:
                pad = new_offsets[t.name] - pos
                if pad < 0:
                    raise AssertionError(f"internal layout error: negative pad for {t.name}")
                if pad:
                    fout.write(b"\x00" * pad)
                fin.seek(data_start_orig + t.offset)
                remaining = sizes[t.name]
                while remaining:
                    chunk = fin.read(min(remaining, 64 * 1024 * 1024))
                    if not chunk:
                        raise ValueError(f"unexpected EOF copying tensor data for {t.name}")
                    fout.write(chunk)
                    remaining -= len(chunk)
                pos = new_offsets[t.name] + sizes[t.name]

        # Atomic install: this is the only line that creates/overwrites output_path.
        os.replace(tmp_path, output_path)
    finally:
        # No-op on the success path (tmp_path was already moved away by os.replace); on any
        # exception -- mid-write or the replace itself -- this leaves zero partial artifacts.
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return {
        "input_layers": header.n_layer,
        "output_layers": max_layers,
        "input_tensors": len(header.tensors),
        "output_tensors": len(kept),
        "output_bytes": output_path.stat().st_size,
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-path", required=True, help="input GGUF file")
    p.add_argument("--max-layers", required=True, type=int, help="number of leading transformer blocks (K) to keep")
    p.add_argument("--output-path", required=True, help="output GGUF file")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = slice_gguf(args.model_path, args.max_layers, args.output_path)
    except Exception as exc:
        print(f"slice_gguf_layers: FAIL-CLOSED: {exc}", file=sys.stderr)
        return 1
    print(f"sliced {result['input_layers']} -> {result['output_layers']} layers, "
          f"{result['input_tensors']} -> {result['output_tensors']} tensors, "
          f"{result['output_bytes']} bytes -> {args.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
