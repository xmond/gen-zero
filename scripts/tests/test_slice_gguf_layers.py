"""Tests for slice_gguf_layers.py: build a minimal legal GGUF byte stream by hand (independent of
the module under test), slice it, and check both the raw bytes and the module's own header parser
agree. Also checks the fail-closed paths the tool must never silently pass through: missing file,
non-GGUF file, --max-layers past the real layer count, input==output path, a write failure mid
atomic-install, an unsupported GGUF version, a non-power-of-two alignment, and a declared layer
count that exceeds the tensors actually present.
"""
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import slice_gguf_layers as slicer

ALIGN = 32


def _kv_string(key, value):
    kb = key.encode("utf-8")
    vb = value.encode("utf-8")
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb


def _kv_u32(key, value):
    kb = key.encode("utf-8")
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 4) + struct.pack("<I", value)


def _kv_u32_array(key, values):
    kb = key.encode("utf-8")
    body = struct.pack("<I", 4) + struct.pack("<Q", len(values))
    for v in values:
        body += struct.pack("<I", v)
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 9) + body


def _tensor_info(name, offset, n_floats=4):
    nb = name.encode("utf-8")
    return (struct.pack("<Q", len(nb)) + nb + struct.pack("<I", 1) + struct.pack("<Q", n_floats)
            + struct.pack("<I", 0) + struct.pack("<Q", offset))


def build_mock_gguf(path, n_layers=3, n_floats_per_tensor=4, version=3, align=ALIGN):
    """3 layers, one tensor for blk.0 (two tensors, to prove multi-tensor blocks work), one for
    blk.1 and blk.2 (one tensor each), plus token_embd/output_norm/output.weight.

    `n_layers` only feeds the llama.block_count metadata value -- the tensor list always goes up
    to blk.2 regardless, which is exactly what a declared-vs-actual-layer-count mismatch test
    needs (pass n_layers > 3 to make block_count lie about tensors that don't exist)."""
    tensor_size = n_floats_per_tensor * 4  # float32

    names_in_order = [
        "token_embd.weight",
        "blk.0.attn_q.weight",
        "blk.0.ffn_down.weight",
        "blk.1.attn_q.weight",
        "blk.2.attn_q.weight",
        "output_norm.weight",
        "output.weight",
    ]
    offsets = {}
    cursor = 0
    for name in names_in_order:
        aligned = ((cursor + align - 1) // align) * align
        offsets[name] = aligned
        cursor = aligned + tensor_size

    kvs = [
        _kv_string("general.architecture", "llama"),
        _kv_u32("general.alignment", align),
        _kv_u32("llama.block_count", n_layers),
        _kv_string("general.name", "tiny-mock"),
        _kv_u32_array("test.array", [1, 2, 3]),
    ]
    tensor_infos = [_tensor_info(name, offsets[name], n_floats_per_tensor) for name in names_in_order]

    header = (b"GGUF" + struct.pack("<I", version) + struct.pack("<Q", len(tensor_infos))
              + struct.pack("<Q", len(kvs)) + b"".join(kvs) + b"".join(tensor_infos))
    header_end = len(header)
    data_start = ((header_end + align - 1) // align) * align

    buf = bytearray(header)
    buf += b"\x00" * (data_start - header_end)
    # Tensor i's byte span in the data section runs [offset_i, offset_{i+1}) for every tensor but
    # the last (the alignment gap before the next tensor rides along, exactly like production's
    # _tensor_sizes convention); the last tensor's span is just its own payload, file ends there.
    for i, name in enumerate(names_in_order):
        payload = struct.pack(f"<{n_floats_per_tensor}f", *([float(i + 1)] * n_floats_per_tensor))
        assert len(payload) == tensor_size
        buf += payload
        if i + 1 < len(names_in_order):
            next_offset = offsets[names_in_order[i + 1]]
            pad = next_offset - (offsets[name] + tensor_size)
            buf += b"\x00" * pad

    path.write_bytes(bytes(buf))
    return names_in_order, offsets


@pytest.fixture
def mock_model(tmp_path):
    model_path = tmp_path / "mock_3layer.gguf"
    build_mock_gguf(model_path)
    return model_path


def test_slices_to_two_layers(mock_model, tmp_path):
    out_path = tmp_path / "sliced.gguf"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "slice_gguf_layers.py"),
         "--model-path", str(mock_model), "--max-layers", "2", "--output-path", str(out_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    raw = out_path.read_bytes()
    assert raw.startswith(b"GGUF")
    assert b"blk.0." in raw
    assert b"blk.1." in raw
    assert b"blk.2." not in raw
    assert b"output.weight" not in raw
    assert b"output_norm.weight" in raw
    assert b"token_embd.weight" in raw

    header = slicer.parse_header(out_path)
    assert header.n_layer == 2
    assert len(header.tensors) == 5  # token_embd, blk.0 x2, blk.1 x1, output_norm
    names = {t.name for t in header.tensors}
    assert names == {"token_embd.weight", "blk.0.attn_q.weight", "blk.0.ffn_down.weight",
                      "blk.1.attn_q.weight", "output_norm.weight"}

    array_kv = next(kv for kv in header.kvs if kv.key == "test.array")
    original_array_kv = next(kv for kv in slicer.parse_header(mock_model).kvs if kv.key == "test.array")
    assert array_kv.raw == original_array_kv.raw  # untouched metadata must round-trip byte-for-byte

    # blk.0.attn_q.weight was written with payload value 2.0 (see build_mock_gguf); confirm the
    # tensor data itself, not just the header, survived the copy byte-for-byte.
    align = header.align
    data_start = ((header.header_end + align - 1) // align) * align
    t = next(t for t in header.tensors if t.name == "blk.0.attn_q.weight")
    payload = raw[data_start + t.offset: data_start + t.offset + 16]
    assert payload == struct.pack("<4f", 2.0, 2.0, 2.0, 2.0)


def test_missing_file_fails_closed(tmp_path):
    missing = tmp_path / "does_not_exist.gguf"
    out_path = tmp_path / "out.gguf"
    with pytest.raises(FileNotFoundError):
        slicer.slice_gguf(missing, 2, out_path)

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "slice_gguf_layers.py"),
         "--model-path", str(missing), "--max-layers", "2", "--output-path", str(out_path)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert result.stderr.strip() != ""
    assert not out_path.exists()


def test_non_gguf_file_fails_closed(tmp_path):
    bogus = tmp_path / "not_a_model.gguf"
    bogus.write_bytes(b"this is not a gguf file, just plain text padding")
    out_path = tmp_path / "out.gguf"

    with pytest.raises(ValueError):
        slicer.slice_gguf(bogus, 2, out_path)

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "slice_gguf_layers.py"),
         "--model-path", str(bogus), "--max-layers", "2", "--output-path", str(out_path)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert result.stderr.strip() != ""
    assert not out_path.exists()


def test_max_layers_exceeds_actual_fails_closed(mock_model, tmp_path):
    out_path = tmp_path / "out.gguf"
    with pytest.raises(ValueError):
        slicer.slice_gguf(mock_model, 10, out_path)

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "slice_gguf_layers.py"),
         "--model-path", str(mock_model), "--max-layers", "10", "--output-path", str(out_path)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert result.stderr.strip() != ""
    assert not out_path.exists()


def test_input_equals_output_path_fails_closed(mock_model):
    snapshot = mock_model.read_bytes()

    with pytest.raises(ValueError):
        slicer.slice_gguf(mock_model, 2, mock_model)

    # source untouched: same n_layer as originally built, and byte-for-byte identical.
    header = slicer.parse_header(mock_model)
    assert header.n_layer == 3
    assert mock_model.read_bytes() == snapshot

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "slice_gguf_layers.py"),
         "--model-path", str(mock_model), "--max-layers", "2", "--output-path", str(mock_model)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert result.stderr.strip() != ""
    assert mock_model.read_bytes() == snapshot


def test_write_failure_leaves_no_partial_output(mock_model, tmp_path, monkeypatch):
    out_path = tmp_path / "sliced.gguf"

    def raiser(*args, **kwargs):
        raise OSError("simulated disk failure during atomic install")

    monkeypatch.setattr(slicer.os, "replace", raiser)

    with pytest.raises(OSError):
        slicer.slice_gguf(mock_model, 2, out_path)

    assert not out_path.exists()
    leftover = list(out_path.parent.glob(f".{out_path.name}.tmp.*"))
    assert leftover == []


def test_unsupported_gguf_version_fails_closed(tmp_path):
    bad_version = tmp_path / "bad_version.gguf"
    build_mock_gguf(bad_version, version=99)
    out_path = tmp_path / "out.gguf"

    with pytest.raises(ValueError, match="version 99"):
        slicer.parse_header(bad_version)

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "slice_gguf_layers.py"),
         "--model-path", str(bad_version), "--max-layers", "2", "--output-path", str(out_path)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert result.stderr.strip() != ""
    assert not out_path.exists()


def test_non_power_of_two_alignment_fails_closed(tmp_path):
    bad_align = tmp_path / "bad_align.gguf"
    build_mock_gguf(bad_align, align=24)
    out_path = tmp_path / "out.gguf"

    with pytest.raises(ValueError, match="alignment"):
        slicer.parse_header(bad_align)

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "slice_gguf_layers.py"),
         "--model-path", str(bad_align), "--max-layers", "2", "--output-path", str(out_path)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert result.stderr.strip() != ""
    assert not out_path.exists()


def test_declared_layer_count_exceeds_actual_tensors_fails_closed(tmp_path):
    # llama.block_count claims 5 layers, but the tensor list (built by build_mock_gguf) only
    # ever goes up to blk.2 -- actual_max_block=2 < n_layer-1=4.
    mismatched = tmp_path / "mismatched_layers.gguf"
    build_mock_gguf(mismatched, n_layers=5)
    out_path = tmp_path / "out.gguf"

    with pytest.raises(ValueError, match="block_count"):
        slicer.slice_gguf(mismatched, 2, out_path)

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "slice_gguf_layers.py"),
         "--model-path", str(mismatched), "--max-layers", "2", "--output-path", str(out_path)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert result.stderr.strip() != ""
    assert not out_path.exists()


def test_launch_profiles_validate_sliced_file_structure(tmp_path):
    """A small serialized GGUF tests file contracts, not llama inference."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "suites"))
    from verify_gguf_model import main as verify_main

    source, sliced = tmp_path / "source.gguf", tmp_path / "slice.gguf"
    build_mock_gguf(source)
    slicer.slice_gguf(source, 2, sliced)
    assert verify_main([str(sliced), "--min-gb", "0", "--profile", "slice", "--expected-layers", "2"]) == 0
    with pytest.raises(ValueError, match="layer count"):
        verify_main([str(sliced), "--min-gb", "0", "--profile", "slice", "--expected-layers", "3"])
    with pytest.raises(ValueError, match="full profile"):
        verify_main([str(sliced), "--min-gb", "0", "--profile", "full", "--expected-layers", "2"])
