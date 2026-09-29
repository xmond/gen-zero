"""Tests for download_benchmark_features.py: a small fake manifest (one task) with
tiny in-memory-sized .npz byte stand-ins, so the hash arithmetic is real but the
test runs instantly. Covers: everything present and correct -> exit 0; a missing
file -> exit 1; a hash mismatch -> exit 1; MASTER_QWEN_DIR/MASTER_LLAMA_DIR env
resolution; and that --no-hash never reports "READY".
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import download_benchmark_features as dbf

QWEN_BYTES = b"fake-qwen72b-massive_en-features"
LLAMA_BYTES = b"fake-llama70b-massive_en-features"
QWEN_HASH = hashlib.sha256(QWEN_BYTES).hexdigest()
LLAMA_HASH = hashlib.sha256(LLAMA_BYTES).hexdigest()


@pytest.fixture
def fake_manifest(monkeypatch):
    """Shrinks the pinned manifest to a single task so tests run instantly."""
    monkeypatch.setattr(dbf, "TASKS", ("massive_en",))
    monkeypatch.setattr(dbf, "FEATURE_SHA256", {"massive_en": {"qwen": QWEN_HASH, "llama": LLAMA_HASH}})


@pytest.fixture
def feature_dirs(tmp_path):
    qdir = tmp_path / "qwen72b" / "features"
    ldir = tmp_path / "llama70b"
    qdir.mkdir(parents=True)
    ldir.mkdir(parents=True)
    (qdir / "massive_en.npz").write_bytes(QWEN_BYTES)
    (ldir / "massive_en.npz").write_bytes(LLAMA_BYTES)
    return qdir, ldir


def test_all_ok_exits_zero(fake_manifest, feature_dirs, capsys):
    qdir, ldir = feature_dirs
    rc = dbf.main(["--qwen-dir", str(qdir), "--llama-dir", str(ldir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "sha256 OK" in out
    assert "MISMATCH" not in out
    assert "MISSING" not in out


def test_missing_file_exits_one(fake_manifest, feature_dirs, capsys):
    qdir, ldir = feature_dirs
    (ldir / "massive_en.npz").unlink()
    rc = dbf.main(["--qwen-dir", str(qdir), "--llama-dir", str(ldir)])
    assert rc == 1
    captured = capsys.readouterr()
    assert "MISSING" in captured.out
    assert "no public mirror" in captured.out or "no public mirror" in captured.err


def test_mismatch_exits_one(fake_manifest, feature_dirs, capsys):
    qdir, ldir = feature_dirs
    (ldir / "massive_en.npz").write_bytes(b"corrupted-bytes-not-matching-the-manifest")
    rc = dbf.main(["--qwen-dir", str(qdir), "--llama-dir", str(ldir)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "MISMATCH" in out


def test_env_var_resolution(fake_manifest, feature_dirs, monkeypatch, tmp_path, capsys):
    qdir, ldir = feature_dirs
    monkeypatch.setenv("MASTER_QWEN_DIR", str(qdir))
    monkeypatch.setenv("MASTER_LLAMA_DIR", str(ldir))
    json_out = tmp_path / "status.json"
    rc = dbf.main(["--json", str(json_out)])
    assert rc == 0
    payload = json.loads(json_out.read_text())
    assert payload["qwen_dir"] == str(qdir)
    assert payload["llama_dir"] == str(ldir)
    assert "MASTER_QWEN_DIR" in payload["qwen_dir_source"]
    assert "MASTER_LLAMA_DIR" in payload["llama_dir_source"]
    assert payload["all_ok"] is True


def test_cli_arg_overrides_env_var(fake_manifest, feature_dirs, monkeypatch, tmp_path):
    qdir, ldir = feature_dirs
    # Point the env vars somewhere wrong; the explicit CLI args must win.
    monkeypatch.setenv("MASTER_QWEN_DIR", str(tmp_path / "wrong_qwen"))
    monkeypatch.setenv("MASTER_LLAMA_DIR", str(tmp_path / "wrong_llama"))
    json_out = tmp_path / "status.json"
    rc = dbf.main(["--qwen-dir", str(qdir), "--llama-dir", str(ldir), "--json", str(json_out)])
    assert rc == 0
    payload = json.loads(json_out.read_text())
    assert payload["qwen_dir"] == str(qdir)
    assert payload["llama_dir"] == str(ldir)


def test_no_hash_never_reports_ready(fake_manifest, feature_dirs, capsys):
    qdir, ldir = feature_dirs
    # Corrupt one file's bytes; --no-hash must still say present, not catch the mismatch.
    (ldir / "massive_en.npz").write_bytes(b"corrupted-but-existence-check-only")
    rc = dbf.main(["--qwen-dir", str(qdir), "--llama-dir", str(ldir), "--no-hash"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "READY" not in out
    assert "PRESENT (unverified)" in out
    assert "hash NOT verified" in out or "hashes NOT verified" in out or "NOT verified" in out


def test_no_hash_missing_file_still_fails(fake_manifest, feature_dirs, capsys):
    qdir, ldir = feature_dirs
    (ldir / "massive_en.npz").unlink()
    rc = dbf.main(["--qwen-dir", str(qdir), "--llama-dir", str(ldir), "--no-hash"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "READY" not in out
    assert "MISSING" in out


def test_bad_args_exit_two():
    with pytest.raises(SystemExit):
        dbf.parse_args(["--not-a-real-flag"])
    rc = dbf.main(["--not-a-real-flag"])
    assert rc == 2
