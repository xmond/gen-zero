"""Unit tests for scripts/download_model_fleet.py.

No network calls. Every test that would otherwise touch the hub monkeypatches
``dl.fetch_listing`` (or the huggingface_hub internals it wraps) with fixed data, following
the same pattern as ``test_gpu_extract_qwen72b.py``'s downloader tests. The registry's real
repo_ids, file names and byte sizes were checked against the live hub on 2026-09-24 (see the
docstring of download_model_fleet.py); this file does not re-verify network reachability, only
the script's own logic.
"""
import hashlib
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import download_model_fleet as dl  # noqa: E402


# ------------------------------------------------------------------ registry sanity

def test_registry_has_exactly_the_five_requested_models():
    assert set(dl.MODEL_REGISTRY) == {"llama70b", "gemma27b", "mixtral8x7b", "deepseek_v2", "gte_qwen7b"}


def test_registry_entries_are_internally_consistent():
    subdirs = set()
    for key, spec in dl.MODEL_REGISTRY.items():
        assert spec.key == key
        assert "/" in spec.repo_id
        assert spec.pattern
        assert spec.subdir not in subdirs, f"duplicate subdir {spec.subdir!r}"
        subdirs.add(spec.subdir)
        assert 0 < spec.min_total_gb < spec.max_total_gb < 200
        assert spec.architecture in {"dense", "moe", "moe-mla", "embedding"}


def test_registry_local_dir_honors_the_local_root_argument():
    spec = dl.MODEL_REGISTRY["gte_qwen7b"]
    assert spec.local_dir(Path("D:/models")) == Path("D:/models/gte-qwen2-7b-instruct")
    assert spec.local_dir(Path("/tmp/other")) == Path("/tmp/other/gte-qwen2-7b-instruct")


# ------------------------------------------------------------------ --models parsing

def test_resolve_models_all_returns_full_registry_in_registry_order():
    specs = dl.resolve_models("all")
    assert [s.key for s in specs] == list(dl.MODEL_REGISTRY)


def test_resolve_models_subset_keeps_registry_order_not_input_order():
    specs = dl.resolve_models("gte_qwen7b,llama70b")
    assert [s.key for s in specs] == ["llama70b", "gte_qwen7b"]


def test_resolve_models_dedupes_repeated_keys():
    specs = dl.resolve_models("llama70b,llama70b")
    assert [s.key for s in specs] == ["llama70b"]


def test_resolve_models_rejects_unknown_key():
    with pytest.raises(dl.DownloadError) as e:
        dl.resolve_models("llama70b,nope")
    assert e.value.code == 3
    assert "nope" in str(e.value)


def test_resolve_models_is_case_and_whitespace_tolerant_for_all():
    assert [s.key for s in dl.resolve_models("  ALL  ")] == list(dl.MODEL_REGISTRY)


# ------------------------------------------------------------------ select_shards (single-file + split)

def test_select_shards_resolves_a_single_file():
    siblings = [dl.Shard("gemma-2-27b-it-Q4_K_M.gguf", 16_645_381_632, "a" * 64),
               dl.Shard("gemma-2-27b-it-Q8_0.gguf", 30_000_000_000, None),
               dl.Shard("README.md", 100, None)]
    got = dl.select_shards(siblings, "gemma-2-27b-it-Q4_K_M")
    assert [s.name for s in got] == ["gemma-2-27b-it-Q4_K_M.gguf"]


def test_select_shards_accepts_a_trailing_gguf_on_the_pattern():
    siblings = [dl.Shard("m-q4_k_m.gguf", 10, None)]
    assert [s.name for s in dl.select_shards(siblings, "m-q4_k_m.gguf")] == ["m-q4_k_m.gguf"]


def test_select_shards_resolves_a_complete_split_set_in_order():
    siblings = [dl.Shard(f"m-q4_k_m-{i:05d}-of-00003.gguf", 10, None) for i in (2, 1, 3)]
    got = dl.select_shards(siblings, "m-q4_k_m")
    assert [s.name for s in got] == ["m-q4_k_m-00001-of-00003.gguf", "m-q4_k_m-00002-of-00003.gguf",
                                     "m-q4_k_m-00003-of-00003.gguf"]


@pytest.mark.parametrize("case", ["no_match", "missing_shard", "single_and_split", "mixed_totals"])
def test_select_shards_refuses_incomplete_or_ambiguous_sets(case):
    siblings = [dl.Shard(f"m-q4_k_m-{i:05d}-of-00003.gguf", 10, None) for i in (1, 2, 3)]
    if case == "no_match":
        siblings = [s for s in siblings if False]
    elif case == "missing_shard":
        siblings = [s for s in siblings if "00002-of" not in s.name]
    elif case == "single_and_split":
        siblings = siblings + [dl.Shard("m-q4_k_m.gguf", 10, None)]
    elif case == "mixed_totals":
        siblings = siblings + [dl.Shard("m-q4_k_m-00004-of-00004.gguf", 10, None)]
    with pytest.raises(dl.DownloadError) as e:
        dl.select_shards(siblings, "m-q4_k_m")
    assert e.value.code == 3


# ------------------------------------------------------------------ size range gate

def test_check_total_gb_accepts_inside_range_and_refuses_outside():
    shards = [dl.Shard("a", int(42.52e9), None)]
    assert dl.check_total_gb(shards, 40.0, 45.0, "x") == pytest.approx(42.52, abs=1e-2)
    with pytest.raises(dl.DownloadError) as e:
        dl.check_total_gb(shards, 50.0, 60.0, "x")
    assert e.value.code == 4


def test_every_registry_entry_hub_verified_size_is_inside_its_own_range():
    verified_bytes = {
        "llama70b": 42_520_398_400,
        "gemma27b": 16_645_381_632,
        "mixtral8x7b": 28_448_468_384,
        "deepseek_v2": 10_364_416_768,
        "gte_qwen7b": 8_095_329_472,
    }
    for key, size in verified_bytes.items():
        spec = dl.MODEL_REGISTRY[key]
        got = dl.check_total_gb([dl.Shard("f", size, None)], spec.min_total_gb, spec.max_total_gb, key)
        assert got == pytest.approx(size / dl.GB, rel=1e-6)


def test_fleet_total_matches_the_documented_estimate():
    total = sum({"llama70b": 42_520_398_400, "gemma27b": 16_645_381_632, "mixtral8x7b": 28_448_468_384,
                "deepseek_v2": 10_364_416_768, "gte_qwen7b": 8_095_329_472}.values()) / dl.GB
    assert 100.0 <= total <= 110.0


# ------------------------------------------------------------------ disk gate

def test_check_disk_blocks_below_and_allows_at_the_threshold(tmp_path, monkeypatch):
    target = tmp_path / "models"
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda p: SimpleNamespace(total=1, used=1, free=int(49.9e9)))
    with pytest.raises(dl.DownloadError, match="need at least 50 GB") as e:
        dl.check_disk(target, 50.0)
    assert e.value.code == 2 and target.is_dir()
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda p: SimpleNamespace(total=1, used=1, free=int(50e9)))
    assert dl.check_disk(target, 50.0) == pytest.approx(50.0)


def test_default_min_free_gb_matches_the_task_brief():
    assert dl.DEFAULT_MIN_FREE_GB == 50.0
    assert dl.build_parser().parse_args([]).min_free_gb == 50.0


# ------------------------------------------------------------------ verify_sizes / sha256

def write_shard(dirpath, shard, magic=b"GGUF"):
    with (dirpath / shard.name).open("wb") as fh:
        fh.write(magic + b"\0" * (shard.size - 4))


def test_verify_sizes_flags_missing_wrong_size_and_bad_magic(tmp_path):
    shards = [dl.Shard(f"m-{i}.gguf", 64, None) for i in range(3)]
    for s in shards:
        write_shard(tmp_path, s)
    assert dl.verify_sizes(tmp_path, shards) == []
    (tmp_path / shards[0].name).write_bytes(b"GGUF" + b"\0" * 10)      # short
    (tmp_path / shards[1].name).write_bytes(b"XXXX" + b"\0" * 60)      # right size, wrong magic
    (tmp_path / shards[2].name).unlink()                                # missing
    problems = dl.verify_sizes(tmp_path, shards)
    assert len(problems) == 3
    assert any("bytes on disk" in p for p in problems)
    assert any("magic" in p for p in problems)
    assert any("missing" in p for p in problems)


def test_verify_sizes_skips_the_magic_check_for_non_gguf_files(tmp_path):
    shard = dl.Shard("weights.safetensors", 8, None)
    (tmp_path / shard.name).write_bytes(b"notgguf!")
    assert dl.verify_sizes(tmp_path, [shard]) == []


def test_verify_sha256_matches_and_mismatches(tmp_path):
    shards = [dl.Shard(f"m-{i}.gguf", 64, None) for i in range(2)]
    for s in shards:
        write_shard(tmp_path, s)
    good = [dl.Shard(s.name, s.size, hashlib.sha256((tmp_path / s.name).read_bytes()).hexdigest()) for s in shards]
    assert dl.verify_sha256(tmp_path, good) == []
    bad = [good[0], dl.Shard(good[1].name, good[1].size, "0" * 64)]
    assert len(dl.verify_sha256(tmp_path, bad)) == 1
    assert "no sha256" in dl.verify_sha256(tmp_path, [dl.Shard(good[0].name, 64, None)])[0]


def test_download_kwargs_only_passes_resume_when_the_signature_accepts_it():
    def old_hub(repo_id, filename, resume_download=False, local_dir=None, revision=None): ...
    def new_hub(repo_id, filename, local_dir=None, revision=None): ...
    assert dl.download_kwargs(old_hub) == {"resume_download": True}
    assert dl.download_kwargs(new_hub) == {}
    from huggingface_hub import hf_hub_download
    assert bool(dl.download_kwargs(hf_hub_download)) == ("resume_download" in inspect.signature(hf_hub_download).parameters)


# ------------------------------------------------------------------ fetch_listing error handling

def test_fetch_listing_wraps_a_missing_repo_as_a_download_error(monkeypatch):
    import httpx
    from huggingface_hub.errors import RepositoryNotFoundError

    resp = httpx.Response(401, request=httpx.Request("GET", "https://huggingface.co/api/models/x"))

    class FakeApi:
        def model_info(self, repo_id, revision=None, files_metadata=False):
            raise RepositoryNotFoundError("nope", response=resp)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    with pytest.raises(dl.DownloadError) as e:
        dl.fetch_listing("bartowski/does-not-exist", None)
    assert e.value.code == 5


# ------------------------------------------------------------------ dry-run
#
# These use the real, hub-verified multi-GB sizes as plain ints (cheap: an int, not bytes on
# disk). dry-run never opens a file for writing, so this is safe.

@pytest.fixture
def hub_data():
    return {
        "llama70b": ("commit-llama", [dl.Shard("Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf", int(42.52e9), None)]),
        "gemma27b": ("commit-gemma", [dl.Shard("gemma-2-27b-it-Q4_K_M.gguf", int(16.65e9), None)]),
        "mixtral8x7b": ("commit-mix", [dl.Shard("Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf", int(28.45e9), None)]),
        "deepseek_v2": ("commit-ds", [dl.Shard("DeepSeek-V2-Lite-Chat.Q4_K_M.gguf", int(10.36e9), None)]),
        "gte_qwen7b": ("commit-gte", [dl.Shard("gte-Qwen2-7B-instruct.Q8_0.gguf", int(8.10e9), None)]),
    }


@pytest.fixture
def hub(monkeypatch, hub_data):
    monkeypatch.setattr(dl, "fetch_listing", lambda repo_id, revision: next(
        v for k, v in hub_data.items() if dl.MODEL_REGISTRY[k].repo_id == repo_id))
    return hub_data


def test_dry_run_downloads_nothing_and_reports_every_model(tmp_path, hub, capsys):
    code = dl.run_dry_run(dl.resolve_models("all"), tmp_path, 50.0)
    assert code == 0
    out = capsys.readouterr().out
    for key in dl.MODEL_REGISTRY:
        assert key in out
        assert not (tmp_path / dl.MODEL_REGISTRY[key].subdir).exists() or \
            not any((tmp_path / dl.MODEL_REGISTRY[key].subdir).iterdir())


def test_dry_run_stops_and_returns_the_error_code_on_a_bad_listing(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "fetch_listing", lambda repo_id, revision:
                        (_ for _ in ()).throw(dl.DownloadError("boom", 5)))
    assert dl.run_dry_run(dl.resolve_models("llama70b"), tmp_path, 50.0) == 5


# ------------------------------------------------------------------ run_fleet + main
#
# These exercise a real (fake) download: bytes actually get written to tmp_path. Using the real
# multi-GB sizes here would write tens of GB of zero-filled files per test run (this cost the
# session a genuine 15 GB /tmp blowup and a 33 GB-RSS pytest process during development -- see
# `write_shard`, which allocates `size` bytes). So these tests fake the network layer
# (`fetch_listing`) with tiny 64-byte shards and relax `check_total_gb`'s range check (which has
# its own dedicated tests above, against the real hub-verified byte counts) so a tiny fake file
# doesn't get rejected against a real model's 10-45 GB expected range. `plan_model` itself,
# `check_disk`, and `verify_sizes` all run for real, unmodified.

TINY_SHARDS = {
    "llama70b": dl.Shard("Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf", 64, None),
    "gemma27b": dl.Shard("gemma-2-27b-it-Q4_K_M.gguf", 64, None),
    "mixtral8x7b": dl.Shard("Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf", 64, None),
    "deepseek_v2": dl.Shard("DeepSeek-V2-Lite-Chat.Q4_K_M.gguf", 64, None),
    "gte_qwen7b": dl.Shard("gte-Qwen2-7B-instruct.Q8_0.gguf", 64, None),
}


@pytest.fixture
def tiny_hub(monkeypatch):
    def fake_fetch(repo_id, revision):
        key = next(k for k, spec in dl.MODEL_REGISTRY.items() if spec.repo_id == repo_id)
        return f"commit-{key}", [TINY_SHARDS[key]]

    monkeypatch.setattr(dl, "fetch_listing", fake_fetch)
    monkeypatch.setattr(dl, "check_total_gb", lambda shards, low, high, what: sum(s.size for s in shards) / dl.GB)
    return TINY_SHARDS


def fake_downloader_factory(shard_map):
    def fake_dl(repo_id, filename, revision, local_dir, resume_download=False):
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        for s in shard_map.values():
            if s.name == filename:
                write_shard(Path(local_dir), s)
                return
        raise AssertionError(f"unexpected filename {filename}")
    return fake_dl


def test_run_fleet_downloads_and_verifies_every_model(tmp_path, tiny_hub):
    results = dl.run_fleet(dl.resolve_models("all"), tmp_path, 50.0, False,
                           downloader=fake_downloader_factory(tiny_hub))
    assert [r.status for r in results] == ["ok"] * 5
    for key, spec in dl.MODEL_REGISTRY.items():
        assert (tmp_path / spec.subdir).is_dir()
        assert (tmp_path / spec.subdir / TINY_SHARDS[key].name).stat().st_size == 64


def test_run_fleet_disk_gate_stops_the_queue_and_marks_the_rest_not_attempted(tmp_path, tiny_hub, monkeypatch):
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda p: SimpleNamespace(total=1, used=1, free=int(10e9)))
    results = dl.run_fleet(dl.resolve_models("all"), tmp_path, 50.0, False,
                           downloader=lambda **kw: pytest.fail("must not download past the disk gate"))
    assert results[0].status == "disk-gate"
    assert [r.status for r in results[1:]] == ["not-attempted"] * 4


def test_run_fleet_continues_past_one_failed_model(tmp_path, tiny_hub, monkeypatch):
    real_plan = dl.plan_model

    def flaky_plan(spec, **kw):
        if spec.key == "gemma27b":
            raise dl.DownloadError("gemma27b: simulated failure", 4)
        return real_plan(spec, **kw)

    monkeypatch.setattr(dl, "plan_model", flaky_plan)
    results = dl.run_fleet(dl.resolve_models("all"), tmp_path, 50.0, False,
                           downloader=fake_downloader_factory(tiny_hub))
    by_key = {r.key: r.status for r in results}
    assert by_key["gemma27b"] == "failed"
    assert by_key["llama70b"] == "ok" and by_key["gte_qwen7b"] == "ok"


def test_main_exit_codes(tmp_path, tiny_hub, monkeypatch):
    assert dl.main(["--dry-run", "--local-root", str(tmp_path), "--models", "llama70b"]) == 0
    assert dl.main(["--local-root", str(tmp_path), "--models", "nope"]) == 3
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda p: SimpleNamespace(total=1, used=1, free=int(1e9)))
    assert dl.main(["--local-root", str(tmp_path / "gated"), "--models", "gte_qwen7b"]) == 2


def test_main_reports_1_when_a_model_fails_without_a_disk_problem(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "plan_model", lambda spec, **kw: (_ for _ in ()).throw(dl.DownloadError("boom", 4)))
    assert dl.main(["--local-root", str(tmp_path), "--models", "gte_qwen7b"]) == 1


# ------------------------------------------------------------------ argument defaults

def test_build_parser_defaults():
    args = dl.build_parser().parse_args([])
    assert args.models == "all"
    assert str(args.local_root).replace("\\", "/") == "D:/models"
    assert args.min_free_gb == 50.0
    assert args.sha256 is False
    assert args.dry_run is False


def test_build_parser_accepts_a_model_subset_and_flags():
    args = dl.build_parser().parse_args(["--models", "llama70b,gte_qwen7b", "--sha256", "--dry-run",
                                         "--min-free-gb", "75"])
    assert args.models == "llama70b,gte_qwen7b"
    assert args.sha256 is True and args.dry_run is True
    assert args.min_free_gb == 75.0
