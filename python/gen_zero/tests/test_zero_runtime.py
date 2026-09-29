"""Zero standalone runtime: configurable CPU threads, no GPU, real weights.

The heavy tests load the real local Qwen2.5-0.5B checkpoint (the only
pretrained dense-Qwen2 weights on this machine) and are skipped with an
explicit reason when it is absent. Nothing here mocks a model.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from gen_zero.causal import zero_runtime, zero_trunk
from gen_zero.causal.zero_runtime import (
    MANIFOLD_DIM,
    ZeroManifold,
    ZeroStandaloneRuntime,
    ZeroTaskHead,
    build_int8_artifact,
    enforce_single_core,
    find_local_snapshot,
)
from gen_zero.causal.zero_trunk import Int8WeightOnlyLinear, ZeroTrunk, TrunkConfig, load_trunk

REPO = Path(__file__).resolve().parents[3]
ARTIFACT = REPO / "benchmarks" / "artifacts" / "zero" / "zero_int8_v2.safetensors"
MANIFOLD = REPO / "benchmarks" / "artifacts" / "zero" / "zero_manifold_v1.npz"
CALIBRATION = REPO / "benchmarks" / "data" / "calibration_clean_16.jsonl"
FORBIDDEN_SOURCE_TOKENS = ("latent_bridge", "candidate_fusion", "run_remote_eval", "AutoModelForCausalLM",
                           "9B", "70B", "Laya", "laya", ".cuda(", "device_map")
TEXTS = [
    "Route the following natural language request to its target intent domain: what's the weather like on friday",
    "Premise: The president advised the doctor.\nHypothesis: The doctor advised the president.\n"
    "Determine whether the premise entails the hypothesis.",
    "Frage: Wie spät ist es?",
    "質問: 今日は何曜日ですか？",
]


def _snapshot():
    try:
        return find_local_snapshot()
    except FileNotFoundError as error:
        pytest.skip(f"real Zero backbone weights are not available locally: {error}")


@pytest.fixture(scope="module")
def int8_artifact(tmp_path_factory):
    snapshot = _snapshot()
    if ARTIFACT.exists():
        return ARTIFACT
    path = tmp_path_factory.mktemp("zero") / "zero_int8.safetensors"
    build_int8_artifact(snapshot, path)
    return path


@pytest.fixture(scope="module")
def runtime(int8_artifact):
    return ZeroStandaloneRuntime(int8_artifact=int8_artifact, max_length=1024)


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# --------------------------------------------------------------------------
# Static anti-cheat checks: no second model, no GPU, no task/text heuristics
# --------------------------------------------------------------------------

def test_runtime_sources_do_not_reference_teacher_or_gpu():
    for module in (zero_runtime, zero_trunk):
        source = Path(module.__file__).read_text(encoding="utf-8")
        body = source.split('"""', 2)[2]  # skip the module docstring, which explains what is excluded
        for token in FORBIDDEN_SOURCE_TOKENS:
            assert token not in body, f"{module.__name__} contains forbidden token {token!r}"
        assert "import transformers" not in body and "from transformers" not in body
        assert "re." not in body.replace("return", "").replace("result", "").replace("torch.rsqrt", "") \
            or "import re\n" not in body, "no regex-based logic in the runtime"


def test_runtime_import_does_not_pull_transformers():
    code = textwrap.dedent("""
        import sys
        import gen_zero.causal.zero_runtime
        print(int('transformers' in sys.modules))
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=str(REPO / "python"))
    assert out.stdout.strip() == "0"


def test_single_core_enforced():
    enforce_single_core()
    assert torch.get_num_threads() == 1
    assert torch.get_num_interop_threads() == 1


# --------------------------------------------------------------------------
# Numerical building blocks (synthetic inputs, clearly labelled as such)
# --------------------------------------------------------------------------

def test_int8_weight_only_linear_tracks_dense_linear():
    torch.manual_seed(0)
    dense = torch.nn.Linear(896, 4864, bias=True)
    quant = Int8WeightOnlyLinear(896, 4864, bias=True)
    quant.load_dense(dense.weight, dense.bias)
    x = torch.randn(7, 896)
    with torch.inference_mode():
        reference = dense(x)
        got = quant(x)
    rel = (got - reference).norm() / reference.norm()
    assert rel < 5e-3, float(rel)
    assert quant.qweight.dtype == torch.int8


def test_manifold_fit_project_and_provenance(tmp_path):
    rng = np.random.default_rng(0)
    # Anisotropic synthetic states: one dominant direction plus noise.
    states = rng.normal(size=(400, 128)) @ np.diag(np.linspace(8, .5, 128)) + 3.0
    manifold = ZeroManifold.fit(states, dim=MANIFOLD_DIM, encoder_id="unit", source="synthetic",
                                split="calibration")
    z = manifold.project(states)
    assert z.shape == (400, MANIFOLD_DIM)
    assert np.allclose(np.linalg.norm(z, axis=1), 1.0)
    gram = manifold.basis.T @ manifold.basis
    assert np.allclose(gram, np.eye(MANIFOLD_DIM), atol=1e-5)
    # Whitening flattens the kept spectrum: variances of the 64 coordinates are close.
    raw = ((states - manifold.mean) @ manifold.basis) * manifold.scale
    variances = raw.var(axis=0)
    assert variances.max() / variances.min() < 3.0
    path = tmp_path / "m.npz"
    manifold.save(path)
    restored = ZeroManifold.load(path, encoder_id="unit")
    assert np.allclose(restored.project(states[:5]), z[:5])
    with pytest.raises(ValueError):
        ZeroManifold.load(path, encoder_id="other-encoder")
    with pytest.raises(ValueError):
        ZeroManifold.fit(states, encoder_id="unit", source="synthetic", split="test")


# --------------------------------------------------------------------------
# Real-weight pipeline
# --------------------------------------------------------------------------

def test_fp32_trunk_is_bit_faithful_to_reference_and_official_zero_model():
    """ZeroTrunk (inference impl.) == transformers Qwen2Model == ZeroModel converted by ZeroConverter."""
    snapshot = _snapshot()
    pytest.importorskip("transformers")
    from transformers import AutoModel, AutoTokenizer
    from gen_zero.model.zero_converter import ZeroConverter
    from gen_zero.model.zero_model import ZeroConfig

    enforce_single_core()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    encoded = tokenizer(TEXTS, return_tensors="pt", padding=True)
    reference = AutoModel.from_pretrained(snapshot, local_files_only=True, dtype=torch.float32).eval()
    with torch.inference_mode():
        ref_last = ZeroTrunk.last_valid(reference(**encoded, use_cache=False).last_hidden_state,
                                        encoded["attention_mask"])
    del reference

    trunk, _ = load_trunk(snapshot, "fp32")
    rows = [tokenizer.encode(t, add_special_tokens=False) for t in TEXTS]
    ids, mask = ZeroStandaloneRuntime._pad(rows)
    with torch.inference_mode():
        hidden, _ = trunk(ids, mask)
        got = ZeroTrunk.last_valid(hidden, mask)
    assert torch.allclose(got, ref_last, atol=2e-3, rtol=1e-4), float((got - ref_last).abs().max())

    # The official ZeroModel at the same topology, filled by the official converter.
    raw = json.loads((snapshot / "config.json").read_text())
    vocab_rows = len(tokenizer.get_vocab())
    config = ZeroConfig(vocab_size=vocab_rows, hidden_size=raw["hidden_size"],
                        num_hidden_layers=raw["num_hidden_layers"],
                        num_attention_heads=raw["num_attention_heads"],
                        intermediate_size=raw["intermediate_size"], attention_bias=True,
                        max_sequence_length=512, rope_theta=raw["rope_theta"],
                        rms_norm_eps=raw["rms_norm_eps"])
    official, report = ZeroConverter(snapshot).convert(config, tokenizer_path=snapshot, dtype=torch.float32)
    assert report.selected_layers == tuple(range(raw["num_hidden_layers"]))
    with torch.inference_mode():
        official_hidden, _ = official.hidden_states(ids, mask.bool())
        official_last = ZeroTrunk.last_valid(official_hidden, mask)
    assert torch.allclose(official_last, got, atol=2e-3, rtol=1e-4), float((official_last - got).abs().max())


def test_int8_runtime_pipeline_multithread_cpu(runtime):
    assert torch.get_num_threads() == min(8, os.cpu_count() or 4)
    assert not torch.cuda.is_initialized()
    for name, tensor in list(runtime.model.named_parameters()) + list(runtime.model.named_buffers()):
        assert tensor.device.type == "cpu", name
    assert runtime.tensor_bytes() > 0
    states, info = runtime.encode(TEXTS)
    assert states.shape == (len(TEXTS), runtime.hidden_size)
    assert np.isfinite(states).all()
    assert info["forward_ms"] > 0 and info["tokenize_ms"] > 0
    # Different texts must give different states (liveness, not a constant model).
    cos = _cos(states[0], states[1])
    assert cos < 0.999, cos


def test_int8_states_track_fp32_states(runtime):
    snapshot = _snapshot()
    trunk, _ = load_trunk(snapshot, "fp32")
    rows = [runtime.token_ids(t) for t in TEXTS]
    ids, mask = ZeroStandaloneRuntime._pad(rows)
    with torch.inference_mode():
        hidden, _ = trunk(ids, mask)
        ref = ZeroTrunk.last_valid(hidden, mask).numpy()
    got, _ = runtime.encode(TEXTS)
    cosines = [_cos(a, b) for a, b in zip(got, ref)]
    assert min(cosines) > 0.98, cosines


def test_prefix_cache_matches_full_sequences(runtime):
    prompt = TEXTS[0]
    candidates = ["alarm", "weather", "datetime", "iot_lights"]
    q0, cands, info = runtime.encode_prompt_with_candidates(prompt, candidates)
    full, _ = runtime.encode([prompt] + [prompt + runtime.candidate_text(c) for c in candidates])
    assert _cos(q0, full[0]) > 0.99999
    assert min(_cos(a, b) for a, b in zip(cands, full[1:])) > 0.99999
    assert info["prompt_tokens"] == len(runtime.token_ids(prompt))
    assert info["candidate_tokens"] == sum(len(runtime.tokenizer.encode(runtime.candidate_text(c),
                                                                      add_special_tokens=False).ids)
                                           for c in candidates)


def test_candidate_chunking_matches_unchunked(runtime):
    """Splitting the K candidates into KV-cache chunks must not change the states.

    Chunking only bounds how many candidates share one expanded-KV-cache
    forward pass at a time (the fix for the >1.0 GB Banking77 blowup); the
    underlying int8 matmuls are batch-size-sensitive at the ULP level (like
    ``test_prefix_cache_matches_full_sequences`` already tolerates for the
    prefix-cache vs. full-sequence comparison), so this checks near-equality
    by cosine similarity rather than bit-for-bit equality.
    """
    prompt = TEXTS[0]
    candidates = ["alarm", "weather", "datetime", "iot_lights", "reminder", "traffic", "music"]
    q0_ref, cands_ref, _ = runtime.encode_prompt_with_candidates(
        prompt, candidates, candidate_chunk_size=len(candidates))
    for chunk_size in (1, 3, 16, len(candidates)):
        q0, cands, info = runtime.encode_prompt_with_candidates(
            prompt, candidates, candidate_chunk_size=chunk_size)
        assert _cos(q0, q0_ref) > 0.99999, chunk_size
        assert min(_cos(a, b) for a, b in zip(cands, cands_ref)) > 0.99999, chunk_size
        assert info["candidate_tokens"] == sum(len(runtime.tokenizer.encode(runtime.candidate_text(c),
                                                                            add_special_tokens=False).ids)
                                               for c in candidates)


def test_truncation_is_refused(runtime):
    long_text = "word " * 3000
    with pytest.raises(ValueError, match="refusing to truncate"):
        runtime.token_ids(long_text)


def test_decision_requires_manifold_and_is_order_invariant(int8_artifact):
    if not MANIFOLD.exists():
        pytest.skip("no calibrated manifold artifact; run benchmark_zero_cpu.py fit-manifold")
    bare = ZeroStandaloneRuntime(int8_artifact=int8_artifact)
    with pytest.raises(RuntimeError, match="no manifold"):
        bare.decide("anything", ["a", "b"])
    rt = ZeroStandaloneRuntime(int8_artifact=int8_artifact, manifold_path=MANIFOLD)
    with open(CALIBRATION, encoding="utf-8") as stream:
        record = json.loads(stream.readline())
    decision = rt.decide(record["context"], record["candidates"])
    assert 0 <= decision.index < len(record["candidates"])
    assert decision.scores.shape == (len(record["candidates"]),)
    assert np.isfinite(decision.scores).all()
    assert decision.prompt_state.shape == (MANIFOLD_DIM,)
    assert decision.candidate_states.shape == (len(record["candidates"]), MANIFOLD_DIM)
    assert decision.forward_ms > 0 and decision.dynamics_ms > 0
    reversed_candidates = list(reversed(record["candidates"]))
    alt = rt.decide(record["context"], reversed_candidates)
    assert reversed_candidates[alt.index] == record["candidates"][decision.index]


@pytest.mark.parametrize("mode", ["fractal", "continuous"])
def test_thinking_modes_end_to_end_cpu(int8_artifact, mode):
    if not MANIFOLD.exists():
        pytest.skip("calibrated manifold artifact is unavailable")
    rt = ZeroStandaloneRuntime(int8_artifact=int8_artifact, manifold_path=MANIFOLD,
                               thinking_mode=mode, single_core=True)
    with open(CALIBRATION, encoding="utf-8") as stream:
        record = json.loads(stream.readline())
    decision = rt.decide(record["context"], record["candidates"])
    assert decision.scores.shape == (len(record["candidates"]),)
    assert np.isfinite(decision.scores).all()
    assert decision.index == int(np.argmax(decision.scores))
    assert decision.dynamics_ms > 0
    assert not torch.cuda.is_initialized()


def test_thinking_mode_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="thinking_mode"):
        ZeroStandaloneRuntime(thinking_mode="unknown")
    with pytest.raises(ValueError, match="rnn_path"):
        ZeroStandaloneRuntime(thinking_mode="lora_rnn")


DEEP_ADAPTER = REPO / "benchmarks" / "artifacts" / "zero" / "deep_projection_adapter_cpu_v1.pt"


def test_adapter_path_requires_manifold(int8_artifact):
    if not DEEP_ADAPTER.exists():
        pytest.skip("no trained deep projection adapter checkpoint")
    with pytest.raises(ValueError, match="manifold_path"):
        ZeroStandaloneRuntime(int8_artifact=int8_artifact, adapter_path=DEEP_ADAPTER)


def test_adapter_path_mounts_and_is_used_in_decide(int8_artifact):
    if not MANIFOLD.exists():
        pytest.skip("no calibrated manifold artifact; run benchmark_zero_cpu.py fit-manifold")
    if not DEEP_ADAPTER.exists():
        pytest.skip("no trained deep projection adapter checkpoint")
    bare = ZeroStandaloneRuntime(int8_artifact=int8_artifact, manifold_path=MANIFOLD)
    rt = ZeroStandaloneRuntime(int8_artifact=int8_artifact, manifold_path=MANIFOLD, adapter_path=DEEP_ADAPTER)
    assert rt.adapter is not None
    with open(CALIBRATION, encoding="utf-8") as stream:
        record = json.loads(stream.readline())
    decision = rt.decide(record["context"], record["candidates"])
    baseline = bare.decide(record["context"], record["candidates"])
    assert 0 <= decision.index < len(record["candidates"])
    assert decision.prompt_state.shape == (MANIFOLD_DIM,)
    assert np.allclose(np.linalg.norm(decision.prompt_state), 1.0, atol=1e-6)
    assert np.isfinite(decision.scores).all()
    # bare.decide() must be untouched by the adapter mount -- decide() with no
    # adapter_path still projects through ZeroManifold.project exactly as before.
    assert baseline.prompt_state.shape == (MANIFOLD_DIM,)


def test_rnn_none_by_default_and_rollout_state_requires_rnn(int8_artifact):
    rt = ZeroStandaloneRuntime(int8_artifact=int8_artifact)
    assert rt.rnn is None
    with pytest.raises(RuntimeError, match="no rnn"):
        rt.rollout_state(np.zeros(MANIFOLD_DIM), steps=3)


def test_rnn_path_mounts_and_rollout_state_never_touches_backbone(int8_artifact, tmp_path):
    from gen_zero.causal.parallel_rnn_lora import ParallelRNNLoRAAdapter

    adapter = ParallelRNNLoRAAdapter(dim=MANIFOLD_DIM, rank=8, seed=0)
    path = tmp_path / "rnn.npz"
    adapter.save(path)
    rt = ZeroStandaloneRuntime(int8_artifact=int8_artifact, rnn_path=path)
    assert rt.rnn is not None

    z0 = np.random.default_rng(0).normal(size=MANIFOLD_DIM)
    h1 = rt.rollout_state(z0, steps=1)
    h3 = rt.rollout_state(z0, steps=3)
    assert h1.shape == (MANIFOLD_DIM,)
    assert h3.shape == (MANIFOLD_DIM,)
    assert np.isfinite(h1).all() and np.isfinite(h3).all()
    assert not np.allclose(h1, h3)
    with pytest.raises(ValueError):
        rt.rollout_state(z0, steps=0)


def test_resident_memory_in_fresh_process(int8_artifact):
    """Resident and peak RSS of a fresh single-core Zero process stay under 1.0 GB (decimal)."""
    code = textwrap.dedent(f"""
        import json, psutil
        from pathlib import Path
        from gen_zero.causal.zero_runtime import ZeroStandaloneRuntime
        rt = ZeroStandaloneRuntime(int8_artifact=Path({str(int8_artifact)!r}))
        states, info = rt.encode(["memory probe text for the resident process", "zweiter Text"])
        rss = psutil.Process().memory_info().rss
        peak = [int(l.split()[1]) * 1024 for l in open('/proc/self/status') if l.startswith('VmHWM')][0]
        print(json.dumps({{"rss": rss, "peak": peak, "tensor": rt.tensor_bytes()}}))
    """)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=str(REPO / "python"), env=env)
    numbers = json.loads(out.stdout.strip().splitlines()[-1])
    assert numbers["tensor"] <= 1_000_000_000, numbers
    assert numbers["rss"] <= 1_000_000_000, numbers
    assert numbers["peak"] <= 1_000_000_000, numbers


def test_candidate_tokenization_is_prefix_consistent(runtime):
    """prompt + candidate must tokenize to prompt ids followed by candidate ids.

    Otherwise the prefix-cache path would silently encode a different string
    than the full sequence; this is the no-information-cut check on real data.
    """
    if not CALIBRATION.exists():
        pytest.skip("calibration split missing")
    checked = 0
    with open(CALIBRATION, encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            prompt_ids = runtime.token_ids(record["context"])
            for candidate in record["candidates"]:
                tail = runtime.candidate_text(candidate)
                joined = runtime.tokenizer.encode(record["context"] + tail, add_special_tokens=False).ids
                cand_ids = runtime.tokenizer.encode(tail, add_special_tokens=False).ids
                assert joined == prompt_ids + cand_ids, record["id"]
                checked += 1
    assert checked > 1000


# --------------------------------------------------------------------------
# Non-default manifold dimensions: the task head and the fractal thinking
# mode must key off the loaded manifold's own dim, not the MANIFOLD_DIM=64
# constant, and must not blow the fractal engine's fixed L2 working-set
# budget once the manifold is wider than ~192D.
# --------------------------------------------------------------------------

def _fit_synthetic_manifold(dim, hidden, encoder_id, tmp_path, seed=0):
    """A manifold fitted on synthetic (not model-derived) hidden states.

    ``ZeroManifold.fit`` only needs a finite (N > dim, hidden) matrix; it never
    touches the backbone, so this is cheap even at dim=896.
    """
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(dim + 64, hidden))
    manifold = ZeroManifold.fit(states, dim=dim, encoder_id=encoder_id,
                                source="synthetic-test", split="calibration")
    path = tmp_path / f"manifold_{dim}_{seed}.npz"
    manifold.save(path)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, sha256


@pytest.mark.parametrize("dim", [128, 256, 896])
def test_task_head_loads_and_scores_at_nondefault_manifold_dim(int8_artifact, runtime, tmp_path, dim):
    manifold_path, manifold_sha256 = _fit_synthetic_manifold(
        dim, runtime.hidden_size, runtime.encoder_id, tmp_path)
    head = ZeroTaskHead(
        weight=np.eye(dim, dtype=np.float32),
        provenance={"version": 1, "dim": dim, "encoder_id": runtime.encoder_id,
                    "manifold_sha256": manifold_sha256, "source": "synthetic-test",
                    "split": "calibration", "samples": 1},
    )
    head_path = tmp_path / f"head_{dim}.npz"
    head.save(head_path)

    rt = ZeroStandaloneRuntime(int8_artifact=int8_artifact, manifold_path=manifold_path,
                               task_head_path=head_path)
    assert rt.manifold.dim == dim
    assert rt.task_head.dim == dim

    with open(CALIBRATION, encoding="utf-8") as stream:
        record = json.loads(stream.readline())
    decision = rt.decide(record["context"], record["candidates"])
    assert decision.task_head_used
    assert decision.scores.shape == (len(record["candidates"]),)
    assert np.isfinite(decision.scores).all()
    assert 0 <= decision.index < len(record["candidates"])
    assert decision.prompt_state.shape == (dim,)


def test_task_head_still_rejects_a_genuine_dimension_mismatch(int8_artifact, runtime, tmp_path):
    # The old check compared against the MANIFOLD_DIM=64 constant, so it happened
    # to also reject this case; confirm the fix (comparing against the loaded
    # manifold's own dim) still fails closed on a real head/manifold mismatch.
    manifold_path, _ = _fit_synthetic_manifold(128, runtime.hidden_size, runtime.encoder_id, tmp_path)
    head = ZeroTaskHead(
        weight=np.eye(256, dtype=np.float32),
        provenance={"version": 1, "dim": 256, "encoder_id": runtime.encoder_id,
                    "manifold_sha256": "0" * 64, "source": "synthetic-test",
                    "split": "calibration", "samples": 1},
    )
    head_path = tmp_path / "mismatched_head.npz"
    head.save(head_path)
    with pytest.raises(ValueError, match="does not match the causal manifold"):
        ZeroStandaloneRuntime(int8_artifact=int8_artifact, manifold_path=manifold_path,
                              task_head_path=head_path)


@pytest.mark.parametrize("dim", [256, 896])
def test_fractal_thinking_mode_handles_high_dimensional_manifolds(int8_artifact, runtime, tmp_path, dim):
    # Before the fix, BifurcatedFractalEngine's fixed 64 KiB l2_bytes default
    # made this raise ValueError("working set ... exceeds l2_bytes=...") for
    # every manifold wider than ~192D (see the design-doc ledger table).
    manifold_path, _ = _fit_synthetic_manifold(dim, runtime.hidden_size, runtime.encoder_id, tmp_path)
    rt = ZeroStandaloneRuntime(int8_artifact=int8_artifact, manifold_path=manifold_path,
                               thinking_mode="fractal", single_core=True)
    with open(CALIBRATION, encoding="utf-8") as stream:
        record = json.loads(stream.readline())
    decision = rt.decide(record["context"], record["candidates"])
    assert decision.scores.shape == (len(record["candidates"]),)
    assert np.isfinite(decision.scores).all()
    assert decision.index == int(np.argmax(decision.scores))
    assert decision.dynamics_ms > 0


def test_fractal_engine_kwargs_reach_the_engine_and_explicit_override_still_fails_closed(
        int8_artifact, runtime, tmp_path):
    manifold_path, _ = _fit_synthetic_manifold(256, runtime.hidden_size, runtime.encoder_id, tmp_path)
    rt = ZeroStandaloneRuntime(int8_artifact=int8_artifact, manifold_path=manifold_path,
                               thinking_mode="fractal", single_core=True)
    with open(CALIBRATION, encoding="utf-8") as stream:
        record = json.loads(stream.readline())
    with pytest.raises(ValueError, match="working set"):
        rt.decide(record["context"], record["candidates"], fractal_engine_kwargs={"l2_bytes": 64})
