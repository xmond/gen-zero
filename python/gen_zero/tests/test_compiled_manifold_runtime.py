"""Tests for the offline knowledge compiler and the compiled CPU runtime.

Three kinds of check, matching the task's own honesty framing:
  * Format/size/corruption tests operate on the SYNTHETIC-SHAPE-GOLDEN artifact
    (`compile_from_task_schemas`, real per-task class counts, no real-world
    signal) -- these validate the binary format and the runtime, not accuracy.
  * `test_runtime_matches_reference_on_real_9b_features` is the one place real
    Qwen3.5-9B hidden states are used (`benchmarks/results/v5_hidden_features.npz`,
    the only real encoder features on this host). The fit happens in `tmp_path`
    and is never persisted to the repo, exactly like
    `benchmarks/suites/cpu_dynamics_clean_eval.py`'s own methodology. It checks
    the compiled runtime AGREES with the reference Python dynamics -- it does
    NOT compute or claim any task accuracy number.
  * Latency tests measure this host, this run; they are not portable claims.
"""
from __future__ import annotations

import inspect
import json
import struct
import time
from pathlib import Path

import numpy as np
import pytest

from gen_zero.causal import compiled_manifold_runtime as runtime_module
from gen_zero.causal.compiled_manifold_runtime import (
    CodebookCorruptionError,
    CompiledManifoldRuntime,
)
from gen_zero.causal.counterfactual_drift_dynamics import (
    CounterfactualDriftDynamics,
    fit_counterfactual_drift_dynamics,
)
from gen_zero.causal.knowledge_compiler import (
    HEADER,
    MAGIC,
    MAX_FILE_BYTES,
    WORKING_SET_BUDGET_BYTES,
    choose_compact_dims,
    compile_from_task_schemas,
    compile_manifold,
    synthetic_shape_golden_dynamics,
)

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "benchmarks" / "data"
V5_FEATURES = ROOT / "benchmarks" / "results" / "v5_hidden_features.npz"
EXPECTED_TASK_IDS = {
    "aegis_safety", "arc_challenge", "boolq", "civil_comments", "gsm8k",
    "massive_de", "massive_en", "multinli", "paws", "pubmedqa", "squad2",
    "summeval", "vitaminc",
}
PAIRED_REAL_TASKS = ("paws", "multinli")


@pytest.fixture(scope="module")
def compiled(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("codebook") / "causal_codebook.bin"
    report = compile_from_task_schemas(DATA_DIR, out)
    return report


@pytest.fixture(scope="module")
def rt(compiled) -> CompiledManifoldRuntime:
    return CompiledManifoldRuntime.load(compiled["path"])


# ---------------------------------------------------------------------------
# Size and manifest
# ---------------------------------------------------------------------------

def test_file_size_under_2mib(compiled):
    size = Path(compiled["path"]).stat().st_size
    assert size == compiled["file_size_bytes"]
    assert size < 2 * 1024 * 1024
    assert size < MAX_FILE_BYTES


def test_all_13_tasks_present(compiled):
    assert set(compiled["task_ids"]) == EXPECTED_TASK_IDS
    assert len(compiled["manifest"]["tasks"]) == 13


def test_notes_disclose_synthetic_provenance(compiled):
    notes = compiled["manifest"]["notes"]
    assert "SYNTHETIC" in notes
    for section in compiled["manifest"]["tasks"]:
        assert section["provenance"]["source"] == "synthetic-shape-golden-v1"


def test_oversize_manifold_is_refused(tmp_path):
    """The compiler must assert-and-refuse, not silently truncate or clip."""
    dyn, _ = synthetic_shape_golden_dynamics("oversize", 4, paired=False, n_components=16,
                                              cf_components=8, seed=1)
    huge = {}
    for i in range(400):  # 400 tasks x ~70KB/task blows the 2MiB cap
        huge[f"t{i}"] = (dyn, False)
    with pytest.raises(ValueError, match="2 MiB|MAX_FILE_BYTES|oversized"):
        compile_manifold(huge, tmp_path / "huge.bin", notes="oversize test")


# ---------------------------------------------------------------------------
# Byte-level serialization / deserialization
# ---------------------------------------------------------------------------

def test_byte_level_roundtrip_exact_f32_arrays(tmp_path):
    dyn, _ = synthetic_shape_golden_dynamics("t_roundtrip", 3, paired=True, seed=7)
    path = tmp_path / "one_task.bin"
    compile_manifold({"t_roundtrip": (dyn, True)}, path, notes="roundtrip test")
    loaded = CompiledManifoldRuntime.load(path)
    sec = loaded.section("t_roundtrip")

    assert np.array_equal(sec.A, dyn.A.astype(np.float32))
    assert np.array_equal(sec.B, dyn.B.astype(np.float32))
    assert np.array_equal(sec.W_c, dyn.W_c.astype(np.float32))
    assert np.array_equal(sec.codebook, dyn.codebook.astype(np.float32))
    assert np.array_equal(sec.x_mean, dyn.x_mean.astype(np.float32))
    assert np.array_equal(sec.c_mean, dyn.c_mean.astype(np.float32))
    # byte-exact, not just value-equal: compare raw buffers
    assert sec.A.tobytes() == dyn.A.astype(np.float32).tobytes()
    assert sec.codebook.tobytes() == dyn.codebook.astype(np.float32).tobytes()

    expected_inv = np.linalg.inv(np.eye(dyn.dim) - dyn.A).astype(np.float32)
    assert np.array_equal(sec.inv_i_minus_a, expected_inv)


def test_byte_level_roundtrip_quantized_basis_within_tolerance(tmp_path):
    dyn, _ = synthetic_shape_golden_dynamics("t_basis", 3, paired=True, seed=8)
    path = tmp_path / "basis.bin"
    compile_manifold({"t_basis": (dyn, True)}, path, notes="basis test")
    loaded = CompiledManifoldRuntime.load(path)
    sec = loaded.section("t_basis")

    x_basis = sec.x_basis()
    c_basis = sec.c_basis()
    assert x_basis.shape == dyn.x_basis.shape
    assert c_basis.shape == dyn.c_basis.shape
    # int8 per-column quantization: each column's max error <= scale/2
    col_scale = sec.x_basis_scale
    err = np.abs(x_basis - dyn.x_basis.astype(np.float32))
    assert np.all(err <= col_scale[None, :] * 0.5 + 1e-6)


def test_corrupted_payload_hash_is_rejected(compiled, tmp_path):
    raw = bytearray(Path(compiled["path"]).read_bytes())
    flip_at = HEADER.size + 5000  # inside the payload, past the header
    raw[flip_at] ^= 0xFF
    bad = tmp_path / "corrupt_hash.bin"
    bad.write_bytes(bytes(raw))
    with pytest.raises(CodebookCorruptionError, match="sha256"):
        CompiledManifoldRuntime.load(bad)


def test_bad_magic_is_rejected(compiled, tmp_path):
    raw = bytearray(Path(compiled["path"]).read_bytes())
    raw[0:8] = b"NOTAGZ00"
    bad = tmp_path / "bad_magic.bin"
    bad.write_bytes(bytes(raw))
    with pytest.raises(CodebookCorruptionError, match="magic"):
        CompiledManifoldRuntime.load(bad)


def test_truncated_file_is_rejected(compiled, tmp_path):
    raw = Path(compiled["path"]).read_bytes()
    bad = tmp_path / "truncated.bin"
    bad.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(CodebookCorruptionError, match="truncated"):
        CompiledManifoldRuntime.load(bad)


def test_non_contractive_A_is_rejected(tmp_path):
    """A corrupted (or maliciously edited) A that breaks the Lyapunov
    contraction certificate must be refused at load, not silently used."""
    dyn, _ = synthetic_shape_golden_dynamics("t_bad_a", 3, paired=False, seed=9)
    path = tmp_path / "bad_a.bin"
    compile_manifold({"t_bad_a": (dyn, False)}, path, notes="bad A test")
    raw = bytearray(path.read_bytes())

    manifest_offset = HEADER.size
    (_, _, _, _, manifest_len, arrays_base_offset, _, _) = HEADER.unpack_from(bytes(raw), 0)
    manifest = json.loads(bytes(raw[manifest_offset:manifest_offset + manifest_len]).decode("utf-8"))
    a_meta = manifest["tasks"][0]["arrays"]["A"]
    a_offset = arrays_base_offset + a_meta["offset_in_arrays"]
    dim = a_meta["shape"][0]
    non_contractive = (np.eye(dim, dtype=np.float32) * 1.5).tobytes()
    raw[a_offset:a_offset + len(non_contractive)] = non_contractive
    # payload changed, so the stored sha256 no longer matches: recompute it so
    # this test isolates the contractivity check, not the hash check.
    header_fields = list(HEADER.unpack_from(bytes(raw), 0))
    payload_len = header_fields[6]
    payload = bytes(raw[HEADER.size:HEADER.size + payload_len])
    import hashlib
    new_sha = hashlib.sha256(payload).digest()
    header_fields[7] = new_sha
    raw[0:HEADER.size] = HEADER.pack(*header_fields)

    bad_path = tmp_path / "bad_a_rehashed.bin"
    bad_path.write_bytes(bytes(raw))
    with pytest.raises(CodebookCorruptionError, match="contractive"):
        CompiledManifoldRuntime.load(bad_path)


# ---------------------------------------------------------------------------
# Working set / latency budgets
# ---------------------------------------------------------------------------

def test_working_set_bytes_within_budget(rt):
    for task_id in rt.task_ids:
        sec = rt.section(task_id)
        ws = sec.working_set_bytes()
        assert ws <= WORKING_SET_BUDGET_BYTES, f"{task_id}: working set {ws} > {WORKING_SET_BUDGET_BYTES}"
        assert ws <= 4096


def _bench(fn, n=4000):
    for _ in range(200):  # warmup
        fn()
    samples = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter_ns()
        fn()
        samples[i] = time.perf_counter_ns() - t0
    return samples / 1e3  # microseconds


def test_single_step_latency_le_15us(rt):
    sec = rt.section("boolq")
    rng = np.random.default_rng(0)
    h = np.zeros(sec.dim, dtype=np.float32)
    x = rng.standard_normal(sec.n_components).astype(np.float32)
    samples = _bench(lambda: rt.step("boolq", h, x, None))
    p50, p99 = np.percentile(samples, [50, 99])
    assert p50 <= 15.0, f"single-step p50={p50:.3f}us exceeds 15us budget (p99={p99:.3f}us)"


def test_end_to_end_latency_le_500us(rt):
    sec = rt.section("multinli")
    rng = np.random.default_rng(1)
    x = rng.standard_normal(sec.n_components).astype(np.float32)
    c = rng.standard_normal(sec.cf_components).astype(np.float32)
    samples = _bench(lambda: rt.infer("multinli", x, c))
    p50, p99 = np.percentile(samples, [50, 99])
    assert p50 <= 500.0, f"end-to-end p50={p50:.3f}us exceeds 500us (0.5ms) budget (p99={p99:.3f}us)"


# ---------------------------------------------------------------------------
# Lyapunov convergence certificate
# ---------------------------------------------------------------------------

def test_lyapunov_certificate_rho_below_one(rt):
    for task_id in rt.task_ids:
        sec = rt.section(task_id)
        assert sec.rho_a < 1.0, f"{task_id}: rho(A)={sec.rho_a} is not contractive"


def test_lyapunov_convergence_certificate_iteration_matches_closed_form(rt):
    """The whole point of `A` being certified contractive is that the recurrence
    `h_{t+1} = A h_t + B x + W_c c` converges to a UNIQUE fixed point regardless
    of h_0. This is the actual Lyapunov content: iterate from an arbitrary
    (nonzero, off-manifold) start and confirm convergence to the runtime's
    closed-form `inv_i_minus_a @ drive`, at a rate consistent with rho(A)."""
    sec = rt.section("summeval")
    rng = np.random.default_rng(2)
    x = rng.standard_normal(sec.n_components).astype(np.float32)
    h = rng.standard_normal(sec.dim).astype(np.float32) * 10.0  # arbitrary start
    result = rt.infer("summeval", x, None)
    residuals = []
    for _ in range(200):
        h = rt.step("summeval", h, x, None)
        residuals.append(float(np.linalg.norm(h - result.fixed_point)))
    assert residuals[-1] < 1e-4, f"iteration did not converge to closed-form fixed point: {residuals[-1]}"
    # monotone geometric decay is the Lyapunov certificate itself: each step's
    # residual should shrink by roughly rho(A), not merely "eventually small"
    ratios = [residuals[i + 1] / residuals[i] for i in range(20) if residuals[i] > 1e-9]
    assert all(r <= sec.rho_a + 0.05 for r in ratios), ratios


# ---------------------------------------------------------------------------
# Real 9B features: runtime agrees with the reference Python dynamics
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not V5_FEATURES.exists(), reason="no real 9B feature artifact on this host")
@pytest.mark.parametrize("task", PAIRED_REAL_TASKS)
def test_runtime_matches_reference_on_real_9b_features(task, tmp_path):
    npz = np.load(V5_FEATURES, allow_pickle=False)
    mask = npz["tasks"] == task
    ids = npz["ids"][mask]
    x = npz["h"][mask].astype(np.float64)
    if task == "paws":
        c = npz["h_difference"][mask].astype(np.float64)
    else:
        c = (npz["h_hypothesis"][mask] - npz["h_premise"][mask]).astype(np.float64)
    n = x.shape[0]
    assert n == 30
    # Deterministic pseudo-labels from row index only (no ground truth file is
    # read): this test checks runtime/reference AGREEMENT, never accuracy.
    k = 3
    y = np.arange(n) % k

    dyn, _info = fit_counterfactual_drift_dynamics(
        x, c, y, sample_ids=[str(i) for i in ids], source=f"real-9b-features-{task}",
        split="calibration", encoder_id="v5_hidden_features.npz", n_classes=k,
        dim=16, n_components=16, cf_components=8, use_counterfactual=True)

    path = tmp_path / f"{task}_real.bin"
    compile_manifold({task: (dyn, True)}, path, notes=f"real-feature equivalence check for {task}")
    loaded = CompiledManifoldRuntime.load(path)

    for i in range(n):
        zx = dyn.project_x(x[i])
        zc = dyn.project_c(c[i])
        ref_h_star = dyn.fixed_point(zx, zc)
        ref_pred = int(np.argmin(np.linalg.norm(dyn.codebook - ref_h_star[None, :], axis=1)))

        res = loaded.infer(task, zx.astype(np.float32), zc.astype(np.float32))
        assert res.prediction == ref_pred, f"{task} row {i}: runtime {res.prediction} != reference {ref_pred}"
        assert np.allclose(res.fixed_point, ref_h_star, atol=1e-4), \
            f"{task} row {i}: fixed point mismatch {res.fixed_point} vs {ref_h_star}"


# ---------------------------------------------------------------------------
# Rule 3 / doc 15 4.4: no task names, no regex, no language-specific parsing
# ---------------------------------------------------------------------------

def test_runtime_source_has_no_hardcoded_task_names_or_regex():
    source = inspect.getsource(runtime_module)
    assert "import re" not in source
    for task_id in EXPECTED_TASK_IDS:
        assert task_id not in source, f"runtime source hardcodes task name {task_id!r}"


def test_routed_runtime_uses_four_orthogonal_contracting_charts(rt):
    from gen_zero.causal.compiled_manifold_runtime import CalibratedMoEGateway
    from gen_zero.causal.wasserstein_moe_router import (
        ExpertManifold, MultiTangentManifoldPool, WassersteinOptimalTransportRouter)

    sec = rt.section("boolq")
    shared_dim = 4
    pool = MultiTangentManifoldPool(tangent_dim=2, seed=71)
    experts = [ExpertManifold(name, mean, np.eye(shared_dim) * 0.1)
               for name, mean in (("codebook", np.zeros(shared_dim)),
                                  ("zero", np.ones(shared_dim)))]
    router = WassersteinOptimalTransportRouter(experts, temperature=2.0)
    rng = np.random.default_rng(72)
    gateway = CalibratedMoEGateway(
        router, pool, rng.normal(size=(shared_dim, sec.dim)),
        rng.normal(size=(shared_dim, shared_dim)),
        rng.normal(size=(pool.ambient_dim, shared_dim)),
        rng.normal(size=(shared_dim, pool.ambient_dim)),
        chart_prototypes={name: rng.normal(size=pool.ambient_dim)
                          for name in pool.CHART_NAMES})
    x = rng.normal(size=sec.n_components).astype(np.float32)
    zero = rng.normal(size=shared_dim)
    a = rt.infer_routed("boolq", x, zero, gateway=gateway)
    b = rt.infer_routed("boolq", x, zero + 1e-6, gateway=gateway)
    assert np.isfinite(a.decision_state).all()
    assert a.weights["codebook"] + a.weights["zero"] == pytest.approx(1.0)
    assert sum(a.tangent_weights.values()) == pytest.approx(1.0)
    assert np.linalg.norm(a.decision_state - b.decision_state) < 1e-3
    assert np.allclose(a.decision_state,
                       a.fused_state + gateway.ambient_to_shared @ a.tangent_state)
    for i, name in enumerate(pool.CHART_NAMES):
        space = pool.chart(name)
        assert max(abs(np.linalg.eigvals(space.a))) < 1.0
        for other in pool.CHART_NAMES[i + 1:]:
            assert np.max(abs(space.basis.T @ pool.chart(other).basis)) < 1e-10
    # Measure the complete routed path on one BLAS thread, including all
    # Bures distances and four fixed-point solves.
    elapsed = _bench(lambda: rt.infer_routed("boolq", x, zero, gateway=gateway), n=100)
    assert np.percentile(elapsed, 50) < 20_000  # microseconds, host-dependent guardrail
