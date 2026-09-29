import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

SPEC = importlib.util.spec_from_file_location(
    "benchmark_multidim_cpu", REPO / "benchmarks" / "suites" / "benchmark_multidim_cpu.py")
multidim = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = multidim
SPEC.loader.exec_module(multidim)

from gen_zero.causal.zero_runtime import ZeroStandaloneRuntime, build_int8_artifact, find_local_snapshot  # noqa: E402

ARTIFACT = REPO / "benchmarks" / "artifacts" / "zero" / "zero_int8_v2.safetensors"


def _reference_project_to_sphere(x: np.ndarray, mu: np.ndarray, zca_mat: np.ndarray) -> np.ndarray:
    """Reimplements gpu_extract_natural_a100.py's project_to_sphere verbatim.

    Not imported from that module: as saved on disk it ends with a stray
    literal ``EOF`` top-level statement (a leftover heredoc terminator,
    unrelated to and outside the guarded ``if __name__ == "__main__":``
    block) that raises ``NameError`` on any import, independent of the
    documented CUDA-availability guard.
    """
    centered = x - mu
    proj = centered @ zca_mat
    norm = np.linalg.norm(proj, axis=-1, keepdims=True)
    return proj / np.maximum(norm, 1e-12)


def test_project_to_sphere_matches_gpu_extract_reference():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 896)).astype(np.float32)
    mu = rng.normal(size=(896,)).astype(np.float32)
    zca_mat = rng.normal(size=(896, 64)).astype(np.float32)
    got = multidim.project_to_sphere(x, mu, zca_mat)
    want = _reference_project_to_sphere(x, mu, zca_mat)
    assert np.allclose(got, want)
    assert np.allclose(np.linalg.norm(got, axis=-1), 1.0, atol=1e-5)


def test_score_candidates_matches_manual_argmax():
    rng = np.random.default_rng(1)
    dim, k = 8, 5
    q0_sphere = rng.normal(size=(dim,)).astype(np.float32)
    q0_sphere /= np.linalg.norm(q0_sphere)
    c_states_sphere = rng.normal(size=(k, dim)).astype(np.float32)
    c_states_sphere /= np.linalg.norm(c_states_sphere, axis=1, keepdims=True)
    W = rng.normal(size=(dim, dim)).astype(np.float32)

    scores = multidim.score_candidates(q0_sphere, c_states_sphere, W)
    assert scores.shape == (k,)

    manual = np.zeros(k)
    q_trans = q0_sphere @ W
    for j in range(k):
        manual[j] = float(q_trans @ c_states_sphere[j])
    assert np.allclose(scores, manual)
    assert int(np.argmax(scores)) == int(np.argmax(manual))


def test_residual_energy_fraction_uses_centered_orthogonal_subspace():
    mean = np.array([2.0, -1.0, 3.0])
    basis = np.array([[1.0], [0.0], [0.0]])
    states = np.array([[5.0, -1.0, 3.0], [2.0, 3.0, 3.0], [5.0, 3.0, 3.0]])
    got = multidim.residual_energy_fraction(states, mean, basis)
    assert np.allclose(got, [0.0, 1.0, 16.0 / 25.0])


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
    return ZeroStandaloneRuntime(int8_artifact=int8_artifact, manifold_path=None,
                                 task_head_path=None, max_length=1024)


@pytest.fixture(scope="module")
def artifacts():
    for dim in multidim.DIMS:
        if not multidim.manifold_path(dim).exists() or not multidim.task_head_path(dim).exists():
            pytest.skip(f"multi-dim artifacts for {dim}-D are not available locally")
    return multidim.load_dim_artifacts(multidim.DIMS)


def test_predict_record_never_reads_ground_truth(runtime, artifacts):
    record = {
        "id": "leakage-probe-1", "task": "probe",
        "context": "Route the following natural language request to its target intent domain: "
                   "what's the weather like on friday",
        "candidates": ["alarm", "weather", "datetime", "iot_lights"],
        "metadata": {},
    }
    assert "ground_truth" not in record
    entry = multidim.predict_record(runtime, artifacts, record, multidim.DIMS)
    assert entry["id"] == "leakage-probe-1"
    for dim in multidim.DIMS:
        assert entry["per_dim"][dim]["prediction"] in record["candidates"]
        assert len(entry["per_dim"][dim]["scores"]) == len(record["candidates"])


# --------------------------------------------------------------------------
# --adapter CLI flag
# --------------------------------------------------------------------------

def test_adapter_flag_defaults_to_none_and_parses_a_path(tmp_path):
    parser = multidim.build_arg_parser()

    args = parser.parse_args(["run"])
    assert args.adapter is None

    fake = tmp_path / "deep_projection_adapter.pt"
    args = parser.parse_args(["run", "--adapter", str(fake)])
    assert args.adapter == fake


DEEP_ADAPTER_MANIFOLD = REPO / "benchmarks" / "artifacts" / "zero" / "zero_manifold_open_v1.npz"


@pytest.fixture(scope="module")
def deep_adapter(runtime):
    if not DEEP_ADAPTER_MANIFOLD.exists():
        pytest.skip("deep projection adapter manifold artifact not available locally")
    from gen_zero.causal.deep_projection_adapter import DeepProjectionAdapter

    return DeepProjectionAdapter(DEEP_ADAPTER_MANIFOLD, encoder_id=runtime.encoder_id)


def test_predict_record_adapter_column_is_additional_and_separate_from_dims(runtime, artifacts, deep_adapter):
    record = {
        "id": "leakage-probe-2", "task": "probe",
        "context": "Route the following natural language request to its target intent domain: "
                   "what's the weather like on friday",
        "candidates": ["alarm", "weather", "datetime", "iot_lights"],
        "metadata": {},
    }
    without_adapter = multidim.predict_record(runtime, artifacts, record, multidim.DIMS)
    with_adapter = multidim.predict_record(runtime, artifacts, record, multidim.DIMS, adapter=deep_adapter)

    assert "adapter" not in without_adapter["per_dim"]
    assert "adapter" in with_adapter["per_dim"]
    assert with_adapter["per_dim"]["adapter"]["prediction"] in record["candidates"]
    assert len(with_adapter["per_dim"]["adapter"]["scores"]) == len(record["candidates"])

    # Mounting the adapter must not perturb the existing integer-keyed dim
    # columns -- in particular dim=64's GPU-fitted-manifold scores are untouched.
    for dim in multidim.DIMS:
        assert with_adapter["per_dim"][dim]["scores"] == without_adapter["per_dim"][dim]["scores"]
        assert with_adapter["per_dim"][dim]["prediction"] == without_adapter["per_dim"][dim]["prediction"]
