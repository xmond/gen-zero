import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "suites"))

import cross_model_manifold_alignment as cma  # noqa: E402


def _rng():
    return np.random.default_rng(0)


def _feature_dict(n_train=12, n_test=6, dim=8, n_cands=4, rng=None, id_offset=0):
    rng = rng or _rng()
    return {
        "train_full": rng.standard_normal((n_train, dim)),
        "test_full": rng.standard_normal((n_test, dim)),
        "cands": rng.standard_normal((n_cands, dim)),
        "train_label": rng.integers(0, n_cands, size=n_train),
        "train_ids": np.array([f"tr-{i + id_offset}" for i in range(n_train)]),
        "test_ids": np.array([f"te-{i + id_offset}" for i in range(n_test)]),
    }


def _write_npz(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    np.savez(path, **data, info_json=np.array("{}"))
    return path


# --------------------------------------------------------------------------- load_features

def test_load_features_round_trips(tmp_path):
    data = _feature_dict()
    path = _write_npz(tmp_path, "a.npz", data)
    loaded = cma.load_features(path)
    for key in cma.REQUIRED_KEYS:
        assert np.array_equal(loaded[key], data[key])


def test_load_features_missing_key_raises(tmp_path):
    data = _feature_dict()
    del data["cands"]
    path = tmp_path / "bad.npz"
    np.savez(path, **data, info_json=np.array("{}"))
    with pytest.raises(ValueError, match="missing keys"):
        cma.load_features(path)


def test_load_features_non_finite_raises(tmp_path):
    data = _feature_dict()
    data["train_full"][0, 0] = np.nan
    path = _write_npz(tmp_path, "nan.npz", data)
    with pytest.raises(ValueError, match="non-finite"):
        cma.load_features(path)

    data2 = _feature_dict()
    data2["test_full"][1, 2] = np.inf
    path2 = _write_npz(tmp_path, "inf.npz", data2)
    with pytest.raises(ValueError, match="non-finite"):
        cma.load_features(path2)


def test_load_features_raises_when_file_is_internally_inconsistent(tmp_path):
    # A file can be broken in isolation, before it is ever compared against a second file: its own
    # feature block row count disagreeing with its own id array.
    data = _feature_dict(n_train=12)
    data["train_ids"] = data["train_ids"][:-1]      # now 11 ids
    data["train_label"] = data["train_label"][:-1]  # keep labels/ids in step so only train_full disagrees
    path = _write_npz(tmp_path, "self_inconsistent.npz", data)
    with pytest.raises(ValueError, match="train_full"):
        cma.load_features(path)

    data2 = _feature_dict(n_train=12)
    data2["train_label"] = data2["train_label"][:-1]
    path2 = _write_npz(tmp_path, "label_mismatch.npz", data2)
    with pytest.raises(ValueError, match="train_label"):
        cma.load_features(path2)


# --------------------------------------------------------------------------- verify_id_alignment

def test_verify_id_alignment_passes_for_identical_ids():
    a = _feature_dict()
    b = _feature_dict()
    b["train_ids"] = a["train_ids"].copy()
    b["test_ids"] = a["test_ids"].copy()
    b["train_label"] = a["train_label"].copy()
    cma.verify_id_alignment(a, "a", b, "b")  # must not raise


def test_verify_id_alignment_raises_on_permuted_ids():
    a = _feature_dict()
    b = _feature_dict()
    b["train_ids"] = a["train_ids"][::-1].copy()
    b["test_ids"] = a["test_ids"].copy()
    b["train_label"] = a["train_label"].copy()
    with pytest.raises(ValueError, match="train_ids"):
        cma.verify_id_alignment(a, "a", b, "b")


def test_verify_id_alignment_raises_on_dropped_row():
    a = _feature_dict(n_train=12)
    b = _feature_dict(n_train=11)
    b["test_ids"] = a["test_ids"].copy()
    with pytest.raises(ValueError, match="train_ids"):
        cma.verify_id_alignment(a, "a", b, "b")


def test_verify_id_alignment_raises_on_different_count():
    a = _feature_dict(n_test=6)
    b = _feature_dict(n_test=9)
    b["train_ids"] = a["train_ids"].copy()
    with pytest.raises(ValueError, match="test_ids"):
        cma.verify_id_alignment(a, "a", b, "b")


def test_verify_id_alignment_raises_on_label_mismatch():
    a = _feature_dict()
    b = _feature_dict()
    b["train_ids"] = a["train_ids"].copy()
    b["test_ids"] = a["test_ids"].copy()
    b["train_label"] = (a["train_label"] + 1) % 4
    with pytest.raises(ValueError, match="train_label"):
        cma.verify_id_alignment(a, "a", b, "b")


# --------------------------------------------------------------------------- procrustes_residual

def test_procrustes_residual_zero_for_identical_matrices():
    rng = _rng()
    X = rng.standard_normal((20, 6))
    # closed form is sqrt(2 - 2*nuclear): float64 round-off of ~1e-16 surfaces as ~1e-8
    assert cma.procrustes_residual(X, X.copy()) == pytest.approx(0.0, abs=1e-6)


def test_procrustes_residual_zero_for_random_orthogonal_rotation():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 5))
    Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    Y = X @ Q
    assert cma.procrustes_residual(X, Y) == pytest.approx(0.0, abs=1e-6)


def test_procrustes_residual_handles_unequal_dims():
    rng = np.random.default_rng(2)
    n = 25
    latent = rng.standard_normal((n, 4))
    A = rng.standard_normal((4, 7))
    B = rng.standard_normal((4, 11))
    X = latent @ A
    Y = latent @ B
    residual = cma.procrustes_residual(X, Y)
    assert np.isfinite(residual)
    assert 0.0 <= residual <= 2.0 + 1e-8
    # Shared latent structure should fit much better than unrelated noise of the same shapes.
    noise_residual = cma.procrustes_residual(rng.standard_normal((n, 7)), rng.standard_normal((n, 11)))
    assert residual < noise_residual


def test_procrustes_residual_symmetric_in_argument_order_when_dims_differ():
    # The SVD solution R = U V^T is only the exact minimizer when the SOURCE (first factor of
    # X^T Y) has dimension <= the target's. procrustes_residual must always pick the
    # lower-dimensional block as the source internally, regardless of which argument position it
    # was passed in -- otherwise swapping (X, Y) -> (Y, X) would silently change the number.
    rng = np.random.default_rng(5)
    n = 25
    latent = rng.standard_normal((n, 3))
    small = latent @ rng.standard_normal((3, 4))    # smaller-dim block (d=4)
    big = latent @ rng.standard_normal((3, 9))       # larger-dim block (d=9)
    forward = cma.procrustes_residual(big, small)    # d1=9 > d2=4: the direction that was buggy
    backward = cma.procrustes_residual(small, big)   # d1=4 < d2=9
    assert forward == pytest.approx(backward, abs=1e-9)


def test_procrustes_residual_exact_isometric_embedding_when_source_dim_larger():
    # A genuine isometric embedding of a higher-dim block into a lower-dim block's exact subspace
    # must still hit ~0 residual even when the naive (unswapped) SVD direction would be suboptimal.
    rng = np.random.default_rng(6)
    n = 30
    Y = rng.standard_normal((n, 5))         # the smaller, "true" space
    # A genuine isometry R0 (9x5, R0^T R0 = I_5); X = Y @ R0^T lives in a 9-dim space whose rows
    # are an exact rotated copy of Y's row space, so a size-9 -> size-5 embedding of X back onto Y
    # exists with zero residual.
    R0, _ = np.linalg.qr(rng.standard_normal((9, 5)))   # 9x5, R0^T R0 = I_5
    X = Y @ R0.T                              # (n,9) = (n,5) @ (5,9)
    residual = cma.procrustes_residual(X, Y)  # d1=9 > d2=5
    assert residual == pytest.approx(0.0, abs=1e-6)


def test_procrustes_residual_raises_on_sample_count_mismatch():
    rng = _rng()
    with pytest.raises(ValueError, match="same number of samples"):
        cma.procrustes_residual(rng.standard_normal((10, 4)), rng.standard_normal((9, 4)))


def test_procrustes_residual_raises_on_non_finite():
    rng = _rng()
    X = rng.standard_normal((10, 4))
    Y = rng.standard_normal((10, 4))
    Y[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        cma.procrustes_residual(X, Y)


def _procrustes_full_svd_oracle(X, Y):
    """Direct definition: R = U V^T from the full SVD of source^T target, residual measured on the
    materialized R. Slow (O(p^3)); the oracle the closed-form implementation must reproduce."""
    Xc, Yc = X - X.mean(axis=0), Y - Y.mean(axis=0)
    Xn, Yn = Xc / np.linalg.norm(Xc), Yc / np.linalg.norm(Yc)
    source, target = (Xn, Yn) if Xn.shape[1] <= Yn.shape[1] else (Yn, Xn)
    u, _, vt = np.linalg.svd(source.T @ target, full_matrices=False)
    return float(np.linalg.norm(source @ (u @ vt) - target) / np.linalg.norm(target))


@pytest.mark.parametrize("n,p,q", [(40, 6, 9), (40, 9, 6), (12, 30, 50), (12, 50, 30), (12, 30, 30), (60, 7, 7)])
def test_procrustes_closed_form_matches_full_svd_oracle(n, p, q):
    # Covers n > p, n < p (the real regime: 250-1000 rows vs 8192 dims) and both orientations.
    rng = np.random.default_rng(n * 1000 + p * 10 + q)
    latent = rng.standard_normal((n, 3))
    X = latent @ rng.standard_normal((3, p)) + 0.5 * rng.standard_normal((n, p))
    Y = latent @ rng.standard_normal((3, q)) + 0.5 * rng.standard_normal((n, q))
    assert cma.procrustes_residual(X, Y) == pytest.approx(_procrustes_full_svd_oracle(X, Y), abs=1e-9)


# --------------------------------------------------------------------------- align_block / linear_cka

def test_align_block_identical_matrices_gives_cka_one_and_residual_zero():
    a = _feature_dict()
    b = {**a, "train_full": a["train_full"].copy(), "test_full": a["test_full"].copy()}
    result = cma.align_block(a, b, "train_full")
    assert result["linear_cka"] == pytest.approx(1.0, abs=1e-6)
    assert result["procrustes_residual"] == pytest.approx(0.0, abs=1e-6)
    assert result["dim_a"] == result["dim_b"]


def test_align_block_raises_on_sample_count_mismatch():
    a = _feature_dict(n_train=12)
    b = _feature_dict(n_train=9)
    with pytest.raises(ValueError, match="sample counts differ"):
        cma.align_block(a, b, "train_full")


# --------------------------------------------------------------------------- compare (end-to-end)

def test_compare_end_to_end_aligned_files(tmp_path):
    rng = np.random.default_rng(3)
    a = _feature_dict(dim=8, rng=rng)
    Q, _ = np.linalg.qr(rng.standard_normal((8, 8)))
    b = dict(a)
    b["train_full"] = a["train_full"] @ Q
    b["test_full"] = a["test_full"] @ Q
    b["cands"] = a["cands"] @ Q
    path_a = _write_npz(tmp_path, "a.npz", a)
    path_b = _write_npz(tmp_path, "b.npz", b)
    report = cma.compare(path_a, path_b)
    assert report["n_train"] == a["train_ids"].shape[0]
    blocks = {block["block"]: block for block in report["blocks"]}
    assert set(blocks) == {"train_full", "test_full"}
    for block in blocks.values():
        assert block["linear_cka"] == pytest.approx(1.0, abs=1e-6)
        assert block["procrustes_residual"] == pytest.approx(0.0, abs=1e-6)


def test_compare_end_to_end_misaligned_ids_raises(tmp_path):
    a = _feature_dict()
    b = _feature_dict(id_offset=100)  # disjoint ids
    path_a = _write_npz(tmp_path, "a.npz", a)
    path_b = _write_npz(tmp_path, "b.npz", b)
    with pytest.raises(ValueError):
        cma.compare(path_a, path_b)


def test_compare_end_to_end_unequal_dims_runs(tmp_path):
    rng = np.random.default_rng(4)
    a = _feature_dict(dim=8, rng=rng)
    b = dict(a)
    proj = rng.standard_normal((8, 13))
    b["train_full"] = a["train_full"] @ proj
    b["test_full"] = a["test_full"] @ proj
    b["cands"] = a["cands"] @ proj
    path_a = _write_npz(tmp_path, "a.npz", a)
    path_b = _write_npz(tmp_path, "b.npz", b)
    report = cma.compare(path_a, path_b)
    for block in report["blocks"]:
        assert block["dim_a"] == 8
        assert block["dim_b"] == 13
        assert np.isfinite(block["linear_cka"])
        assert np.isfinite(block["procrustes_residual"])


# --------------------------------------------------------------------------- controls / batch

def test_row_permutation_null_is_worse_than_paired_and_seeded():
    rng = np.random.default_rng(11)
    latent = rng.standard_normal((80, 4))
    X = latent @ rng.standard_normal((4, 20))
    Y = latent @ rng.standard_normal((4, 30))
    null_a = cma.row_permutation_null(X, Y, permutations=3, seed=7)
    null_b = cma.row_permutation_null(X, Y, permutations=3, seed=7)
    assert null_a == null_b
    assert null_a["linear_cka_mean"] < cma.PhaseTransitionLayerExtractor.linear_cka(X, Y)
    assert null_a["procrustes_residual_mean"] > cma.procrustes_residual(X, Y)


def test_standardized_cka_ignores_a_dominant_feature_scale():
    rng = np.random.default_rng(12)
    X = rng.standard_normal((60, 10))
    Y = rng.standard_normal((60, 10))
    X_big, Y_big = X.copy(), Y.copy()
    X_big[:, 0] *= 1e4
    Y_big[:, 0] = X_big[:, 0]                 # one shared giant column fakes similarity
    raw = cma.PhaseTransitionLayerExtractor.linear_cka(X_big, Y_big)
    z = cma.standardized_linear_cka(X_big, Y_big)
    assert raw > 0.99 and z < 0.5


def test_compare_directories_all_tasks_and_missing_task_raises(tmp_path):
    rng = _rng()
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
    for task in ("t1", "t2"):
        data_a = _feature_dict(rng=rng)
        data_b = dict(data_a, train_full=data_a["train_full"] @ np.linalg.qr(rng.standard_normal((8, 8)))[0],
                      test_full=data_a["test_full"] @ np.linalg.qr(rng.standard_normal((8, 8)))[0])
        _write_npz(tmp_path / "a", f"{task}.npz", data_a)
        _write_npz(tmp_path / "b", f"{task}.npz", data_b)
    report = cma.compare_directories(tmp_path / "a", tmp_path / "b", ["t1", "t2"], controls=True)
    assert report["n_tasks"] == 2
    for task in ("t1", "t2"):
        for blk in report["per_task"][task]["blocks"]:
            assert blk["linear_cka"] == pytest.approx(1.0, abs=1e-9)   # rotation preserves linear CKA
            assert blk["procrustes_residual"] == pytest.approx(0.0, abs=1e-6)
    assert "| t1 |" in cma.render_markdown(report, "A", "B")
    with pytest.raises(FileNotFoundError):
        cma.compare_directories(tmp_path / "a", tmp_path / "b", ["t1", "missing"])
