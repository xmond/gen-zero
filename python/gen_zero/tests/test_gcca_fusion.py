import json
import numpy as np
import pytest

from gen_zero.manifold import GCCAMidFusion, RelativeAnchorEncoder
from gen_zero.manifold.gcca_fusion import _numeric_rank, _ridge_inverse_and_inverse_sqrt


def _principal_angle_cosines(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    """Singular values of Qa^T Qb for orthonormal bases Qa, Qb: 1.0 means identical subspaces."""
    qa, _ = np.linalg.qr(basis_a)
    qb, _ = np.linalg.qr(basis_b)
    return np.linalg.svd(qa.T @ qb, compute_uv=False)


def _make_synthetic_views(n_samples=500, shared_rank=5, dims=(30, 40, 50), noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    g_true, _ = np.linalg.qr(rng.normal(size=(n_samples, shared_rank)))
    views = []
    for d in dims:
        loading = rng.normal(size=(shared_rank, d))
        z = g_true @ loading + noise * rng.normal(size=(n_samples, d))
        views.append(z)
    return g_true, views


def test_gcca_recovers_shared_subspace_on_synthetic_multiview_data():
    g_true, views = _make_synthetic_views(n_samples=500, shared_rank=5, dims=(30, 40, 50), noise=0.01, seed=0)

    model = GCCAMidFusion()
    model.fit(views, shared_dim=5, residual_dim=4, reg=1e-4)
    fused = model.transform(views)
    shared_hat = fused[:, :5]

    cosines = _principal_angle_cosines(shared_hat, g_true)
    assert cosines.shape == (5,)
    assert np.all(cosines > 0.9), f"principal angle cosines too low: {cosines}"


def test_gcca_residual_is_orthogonal_to_shared_basis():
    _, views = _make_synthetic_views(n_samples=300, shared_rank=4, dims=(20, 25), noise=0.05, seed=1)

    model = GCCAMidFusion()
    model.fit(views, shared_dim=4, residual_dim=6, reg=1e-3)

    for z, view in zip(views, model.views_):
        centered = z - view.mean
        residual = centered - centered @ view.shared_basis @ view.shared_basis.T

        # U_m^T r_m^T == (r_m @ U_m)^T must vanish: residual carries none of the
        # shared-basis directions that were projected out of it.
        cross = residual @ view.shared_basis
        np.testing.assert_allclose(cross, np.zeros_like(cross), atol=1e-8)

        # Truncated residual directions come from residual's own row space, which is
        # orthogonal to shared_basis by the fundamental theorem of linear algebra, so
        # this must also vanish after PCA truncation.
        truncated_cross = view.residual_directions.T @ view.shared_basis
        np.testing.assert_allclose(truncated_cross, np.zeros_like(truncated_cross), atol=1e-8)


def test_gcca_fused_output_shape_and_input_validation():
    _, views = _make_synthetic_views(n_samples=100, shared_rank=3, dims=(10, 12), noise=0.02, seed=2)

    model = GCCAMidFusion()
    model.fit(views, shared_dim=3, residual_dim=5, reg=1e-3)
    fused = model.transform(views)
    assert fused.shape == (100, 3 + 5 + 5)

    with pytest.raises(RuntimeError, match="fit"):
        GCCAMidFusion().transform(views)

    with pytest.raises(ValueError, match="number of aligned samples"):
        GCCAMidFusion().fit([views[0], views[1][:-1]])

    with pytest.raises(ValueError, match="fitted views"):
        model.transform(views[:1])


def test_ridge_inverse_helper_matches_direct_computation():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 6))
    gram = x.T @ x
    reg = 0.1
    inv, inv_sqrt = _ridge_inverse_and_inverse_sqrt(gram, reg)

    regularized = gram + reg * np.eye(6)
    np.testing.assert_allclose(inv @ regularized, np.eye(6), atol=1e-8)
    np.testing.assert_allclose(inv_sqrt @ inv_sqrt @ regularized, np.eye(6), atol=1e-6)


def test_numeric_rank_counts_only_significant_singular_values():
    assert _numeric_rank(np.array([5.0, 4.0, 1e-14]), ambient_dim=3) == 2
    assert _numeric_rank(np.array([]), ambient_dim=3) == 0
    assert _numeric_rank(np.array([0.0, 0.0]), ambient_dim=2) == 0


def test_gcca_residual_dim_overshoot_does_not_leak_shared_directions_back_in():
    # dims=(10, 12) with shared_dim=8 leaves only 2 (resp. 4) private dimensions per view;
    # requesting residual_dim=16 used to silently keep the trailing near-zero singular
    # vectors of `residual`, which span exactly shared_basis's column space (its null
    # space), pulling shared directions back into the supposedly "private" residual.
    _, views = _make_synthetic_views(n_samples=200, shared_rank=6, dims=(10, 12), noise=0.05, seed=5)

    model = GCCAMidFusion()
    model.fit(views, shared_dim=8, residual_dim=16, reg=1e-3)

    for z, view in zip(views, model.views_):
        assert view.residual_directions.shape[1] <= z.shape[1] - view.shared_basis.shape[1]
        truncated_cross = view.residual_directions.T @ view.shared_basis
        np.testing.assert_allclose(truncated_cross, np.zeros_like(truncated_cross), atol=1e-8)


def test_gcca_transform_generalizes_to_held_out_samples():
    rng = np.random.default_rng(9)
    shared_rank, dims, noise = 4, (20, 30), 0.01
    g_train, _ = np.linalg.qr(rng.normal(size=(400, shared_rank)))
    g_test, _ = np.linalg.qr(rng.normal(size=(150, shared_rank)))
    loadings = [rng.normal(size=(shared_rank, d)) for d in dims]
    train_views = [g_train @ w + noise * rng.normal(size=(400, d)) for w, d in zip(loadings, dims)]
    test_views = [g_test @ w + noise * rng.normal(size=(150, d)) for w, d in zip(loadings, dims)]

    model = GCCAMidFusion()
    model.fit(train_views, shared_dim=shared_rank, residual_dim=4, reg=1e-4)
    fused_test = model.transform(test_views)

    cosines = _principal_angle_cosines(fused_test[:, :shared_rank], g_test)
    assert np.all(cosines > 0.85), f"held-out subspace recovery too weak: {cosines}"


def test_gcca_nonuniform_weights_change_the_shared_fit():
    _, views = _make_synthetic_views(n_samples=200, shared_rank=4, dims=(15, 15), noise=0.2, seed=6)

    uniform = GCCAMidFusion().fit(views, shared_dim=4, residual_dim=3, reg=1e-3, weights=[1.0, 1.0])
    skewed = GCCAMidFusion().fit(views, shared_dim=4, residual_dim=3, reg=1e-3, weights=[10.0, 0.1])

    uniform_shared = uniform.transform(views)[:, :4]
    skewed_shared = skewed.transform(views)[:, :4]
    assert not np.allclose(uniform_shared, skewed_shared, atol=1e-3)


def test_gcca_fail_closed_on_invalid_arguments():
    _, views = _make_synthetic_views(n_samples=50, shared_rank=3, dims=(8, 8), noise=0.02, seed=8)

    with pytest.raises(ValueError, match="shared_dim"):
        GCCAMidFusion().fit(views, shared_dim=0)
    with pytest.raises(ValueError, match="shared_dim"):
        GCCAMidFusion().fit(views, shared_dim=100)  # exceeds min(n_samples, sum feature dims)
    with pytest.raises(ValueError, match="residual_dim"):
        GCCAMidFusion().fit(views, residual_dim=-1)
    with pytest.raises(ValueError, match="reg"):
        GCCAMidFusion().fit(views, reg=0.0)
    with pytest.raises(ValueError, match="reg"):
        GCCAMidFusion().fit(views, reg=-1.0)
    with pytest.raises(ValueError, match="weights"):
        GCCAMidFusion().fit(views, weights=[1.0, -2.0])
    with pytest.raises(ValueError, match="weights"):
        GCCAMidFusion().fit(views, weights=[1.0])  # wrong length


def test_gcca_save_load_round_trip(tmp_path):
    _, views = _make_synthetic_views(n_samples=150, shared_rank=4, dims=(12, 18), noise=0.03, seed=10)

    model = GCCAMidFusion().fit(views, shared_dim=4, residual_dim=5, reg=1e-3, weights=[1.0, 2.0])
    expected = model.transform(views)

    path = tmp_path / "gcca.npz"
    meta = model.save(path)
    assert meta["format_version"] == 1

    loaded = GCCAMidFusion.load(path)
    actual = loaded.transform(views)
    np.testing.assert_allclose(actual, expected, atol=1e-10)

    # tamper with the payload: load must reject it rather than silently using corrupt data
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files if k != "metadata"}
        metadata = str(data["metadata"])
    arrays["encoder_0"] = arrays["encoder_0"] + 1.0
    with open(path, "wb") as f:
        np.savez(f, metadata=metadata, **arrays)
    with pytest.raises(ValueError, match="sha256"):
        GCCAMidFusion.load(path)


def _make_anchor_views(n_samples=200, dims=(16, 64), seed=7):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(n_samples, d)) for d in dims]


def test_relative_anchor_encoder_dimension_alignment_across_heterogeneous_views():
    views = _make_anchor_views(n_samples=200, dims=(16, 64), seed=7)

    encoder = RelativeAnchorEncoder()
    encoder.fit(views, n_anchors=24, seed=42)
    z_anchor = encoder.transform(views)

    assert z_anchor.shape == (200, 2 * 24)


def test_relative_anchor_encoder_is_bounded_and_nonnegative():
    views = _make_anchor_views(n_samples=150, dims=(8, 32, 100), seed=11)

    encoder = RelativeAnchorEncoder()
    encoder.fit(views, n_anchors=16, seed=42)
    z_anchor = encoder.transform(views)

    assert np.all(z_anchor >= 0.0)
    assert np.all(z_anchor <= 1.0)


def test_relative_anchor_encoder_self_similarity_at_anchor_points():
    views = _make_anchor_views(n_samples=100, dims=(10, 20), seed=13)

    encoder = RelativeAnchorEncoder()
    encoder.fit(views, n_anchors=8, seed=42)
    z_anchor = encoder.transform(views)

    indices = encoder.anchor_indices_
    for view_idx in range(len(views)):
        block = z_anchor[:, view_idx * 8:(view_idx + 1) * 8]
        for anchor_pos, sample_idx in enumerate(indices):
            # cos(h_m(a_j) - mu_m, h_m(a_j) - mu_m) == 1 -> rescaled coordinate == 1
            assert block[sample_idx, anchor_pos] == pytest.approx(1.0, abs=1e-8)


def test_relative_anchor_encoder_input_validation():
    views = _make_anchor_views(n_samples=50, dims=(10, 10), seed=17)

    with pytest.raises(RuntimeError, match="fit"):
        RelativeAnchorEncoder().transform(views)

    with pytest.raises(ValueError, match="number of aligned samples"):
        RelativeAnchorEncoder().fit([views[0], views[1][:-1]])

    encoder = RelativeAnchorEncoder()
    encoder.fit(views, n_anchors=5, seed=42)
    with pytest.raises(ValueError, match="fitted views"):
        encoder.transform(views[:1])


def test_relative_anchor_encoder_fail_closed_on_invalid_arguments():
    views = _make_anchor_views(n_samples=30, dims=(6, 6), seed=19)

    with pytest.raises(ValueError, match="n_anchors"):
        RelativeAnchorEncoder().fit(views, n_anchors=0)
    with pytest.raises(ValueError, match="n_anchors"):
        RelativeAnchorEncoder().fit(views, n_anchors=1000)


def test_relative_anchor_encoder_rejects_zero_norm_centered_row_instead_of_faking_similarity():
    # A row exactly at the view's own mean has an undefined cosine direction; silently
    # returning 0.5 there (the (cos+1)/2 midpoint) would fabricate a similarity score.
    views = _make_anchor_views(n_samples=40, dims=(5,), seed=23)
    encoder = RelativeAnchorEncoder()
    encoder.fit(views, n_anchors=6, seed=42)

    degenerate = [views[0].copy()]
    degenerate[0][0] = encoder.views_[0].mean[0]  # exactly at the view mean after centering
    with pytest.raises(ValueError, match="cosine is undefined"):
        encoder.transform(degenerate)


def test_relative_anchor_encoder_save_load_round_trip(tmp_path):
    views = _make_anchor_views(n_samples=120, dims=(14, 22), seed=29)
    encoder = RelativeAnchorEncoder().fit(views, n_anchors=10, seed=42)
    expected = encoder.transform(views)

    path = tmp_path / "anchor.npz"
    meta = encoder.save(path)
    assert meta["format_version"] == 1

    loaded = RelativeAnchorEncoder.load(path)
    actual = loaded.transform(views)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    np.testing.assert_array_equal(loaded.anchor_indices_, encoder.anchor_indices_)


def test_cli_manifold_fuse_gcca_fit_save_load_end_to_end(tmp_path):
    # Exercises the real `gen-zero manifold-fuse` production entry point (cli.py), not
    # just the library classes, so this module is actually reachable from the CLI.
    from gen_zero.cli import main

    _, views = _make_synthetic_views(n_samples=80, shared_rank=3, dims=(9, 11), noise=0.02, seed=31)
    features_path = tmp_path / "views.npz"
    np.savez(features_path, view_a=views[0], view_b=views[1])

    artifact_path = tmp_path / "gcca_artifact.npz"
    output_path = tmp_path / "fused.npy"
    exit_code = main([
        "manifold-fuse", "--no-color",
        "--features", str(features_path),
        "--view-keys", "view_a", "view_b",
        "--shared-dim", "3", "--residual-dim", "2", "--reg", "1e-3",
        "--save-artifact", str(artifact_path),
        "--output", str(output_path),
        "--json",
    ])
    assert exit_code == 0
    assert artifact_path.exists()
    fused = np.load(output_path)
    assert fused.shape == (80, 3 + 2 + 2)

    # Re-run loading the saved artifact instead of refitting; must reproduce the same output.
    output_path_2 = tmp_path / "fused_from_artifact.npy"
    exit_code = main([
        "manifold-fuse", "--no-color",
        "--features", str(features_path),
        "--view-keys", "view_a", "view_b",
        "--artifact", str(artifact_path),
        "--output", str(output_path_2),
        "--json",
    ])
    assert exit_code == 0
    np.testing.assert_allclose(np.load(output_path_2), fused, atol=1e-10)


def test_cli_manifold_fuse_anchor_mode_end_to_end(tmp_path):
    from gen_zero.cli import main

    views = _make_anchor_views(n_samples=60, dims=(7, 13), seed=37)
    features_path = tmp_path / "views.npz"
    np.savez(features_path, view_a=views[0], view_b=views[1])
    output_path = tmp_path / "anchor_fused.npy"

    exit_code = main([
        "manifold-fuse", "--no-color",
        "--mode", "anchor",
        "--features", str(features_path),
        "--view-keys", "view_a", "view_b",
        "--n-anchors", "5", "--seed", "42",
        "--output", str(output_path),
        "--json",
    ])
    assert exit_code == 0
    fused = np.load(output_path)
    assert fused.shape == (60, 2 * 5)
    assert np.all(fused >= 0.0) and np.all(fused <= 1.0)


def test_cli_manifold_fuse_reports_error_on_missing_view_key(tmp_path):
    from gen_zero.cli import main

    views = _make_anchor_views(n_samples=20, dims=(4,), seed=41)
    features_path = tmp_path / "views.npz"
    np.savez(features_path, only_view=views[0])

    exit_code = main([
        "manifold-fuse", "--no-color",
        "--features", str(features_path),
        "--view-keys", "only_view", "missing_view",
    ])
    assert exit_code == 1


def test_gcca_full_shared_dim_has_zero_residual_directions():
    # Counterexample from Astra: full shared space should yield empty residual basis (0 columns),
    # never a spurious private basis overlapping the shared space.
    rng = np.random.default_rng(42)
    x = rng.normal(size=(100, 4)) * 1e6
    m = GCCAMidFusion().fit([x], shared_dim=4, residual_dim=4, reg=1.0)
    v = m.views_[0]
    assert v.residual_directions.shape == (4, 0), f"expected 0 residual directions, got {v.residual_directions.shape}"
    # Transform output should strictly contain only the shared dimensions
    fused = m.transform([x])
    assert fused.shape == (100, 4)


def test_gcca_rejects_nan_and_inf_fail_closed():
    # Counterexample from Astra: Non-finite inputs must raise ValueError, not silently produce NaNs
    rng = np.random.default_rng(123)
    x_nan = rng.normal(size=(50, 8))
    x_nan[5, 2] = np.nan
    x_valid = rng.normal(size=(50, 8))

    model = GCCAMidFusion()
    with pytest.raises(ValueError, match="non-finite"):
        model.fit([x_nan], shared_dim=4)

    model.fit([x_valid], shared_dim=4)
    with pytest.raises(ValueError, match="non-finite"):
        model.transform([x_nan])

    anchor = RelativeAnchorEncoder()
    with pytest.raises(ValueError, match="non-finite"):
        anchor.fit([x_nan], n_anchors=10)

    anchor.fit([x_valid], n_anchors=10)
    with pytest.raises(ValueError, match="non-finite"):
        anchor.transform([x_nan])


def test_gcca_tampered_weights_rejected_by_sha256(tmp_path):
    # Counterexample from Fable & Astra: Modifying weights in artifact metadata must trigger sha256 mismatch
    _, views = _make_synthetic_views(n_samples=60, shared_rank=4, dims=(8, 10), seed=42)
    model = GCCAMidFusion().fit(views, shared_dim=4, residual_dim=2, reg=1e-2, weights=[1.0, 2.0])
    save_path = tmp_path / "gcca_tampered.npz"
    model.save(save_path)

    # Tamper with weights inside the saved npz
    with np.load(save_path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        meta["weights"] = [100.0, 0.001]
        payload = {k: data[k] for k in data.files if k != "metadata"}

    with open(save_path, "wb") as f:
        np.savez(f, metadata=json.dumps(meta), **payload)

    with pytest.raises(ValueError, match="payload_sha256 mismatch"):
        GCCAMidFusion.load(save_path)


def test_gcca_private_residuals_do_not_leak_shared_subspace():
    # Counterexample from Fable: G must not be linearly predictable from the residual representations
    g_true, views = _make_synthetic_views(n_samples=200, shared_rank=4, dims=(16, 20), seed=77)
    model = GCCAMidFusion().fit(views, shared_dim=4, residual_dim=4, reg=1e-2)
    fused = model.transform(views)
    residuals = fused[:, 4:]  # (200, 8)
    assert residuals.shape[1] > 0

    # Ridge regression from residuals -> G: R^2 must be near zero (< 0.1)
    r_mean = residuals - residuals.mean(axis=0)
    g_mean = g_true - g_true.mean(axis=0)
    w_leak = np.linalg.solve(r_mean.T @ r_mean + 1e-3 * np.eye(r_mean.shape[1]), r_mean.T @ g_mean)
    g_pred = r_mean @ w_leak
    r2 = 1.0 - np.sum((g_mean - g_pred) ** 2) / np.sum(g_mean ** 2)
    assert r2 < 0.1, f"Residuals leak shared signal G with R^2={r2:.4f}, expected < 0.1"
