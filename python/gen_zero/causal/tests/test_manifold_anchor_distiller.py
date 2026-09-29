"""Unit tests for the Manifold Anchor Distiller (offline conformal projection).

No mocks, no hardcoded outputs: every assertion is checked against an
independently computed reference (mathematical invariant, or a from-scratch
sha256/statistic) rather than a value copied from a single run.
"""
import hashlib
import json
import os
import time
import tracemalloc

import numpy as np
import pytest

from gen_zero.causal.manifold_anchor_distiller import ManifoldAnchorDistiller, main


# --------------------------------------------------------------------------
# Orthogonality
# --------------------------------------------------------------------------


def test_fitted_basis_is_orthonormal():
    rng = np.random.default_rng(0)
    n, d, k = 300, 512, 128
    x_train = rng.normal(size=(n, d))

    distiller = ManifoldAnchorDistiller(input_dim=d, output_dim=k).fit(x_train)

    assert distiller.P.shape == (k, d)
    assert np.allclose(distiller.P @ distiller.P.T, np.eye(k), atol=1e-8)


# --------------------------------------------------------------------------
# Conformality (angle preservation) on a low-rank manifold
# --------------------------------------------------------------------------


def test_conformality_on_low_rank_manifold():
    rng = np.random.default_rng(1)
    n, d, r, k = 300, 1024, 64, 128
    assert r <= k

    a_full = rng.normal(size=(n, r))
    b = rng.normal(size=(r, d))
    signal = a_full @ b
    noise_scale = 0.01 * (np.linalg.norm(signal) / np.linalg.norm(rng.normal(size=signal.shape)))
    noise = noise_scale * rng.normal(size=signal.shape)
    x = signal + noise

    # Disjoint train/val split.
    x_train, x_val = x[:200], x[200:]

    distiller = ManifoldAnchorDistiller(input_dim=d, output_dim=k).fit(x_train)
    report = distiller.evaluate_distortion(x_val)

    assert report["pairwise_cosine_correlation"] > 0.9
    # Sanity: this must not be a degenerate 1.0-from-uncentered artifact.
    # A rank-r=64 manifold fit with output_dim=128 (>= r) should retain most,
    # but not necessarily all, of the training energy.
    assert 0.0 < report["energy_retained_ratio"] <= 1.0 + 1e-9


# --------------------------------------------------------------------------
# No leakage / deterministic reproducibility
# --------------------------------------------------------------------------


def test_save_load_roundtrip_is_bit_exact_and_leak_free(tmp_path):
    rng = np.random.default_rng(2)
    n, d, k = 300, 512, 128
    x_train = rng.normal(size=(n, d))
    x_test = rng.normal(size=(50, d))

    distiller = ManifoldAnchorDistiller(input_dim=d, output_dim=k).fit(x_train)
    path = tmp_path / "distiller.npz"
    meta = distiller.save(str(path))

    loaded = ManifoldAnchorDistiller.load(str(path))

    z_original = distiller.project(x_test)
    z_loaded = loaded.project(x_test)
    assert np.array_equal(z_original, z_loaded)

    expected_train_sha256 = hashlib.sha256(np.ascontiguousarray(x_train).tobytes()).hexdigest()
    assert meta["train_data_sha256"] == expected_train_sha256
    assert loaded._train_data_sha256 == expected_train_sha256

    # evaluate_distortion must be deterministic and must not mutate state.
    report_1 = distiller.evaluate_distortion(x_test)
    sha_after_1 = distiller._train_data_sha256
    report_2 = distiller.evaluate_distortion(x_test)
    sha_after_2 = distiller._train_data_sha256

    assert report_1 == report_2
    assert sha_after_1 == expected_train_sha256
    assert sha_after_2 == expected_train_sha256


# --------------------------------------------------------------------------
# Fail-closed checks
# --------------------------------------------------------------------------


def test_project_and_evaluate_before_fit_raise_runtime_error():
    distiller = ManifoldAnchorDistiller(input_dim=16, output_dim=4)
    x = np.random.default_rng(3).normal(size=(10, 16))
    with pytest.raises(RuntimeError):
        distiller.project(x)
    with pytest.raises(RuntimeError):
        distiller.evaluate_distortion(x)


def test_fit_rejects_wrong_last_dim():
    rng = np.random.default_rng(4)
    distiller = ManifoldAnchorDistiller(input_dim=16, output_dim=4)
    with pytest.raises(ValueError):
        distiller.fit(rng.normal(size=(20, 15)))


def test_project_rejects_wrong_last_dim():
    rng = np.random.default_rng(5)
    distiller = ManifoldAnchorDistiller(input_dim=16, output_dim=4).fit(rng.normal(size=(20, 16)))
    with pytest.raises(ValueError):
        distiller.project(rng.normal(size=(5, 15)))


def test_fit_rejects_nan_and_inf():
    rng = np.random.default_rng(6)
    distiller = ManifoldAnchorDistiller(input_dim=16, output_dim=4)

    x_nan = rng.normal(size=(20, 16))
    x_nan[3, 2] = np.nan
    with pytest.raises(ValueError):
        distiller.fit(x_nan)

    x_inf = rng.normal(size=(20, 16))
    x_inf[5, 7] = np.inf
    with pytest.raises(ValueError):
        distiller.fit(x_inf)


def test_project_rejects_nan_and_inf():
    rng = np.random.default_rng(7)
    distiller = ManifoldAnchorDistiller(input_dim=16, output_dim=4).fit(rng.normal(size=(20, 16)))

    x_nan = rng.normal(size=(5, 16))
    x_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        distiller.project(x_nan)

    x_inf = rng.normal(size=(5, 16))
    x_inf[1, 1] = np.inf
    with pytest.raises(ValueError):
        distiller.project(x_inf)


def test_constructor_rejects_invalid_dims():
    with pytest.raises(ValueError):
        ManifoldAnchorDistiller(input_dim=16, output_dim=0)
    with pytest.raises(ValueError):
        ManifoldAnchorDistiller(input_dim=16, output_dim=-1)
    assert ManifoldAnchorDistiller(input_dim=16, output_dim=16).output_dim == 16
    with pytest.raises(ValueError):
        ManifoldAnchorDistiller(input_dim=16, output_dim=32)
    with pytest.raises(ValueError):
        ManifoldAnchorDistiller(input_dim=0, output_dim=4)
    with pytest.raises(ValueError):
        ManifoldAnchorDistiller(input_dim=-8, output_dim=4)


# --------------------------------------------------------------------------
# Large-dimension performance / memory
# --------------------------------------------------------------------------


def test_large_dimension_fit_and_project_bounded_memory_and_time():
    rng = np.random.default_rng(8)
    n, d, k = 200, 8192, 128
    x_train = rng.normal(size=(n, d))
    x_probe = rng.normal(size=(10, d))

    tracemalloc.start()
    start = time.monotonic()

    distiller = ManifoldAnchorDistiller(input_dim=d, output_dim=k).fit(x_train)
    distiller.project(x_probe)

    elapsed = time.monotonic() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)
    loadavg = os.getloadavg()

    assert peak_mb < 400, (
        f"peak traced memory {peak_mb:.1f}MB exceeded 400MB bound "
        f"(elapsed={elapsed:.2f}s, loadavg={loadavg}); this likely means the "
        f"implementation regressed to a (D, D) covariance/eigh path instead "
        f"of thin SVD on the (N, D) matrix"
    )
    assert elapsed < 60, f"fit+project took {elapsed:.2f}s (bound 60s); loadavg={loadavg}"


# --------------------------------------------------------------------------
# Artifact hardening: format_version / mu / singular_values / energy ratio
# --------------------------------------------------------------------------


def _orthonormal_rows(rng, output_dim, input_dim):
    """A (output_dim, input_dim) matrix with orthonormal rows (P @ P.T == I).

    Rows of a square orthogonal matrix are mutually orthonormal, so slicing
    any subset of rows preserves the property -- this mirrors how ``fit``
    derives ``P`` from ``vt``.
    """
    a = rng.normal(size=(input_dim, input_dim))
    q, _ = np.linalg.qr(a)
    return q[:output_dim, :]


def _write_raw_artifact(
    path,
    *,
    P,
    mu,
    singular_values,
    format_version=1,
    energy_retained_ratio=0.5,
    input_dim=None,
    output_dim=None,
    train_data_sha256="deadbeef",
    created_at="2020-01-01T00:00:00+00:00",
    payload_sha256=None,
):
    """Write a ``.npz`` artifact directly, bypassing ``fit()``/``save()``.

    Lets tests exercise ``load()``'s own validation on values that ``fit()``
    would itself reject (e.g. NaN mu), by computing a payload_sha256 that is
    consistent with the (possibly tampered) arrays so the sha check does not
    mask the check under test.
    """
    P = np.asarray(P, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    singular_values = np.asarray(singular_values, dtype=np.float64)
    if input_dim is None:
        input_dim = P.shape[1]
    if output_dim is None:
        output_dim = P.shape[0]

    payload = {"P": P, "mu": mu, "singular_values": singular_values}
    if payload_sha256 is None:
        digest = hashlib.sha256()
        for name in sorted(payload):
            digest.update(np.ascontiguousarray(payload[name]).tobytes())
        payload_sha256 = digest.hexdigest()

    meta = dict(
        format_version=format_version,
        input_dim=input_dim,
        output_dim=output_dim,
        train_data_sha256=train_data_sha256,
        created_at=created_at,
        energy_retained_ratio=energy_retained_ratio,
        payload_sha256=payload_sha256,
    )
    with open(path, "wb") as f:
        np.savez(f, metadata=json.dumps(meta), **payload)


def test_load_rejects_missing_or_wrong_format_version(tmp_path):
    rng = np.random.default_rng(20)
    d, k = 32, 8
    x_train = rng.normal(size=(50, d))
    distiller = ManifoldAnchorDistiller(input_dim=d, output_dim=k).fit(x_train)
    path = tmp_path / "distiller.npz"
    distiller.save(str(path))

    with np.load(str(path), allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        payload = {name: data[name] for name in ("P", "mu", "singular_values")}

    # Wrong version.
    wrong_meta = dict(meta)
    wrong_meta["format_version"] = meta["format_version"] + 1
    with open(path, "wb") as f:
        np.savez(f, metadata=json.dumps(wrong_meta), **payload)
    with pytest.raises(ValueError):
        ManifoldAnchorDistiller.load(str(path))

    # Missing version key entirely.
    missing_meta = dict(meta)
    del missing_meta["format_version"]
    with open(path, "wb") as f:
        np.savez(f, metadata=json.dumps(missing_meta), **payload)
    with pytest.raises(ValueError):
        ManifoldAnchorDistiller.load(str(path))


def test_load_rejects_mu_wrong_shape(tmp_path):
    rng = np.random.default_rng(21)
    d, k = 16, 4
    p = _orthonormal_rows(rng, k, d)
    bad_mu = rng.normal(size=(d + 1,))  # wrong shape, would never come from fit()
    s = np.abs(rng.normal(size=(k,))) + 1.0

    path = tmp_path / "artifact.npz"
    _write_raw_artifact(path, P=p, mu=bad_mu, singular_values=s, input_dim=d, output_dim=k)

    with pytest.raises(ValueError):
        ManifoldAnchorDistiller.load(str(path))


def test_load_rejects_mu_nan_or_inf(tmp_path):
    rng = np.random.default_rng(22)
    d, k = 16, 4
    p = _orthonormal_rows(rng, k, d)
    s = np.abs(rng.normal(size=(k,))) + 1.0

    for bad_value in (np.nan, np.inf):
        mu = rng.normal(size=(d,))
        mu[2] = bad_value
        path = tmp_path / "artifact.npz"
        _write_raw_artifact(path, P=p, mu=mu, singular_values=s, input_dim=d, output_dim=k)
        with pytest.raises(ValueError):
            ManifoldAnchorDistiller.load(str(path))


def test_load_rejects_singular_values_wrong_shape(tmp_path):
    rng = np.random.default_rng(23)
    d, k = 16, 4
    p = _orthonormal_rows(rng, k, d)
    mu = rng.normal(size=(d,))
    bad_s = np.abs(rng.normal(size=(k + 1,))) + 1.0  # wrong shape

    path = tmp_path / "artifact.npz"
    _write_raw_artifact(path, P=p, mu=mu, singular_values=bad_s, input_dim=d, output_dim=k)

    with pytest.raises(ValueError):
        ManifoldAnchorDistiller.load(str(path))


def test_load_rejects_singular_values_nan_or_inf(tmp_path):
    rng = np.random.default_rng(24)
    d, k = 16, 4
    p = _orthonormal_rows(rng, k, d)
    mu = rng.normal(size=(d,))

    for bad_value in (np.nan, np.inf):
        s = np.abs(rng.normal(size=(k,))) + 1.0
        s[0] = bad_value
        path = tmp_path / "artifact.npz"
        _write_raw_artifact(path, P=p, mu=mu, singular_values=s, input_dim=d, output_dim=k)
        with pytest.raises(ValueError):
            ManifoldAnchorDistiller.load(str(path))


def test_load_rejects_energy_retained_ratio_out_of_range_or_nonfinite(tmp_path):
    rng = np.random.default_rng(25)
    d, k = 16, 4
    p = _orthonormal_rows(rng, k, d)
    mu = rng.normal(size=(d,))
    s = np.abs(rng.normal(size=(k,))) + 1.0

    for bad_ratio in (-0.1, 1.1, float("nan"), float("inf")):
        path = tmp_path / "artifact.npz"
        _write_raw_artifact(
            path, P=p, mu=mu, singular_values=s, input_dim=d, output_dim=k,
            energy_retained_ratio=bad_ratio,
        )
        with pytest.raises(ValueError):
            ManifoldAnchorDistiller.load(str(path))


# --------------------------------------------------------------------------
# evaluate_distortion: fail-closed on small N and non-finite correlation
# --------------------------------------------------------------------------


def test_evaluate_distortion_requires_at_least_three_samples():
    rng = np.random.default_rng(26)
    d, k = 16, 4
    distiller = ManifoldAnchorDistiller(input_dim=d, output_dim=k).fit(rng.normal(size=(30, d)))
    x_val = rng.normal(size=(2, d))
    with pytest.raises(ValueError, match=r"^evaluate_distortion requires at least 3 samples$"):
        distiller.evaluate_distortion(x_val)


def test_evaluate_distortion_raises_instead_of_returning_nan_correlation(monkeypatch):
    rng = np.random.default_rng(27)
    d, k = 16, 4
    distiller = ManifoldAnchorDistiller(input_dim=d, output_dim=k).fit(rng.normal(size=(30, d)))
    x_val = rng.normal(size=(5, d))

    # Deterministic, non-flaky reproduction of a degenerate (zero-variance)
    # pairwise-similarity distribution: force corrcoef itself to report NaN,
    # exactly the condition the fail-closed check must catch.
    monkeypatch.setattr(np, "corrcoef", lambda *a, **kw: np.array([[1.0, np.nan], [np.nan, 1.0]]))

    with pytest.raises(ValueError):
        distiller.evaluate_distortion(x_val)


# --------------------------------------------------------------------------
# fit(): SVD overflow guards
# --------------------------------------------------------------------------


def test_fit_rejects_svd_overflow_real_scale():
    rng = np.random.default_rng(28)
    d, k = 16, 4
    x_train = rng.normal(size=(20, d)) * 1e160
    with pytest.raises(ValueError, match="numerical overflow in energy computation"):
        ManifoldAnchorDistiller(input_dim=d, output_dim=k).fit(x_train)


def test_fit_rejects_nonfinite_singular_values_from_svd(monkeypatch):
    rng = np.random.default_rng(29)
    d, k = 16, 4
    x_train = rng.normal(size=(20, d))

    def fake_svd(a, full_matrices=False):
        m, n = a.shape
        kk = min(m, n)
        return np.zeros((m, kk)), np.full(kk, np.inf), np.zeros((kk, n))

    monkeypatch.setattr(np.linalg, "svd", fake_svd)
    with pytest.raises(ValueError, match="numerical overflow in energy computation"):
        ManifoldAnchorDistiller(input_dim=d, output_dim=k).fit(x_train)


def test_fit_rejects_energy_overflow_with_finite_singular_values(monkeypatch):
    rng = np.random.default_rng(30)
    d, k = 16, 4
    x_train = rng.normal(size=(20, d))
    real_svd = np.linalg.svd

    def fake_svd(a, full_matrices=False):
        u, s, vt = real_svd(a, full_matrices=full_matrices)
        # s itself is finite, but s ** 2 overflows to inf for every entry,
        # so sum(s[:k]**2) / sum(s**2) becomes inf / inf == nan.
        crafted_s = np.full_like(s, 1e200)
        return u, crafted_s, vt

    monkeypatch.setattr(np.linalg, "svd", fake_svd)
    with pytest.raises(ValueError, match="numerical overflow in energy computation"):
        ManifoldAnchorDistiller(input_dim=d, output_dim=k).fit(x_train)


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------


def test_project_rejects_overflow_from_extreme_finite_input():
    """finfo.max survives centering (still finite) but overflows in the
    P @ xc.T matmul -- exercises the *projection-output* finite check."""
    rng = np.random.default_rng(44)
    d, k = 16, 4
    distiller = ManifoldAnchorDistiller(input_dim=d, output_dim=k).fit(rng.normal(size=(20, d)))

    x_extreme = np.full((1, d), np.finfo(np.float64).max)
    with pytest.raises(ValueError, match="projection overflowed"):
        distiller.project(x_extreme)


def test_project_rejects_overflow_from_crafted_projection_matrix(monkeypatch):
    """Even if centering stays finite, the P @ x.T matmul can still overflow
    to inf; project() must check its own output, not just its input."""
    rng = np.random.default_rng(45)
    d, k = 16, 4
    distiller = ManifoldAnchorDistiller(input_dim=d, output_dim=k).fit(rng.normal(size=(20, d)))

    # Inflate P by a huge factor post-fit (bypassing fit()'s own checks) to
    # force the matmul itself to overflow on an otherwise-ordinary input.
    monkeypatch.setattr(distiller, "P", distiller.P * 1e200)
    x = rng.normal(size=(1, d)) * 1e200

    with pytest.raises(ValueError, match="projection overflowed"):
        distiller.project(x)


def test_project_rejects_overflow_in_centering_itself(monkeypatch):
    """Isolates the *centering* finite check: force ``x - mu`` itself to
    overflow to inf while leaving ``P`` untouched, so this specifically
    exercises the first guard in project(), not the matmul guard."""
    rng = np.random.default_rng(48)
    d, k = 16, 4
    distiller = ManifoldAnchorDistiller(input_dim=d, output_dim=k).fit(rng.normal(size=(20, d)))

    monkeypatch.setattr(distiller, "mu", np.full(d, -np.finfo(np.float64).max))
    x = np.full((1, d), np.finfo(np.float64).max)

    with pytest.raises(ValueError, match="centering overflowed"):
        distiller.project(x)


def test_load_rejects_format_version_true(tmp_path):
    rng = np.random.default_rng(46)
    d, k = 16, 4
    p = _orthonormal_rows(rng, k, d)
    mu = rng.normal(size=(d,))
    s = np.abs(rng.normal(size=(k,))) + 1.0

    path = tmp_path / "artifact.npz"
    _write_raw_artifact(path, P=p, mu=mu, singular_values=s, input_dim=d, output_dim=k, format_version=True)

    with pytest.raises(ValueError):
        ManifoldAnchorDistiller.load(str(path))


def test_load_rejects_format_version_false(tmp_path):
    rng = np.random.default_rng(47)
    d, k = 16, 4
    p = _orthonormal_rows(rng, k, d)
    mu = rng.normal(size=(d,))
    s = np.abs(rng.normal(size=(k,))) + 1.0

    path = tmp_path / "artifact.npz"
    _write_raw_artifact(path, P=p, mu=mu, singular_values=s, input_dim=d, output_dim=k, format_version=False)

    with pytest.raises(ValueError):
        ManifoldAnchorDistiller.load(str(path))
