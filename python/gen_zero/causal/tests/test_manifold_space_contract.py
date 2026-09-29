"""Numerical contract tests; generated matrices are not model quality evidence."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "benchmarks" / "suites"))
from generalized_cca_manifold_interference import GeneralizedCCAManifoldInterference
from gen_zero.causal.manifold_anchor_distiller import ManifoldAnchorDistiller, main
from gen_zero.causal.nanocore_bridge import NanocoreAnchorBridge


def make_transform(tmp_path, width=128, n_fit=270, n_eval=30):
    """Returns (x, fit, path, source_space). ``source_space`` is the 3-key RAW
    SOURCE identity {source_model, layer, norm} -- NOT the full GCCA manifest
    ``save_transform`` writes into the artifact (which additionally carries
    GCCA_map). See ``manifold_anchor_distiller.transform_source``: a raw
    sample's space can never carry GCCA_map, since the sample is written
    before any GCCA transform exists.
    """
    rng = np.random.default_rng(471)
    x = rng.normal(size=(n_fit + n_eval, width)) + rng.normal(size=width)
    y = x @ rng.normal(size=(width, width)) + rng.normal(size=x.shape) * .01
    op = GeneralizedCCAManifoldInterference(max_rank=width, n_shared=width, standardize=True)
    fit = op.fit([x[:n_fit], y[:n_fit]])
    path = tmp_path / "gcca.npz"
    manifest = fit.save_transform(path, 0, source_model="numerical-test-source", layer="last",
                                  norm="none", source_data=x, fit_indices=np.arange(n_fit),
                                  eval_indices=np.arange(n_fit, n_fit + n_eval))
    source_space = {k: manifest[k] for k in ("source_model", "layer", "norm")}
    return x, fit, path, source_space


def test_maxvar_map_matches_primal_ridge_and_heldout_statistics(tmp_path):
    x, fit, path, space = make_transform(tmp_path, 8, 40, 10)
    xc = (x[:40] - fit.means[0]) / fit.scales[0]
    expected = np.linalg.solve(xc.T @ xc + fit.ridges[0] * 39 * np.eye(8), xc.T @ fit.shared_basis)
    np.testing.assert_allclose(fit.maps[0], expected, atol=1e-12)
    anchor = ManifoldAnchorDistiller(8, 8).attach_gcca(path).fit(fit.transform(x[:40]))
    loaded_x = ((x[40:] - anchor.gcca["mean"]) / anchor.gcca["scale"]) @ anchor.gcca["W"]
    np.testing.assert_allclose(fit.transform(x[40:]), loaded_x)
    np.testing.assert_allclose(anchor.transform_source(x[40:], space), anchor.project(loaded_x))
    with pytest.raises(ValueError, match="already fitted"):
        anchor.fit(loaded_x)


def test_cli_saved_128_space_to_bridge_and_same_width_rejection(tmp_path, capsys):
    x, fit, transform, space = make_transform(tmp_path)
    features = tmp_path / "features.npz"
    np.savez(features, train_full=x)
    artifact = tmp_path / "anchor.npz"
    core_digest = hashlib.sha256(b"numerical test core artifact").hexdigest()
    assert main(["--features", str(features), "--gcca-transform", str(transform),
                 "--core-sha256", core_digest, "--domain-id", "7", "--out", str(artifact)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert (summary["n_fit"], summary["n_eval"], summary["input_dim"]) == (270, 30, 128)
    anchor = ManifoldAnchorDistiller.load(artifact)
    bridge = NanocoreAnchorBridge(artifact, core_manifest=anchor.space)
    sample = {"values": x[-1], "space": space}
    payload = bridge.generate_mcp_ask_payload(sample, 7, ["continue", "stop"])
    np.testing.assert_allclose(payload["nanocore_state"], anchor.project(fit.transform(x[-1:]))[0], rtol=1e-6, atol=1e-8)
    for field in ("source_model", "layer", "norm"):
        with pytest.raises(ValueError, match="space identity"):
            bridge.project_to_nanocore_state({"values": x[-1], "space": {**space, field: "other"}})
    # GCCA_map is not part of a raw sample's 3-key source space (Defect 1): a
    # raw feature file is written before any GCCA transform exists, so it can
    # never carry a GCCA_map. That identity instead lives in the anchor's own
    # manifest and is checked when the bridge compares the artifact's space
    # against an independently supplied core manifest.
    for field in ("anchor_basis", "core", "domain_id", "GCCA_map"):
        with pytest.raises(ValueError, match="core manifest"):
            NanocoreAnchorBridge(artifact, core_manifest={**anchor.space, field: "other"})
    with pytest.raises(ValueError, match="explicit source"):
        bridge.project_to_nanocore_state(x[-1])
    with pytest.raises(ValueError, match="domain"):
        bridge.generate_mcp_ask_payload(sample, 8, ["stop"])
    with pytest.raises(ValueError, match="core manifest"):
        NanocoreAnchorBridge(artifact)


@pytest.mark.parametrize("problem", ["overlap", "out_of_range", "wrong_data", "small_fit", "small_eval"])
def test_cli_refuses_bad_partition(tmp_path, problem):
    x, _, transform, _ = make_transform(tmp_path, 8, 40, 10)
    with np.load(transform, allow_pickle=False) as data:
        contents = dict(data)
    if problem == "overlap":
        contents["eval_indices"][0] = 0
    elif problem == "out_of_range":
        contents["eval_indices"][0] = len(x)
    elif problem == "wrong_data":
        x[0, 0] += 1
    elif problem == "small_fit":
        contents["fit_indices"] = np.arange(4)
    else:
        contents["eval_indices"] = np.array([49])
    np.savez(transform, **contents)
    features = tmp_path / "features.npz"
    np.savez(features, train_full=x)
    with pytest.raises(ValueError):
        main(["--features", str(features), "--gcca-transform", str(transform),
              "--core-sha256", hashlib.sha256(b"core").hexdigest(), "--domain-id", "0",
              "--output-dim", "8", "--out", str(tmp_path / "anchor.npz")])


def test_rank64_cannot_manufacture_128_dimensions():
    rng = np.random.default_rng(51)
    views = [rng.normal(size=(200, 160)) for _ in range(2)]
    with pytest.raises(ValueError, match="available shared"):
        GeneralizedCCAManifoldInterference(max_rank=64, n_shared=128).fit(views)
    with pytest.raises(ValueError):
        ManifoldAnchorDistiller(64, 128)


def extraction_info():
    return {
        "encoder": "gguf:test-model:llama-server:pooling=last:embd_normalize=-1",
        "pooling": "llama-server --pooling last (final post-norm state, last token)",
        "embd_normalize": -1,
    }


@pytest.mark.parametrize("normalization", [0, 2, None, True, "-1"])
def test_source_space_rejects_conflicting_or_invalid_normalization(normalization):
    from gen_zero.causal.feature_space import source_space_from_info
    with pytest.raises(ValueError, match="normaliz"):
        source_space_from_info({**extraction_info(), "embd_normalize": normalization})


@pytest.mark.parametrize("info", [None, {}, [], {"encoder": extraction_info()["encoder"]}])
def test_explicit_space_cannot_bypass_invalid_info(tmp_path, info):
    from gen_zero.causal.feature_space import load_source_space, source_space_from_info
    path = tmp_path / "features.npz"
    space = source_space_from_info(extraction_info())
    np.savez(path, space=json.dumps(space), info_json=json.dumps(info))
    with pytest.raises(ValueError):
        load_source_space(path)


def test_feature_store_dual_source_consistency_and_logged_derivation(tmp_path, caplog):
    from gen_zero.causal.feature_space import load_source_space, source_space_from_info
    info = extraction_info()
    space = source_space_from_info(info)
    path = tmp_path / "features.npz"
    np.savez(path, space=json.dumps(space), info_json=json.dumps(info))
    assert load_source_space(path) == space
    np.savez(path, space=json.dumps(space), info_json=json.dumps({**info, "embd_normalize": 2}))
    with pytest.raises(ValueError, match="normalization disagrees"):
        load_source_space(path)
    np.savez(path, space=json.dumps({**space, "norm": "other"}), info_json=json.dumps(info))
    with pytest.raises(ValueError, match="disagrees"):
        load_source_space(path)
    np.savez(path, info_json=json.dumps(info))
    with caplog.at_level("WARNING"):
        assert load_source_space(path) == space
    assert "no explicit 'space' key found" in caplog.text
