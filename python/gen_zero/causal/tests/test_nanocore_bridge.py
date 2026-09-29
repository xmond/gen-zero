"""Bridge identity and wire contract tests; no claim of trained core inference.

The old BoolQ artifact lacks GCCA/source/core provenance and must be rejected.
Numerical fixtures exercise the new bridge; live acceptance requires a trained
core with the matching manifest, which this repository fixture does not supply.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gen_zero.causal.manifold_anchor_distiller import ManifoldAnchorDistiller
from gen_zero.causal.nanocore_bridge import NanocoreAnchorBridge
from test_manifold_space_contract import make_transform

ARTIFACT_PATH = Path(__file__).resolve().parents[4] / "benchmarks/results/manifold/distilled_128d_llama70b_boolq.npz"


@pytest.fixture(scope="module")
def bound_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("bound-anchor")
    x, fit, path, space = make_transform(root)
    anchor = ManifoldAnchorDistiller(128, 128).attach_gcca(path).fit(fit.transform(x[:270]))
    anchor.bind_core(hashlib.sha256(b"numerical contract test, not trained core").hexdigest(), 7)
    artifact = root / "anchor.npz"
    anchor.save(artifact)
    return root, artifact, anchor.space, {"values": x[-1], "space": space}


def test_legacy_real_artifact_is_rejected_before_projection():
    with pytest.raises(ValueError, match="format_version"):
        NanocoreAnchorBridge(ARTIFACT_PATH)


@pytest.mark.parametrize("domain", [-1, 2**32, True, 1.0, "7", None])
def test_invalid_domain(bound_fixture, domain):
    _, path, manifest, row = bound_fixture
    bridge = NanocoreAnchorBridge(path, core_manifest=manifest)
    with pytest.raises(ValueError, match="domain_id"):
        bridge.generate_mcp_ask_payload(row, domain, ["stop"])


@pytest.mark.parametrize("candidates", ["stop", b"stop", [], None, [" "], [1], ["go", "go"], [str(i) for i in range(17)]])
def test_invalid_candidates(bound_fixture, candidates):
    _, path, manifest, row = bound_fixture
    bridge = NanocoreAnchorBridge(path, core_manifest=manifest)
    with pytest.raises((ValueError, TypeError), match="candidates"):
        bridge.generate_mcp_ask_payload(row, 7, candidates)


@pytest.mark.parametrize("value", [np.zeros((2, 128)), np.zeros(127), np.full(128, np.nan), np.full(128, np.inf)])
def test_invalid_source_vectors(bound_fixture, value):
    _, path, manifest, row = bound_fixture
    bridge = NanocoreAnchorBridge(path, core_manifest=manifest)
    with pytest.raises(ValueError):
        bridge.project_to_nanocore_state({**row, "values": value})


def test_float32_overflow_is_rejected_without_mocking_projection(bound_fixture):
    _, path, manifest, row = bound_fixture
    bridge = NanocoreAnchorBridge(path, core_manifest=manifest)
    with pytest.raises(ValueError, match="float32"):
        bridge.project_to_nanocore_state({**row, "values": np.full(128, 1e100)})


def test_cli_real_entry_builds_payload_from_bound_numerical_artifact(bound_fixture, capsys):
    from gen_zero.cli import main
    root, artifact, manifest, row = bound_fixture
    features = root / "features.npz"
    np.savez(features, test_full=row["values"][None, :], space=json.dumps(row["space"]))
    core_manifest = root / "core.json"
    core_manifest.write_text(json.dumps(manifest))
    rc = main(["anchor", "--features", str(features), "--artifact", str(artifact),
               "--core-manifest", str(core_manifest), "--domain-id", "7", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verb"] == "ask" and payload["nanocore_domain"] == 7
    assert len(payload["nanocore_state"]) == 128
    expected = NanocoreAnchorBridge(artifact, core_manifest=manifest).generate_mcp_ask_payload(row, 7, ["proceed", "abort"])
    assert payload == expected


def test_client_refuses_unbound_registered_core(bound_fixture):
    from gen_zero.client import GenZero
    _, artifact, manifest, row = bound_fixture
    client = GenZero()
    client.load_manifold_anchor_artifact(artifact, core_manifest=manifest)
    client.register_nanocore(7, core=client.nanocore_choice_head)
    with pytest.raises(ValueError, match="core space manifest mismatch"):
        client.decide_nanocore(row, 7, ["proceed", "abort"])


def test_decide_nanocore_rejects_unregistered_domain():
    from gen_zero.client import GenZero
    with pytest.raises(ValueError, match=r"micro-core unavailable \(domain 99999\)"):
        GenZero().decide_nanocore(np.zeros(128), 99999, ["stop"])


@pytest.mark.parametrize("missing", [False, True])
def test_decide_nanocore_rejects_empty_or_missing_registry(missing):
    from gen_zero.client import GenZero
    client = GenZero()
    if missing:
        del client.registered_nanocores
    else:
        client.registered_nanocores.clear()
    with pytest.raises(ValueError, match="micro-core unavailable"):
        client.decide_nanocore(np.zeros(128), 0, ["stop"])


@pytest.fixture
def registered_client(bound_fixture, tmp_path):
    """Real untrained ETF head and numerical anchor: lifecycle evidence only."""
    from gen_zero.client import GenZero
    client = GenZero()
    _, artifact, _, row = bound_fixture
    anchor = ManifoldAnchorDistiller.load(artifact)
    anchor.bind_core(client.nanocore_choice_head.core_digest(), 7)
    path = tmp_path / "registered.npz"
    anchor.save(path)
    client.load_manifold_anchor_artifact(path, core_manifest=anchor.space)
    client.register_nanocore(7, space_manifest=anchor.space)
    return client, anchor.space, row


def test_register_and_unregister_nanocore_lifecycle(registered_client):
    client, manifest, row = registered_client
    result = client.decide_nanocore(row, 7, ["proceed", "abort"])
    expected = client.nanocore_choice_head.decide(result["nanocore_state"], ["proceed", "abort"])
    assert result["action_probabilities"] == expected.action_probabilities
    assert result["chosen_action"] == expected.selected_action
    client.unregister_nanocore(7)
    assert 7 not in client.nanocore_space_manifests
    assert 7 not in client.registered_nanocores
    with pytest.raises(ValueError, match="micro-core unavailable"):
        client.decide_nanocore(row, 7, ["stop"])
    client.unregister_nanocore(7)
    with pytest.raises(TypeError, match="callable decide"):
        client.register_nanocore(7, core=object())
    client.register_nanocore(7, space_manifest=manifest)
    assert client.decide_nanocore(row, 7, ["stop"])["gate_status"] == "passed"


def test_replacing_core_clears_manifest_and_rejects_old_binding(registered_client):
    from gen_zero.nanocore.choice_head import ActionETFChoiceHead
    client, manifest, row = registered_client
    replacement = ActionETFChoiceHead(seed=43)
    assert replacement.core_digest() != manifest["core"]
    client.register_nanocore(7, core=replacement)
    assert client.registered_nanocores[7] is replacement
    assert 7 not in client.nanocore_space_manifests
    with pytest.raises(ValueError, match="core space manifest mismatch"):
        client.decide_nanocore(row, 7, ["stop"])
    with pytest.raises(ValueError, match="actual core's digest"):
        client.register_nanocore(7, core=replacement, space_manifest=manifest)
    assert 7 not in client.nanocore_space_manifests
    new_manifest = {**manifest, "core": replacement.core_digest()}
    client.register_nanocore(7, core=replacement, space_manifest=new_manifest)
    assert client.nanocore_space_manifests[7] == new_manifest
    with pytest.raises(ValueError, match="core space manifest mismatch"):
        client.decide_nanocore(row, 7, ["stop"])


def test_invalid_replacement_preserves_existing_registration(registered_client):
    from gen_zero.nanocore.choice_head import ActionETFChoiceHead
    client, manifest, _ = registered_client
    old_core = client.registered_nanocores[7]
    with pytest.raises(ValueError, match="actual core's digest"):
        client.register_nanocore(7, core=ActionETFChoiceHead(seed=43), space_manifest=manifest)
    assert client.registered_nanocores[7] is old_core
    assert client.nanocore_space_manifests[7] == manifest
