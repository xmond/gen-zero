"""Inter-layer hook, episodic manifold memory and MCTS compute pruning on a real Qwen2 forward.

The model is a tiny randomly initialised ``Qwen2ForCausalLM`` (transformers), so every hidden
state comes from the real Qwen2 layer math, but its weights are untrained. The labelled task
is synthetic and built to be learnable by nearest-neighbour retrieval: a sequence's label is
the class of its final trigger token, after random filler tokens. With the base ``model``
fixture the verbalizer readout is near chance, and any gain comes from the memory reading
real hidden states. The early-exit test uses ``_contracting_model(model)`` instead: it adds
a partial label signal and a constant class-0 offset to the trigger embeddings and scales
the residual write-back of layer ``l`` by ``decay ** l``, so the z-scored probe readout and
the memory both contribute (``0 < lambda < 1``). No claim about a trained Qwen2.5 checkpoint
follows from these tests.

torch and transformers are imported unconditionally: a missing dependency is a collection
error, never a skip. Pure inference and non-parametric retrieval, no training, so the
gradient/weight-update checks (M3) do not apply.
"""
from __future__ import annotations

import numpy as np
import json
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from gen_zero.causal import (
    EpisodicManifoldMemory,
    MctsLayerwiseQwenRunner,
    QwenInterLayerHook,
    VerbalizerProbe,
    select_memory_lambda,
)
from gen_zero.causal.qwen_inter_layer_mcts import _LayerEarlyExit


def test_cli_layerwise_mcts_entrypoint(model, tmp_path, monkeypatch, capsys):
    from gen_zero import cli
    from transformers import AutoModelForCausalLM

    # The test fixture is a genuine Qwen2 forward pass with untrained weights.
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *a, **k: model)
    monkeypatch.setenv("GENZERO_LAYERWISE_HF_MODEL", str(tmp_path))
    monkeypatch.setenv("GENZERO_LAYERWISE_INPUT_IDS", "[41,42,43,11]")
    monkeypatch.setenv("GENZERO_LAYERWISE_LABEL_TOKEN_IDS", "[[1],[2],[3]]")
    assert cli.main(["--no-color", "layerwise-mcts", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["total_layers"] == NUM_LAYERS
    assert 1 <= result["layers_executed"] <= NUM_LAYERS
    assert 0 <= result["predicted_index"] < 3


def test_cli_layerwise_mcts_requires_a_real_checkpoint(monkeypatch, capsys):
    # No --hf-model must fail closed: there is no built-in random stand-in model.
    from gen_zero import cli
    from transformers import AutoModelForCausalLM

    def _forbid_load(*a, **k):
        raise AssertionError("model must not load without --hf-model")

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", _forbid_load)
    monkeypatch.delenv("GENZERO_LAYERWISE_HF_MODEL", raising=False)
    assert cli.main(["--no-color", "layerwise-mcts", "--json", "--input-ids", "[41,42,43,11]",
                     "--label-token-ids", "[[1],[2],[3]]"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--hf-model" in captured.err


@pytest.mark.parametrize("payload", [[], [[]], [[41, -1]], [[41, True]], {"rows": [[41]]}])
def test_cli_rejects_bad_candidates_before_model_load(model, payload, tmp_path, monkeypatch, capsys):
    from gen_zero import cli
    from transformers import AutoModelForCausalLM

    def _forbid_load(*a, **k):
        raise AssertionError("model must not load with an invalid --candidates file")

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", _forbid_load)
    cand_file = tmp_path / "candidates.json"
    cand_file.write_text(json.dumps(payload))
    argv = ["--no-color", "layerwise-mcts", "--json", "--hf-model", str(tmp_path),
            "--input-ids", "[41,42,43,11]", "--label-token-ids", "[[1],[2],[3]]",
            "--candidates", str(cand_file)]
    assert cli.main(argv) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ValueError" in captured.err and "--candidates" in captured.err


def test_cli_layerwise_mcts_text_output_without_json(model, tmp_path, monkeypatch, capsys):
    from gen_zero import cli
    from transformers import AutoModelForCausalLM

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *a, **k: model)
    argv = ["--no-color", "layerwise-mcts", "--hf-model", str(tmp_path),
            "--input-ids", "[41,42,43,11]", "--label-token-ids", "[[1],[2],[3]]"]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert out.startswith("[Gen-Zero Layerwise MCTS]")
    assert f"/{NUM_LAYERS} (" in out
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


def test_cli_layerwise_mcts_anchors_fit_probe_stats(model, data, tmp_path, monkeypatch, capsys):
    from gen_zero import cli
    from transformers import AutoModelForCausalLM

    (anchors, labels), _ = data
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *a, **k: model)
    anchor_file = tmp_path / "anchors.json"
    anchor_file.write_text(json.dumps({"input_ids": [a[0].tolist() for a in anchors],
                                       "labels": labels}))
    argv = ["--no-color", "layerwise-mcts", "--json", "--hf-model", str(tmp_path),
            "--input-ids", "[41,42,43,11]", "--label-token-ids", "[[1],[2],[3]]",
            "--anchors", str(anchor_file)]
    assert cli.main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    read_layers = {*result["checkpoints"], NUM_LAYERS - 1}
    assert {int(k) for k in result["probe_stats"]} == read_layers
    assert all(min(v["std"]) > 0 for v in result["probe_stats"].values())


def test_cli_deep_model_requires_anchors(data, tmp_path, monkeypatch, capsys):
    from gen_zero import cli
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    deep = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=16,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        tie_word_embeddings=True,
    )).eval()
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *a, **k: deep)
    argv = ["--no-color", "layerwise-mcts", "--json", "--hf-model", str(tmp_path),
            "--input-ids", "[41,42,43,11]", "--label-token-ids", "[[1],[2],[3]]"]
    assert cli.main(argv) != 0
    assert "--anchors is required for models with >= 16 layers" in capsys.readouterr().err
    anchors, labels = data[0]
    anchor_file = tmp_path / "anchors.json"
    anchor_file.write_text(json.dumps({"input_ids": [a[0].tolist() for a in anchors],
                                       "labels": labels}))
    assert cli.main([*argv, "--anchors", str(anchor_file)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["total_layers"] == 16
    assert result["memory_anchor_count"] == len(anchors)
    assert result["probe_stats"]


@pytest.mark.parametrize("payload", [
    {"input_ids": [], "labels": []},
    {"input_ids": [[41, 42, 43, 11]], "labels": [0]},
    {"input_ids": [[41, 42, 43, 11], [41, 42, 43, 21]], "labels": [0]},
    {"input_ids": [[41, 42, 43, 11], [41, 42, 43, 21], [41, 42, 43, 31]], "labels": [True, 0, 1]},
    {"input_ids": [[True, 42, 43, 11], [41, 42, 43, 21]], "labels": [0, 1]},
])
def test_cli_rejects_empty_or_short_anchors_file(model, payload, tmp_path, monkeypatch, capsys):
    from gen_zero import cli
    from transformers import AutoModelForCausalLM

    load_calls = 0

    def _forbid_load(*a, **k):
        nonlocal load_calls
        load_calls += 1
        return model

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", _forbid_load)
    anchor_file = tmp_path / "anchors.json"
    anchor_file.write_text(json.dumps(payload))
    base = ["--no-color", "layerwise-mcts", "--json", "--hf-model", str(tmp_path),
            "--input-ids", "[41,42,43,11]", "--label-token-ids", "[[1],[2],[3]]",
            "--anchors", str(anchor_file)]
    for extra in ([], ["--mcts-layer-weights"], ["--ffn-keep-ratio", "0.5"]):
        assert cli.main([*base, *extra]) != 0
        assert load_calls == 0
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "ValueError" in captured.err and "--anchors" in captured.err


def test_cli_layerwise_mcts_weights_and_ffn_pruning_order(model, data, tmp_path, monkeypatch, capsys):
    from gen_zero import cli
    from gen_zero.causal import QwenSparseFfnHook
    from transformers import AutoModelForCausalLM

    (anchors, labels), _ = data
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *a, **k: model)
    anchor_file = tmp_path / "anchors.json"
    anchor_file.write_text(json.dumps({"input_ids": [a[0].tolist() for a in anchors],
                                       "labels": labels}))
    argv = ["--no-color", "layerwise-mcts", "--json", "--hf-model", str(tmp_path),
            "--input-ids", "[41,42,43,11]", "--label-token-ids", "[[1],[2],[3]]",
            "--anchors", str(anchor_file), "--mcts-layer-weights", "--ffn-keep-ratio", "0.5"]
    assert cli.main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ffn_channels_executed"] < result["ffn_dense_channels"]
    assert result["layer_weights"]

    # Reference: prune first, then read the probe on the pruned forward pass directly.
    probe = VerbalizerProbe(model, LABEL_TOKENS)
    ref = MctsLayerwiseQwenRunner(model, probe, checkpoints=result["checkpoints"])
    masks = ref.fit_ffn_pruning(anchors, list(range(NUM_LAYERS)), keep_ratio=0.5)
    layers = sorted(int(k) for k in result["probe_stats"])
    assert layers == sorted({*result["checkpoints"], NUM_LAYERS - 1})

    def stats(sparse: bool):
        states = {layer: [] for layer in layers}
        for ids in anchors:
            hook = QwenInterLayerHook(model, capture_layers=layers)
            with torch.no_grad(), QwenSparseFfnHook(model, masks if sparse else {}), hook:
                model.model(input_ids=ids, use_cache=False)
            for layer in layers:
                states[layer].append(hook.captured[layer].cpu().numpy())
        out = {}
        for layer in layers:
            z = probe(torch.as_tensor(np.concatenate(states[layer]), dtype=torch.float32))
            out[layer] = (z.mean(axis=0), z.std(axis=0))
        return out

    pruned, dense = stats(sparse=True), stats(sparse=False)
    for layer in layers:
        got = result["probe_stats"][str(layer)]
        np.testing.assert_allclose(got["mean"], pruned[layer][0], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(got["std"], pruned[layer][1], rtol=1e-6, atol=1e-7)
    # The check discriminates: the dense forward gives different stats at the last layer.
    assert not np.allclose(result["probe_stats"][str(NUM_LAYERS - 1)]["mean"],
                           dense[NUM_LAYERS - 1][0], rtol=1e-6, atol=1e-7)

NUM_LAYERS = 6
TRIGGERS = ((10, 11, 12), (20, 21, 22), (30, 31, 32))
LABEL_TOKENS = ((1,), (2,), (3,))
FILLER = np.arange(40, 64)
SEQ_FILLER = 5


def _model() -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    return Qwen2ForCausalLM(cfg).eval()


def _dataset(n_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for label, triggers in enumerate(TRIGGERS):
        for _ in range(n_per_class):
            ids = list(rng.choice(FILLER, SEQ_FILLER)) + [int(rng.choice(triggers))]
            rows.append(torch.tensor([ids], dtype=torch.long))
            labels.append(label)
    return rows, labels


@pytest.fixture(scope="module")
def model():
    return _model()


@pytest.fixture(scope="module")
def data():
    return _dataset(12, seed=1), _dataset(10, seed=2)


def _fitted_runner(model, anchors, labels, tau, checkpoints=(1, 2), beta=12.0):
    probe = VerbalizerProbe(model, LABEL_TOKENS)
    runner = MctsLayerwiseQwenRunner(model, probe, checkpoints=checkpoints, tau=tau, gamma=0.5)
    report = runner.fit_memories(anchors, labels, layers=[*checkpoints, NUM_LAYERS - 1], beta=beta)
    return runner, report


# -- hook ---------------------------------------------------------------------------------

def test_hook_identity_is_bit_exact_when_action_or_eta_is_zero(model):
    ids = torch.tensor([[41, 42, 43, 11]])
    with torch.no_grad():
        ref = model(ids, use_cache=False).logits
    basis = np.random.default_rng(0).normal(size=(2, 64))
    for action, eta in ((0, 0.7), (1, 0.0)):
        hook = QwenInterLayerHook(model, steering_basis=basis, steering_layers=[1, 3],
                                  eta=eta, action=action)
        with torch.no_grad(), hook:
            out = model(ids, use_cache=False).logits
        assert torch.equal(out, ref), (action, eta)
        assert hook.layers_executed == NUM_LAYERS
        assert hook.steering_delta(1, out) is None
    steered = QwenInterLayerHook(model, steering_basis=basis, steering_layers=[1], eta=0.5, action=1)
    with torch.no_grad(), steered:
        moved = model(ids, use_cache=False).logits
    assert not torch.allclose(moved, ref)


def test_hook_kinetic_delta_matches_definition(model):
    ids = torch.tensor([[41, 42, 43, 21]])
    hook = QwenInterLayerHook(model, capture_layers=range(NUM_LAYERS))
    with torch.no_grad(), hook:
        out = model(ids, use_cache=False, output_hidden_states=True)
    hs = out.hidden_states  # embeddings + one per layer (the last one is post-norm)
    for layer in range(NUM_LAYERS - 1):
        h, prev = hs[layer + 1][0, -1], hs[layer][0, -1]
        assert torch.allclose(hook.captured[layer][0], h, atol=1e-6)
        expected = float((h - prev).norm() / (h.norm() + 1e-6))
        assert hook.deltas[layer][0] == pytest.approx(expected, rel=1e-5)
        assert np.isfinite(hook.deltas[layer][0]) and hook.deltas[layer][0] > 0


def test_hook_exit_skips_later_layers(model):
    ids = torch.tensor([[41, 42, 43, 31]])
    hook = QwenInterLayerHook(model, exit_policy=lambda *_: True, exit_layers=[2])
    with torch.no_grad(), hook, pytest.raises(_LayerEarlyExit) as stop:
        model.model(input_ids=ids, use_cache=False)
    assert stop.value.layer_idx == 2
    assert hook.layer_calls == [1, 1, 1, 0, 0, 0]
    assert hook.layers_executed == 3


# -- memory -------------------------------------------------------------------------------

def test_memory_is_cosine_scale_invariant_and_loo_masks_self():
    rng = np.random.default_rng(3)
    keys = rng.normal(size=(20, 16))
    labels = rng.integers(0, 3, size=20)
    memory = EpisodicManifoldMemory(16, 3, beta=12.0)
    memory.write(keys, labels, ids=list(range(20)))
    q = rng.normal(size=(4, 16))
    np.testing.assert_allclose(memory.query(q), memory.query(3.0 * q), atol=1e-12)
    # Without masking, an entry finds itself at cosine 1 and dominates its own read.
    own = memory.query(keys[:1])[0]
    loo = memory.loo_query()[0]
    assert own[labels[0]] > loo[labels[0]]
    np.testing.assert_allclose(loo, memory.query(keys[:1], exclude_ids=[0])[0])
    with pytest.raises(ValueError):
        EpisodicManifoldMemory(16, 3).query(q)


@pytest.mark.parametrize("targets", [
    [[0.0, 0.0, 0.0]],
    [[np.nan, 1.0, 0.0]],
    [[np.inf, 1.0, 0.0]],
    [[-np.inf, 1.0, 0.0]],
    [[0.5, -0.1, 0.6]],
    [[1e308, 1e308, 1e308]],
])
def test_memory_write_rejects_degenerate_target_rows(targets):
    memory = EpisodicManifoldMemory(4, 3)
    with pytest.raises(ValueError):
        memory.write(np.ones((1, 4)), np.asarray(targets), ids=[0])
    assert len(memory) == 0


@pytest.mark.parametrize("hidden", [np.zeros((1, 4)), np.full((1, 4), np.nan),
                                    np.full((1, 4), np.inf), np.full((1, 4), 1e300)])
def test_memory_write_rejects_zero_or_non_finite_keys(hidden):
    memory = EpisodicManifoldMemory(4, 3, projection=np.full((4, 2), 1e10))
    with pytest.raises(ValueError):
        memory.write(hidden, np.array([0]), ids=[0])
    assert len(memory) == 0


def test_lambda_gate_rejects_a_prior_only_memory_against_a_saturated_probe():
    labels = np.array([0, 1] * 10)
    n = len(labels)
    # A saturated, wrong-half probe: every row says label 0 with probability ~1.
    base = np.tile([1 - 1e-9, 1e-9], (n, 1))
    # Identical keys: every cosine is 1, so the LOO read is exactly the LOO class prior.
    memory = EpisodicManifoldMemory(4, 2)
    memory.write(np.ones((n, 4)), labels, ids=list(range(n)))
    onehot = np.eye(2)[labels]
    loo_prior = (onehot.sum(axis=0)[None, :] - onehot) / (n - 1)
    np.testing.assert_allclose(memory.loo_query(), loo_prior, atol=1e-12)
    lam_plain, _ = select_memory_lambda(base, memory.loo_query(), labels)
    assert lam_plain > 0  # the old test alone is fooled by the saturated base
    lam_null, report = select_memory_lambda(base, memory.loo_query(), labels, null_probs=loo_prior)
    assert lam_null == 0.0, report


def test_memory_center_is_applied_to_writes_and_queries():
    rng = np.random.default_rng(9)
    offset = np.zeros(8)
    offset[0] = 1e3  # one outlier dimension, as in real decoder states
    keys = rng.normal(size=(12, 8)) + offset
    raw = EpisodicManifoldMemory(8, 2)
    centred = EpisodicManifoldMemory(8, 2, center=keys.mean(axis=0))
    assert np.min(raw.encode(keys) @ raw.encode(keys).T) > 0.99
    assert np.min(centred.encode(keys) @ centred.encode(keys).T) < 0.5
    with pytest.raises(ValueError):
        EpisodicManifoldMemory(8, 2, center=np.full(8, np.nan))


def test_lambda_gate_rejects_label_shuffled_memory(model, data):
    (anchors, labels), _ = data
    runner, report = _fitted_runner(model, anchors, labels, tau=1.0)
    lam = {layer: runner.memories[layer][1] for layer in runner.memories}
    assert all(v > 0 for v in lam.values()), report
    shuffled = np.random.default_rng(7).permutation(labels)
    states = []
    for ids in anchors:
        hook = QwenInterLayerHook(model, capture_layers=[2])
        with torch.no_grad(), hook:
            model.model(input_ids=ids, use_cache=False)
        states.append(hook.captured[2])
    h = torch.cat(states)
    logits = runner.probe(h)
    base = np.exp(logits - logits.max(axis=1, keepdims=True))
    base = base / base.sum(axis=1, keepdims=True)
    memory = EpisodicManifoldMemory(h.shape[1], 3)
    memory.write(h.numpy(), shuffled, ids=list(range(len(labels))))
    lam_shuffled, rep = select_memory_lambda(base, memory.loo_query(), labels)
    assert lam_shuffled == 0.0, rep


# -- per-layer probe logit calibration ----------------------------------------------------

def _last_residual(model, ids, layer):
    hook = QwenInterLayerHook(model, capture_layers=[layer])
    with torch.no_grad(), hook:
        model.model(input_ids=ids, use_cache=False)
    return hook.captured[layer]


def test_probe_stats_absent_keeps_raw_readout_and_present_standardises(model, data):
    (anchors, _), _ = data
    probe = VerbalizerProbe(model, LABEL_TOKENS)
    h = _last_residual(model, anchors[0], 2)
    z = probe(h)
    raw = MctsLayerwiseQwenRunner(model, probe, checkpoints=(1, 2))
    expected = np.exp(z - z.max(axis=1, keepdims=True))
    expected = expected / expected.sum(axis=1, keepdims=True)
    assert np.array_equal(raw.fused_probs(2, h), expected)
    mean, std = np.array([[0.5, -1.0, 2.0]]), np.array([[2.0, 0.5, 4.0]])
    cal = MctsLayerwiseQwenRunner(model, probe, checkpoints=(1, 2), probe_stats={2: (mean, std)})
    zt = (z - mean) / std
    expected = np.exp(zt - zt.max(axis=1, keepdims=True))
    np.testing.assert_allclose(cal.fused_probs(2, h), expected / expected.sum(axis=1, keepdims=True))
    # Stats exist but not for this layer: fail closed instead of mixing raw and calibrated.
    with pytest.raises(ValueError, match="has none"):
        cal.fused_probs(1, h)


@pytest.mark.parametrize("mean,std", [
    (np.zeros((1, 2)), np.ones((1, 2))),           # wrong K
    (np.zeros(3), np.ones(3)),                     # not (1, K)
    (np.zeros((1, 3)), np.array([[1.0, 0.0, 1.0]])),
    (np.array([[0.0, np.nan, 0.0]]), np.ones((1, 3))),
])
def test_probe_stats_rejects_bad_shapes_and_values(model, mean, std):
    probe = VerbalizerProbe(model, LABEL_TOKENS)
    with pytest.raises(ValueError):
        MctsLayerwiseQwenRunner(model, probe, checkpoints=(1, 2), probe_stats={2: (mean, std)})


def test_fit_memories_sets_anchor_only_probe_stats(model, data):
    (anchors, labels), (test_rows, _) = data
    layers = (1, 2, NUM_LAYERS - 1)
    runner, report = _fitted_runner(model, anchors, labels, tau=1.0)
    assert sorted(runner.probe_stats) == sorted(layers)
    for layer in layers:
        z = np.concatenate([runner.probe(_last_residual(model, ids, layer)) for ids in anchors])
        mean, std = runner.probe_stats[layer]
        assert mean.shape == std.shape == (1, 3)
        np.testing.assert_allclose(mean, z.mean(axis=0, keepdims=True), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(std, z.std(axis=0, keepdims=True), rtol=1e-6, atol=1e-6)
        assert report[layer]["probe_logit_mean"] == mean[0].tolist()
    before = {k: (m.copy(), s.copy()) for k, (m, s) in runner.probe_stats.items()}
    for ids in test_rows[:6]:
        runner.predict(ids)
    for k, (m, s) in runner.probe_stats.items():
        assert np.array_equal(m, before[k][0]) and np.array_equal(s, before[k][1])


def test_leave_one_out_logit_stats_match_brute_force():
    from gen_zero.causal.qwen_inter_layer_mcts import _loo_mean, _loo_std
    z = np.random.default_rng(3).normal(size=(9, 3)) * [1.0, 5.0, 0.2] + [4.0, -2.0, 0.0]
    for i in range(len(z)):
        rest = np.delete(z, i, axis=0)
        np.testing.assert_allclose(_loo_mean(z)[i], rest.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(_loo_std(z)[i], rest.std(axis=0), atol=1e-9)


@pytest.mark.parametrize("z", [
    np.array([[1.0, 2.0], [1.0, 3.0]]),
    # A constant 0.1 column: the float mean is off by 1e-17, so std is 1.4e-17, not 0.
    np.array([[0.1, 2.0], [0.1, 3.0], [0.1, 4.0]]),
    np.array([[np.inf, 2.0], [1.0, 3.0], [2.0, 4.0]]),
    np.array([[np.nan, 2.0], [1.0, 3.0], [2.0, 4.0]]),
])
def test_anchor_logit_stats_reject_zero_variance(z):
    from gen_zero.causal.qwen_inter_layer_mcts import _anchor_logit_stats
    with pytest.raises(ValueError, match="degenerate anchor logits with zero variance"):
        _anchor_logit_stats(z)


@pytest.mark.parametrize("z", [
    np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]),  # constant column: every LOO var is 0
    np.array([[0.0, 2.0], [5.0, 3.0], [5.0, 4.0]]),  # dropping row 0 leaves a constant pair
    # Dropping row 0 leaves [1.1, 1.1]: second - mu ** 2 cancels to 2.2e-16, not 0.
    np.array([[0.3, 2.0], [1.1, 3.0], [1.1, 4.0]]),
    np.array([[np.inf, 2.0], [1.0, 3.0], [2.0, 4.0]]),
])
def test_loo_std_rejects_zero_or_non_finite_variance(z):
    from gen_zero.causal.qwen_inter_layer_mcts import _loo_std
    with pytest.raises(ValueError, match="degenerate leave-one-out anchor logits with zero variance"):
        _loo_std(z)


def test_fit_memories_loo_selection_excludes_own_outlier(model, data, monkeypatch):
    import gen_zero.causal.qwen_inter_layer_mcts as module

    anchors, labels = data[0]
    probe = VerbalizerProbe(model, LABEL_TOKENS)
    captured = []
    logits = []
    original_call = VerbalizerProbe.__call__
    calls = 0
    def injected(self, h):
        nonlocal calls
        z = original_call(self, h)
        calls += 1
        if calls == 1:
            z = z.copy()
            z[0, 0] += 1000.0
            logits.append(z.copy())
        return z
    monkeypatch.setattr(VerbalizerProbe, "__call__", injected)
    original_select = module.select_memory_lambda
    def record(base, *args, **kwargs):
        captured.append(base.copy())
        return original_select(base, *args, **kwargs)
    monkeypatch.setattr(module, "select_memory_lambda", record)
    runner = MctsLayerwiseQwenRunner(model, probe, checkpoints=(2,))
    runner.fit_memories(anchors, labels, layers=[2])
    assert captured
    z = logits[0]
    from gen_zero.causal.qwen_inter_layer_mcts import _loo_mean, _loo_std, _softmax
    expected = _softmax((z - _loo_mean(z)) / _loo_std(z), axis=1)
    np.testing.assert_allclose(captured[0], expected, atol=1e-6)
    in_sample = _softmax((z - z.mean(axis=0, keepdims=True)) /
                         z.std(axis=0, keepdims=True), axis=1)
    assert np.max(np.abs(captured[0] - in_sample)) > 0.01


# -- MCTS early exit ----------------------------------------------------------------------

def _evaluate(runner, rows, labels):
    decisions = [runner.predict(ids) for ids in rows]
    acc = float(np.mean([d.label == y for d, y in zip(decisions, labels)]))
    return decisions, acc


def _contracting_model(model, decay=0.8, signal=0.5, offset=1.0, label_scale=10.0):
    """A copy of ``model`` with shrinking residual updates and a partial label signal.

    Every decoder layer stays a real, non-zero Qwen2 block. Changes, in order:

    * The trigger embeddings get ``signal`` times their class's label row direction, so the
      probe has a real but partial readout, and all of them get ``offset`` times label 0's
      direction: a constant raw bias that per-layer z-scoring removes.
    * The third trigger of each class shares one token-specific vector. Those rows differ
      only along the label directions, so the cosine memory confuses them and the probe
      separates them. This keeps the memory lambda strictly between 0 and 1.
    * The label rows of the tied head are scaled by ``label_scale``. A z-scored readout does
      not change; a raw readout becomes confidently biased instead of near uniform.
    * The residual write-back of layer ``l`` (``o_proj`` and ``down_proj``) is scaled by
      ``decay ** l``: the readout still moves at layers 2 to 3, and less after that.
    """
    import copy

    model = copy.deepcopy(model)
    with torch.no_grad():
        emb = model.model.embed_tokens.weight
        label_dirs = emb[[1, 2, 3]].clone()
        label_dirs = label_dirs / label_dirs.norm(dim=1, keepdim=True)
        scale = float(emb[min(TRIGGERS[0]):max(TRIGGERS[-1]) + 1].norm(dim=1).mean())
        shared = emb[TRIGGERS[0][2]].clone()
        for triggers in TRIGGERS:
            emb[triggers[2]] = shared
        for label, triggers in enumerate(TRIGGERS):
            for t in triggers:
                emb[t] += scale * (signal * label_dirs[label] + offset * label_dirs[0])
        emb[[1, 2, 3]] *= label_scale
        for idx, layer in enumerate(model.model.layers):
            layer.self_attn.o_proj.weight.mul_(decay ** idx)
            layer.mlp.down_proj.weight.mul_(decay ** idx)
    return model


def test_layerwise_mcts_early_exit_cuts_layers_and_beats_baseline(model, data):
    model = _contracting_model(model)
    (anchors, labels), (test_rows, test_labels) = data
    n = len(test_rows)
    probe = VerbalizerProbe(model, LABEL_TOKENS)
    checkpoints = (2, 3, 4)
    baseline = MctsLayerwiseQwenRunner(model, probe, checkpoints=(2,), tau=1e9)
    base_dec, base_acc = _evaluate(baseline, test_rows, test_labels)
    assert all(not d.early_exit for d in base_dec)
    assert sum(d.layers_executed for d in base_dec) == n * NUM_LAYERS

    # Measured sweep (dev, 2026-10-18, decay=0.8, n=30): beta 7 sets lambda_5 = 0 and fails.
    # beta 7.5 / 8: acc 29/30, min lambda 0.2 / 0.3. beta 8.5: acc 27/30, agree 29/30.
    # beta 9: acc 27/30 (26/30 still passes, 25/30 fails), agree 30/30, min lambda 0.5. Kept
    # because beta 9 also passes at decay 0.85, where beta 8 sets lambda_5 = 0. beta 9 fails at
    # decay 0.75 (no held row); beta 8 passes there.
    runner, report = _fitted_runner(model, anchors, labels, tau=1.0, checkpoints=checkpoints,
                                    beta=9.0)
    assert runner.min_exit_layer == NUM_LAYERS // 2
    lam = {layer: runner.memories[layer][1] for layer in runner.memories}
    print(f"\nmemory_lambda={lam}")
    # 0 < lambda < 1 at every read layer: the calibrated probe and the memory both enter
    # fused_probs, so probe_stats matters for the exit and for the full-depth target.
    assert sorted(lam) == [*checkpoints, NUM_LAYERS - 1]
    assert all(0 < v < 1 for v in lam.values()), report
    full_depth = MctsLayerwiseQwenRunner(model, probe, checkpoints=(2,), tau=1e9,
                                         memories=runner.memories, probe_stats=runner.probe_stats)
    full_dec, full_acc = _evaluate(full_depth, test_rows, test_labels)
    dec, acc = _evaluate(runner, test_rows, test_labels)
    executed = sum(d.layers_executed for d in dec)
    exited = [d for d in dec if d.early_exit]
    print(f"layerwise: n={n} chance=1/3 baseline_acc={base_acc:.3f} memory_mcts_acc={acc:.3f} "
          f"layers={executed}/{n * NUM_LAYERS} exits={len(exited)} "
          f"exit_layers={sorted(d.exit_layer for d in exited)} "
          f"paired_wins={sum(d.label == y and b.label != y for d, b, y in zip(dec, base_dec, test_labels))} "
          f"paired_losses={sum(d.label != y and b.label == y for d, b, y in zip(dec, base_dec, test_labels))}")
    assert exited, "no sample exited early"
    assert executed < n * NUM_LAYERS
    agree = sum(d.label == f.label for d, f in zip(dec, full_dec))
    print(f"agreement_with_full_depth_memory={agree}/{n} full_depth_memory_acc={full_acc:.3f}")
    for d in exited:
        # Same-unit certificate: probability gap against the simplex tail bound, never
        # at the first checkpoint and never before min_exit_layer. A zero drift would make
        # the bound 0 and the gap test vacuous, so it must be positive here.
        assert d.exit_layer >= runner.min_exit_layer and d.exit_layer != checkpoints[0]
        assert d.drift > 0 and d.threshold > 0
        assert d.margin > d.threshold
        assert d.threshold == pytest.approx(2 * d.gamma_hat / (1 - d.gamma_hat) * d.drift)
        assert 0.5 <= d.gamma_hat <= 0.95
        assert d.mcts_margin > runner.mcts_margin_min
        assert d.label == int(np.argmax(d.probs))
        assert d.layers_executed == d.exit_layer + 1
        assert all(c == 0 for c in d.layer_calls[d.exit_layer + 1:])
        assert sum(d.layer_calls) == d.layers_executed
    # Rows the gap test holds back at layer 3: top-1 stable from layer 2, gap <= bound.
    # Recomputed from the layer 2 and 3 residuals, independent of predict().
    held = []
    for i, ids in enumerate(test_rows):
        hook = QwenInterLayerHook(model, capture_layers=[2, 3])
        with torch.no_grad(), hook:
            model.model(input_ids=ids, use_cache=False)
        p2 = runner.fused_probs(2, hook.captured[2])[0]
        p3 = runner.fused_probs(3, hook.captured[3])[0]
        drift = float(np.max(np.abs(p3 - p2)))
        assert drift > 0
        bound = runner.tail_bound(drift, runner.contraction(drift, None))
        top2 = np.sort(p3)[-2:]
        if np.argmax(p3) == np.argmax(p2) and top2[1] - top2[0] <= bound:
            held.append(i)
    print(f"held_at_layer3_by_gap={held} exit_layers_of_held="
          f"{[dec[i].exit_layer if dec[i].early_exit else None for i in held]}")
    assert held, "no row is held back by gap <= bound at layer 3"
    assert all(not dec[i].early_exit or dec[i].exit_layer > 3 for i in held)
    assert any(not d.early_exit or d.exit_layer > 3 for d in dec)
    assert agree >= n - 2
    assert acc >= 0.85
    assert acc > base_acc


def test_certificate_needs_two_checkpoints_and_min_exit_layer(model, data):
    (anchors, labels), (test_rows, _) = data
    # One checkpoint: there is no drift, so no certificate and no exit.
    single, _ = _fitted_runner(model, anchors, labels, tau=0.0, checkpoints=(3,))
    assert not any(single.predict(ids).early_exit for ids in test_rows[:6])
    # tau = 0 makes the bound 0; exits still wait for min_exit_layer.
    late, _ = _fitted_runner(model, anchors, labels, tau=0.0, checkpoints=(1, 2, 3, 4))
    late.min_exit_layer = 4
    for ids in test_rows[:6]:
        d = late.predict(ids)
        assert not d.early_exit or d.exit_layer == 4
    with pytest.raises(ValueError):
        MctsLayerwiseQwenRunner(model, late.probe, checkpoints=(2, 3), min_exit_layer=NUM_LAYERS)
    with pytest.raises(ValueError):
        MctsLayerwiseQwenRunner(model, late.probe, checkpoints=(2, 3), gamma=0.95)


# -- MCTS multi-candidate pruning ---------------------------------------------------------

def test_candidate_pruning_skips_deep_layers_and_matches_full_forward(model, data):
    (anchors, labels), _ = data
    runner, _ = _fitted_runner(model, anchors, labels, tau=1.0, checkpoints=(1, 2))
    shallow, k, m = 3, 6, 2
    rng = np.random.default_rng(11)
    queries, wins_pruned, wins_base = 12, 0, 0
    probe_only = MctsLayerwiseQwenRunner(model, runner.probe, checkpoints=(1,), tau=1e9)
    for qi in range(queries):
        target = qi % 3
        good = int(rng.integers(k))
        cands = []
        for c in range(k):
            cls = target if c == good else (target + 1 + c % 2) % 3
            ids = list(rng.choice(FILLER, SEQ_FILLER)) + [int(rng.choice(TRIGGERS[cls]))]
            cands.append(torch.tensor([ids], dtype=torch.long))
        res = runner.score_candidates_mcts_pruned(cands, target, shallow_layers=shallow, top_m=m)
        assert res.layers_executed == k * shallow + m * (NUM_LAYERS - shallow)
        assert res.layers_executed < res.full_layers == k * NUM_LAYERS
        assert len(res.survivors) == m and res.best in res.survivors
        wins_pruned += res.best == good
        base_scores = [probe_only.predict(c).probs[target] for c in cands]
        wins_base += int(np.argmax(base_scores)) == good
    print(f"\npruning: queries={queries} K={k} m={m} chance=1/{k} "
          f"baseline_hits={wins_base} pruned_memory_hits={wins_pruned} "
          f"layers={k * shallow + m * (NUM_LAYERS - shallow)}/{k * NUM_LAYERS} per query")
    assert wins_pruned > wins_base

    ids = torch.tensor([[44, 45, 46, 47, 48, 22]])
    # The same shallow-cut and replay path that score_candidates_mcts_pruned runs.
    stop, shallow_run = runner._shallow_pass(ids, shallow - 1)
    resumed, deep_run = runner._resume_pass(ids, stop.hidden_state, shallow)
    executed = shallow_run + deep_run
    full = QwenInterLayerHook(model, capture_layers=[NUM_LAYERS - 1])
    with torch.no_grad(), full:
        model.model(input_ids=ids, use_cache=False)
    assert executed == NUM_LAYERS
    assert torch.equal(resumed, full.captured[NUM_LAYERS - 1])

# -- sparse FFN and layer-weight regression checks ----------------------------------------

def test_sparse_ffn_physical_slice_matches_reference(model):
    from gen_zero.causal.qwen_inter_layer_mcts import QwenSparseFfnHook
    ids = torch.tensor([[41, 42, 43, 11]])
    mlp = model.model.layers[0].mlp
    selected = np.arange(0, mlp.gate_proj.weight.shape[0], 2)
    seen = []
    def capture(module, inputs):
        seen.append(inputs[0].detach())
    handle = mlp.register_forward_pre_hook(capture)
    with torch.no_grad(), QwenSparseFfnHook(model, {0: selected}) as sparse:
        model(ids, use_cache=False)
    handle.remove()
    x = seen[0]
    import torch.nn.functional as F
    expected = F.linear(F.silu(F.linear(x, mlp.gate_proj.weight[selected])) *
                        F.linear(x, mlp.up_proj.weight[selected]), mlp.down_proj.weight[:, selected])
    with torch.no_grad(), QwenSparseFfnHook(model, {0: selected}):
        actual = mlp(x)
    torch.testing.assert_close(actual, expected)
    assert sparse.ffn_layer_calls == [1] * NUM_LAYERS
    assert sparse.ffn_channels_executed == 5 * mlp.gate_proj.weight.shape[0] + len(selected)
    assert sparse.ffn_dense_channels == NUM_LAYERS * mlp.gate_proj.weight.shape[0]
    assert sparse.ffn_macs_executed < sparse.ffn_dense_macs


def test_sparse_ffn_restores_forward_after_exception(model):
    from gen_zero.causal.qwen_inter_layer_mcts import QwenSparseFfnHook
    mlp = model.model.layers[0].mlp
    original = mlp.forward.__func__
    with pytest.raises(RuntimeError, match="break"):
        with QwenSparseFfnHook(model, {0: [0, 1]}):
            raise RuntimeError("break")
    assert mlp.forward.__func__ is original


def test_sparse_ffn_rejects_bad_masks(model):
    from gen_zero.causal.qwen_inter_layer_mcts import QwenSparseFfnHook
    for mask in ([], [1, 1], [2, 1], [-1], [128]):
        with pytest.raises(ValueError):
            QwenSparseFfnHook(model, {0: mask})
    with pytest.raises(ValueError):
        QwenSparseFfnHook(model, {NUM_LAYERS: [0]})


def test_score_ffn_channels_is_finite_and_selects_top(model):
    from gen_zero.causal.qwen_inter_layer_mcts import score_ffn_channels, select_top_ffn_channels
    anchors = [torch.tensor([[41, 42, 43, 11]]), torch.tensor([[41, 42, 43, 21]])]
    scores = score_ffn_channels(model, anchors, [0, 2])
    assert set(scores) == {0, 2}
    assert scores[0].dtype == np.float64 and scores[0].shape == (128,)
    assert np.all(np.isfinite(scores[0]))
    chosen = select_top_ffn_channels(scores, 0.5)
    assert len(chosen[0]) == 64 and np.all(np.diff(chosen[0]) > 0)
    np.testing.assert_array_equal(chosen[0], np.sort(np.argsort(scores[0])[-64:]))


def test_score_ffn_channels_rejects_empty_anchors(model):
    from gen_zero.causal.qwen_inter_layer_mcts import score_ffn_channels
    with pytest.raises(ValueError):
        score_ffn_channels(model, [], [0])


def test_fit_layer_weights_normalizes_and_keeps_top_k(model):
    anchors, labels = _dataset(3, seed=5)
    runner = MctsLayerwiseQwenRunner(model, VerbalizerProbe(model, LABEL_TOKENS), [1, 2])
    runner.fit_memories(anchors, labels, [1, 2, NUM_LAYERS - 1], fit_mcts_layer_weights=True,
                        top_k_hooks=2)
    assert sum(runner.layer_weights.values()) == pytest.approx(1.0)
    assert 1 <= sum(weight > 0 for weight in runner.layer_weights.values()) <= 2
    decision = runner.predict(anchors[0])
    assert decision.active_hook_weights == runner.layer_weights


def test_fit_ffn_pruning_changes_predict_compute(model):
    runner = MctsLayerwiseQwenRunner(model, VerbalizerProbe(model, LABEL_TOKENS), [1, 2])
    ids = torch.tensor([[41, 42, 43, 11]])
    masks = runner.fit_ffn_pruning([ids], [0, 1], 0.5)
    assert len(masks[0]) == 64
    decision = runner.predict(ids)
    assert decision.ffn_channels_executed < decision.ffn_dense_channels
    assert decision.ffn_macs_executed < decision.ffn_dense_macs


def test_sparse_hook_reused_across_predict(model):
    runner = MctsLayerwiseQwenRunner(model, VerbalizerProbe(model, LABEL_TOKENS), [1, 2],
                                     ffn_channel_masks={0: np.arange(64)})
    ids = torch.tensor([[41, 42, 43, 11]])
    runner.predict(ids)
    sparse = runner._last_sparse_hook
    slices = sparse._slices[0]
    runner.predict(ids)
    assert runner._last_sparse_hook is sparse
    assert runner._last_sparse_hook._slices[0] is slices
    runner.ffn_channel_masks[0] = np.arange(32)
    runner.predict(ids)
    assert runner._last_sparse_hook is not sparse


def test_verbalizer_probe_moves_residual_to_norm_device(model):
    probe = VerbalizerProbe(model, LABEL_TOKENS)
    residual = torch.randn(1, model.config.hidden_size)
    expected = probe(residual)
    if torch.cuda.is_available():
        model = model.to("cuda")
        probe = VerbalizerProbe(model, LABEL_TOKENS)
        np.testing.assert_allclose(probe(residual), expected, rtol=1e-4, atol=1e-4)
    else:
        pytest.skip("CUDA required to verify cross-device transfer")
