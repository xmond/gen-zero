"""Numerical and input-contract tests for continuous ZeroTrunk decoding."""
import pytest
import torch

from gen_zero.causal.zero_trunk import TrunkConfig, ZeroTrunk


@pytest.fixture(params=["fp32", "bf16"])
def trunk(request):
    torch.manual_seed(17)
    cfg = TrunkConfig(dict(model_type="qwen2", hidden_size=16, intermediate_size=32,
                           num_hidden_layers=2, num_attention_heads=4,
                           num_key_value_heads=2, rms_norm_eps=1e-6,
                           rope_theta=10000, vocab_size=32))
    model = ZeroTrunk(cfg, request.param)
    for name, buffer in model.named_buffers():
        if name == "inv_freq":
            continue
        if "layernorm.weight" in name or name == "norm.weight":
            buffer.fill_(1)
        elif name == "embed_tokens.weight":
            buffer.copy_(torch.randn_like(buffer.float()).to(buffer.dtype) * .1)
        else:
            buffer.copy_(torch.randn_like(buffer.float()).to(buffer.dtype) * .05)
    return model


def assert_cache_close(left, right, tolerance):
    assert len(left) == len(right)
    for (lk, lv), (rk, rv) in zip(left, right):
        torch.testing.assert_close(lk, rk, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(lv, rv, atol=tolerance, rtol=tolerance)


def test_embedding_entry_bit_exact(trunk):
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    embeds = trunk.embed_tokens(ids)
    with torch.inference_mode():
        by_ids, cache_ids = trunk(ids, return_cache=True)
        by_embeds, cache_embeds = trunk(inputs_embeds=embeds, return_cache=True)
    assert torch.equal(by_ids, by_embeds)
    assert_cache_close(cache_ids, cache_embeds, 0)


def test_single_step_matches_full_sequence(trunk):
    ids = torch.tensor([[1, 2, 3]])
    extra = torch.randn(1, 1, trunk.cfg.hidden_size).to(trunk.act_dtype) * .1
    with torch.inference_mode():
        _, prefix_cache = trunk(ids, return_cache=True)
        stepped, stepped_cache, mask = trunk.forward_latent_step(extra, prefix_cache, torch.ones_like(ids))
        full, full_cache = trunk(inputs_embeds=torch.cat((trunk.embed_tokens(ids), extra), 1), return_cache=True)
    tol = 2e-5 if trunk.precision == "fp32" else .03
    torch.testing.assert_close(stepped, full[:, -1:], atol=tol, rtol=tol)
    assert_cache_close(stepped_cache, full_cache, tol)
    assert mask.tolist() == [[1, 1, 1, 1]]


@pytest.mark.parametrize("steps", range(1, 9))
def test_continuous_rollout(trunk, steps):
    seed = torch.randn(2, 2, trunk.cfg.hidden_size).to(trunk.act_dtype) * .1
    with torch.inference_mode():
        hidden, cache = trunk(inputs_embeds=seed, return_cache=True)
        mask = torch.ones(2, 2, dtype=torch.bool)
        latent = hidden[:, -1:]
        for index in range(steps):
            latent, cache, mask = trunk.forward_latent_step(latent, cache, mask)
            assert torch.isfinite(latent).all()
            assert mask.shape == (2, index + 3)
            assert all(k.shape[2] == index + 3 for k, _ in cache)


def test_invalid_inputs(trunk):
    dtype = torch.float32 if trunk.act_dtype == torch.bfloat16 else torch.bfloat16
    good = torch.zeros(1, 1, trunk.cfg.hidden_size, dtype=trunk.act_dtype)
    with pytest.raises(ValueError):
        trunk()
    with pytest.raises(ValueError):
        trunk(torch.ones(1, 1, dtype=torch.long), inputs_embeds=good)
    for bad in (good[:, :0], good[:, :, :-1], good.squeeze(1), good.to(dtype),
                torch.full_like(good, float("nan"))):
        with pytest.raises(ValueError):
            trunk(inputs_embeds=bad)
    with torch.inference_mode():
        _, cache = trunk(inputs_embeds=good, return_cache=True)
    for bad in (good.squeeze(1), good[:, :0], good.to(dtype)):
        with pytest.raises(ValueError):
            trunk.forward_latent_step(bad, cache, torch.ones(1, 1))
    with pytest.raises(ValueError):
        trunk.forward_latent_step(good, cache, torch.ones(1, 2))


def test_runtime_rollout_direct_embeddings(trunk):
    from gen_zero.causal.zero_runtime import ZeroStandaloneRuntime

    runtime = object.__new__(ZeroStandaloneRuntime)
    runtime.model = trunk
    runtime.max_length = 12
    seed = torch.randn(1, 2, trunk.cfg.hidden_size).to(trunk.act_dtype) * .1
    states = runtime.rollout_latent(inputs_embeds=seed, steps=3)
    assert states.shape == (1, 3, trunk.cfg.hidden_size)
    assert torch.isfinite(states).all()
    with pytest.raises(ValueError):
        runtime.rollout_latent(inputs_embeds=seed, steps=0)
    with pytest.raises(ValueError):
        runtime.rollout_latent(inputs_embeds=seed, steps=11)
