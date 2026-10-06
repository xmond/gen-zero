"""Real computation tests; compact configs test math, production config tests storage/forward."""
# anti-leakage: allow-mock-tensor
import gc
import os
from pathlib import Path

import pytest
import torch

from gen_zero.model.zero_model import ZeroConfig, ZeroModel, load_zero_tokenizer
from gen_zero.model.zero_converter import ZeroConverter, distillation_loss


def tiny(**kwargs):
    return ZeroConfig(vocab_size=37, hidden_size=32, num_hidden_layers=2,
                      num_attention_heads=4, intermediate_size=48, **kwargs)


def tiny_sparse(**kwargs):
    """4 layers, sparse_layer_start=3 (1-indexed) -> 0-indexed layers 2,3 are sparse (2 dense, 2 sparse)."""
    defaults = dict(vocab_size=37, hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
                    intermediate_size=48, sparse_moe=True, sparse_layer_start=3,
                    num_experts=8, experts_per_token=1, expert_intermediate_size=12)
    defaults.update(kwargs)
    return ZeroConfig(**defaults)


@pytest.fixture(scope='module')
def qwen_path():
    specified = os.environ.get('ZERO_TEST_QWEN2_PATH')
    if specified:
        return Path(specified)
    paths = sorted((Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots').glob('*'))
    if not paths:
        pytest.skip('Real local Qwen2 tokenizer/checkpoint unavailable; set ZERO_TEST_QWEN2_PATH')
    return paths[-1]


@pytest.fixture(autouse=True)
def single_core():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_production_bf16_storage_and_native_text_forward(qwen_path):
    from transformers import AutoTokenizer
    native = AutoTokenizer.from_pretrained(qwen_path, local_files_only=True)
    with pytest.raises(ValueError, match='exact alignment'):
        load_zero_tokenizer(qwen_path, ZeroConfig())
    # Include all native special tokens, with unchanged IDs and tokenizer pipeline.
    config = ZeroConfig(vocab_size=len(native))
    tokenizer = load_zero_tokenizer(qwen_path, config)
    model = ZeroModel(config).eval()
    count = sum(p.numel() for p in model.parameters())
    assert count == config.parameter_count()
    assert 450_000_000 <= count <= 490_000_000
    assert model.storage_bytes() == config.storage_bytes() <= 1_000_000_000
    assert all(p.device.type == 'cpu' and p.dtype == torch.bfloat16 for p in model.parameters())
    texts = ['Causal relations require interventional verification.', 'التدخل يغير النتيجة.', 'cause and effect', 'Intervention changes outcomes.']
    tokens = tokenizer(texts, padding=True, return_tensors='pt', add_special_tokens=False)
    assert tokenizer.batch_decode(tokenizer(texts, add_special_tokens=False)['input_ids']) == texts
    with torch.inference_mode():
        output = model(tokens['input_ids'], tokens['attention_mask'])
    assert output.shape == (4, 64) and torch.isfinite(output).all()
    print(f'production parameters={count}; BF16 bytes={model.storage_bytes()}; '
          f'CPU threads={torch.get_num_threads()}; output={tuple(output.shape)}')
    del model
    gc.collect()


def test_default_exact_count():
    config = ZeroConfig()
    model = ZeroModel(config, device='meta')
    assert sum(p.numel() for p in model.parameters()) == config.parameter_count()
    assert 450_000_000 <= config.parameter_count() <= 490_000_000
    assert config.storage_bytes() <= 1_000_000_000
    print(f'default parameters={config.parameter_count()}; BF16 bytes={config.storage_bytes()}')


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16])
def test_backward_finite(dtype):
    torch.manual_seed(314)
    model = ZeroModel(tiny(), dtype=dtype)
    output = model(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    output.float().square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_padding_invariance_and_causality():
    model = ZeroModel(tiny(), dtype=torch.float32).eval()
    with torch.no_grad():
        plain = model(torch.tensor([[1, 2, 3]]))
        left = model(torch.tensor([[0, 0, 1, 2, 3]]), torch.tensor([[0, 0, 1, 1, 1]]))
        right = model(torch.tensor([[1, 2, 3, 0, 0]]), torch.tensor([[1, 1, 1, 0, 0]]))
        a, _ = model.hidden_states(torch.tensor([[1, 2, 3]]))
        b, _ = model.hidden_states(torch.tensor([[1, 2, 4]]))
    torch.testing.assert_close(plain, left)
    torch.testing.assert_close(plain, right)
    torch.testing.assert_close(a[:, :2], b[:, :2])
    for ids, mask in [(torch.tensor([[37]]), None), (torch.tensor([[1]]), torch.tensor([[0]])),
                      (torch.tensor([[1]]), torch.tensor([[0.5]]))]:
        with pytest.raises(ValueError):
            model(ids, mask)


def test_missing_checkpoint_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match='No teacher checkpoint'):
        ZeroConverter(tmp_path)


def test_real_checkpoint_slice(qwen_path):
    from transformers import AutoTokenizer
    native = AutoTokenizer.from_pretrained(qwen_path, local_files_only=True)
    converter = ZeroConverter(qwen_path)
    config = ZeroConfig(vocab_size=len(native), hidden_size=32, num_hidden_layers=2,
                        num_attention_heads=4, intermediate_size=48, attention_bias=True,
                        rope_theta=converter.source['rope_theta'], rms_norm_eps=converter.source['rms_norm_eps'])
    model, report = converter.convert(config, tokenizer_path=qwen_path)
    original = converter._read('model.embed_tokens.weight')
    torch.testing.assert_close(model.embed_tokens.weight, original[:len(native), :32].to(torch.bfloat16), rtol=0, atol=0)
    assert len(report.tensor_sha256) == 26
    assert report.initialized_only == ('manifold_head.weight',) and not report.distilled
    assert report.source_layers == (0, converter.source['num_hidden_layers']-1)
    source = converter.source
    dimension = source['hidden_size']//source['num_attention_heads']
    heads = torch.linspace(0, source['num_attention_heads']-1, 4).round().long()
    coordinates = torch.tensor([0, 1, 2, 3, dimension//2, dimension//2+1, dimension//2+2, dimension//2+3])
    rows = (heads[:, None]*dimension+coordinates).flatten()
    original_q = converter._read(f'model.layers.{report.source_layers[-1]}.self_attn.q_proj.weight')
    torch.testing.assert_close(model.layers[-1].q_proj.weight,
                               original_q[rows, :32].to(torch.bfloat16), rtol=0, atol=0)
    kv = heads//(source['num_attention_heads']//source['num_key_value_heads'])
    kv_rows = (kv[:, None]*dimension+coordinates).flatten()
    original_k = converter._read(f'model.layers.{report.source_layers[-1]}.self_attn.k_proj.weight')
    torch.testing.assert_close(model.layers[-1].k_proj.weight,
                               original_k[kv_rows, :32].to(torch.bfloat16), rtol=0, atol=0)
    with pytest.raises(ValueError, match='structured student dimensions'):
        converter.convert(ZeroConfig(vocab_size=len(native), attention_bias=True), tokenizer_path=qwen_path)
    tokens = native('real weight slice verification', return_tensors='pt')
    with torch.no_grad():
        assert torch.isfinite(model(tokens['input_ids'])).all()
    print(f'real slice source={report.checkpoint}; layers={report.source_layers}; tensors={len(report.tensor_sha256)}')


def test_offline_loss_gradients_and_teacher_detachment():
    torch.manual_seed(17)
    model = ZeroModel(tiny(), dtype=torch.float32)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    logits = model.token_logits(ids)
    state = model(ids)
    counterfactual = model(ids.flip(-1))
    # Numerical inputs to exercise the objective; not converted teacher evidence.
    teacher_logits = torch.randn_like(logits, requires_grad=True)
    teacher_state = torch.randn(2, 64, requires_grad=True)
    teacher_cf = -teacher_state
    labels = torch.tensor([[2, 3, -100], [5, 6, -100]])
    losses = distillation_loss(logits, teacher_logits, labels, state, teacher_state, counterfactual, teacher_cf)
    assert all(torch.isfinite(x) for x in losses.values())
    losses['loss'].backward()
    assert teacher_logits.grad is None and teacher_state.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError, match='not separated'):
        distillation_loss(logits, teacher_logits, labels, state, teacher_state, counterfactual, teacher_state)
    with pytest.raises(ValueError, match='supervised token'):
        distillation_loss(logits, teacher_logits, torch.full_like(labels, -100), state, teacher_state, counterfactual, teacher_cf)


# --- Elastic scaling spectrum (docs/zero/01 section 1.1) ---------------------------------------

def test_zero_lite_is_the_unchanged_default():
    assert ZeroConfig.zero_lite() == ZeroConfig()


def test_elastic_presets_parameter_and_bf16_budget():
    lite, plus, maxc = ZeroConfig.zero_lite(), ZeroConfig.zero_plus(), ZeroConfig.zero_max()
    for cfg in (lite, plus, maxc):
        assert cfg.vocab_size == 151643 and cfg.manifold_dim == 64 and cfg.head_dim == 64

    assert (lite.num_hidden_layers, lite.hidden_size, lite.intermediate_size, lite.num_attention_heads) \
        == (24, 1024, 2816, 16)
    assert lite.parameter_count() == 463_679_488
    assert lite.storage_bytes() == 927_358_976
    assert 450_000_000 <= lite.parameter_count() <= 490_000_000
    assert lite.storage_bytes() <= 1_000_000_000

    assert (plus.num_hidden_layers, plus.hidden_size, plus.intermediate_size, plus.num_attention_heads) \
        == (28, 1536, 4096, 24)
    assert plus.parameter_count() == 1_025_832_960
    assert plus.storage_bytes() == 2_051_665_920
    assert 1.0e9 <= plus.parameter_count() <= 1.2e9
    assert 2.0e9 <= plus.storage_bytes() <= 2.4e9  # doc says ~2.1-2.4GB; exact value is 2.052GB

    assert (maxc.num_hidden_layers, maxc.hidden_size, maxc.intermediate_size, maxc.num_attention_heads) \
        == (32, 2048, 5632, 32)
    assert maxc.parameter_count() == 1_954_996_224
    assert maxc.storage_bytes() == 3_909_992_448
    assert 1.8e9 <= maxc.parameter_count() <= 2.2e9
    assert 3.8e9 <= maxc.storage_bytes() <= 4.4e9

    # Cross-check parameter_count() against an instantiated (meta-device) model, not just itself.
    for cfg in (lite, plus, maxc):
        model = ZeroModel(cfg, device='meta')
        assert sum(p.numel() for p in model.parameters()) == cfg.parameter_count()
    print(f'lite={lite.parameter_count()} plus={plus.parameter_count()} max={maxc.parameter_count()}')


def test_elastic_presets_forward_real_computation():
    """Real (non-meta) forward on small batches for all three presets, CPU single-thread."""
    torch.manual_seed(5)
    for cfg in (ZeroConfig.zero_lite(), ZeroConfig.zero_plus(), ZeroConfig.zero_max()):
        model = ZeroModel(cfg, dtype=torch.bfloat16).eval()
        ids = torch.randint(0, cfg.vocab_size, (2, 6))
        with torch.inference_mode():
            out = model(ids)
        assert out.shape == (2, 64) and torch.isfinite(out.float()).all()
        del model
        gc.collect()


# --- Dynamical sparse MoE, docs/zero/05 dimension 3, Phase 2.3 ---------------------------------

def test_sparse_moe_disabled_by_default_matches_dense_formula():
    dense = ZeroConfig.zero_lite()
    assert dense.sparse_layer_indices == frozenset()
    assert dense.active_parameter_count() == dense.parameter_count()


def test_sparse_config_validation():
    with pytest.raises(ValueError, match='experts_per_token'):
        tiny_sparse(experts_per_token=9)
    with pytest.raises(ValueError, match='sparse_layer_start'):
        tiny_sparse(sparse_layer_start=5)  # 1-indexed, must be within 1..num_hidden_layers (4)
    with pytest.raises(ValueError, match='num_experts'):
        tiny_sparse(num_experts=0)


def test_zero_lite_sparse_moe_active_parameter_reduction():
    """Real numbers for the doc's own baseline: sparse_layer_start=16, 8 experts, top-1,
    expert width = intermediate_size//4 (2816//4=704, the doc's own suggested option).

    sparse_layer_start=16 is 1-indexed inclusive (sparse_layer_indices docstring, zero_model.py):
    layers 16-24 (1-indexed) = 0-indexed 15-23 = 9 sparse layers, 15 dense. (This test previously
    hardcoded frozenset(range(16, 24)), 8 layers, and stale totals of 532_951_040/411_840_512 --
    a one-off transcription bug against the property's own documented convention, unrelated to
    factorized embedding; corrected here against the real formula output, verified independently
    against ZeroModel(cfg, device='meta') below.)

    Honest result: total stored parameters GROW to ~541.6M (8 resident experts per sparse layer,
    each ~2.16M params, not the doc's claimed ~8M/branch -- 3*1024*704 = 2,162,688), and active
    (per-token) parameters fall to ~405.4M, not the doc's claimed ~220M. 220M is not reachable
    with a non-factorized embedding: the full embedding table (155.3M) plus full dense attention
    across all 24 layers (100.7M) alone is a 256M floor, already above 220M before any MLP
    compute. Breaking that floor needs Factorized Embedding Parameterization, see
    ZeroConfig.zero_compact_220m() and test_zero_compact_220m_active_and_storage_budgets.
    """
    cfg = ZeroConfig.zero_lite(sparse_moe=True)
    assert cfg.sparse_layer_start == 16
    assert cfg.num_experts == 8
    assert cfg.experts_per_token == 1
    assert cfg.effective_expert_intermediate_size == 704
    assert cfg.sparse_layer_indices == frozenset(range(15, 24))

    assert cfg.parameter_count() == 541_609_984
    assert cfg.active_parameter_count() == 405_360_640
    assert cfg.active_parameter_count() < ZeroConfig.zero_lite().parameter_count()

    model = ZeroModel(cfg, device='meta')
    assert sum(p.numel() for p in model.parameters()) == cfg.parameter_count()
    print(f'zero-lite sparse: total={cfg.parameter_count()} active={cfg.active_parameter_count()} '
          f'(dense baseline={ZeroConfig.zero_lite().parameter_count()})')


def test_active_parameter_count_independent_recomputation():
    """Recompute the active-parameter formula independently of ZeroConfig internals."""
    cfg = tiny_sparse()
    h, i, e = cfg.hidden_size, cfg.intermediate_size, cfg.effective_expert_intermediate_size
    n_sparse = 2  # layers 2,3 of 4
    n_dense = cfg.num_hidden_layers - n_sparse
    attn = 4*h*h
    norms = 2*h
    dense_mlp = 3*h*i
    router = h*cfg.num_experts
    expert = 3*h*e
    expected = (cfg.vocab_size*h
                + n_dense*(attn+norms+dense_mlp)
                + n_sparse*(attn+norms+router+cfg.experts_per_token*expert)
                + h + h*64)
    assert cfg.active_parameter_count() == expected == 31936


def test_top1_routing_matches_manual_argmax_and_gate_weight():
    torch.manual_seed(11)
    cfg = tiny_sparse(experts_per_token=1)
    model = ZeroModel(cfg, dtype=torch.float32).eval()
    pool = model.layers[2].mlp_pool
    z = torch.randn(3, 5, cfg.hidden_size)
    with torch.no_grad():
        logits = pool.router(z.reshape(-1, cfg.hidden_size).float())
        probs = logits.softmax(-1)
        best_prob, best_idx = probs.max(-1)
        routed = pool.routed_experts(z).view(-1)
        assert torch.equal(routed, best_idx)

        expected = torch.zeros(15, cfg.hidden_size)
        for row in range(15):
            e = best_idx[row].item()
            expected[row] = best_prob[row]*pool.experts[e](z.reshape(-1, cfg.hidden_size)[row:row+1])[0]
        actual = pool(z).reshape(-1, cfg.hidden_size)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_top2_routing_matches_manual_topk_and_differs_from_top1():
    torch.manual_seed(11)
    cfg1, cfg2 = tiny_sparse(experts_per_token=1), tiny_sparse(experts_per_token=2)
    torch.manual_seed(3)
    model1 = ZeroModel(cfg1, dtype=torch.float32).eval()
    torch.manual_seed(3)
    model2 = ZeroModel(cfg2, dtype=torch.float32).eval()
    pool1, pool2 = model1.layers[2].mlp_pool, model2.layers[2].mlp_pool
    # Identically-initialized experts/router up to the k-dependent forward.
    for e1, e2 in zip(pool1.experts, pool2.experts):
        torch.testing.assert_close(e1.gate_proj.weight, e2.gate_proj.weight)
    z = torch.randn(2, 4, cfg1.hidden_size)
    with torch.no_grad():
        flat = z.reshape(-1, cfg1.hidden_size)
        logits = pool2.router(flat.float())
        probs = logits.softmax(-1)
        top_probs, top_idx = probs.topk(2, dim=-1)
        expected = torch.zeros(8, cfg1.hidden_size)
        for row in range(8):
            for slot in range(2):
                e = top_idx[row, slot].item()
                expected[row] += top_probs[row, slot]*pool2.experts[e](flat[row:row+1])[0]
        actual2 = pool2(z).reshape(-1, cfg1.hidden_size)
        torch.testing.assert_close(actual2, expected, rtol=1e-5, atol=1e-5)
        actual1 = pool1(z).reshape(-1, cfg1.hidden_size)
        assert not torch.allclose(actual1, actual2)


def test_sparse_forward_shape_finite_deterministic_and_padding_invariant():
    torch.manual_seed(9)
    for k in (1, 2):
        cfg = tiny_sparse(experts_per_token=k)
        model = ZeroModel(cfg, dtype=torch.float32).eval()
        ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        with torch.no_grad():
            out1 = model(ids)
            out2 = model(ids)
        assert out1.shape == (2, 64) and torch.isfinite(out1).all()
        assert torch.equal(out1, out2)  # deterministic: no dithering/noise in routing

        with torch.no_grad():
            plain = model(torch.tensor([[1, 2, 3]]))
            left = model(torch.tensor([[0, 0, 1, 2, 3]]), torch.tensor([[0, 0, 1, 1, 1]]))
            right = model(torch.tensor([[1, 2, 3, 0, 0]]), torch.tensor([[1, 1, 1, 0, 0]]))
        torch.testing.assert_close(plain, left, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(plain, right, rtol=1e-5, atol=1e-5)


def test_sparse_backward_touched_experts_match_active_formula():
    """The trap: a naive test would assert every expert gets nonzero grad every time, which is
    false for real top-k MoE. With exactly one token in the batch, exactly experts_per_token
    experts per sparse layer receive gradient; the rest are untouched (grad is None) because
    they were never called (row-gather dispatch, not a densely-computed-then-masked pool)."""
    torch.manual_seed(21)
    cfg = tiny_sparse(experts_per_token=1)
    model = ZeroModel(cfg, dtype=torch.float32)
    ids = torch.tensor([[3]])  # batch=1, seq=1: exactly one token routed per sparse layer
    out = model(ids)
    out.float().square().mean().backward()

    touched_params = 0
    for i, layer in enumerate(model.layers):
        h = cfg.hidden_size
        for p in (layer.q_proj, layer.k_proj, layer.v_proj, layer.o_proj):
            assert p.weight.grad is not None and torch.isfinite(p.weight.grad).all()
            touched_params += p.weight.numel()
        for n in (layer.input_layernorm, layer.post_attention_layernorm):
            assert n.weight.grad is not None
            touched_params += n.weight.numel()
        if i in cfg.sparse_layer_indices:
            pool = layer.mlp_pool
            assert pool.router.weight.grad is not None and torch.isfinite(pool.router.weight.grad).all()
            touched_params += pool.router.weight.numel()
            touched_experts = [e for e, exp in enumerate(pool.experts) if exp.gate_proj.weight.grad is not None]
            assert len(touched_experts) == cfg.experts_per_token
            for e in touched_experts:
                exp = pool.experts[e]
                for p in (exp.gate_proj, exp.up_proj, exp.down_proj):
                    assert p.weight.grad is not None and torch.isfinite(p.weight.grad).all()
                    touched_params += p.weight.numel()
            untouched = set(range(cfg.num_experts)) - set(touched_experts)
            for e in untouched:
                assert pool.experts[e].gate_proj.weight.grad is None
        else:
            for p in (layer.gate_proj, layer.up_proj, layer.down_proj):
                assert p.weight.grad is not None and torch.isfinite(p.weight.grad).all()
                touched_params += p.weight.numel()

    expected_backbone_active = cfg.active_parameter_count() - cfg.vocab_size*cfg.hidden_size \
        - cfg.hidden_size - cfg.hidden_size*64
    assert touched_params == expected_backbone_active == 28672
    assert model.embed_tokens.weight.grad is not None
    assert model.manifold_head.weight.grad is not None


def tiny_factorized(**kwargs):
    defaults = dict(vocab_size=100, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                    intermediate_size=48, factorized_embedding=True, embedding_dim=16)
    defaults.update(kwargs)
    return ZeroConfig(**defaults)


# --- Factorized Embedding Parameterization + zero_compact_220m preset --------------------------

def test_factorized_embedding_disabled_by_default_matches_dense_formula():
    cfg = ZeroConfig()
    assert cfg.factorized_embedding is False
    assert cfg.embedding_table_cost == cfg.vocab_size * cfg.hidden_size


def test_factorized_embedding_validation():
    with pytest.raises(ValueError, match='embedding_dim must be a positive integer'):
        ZeroConfig(factorized_embedding=True, embedding_dim=0)
    with pytest.raises(ValueError, match='embedding_dim must be smaller than hidden_size'):
        ZeroConfig(factorized_embedding=True, hidden_size=1024, embedding_dim=1024)


def test_factorized_embedding_shrinks_table_cost_and_matches_real_model():
    cfg = ZeroConfig(vocab_size=151643, hidden_size=1024, factorized_embedding=True, embedding_dim=128)
    assert cfg.embedding_table_cost == 151643*128 + 128*1024 == 19_541_376
    plain = ZeroConfig(vocab_size=151643, hidden_size=1024)
    assert cfg.embedding_table_cost < plain.embedding_table_cost  # 19.5M vs 155.3M

    model = ZeroModel(cfg, device='meta')
    assert sum(p.numel() for p in model.parameters()) == cfg.parameter_count()
    assert model.embed_tokens.weight.shape == (151643, 128)
    assert model.embed_proj.weight.shape == (1024, 128)


def test_zero_compact_220m_active_and_storage_budgets():
    cfg = ZeroConfig.zero_compact_220m()
    assert cfg.vocab_size == 151643 and cfg.hidden_size == 1024 and cfg.embedding_dim == 128
    assert cfg.factorized_embedding is True
    assert cfg.num_hidden_layers == 24 and cfg.num_attention_heads == 16
    assert cfg.sparse_moe is True and cfg.num_experts == 8 and cfg.experts_per_token == 1

    assert cfg.active_parameter_count() == 210_637_184
    assert cfg.active_parameter_count() <= 230_000_000
    assert cfg.parameter_count() == 346_886_528
    assert cfg.parameter_count() <= 350_000_000
    assert cfg.storage_bytes() == 693_773_056
    assert cfg.storage_bytes() <= 1_000_000_000

    model = ZeroModel(cfg, device='meta')
    assert sum(p.numel() for p in model.parameters()) == cfg.parameter_count()
    print(f'zero_compact_220m: active={cfg.active_parameter_count():,} '
          f'total={cfg.parameter_count():,} bf16_bytes={cfg.storage_bytes():,}')


def test_zero_compact_220m_real_forward_within_storage_budget():
    """One real (non-meta) instantiation: exact BF16 storage, finite (B, 64) forward."""
    torch.manual_seed(7)
    cfg = ZeroConfig.zero_compact_220m()
    model = ZeroModel(cfg, dtype=torch.bfloat16).eval()
    assert model.storage_bytes() == cfg.storage_bytes() <= 1_000_000_000
    ids = torch.randint(0, cfg.vocab_size, (2, 6))
    with torch.inference_mode():
        out = model(ids)
    assert out.shape == (2, 64) and torch.isfinite(out.float()).all()
    del model
    gc.collect()


def test_factorized_embedding_forward_shape_and_determinism():
    torch.manual_seed(19)
    cfg = tiny_factorized()
    model = ZeroModel(cfg, dtype=torch.float32).eval()
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    with torch.no_grad():
        out1 = model(ids)
        out2 = model(ids)
    assert out1.shape == (2, 64) and torch.isfinite(out1).all()
    assert torch.equal(out1, out2)


def test_factorized_embedding_gradients_reach_both_matrices():
    torch.manual_seed(23)
    cfg = tiny_factorized()
    model = ZeroModel(cfg, dtype=torch.float32)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    out = model(ids)
    out.float().square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.embed_tokens.weight.grad.abs().sum() > 0
    assert model.embed_proj.weight.grad.abs().sum() > 0


def test_factorized_embedding_token_logits_rejected():
    cfg = tiny_factorized()
    model = ZeroModel(cfg, dtype=torch.float32)
    with pytest.raises(ValueError, match='not defined for factorized_embedding'):
        model.token_logits(torch.tensor([[1, 2, 3]]))


def test_sparse_moe_converter_guard(qwen_path):
    converter = ZeroConverter(qwen_path)
    from transformers import AutoTokenizer
    native = AutoTokenizer.from_pretrained(qwen_path, local_files_only=True)
    cfg = ZeroConfig(vocab_size=len(native), hidden_size=32, num_hidden_layers=4,
                     num_attention_heads=4, intermediate_size=48, sparse_moe=True,
                     sparse_layer_start=2, num_experts=8, experts_per_token=1,
                     expert_intermediate_size=12)
    with pytest.raises(ValueError, match='no dense teacher slice'):
        converter.convert(cfg, tokenizer_path=qwen_path)
