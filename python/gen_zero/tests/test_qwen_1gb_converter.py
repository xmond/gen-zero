"""Tests for the native Qwen student contract without allocating a 1 GB model."""
from __future__ import annotations

from pathlib import Path
from dataclasses import asdict

import pytest
import torch

from gen_zero.model.qwen_1gb_converter import (
    Qwen1GBSpec,
    QwenCausalBridge,
    convert_qwen_9b,
    evenly_spaced_indices,
    load_native_artifact,
    parameter_count_for_spec,
    project_tensor,
    selected_qwen35_layers,
    selected_teacher_layers,
)


QWEN_VOCAB_SIZE = 248_320


def test_default_spec_has_real_500m_bf16_budget() -> None:
    spec = Qwen1GBSpec()
    count = parameter_count_for_spec(QWEN_VOCAB_SIZE, spec)
    assert count == 471_272_128
    assert 450_000_000 <= count <= 520_000_000
    assert count * 2 <= 1_050_000_000


def test_formula_matches_meta_qwen_model_exactly() -> None:
    transformers = pytest.importorskip("transformers")
    spec = Qwen1GBSpec()
    config = transformers.Qwen3_5TextConfig(
        vocab_size=QWEN_VOCAB_SIZE,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_hidden_layers=spec.num_hidden_layers,
        num_attention_heads=spec.num_attention_heads,
        num_key_value_heads=spec.num_key_value_heads,
        head_dim=spec.head_dim,
        linear_key_head_dim=spec.linear_key_head_dim,
        linear_value_head_dim=spec.linear_value_head_dim,
        linear_num_key_heads=spec.linear_num_key_heads,
        linear_num_value_heads=spec.linear_num_value_heads,
        full_attention_interval=spec.full_attention_interval,
        tie_word_embeddings=True,
    )
    with torch.device("meta"):
        model = transformers.Qwen3_5ForCausalLM(config)
        bridge = QwenCausalBridge(spec.hidden_size, spec.manifold_size)
    instantiated = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in bridge.parameters())
    assert instantiated == parameter_count_for_spec(QWEN_VOCAB_SIZE, spec)


def test_small_qwen_forward_aligns_with_64d_bridge() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3_5TextConfig(
        vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=4, linear_num_value_heads=8, full_attention_interval=4,
        tie_word_embeddings=True,
    )
    model = transformers.Qwen3_5ForCausalLM(config).eval()
    bridge = QwenCausalBridge(config.hidden_size, 64).eval()
    tokens = torch.tensor([[1, 5, 9, 2], [1, 8, 2, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    with torch.inference_mode():
        output = model(input_ids=tokens, attention_mask=mask, output_hidden_states=True,
                       use_cache=False, return_dict=True)
        state = bridge(output.hidden_states[-1])
    assert output.logits.shape == (2, 4, 97)
    assert state.shape == (2, 4, 64)
    assert torch.isfinite(state).all()


def test_native_artifact_round_trip_without_weight_substitution(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3_5TextConfig(
        vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=4, linear_num_value_heads=8, full_attention_interval=4,
        tie_word_embeddings=True,
    )
    model = transformers.Qwen3_5ForCausalLM(config).to(torch.bfloat16)
    bridge = QwenCausalBridge(64, 64).to(torch.bfloat16)
    path = tmp_path / "native.pt"
    torch.save({
        "format": "gen-zero-qwen-1gb-v1", "config": config.to_dict(),
        "spec": {**asdict(Qwen1GBSpec()), "hidden_size": 64},
        "model_state_dict": model.state_dict(),
        "causal_bridge_state_dict": bridge.state_dict(),
        "provenance": {"fixture": True},
    }, path)
    restored, restored_bridge, provenance = load_native_artifact(path)
    assert provenance == {"fixture": True}
    assert sum(p.numel() for p in restored.parameters()) == sum(p.numel() for p in model.parameters())
    assert restored_bridge.projection.weight.shape == (64, 64)


def test_projection_and_layer_slicing_are_deterministic_and_generic() -> None:
    source = torch.arange(7 * 5, dtype=torch.float32).reshape(7, 5)
    first = project_tensor(source, torch.Size((4, 3)))
    second = project_tensor(source, torch.Size((4, 3)))
    assert torch.equal(first, second)
    assert first.shape == (4, 3)
    assert selected_teacher_layers(40, 24)[0] == 0
    assert selected_teacher_layers(40, 24)[-1] == 39
    layer_map = selected_qwen35_layers(32, 16)
    assert len(layer_map) == 16
    assert all((student % 4) == (teacher % 4) for student, teacher in enumerate(layer_map))
    assert evenly_spaced_indices(9, 4, torch.device("cpu")).tolist() == [0, 3, 5, 8]


def test_missing_teacher_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="local Qwen checkpoint is required"):
        convert_qwen_9b(tmp_path / "absent", tmp_path / "student.pt")
    assert not (tmp_path / "student.pt").exists()
