"""Structure-preserving Qwen-9B -> CPU student conversion.

This module intentionally has no random-weight or download fallback.  A conversion
requires a local Hugging Face Qwen checkpoint and copies every student tensor from
the teacher by deterministic layer and coordinate selection.  The tokenizer files
are copied verbatim by ``save_pretrained``.

The default text-only Qwen3.5 architecture has 471,272,128 parameters with its
248,320-entry vocabulary: 16 layers, width 1024, SwiGLU width 3072, and the
native 3:1 linear/full-attention cadence.  In BF16 its raw tensors occupy
942,544,256 bytes.  Input and output embeddings are tied to make that budget
possible; the 9B teacher itself uses separate matrices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class Qwen1GBSpec:
    """Default student shape; vocabulary size is inherited from the teacher."""

    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 64
    linear_key_head_dim: int = 32
    linear_value_head_dim: int = 32
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    full_attention_interval: int = 4
    manifold_size: int = 64
    dtype: str = "bfloat16"

    def validate(self) -> None:
        numeric = (
            self.hidden_size, self.intermediate_size, self.num_hidden_layers,
            self.num_attention_heads, self.num_key_value_heads, self.manifold_size,
            self.head_dim, self.linear_key_head_dim, self.linear_value_head_dim,
            self.linear_num_key_heads, self.linear_num_value_heads, self.full_attention_interval,
        )
        if any(value <= 0 for value in numeric):
            raise ValueError("all dimensions must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.manifold_size != 64:
            raise ValueError("Gen-Zero causal manifold must have exactly 64 dimensions")
        if self.dtype != "bfloat16":
            raise ValueError("the native artifact format currently requires bfloat16")


def evenly_spaced_indices(source_size: int, target_size: int, device: torch.device) -> Tensor:
    """Select a deterministic, endpoint-preserving subspace without language rules."""
    if target_size > source_size:
        raise ValueError(f"cannot project dimension {source_size} into larger dimension {target_size}")
    if target_size == source_size:
        return torch.arange(source_size, device=device)
    return torch.linspace(0, source_size - 1, target_size, device=device).round().long()


def project_tensor(source: Tensor, shape: torch.Size) -> Tensor:
    """Coordinate-project ``source`` to ``shape`` on every axis.

    This is lossy structured weight projection, not distillation and not a claim of
    exact functional equivalence.
    """
    if source.ndim != len(shape):
        raise ValueError(f"rank mismatch: teacher {tuple(source.shape)}, student {tuple(shape)}")
    result = source.detach()
    for axis, wanted in enumerate(shape):
        index = evenly_spaced_indices(result.shape[axis], wanted, result.device)
        result = result.index_select(axis, index)
    return result


def selected_teacher_layers(teacher_layers: int, student_layers: int) -> list[int]:
    if student_layers > teacher_layers:
        raise ValueError("student cannot contain more layers than teacher")
    return torch.linspace(0, teacher_layers - 1, student_layers).round().long().tolist()


def parameter_count_for_spec(vocab_size: int, spec: Qwen1GBSpec, *, tie_embeddings: bool = True) -> int:
    """Exact Qwen3.5 text parameter count including the 64-D bridge."""
    h, m, layers = spec.hidden_size, spec.intermediate_size, spec.num_hidden_layers
    embeddings = vocab_size * h * (1 if tie_embeddings else 2)
    full_layers = layers // spec.full_attention_interval
    linear_layers = layers - full_layers
    linear_qkv = (
        2 * spec.linear_num_key_heads * spec.linear_key_head_dim
        + spec.linear_num_value_heads * spec.linear_value_head_dim
    )
    per_linear = (
        3 * spec.linear_num_value_heads + linear_qkv * 4 + h * h
        + linear_qkv * h + h * h + 2 * spec.linear_num_value_heads * h
        + 3 * h * m + 2 * h
    )
    per_full = (
        2 * spec.num_attention_heads * spec.head_dim * h
        + 2 * spec.num_key_value_heads * spec.head_dim * h + h * h
        + 2 * spec.head_dim + 3 * h * m + 2 * h
    )
    final_norm = h
    bridge = spec.manifold_size * h + spec.manifold_size
    return embeddings + linear_layers * per_linear + full_layers * per_full + final_norm + bridge


class QwenCausalBridge(nn.Module):
    """Trainable numeric-only adapter from native Qwen states to Gen-Zero 64-D."""

    def __init__(self, hidden_size: int, manifold_size: int = 64) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, manifold_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim not in (2, 3) or hidden_states.shape[-1] != self.projection.in_features:
            raise ValueError("hidden_states have incompatible shape")
        return self.projection(hidden_states)


def _require_local_source(path: Path) -> None:
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(f"local Qwen checkpoint is required: {path}")
    if not any(path.glob("*.safetensors")) and not any(path.glob("pytorch_model*.bin")):
        raise FileNotFoundError(f"no model weight shards found in {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _student_config(teacher: Any, spec: Qwen1GBSpec) -> Any:
    from transformers import Qwen3_5TextConfig

    teacher_text = teacher.config.text_config
    config = Qwen3_5TextConfig.from_dict(teacher_text.to_dict())
    config.hidden_size = spec.hidden_size
    config.intermediate_size = spec.intermediate_size
    config.num_hidden_layers = spec.num_hidden_layers
    config.num_attention_heads = spec.num_attention_heads
    config.num_key_value_heads = spec.num_key_value_heads
    config.head_dim = spec.head_dim
    config.linear_key_head_dim = spec.linear_key_head_dim
    config.linear_value_head_dim = spec.linear_value_head_dim
    config.linear_num_key_heads = spec.linear_num_key_heads
    config.linear_num_value_heads = spec.linear_num_value_heads
    config.full_attention_interval = spec.full_attention_interval
    config.layer_types = None
    config.tie_word_embeddings = True
    config.dtype = "bfloat16"
    return config


def selected_qwen35_layers(teacher_layers: int, student_layers: int,
                           interval: int = 4) -> list[int]:
    """Slice complete 3-linear/1-full groups, preserving layer architecture."""
    if teacher_layers % interval or student_layers % interval:
        raise ValueError("teacher and student layer counts must contain complete attention groups")
    teacher_groups, student_groups = teacher_layers // interval, student_layers // interval
    groups = selected_teacher_layers(teacher_groups, student_groups)
    return [group * interval + offset for group in groups for offset in range(interval)]


def _teacher_key(student_key: str, layer_map: list[int]) -> str:
    fields = student_key.split(".")
    try:
        pos = fields.index("layers") + 1
    except ValueError:
        return student_key
    fields[pos] = str(layer_map[int(fields[pos])])
    return ".".join(fields)


def convert_qwen_9b(source: Path, output: Path, *, spec: Qwen1GBSpec = Qwen1GBSpec()) -> dict[str, Any]:
    """Convert a local Qwen3.5-9B checkpoint and write one native ``.pt`` file."""
    spec.validate()
    source, output = Path(source).resolve(), Path(output).resolve()
    _require_local_source(source)
    from transformers import AutoModelForImageTextToText, AutoTokenizer, Qwen3_5ForCausalLM

    teacher = AutoModelForImageTextToText.from_pretrained(
        source, local_files_only=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=False,
    ).eval()
    model_type = getattr(teacher.config, "model_type", "")
    if model_type != "qwen3_5":
        raise ValueError(f"unsupported teacher architecture {model_type!r}; expected Qwen3.5-9B")
    if not (8_000_000_000 <= sum(p.numel() for p in teacher.parameters()) <= 11_000_000_000):
        raise ValueError("source is not a Qwen 9B-class checkpoint (expected 8B..11B parameters)")
    config = _student_config(teacher, spec)
    student = Qwen3_5ForCausalLM(config).to(dtype=torch.bfloat16).eval()
    layer_map = selected_qwen35_layers(teacher.config.text_config.num_hidden_layers,
                                      spec.num_hidden_layers, spec.full_attention_interval)
    # Conditional-generation checkpoints namespace text weights under model.language_model.
    teacher_state = {f"model.{key}": value for key, value in teacher.model.language_model.state_dict().items()}
    with torch.no_grad():
        for key, target in student.state_dict().items():
            if key == "lm_head.weight":
                continue  # tied to the already-projected input embedding
            source_key = _teacher_key(key, layer_map)
            if source_key not in teacher_state:
                raise KeyError(f"teacher has no tensor for {source_key}")
            target.copy_(project_tensor(teacher_state[source_key], target.shape).to(dtype=target.dtype))

    # PCA on a deterministic vocabulary sample initializes the bridge from teacher-derived
    # geometry.  It is an initialization only; semantic decisions remain uncalibrated.
    embedding = student.get_input_embeddings().weight.float()
    sample_idx = evenly_spaced_indices(embedding.shape[0], min(8192, embedding.shape[0]), embedding.device)
    sample = embedding.index_select(0, sample_idx)
    sample = sample - sample.mean(0, keepdim=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        _, _, components = torch.pca_lowrank(sample, q=spec.manifold_size, center=False, niter=4)
    bridge = QwenCausalBridge(spec.hidden_size, spec.manifold_size).to(dtype=torch.bfloat16)
    with torch.no_grad():
        bridge.projection.weight.copy_(components.T.to(torch.bfloat16))
        bridge.projection.bias.zero_()

    count = sum(p.numel() for p in student.parameters()) + sum(p.numel() for p in bridge.parameters())
    expected = parameter_count_for_spec(config.vocab_size, spec, tie_embeddings=True)
    if count != expected:
        raise RuntimeError(f"parameter accounting mismatch: instantiated={count}, expected={expected}")
    if not 450_000_000 <= count <= 520_000_000:
        raise RuntimeError(f"student parameter count {count} violates 450M..520M contract")
    payload = {
        "format": "gen-zero-qwen-1gb-v1",
        "config": config.to_dict(),
        "spec": asdict(spec),
        "model_state_dict": {k: v.cpu() for k, v in student.state_dict().items()},
        "causal_bridge_state_dict": {k: v.cpu() for k, v in bridge.state_dict().items()},
        "provenance": {
            "source": str(source), "source_model_type": model_type,
            "source_files_sha256": {
                item.name: _sha256(item)
                for item in sorted((*source.glob("*.safetensors"), *source.glob("pytorch_model*.bin"),
                                    source / "config.json"))
                if item.is_file()
            },
            "teacher_layers": teacher.config.text_config.num_hidden_layers, "layer_map": layer_map,
            "method": "deterministic_coordinate_projection_and_layer_slicing",
            "lossless": False, "calibrated_for_decisions": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    size = temporary.stat().st_size
    if size > 1_050_000_000:
        temporary.unlink()
        raise RuntimeError(f"artifact would be {size} bytes, above the 1.05 GB limit")
    temporary.replace(output)
    tokenizer_dir = output.with_suffix("").with_name(output.stem + "_tokenizer")
    AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False).save_pretrained(tokenizer_dir)
    return {"output": str(output), "tokenizer": str(tokenizer_dir), "parameters": count,
            "raw_bf16_bytes": count * 2, "artifact_bytes": size, "layer_map": layer_map}


def load_native_artifact(path: Path) -> tuple[Any, QwenCausalBridge, Mapping[str, Any]]:
    """Load only the native artifact; never downloads or substitutes weights."""
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    payload = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
    if payload.get("format") != "gen-zero-qwen-1gb-v1":
        raise ValueError("not a gen-zero-qwen-1gb-v1 artifact")
    config = Qwen3_5TextConfig.from_dict(payload["config"])
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
        bridge = QwenCausalBridge(config.hidden_size, payload["spec"]["manifold_size"])
    model.load_state_dict(payload["model_state_dict"], strict=True, assign=True)
    model.tie_weights()
    bridge.load_state_dict(payload["causal_bridge_state_dict"], strict=True, assign=True)
    return model.eval(), bridge.eval(), payload["provenance"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="local Qwen-9B Hugging Face directory")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/gen_zero_qwen_1gb.pt"))
    args = parser.parse_args()
    print(json.dumps(convert_qwen_9b(args.source, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
