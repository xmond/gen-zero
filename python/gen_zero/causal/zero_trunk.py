"""Zero backbone: a self-contained Qwen2-architecture trunk in plain PyTorch.

No ``transformers`` import at runtime. The architecture (RMSNorm, rotary
positions, grouped-query attention with q/k/v bias, SwiGLU MLP, tied
embedding without a language-model head) is written out here so the resident
process needs only ``torch`` and the ``tokenizers`` library. Weights come from
the local Qwen2.5-0.5B safetensors file (FP32/BF16 paths) or from the INT8
weight-only artifact built by :func:`build_int8_artifact`.

Numerical contract: with FP32 weights the last-token hidden state matches the
reference ``transformers`` implementation to float precision; the test suite
checks this against the real Hugging Face model, not a mock.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

__all__ = [
    "ZERO_ENCODER_FAMILY",
    "INT8_ARTIFACT_VERSION",
    "TrunkConfig",
    "Int8WeightOnlyLinear",
    "BF16Embedding",
    "ZeroTrunk",
    "KVCache",
    "build_int8_artifact",
    "load_trunk",
    "snapshot_digest",
]

ZERO_ENCODER_FAMILY = "zero-qwen2.5-0.5b-trunk"
INT8_ARTIFACT_VERSION = 2
PRECISIONS = ("int8", "bf16", "fp32")

KVCache = List[Tuple[Tensor, Tensor]]  # per layer (k, v) of shape (B, kv_heads, T, head_dim)


def malloc_trim() -> None:
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:  # non-glibc platform
        pass


class TrunkConfig:
    """The subset of the HF config the trunk needs, validated once."""

    REQUIRED = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "rms_norm_eps", "rope_theta", "vocab_size")

    def __init__(self, raw: dict) -> None:
        if raw.get("model_type") != "qwen2":
            raise ValueError("Zero's trunk is a qwen2 architecture; refusing another model type")
        if raw.get("use_sliding_window") or raw.get("use_mrope"):
            raise ValueError("sliding-window / mrope variants are not implemented")
        for key in self.REQUIRED:
            if key not in raw:
                raise ValueError(f"config is missing {key}")
        self.hidden_size = int(raw["hidden_size"])
        self.intermediate_size = int(raw["intermediate_size"])
        self.num_layers = int(raw["num_hidden_layers"])
        self.num_heads = int(raw["num_attention_heads"])
        self.num_kv_heads = int(raw["num_key_value_heads"])
        self.rms_norm_eps = float(raw["rms_norm_eps"])
        self.rope_theta = float(raw["rope_theta"])
        self.vocab_size = int(raw["vocab_size"])
        if self.hidden_size % self.num_heads or self.num_heads % self.num_kv_heads:
            raise ValueError("head geometry is inconsistent")
        self.head_dim = self.hidden_size // self.num_heads

    @classmethod
    def from_snapshot(cls, snapshot: Path) -> "TrunkConfig":
        return cls(json.loads((Path(snapshot) / "config.json").read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# Weight modules
# --------------------------------------------------------------------------

class Int8WeightOnlyLinear(nn.Module):
    """``y = x @ dequant(W)^T + b``; W is int8 with one FP32 scale per output row.

    Dequantization goes through one scratch buffer shared by all layers, so
    the transient FP32 copy never exceeds the largest single weight matrix.
    Activations are never quantized (see zero_runtime module docstring).
    """

    _scratch: Optional[Tensor] = None

    def __init__(self, in_features: int, out_features: int, bias: bool) -> None:
        super().__init__()
        self.in_features, self.out_features = int(in_features), int(out_features)
        self.register_buffer("qweight", torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.empty(out_features, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=torch.float32))
        else:
            self.bias = None

    def load_dense(self, weight: Tensor, bias: Optional[Tensor]) -> None:
        weight = weight.detach().to(torch.float32)
        if weight.shape != self.qweight.shape or not torch.isfinite(weight).all():
            raise ValueError("dense weight has the wrong shape or is non-finite")
        scale = weight.abs().amax(dim=1).clamp_min(1e-8) / 127.0
        self.qweight.copy_(torch.round(weight / scale[:, None]).clamp_(-127, 127).to(torch.int8))
        self.scale.copy_(scale)
        if (bias is None) != (self.bias is None):
            raise ValueError("bias presence mismatch")
        if bias is not None:
            self.bias.copy_(bias.detach().to(torch.float32))

    @classmethod
    def scratch(cls, numel: int) -> Tensor:
        if cls._scratch is None or cls._scratch.numel() < numel:
            cls._scratch = torch.empty(numel, dtype=torch.float32)
        return cls._scratch

    def forward(self, x: Tensor) -> Tensor:
        dense = self.scratch(self.qweight.numel())[: self.qweight.numel()].view_as(self.qweight)
        dense.copy_(self.qweight)
        out = nn.functional.linear(x, dense)
        out.mul_(self.scale)
        if self.bias is not None:
            out.add_(self.bias)
        return out


class DenseLinear(nn.Module):
    """Frozen FP32/BF16 linear stored as buffers (no autograd bookkeeping)."""

    def __init__(self, in_features: int, out_features: int, bias: bool, dtype: torch.dtype) -> None:
        super().__init__()
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=dtype))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=dtype))
        else:
            self.bias = None

    def load_dense(self, weight: Tensor, bias: Optional[Tensor]) -> None:
        self.weight.copy_(weight.detach().to(self.weight.dtype))
        if (bias is None) != (self.bias is None):
            raise ValueError("bias presence mismatch")
        if bias is not None:
            self.bias.copy_(bias.detach().to(self.bias.dtype))

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.linear(x.to(self.weight.dtype), self.weight, self.bias)


class BF16Embedding(nn.Module):
    def __init__(self, num_embeddings: int, dim: int, out_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.out_dtype = out_dtype
        self.register_buffer("weight", torch.empty(num_embeddings, dim, dtype=torch.bfloat16))

    def forward(self, ids: Tensor) -> Tensor:
        return nn.functional.embedding(ids, self.weight).to(self.out_dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.empty(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        h = x.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * h.to(dtype)) if dtype == torch.float32 else (self.weight.to(dtype) * h.to(dtype))


# --------------------------------------------------------------------------
# Architecture
# --------------------------------------------------------------------------

def _make_linear(cfg_precision: str, in_f: int, out_f: int, bias: bool) -> nn.Module:
    if cfg_precision == "int8":
        return Int8WeightOnlyLinear(in_f, out_f, bias)
    return DenseLinear(in_f, out_f, bias, torch.bfloat16 if cfg_precision == "bf16" else torch.float32)


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: TrunkConfig, precision: str) -> None:
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        self.q_proj = _make_linear(precision, cfg.hidden_size, cfg.num_heads * cfg.head_dim, True)
        self.k_proj = _make_linear(precision, cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, True)
        self.v_proj = _make_linear(precision, cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, True)
        self.o_proj = _make_linear(precision, cfg.num_heads * cfg.head_dim, cfg.hidden_size, False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor,
                past: Optional[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        present = (k, v)
        groups = self.num_heads // self.num_kv_heads
        k_full = k.repeat_interleave(groups, dim=1)
        v_full = v.repeat_interleave(groups, dim=1)
        out = nn.functional.scaled_dot_product_attention(q, k_full, v_full, attn_mask=mask)
        out = out.transpose(1, 2).reshape(batch, seq, self.num_heads * self.head_dim)
        return self.o_proj(out), present


class MLP(nn.Module):
    def __init__(self, cfg: TrunkConfig, precision: str) -> None:
        super().__init__()
        self.gate_proj = _make_linear(precision, cfg.hidden_size, cfg.intermediate_size, False)
        self.up_proj = _make_linear(precision, cfg.hidden_size, cfg.intermediate_size, False)
        self.down_proj = _make_linear(precision, cfg.intermediate_size, cfg.hidden_size, False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, cfg: TrunkConfig, precision: str) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg, precision)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg, precision)

    def forward(self, x, cos, sin, mask, past):
        attn, present = self.self_attn(self.input_layernorm(x), cos, sin, mask, past)
        h = x + attn
        return h + self.mlp(self.post_attention_layernorm(h)), present


class ZeroTrunk(nn.Module):
    """Embedding -> N decoder layers -> final RMSNorm. No LM head."""

    def __init__(self, cfg: TrunkConfig, precision: str) -> None:
        super().__init__()
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        self.cfg = cfg
        self.precision = precision
        act_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        self.act_dtype = act_dtype
        self.embed_tokens = BF16Embedding(cfg.vocab_size, cfg.hidden_size, act_dtype)
        self.layers = nn.ModuleList(Layer(cfg, precision) for _ in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rotary(self, positions: Tensor) -> Tuple[Tensor, Tensor]:
        freqs = positions.to(torch.float32)[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(self.act_dtype)[None, None], emb.sin().to(self.act_dtype)[None, None]

    def forward(self, input_ids: Optional[Tensor] = None, attention_mask: Optional[Tensor] = None,
                *, inputs_embeds: Optional[Tensor] = None, past: Optional[KVCache] = None,
                past_mask: Optional[Tensor] = None, return_cache: bool = False
                ) -> Tuple[Tensor, Optional[KVCache]]:
        """Return hidden states (B, T, hidden) and optionally the per-layer KV cache.

        ``attention_mask`` is (B, T) for the new tokens; ``past_mask`` is (B, P)
        for cached positions. Query row i may attend to cached position j when
        ``past_mask[b, j]`` and to new position j <= i when ``attention_mask[b, j]``.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids and inputs_embeds")
        if inputs_embeds is not None:
            if inputs_embeds.ndim != 3 or inputs_embeds.shape[-1] != self.cfg.hidden_size:
                raise ValueError("inputs_embeds must be (B, T, hidden_size)")
            if inputs_embeds.dtype != self.act_dtype or inputs_embeds.device != self.inv_freq.device:
                raise ValueError("inputs_embeds dtype and device must match trunk activations")
            if not torch.isfinite(inputs_embeds).all():
                raise ValueError("inputs_embeds must be finite")
            batch, seq = inputs_embeds.shape[:2]
        else:
            if input_ids.ndim != 2 or input_ids.dtype != torch.long:
                raise ValueError("input_ids must be a (B, T) long tensor")
            batch, seq = input_ids.shape
        if batch == 0 or seq == 0:
            raise ValueError("input batch and sequence must be nonempty")
        if attention_mask is None:
            attention_mask = torch.ones((batch, seq), dtype=torch.bool, device=self.inv_freq.device)
        elif attention_mask.shape != (batch, seq) or attention_mask.device != self.inv_freq.device:
            raise ValueError("attention_mask must be (B, T) on the trunk device")
        if past is not None and len(past) != len(self.layers):
            raise ValueError("KV cache layer count mismatch")
        past_len = 0 if past is None else int(past[0][0].shape[2])
        if past is not None and (len(past) != len(self.layers) or past_mask is None or
                                 past_mask.shape != (batch, past_len)):
            raise ValueError("past_mask must be (B, P) when a cache is supplied")
        if past is None and past_mask is not None:
            raise ValueError("past_mask requires a cache")
        if past is not None:
            if past_mask.device != self.inv_freq.device:
                raise ValueError("past_mask must be on the trunk device")
            for key, value in past:
                expected = (batch, self.cfg.num_kv_heads, past_len, self.cfg.head_dim)
                if key.shape != expected or value.shape != expected or key.dtype != self.act_dtype or value.dtype != self.act_dtype:
                    raise ValueError("KV cache shape or dtype mismatch")
        positions = torch.arange(past_len, past_len + seq, device=self.inv_freq.device)
        cos, sin = self.rotary(positions)
        new_valid = attention_mask != 0
        causal = torch.ones(seq, seq, dtype=torch.bool).tril()
        mask_new = causal[None] & new_valid[:, None, :]
        if past is not None:
            mask = torch.cat([(past_mask != 0)[:, None, :].expand(batch, seq, past_len), mask_new], dim=2)
        else:
            mask = mask_new
        # Padded query rows attend to nothing in a causal batch only if they are
        # the first token; guarantee at least one key so SDPA never yields NaN.
        mask = mask | (~mask.any(dim=2, keepdim=True))
        mask = mask[:, None]
        h = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        presents: KVCache = []
        for i, layer in enumerate(self.layers):
            h, present = layer(h, cos, sin, mask, None if past is None else past[i])
            if return_cache:
                presents.append(present)
        return self.norm(h), (presents if return_cache else None)

    def forward_latent_step(self, latent_vector: Tensor, past: KVCache,
                            past_mask: Tensor) -> Tuple[Tensor, KVCache, Tensor]:
        """Append one continuous vector; return its hidden state, cache, and extended mask."""
        if latent_vector.ndim != 3 or latent_vector.shape[1] != 1:
            raise ValueError("latent_vector must be (B, 1, hidden_size)")
        if past is None or past_mask is None:
            raise ValueError("latent step requires cache and past_mask")
        hidden, cache = self.forward(inputs_embeds=latent_vector, past=past,
                                     past_mask=past_mask, return_cache=True)
        new_mask = torch.cat((past_mask, torch.ones_like(past_mask[:, :1])), dim=1)
        return hidden, cache, new_mask

    @staticmethod
    def last_valid(hidden: Tensor, attention_mask: Tensor) -> Tensor:
        batch, seq = attention_mask.shape
        valid = attention_mask != 0
        index = torch.arange(seq).expand(batch, seq).masked_fill(~valid, -1).max(dim=1).values
        if (index < 0).any():
            raise ValueError("every sequence must contain at least one token")
        return hidden[torch.arange(batch), index]

    def tensor_bytes(self) -> int:
        seen, total = set(), 0
        for tensor in list(self.parameters()) + list(self.buffers()):
            if tensor.data_ptr() in seen:
                continue
            seen.add(tensor.data_ptr())
            total += tensor.numel() * tensor.element_size()
        return total


# --------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------

def snapshot_digest(snapshot: Path) -> str:
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors"):
        with open(Path(snapshot) / name, "rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _hf_key(prefix: str, name: str) -> str:
    return f"{prefix}{name}"


def load_dense_weights(trunk: ZeroTrunk, snapshot: Path) -> None:
    """Stream tensors from the local safetensors file into the trunk, one at a time."""
    from safetensors import safe_open
    path = Path(snapshot) / "model.safetensors"
    with safe_open(str(path), framework="pt") as store:
        keys = set(store.keys())
        prefix = "model." if "model.embed_tokens.weight" in keys else ""

        def take(name: str) -> Tensor:
            key = _hf_key(prefix, name)
            if key not in keys:
                raise KeyError(f"weight {key} missing from {path}")
            keys.discard(key)
            return store.get_tensor(key)

        trunk.embed_tokens.weight.copy_(take("embed_tokens.weight").to(torch.bfloat16))
        trunk.norm.weight.copy_(take("norm.weight").to(torch.float32))
        for i, layer in enumerate(trunk.layers):
            base = f"layers.{i}."
            layer.input_layernorm.weight.copy_(take(base + "input_layernorm.weight").to(torch.float32))
            layer.post_attention_layernorm.weight.copy_(
                take(base + "post_attention_layernorm.weight").to(torch.float32))
            for proj in ("q_proj", "k_proj", "v_proj"):
                getattr(layer.self_attn, proj).load_dense(take(f"{base}self_attn.{proj}.weight"),
                                                          take(f"{base}self_attn.{proj}.bias"))
            layer.self_attn.o_proj.load_dense(take(base + "self_attn.o_proj.weight"), None)
            for proj in ("gate_proj", "up_proj", "down_proj"):
                getattr(layer.mlp, proj).load_dense(take(f"{base}mlp.{proj}.weight"), None)
        leftover = [k for k in keys if not k.endswith("lm_head.weight")]
        if leftover:
            raise ValueError(f"unconsumed weights in checkpoint: {leftover[:5]}")


def build_int8_artifact(snapshot: Path, artifact: Path) -> dict:
    """Quantize once (offline) and write a self-describing safetensors artifact.

    safetensors is used so that :func:`load_trunk` can stream one tensor at a
    time; the process peak then stays near the resident size instead of
    holding a second mapped copy of the whole file.
    """
    from safetensors.torch import save_file
    cfg = TrunkConfig.from_snapshot(snapshot)
    trunk = ZeroTrunk(cfg, "int8")
    load_dense_weights(trunk, snapshot)
    metadata = {
        "version": str(INT8_ARTIFACT_VERSION),
        "family": ZERO_ENCODER_FAMILY,
        "scheme": "int8-weight-only-per-row-symmetric, bf16-embedding, fp32-compute",
        "source_snapshot": str(snapshot),
        "source_sha256": snapshot_digest(snapshot),
        "quantized_linears": str(sum(isinstance(m, Int8WeightOnlyLinear) for m in trunk.modules())),
    }
    artifact = Path(artifact)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in trunk.state_dict().items()}, str(artifact), metadata=metadata)
    return metadata


_SAFETENSORS_DTYPES = {"I8": torch.int8, "BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}


def _stream_safetensors_into(path: Path, state: dict) -> dict:
    """Copy every tensor of a safetensors file into ``state`` with plain reads.

    Deliberately avoids mmap: a mapped file keeps its pages resident until the
    handle closes, which doubles the process peak. Here the peak is the
    destination tensors plus one tensor-sized read buffer.
    """
    import struct
    with open(path, "rb") as stream:
        header_len = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_len).decode("utf-8"))
        metadata = dict(header.pop("__metadata__", {}) or {})
        if set(header) != set(state):
            raise ValueError("int8 artifact keys mismatch")
        base = 8 + header_len
        for key, entry in header.items():
            target = state[key]
            dtype = _SAFETENSORS_DTYPES.get(entry["dtype"])
            if dtype is None or dtype != target.dtype or tuple(entry["shape"]) != tuple(target.shape):
                raise ValueError(f"int8 artifact tensor {key} has the wrong shape/dtype")
            start, end = entry["data_offsets"]
            nbytes = end - start
            if nbytes != target.numel() * target.element_size():
                raise ValueError(f"int8 artifact tensor {key} has the wrong byte length")
            stream.seek(base + start)
            view = target.contiguous().view(torch.uint8) if target.dim() else target.view(1).view(torch.uint8)
            if not target.is_contiguous():
                raise ValueError("destination tensors must be contiguous")
            buffer = memoryview(view.numpy())
            read = stream.readinto(buffer)
            if read != nbytes:
                raise ValueError(f"short read for {key}")
    return metadata


def load_trunk(snapshot: Path, precision: str, int8_artifact: Optional[Path] = None
               ) -> Tuple[ZeroTrunk, dict]:
    """Materialize the trunk for ``precision``. INT8 needs a prebuilt artifact.

    The INT8 path never holds an FP32 copy: the module is allocated in its
    final int8/bf16 layout and the artifact is streamed in through mmap.
    """
    cfg = TrunkConfig.from_snapshot(snapshot)
    trunk = ZeroTrunk(cfg, precision)
    if precision == "int8":
        if int8_artifact is None:
            raise ValueError("precision='int8' needs an artifact from build_int8_artifact")
        state = trunk.state_dict()
        metadata = _stream_safetensors_into(Path(int8_artifact), state)
        if metadata.get("version") != str(INT8_ARTIFACT_VERSION) or metadata.get("family") != ZERO_ENCODER_FAMILY:
            raise ValueError("int8 artifact version/family mismatch")
        if metadata.get("source_sha256") != snapshot_digest(snapshot):
            raise ValueError("int8 artifact was built from a different backbone snapshot")
    else:
        load_dense_weights(trunk, snapshot)
        metadata = {"scheme": precision, "source_snapshot": str(snapshot),
                    "source_sha256": snapshot_digest(snapshot)}
    for tensor in list(trunk.parameters()) + list(trunk.buffers()):
        if tensor.device.type != "cpu":
            raise RuntimeError("Zero tensor left the cpu")
    trunk.eval()
    malloc_trim()
    return trunk, metadata
