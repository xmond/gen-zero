"""Offline structured initialization and distillation, never runtime teacher loading.

Slicing is an initialization heuristic, not a function-preserving compression or
proof of causal knowledge transfer. Unsupported architectures fail explicitly.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .zero_model import ZeroConfig, ZeroModel, load_zero_tokenizer


@dataclass(frozen=True)
class ConversionReport:
    checkpoint: str
    source_layers: tuple
    tensor_sha256: dict
    initialized_only: tuple = ('manifold_head.weight',)
    distilled: bool = False

    @property
    def selected_layers(self) -> tuple:
        return self.source_layers


class ZeroConverter:
    """Stream local safetensors; supports dense Qwen2 MHA/GQA, not hybrid Qwen3.5.

    Capacity labels (9B/70B) are not architecture contracts. Inspect config first.
    Future teachers with unsupported layouts need an explicit adapter or output
    distillation; silently discarding gated/delta/MoE components is prohibited.
    """
    def __init__(self, checkpoint):
        self.path = Path(checkpoint).expanduser().resolve()
        if not self.path.is_dir() or not (self.path / 'config.json').is_file():
            raise FileNotFoundError(f'No teacher checkpoint at {self.path}. Supply a local '
                                    'config.json + safetensors directory; use ZeroModel(ZeroConfig()) '
                                    'for explicitly untrained architecture validation.')
        self.source = json.loads((self.path / 'config.json').read_text())
        from safetensors import safe_open
        self.locations = {}
        for file in sorted(self.path.glob('*.safetensors')):
            with safe_open(file, framework='pt', device='cpu') as reader:
                for name in reader.keys():
                    if name in self.locations:
                        raise ValueError(f'Duplicate checkpoint tensor: {name}')
                    self.locations[name] = file
        if not self.locations:
            raise FileNotFoundError(f'No safetensors weights in {self.path}; no conversion performed')

    def _read(self, name):
        from safetensors import safe_open
        if name not in self.locations:
            raise ValueError(f'Missing required teacher tensor: {name}')
        with safe_open(self.locations[name], framework='pt', device='cpu') as reader:
            return reader.get_tensor(name)

    def convert(self, config: ZeroConfig, *, tokenizer_path, dtype=torch.bfloat16):
        src = self.source
        if config.sparse_moe:
            raise ValueError('Dynamical sparse MoE layers have no dense teacher slice to copy; '
                             'convert a sparse_moe=False config, then port weights separately')
        if src.get('model_type') != 'qwen2':
            raise ValueError(f"Unsupported teacher architecture {src.get('model_type')!r}; "
                             'Qwen3.5 hybrid attention and its different vocabulary cannot be sliced as Qwen2')
        if src.get('rope_scaling') or src.get('use_sliding_window', False):
            raise ValueError('Scaled RoPE/sliding-window teachers require an explicit conversion adapter')
        teacher_tok = load_zero_tokenizer(self.path, config)
        zero_tok = load_zero_tokenizer(tokenizer_path, config)
        if teacher_tok.backend_tokenizer.to_str() != zero_tok.backend_tokenizer.to_str():
            raise ValueError('Teacher and Zero tokenizer pipelines differ; logits KL requires identical token semantics')
        h, heads = src['hidden_size'], src['num_attention_heads']
        kv_heads = src.get('num_key_value_heads', heads)
        d = h // heads
        if h % heads or heads % kv_heads or d % 2:
            raise ValueError('Invalid teacher attention geometry')
        if (h < config.hidden_size or heads < config.num_attention_heads or d < config.head_dim
                or src['intermediate_size'] < config.intermediate_size
                or src['num_hidden_layers'] < config.num_hidden_layers):
            raise ValueError('Teacher must contain the requested structured student dimensions')
        if not config.attention_bias:
            raise ValueError('Qwen2 q/k/v biases require ZeroConfig(attention_bias=True)')
        if config.rope_theta != src.get('rope_theta', 10000.0) or config.rms_norm_eps != src['rms_norm_eps']:
            raise ValueError('Use teacher rope_theta and rms_norm_eps for initialization')
        selected = torch.linspace(0, src['num_hidden_layers']-1, config.num_hidden_layers).round().long().tolist()
        # Select complete head identities and paired rotary coordinates, not a
        # contiguous flattened prefix that mixes GQA heads or rotary halves.
        q_heads = torch.linspace(0, heads-1, config.num_attention_heads).round().long()
        coordinates = torch.cat((torch.arange(config.head_dim//2),
                                 torch.arange(config.head_dim//2)+d//2))
        q_rows = (q_heads[:, None]*d + coordinates).flatten()
        kv_rows = ((q_heads//(heads//kv_heads))[:, None]*d + coordinates).flatten()
        model = ZeroModel(config, dtype=dtype)
        evidence = {}
        def copy(destination, name, rows=None, cols=None):
            value = self._read(name)
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f'Nonfinite/nonfloating teacher tensor {name}')
            if rows is not None:
                value = value[rows]
            if cols is not None:
                value = value[:, cols]
            if value.shape != destination.shape:
                raise ValueError(f'{name}: selected shape {value.shape} != {destination.shape}')
            evidence[name] = hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            with torch.no_grad():
                converted = value.to(dtype=destination.dtype)
                if not torch.isfinite(converted).all():
                    raise ValueError(f'{name} overflows target dtype')
                destination.copy_(converted)
        residual = slice(0, config.hidden_size)
        intermediate = slice(0, config.intermediate_size)
        copy(model.embed_tokens.weight, 'model.embed_tokens.weight', slice(0, config.vocab_size), residual)
        copy(model.norm.weight, 'model.norm.weight', residual)
        for layer, source_index in zip(model.layers, selected):
            prefix = f'model.layers.{source_index}.'
            for name in ('input_layernorm', 'post_attention_layernorm'):
                copy(getattr(layer, name).weight, prefix+name+'.weight', residual)
            for name, rows in (('q_proj', q_rows), ('k_proj', kv_rows), ('v_proj', kv_rows)):
                projection = getattr(layer, name)
                copy(projection.weight, prefix+'self_attn.'+name+'.weight', rows, residual)
                copy(projection.bias, prefix+'self_attn.'+name+'.bias', rows)
            copy(layer.o_proj.weight, prefix+'self_attn.o_proj.weight', residual, q_rows)
            for name in ('gate_proj', 'up_proj'):
                copy(getattr(layer, name).weight, prefix+'mlp.'+name+'.weight', intermediate, residual)
            copy(layer.down_proj.weight, prefix+'mlp.down_proj.weight', residual, intermediate)
        return model, ConversionReport(str(self.path), tuple(selected), evidence)


def distillation_loss(student_logits, teacher_logits, labels, student_state,
                      teacher_state, student_counterfactual, teacher_counterfactual,
                      *, temperature=2.0, margin=0.2, ce_weight=1.0, kl_weight=1.0,
                      manifold_weight=1.0, counterfactual_weight=1.0):
    """Offline objective; caller supplies aligned, independently supervised states.

    Labels are already aligned prediction targets (-100 excludes padding). Teacher
    states must share one calibrated 64-D coordinate system, not random projections.
    Counterfactual examples must be real interventions with known changed outcomes.
    No teacher module or weights are captured by this function or ZeroModel.
    """
    import math
    weights = (ce_weight, kl_weight, manifold_weight, counterfactual_weight)
    if not math.isfinite(temperature) or temperature <= 0 or not 0 < margin <= 2:
        raise ValueError('Invalid temperature or cosine margin')
    if any(not math.isfinite(w) or w < 0 for w in weights) or not any(weights):
        raise ValueError('Loss weights must be finite, nonnegative and not all zero')
    if student_logits.ndim != 3 or student_logits.shape != teacher_logits.shape or labels.shape != student_logits.shape[:-1]:
        raise ValueError('Logits must be aligned [batch, sequence, vocabulary] with matching labels')
    expected = (student_logits.shape[0], 64)
    states = (student_state, teacher_state, student_counterfactual, teacher_counterfactual)
    for value in (*states, student_logits, teacher_logits):
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError('Distillation tensors must be finite floating-point values')
    if any(s.shape != expected for s in states):
        raise ValueError('All manifold states must have shape [batch, 64]')
    if any((s.float().norm(dim=-1) <= 1e-8).any() for s in states):
        raise ValueError('Cosine supervision requires nonzero states')
    valid = labels != -100
    if not valid.any():
        raise ValueError('At least one supervised token is required')
    sl, tl = student_logits[valid].float(), teacher_logits.detach()[valid].float()
    ce = F.cross_entropy(sl, labels[valid])
    kl = F.kl_div(F.log_softmax(sl/temperature, -1), F.softmax(tl/temperature, -1), reduction='batchmean')*temperature**2
    s, t, sc, tc = (x.float() for x in states)
    t, tc = t.detach(), tc.detach()
    cosine = lambda a, b: F.cosine_similarity(a, b, dim=-1)
    if (1-cosine(t, tc) < margin).any():
        raise ValueError('Teacher intervention pairs are not separated by the requested margin')
    manifold = ((1-cosine(s, t)) + (1-cosine(sc, tc))).mean()/2
    contrastive = (F.relu(margin-cosine(s, t)+cosine(s, tc)) +
                   F.relu(margin-cosine(sc, tc)+cosine(sc, t))).mean()/2
    total = ce_weight*ce + kl_weight*kl + manifold_weight*manifold + counterfactual_weight*contrastive
    return {'loss': total, 'ce': ce, 'kl': kl, 'manifold': manifold, 'counterfactual': contrastive}
