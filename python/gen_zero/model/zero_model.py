"""Zero: standalone causal Transformer. Initialization is NOT learned causality.

The byte budget covers parameters/buffers, not process RSS or training memory.
No teacher, tokenizer, KV cache, or external model is retained by ZeroModel.
"""
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ZeroConfig:
    vocab_size: int = 151643
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 2816
    attention_bias: bool = False
    manifold_dim: int = 64
    max_sequence_length: int = 512
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    # Phase 2.3 dynamical sparse MoE (docs/zero/05 dimension 3): opt-in, off by default so every
    # existing dense preset/converter/test keeps its exact parameter count unchanged.
    sparse_moe: bool = False
    sparse_layer_start: int = 16
    num_experts: int = 8
    experts_per_token: int = 1
    expert_intermediate_size: int | None = None
    # Factorized Embedding Parameterization (ALBERT-style): vocab -> embedding_dim -> hidden_size,
    # opt-in so every existing dense preset/converter/test keeps its exact parameter count unchanged.
    factorized_embedding: bool = False
    embedding_dim: int = 128

    def __post_init__(self):
        for name in ('vocab_size', 'hidden_size', 'num_hidden_layers',
                     'num_attention_heads', 'intermediate_size', 'max_sequence_length'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.factorized_embedding:
            if not isinstance(self.embedding_dim, int) or isinstance(self.embedding_dim, bool) \
                    or self.embedding_dim <= 0:
                raise ValueError('embedding_dim must be a positive integer')
            if self.embedding_dim >= self.hidden_size:
                raise ValueError('embedding_dim must be smaller than hidden_size to save parameters')
        if self.manifold_dim != 64:
            raise ValueError('Zero causal state must have 64 dimensions')
        if self.hidden_size % self.num_attention_heads or self.head_dim % 2:
            raise ValueError('attention heads must divide hidden size with even head dimension')
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError('rope_theta must be finite and positive')
        if not math.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0:
            raise ValueError('rms_norm_eps must be finite and positive')
        if self.sparse_moe:
            if not isinstance(self.num_experts, int) or isinstance(self.num_experts, bool) or self.num_experts < 1:
                raise ValueError('num_experts must be a positive integer')
            if not isinstance(self.experts_per_token, int) or isinstance(self.experts_per_token, bool) \
                    or not 1 <= self.experts_per_token <= self.num_experts:
                raise ValueError('experts_per_token must be between 1 and num_experts')
            if not isinstance(self.sparse_layer_start, int) or isinstance(self.sparse_layer_start, bool) \
                    or not 1 <= self.sparse_layer_start <= self.num_hidden_layers:
                raise ValueError('sparse_layer_start must be a 1-indexed layer number within range')
            if self.expert_intermediate_size is not None and (
                    not isinstance(self.expert_intermediate_size, int)
                    or isinstance(self.expert_intermediate_size, bool)
                    or self.expert_intermediate_size <= 0):
                raise ValueError('expert_intermediate_size must be a positive integer')
            if self.effective_expert_intermediate_size <= 0:
                raise ValueError('intermediate_size // 4 must be positive to derive an expert width')

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def effective_expert_intermediate_size(self):
        return self.expert_intermediate_size or (self.intermediate_size // 4)

    @property
    def embedding_table_cost(self):
        """Stored parameters in the vocabulary embedding: factorized (vocab*d + d*hidden)
        or the plain vocab*hidden table. Active == total since every forward touches all of it
        (row lookup selects one row per token, but the projection matrix is dense either way)."""
        if not self.factorized_embedding:
            return self.vocab_size * self.hidden_size
        return self.vocab_size * self.embedding_dim + self.embedding_dim * self.hidden_size

    @property
    def sparse_layer_indices(self):
        """0-indexed layers using the sparse operator pool; empty unless sparse_moe is set.
        sparse_layer_start is 1-indexed inclusive, per docs/zero/05 ("layers 16-24"): the default 16
        on a 24-layer model marks layers 16-24 (1-indexed) i.e. 0-indexed 15-23, 9 layers."""
        if not self.sparse_moe:
            return frozenset()
        return frozenset(i for i in range(self.num_hidden_layers) if i >= self.sparse_layer_start - 1)

    def _layer_mlp_costs(self):
        """(dense_layer_mlp, sparse_layer_mlp_total, sparse_layer_mlp_active) parameter counts."""
        h, i = self.hidden_size, self.intermediate_size
        dense = 3*h*i
        if not self.sparse_moe:
            return dense, dense, dense
        e = self.effective_expert_intermediate_size
        router = h*self.num_experts
        expert = 3*h*e
        return dense, router + self.num_experts*expert, router + self.experts_per_token*expert

    def parameter_count(self):
        """Total stored parameters (every expert resident, matches BF16 file size on disk)."""
        h = self.hidden_size
        attn_norm = 4*h*h + 2*h + (3*h if self.attention_bias else 0)
        dense_mlp, sparse_total_mlp, _ = self._layer_mlp_costs()
        n_sparse = len(self.sparse_layer_indices)
        n_dense = self.num_hidden_layers - n_sparse
        backbone = n_dense*(attn_norm + dense_mlp) + n_sparse*(attn_norm + sparse_total_mlp)
        return self.embedding_table_cost + backbone + h + h*64

    def active_parameter_count(self):
        """Parameters touched by a single token's forward pass: dense layers in full, sparse
        layers only pay for their router plus the experts_per_token experts actually routed to.
        Embedding and the manifold head are counted in full (matches parameter_count() when
        sparse_moe is off, so the two are directly comparable)."""
        h = self.hidden_size
        attn_norm = 4*h*h + 2*h + (3*h if self.attention_bias else 0)
        dense_mlp, _, sparse_active_mlp = self._layer_mlp_costs()
        n_sparse = len(self.sparse_layer_indices)
        n_dense = self.num_hidden_layers - n_sparse
        backbone = n_dense*(attn_norm + dense_mlp) + n_sparse*(attn_norm + sparse_active_mlp)
        return self.embedding_table_cost + backbone + h + h*64

    def storage_bytes(self, dtype=torch.bfloat16):
        return self.parameter_count() * torch.empty((), dtype=dtype).element_size()

    def active_storage_bytes(self, dtype=torch.bfloat16):
        return self.active_parameter_count() * torch.empty((), dtype=dtype).element_size()

    @classmethod
    def zero_lite(cls, **overrides):
        """~460M dense params / ~927MB BF16; edge / single-core CPU. The library default."""
        return cls(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
                    intermediate_size=2816, **overrides)

    @classmethod
    def zero_plus(cls, **overrides):
        """~1.0B-1.2B dense params / ~2.1-2.4GB BF16; production server CPU."""
        return cls(hidden_size=1536, num_hidden_layers=28, num_attention_heads=24,
                    intermediate_size=4096, **overrides)

    @classmethod
    def zero_max(cls, **overrides):
        """~1.8B-2.2B dense params / ~3.8-4.4GB BF16; edge industrial controller, offline hub."""
        return cls(hidden_size=2048, num_hidden_layers=32, num_attention_heads=32,
                    intermediate_size=5632, **overrides)

    @classmethod
    def zero_compact_220m(cls, **overrides):
        """~211M active / ~347M stored params / ~694MB BF16; breaks the ~256M active-parameter
        floor that a full (non-factorized) embedding table imposes on this architecture family
        (see test_zero_lite_sparse_moe_active_parameter_reduction's own finding: embedding
        155.3M + dense attention 100.7M alone = 256M, already over a 220M-class active budget
        before any MLP compute). Factorized Embedding Parameterization (vocab -> 128 -> hidden)
        cuts the embedding from 151643*1024=155.3M to 151643*128+128*1024=~19.5M stored/active,
        which is what makes an active budget under 230M reachable at all.

        intermediate_size is tuned to 1536 (not the naive 2816 dense width) so the 230M active /
        350M stored budgets are actually met, not just asserted: at intermediate_size=2816 with
        expert_intermediate_size=704 the honest active count is ~269.6M (verified via
        active_parameter_count(), not estimated), still over budget by ~40M. Shrinking dense MLP
        width is the lever this preset's own docs explicitly permit ("set compact factorization for
        the first 16 MLP layers or specify expert_intermediate_size"); expert_intermediate_size is kept at the doc's own 704
        (intermediate_size//4 of the *original* 2816) since only the dense layers were oversized.

        Exact counts (computed by active_parameter_count()/parameter_count(), not estimated):
        embedding 19,541,376 + attention/norms 100,712,448 (24 layers) + 15 dense MLPs
        (3*1024*1536 each) 70,778,880 + 9 sparse layers (router 1024*8 + top-1 of 8 experts at
        3*1024*704 each) 19,537,920 + tail (norm + manifold head) 66,560
        = active_parameter_count() 210,637,184 (<= 230,000,000).
        Stored adds all 8 resident experts per sparse layer instead of 1:
        parameter_count() 346,886,528 (<= 350,000,000), storage_bytes() 693,773,056 (<= 1.0GB)."""
        return cls(vocab_size=151643, hidden_size=1024, num_hidden_layers=24,
                    intermediate_size=1536, num_attention_heads=16,
                    factorized_embedding=True, embedding_dim=128,
                    sparse_moe=True, sparse_layer_start=16, num_experts=8,
                    experts_per_token=1, expert_intermediate_size=704, **overrides)


class RMSNorm(nn.Module):
    def __init__(self, size, eps, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, **kwargs))
        self.eps = eps

    def forward(self, x):
        value = x.float()
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


class ZeroBlock(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.input_layernorm = RMSNorm(h, config.rms_norm_eps, **kwargs)
        self.post_attention_layernorm = RMSNorm(h, config.rms_norm_eps, **kwargs)
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            setattr(self, name, nn.Linear(h, h, bias=config.attention_bias and name != "o_proj", **kwargs))
        self._build_mlp(config, **kwargs)

    def _build_mlp(self, config, **kwargs):
        h, i = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False, **kwargs)
        self.up_proj = nn.Linear(h, i, bias=False, **kwargs)
        self.down_proj = nn.Linear(i, h, bias=False, **kwargs)

    def _mlp(self, z):
        return self.down_proj(F.silu(self.gate_proj(z))*self.up_proj(z))

    def forward(self, x, allowed, cos, sin):
        b, s, _ = x.shape
        c = self.config
        z = self.input_layernorm(x)
        def split(projection):
            return projection(z).view(b, s, c.num_attention_heads, c.head_dim).transpose(1, 2)
        def rotate(v):
            first, second = v.chunk(2, dim=-1)
            return v*cos + torch.cat((-second, first), dim=-1)*sin
        q, k, v = rotate(split(self.q_proj)), rotate(split(self.k_proj)), split(self.v_proj)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        x = x + self.o_proj(attn.transpose(1, 2).reshape(b, s, c.hidden_size))
        z = self.post_attention_layernorm(x)
        return x + self._mlp(z)


class SwiGLUExpert(nn.Module):
    """One compact dynamical branch of the sparse operator pool."""
    def __init__(self, hidden_size, expert_intermediate_size, **kwargs):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, expert_intermediate_size, bias=False, **kwargs)
        self.up_proj = nn.Linear(hidden_size, expert_intermediate_size, bias=False, **kwargs)
        self.down_proj = nn.Linear(expert_intermediate_size, hidden_size, bias=False, **kwargs)

    def forward(self, z):
        return self.down_proj(F.silu(self.gate_proj(z))*self.up_proj(z))


class DynamicalSparseOperatorPool(nn.Module):
    """Top-k routed pool of compact SwiGLU experts (docs/zero/05 dimension 3, Phase 2.3).

    Routing softmaxes over ALL experts before taking top-k, so the selected gate weight is the
    expert's true softmax probability (not a constant 1 for k=1) and the router keeps receiving
    gradient even from tokens that only ever pick one expert. Dispatch is by row-gather per
    expert: an expert that receives no token in a given forward pass is never called, so it gets
    no gradient and contributes nothing to that pass's compute -- this is what makes the "active
    parameters" count real rather than a label on a densely-computed-then-masked pool.
    """
    def __init__(self, config, **kwargs):
        super().__init__()
        h = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.experts_per_token
        self.router = nn.Linear(h, self.num_experts, bias=False, **kwargs)
        self.experts = nn.ModuleList([
            SwiGLUExpert(h, config.effective_expert_intermediate_size, **kwargs)
            for _ in range(self.num_experts)
        ])

    def forward(self, z):
        b, s, h = z.shape
        flat = z.reshape(-1, h)
        logits = self.router(flat)
        probs = logits.float().softmax(-1)
        top_probs, top_idx = probs.topk(self.top_k, dim=-1)
        top_probs = top_probs.to(flat.dtype)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            row, slot = (top_idx == e).nonzero(as_tuple=True)
            if row.numel() == 0:
                continue
            out.index_add_(0, row, top_probs[row, slot, None]*expert(flat[row]))
        return out.view(b, s, h)

    def routed_experts(self, z):
        """Top-k expert indices per token, for routing tests -- no grad, diagnostic only."""
        with torch.no_grad():
            b, s, h = z.shape
            probs = self.router(z.reshape(-1, h)).float().softmax(-1)
            return probs.topk(self.top_k, dim=-1).indices.view(b, s, self.top_k)


class DynamicalSparseZeroBlock(ZeroBlock):
    """ZeroBlock whose MLP is replaced by the dynamical sparse operator pool."""
    def _build_mlp(self, config, **kwargs):
        self.mlp_pool = DynamicalSparseOperatorPool(config, **kwargs)

    def _mlp(self, z):
        return self.mlp_pool(z)


class ZeroModel(nn.Module):
    def __init__(self, config=None, *, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.config = config or ZeroConfig()
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError('Zero requires a floating-point parameter dtype')
        c = self.config
        kwargs = dict(device=device, dtype=dtype)
        if c.factorized_embedding:
            self.embed_tokens = nn.Embedding(c.vocab_size, c.embedding_dim, **kwargs)
            self.embed_proj = nn.Linear(c.embedding_dim, c.hidden_size, bias=False, **kwargs)
        else:
            self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size, **kwargs)
            self.embed_proj = None
        sparse = c.sparse_layer_indices
        self.layers = nn.ModuleList([
            DynamicalSparseZeroBlock(c, **kwargs) if i in sparse else ZeroBlock(c, **kwargs)
            for i in range(c.num_hidden_layers)
        ])
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps, **kwargs)
        self.manifold_head = nn.Linear(c.hidden_size, 64, bias=False, **kwargs)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def storage_bytes(self):
        return sum(t.numel()*t.element_size() for t in self.parameters()) + sum(
            t.numel()*t.element_size() for t in self.buffers())

    def hidden_states(self, input_ids: Tensor, attention_mask=None):
        c = self.config
        if input_ids.ndim != 2 or input_ids.dtype != torch.long or input_ids.shape[0] == 0:
            raise ValueError('input_ids must be nonempty [batch, sequence] int64')
        b, s = input_ids.shape
        if not 0 < s <= c.max_sequence_length:
            raise ValueError('sequence length exceeds the configured runtime bound')
        if (input_ids < 0).any() or (input_ids >= c.vocab_size).any():
            raise ValueError('token ID outside Zero vocabulary; verify tokenizer alignment')
        mask = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask
        if mask.shape != input_ids.shape or mask.device != input_ids.device:
            raise ValueError('mask must match input shape and device')
        if not ((mask == 0) | (mask == 1)).all() or not mask.bool().any(-1).all():
            raise ValueError('mask must be binary with a valid token in every row')
        mask = mask.bool()
        positions = (mask.long().cumsum(-1)-1).clamp_min(0)
        frequency = c.rope_theta ** (-torch.arange(0, c.head_dim, 2, device=input_ids.device).float()/c.head_dim)
        angles = positions.float()[..., None] * frequency
        angles = torch.cat((angles, angles), -1)[:, None]
        x = self.embed_tokens(input_ids)
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        causal = torch.ones(s, s, device=x.device, dtype=torch.bool).tril()
        allowed = causal[None, None] & mask[:, None, None, :]
        for layer in self.layers:
            x = layer(x, allowed, cos, sin)
        return self.norm(x), mask

    def forward(self, input_ids, attention_mask=None):
        hidden, mask = self.hidden_states(input_ids, attention_mask)
        last = torch.arange(hidden.shape[1], device=hidden.device).expand_as(mask).masked_fill(~mask, -1).max(-1).values
        return self.manifold_head(hidden[torch.arange(hidden.shape[0], device=hidden.device), last])

    def token_logits(self, input_ids, attention_mask=None):
        """Offline training only: tied embedding avoids a second vocabulary matrix."""
        if self.embed_proj is not None:
            raise ValueError('token_logits is not defined for factorized_embedding configs: '
                              'embed_tokens.weight is [vocab, embedding_dim], not [vocab, hidden], '
                              'so it cannot tie against hidden-size logits without a second matrix')
        hidden, _ = self.hidden_states(input_ids, attention_mask)
        return F.linear(hidden, self.embed_tokens.weight)


def load_zero_tokenizer(path, config):
    """Load local native tokenizer without remapping, dropping or aliasing token IDs."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    ids = set(tokenizer.get_vocab().values())
    if ids != set(range(config.vocab_size)):
        raise ValueError(f'Native tokenizer has {len(ids)} IDs (max={max(ids)}), '
                         f'Zero has {config.vocab_size} rows; exact alignment required')
    return tokenizer
