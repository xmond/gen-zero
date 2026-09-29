"""Gen-Zero Layer 1: Policy + Value Dual-Head Decision Network.

Features:
1. DeepSetAttentionHead: Permutation-Equivariant candidate self-attention & state cross-attention.
2. GlobalStateValueHead: Predicts scalar V(s) in [-1, 1] for AlphaZero MCTS.
3. AbstainModule: Active abstain slot (I-09) to avoid forced hazardous actions.
4. Safe import fallback for universal portability.
"""

import math
import dataclasses
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = object
    F = None
    HAS_TORCH = False



def _canonicalize_state_obj(x: Any) -> Any:
    """Recursively converts structured state data (including PyTorch Tensors and NumPy ndarrays)
    into deterministic, hashable primitives without string ellipsis truncation.
    """
    import hashlib
    if x is None:
        return None
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (int, bool)):
        return x
    if isinstance(x, float):
        return x

    # PyTorch Tensor: convert to contiguous CPU array and hash exact bytes
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        try:
            arr = x.detach().cpu().numpy()
            arr = np.ascontiguousarray(arr)
            h = hashlib.sha256(arr.tobytes()).hexdigest()
            return f"arr_{arr.dtype}_{arr.shape}_{h}"
        except Exception:
            pass

    # NumPy ndarray: hash exact raw bytes for numerical types; recursively traverse object arrays
    if hasattr(x, "tobytes") and hasattr(x, "shape") and hasattr(x, "dtype"):
        if getattr(x, "dtype", None) == object:
            return [_canonicalize_state_obj(v) for v in x.flat]
        arr = np.ascontiguousarray(x)
        h = hashlib.sha256(arr.tobytes()).hexdigest()
        return f"arr_{arr.dtype}_{arr.shape}_{h}"

    if isinstance(x, dict):
        # Unwrap single-key container wrappers like {"state": ...} or {"state_repr": ...}
        if len(x) == 1 and ("state" in x or "state_repr" in x):
            inner = x.get("state", x.get("state_repr"))
            return _canonicalize_state_obj(inner)
        return {str(k): _canonicalize_state_obj(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [_canonicalize_state_obj(v) for v in x]

    return str(x)


def normalize_state_repr(state: Any) -> str:
    """Recursively normalizes any state representation (string, dict, wrapper, numeric sequence, ndarray, or Tensor)
    into a deterministic, canonical string representation that preserves all structured fields without truncation.
    """
    import json
    if state is None:
        return ""
    if isinstance(state, str):
        return state.strip()

    canonical = _canonicalize_state_obj(state)
    if isinstance(canonical, str):
        return canonical

    try:
        if isinstance(canonical, dict):
            return json.dumps(canonical, sort_keys=True, separators=(',', ':'), ensure_ascii=False, default=str)
        elif isinstance(canonical, (list, tuple)):
            return json.dumps(canonical, separators=(',', ':'), ensure_ascii=False, default=str)
        return str(canonical)
    except Exception:
        return str(canonical)


def encode_leaf_tokens(state: Any, candidate: str, candidate_desc: Optional[str] = None, vocab_size: int = 19999) -> List[int]:
    """Unified token encoder ensuring 100% representation equivalence across train, eval, and reflex."""
    import hashlib
    s_str = normalize_state_repr(state)
    desc_str = f"_{candidate_desc}" if candidate_desc else ""
    raw = f"{s_str}_{candidate}{desc_str}"
    h = int(hashlib.md5(raw.encode("utf-8")).hexdigest()[:8], 16) % vocab_size
    return [101, h, 102]


def compute_normalized_attention_entropy(
    attn_weights: Any,
    eps: float = 1e-12
) -> Any:
    """Computes normalized Shannon attention entropy H_norm(a) = -sum(A_{a,j} * ln(A_{a,j} + eps)) / ln(L).

    Supports:
    - PyTorch Tensor of shape (..., L)
    - NumPy ndarray of shape (..., L)
    - Python float, list, or nested list

    Properties:
    - If L <= 1: returns 0.0 (no diffusion possible)
    - Boundary: strictly clamped within [0.0, 1.0]
    - Concentrated distribution (one-hot) -> 0.0
    - Perfectly uniform distribution (1/L, ..., 1/L) -> 1.0
    """
    if attn_weights is None:
        return 0.0

    if HAS_TORCH and isinstance(attn_weights, torch.Tensor):
        if attn_weights.numel() == 0:
            return torch.zeros(attn_weights.shape[:-1], dtype=attn_weights.dtype, device=attn_weights.device)
        l = attn_weights.shape[-1]
        if l <= 1:
            return torch.zeros(attn_weights.shape[:-1], dtype=attn_weights.dtype, device=attn_weights.device)
        p = attn_weights.clamp(min=0.0)
        p = p / p.sum(dim=-1, keepdim=True).clamp(min=eps)
        ent = -(p * (p + eps).log()).sum(dim=-1) / math.log(l)
        return ent.clamp(0.0, 1.0)

    arr = np.asarray(attn_weights, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    l = arr.shape[-1]
    if l <= 1:
        if arr.ndim == 1:
            return 0.0
        return np.zeros(arr.shape[:-1], dtype=np.float32)

    p = np.clip(arr, 0.0, None)
    denom = np.sum(p, axis=-1, keepdims=True)
    denom[denom <= 0] = 1.0
    p = p / denom
    ent = -np.sum(p * np.log(p + eps), axis=-1) / math.log(l)
    ent = np.clip(ent, 0.0, 1.0)
    if ent.ndim == 0:
        return float(ent)
    return ent


@dataclasses.dataclass
class CredibilityVerdict:
    confidence: float
    attention_entropy: float
    status: str              # 'PASS', 'CIRCUIT_BREAKER_BLIND_CONFIDENCE', 'ABSTAIN_UNCERTAIN', 'FALLBACK_LOW_CONFIDENCE'
    passed: bool
    is_blind_confidence: bool
    recommended_action: str  # 'execute', 'abstain', 'replan'
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": round(self.confidence, 4),
            "attention_entropy": round(self.attention_entropy, 4),
            "status": self.status,
            "passed": self.passed,
            "is_blind_confidence": self.is_blind_confidence,
            "recommended_action": self.recommended_action,
            "explanation": self.explanation,
        }


class TwoDimensionalCredibilityGate:
    """Two-dimensional decision credibility defense matrix: Confidence x Attention Entropy."""

    def __init__(self, conf_threshold: float = 0.60, entropy_threshold: float = 0.85):
        self.conf_threshold = conf_threshold
        self.entropy_threshold = entropy_threshold

    def evaluate(self, confidence: float, attention_entropy: float) -> CredibilityVerdict:
        conf = float(max(0.0, min(1.0, confidence)))
        ent = float(max(0.0, min(1.0, attention_entropy)))

        is_high_conf = conf >= self.conf_threshold
        is_high_ent = ent > self.entropy_threshold

        if is_high_conf and not is_high_ent:
            return CredibilityVerdict(
                confidence=conf,
                attention_entropy=ent,
                status="PASS",
                passed=True,
                is_blind_confidence=False,
                recommended_action="execute",
                explanation=f"Valid sharp attention (H={ent:.3f}) and confident decision (C={conf:.3f})."
            )
        elif is_high_conf and is_high_ent:
            # Blind confidence / False prosperity
            return CredibilityVerdict(
                confidence=conf,
                attention_entropy=ent,
                status="CIRCUIT_BREAKER_BLIND_CONFIDENCE",
                passed=False,
                is_blind_confidence=True,
                recommended_action="abstain",
                explanation=f"Circuit Breaker Triggered: High confidence (C={conf:.3f}) with diffuse attention (H={ent:.3f} > {self.entropy_threshold:.2f})."
            )
        elif not is_high_conf and is_high_ent:
            return CredibilityVerdict(
                confidence=conf,
                attention_entropy=ent,
                status="ABSTAIN_UNCERTAIN",
                passed=False,
                is_blind_confidence=False,
                recommended_action="replan",
                explanation=f"Uncertain state: Low confidence (C={conf:.3f}) and diffuse attention (H={ent:.3f}). Degrade to System 2 replanning."
            )
        else:
            return CredibilityVerdict(
                confidence=conf,
                attention_entropy=ent,
                status="FALLBACK_LOW_CONFIDENCE",
                passed=False,
                is_blind_confidence=False,
                recommended_action="abstain",
                explanation=f"Low confidence (C={conf:.3f}) with focused attention (H={ent:.3f})."
            )


if HAS_TORCH:

    class GlobalStateValueHead(nn.Module):
        """Global state valuation head V(s) in [-1, 1]."""
        def __init__(self, hidden_dim: int, mlp_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, mlp_dim),
                nn.LayerNorm(mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Tanh()
            )
            # Small weight initialization around 0 for neutral prior
            nn.init.normal_(self.net[-2].weight, std=0.01)
            nn.init.zeros_(self.net[-2].bias)

        def forward(self, state_repr: torch.Tensor) -> torch.Tensor:
            return self.net(state_repr).squeeze(-1)

    class DeepSetAttentionBlock(nn.Module):
        def __init__(self, embed_dim: int = 128, num_heads: int = 4, context_dim: Optional[int] = None, dropout: float = 0.0):
            super().__init__()
            self.ln_self = nn.LayerNorm(embed_dim)
            self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

            self.has_cross = context_dim is not None
            if self.has_cross:
                self.ln_cross = nn.LayerNorm(embed_dim)
                self.context_proj = nn.Linear(context_dim, embed_dim)
                self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

            self.ln_ffn = nn.LayerNorm(embed_dim)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout)
            )

        def forward(
            self,
            u: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            context: Optional[torch.Tensor] = None,
            return_attention: bool = False
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            u_norm = self.ln_self(u)
            self_out, self_attn = self.self_attn(u_norm, u_norm, u_norm, key_padding_mask=key_padding_mask, need_weights=return_attention)
            u = u + self_out

            cross_attn = None
            if self.has_cross and context is not None:
                c = self.context_proj(context)
                u_norm = self.ln_cross(u)
                cross_out, cross_attn = self.cross_attn(u_norm, c, c, need_weights=return_attention)
                u = u + cross_out

            u = u + self.ffn(self.ln_ffn(u))
            attn_matrix = cross_attn if (self.has_cross and cross_attn is not None and cross_attn.shape[-1] > 1) else self_attn
            return u, attn_matrix


    class DeepSetAttentionHead(nn.Module):
        """Permutation-equivariant candidate action evaluation head."""
        def __init__(self, hidden_dim: int, embed_dim: int = 128, num_layers: int = 2, num_heads: int = 4, context_dim: Optional[int] = None, dropout: float = 0.0):
            super().__init__()
            self.project = nn.Linear(hidden_dim + 1, embed_dim)
            self.blocks = nn.ModuleList([
                DeepSetAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, context_dim=context_dim, dropout=dropout)
                for _ in range(num_layers)
            ])
            self.ln_out = nn.LayerNorm(embed_dim)
            self.output_proj = nn.Linear(embed_dim, 1)

            # Zero-init ensures initial behavior equals standard scalar head without disruption
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

        def forward(
            self,
            h: torch.Tensor,
            valid: torch.Tensor,
            context: Optional[torch.Tensor] = None,
            return_attention: bool = False
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            kmax = h.shape[1]
            p_dtype = self.project.weight.dtype
            log_k = valid.sum(-1, keepdim=True).float().clamp(min=1.0).log().unsqueeze(-1).expand(-1, kmax, 1)
            u = self.project(torch.cat([h.to(p_dtype), log_k.to(p_dtype)], dim=-1))
            if context is not None:
                context = context.to(p_dtype)

            key_padding_mask = ~valid
            last_attn = None
            for block in self.blocks:
                u, last_attn = block(u, key_padding_mask=key_padding_mask, context=context, return_attention=return_attention)

            delta = self.output_proj(self.ln_out(u)).squeeze(-1)
            if return_attention:
                if last_attn is not None:
                    entropy = compute_normalized_attention_entropy(last_attn)
                else:
                    entropy = torch.zeros((h.shape[0], kmax), device=h.device, dtype=torch.float32)
                return delta, entropy
            return delta

    class AbstainModule(nn.Module):
        """Active Abstain / No-op slot (I-09)."""
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.abstain_token = nn.Parameter(torch.randn(hidden_dim) * 0.02)

        def append_abstain(self, h: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            bsz = h.shape[0]
            abstain_expanded = self.abstain_token.unsqueeze(0).unsqueeze(1).expand(bsz, 1, -1)
            h_with_abstain = torch.cat([h, abstain_expanded], dim=1)
            valid_with_abstain = torch.cat([valid, torch.ones((bsz, 1), dtype=torch.bool, device=valid.device)], dim=1)
            return h_with_abstain, valid_with_abstain

    class GenZeroDualHeadModel(nn.Module):
        """Unified Gen-Zero Dual-Head Decision Model."""
        def __init__(
            self,
            backbone: Any = None,
            hidden_dim: int = 2048,
            embed_dim: int = 128,
            num_layers: int = 2,
            num_heads: int = 4,
            use_value_head: bool = True,
            enable_abstain: bool = True
        ):
            super().__init__()
            self.backbone = backbone
            self.hidden_dim = hidden_dim
            self.norm = nn.LayerNorm(hidden_dim)
            self.scalar = nn.Linear(hidden_dim, 1)
            nn.init.normal_(self.scalar.weight, std=0.02)
            nn.init.zeros_(self.scalar.bias)

            self.deep_set_attention = DeepSetAttentionHead(
                hidden_dim=hidden_dim,
                embed_dim=embed_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                context_dim=hidden_dim
            )

            self.use_value_head = use_value_head
            if use_value_head:
                self.value_head = GlobalStateValueHead(hidden_dim=hidden_dim)

            self.enable_abstain = enable_abstain
            if enable_abstain:
                self.abstain_module = AbstainModule(hidden_dim=hidden_dim)

            # Compact learned embedding table for genuine input-dependent representations when no heavy backbone is loaded
            self.token_embedding = nn.Embedding(20000, hidden_dim)
            nn.init.normal_(self.token_embedding.weight, std=0.02)

        def forward(
            self,
            examples: List[dict],
            pad_token: int,
            return_value: bool = False,
            return_abstain: bool = False,
            return_attention_entropy: bool = False
        ) -> Any:
            device = self.scalar.weight.device
            if not examples:
                empty_l = torch.empty((0, 0), device=device)
                empty_v = torch.empty((0, 0), dtype=torch.bool, device=device)
                empty_e = torch.empty((0, 0), device=device)
                if return_value and return_attention_entropy:
                    return empty_l, empty_v, torch.empty((0, 1), device=device), empty_e
                elif return_value:
                    return empty_l, empty_v, torch.empty((0, 1), device=device)
                elif return_attention_entropy:
                    return empty_l, empty_v, empty_e
                return empty_l, empty_v

            paths = [ids for ex in examples for ids in ex.get('leaf_tokens', [])]
            if not paths:
                kmax = max((len(ex.get('candidate_ids', [])) for ex in examples), default=0)
                empty_l = torch.zeros((len(examples), kmax), device=device)
                empty_v = torch.zeros((len(examples), kmax), dtype=torch.bool, device=device)
                empty_e = torch.zeros((len(examples), kmax), device=device)
                if return_value and return_attention_entropy:
                    return empty_l, empty_v, torch.zeros((len(examples), 1), device=device), empty_e
                elif return_value:
                    return empty_l, empty_v, torch.zeros((len(examples), 1), device=device)
                elif return_attention_entropy:
                    return empty_l, empty_v, empty_e
                return empty_l, empty_v

            lengths = torch.tensor([len(ids) for ids in paths], device=device)
            width = int(lengths.max())
            tokens = torch.full((len(paths), width), pad_token, dtype=torch.long, device=device)
            for i, ids in enumerate(paths):
                tokens[i, :len(ids)] = torch.tensor(ids, device=device)
            attention = torch.arange(width, device=device)[None, :] < lengths[:, None]
            
            if self.backbone is not None:
                hidden = self.backbone(input_ids=tokens, attention_mask=attention, use_cache=False).last_hidden_state
                leaves = hidden[torch.arange(len(paths), device=device), lengths - 1]
            else:
                # Genuine learned token representation without requiring gigabyte-scale external backbone
                tok_embs = self.token_embedding(tokens.clamp(0, 19999))  # [len(paths), width, hidden_dim]
                mask = attention.unsqueeze(-1).float()
                leaves = (tok_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

            kmax = max(len(ex['candidate_ids']) for ex in examples)
            h = leaves.new_zeros((len(examples), kmax, leaves.shape[-1]))
            valid = torch.zeros((len(examples), kmax), dtype=torch.bool, device=device)
            offset = 0
            for i, ex in enumerate(examples):
                n = len(ex['leaf_tokens'])
                h[i, :n] = leaves[offset:offset+n]
                valid[i, :len(ex['candidate_ids'])] = True
                offset += n

            # Global context token: mean pool across valid candidate paths
            state_repr = (h * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp(min=1.0)
            norm_dtype = self.norm.weight.dtype
            state_repr = self.norm(state_repr.to(norm_dtype)).to(self.scalar.weight.dtype)

            values = self.value_head(state_repr) if self.use_value_head else None

            # Abstain slot if requested
            if self.enable_abstain and return_abstain:
                h, valid = self.abstain_module.append_abstain(h, valid)
                kmax += 1

            h_norm = self.norm(h.to(norm_dtype)).to(self.scalar.weight.dtype)
            z = self.scalar(h_norm).squeeze(-1).float()

            entropy = torch.zeros((len(examples), kmax), device=device, dtype=torch.float32)
            choice = torch.tensor([i for i, ex in enumerate(examples) if ex['type'] == 'choice'], device=device)
            if len(choice):
                if return_attention_entropy:
                    delta, choice_entropy = self.deep_set_attention(
                        h=h_norm[choice],
                        valid=valid[choice],
                        context=state_repr[choice].unsqueeze(1),
                        return_attention=True
                    )
                    z = z.index_add(0, choice, delta.float())
                    entropy[choice] = choice_entropy.float()
                else:
                    delta = self.deep_set_attention(
                        h=h_norm[choice],
                        valid=valid[choice],
                        context=state_repr[choice].unsqueeze(1)
                    ).float()
                    z = z.index_add(0, choice, delta)

            out = []
            for i, ex in enumerate(examples):
                if ex['type'] == 'boolean':
                    out.append(F.pad(torch.stack([z[i, 0] * 0, z[i, 0]]), (0, kmax - 2)))
                else:
                    out.append(z[i])
            logits = torch.stack(out).masked_fill(~valid, -1e9)

            if return_value and return_attention_entropy:
                return logits, valid, values, entropy
            elif return_value:
                return logits, valid, values
            elif return_attention_entropy:
                return logits, valid, entropy
            return logits, valid

        def export_to_scorer_weights(self) -> Dict[str, np.ndarray]:
            """Exports trained neural weights into dictionary format for CPU QuantizedCandidateScorer."""
            weights: Dict[str, np.ndarray] = {}
            embed_dim = 128
            h_dim = getattr(self, "hidden_dim", 2048)
            state_dim = h_dim
            cand_dim = h_dim

            # 1. State / Candidate projections from learned embedding
            if hasattr(self, "token_embedding"):
                emb_w = self.token_embedding.weight.detach().cpu().float().numpy()
                weights["state_proj"] = emb_w[:embed_dim, :state_dim].copy()
                weights["cand_proj"] = emb_w[embed_dim:embed_dim * 2, :cand_dim].copy()

            # 2. Self Attention blocks
            if hasattr(self, "deep_set_attention") and hasattr(self.deep_set_attention, "blocks"):
                if len(self.deep_set_attention.blocks) > 0:
                    block0 = self.deep_set_attention.blocks[0]
                    sa = block0.self_attn
                    if hasattr(sa, "in_proj_weight") and sa.in_proj_weight is not None:
                        in_w = sa.in_proj_weight.detach().cpu().float().numpy()
                        weights["q_proj"] = in_w[:embed_dim, :].copy()
                        weights["k_proj"] = in_w[embed_dim:2 * embed_dim, :].copy()
                        weights["v_proj"] = in_w[2 * embed_dim:, :].copy()
                    if hasattr(sa, "out_proj") and hasattr(sa.out_proj, "weight"):
                        weights["out_proj"] = sa.out_proj.weight.detach().cpu().float().numpy().copy()

            # 3. Scalar scoring head & score MLP
            weights["score_mlp_0"] = np.zeros((embed_dim, embed_dim * 2), dtype=np.float32)
            weights["score_mlp_0"][:, :embed_dim] = np.eye(embed_dim, dtype=np.float32)
            # Ensure state representation slice [embed_dim:] is NOT zeroed out in CPU scorer!
            weights["score_mlp_0"][:, embed_dim:] = np.eye(embed_dim, dtype=np.float32) * 0.5
            weights["score_mlp_0_bias"] = np.zeros(embed_dim, dtype=np.float32)

            if hasattr(self, "scalar"):
                sc_w = self.scalar.weight.detach().cpu().float().numpy()
                weights["score_mlp_1"] = sc_w[:, :embed_dim].copy()
                if self.scalar.bias is not None:
                    weights["score_mlp_1_bias"] = self.scalar.bias.detach().cpu().float().numpy().copy()
            else:
                weights["score_mlp_1"] = np.zeros((1, embed_dim), dtype=np.float32)
                weights["score_mlp_1_bias"] = np.zeros(1, dtype=np.float32)

            # 4. Value head (net[3] is Linear(128, 64), net[5] is Linear(64, 1))
            if hasattr(self, "value_head") and hasattr(self.value_head, "net"):
                v_net = self.value_head.net
                if len(v_net) > 3 and hasattr(v_net[3], "weight"):
                    w0 = v_net[3].weight.detach().cpu().float().numpy()
                    weights["val_mlp_0"] = w0[:64, :embed_dim].copy()
                    if v_net[3].bias is not None:
                        weights["val_mlp_0_bias"] = v_net[3].bias.detach().cpu().float().numpy()[:64].copy()
                if len(v_net) > 5 and hasattr(v_net[5], "weight"):
                    w1 = v_net[5].weight.detach().cpu().float().numpy()
                    weights["val_mlp_1"] = w1[:1, :64].copy()
                    if v_net[5].bias is not None:
                        weights["val_mlp_1_bias"] = v_net[5].bias.detach().cpu().float().numpy()[:1].copy()

            return weights

else:
    # Lightweight CPU Mock for testing and non-torch environments
    class GlobalStateValueHead:
        def __init__(self, *args, **kwargs): pass
    class DeepSetAttentionHead:
        def __init__(self, *args, **kwargs): pass
    class AbstainModule:
        def __init__(self, *args, **kwargs): pass
    class GenZeroDualHeadModel:
        def __init__(self, *args, **kwargs):
            self._mock_weights = {
                "state_proj": np.zeros((128, 2048), dtype=np.float32),
                "cand_proj": np.zeros((128, 2048), dtype=np.float32),
                "q_proj": np.zeros((128, 128), dtype=np.float32),
                "k_proj": np.zeros((128, 128), dtype=np.float32),
                "v_proj": np.zeros((128, 128), dtype=np.float32),
                "out_proj": np.zeros((128, 128), dtype=np.float32),
                "score_mlp_0": np.zeros((128, 256), dtype=np.float32),
                "score_mlp_1": np.zeros((1, 128), dtype=np.float32),
                "val_mlp_0": np.zeros((64, 128), dtype=np.float32),
                "val_mlp_1": np.zeros((1, 64), dtype=np.float32),
            }
        def export_to_scorer_weights(self) -> Dict[str, np.ndarray]:
            return {k: v.copy() for k, v in self._mock_weights.items()}

        def forward(
            self,
            examples: List[dict],
            pad_token: int = 0,
            return_value: bool = False,
            return_abstain: bool = False,
            return_attention_entropy: bool = False
        ):
            kmax = max((len(ex.get('candidate_ids', [])) for ex in examples), default=1)
            logits = np.zeros((len(examples), kmax), dtype=np.float32)
            valid = np.ones((len(examples), kmax), dtype=bool)
            values = np.zeros((len(examples), 1), dtype=np.float32)
            entropy = np.zeros((len(examples), kmax), dtype=np.float32)
            if return_value and return_attention_entropy:
                return logits, valid, values, entropy
            elif return_value:
                return logits, valid, values
            elif return_attention_entropy:
                return logits, valid, entropy
            return logits, valid

        def __call__(self, state: Any, candidates: List[str]) -> Dict[str, Any]:
            return {"best_action": candidates[0] if candidates else None}

