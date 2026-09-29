"""Nano-GenZero Compact Decision Core.

A compact, highly efficient decision backbone (~25M equivalent parameter footprint)
designed for edge devices, industrial control, and pure CPU deployments:
1. Feature Projection: 1024 -> 128 dim bottleneck.
2. Permutation-Equivariant Set Attention across K candidate actions.
3. Policy Head + Value Head.
4. Exportable weight dictionary for zero-dependency INT8 quantization.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import math

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

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


if HAS_TORCH:
    class NanoGenZeroCore(nn.Module):
        """Ultra-compact decision backbone for edge deployment."""

        def __init__(
            self,
            state_dim: int = 1024,
            candidate_dim: int = 1024,
            embed_dim: int = 128,
            num_heads: int = 4
        ):
            super().__init__()
            self.state_dim = state_dim
            self.candidate_dim = candidate_dim
            self.embed_dim = embed_dim

            # 1. State projection
            self.state_proj = nn.Linear(state_dim, embed_dim, bias=False)

            # 2. Candidate projection
            self.cand_proj = nn.Linear(candidate_dim, embed_dim, bias=False)

            # 3. Permutation-Equivariant Cross/Self-Attention Block
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

            # 4. Scoring MLP (maps pooled attention feature -> scalar score per candidate)
            self.score_mlp = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 1)
            )

            # 5. Value Head V(s) in [-1, 1]
            self.value_head = nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Tanh()
            )

        @classmethod
        def from_config(
            cls,
            config: Any,
            candidate_dim: Optional[int] = None,
            embed_dim: int = 128,
            num_heads: int = 4
        ) -> "NanoGenZeroCore":
            """Dynamically instantiates NanoGenZeroCore by extracting hidden_size from base model config."""
            state_dim = getattr(config, "hidden_size", getattr(config, "state_dim", 1024))
            cand_dim = candidate_dim if candidate_dim is not None else state_dim
            return cls(
                state_dim=state_dim,
                candidate_dim=cand_dim,
                embed_dim=embed_dim,
                num_heads=num_heads
            )

        def forward(
            self,
            state_repr: torch.Tensor,
            candidate_reprs: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass.
            
            Args:
                state_repr: [Batch, state_dim]
                candidate_reprs: [Batch, K, candidate_dim]
            Returns:
                logits: [Batch, K]
                value: [Batch, 1]
            """
            if state_repr.shape[-1] != self.state_dim:
                raise ValueError(
                    f"State dimension mismatch: expected {self.state_dim}, got {state_repr.shape[-1]}"
                )
            if candidate_reprs.shape[-1] != self.candidate_dim:
                raise ValueError(
                    f"Candidate dimension mismatch: expected {self.candidate_dim}, got {candidate_reprs.shape[-1]}"
                )

            b, k, _ = candidate_reprs.shape

            s_emb = self.state_proj(state_repr)  # [B, embed_dim]
            c_emb = self.cand_proj(candidate_reprs)  # [B, K, embed_dim]

            # Scaled Dot-Product Attention between candidates
            q = self.q_proj(c_emb)
            k_mat = self.k_proj(c_emb)
            v = self.v_proj(c_emb)

            scale = 1.0 / math.sqrt(self.embed_dim)
            attn_scores = torch.matmul(q, k_mat.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_out = self.out_proj(torch.matmul(attn_weights, v))  # [B, K, embed_dim]

            # Condition on state representation
            s_expanded = s_emb.unsqueeze(1).expand(-1, k, -1)  # [B, K, embed_dim]
            combined = torch.cat([attn_out, s_expanded], dim=-1)  # [B, K, embed_dim*2]

            logits = self.score_mlp(combined).squeeze(-1)  # [B, K]
            value = self.value_head(s_emb)  # [B, 1]

            return logits, value

        def export_weight_dict(self) -> Dict[str, np.ndarray]:
            """Exports all parameters as pure float32 NumPy arrays with dimension metadata."""
            weights = {}
            for name, param in self.named_parameters():
                weights[name] = param.detach().cpu().numpy().astype(np.float32)
            # Persist architecture dimensions metadata for verification on load
            weights["__metadata__"] = np.array(
                [self.state_dim, self.candidate_dim, self.embed_dim],
                dtype=np.int32
            )
            return weights
else:
    class NanoGenZeroCore:
        pass

