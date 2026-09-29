"""Gen-Zero Layer 1: Option Isolation Attention Mask & Native Permutation-Equivariant Engine.

RFC Implementation for Issue #7:
1. Block-Causal Masking (Option Isolation):
   - Sequence: [Prefix (State + Instruction)] + [Option 0] + [Option 1] + ... + [Option K-1] + [Gather Token]
   - Mask:
     - Prefix can attend to Prefix tokens.
     - Option k can attend to Prefix tokens and Option k tokens (self).
     - Option k cannot attend to Option m (m != k).
     - Gather Token attends to Prefix and all Option branches.
2. Shared Position IDs:
   - Prefix has position IDs [0, ..., L_prefix - 1].
   - Each Option k has position IDs starting at L_prefix: [L_prefix, L_prefix + 1, ..., L_prefix + len(Option k) - 1].
   - Gather token has position ID L_prefix + max_option_len.
   - Completely eliminates position embedding bias across candidate options.
3. Decision Gathering:
   - Equivariant projection across candidate terminal states.
   - Mathematically guarantees 0.00% argmax flip rate under candidate order permutations.
"""

import copy
import hashlib
import itertools
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from .sanitization import sanitize_candidates, sanitize_state, sanitize_input_text
from .dual_head import normalize_state_repr

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


def build_option_isolation_mask(
    prefix_len: int,
    option_lens: List[int],
    include_gather: bool = True,
    is_causal_prefix: bool = False,
    is_causal_option: bool = True,
    dtype: Optional[Any] = None,
    device: Optional[Any] = None
) -> Dict[str, Any]:
    """Constructs the block-causal attention mask and shared position IDs.

    Args:
        prefix_len: Length of Prefix (State + Instruction) span.
        option_lens: List of token lengths for each candidate option span.
        include_gather: Whether to append a terminal gather token.
        is_causal_prefix: If True, prefix tokens attend causally; if False, bidirectional.
        is_causal_option: If True, option tokens attend causally to self; if False, bidirectional.
        dtype: PyTorch tensor dtype (default torch.float32 or bool).
        device: PyTorch device.

    Returns:
        Dict containing:
        - 'mask': 2D attention mask (shape [total_len, total_len]), 1.0 (or 0.0 additive) for visible, 0.0 (-1e9) for blocked.
        - 'position_ids': 1D tensor/array of shape [total_len].
        - 'span_indices': Dict mapping span names to (start_idx, end_idx) in the concatenated sequence.
        - 'option_terminal_indices': Indices of the last token for each option span.
        - 'total_len': Total sequence length.
    """
    k = len(option_lens)
    gather_len = 1 if include_gather else 0
    total_len = prefix_len + sum(option_lens) + gather_len

    # Track start and end indices for each span
    span_indices: Dict[str, Tuple[int, int]] = {
        "prefix": (0, prefix_len)
    }
    opt_starts = []
    opt_ends = []
    curr = prefix_len
    for i, olen in enumerate(option_lens):
        opt_starts.append(curr)
        opt_ends.append(curr + olen)
        span_indices[f"option_{i}"] = (curr, curr + olen)
        curr += olen

    gather_idx = curr if include_gather else None
    if include_gather:
        span_indices["gather"] = (curr, curr + 1)

    # Build mask matrix (1 = attend, 0 = mask out)
    if HAS_TORCH and (dtype is not None or device is not None or torch is not None):
        t_device = device if device is not None else torch.device("cpu")
        mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=t_device)

        # 1. Prefix visibility
        if is_causal_prefix:
            prefix_mask = torch.tril(torch.ones((prefix_len, prefix_len), dtype=torch.bool, device=t_device))
        else:
            prefix_mask = torch.ones((prefix_len, prefix_len), dtype=torch.bool, device=t_device)
        mask[0:prefix_len, 0:prefix_len] = prefix_mask

        # 2. Options visibility
        for i in range(k):
            s_i, e_i = opt_starts[i], opt_ends[i]
            olen = option_lens[i]
            # Option sees full prefix
            mask[s_i:e_i, 0:prefix_len] = True
            # Option sees self
            if is_causal_option:
                opt_self_mask = torch.tril(torch.ones((olen, olen), dtype=torch.bool, device=t_device))
            else:
                opt_self_mask = torch.ones((olen, olen), dtype=torch.bool, device=t_device)
            mask[s_i:e_i, s_i:e_i] = opt_self_mask
            # Option CANNOT see other options (mask[s_i:e_i, s_j:e_j] remains False)

        # 3. Gather token visibility
        if include_gather and gather_idx is not None:
            # Gather token attends to Prefix and all Options
            mask[gather_idx, 0:gather_idx + 1] = True

        # Position IDs: Shared across all option branches!
        position_ids = torch.zeros(total_len, dtype=torch.long, device=t_device)
        # Prefix positions: [0, ..., prefix_len - 1]
        position_ids[0:prefix_len] = torch.arange(prefix_len, dtype=torch.long, device=t_device)
        # Each option branch starts at exactly prefix_len
        max_olen = max(option_lens) if option_lens else 0
        for i in range(k):
            s_i, e_i = opt_starts[i], opt_ends[i]
            olen = option_lens[i]
            position_ids[s_i:e_i] = prefix_len + torch.arange(olen, dtype=torch.long, device=t_device)

        if include_gather and gather_idx is not None:
            position_ids[gather_idx] = prefix_len + max_olen

        terminal_indices = [opt_ends[i] - 1 for i in range(k)]

        # Convert mask to additive attention mask if float requested
        if dtype in (torch.float32, torch.float16, torch.bfloat16):
            additive_mask = torch.where(mask, 0.0, -1e9).to(dtype=dtype)
            mask_out = additive_mask
        else:
            mask_out = mask

        return {
            "mask": mask_out,
            "bool_mask": mask,
            "position_ids": position_ids,
            "span_indices": span_indices,
            "option_terminal_indices": terminal_indices,
            "gather_idx": gather_idx,
            "total_len": total_len
        }

    else:
        # NumPy / Pure CPU fallback
        mask = np.zeros((total_len, total_len), dtype=bool)

        # 1. Prefix
        if is_causal_prefix:
            mask[0:prefix_len, 0:prefix_len] = np.tril(np.ones((prefix_len, prefix_len), dtype=bool))
        else:
            mask[0:prefix_len, 0:prefix_len] = True

        # 2. Options
        for i in range(k):
            s_i, e_i = opt_starts[i], opt_ends[i]
            olen = option_lens[i]
            mask[s_i:e_i, 0:prefix_len] = True
            if is_causal_option:
                mask[s_i:e_i, s_i:e_i] = np.tril(np.ones((olen, olen), dtype=bool))
            else:
                mask[s_i:e_i, s_i:e_i] = True

        # 3. Gather
        if include_gather and gather_idx is not None:
            mask[gather_idx, 0:gather_idx + 1] = True

        position_ids = np.zeros(total_len, dtype=int)
        position_ids[0:prefix_len] = np.arange(prefix_len)
        max_olen = max(option_lens) if option_lens else 0
        for i in range(k):
            s_i, e_i = opt_starts[i], opt_ends[i]
            olen = option_lens[i]
            position_ids[s_i:e_i] = prefix_len + np.arange(olen)

        if include_gather and gather_idx is not None:
            position_ids[gather_idx] = prefix_len + max_olen

        terminal_indices = [opt_ends[i] - 1 for i in range(k)]

        return {
            "mask": mask,
            "bool_mask": mask,
            "position_ids": position_ids,
            "span_indices": span_indices,
            "option_terminal_indices": terminal_indices,
            "gather_idx": gather_idx,
            "total_len": total_len
        }


def tokenize_option_isolation_sequence(
    state: Any,
    candidates: List[str],
    instruction: str = "Select the optimal action.",
    vocab_size: int = 19999
) -> Tuple[List[int], List[List[int]], int]:
    """Tokenizes State+Instruction prefix and candidate option branches with boundary safety.

    Sanitizes untrusted control sequences before tokenizing to prevent boundary forgery.
    """
    clean_state = sanitize_state(state)
    clean_cands = sanitize_candidates(candidates)
    clean_instr = sanitize_input_text(instruction)

    # Prefix tokens
    state_str = normalize_state_repr(clean_state)
    prefix_str = f"[PREFIX]{clean_instr}:{state_str}"
    
    # Hash deterministic tokens for prefix
    def _str_to_tokens(s: str, min_tokens: int = 3) -> List[int]:
        tokens = [101]
        for piece in s.split():
            h = int(hashlib.md5(piece.encode("utf-8")).hexdigest()[:8], 16) % vocab_size
            tokens.append(h)
        if len(tokens) < min_tokens:
            tokens.extend([100] * (min_tokens - len(tokens)))
        return tokens

    prefix_tokens = _str_to_tokens(prefix_str, min_tokens=4)

    # Option branch tokens
    option_token_lists = []
    for c in clean_cands:
        c_str = str(c)
        c_tokens = _str_to_tokens(f"[OPT]{c_str}", min_tokens=2)
        c_tokens.append(102)  # EOS marker for option
        option_token_lists.append(c_tokens)

    gather_token = 103  # Special GATHER token

    return prefix_tokens, option_token_lists, gather_token


if HAS_TORCH:
    class OptionIsolationGatherHead(nn.Module):
        """Cross-option gathering head calculating candidate logits."""
        def __init__(self, hidden_dim: int, embed_dim: int = 128):
            super().__init__()
            self.cand_proj = nn.Linear(hidden_dim, embed_dim)
            self.state_proj = nn.Linear(hidden_dim, embed_dim)
            self.scorer = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 1)
            )

        def forward(self, prefix_rep: torch.Tensor, option_reps: torch.Tensor) -> torch.Tensor:
            """
            prefix_rep: [hidden_dim]
            option_reps: [K, hidden_dim]
            Returns: [K] logits
            """
            s_proj = self.state_proj(prefix_rep).unsqueeze(0)  # [1, embed_dim]
            c_proj = self.cand_proj(option_reps)               # [K, embed_dim]
            combined = torch.cat([s_proj.expand(c_proj.shape[0], -1), c_proj], dim=-1)
            scores = self.scorer(combined).squeeze(-1)         # [K]
            return scores


    class OptionIsolationEngine(nn.Module):
        """Native Option-Isolation Execution Engine guaranteeing 0.00% permutation flip rate."""
        def __init__(
            self,
            hidden_dim: int = 128,
            vocab_size: int = 20000,
            num_heads: int = 4,
            num_layers: int = 2
        ):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.vocab_size = vocab_size
            self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
            self.pos_embedding = nn.Embedding(2048, hidden_dim)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.gather_head = OptionIsolationGatherHead(hidden_dim=hidden_dim, embed_dim=hidden_dim)

            # Initialize weights
            nn.init.normal_(self.token_embedding.weight, std=0.02)
            nn.init.normal_(self.pos_embedding.weight, std=0.02)

        def forward(
            self,
            state: Any,
            candidates: List[str],
            instruction: str = "Select the optimal action."
        ) -> Dict[str, Any]:
            """Performs forward inference with block-causal mask and shared position IDs."""
            device = self.token_embedding.weight.device
            if not candidates:
                return {
                    "scores": {},
                    "best_action": None,
                    "confidence": 0.0,
                    "logits": torch.empty(0, device=device)
                }

            prefix_toks, opt_toks_list, gather_tok = tokenize_option_isolation_sequence(
                state, candidates, instruction, self.vocab_size
            )

            prefix_len = len(prefix_toks)
            option_lens = [len(ot) for ot in opt_toks_list]

            mask_data = build_option_isolation_mask(
                prefix_len=prefix_len,
                option_lens=option_lens,
                include_gather=True,
                is_causal_prefix=False,
                is_causal_option=False,
                device=device
            )

            bool_mask = mask_data["bool_mask"]
            attn_mask = torch.where(bool_mask, 0.0, -1e9)  # [total_len, total_len]
            pos_ids = mask_data["position_ids"]            # [total_len]
            term_indices = mask_data["option_terminal_indices"]

            # Assemble full sequence tokens
            full_tokens: List[int] = list(prefix_toks)
            for ot in opt_toks_list:
                full_tokens.extend(ot)
            full_tokens.append(gather_tok)

            tokens_tensor = torch.tensor(full_tokens, dtype=torch.long, device=device).unsqueeze(0)  # [1, total_len]
            pos_tensor = pos_ids.unsqueeze(0)  # [1, total_len]

            # Embeddings = Token + Position
            x = self.token_embedding(tokens_tensor) + self.pos_embedding(pos_tensor)

            # Run transformer with block-causal mask (2D attention mask)
            hidden = self.transformer(x, mask=attn_mask).squeeze(0)  # [total_len, hidden_dim]

            # Extract prefix representation (mean of prefix tokens)
            prefix_rep = hidden[0:prefix_len].mean(dim=0)

            # Extract option representations at their terminal tokens
            opt_reps = hidden[term_indices]  # [K, hidden_dim]

            # Compute permutation-equivariant candidate logits
            logits = self.gather_head(prefix_rep, opt_reps)  # [K]
            probs = F.softmax(logits, dim=-1)

            scores_dict = {
                cand: float(probs[i].detach().cpu().item())
                for i, cand in enumerate(candidates)
            }
            best_idx = int(torch.argmax(logits).item())
            best_action = candidates[best_idx]
            conf = float(probs[best_idx].detach().cpu().item())

            return {
                "scores": scores_dict,
                "best_action": best_action,
                "confidence": round(conf, 4),
                "logits": logits,
                "probabilities": probs
            }

        def verify_permutation_equivariance(
            self,
            state: Any,
            candidates: List[str],
            num_permutations: int = 5
        ) -> Dict[str, Any]:
            """Verifies that candidate scores and argmax are 100% identical under arbitrary permutations."""
            if len(candidates) <= 1:
                return {"is_equivariant": True, "argmax_flip_rate": 0.0, "tested_permutations": 1}

            # Generate all permutations (or sample up to num_permutations)
            all_perms = list(itertools.permutations(candidates))
            if len(all_perms) > num_permutations:
                sampled_perms = [list(candidates)]
                remaining = [p for p in all_perms if list(p) != list(candidates)]
                sampled_perms.extend([list(p) for p in random.sample(remaining, num_permutations - 1)])
            else:
                sampled_perms = [list(p) for p in all_perms]

            base_res = self.forward(state, sampled_perms[0])
            base_best = base_res["best_action"]
            base_scores = base_res["scores"]

            flips = 0
            max_score_diff = 0.0

            for perm in sampled_perms[1:]:
                res = self.forward(state, perm)
                if res["best_action"] != base_best:
                    flips += 1
                for c in candidates:
                    diff = abs(res["scores"][c] - base_scores[c])
                    if diff > max_score_diff:
                        max_score_diff = diff

            flip_rate = flips / max(1, len(sampled_perms) - 1)
            is_equivariant = (flips == 0) and (max_score_diff < 1e-4)

            return {
                "is_equivariant": is_equivariant,
                "argmax_flip_rate": round(flip_rate, 4),
                "max_score_diff": round(max_score_diff, 8),
                "tested_permutations": len(sampled_perms),
                "base_best_action": base_best
            }

else:
    class OptionIsolationEngine:
        """Placeholder when PyTorch is missing: fails closed on construction.

        An earlier fallback scored options with ``math.sin`` of token-id sums. Those scores
        carried no model signal, yet looked like real decisions, so it was removed.
        """
        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError(
                "OptionIsolationEngine requires PyTorch; install torch to use it. "
                "There is no CPU fallback that produces real option scores."
            )
