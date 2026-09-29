"""Gen-Zero Layer 1: Tree-Structured Prefix Attention & Context Quota Engine (Issue #14).

Architectural Implementation:
1. Shared Prefix Prefill: Encodes state prefix once and holds in KV-cache.
2. Cross-Branch Zero-Leakage Mask: Suffix question branches share state KV-cache,
   while attention between sibling branches is strictly zero-masked.
3. Standardized Context Quotas:
   - Single branch limit: 32,768 tokens (2^15)
   - Aggregated sequence package limit: 65,536 tokens (2^16)
4. Shared Position IDs & Packed Sequence Layout:
   - Branch position IDs start at prefix_len to eliminate position bias.
   - Computes cumulative sequence lengths (cu_seqlens) for FlashAttention varlen kernels.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import copy

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


SINGLE_BRANCH_CONTEXT_LIMIT: int = 32768   # 2^15
AGGREGATED_PACKAGE_LIMIT: int = 65536      # 2^16


def validate_context_quotas(prefix_len: int, branch_lens: Sequence[int]) -> bool:
    """Validates that token lengths strictly adhere to the standardized context quotas.

    Args:
        prefix_len: Number of tokens in shared state prefix.
        branch_lens: Token counts for each question branch.

    Raises:
        ValueError: If single branch or aggregated package exceeds limits.

    Returns:
        True if all quotas are respected.
    """
    if prefix_len < 0:
        raise ValueError(f"prefix_len must be non-negative, got {prefix_len}")

    for idx, b_len in enumerate(branch_lens):
        if b_len < 0:
            raise ValueError(f"branch length at index {idx} must be non-negative, got {b_len}")
        branch_total = prefix_len + b_len
        if branch_total > SINGLE_BRANCH_CONTEXT_LIMIT:
            raise ValueError(
                f"Single branch context {branch_total} tokens at index {idx} "
                f"(prefix {prefix_len} + branch {b_len}) exceeds quota limit of "
                f"{SINGLE_BRANCH_CONTEXT_LIMIT} tokens."
            )

    aggregated_total = prefix_len + sum(branch_lens)
    if aggregated_total > AGGREGATED_PACKAGE_LIMIT:
        raise ValueError(
            f"Aggregated sequence package {aggregated_total} tokens "
            f"(prefix {prefix_len} + {len(branch_lens)} branches) exceeds quota limit of "
            f"{AGGREGATED_PACKAGE_LIMIT} tokens."
        )

    return True


def build_prefix_tree_attention_mask(
    prefix_len: int,
    branch_lens: Sequence[int],
    is_causal_prefix: bool = False,
    is_causal_branch: bool = True,
    dtype: Optional[Any] = None,
    device: Optional[Any] = None
) -> Dict[str, Any]:
    """Constructs the tree-structured prefix attention mask and shared position IDs.

    Topology:
      - Tokens [0, prefix_len): Shared Prefix (State).
        Can attend to Prefix tokens (causal or full bidirectional).
        Cannot attend to any branch tokens.
      - Tokens for branch m [start_m, end_m):
        Can attend to Shared Prefix [0, prefix_len).
        Can attend to own branch tokens [start_m, end_m) (causal or bidirectional).
        Strictly CANNOT attend to sibling branches k != m (zero leakage).

    Args:
        prefix_len: Number of tokens in shared prefix.
        branch_lens: Sequence of token counts for each branch.
        is_causal_prefix: Whether prefix self-attention is causal (tril) or full bidirectional.
        is_causal_branch: Whether branch self-attention is causal (tril) or bidirectional.
        dtype: PyTorch tensor dtype (default torch.bool).
        device: PyTorch device.

    Returns:
        Dict containing:
          - 'mask': 2D boolean tensor (or 2D list/numpy array if torch not present) of shape [N, N].
          - 'position_ids': 1D tensor / list of shape [N].
          - 'span_indices': Dict mapping span names ('prefix', 'branch_0', ...) to (start, end) tuples.
          - 'total_len': Total sequence length N.
          - 'cu_seqlens': List of cumulative sequence lengths.
          - 'zero_leakage_verified': True.
    """
    validate_context_quotas(prefix_len, branch_lens)

    m = len(branch_lens)
    total_len = prefix_len + sum(branch_lens)

    span_indices: Dict[str, Tuple[int, int]] = {
        "prefix": (0, prefix_len)
    }
    branch_starts: List[int] = []
    branch_ends: List[int] = []

    curr = prefix_len
    for i, blen in enumerate(branch_lens):
        branch_starts.append(curr)
        branch_ends.append(curr + blen)
        span_indices[f"branch_{i}"] = (curr, curr + blen)
        curr += blen

    # Shared Position IDs
    # Prefix: [0, 1, ..., prefix_len - 1]
    # Each Branch: starts at prefix_len to eliminate relative order bias
    pos_ids_list: List[int] = list(range(prefix_len))
    for blen in branch_lens:
        pos_ids_list.extend([prefix_len + j for j in range(blen)])

    # cu_seqlens for FlashAttention packed sequence layout
    cu_seqlens: List[int] = [0, prefix_len]
    acc = prefix_len
    for blen in branch_lens:
        acc += blen
        cu_seqlens.append(acc)

    if HAS_TORCH and (dtype is not None or device is not None or torch is not None):
        t_device = device if device is not None else torch.device("cpu")
        t_dtype = dtype if dtype is not None else torch.bool

        mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=t_device)

        # 1. Prefix visibility
        if prefix_len > 0:
            if is_causal_prefix:
                prefix_mask = torch.tril(torch.ones((prefix_len, prefix_len), dtype=torch.bool, device=t_device))
            else:
                prefix_mask = torch.ones((prefix_len, prefix_len), dtype=torch.bool, device=t_device)
            mask[0:prefix_len, 0:prefix_len] = prefix_mask

        # 2. Branch visibility
        for i in range(m):
            s_i, e_i = branch_starts[i], branch_ends[i]
            blen = branch_lens[i]
            if blen <= 0:
                continue

            # Branch sees full shared prefix
            if prefix_len > 0:
                mask[s_i:e_i, 0:prefix_len] = True

            # Branch sees own tokens
            if is_causal_branch:
                branch_self = torch.tril(torch.ones((blen, blen), dtype=torch.bool, device=t_device))
            else:
                branch_self = torch.ones((blen, blen), dtype=torch.bool, device=t_device)
            mask[s_i:e_i, s_i:e_i] = branch_self

            # Sibling branches k != i remain False (strictly zero-leakage)

        position_ids_tensor = torch.tensor(pos_ids_list, dtype=torch.long, device=t_device)
        if t_dtype != torch.bool:
            mask = mask.to(t_dtype)

        # Verify zero leakage between distinct branches
        zero_leakage = True
        for i in range(m):
            s_i, e_i = branch_starts[i], branch_ends[i]
            for j in range(m):
                if i != j:
                    s_j, e_j = branch_starts[j], branch_ends[j]
                    if torch.any(mask[s_i:e_i, s_j:e_j]):
                        zero_leakage = False
                        break

        return {
            "mask": mask,
            "position_ids": position_ids_tensor,
            "span_indices": span_indices,
            "total_len": total_len,
            "cu_seqlens": cu_seqlens,
            "zero_leakage_verified": zero_leakage
        }

    # Pure Python / NumPy fallback
    mask_matrix = [[False] * total_len for _ in range(total_len)]

    if prefix_len > 0:
        for r in range(prefix_len):
            for c in range(prefix_len):
                if not is_causal_prefix or c <= r:
                    mask_matrix[r][c] = True

    for i in range(m):
        s_i, e_i = branch_starts[i], branch_ends[i]
        blen = branch_lens[i]
        for r_offset in range(blen):
            r = s_i + r_offset
            # Sees prefix
            for c in range(prefix_len):
                mask_matrix[r][c] = True
            # Sees self
            for c_offset in range(blen):
                c = s_i + c_offset
                if not is_causal_branch or c_offset <= r_offset:
                    mask_matrix[r][c] = True

    # Verify zero leakage
    zero_leakage = True
    for i in range(m):
        s_i, e_i = branch_starts[i], branch_ends[i]
        for j in range(m):
            if i != j:
                s_j, e_j = branch_starts[j], branch_ends[j]
                for r in range(s_i, e_i):
                    for c in range(s_j, e_j):
                        if mask_matrix[r][c]:
                            zero_leakage = False

    return {
        "mask": mask_matrix,
        "position_ids": pos_ids_list,
        "span_indices": span_indices,
        "total_len": total_len,
        "cu_seqlens": cu_seqlens,
        "zero_leakage_verified": zero_leakage
    }


class PrefixTreePackedLayout:
    """Packed sequence layout helper for tree-structured prefill and FlashAttention."""

    @staticmethod
    def pack_inputs(
        prefix_tokens: List[int],
        branch_token_lists: List[List[int]]
    ) -> Tuple[List[int], Dict[str, Any]]:
        """Packs prefix tokens and multiple question branches into a single sequence package.

        Args:
            prefix_tokens: Token list for state prefix.
            branch_token_lists: Token lists for each question branch.

        Returns:
            Tuple of:
              - Concatenated token sequence [N]
              - Mask and geometry metadata dict
        """
        prefix_len = len(prefix_tokens)
        branch_lens = [len(b) for b in branch_token_lists]
        validate_context_quotas(prefix_len, branch_lens)

        packed_tokens: List[int] = list(prefix_tokens)
        for b in branch_token_lists:
            packed_tokens.extend(b)

        geom = build_prefix_tree_attention_mask(prefix_len, branch_lens)
        return packed_tokens, geom

    @staticmethod
    def extract_branch_hidden_states(
        hidden_states: Any,
        span_indices: Dict[str, Tuple[int, int]],
        branch_count: int,
        pooling: str = "last"
    ) -> List[Any]:
        """Extracts branch representations from packed hidden states.

        Args:
            hidden_states: Tensor [1, N, D] or [N, D] or nested list.
            span_indices: Dict of span coordinates.
            branch_count: Number of branches M.
            pooling: 'last' token representation or 'mean'.

        Returns:
            List of branch terminal vectors [D].
        """
        results = []
        is_tensor = HAS_TORCH and isinstance(hidden_states, torch.Tensor)

        for i in range(branch_count):
            span_key = f"branch_{i}"
            if span_key not in span_indices:
                continue
            start, end = span_indices[span_key]
            if start >= end:
                continue

            if is_tensor:
                if hidden_states.dim() == 3:
                    branch_hidden = hidden_states[0, start:end, :]
                else:
                    branch_hidden = hidden_states[start:end, :]

                if pooling == "last":
                    results.append(branch_hidden[-1])
                elif pooling == "mean":
                    results.append(branch_hidden.mean(dim=0))
                else:
                    results.append(branch_hidden[-1])
            else:
                branch_slice = hidden_states[start:end]
                if pooling == "last":
                    results.append(branch_slice[-1])
                elif pooling == "mean" and branch_slice and isinstance(branch_slice[0], (list, tuple)):
                    dim = len(branch_slice[0])
                    mean_vec = [sum(row[d] for row in branch_slice) / len(branch_slice) for d in range(dim)]
                    results.append(mean_vec)
                else:
                    results.append(branch_slice[-1])

        return results
