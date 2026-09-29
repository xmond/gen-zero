"""Gen-Zero Layer 1: Parallel Prefix-Suffix KV-Cache Reordering & Selective Logits Engine (I-12).

Architectural Implementation:
1. One-pass Prefix Prefill: Computes KV activations for common state prefix once.
2. Parallel Suffix Branching: Broadcasts/reorders prefix KV cache across K candidates with chunking (K <= 32).
3. Right-Padded Variable-Length Alignment: Enforces causal attention mask [K, P + S_max] and exact position_ids.
4. lm_head Bypass & Vectorized Terminal Gathering: Bypasses vocab projection GEMM, extracting exact [K, D] representations via torch.gather.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
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


class ParallelPrefixCacheEngine:
    """Parallel Suffix KV-Cache Engine with Speculative Attention Chunking."""

    MAX_CHUNK_SIZE: int = 32

    @classmethod
    def clone_or_reorder_cache(cls, cached_kv: Any, target_k: int) -> Any:
        """Safely clones and expands prefix KV cache across target_k branches
        without in-place mutation of the master prefix cache.
        """
        if not HAS_TORCH or cached_kv is None:
            return None

        # 1. Modern HuggingFace DynamicCache (transformers >= 4.43+)
        if hasattr(cached_kv, "layers"):
            new_cache = copy.copy(cached_kv)
            new_cache.layers = []
            for layer in cached_kv.layers:
                new_layer = copy.copy(layer)
                if hasattr(layer, "keys") and hasattr(layer, "values") and layer.keys is not None:
                    repeats_k = [target_k] + [1] * (layer.keys.dim() - 1)
                    repeats_v = [target_k] + [1] * (layer.values.dim() - 1)
                    new_layer.keys = layer.keys.repeat(*repeats_k)
                    new_layer.values = layer.values.repeat(*repeats_v)
                new_cache.layers.append(new_layer)
            return new_cache

        # 2. Legacy HuggingFace DynamicCache (transformers < 4.43)
        if hasattr(cached_kv, "key_cache") and hasattr(cached_kv, "value_cache"):
            new_cache = copy.copy(cached_kv)
            new_cache.key_cache = []
            new_cache.value_cache = []
            for k_tensor, v_tensor in zip(cached_kv.key_cache, cached_kv.value_cache):
                repeats = [target_k] + [1] * (k_tensor.dim() - 1)
                new_cache.key_cache.append(k_tensor.repeat(*repeats))
                new_cache.value_cache.append(v_tensor.repeat(*repeats))
            return new_cache

        # 3. Legacy tuple of (key, value) per layer
        if isinstance(cached_kv, (tuple, list)):
            expanded_layers = []
            for layer in cached_kv:
                if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                    k_t, v_t = layer[0], layer[1]
                    if hasattr(k_t, "repeat"):
                        repeats_k = [target_k] + [1] * (k_t.dim() - 1)
                        new_k = k_t.repeat(*repeats_k)
                    else:
                        new_k = k_t
                    if hasattr(v_t, "repeat"):
                        repeats_v = [target_k] + [1] * (v_t.dim() - 1)
                        new_v = v_t.repeat(*repeats_v)
                    else:
                        new_v = v_t
                    expanded_layers.append((new_k, new_v))
                else:
                    expanded_layers.append(layer)
            return tuple(expanded_layers) if isinstance(cached_kv, tuple) else expanded_layers

        try:
            return copy.deepcopy(cached_kv)
        except Exception:
            return copy.copy(cached_kv)

    @classmethod
    def evaluate_parallel_suffixes(
        cls,
        backbone: Any,
        prefix_tokens: Any,
        candidate_suffixes: List[Any],
        pad_token_id: int = 0,
        max_chunk_size: int = MAX_CHUNK_SIZE
    ) -> Any:
        """Evaluates K candidate suffixes in parallel against the shared prefix KV-cache.
        
        Args:
            backbone: Transformer model (e.g. Qwen2, LLaMA, or mock model).
            prefix_tokens: Shared state prefix token tensor [1, P].
            candidate_suffixes: List of candidate suffix token 1D/2D tensors.
            pad_token_id: Token ID used for right-padding variable-length suffixes.
            max_chunk_size: Maximum candidates processed in a single parallel GPU forward pass (<= 32).
        
        Returns:
            Terminal leaf representations tensor of shape [K, D].
        """
        # Determine execution device
        if hasattr(prefix_tokens, "device") and prefix_tokens.device is not None:
            device = prefix_tokens.device
        elif hasattr(backbone, "device"):
            device = backbone.device
        elif hasattr(backbone, "parameters"):
            try:
                device = next(backbone.parameters()).device
            except (StopIteration, Exception):
                device = "cpu"
        else:
            device = "cpu"

        k = len(candidate_suffixes)
        if k == 0:
            if not HAS_TORCH:
                return []
            dim = getattr(backbone.config, "hidden_size", 128) if (backbone and hasattr(backbone, "config")) else 128
            return torch.zeros((0, dim), device=device)

        if not HAS_TORCH:
            # Fallback for pure CPU / mock environments
            dim = 128
            if HAS_NUMPY:
                return np.zeros((k, dim), dtype=np.float32)
            return [[0.0] * dim for _ in range(k)]

        if not isinstance(prefix_tokens, torch.Tensor):
            prefix_tokens = torch.tensor(prefix_tokens, dtype=torch.long, device=device)

        if prefix_tokens.dim() == 1:
            prefix_tokens = prefix_tokens.unsqueeze(0)
        prefix_len = prefix_tokens.shape[1]

        # 1. Master Prefix Prefill Pass (Single pass over prefix context P)
        with torch.no_grad():
            trunk = getattr(backbone, "model", backbone)
            prefix_outputs = trunk(
                input_ids=prefix_tokens,
                use_cache=True,
                return_dict=True
            )
            master_kv = prefix_outputs.past_key_values

        collected_leaf_reprs = []

        # 2. Chunked Parallel Execution (B = min(K, max_chunk_size) <= 32)
        for chunk_start in range(0, k, max_chunk_size):
            chunk_suffixes = candidate_suffixes[chunk_start : chunk_start + max_chunk_size]
            b_chunk = len(chunk_suffixes)

            suffix_lens = [s.shape[-1] if hasattr(s, "shape") else len(s) for s in chunk_suffixes]
            s_max = max(suffix_lens)

            # Build right-padded batch tensor [b_chunk, s_max]
            padded_input_ids = torch.full(
                (b_chunk, s_max),
                fill_value=pad_token_id,
                dtype=torch.long,
                device=device
            )
            for i, s in enumerate(chunk_suffixes):
                s_t = s.to(device=device, dtype=torch.long) if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.long, device=device)
                s_flat = s_t.flatten()
                padded_input_ids[i, :len(s_flat)] = s_flat

            # Exact Vectorized position_ids [b_chunk, s_max] starting at prefix_len
            position_ids = torch.zeros((b_chunk, s_max), dtype=torch.long, device=device)
            for i, s_len in enumerate(suffix_lens):
                position_ids[i, :s_len] = torch.arange(prefix_len, prefix_len + s_len, device=device)
                if s_len < s_max:
                    # Clamp padded positions to avoid out-of-range RoPE extrapolation
                    position_ids[i, s_len:] = prefix_len + s_len - 1

            # 2D Attention Mask [b_chunk, prefix_len + s_max]
            # Prefix tokens (1) + active suffix tokens (1) + right padding (0)
            attention_mask = torch.zeros((b_chunk, prefix_len + s_max), dtype=torch.long, device=device)
            attention_mask[:, :prefix_len] = 1  # Full causal attention to prefix
            for i, s_len in enumerate(suffix_lens):
                attention_mask[i, prefix_len : prefix_len + s_len] = 1

            # Clone prefix KV cache for this chunk without mutating master_kv
            chunk_kv = cls.clone_or_reorder_cache(master_kv, target_k=b_chunk)

            # Forward pass through model trunk (bypassing lm_head full-vocab GEMM)
            with torch.no_grad():
                out = trunk(
                    input_ids=padded_input_ids,
                    past_key_values=chunk_kv,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True
                )

                hidden_states = out.last_hidden_state  # [b_chunk, s_max, D]
                hidden_dim = hidden_states.shape[-1]

                # Vectorized Terminal State Extraction via torch.gather
                lens_tensor = torch.tensor(suffix_lens, dtype=torch.long, device=device)
                gather_idx = (lens_tensor - 1).view(b_chunk, 1, 1).expand(b_chunk, 1, hidden_dim)
                chunk_leaf_reprs = torch.gather(hidden_states, dim=1, index=gather_idx).squeeze(1)  # [b_chunk, D]

                collected_leaf_reprs.append(chunk_leaf_reprs)

            # Explicit deallocation to assist PyTorch caching allocator
            del chunk_kv, out, position_ids, attention_mask, padded_input_ids

        return torch.cat(collected_leaf_reprs, dim=0)


# Backwards compatibility alias
GenZeroPrefixCacheEngine = ParallelPrefixCacheEngine
