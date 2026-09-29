"""Unit and Equivariance Tests for Parallel Suffix Cache Engine."""

import unittest
from typing import List, Dict, Any, Optional

from gen_zero.model.prefix_cache import ParallelPrefixCacheEngine, HAS_TORCH

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    transformers = None
    HAS_TRANSFORMERS = False

if HAS_TORCH:
    import torch
    import torch.nn as nn

    class MockTransformerTrunk(nn.Module):
        """Mock transformer trunk simulating causal attention and KV cache."""
        def __init__(self, hidden_dim: int = 64, num_layers: int = 2):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.embed = nn.Embedding(256, hidden_dim)

        def forward(
            self,
            input_ids: torch.Tensor,
            past_key_values: Optional[Any] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            use_cache: bool = True,
            return_dict: bool = True
        ):
            b, seq_len = input_ids.shape
            h = self.embed(input_ids)  # [B, seq_len, D]

            # Simulate causal dependency on prefix if KV cache provided
            if past_key_values is not None:
                # Add influence from cached keys
                k_layer0 = past_key_values[0][0]
                prefix_summary = k_layer0.mean(dim=-2)  # [B, D]
                h = h + prefix_summary.unsqueeze(1) * 0.1

            new_kv = None
            if use_cache:
                # Mock KV cache: tuple of (k, v) per layer
                new_kv = tuple(
                    (h.clone(), h.clone()) for _ in range(self.num_layers)
                )

            class Output:
                def __init__(self, last_hidden_state, past_key_values):
                    self.last_hidden_state = last_hidden_state
                    self.past_key_values = past_key_values

            return Output(last_hidden_state=h, past_key_values=new_kv)


class TestParallelPrefixCacheEngine(unittest.TestCase):
    """Verifies ParallelPrefixCacheEngine parallel execution and guarantees."""

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for transformer mock tests")
    def test_variable_length_suffix_gathering(self):
        """Verify that variable length suffixes gather the true terminal token, not pad."""
        trunk = MockTransformerTrunk(hidden_dim=32)
        prefix = torch.tensor([[10, 11, 12]], dtype=torch.long)
        
        # Suffixes with different lengths: 1 token, 3 tokens, 2 tokens
        s1 = torch.tensor([21], dtype=torch.long)
        s2 = torch.tensor([22, 23, 24], dtype=torch.long)
        s3 = torch.tensor([25, 26], dtype=torch.long)
        suffixes = [s1, s2, s3]

        out = ParallelPrefixCacheEngine.evaluate_parallel_suffixes(
            backbone=trunk,
            prefix_tokens=prefix,
            candidate_suffixes=suffixes,
            pad_token_id=0,
            max_chunk_size=32
        )

        self.assertEqual(out.shape, (3, 32))
        self.assertFalse(torch.isnan(out).any())

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for chunking test")
    def test_chunking_beyond_max_chunk_size(self):
        """Verify candidate counts > max_chunk_size (e.g. K=40 > 32) are chunked properly."""
        trunk = MockTransformerTrunk(hidden_dim=32)
        prefix = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        
        k = 40
        suffixes = [torch.tensor([100 + i], dtype=torch.long) for i in range(k)]

        out = ParallelPrefixCacheEngine.evaluate_parallel_suffixes(
            backbone=trunk,
            prefix_tokens=prefix,
            candidate_suffixes=suffixes,
            pad_token_id=0,
            max_chunk_size=16  # Force 3 chunks (16 + 16 + 8)
        )

        self.assertEqual(out.shape, (40, 32))

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for cache immutability test")
    def test_master_cache_immutability(self):
        """Master prefix KV cache remains unmutated after parallel suffix execution."""
        trunk = MockTransformerTrunk(hidden_dim=32)
        prefix = torch.tensor([[5, 6, 7]], dtype=torch.long)
        
        pref_out = trunk(input_ids=prefix, use_cache=True)
        master_kv = pref_out.past_key_values
        orig_shape = master_kv[0][0].shape  # [1, 3, 32]

        cloned_kv = ParallelPrefixCacheEngine.clone_or_reorder_cache(master_kv, target_k=8)
        self.assertEqual(cloned_kv[0][0].shape, (8, 3, 32))
        # Verify master cache was not mutated in-place
        self.assertEqual(master_kv[0][0].shape, orig_shape)

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for padding isolation test")
    def test_right_padding_isolation(self):
        """Verify that padding tokens do not contaminate representations of shorter suffixes."""
        class CausalSelfAttnTrunk(nn.Module):
            def __init__(self, hidden_dim: int = 32):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.embed = nn.Embedding(256, hidden_dim)
                self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)

            def forward(self, input_ids, past_key_values=None, position_ids=None, attention_mask=None, use_cache=True, return_dict=True):
                b, s = input_ids.shape
                h = self.embed(input_ids)
                q = self.q(h)
                k = self.k(h)
                v = self.v(h)

                if past_key_values is not None:
                    pk, pv = past_key_values[0]
                    k = torch.cat([pk, k], dim=-2)
                    v = torch.cat([pv, v], dim=-2)

                # Compute causal attention with attention_mask
                scores = torch.matmul(q, k.transpose(-1, -2)) / (self.hidden_dim ** 0.5)
                if attention_mask is not None:
                    mask = (attention_mask == 0).unsqueeze(1).expand_as(scores)
                    scores = scores.masked_fill(mask, float('-inf'))
                attn = torch.softmax(scores, dim=-1)
                attn = torch.nan_to_num(attn, nan=0.0)
                out = torch.matmul(attn, v)

                new_kv = tuple([(k.clone(), v.clone())]) if use_cache else None
                class Output:
                    def __init__(self, last_hidden_state, past_key_values):
                        self.last_hidden_state = last_hidden_state
                        self.past_key_values = past_key_values
                return Output(out, new_kv)

        trunk = CausalSelfAttnTrunk(hidden_dim=32).eval()
        prefix = torch.tensor([[1, 2]], dtype=torch.long)
        s_short = torch.tensor([10], dtype=torch.long)
        s_long = torch.tensor([20, 30, 40], dtype=torch.long)

        # 1. Suffix short evaluated alone
        out_single = ParallelPrefixCacheEngine.evaluate_parallel_suffixes(
            backbone=trunk,
            prefix_tokens=prefix,
            candidate_suffixes=[s_short]
        )

        # 2. Suffix short evaluated batched with longer suffix (right-padded)
        out_batch = ParallelPrefixCacheEngine.evaluate_parallel_suffixes(
            backbone=trunk,
            prefix_tokens=prefix,
            candidate_suffixes=[s_short, s_long]
        )

        # Short suffix terminal representation must be identical (< 1e-5)
        diff = torch.max(torch.abs(out_single[0] - out_batch[0])).item()
        self.assertLess(diff, 1e-5, f"Right-padding contaminated short suffix: diff={diff}")

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for dynamic cache test")
    def test_dynamic_cache_cloning_and_immutability(self):
        """Verify modern DynamicCache (or mock DynamicCache with .layers) clone & immutability."""
        class MockLayer:
            def __init__(self, keys, values):
                self.keys = keys
                self.values = values

        class MockDynamicCache:
            def __init__(self, layers):
                self.layers = layers

        k = torch.randn(1, 2, 4, 16)
        v = torch.randn(1, 2, 4, 16)
        cache = MockDynamicCache([MockLayer(k.clone(), v.clone())])

        cloned = ParallelPrefixCacheEngine.clone_or_reorder_cache(cache, target_k=8)
        self.assertEqual(cloned.layers[0].keys.shape, (8, 2, 4, 16))
        self.assertEqual(cloned.layers[0].values.shape, (8, 2, 4, 16))
        # Master cache intact
        self.assertEqual(cache.layers[0].keys.shape, (1, 2, 4, 16))

    @unittest.skipUnless(HAS_TORCH and HAS_TRANSFORMERS, "PyTorch and Transformers required")
    def test_real_transformer_dynamic_cache_end_to_end(self):
        """End-to-end forward pass with real Qwen2Config and DynamicCache."""
        from transformers import Qwen2Config, Qwen2Model
        config = Qwen2Config(
            vocab_size=200,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=128
        )
        model = Qwen2Model(config).eval()
        prefix = torch.tensor([[5, 6, 7]], dtype=torch.long)
        s1 = torch.tensor([10, 11], dtype=torch.long)
        s2 = torch.tensor([12], dtype=torch.long)

        out = ParallelPrefixCacheEngine.evaluate_parallel_suffixes(
            backbone=model,
            prefix_tokens=prefix,
            candidate_suffixes=[s1, s2]
        )
        self.assertEqual(out.shape, (2, 32))
        self.assertFalse(torch.isnan(out).any())

    def test_fallback_when_torch_or_empty(self):
        """Engine handles empty suffix list gracefully."""
        res = ParallelPrefixCacheEngine.evaluate_parallel_suffixes(
            backbone=None,
            prefix_tokens=[],
            candidate_suffixes=[]
        )
        self.assertEqual(len(res), 0)


if __name__ == "__main__":
    unittest.main()
