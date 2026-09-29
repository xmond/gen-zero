"""Unit tests for CandidateMultiLayerFusionEngine (Strategy 1 + Strategy 2 fusion, 0-token regime).

Verifies:
1. Batch construction: shared-prefix padding, attention mask, and position-id alignment.
2. Strategy 1 (candidate-conditioned log-likelihood) matches a hand-derived reference computed
   from the same mock model's deterministic logits.
3. Strategy 2 (multi-layer ETF projection) mathematical properties: unit-norm embeddings and an
   equiangular ETF frame.
4. Exactly one batched forward pass runs and exactly 0 tokens are ever generated.
5. Degenerate single-candidate input does not produce NaN/Inf.
"""

from types import SimpleNamespace

import unittest
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from gen_zero.model.candidate_fusion import (
        CandidateMultiLayerFusionEngine,
        CandidateFusionResult,
        DEFAULT_LAYER_OFFSETS,
    )

    class MockTokenizer:
        """Deterministic tokenizer: one token per character, no external vocab file."""
        pad_token_id = 0
        eos_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [(ord(c) % 60) + 1 for c in text]

    class MockMergingTokenizer:
        """Char-tokenizer with one BPE-like merge rule: a contiguous "ab" substring
        becomes a single token. This lets joint-vs-naive tokenization diverge at a
        prompt/candidate boundary, the way real BPE tokenizers merge across word
        boundaries that separately-encoded halves can never produce.
        """
        pad_token_id = 0
        eos_token_id = 0
        MERGED_TOKEN_ID = 61  # stays within MockCandidateModel's default vocab_size=64

        def encode(self, text, add_special_tokens=False):
            ids = []
            i = 0
            while i < len(text):
                if text[i : i + 2] == "ab":
                    ids.append(self.MERGED_TOKEN_ID)
                    i += 2
                else:
                    ids.append((ord(text[i]) % 60) + 1)
                    i += 1
            return ids

    class MockCandidateModel(nn.Module):
        """Deterministic causal-LM stand-in with a `generate` guard.

        Each layer applies a fixed per-layer affine transform so the 10 hidden-state
        layers (plus the embedding layer) are distinguishable, which lets Strategy 2
        pull different vectors from different layer offsets.
        """

        def __init__(self, vocab_size: int = 64, hidden_dim: int = 16, num_layers: int = 10):
            super().__init__()
            self.vocab_size = vocab_size
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.embed = nn.Embedding(vocab_size, hidden_dim)
            self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
            self.config = SimpleNamespace(num_hidden_layers=num_layers, hidden_size=hidden_dim)
            self.forward_call_count = 0
            self.generate_call_count = 0

        def forward(
            self,
            input_ids,
            attention_mask=None,
            position_ids=None,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        ):
            self.forward_call_count += 1
            h = self.embed(input_ids)  # [B, L, D]
            hidden_states = [h]
            for i in range(self.num_layers):
                h = h * (0.5 + 0.1 * i) + 0.01 * i
                hidden_states.append(h)
            logits = self.lm_head(h)
            return SimpleNamespace(
                logits=logits,
                hidden_states=tuple(hidden_states) if output_hidden_states else None,
            )

        def generate(self, *args, **kwargs):
            self.generate_call_count += 1
            raise AssertionError("generate() must never be called in the 0-token candidate fusion regime")


@unittest.skipUnless(HAS_TORCH, "PyTorch required for candidate fusion tests")
class TestCandidateFusionBatchConstruction(unittest.TestCase):
    def setUp(self):
        self.model = MockCandidateModel()
        self.tokenizer = MockTokenizer()
        self.engine = CandidateMultiLayerFusionEngine(self.model, self.tokenizer)

    def test_batch_shapes_and_shared_prefix(self):
        prompt = "abc"
        candidates = ["x", "yz", "www"]
        batch = self.engine.build_batch(prompt, candidates)

        prompt_len = len(prompt)
        c_max = max(len(c) for c in candidates)
        k = len(candidates)

        self.assertEqual(batch["prompt_len"], prompt_len)
        self.assertEqual(batch["candidate_lens"], [1, 2, 3])
        self.assertEqual(batch["input_ids"].shape, (k, prompt_len + c_max))
        self.assertEqual(batch["attention_mask"].shape, (k, prompt_len + c_max))

        prompt_ids = self.tokenizer.encode(prompt)
        for i in range(k):
            self.assertEqual(
                batch["input_ids"][i, :prompt_len].tolist(), prompt_ids,
                "every row must share the identical tokenized prompt prefix",
            )

    def test_attention_mask_and_position_id_alignment(self):
        prompt = "hi"
        candidates = ["a", "bcd"]
        batch = self.engine.build_batch(prompt, candidates)
        prompt_len = batch["prompt_len"]

        for i, c_len in enumerate(batch["candidate_lens"]):
            real_len = prompt_len + c_len
            mask_row = batch["attention_mask"][i]
            self.assertEqual(int(mask_row.sum().item()), real_len)
            self.assertTrue(bool(mask_row[:real_len].all()))
            if real_len < mask_row.shape[0]:
                self.assertFalse(bool(mask_row[real_len:].any()))

            pos_row = batch["position_ids"][i]
            self.assertEqual(pos_row[:real_len].tolist(), list(range(real_len)))
            if real_len < pos_row.shape[0]:
                # Padded RoPE positions are clamped to the last real position.
                self.assertTrue(bool((pos_row[real_len:] == real_len - 1).all()))

    def test_rejects_empty_candidate_list(self):
        with self.assertRaises(ValueError):
            self.engine.build_batch("hello", [])

    def test_rejects_zero_token_candidate(self):
        with self.assertRaises(ValueError):
            self.engine.build_batch("hello", ["ok", ""])

    def test_rejects_empty_prompt(self):
        with self.assertRaises(ValueError):
            self.engine.build_batch("", ["ok"])


@unittest.skipUnless(HAS_TORCH, "PyTorch required for candidate fusion tests")
class TestCandidateFusionStrategy1Likelihood(unittest.TestCase):
    def test_sequence_logprob_matches_manual_reference(self):
        model = MockCandidateModel()
        tokenizer = MockTokenizer()
        engine = CandidateMultiLayerFusionEngine(model, tokenizer)

        prompt = "the cat"
        candidates = ["sat", "jumped over"]
        result = engine.score(prompt, candidates)

        # Independently recompute Strategy 1 from the same deterministic model, without
        # reusing any of the engine's internal helper methods.
        batch = engine.build_batch(prompt, candidates)
        with torch.no_grad():
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                position_ids=batch["position_ids"],
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        log_probs = F.log_softmax(out.logits.float(), dim=-1)

        for i, cand_len in enumerate(batch["candidate_lens"]):
            prompt_len = batch["prompt_lens"][i]
            total = 0.0
            for t in range(cand_len):
                pred_pos = prompt_len - 1 + t
                target_tok = int(batch["input_ids"][i, prompt_len + t].item())
                total += float(log_probs[i, pred_pos, target_tok].item())
            expected = total / cand_len
            self.assertAlmostEqual(result.scores[i].sequence_logprob, expected, places=5)

    def test_variable_length_candidates_use_correct_denominator(self):
        model = MockCandidateModel()
        tokenizer = MockTokenizer()
        engine = CandidateMultiLayerFusionEngine(model, tokenizer)

        result = engine.score("context", ["a", "abcdefgh"])
        self.assertEqual(result.scores[0].num_candidate_tokens, 1)
        self.assertEqual(result.scores[1].num_candidate_tokens, 8)
        # Padding must not leak into the normalized log-likelihood of the shorter candidate.
        for s in result.scores:
            self.assertTrue(abs(s.sequence_logprob) < 50.0)


@unittest.skipUnless(HAS_TORCH, "PyTorch required for candidate fusion tests")
class TestCandidateFusionStrategy2ETF(unittest.TestCase):
    def test_multilayer_etf_embeddings_are_unit_norm_and_frame_is_equiangular(self):
        model = MockCandidateModel()
        tokenizer = MockTokenizer()
        engine = CandidateMultiLayerFusionEngine(model, tokenizer)

        candidates = ["red", "green", "blue", "yellow"]
        batch = engine.build_batch("color:", candidates)
        with torch.no_grad():
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                position_ids=batch["position_ids"],
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )

        self_alignment, margin = engine._strategy2_multilayer_etf(
            out.hidden_states, batch["prompt_lens"], batch["candidate_lens"]
        )
        self.assertEqual(self_alignment.shape, (4,))
        self.assertEqual(margin.shape, (4,))
        self.assertFalse(torch.isnan(self_alignment).any())
        self.assertFalse(torch.isinf(margin).any())

        # The frame itself (independent of which layers fed it) must be an exact simplex ETF.
        from gen_zero.nanocore.action_etf_embedding import generate_simplex_etf
        import numpy as np

        etf = generate_simplex_etf(k=4, dim=3)
        norms = np.linalg.norm(etf, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)
        gram = etf @ etf.T
        off_diag = gram[~np.eye(4, dtype=bool)]
        np.testing.assert_allclose(off_diag, -1.0 / 3.0, atol=1e-10)

    def test_layer_offsets_select_distinguishable_layers(self):
        model = MockCandidateModel(num_layers=10)
        tokenizer = MockTokenizer()
        engine_default = CandidateMultiLayerFusionEngine(model, tokenizer, layer_offsets=DEFAULT_LAYER_OFFSETS)
        engine_last_only = CandidateMultiLayerFusionEngine(
            model, tokenizer, layer_offsets=(-1,)
        )

        candidates = ["alpha", "beta", "gamma"]
        result_multi = engine_default.score("q:", candidates)
        result_last = engine_last_only.score("q:", candidates)

        # Both must produce valid, finite alignments, and (since the mock scales each
        # layer differently) the multi-layer signal differs from the last-layer-only signal.
        for s in result_multi.scores + result_last.scores:
            self.assertTrue(abs(s.etf_alignment) < 10.0)
        alignments_multi = [s.etf_alignment for s in result_multi.scores]
        alignments_last = [s.etf_alignment for s in result_last.scores]
        self.assertNotEqual(alignments_multi, alignments_last)

    def test_out_of_range_layer_offset_raises(self):
        model = MockCandidateModel(num_layers=3)
        tokenizer = MockTokenizer()
        engine = CandidateMultiLayerFusionEngine(model, tokenizer, layer_offsets=(-99,))
        with self.assertRaises(ValueError):
            engine.score("q:", ["a", "b"])


@unittest.skipUnless(HAS_TORCH, "PyTorch required for candidate fusion tests")
class TestCandidateFusionZeroTokenGeneration(unittest.TestCase):
    def test_exactly_one_forward_pass_and_zero_generated_tokens(self):
        model = MockCandidateModel()
        tokenizer = MockTokenizer()
        engine = CandidateMultiLayerFusionEngine(model, tokenizer)

        result = engine.score("prompt text", ["candidate one", "candidate two", "c"])

        self.assertIsInstance(result, CandidateFusionResult)
        self.assertEqual(model.forward_call_count, 1)
        self.assertEqual(model.generate_call_count, 0)
        self.assertEqual(result.tokens_generated, 0)
        self.assertEqual(result.forward_pass_count, 1)
        self.assertFalse(hasattr(engine, "generate"))

    def test_fused_confidence_is_a_calibrated_distribution(self):
        model = MockCandidateModel()
        tokenizer = MockTokenizer()
        engine = CandidateMultiLayerFusionEngine(model, tokenizer)

        result = engine.score("prompt", ["short", "a much longer candidate span"])
        total = sum(s.fused_confidence for s in result.scores)
        self.assertAlmostEqual(total, 1.0, places=5)
        for s in result.scores:
            self.assertGreaterEqual(s.fused_confidence, 0.0)
            self.assertLessEqual(s.fused_confidence, 1.0)
        self.assertEqual(result.best_index, int(
            max(range(len(result.scores)), key=lambda i: result.scores[i].fused_confidence)
        ))

    def test_single_candidate_degenerate_case_is_finite(self):
        model = MockCandidateModel()
        tokenizer = MockTokenizer()
        engine = CandidateMultiLayerFusionEngine(model, tokenizer)

        result = engine.score("prompt", ["only-option"])
        self.assertEqual(len(result.scores), 1)
        self.assertEqual(result.best_index, 0)
        self.assertAlmostEqual(result.scores[0].fused_confidence, 1.0, places=5)
        self.assertFalse(torch.isnan(torch.tensor(result.scores[0].etf_margin)))
        self.assertFalse(torch.isinf(torch.tensor(result.scores[0].etf_margin)))

    def test_candidate_order_permutation_equivariance(self):
        """Permuting candidates must permute scores equivariantly with zero slot bias."""
        model = MockCandidateModel()
        tokenizer = MockTokenizer()
        engine = CandidateMultiLayerFusionEngine(model, tokenizer)

        # Distinct terminal characters ('a', 't', 'o') so terminal-token hidden states
        # cannot accidentally collide across candidates in this per-char mock tokenizer.
        order_a = ["alpha", "belt", "gizmo"]
        order_b = ["gizmo", "alpha", "belt"]

        res_a = engine.score("question prompt", order_a)
        res_b = engine.score("question prompt", order_b)

        # Winning candidate must be identical regardless of order
        self.assertEqual(res_a.best_candidate, res_b.best_candidate)

        # Scores for 'alpha' must be strictly identical across both runs
        score_alpha_a = next(s for s in res_a.scores if s.candidate == "alpha")
        score_alpha_b = next(s for s in res_b.scores if s.candidate == "alpha")
        self.assertAlmostEqual(score_alpha_a.sequence_logprob, score_alpha_b.sequence_logprob, places=5)
        self.assertAlmostEqual(score_alpha_a.etf_alignment, score_alpha_b.etf_alignment, places=5)
        self.assertAlmostEqual(score_alpha_a.etf_margin, score_alpha_b.etf_margin, places=5)
        self.assertAlmostEqual(score_alpha_a.fused_confidence, score_alpha_b.fused_confidence, places=5)

    def test_calibration_entropy_reflects_probabilities(self):
        """Reported calibration entropy must accurately reflect the returned distribution."""
        model = MockCandidateModel()
        tokenizer = MockTokenizer()
        engine = CandidateMultiLayerFusionEngine(model, tokenizer)

        res = engine.score("q:", ["opt_a", "opt_b", "opt_c"])
        p = np.array([s.fused_confidence for s in res.scores])
        expected_entropy = float(-np.sum(p * np.log(np.clip(p, 1e-12, 1.0))) / np.log(3))
        self.assertAlmostEqual(res.calibration_entropy, expected_entropy, places=5)


@unittest.skipUnless(HAS_TORCH, "PyTorch required for candidate fusion tests")
class TestCandidateFusionLengthPenaltyAlpha(unittest.TestCase):
    def _summed_logprob(self, model, tokenizer, engine, prompt, candidate):
        """Recomputes the raw (un-normalized) summed log-likelihood independently."""
        batch = engine.build_batch(prompt, [candidate])
        with torch.no_grad():
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                position_ids=batch["position_ids"],
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        log_probs = F.log_softmax(out.logits.float(), dim=-1)
        prompt_len = batch["prompt_lens"][0]
        cand_len = batch["candidate_lens"][0]
        total = 0.0
        for t in range(cand_len):
            pred_pos = prompt_len - 1 + t
            target_tok = int(batch["input_ids"][0, prompt_len + t].item())
            total += float(log_probs[0, pred_pos, target_tok].item())
        return total, cand_len

    def test_alpha_one_matches_plain_mean(self):
        model = MockCandidateModel()
        tokenizer = MockTokenizer()
        engine = CandidateMultiLayerFusionEngine(model, tokenizer, length_penalty_alpha=1.0)

        result = engine.score("the cat", ["jumped over"])
        total, cand_len = self._summed_logprob(model, tokenizer, engine, "the cat", "jumped over")
        self.assertAlmostEqual(result.scores[0].sequence_logprob, total / cand_len, places=5)

    def test_alpha_half_matches_sqrt_denominator(self):
        model = MockCandidateModel()
        tokenizer = MockTokenizer()
        engine = CandidateMultiLayerFusionEngine(model, tokenizer, length_penalty_alpha=0.5)

        result = engine.score("the cat", ["jumped over"])
        total, cand_len = self._summed_logprob(model, tokenizer, engine, "the cat", "jumped over")
        self.assertAlmostEqual(result.scores[0].sequence_logprob, total / (cand_len**0.5), places=5)

    def test_different_alphas_change_normalized_score(self):
        model = MockCandidateModel()
        tokenizer = MockTokenizer()
        engine_alpha1 = CandidateMultiLayerFusionEngine(model, tokenizer, length_penalty_alpha=1.0)
        engine_alpha0 = CandidateMultiLayerFusionEngine(model, tokenizer, length_penalty_alpha=0.0)

        result1 = engine_alpha1.score("context", ["a much longer candidate span"])
        result0 = engine_alpha0.score("context", ["a much longer candidate span"])

        # alpha=0.0 means no length normalization: score equals the raw summed log-likelihood.
        cand_len = result1.scores[0].num_candidate_tokens
        self.assertAlmostEqual(
            result0.scores[0].sequence_logprob,
            result1.scores[0].sequence_logprob * cand_len,
            places=4,
        )


@unittest.skipUnless(HAS_TORCH, "PyTorch required for candidate fusion tests")
class TestCandidateFusionJointTokenization(unittest.TestCase):
    def test_joint_tokenization_merges_across_boundary(self):
        model = MockCandidateModel()
        tokenizer = MockMergingTokenizer()
        engine_joint = CandidateMultiLayerFusionEngine(model, tokenizer, joint_tokenization=True)
        engine_naive = CandidateMultiLayerFusionEngine(model, tokenizer, joint_tokenization=False)

        prompt = "xa"      # tokenizes to [x, a]
        candidates = ["bY"]  # "a" (prompt's last char) + "b" (candidate's first char) = "ab" merge

        batch_joint = engine_joint.build_batch(prompt, candidates)
        batch_naive = engine_naive.build_batch(prompt, candidates)

        # The joint row must be EXACTLY the tokenizer's own joint encoding of "xabY" (never a
        # concatenation of the separately-encoded prompt and candidate, which would duplicate
        # or corrupt whatever token the merge absorbed).
        real_len_joint = batch_joint["prompt_lens"][0] + batch_joint["candidate_lens"][0]
        joint_row = batch_joint["input_ids"][0, :real_len_joint].tolist()
        self.assertEqual(joint_row, tokenizer.encode(prompt + candidates[0]))

        prompt_len_joint = batch_joint["prompt_lens"][0]
        prompt_len_naive = batch_naive["prompt_lens"][0]
        joint_cand_ids = batch_joint["input_ids"][0, prompt_len_joint:real_len_joint].tolist()
        naive_cand_ids = batch_naive["input_ids"][0, prompt_len_naive:].tolist()

        self.assertNotEqual(joint_cand_ids, naive_cand_ids)
        self.assertIn(MockMergingTokenizer.MERGED_TOKEN_ID, joint_cand_ids)
        self.assertNotIn(MockMergingTokenizer.MERGED_TOKEN_ID, naive_cand_ids)
        # The merge absorbed the prompt's last char, so the joint row's prompt boundary is
        # shorter than the standalone prompt length.
        self.assertLess(prompt_len_joint, batch_joint["prompt_len"])

    def test_joint_tokenization_no_boundary_interaction_is_unaffected(self):
        model = MockCandidateModel()
        tokenizer = MockMergingTokenizer()
        engine_joint = CandidateMultiLayerFusionEngine(model, tokenizer, joint_tokenization=True)
        engine_naive = CandidateMultiLayerFusionEngine(model, tokenizer, joint_tokenization=False)

        # No "ab" spans the boundary here, so joint and naive tokenization must agree.
        prompt = "hello"
        candidates = ["world"]

        batch_joint = engine_joint.build_batch(prompt, candidates)
        batch_naive = engine_naive.build_batch(prompt, candidates)
        self.assertEqual(batch_joint["prompt_lens"], batch_naive["prompt_lens"])
        self.assertEqual(batch_joint["candidate_lens"], batch_naive["candidate_lens"])
        real_len = batch_joint["prompt_lens"][0] + batch_joint["candidate_lens"][0]
        self.assertEqual(
            batch_joint["input_ids"][0, :real_len].tolist(),
            batch_naive["input_ids"][0, :real_len].tolist(),
        )

    def test_joint_tokenization_scores_without_crashing(self):
        model = MockCandidateModel()
        tokenizer = MockMergingTokenizer()
        engine = CandidateMultiLayerFusionEngine(model, tokenizer, joint_tokenization=True)

        # "zW" first (no boundary merge) so candidate 0 still satisfies score()'s fail-closed
        # prompt_hidden_state/prompt_last_logits invariant; "bY" second still exercises a
        # boundary merge (row 1) so joint tokenization's own crash-safety is still covered.
        result = engine.score("xa", ["zW", "bY"])
        self.assertEqual(len(result.scores), 2)
        for s in result.scores:
            self.assertFalse(torch.isnan(torch.tensor(s.fused_confidence)))

    def test_joint_tokenization_degenerate_prefix_raises_value_error(self):
        """Fail-closed regression test: strictly no silent fallbacks on degenerate prefix."""
        model = MockCandidateModel()

        class DegenerateTokenizer(MockTokenizer):
            def encode(self, text, add_special_tokens=False):
                if text == "prompt":
                    return [10, 20]
                if text == "prompt_mutated":
                    return [99, 99]  # Common prefix is 0 tokens
                return [10]

        tokenizer = DegenerateTokenizer()
        engine = CandidateMultiLayerFusionEngine(model, tokenizer, joint_tokenization=True)
        with self.assertRaises(ValueError):
            engine.build_batch("prompt", ["_mutated"])


@unittest.skipUnless(HAS_TORCH, "PyTorch required for candidate fusion tests")
class TestCandidateFusionPromptReuse(unittest.TestCase):
    """`CandidateFusionResult.prompt_hidden_state`/`prompt_last_logits`/`forward_ms`: the
    fields that let a caller reuse row 0 of the batched forward instead of running a second,
    standalone `model(prompt)` forward.
    """

    def test_prompt_hidden_state_and_logits_match_standalone_forward(self):
        model = MockCandidateModel()
        tokenizer = MockTokenizer()
        engine = CandidateMultiLayerFusionEngine(model, tokenizer)

        prompt = "the cat"
        # Deliberately variable-length candidates so row 0 (the shorter one) sits next to
        # padding in the batch; the extraction must still read row 0 at its own boundary
        # position, not at the batch's (longer) max length.
        candidates = ["sat", "jumped over"]
        result = engine.score(prompt, candidates)

        self.assertIsNotNone(result.prompt_hidden_state)
        self.assertIsNotNone(result.prompt_last_logits)
        self.assertIsInstance(result.prompt_hidden_state, np.ndarray)
        self.assertEqual(result.prompt_hidden_state.shape, (model.hidden_dim,))
        self.assertEqual(tuple(result.prompt_last_logits.shape), (model.vocab_size,))

        # Independently reproduce via a standalone forward over just the prompt tokens
        # (mirrors what `A100Engine.forward(prompt)` would compute), without reusing any
        # of the engine's internal helpers.
        prompt_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(len(prompt_ids)).unsqueeze(0)
        with torch.no_grad():
            standalone = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        expected_h = standalone.hidden_states[-1][0, -1, :].float().numpy()
        expected_logits = standalone.logits[0, -1, :].float()

        np.testing.assert_allclose(result.prompt_hidden_state, expected_h, atol=1e-6)
        self.assertTrue(torch.allclose(result.prompt_last_logits, expected_logits, atol=1e-6))

    def test_boundary_merge_on_candidate_zero_raises(self):
        """Fail-closed regression test: when candidate 0's joint tokenization merges across
        the prompt boundary, `prompt_hidden_state`/`prompt_last_logits` would only be
        reproducible over a shorter, wrong prefix, so `score()` must raise instead of
        silently returning a mismatched value.
        """
        model = MockCandidateModel()
        tokenizer = MockMergingTokenizer()
        engine = CandidateMultiLayerFusionEngine(model, tokenizer, joint_tokenization=True)

        # "bY" as candidate 0 merges "a"(prompt's last char) + "b"(candidate's first char)
        # across the boundary (see test_joint_tokenization_merges_across_boundary), so
        # prompt_lens[0] < prompt_len for this row specifically.
        with self.assertRaises(ValueError):
            engine.score("xa", ["bY", "zW"])

    def test_forward_ms_is_populated_and_non_negative(self):
        model = MockCandidateModel()
        tokenizer = MockTokenizer()
        engine = CandidateMultiLayerFusionEngine(model, tokenizer)

        result = engine.score("prompt text", ["a", "bb"])
        self.assertIsNotNone(result.forward_ms)
        self.assertGreaterEqual(result.forward_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
