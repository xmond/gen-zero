"""Unit tests for Issue #72: Candidate Action Space Simplex ETF Equiangular Tight Frame Embedding.

Verifies:
1. Strict equiangularity: <v_i, v_j> = -1 / (K-1) and ||v_i||_2 = 1.0.
2. Permutation equivariance: 0.00% argmax flip rate under candidate action order reordering.
3. Attention shunting mitigation: eliminates probability dilution on near-synonym action pairs.
4. ActionETFChoiceHead end-to-end decision and verification report.
5. PyTorch differentiable head integration (if torch is available).
"""
# anti-leakage: allow-mock-tensor

import math
import unittest
import numpy as np

from gen_zero.nanocore.action_etf_embedding import (
    ActionSpaceETFEmbedding,
    ETFVerificationReport,
    generate_simplex_etf,
)
from gen_zero.nanocore.choice_head import (
    ActionETFChoiceHead,
    ChoiceDecisionResult,
)

try:
    import torch
    from gen_zero.nanocore.choice_head import PyTorchActionETFChoiceHead
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestIssue72ActionSimplexETF(unittest.TestCase):
    """Test suite for Issue #72 Simplex ETF action space embedding."""

    def test_01_simplex_etf_mathematical_guarantees(self):
        """Validates that Simplex ETF satisfies theoretical tight frame properties for all K."""
        dim = 64
        for k in [2, 3, 4, 5, 8, 12, 16]:
            etf = generate_simplex_etf(k=k, dim=dim)
            self.assertEqual(etf.shape, (k, dim))

            # 1. Unit norm check: ||v_i||_2 = 1.0
            norms = np.linalg.norm(etf, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-12, err_msg=f"Norms not unit for K={k}")

            # 2. Pairwise inner product: <v_i, v_j> = -1 / (K - 1)
            gram = np.dot(etf, etf.T)
            expected_ip = -1.0 / (k - 1)
            off_diag = gram[~np.eye(k, dtype=bool)]

            np.testing.assert_allclose(
                off_diag, expected_ip, atol=1e-12,
                err_msg=f"Inner product deviation for K={k}: expected {expected_ip}, got {off_diag}"
            )

            # 3. Variance across pairs must be 0 (equiangular)
            var_ip = np.var(off_diag)
            self.assertLessEqual(var_ip, 1e-15, f"Variance > 1e-15 for K={k}")

            # 4. Angle in degrees
            expected_angle_deg = math.degrees(math.acos(expected_ip))
            angles = np.degrees(np.arccos(np.clip(off_diag, -1.0, 1.0)))
            np.testing.assert_allclose(angles, expected_angle_deg, atol=1e-10)

    def test_02_permutation_equivariance_zero_argmax_flip(self):
        """Verifies that reordering candidate actions produces 0.00% argmax flip rate."""
        head = ActionETFChoiceHead(hidden_dim=64, action_dim=64, blend_alpha=0.9, seed=123)
        state = "Production memory spike detected on node worker-04. Recommend remediation."
        base_actions = ["ROLLBACK", "RESTART_POD", "SCALE_UP", "WAIT_METRICS"]

        base_result = head.decide(state, base_actions)
        expected_winner = base_result.selected_action

        import itertools
        permutations = list(itertools.permutations(base_actions))

        for perm in permutations:
            res = head.decide(state, list(perm))
            self.assertEqual(
                res.selected_action, expected_winner,
                f"Argmax flipped on permutation {perm}: expected {expected_winner}, got {res.selected_action}"
            )
            # Compare every action by identity, not its position in the list.
            for action_id in base_actions:
                self.assertAlmostEqual(
                    res.action_probabilities[action_id],
                    base_result.action_probabilities[action_id],
                    places=6,
                    msg=f"score changed for {action_id} under {perm}",
                )

    def test_03_high_confusion_synonym_mitigation(self):
        """Tests that Simplex ETF resolves semantic attention shunting between near-synonyms."""
        engine = ActionSpaceETFEmbedding(dim=64, blend_alpha=1.0, temperature=1.0)
        actions = ["ABORT", "CANCEL", "PROCEED"]

        # Synthetic semantic vectors where ABORT and CANCEL have 0.90 correlation
        sem_abort = np.ones(64) / np.sqrt(64)
        noise = np.random.RandomState(42).randn(64) * 0.05
        sem_cancel = (sem_abort + noise) / np.linalg.norm(sem_abort + noise)
        sem_proceed = -sem_abort  # Opposite

        sem_matrix = np.vstack([sem_abort, sem_cancel, sem_proceed])

        # Verify semantic correlation before ETF
        raw_corr = np.dot(sem_abort, sem_cancel)
        self.assertGreater(raw_corr, 0.85, "Synthetic synonyms must have high raw correlation")

        # In pure ETF space: correlation is exactly -1 / (3 - 1) = -0.5
        etf_vectors = engine.embed_actions(actions, semantic_embeddings=sem_matrix, alpha=1.0)
        etf_corr = np.dot(etf_vectors[0], etf_vectors[1])
        self.assertAlmostEqual(etf_corr, -0.5, places=10)

        # Audit frame verification
        report = engine.verify_frame(etf_vectors)
        self.assertTrue(report.is_equiangular)
        self.assertAlmostEqual(report.expected_inner_product, -0.5, places=10)
        self.assertAlmostEqual(report.min_angle_degrees, 120.0, places=5)
        self.assertAlmostEqual(report.max_angle_degrees, 120.0, places=5)

    def test_04_choice_head_end_to_end_decision(self):
        """Tests ActionETFChoiceHead decision pipeline and report generation."""
        head = ActionETFChoiceHead(hidden_dim=32, action_dim=32, blend_alpha=0.85, seed=42)
        state_vec = np.random.RandomState(101).randn(32)
        candidates = ["ACT_A", "ACT_B", "ACT_C", "ACT_D"]

        result = head.decide(state_vec, candidates)
        self.assertIsInstance(result, ChoiceDecisionResult)
        self.assertIn(result.selected_action, candidates)
        self.assertGreater(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertAlmostEqual(sum(result.action_probabilities.values()), 1.0, places=6)
        self.assertTrue(result.is_equiangular)
        self.assertGreater(result.snr, 0.0)

        d = result.to_dict()
        self.assertIn("verification", d)
        self.assertIn("attention_entropy", d)

    def test_05_pytorch_differentiable_head(self):
        """Tests PyTorch module backpropagation through ETF manifold if torch is installed."""
        if not HAS_TORCH:
            self.skipTest("PyTorch not installed; skipping differentiable head test.")

        torch.manual_seed(42)
        head = PyTorchActionETFChoiceHead(hidden_dim=32, action_dim=32, blend_alpha=0.85)
        state_tensor = torch.randn(4, 32, requires_grad=True)
        actions = ["OP_1", "OP_2", "OP_3"]

        probs, logits = head(state_tensor, actions)
        self.assertEqual(probs.shape, (4, 3))
        self.assertEqual(logits.shape, (4, 3))

        # Check gradient flow
        loss = probs.sum()
        loss.backward()
        self.assertIsNotNone(state_tensor.grad)
        self.assertFalse(torch.isnan(state_tensor.grad).any())


if __name__ == "__main__":
    unittest.main()
