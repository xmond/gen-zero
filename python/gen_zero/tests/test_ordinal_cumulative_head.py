"""Unit and Integration Tests for Ordinal Cumulative Head.

Validates the Proportional Odds model on SummEval continuous ordinal manifolds:
1. Cumulative monotonicity: P(Y > 1) >= P(Y > 2) >= P(Y > 3) >= P(Y > 4).
2. Discrete probabilities sum to 1.0.
3. Expected quality score: E[Y] = 1 + \sum P(Y > k).
4. Quantized decision: \hat{y} = \arg\max P(Y = k).
5. Wasserstein metric topology: penalizes distant prediction errors exponentially harder than neighboring scores (|4 - 5| vs |1 - 5|).
6. Entropy explosion elimination: prevents false 1.56 bit categorical entropy and 27/30 STOP gate trigger.
7. PyTorch module execution and differentiability (when available).
"""
# anti-leakage: allow-mock-tensor

import unittest
import numpy as np

from gen_zero.model.choice_head import (
    OrdinalCumulativeHead,
    OrdinalEvaluationResult,
    PyTorchOrdinalCumulativeHead,
    HAS_TORCH,
)


class TestOrdinalCumulativeHead(unittest.TestCase):
    def setUp(self):
        self.head = OrdinalCumulativeHead(dimension=1024, num_classes=5)

    def test_thresholds_and_num_classes(self):
        self.assertEqual(self.head.num_classes, 5)
        self.assertEqual(len(self.head.thresholds), 4)
        # Verify strictly monotonic cutoffs
        for i in range(1, len(self.head.thresholds)):
            self.assertGreater(self.head.thresholds[i], self.head.thresholds[i - 1])

    def test_proportional_odds_cumulative_monotonicity(self):
        # High quality score: s = 2.0
        res = self.head.evaluate_scalar(2.0)
        self.assertEqual(len(res.cumulative_probs), 4)
        self.assertEqual(len(res.probabilities), 5)

        # Monotonicity check
        for i in range(1, 4):
            self.assertGreaterEqual(res.cumulative_probs[i - 1], res.cumulative_probs[i])

        # Probabilities sum to 1.0
        self.assertAlmostEqual(sum(res.probabilities), 1.0, places=5)

        # Expected score formula: E[Y] = 1 + \sum P(Y > k)
        expected_manual = 1.0 + sum(res.cumulative_probs)
        self.assertAlmostEqual(res.expected_score, expected_manual, places=5)
        self.assertGreaterEqual(res.expected_score, 1.0)
        self.assertLessEqual(res.expected_score, 5.0)

        # High quality should quantize to 4 or 5
        self.assertIn(res.quantized_decision, [4, 5])

    def test_low_quality_projection(self):
        # Low quality score: s = -2.0
        res = self.head.evaluate_scalar(-2.0)
        self.assertAlmostEqual(sum(res.probabilities), 1.0, places=5)
        self.assertLess(res.expected_score, 2.0)
        self.assertEqual(res.quantized_decision, 1)

    def test_wasserstein_metric_topology(self):
        """Validates that |4 - 5| is penalized far less than |1 - 5|."""
        target_y = 5

        # Neighboring prediction (pred = 4)
        probs_neighbor = [0.01, 0.02, 0.07, 0.60, 0.30]
        loss_neighbor = self.head.wasserstein_exponential_loss(probs_neighbor, target_y)
        dist_neighbor = self.head.wasserstein_distance(probs_neighbor, target_y)

        # Distant prediction (pred = 1)
        probs_distant = [0.60, 0.30, 0.07, 0.02, 0.01]
        loss_distant = self.head.wasserstein_exponential_loss(probs_distant, target_y)
        dist_distant = self.head.wasserstein_distance(probs_distant, target_y)

        self.assertLess(dist_neighbor, dist_distant)
        # Exponential penalty: distant error penalty is vastly higher
        self.assertGreater(loss_distant, loss_neighbor * 4.0)

    def test_summeval_entropy_explosion_elimination(self):
        """Validates that boundary predictions (between 4 and 5) don't trigger Gate=STOP.
        
        Categorical softmax blows up to ~1.56 bits when probability mass splits across scores.
        Ordinal cumulative head respects continuous metric topology, maintaining low metric entropy.
        """
        # Boundary score between 4 and 5
        res = self.head.evaluate_scalar(1.5)

        # Normalized metric entropy should be low (well below 0.70 STOP gate threshold)
        self.assertLess(res.normalized_entropy, 0.65)
        self.assertLess(res.ordinal_variance, 1.5)

    def test_latent_vector_evaluation(self):
        latent = np.zeros(1024, dtype=np.float64)
        latent[0] = 2.5
        res = self.head.evaluate(latent)
        self.assertEqual(res.quantized_decision, 5)
        self.assertGreater(res.expected_score, 4.0)

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
    def test_pytorch_ordinal_cumulative_head(self):
        import torch
        model = PyTorchOrdinalCumulativeHead(hidden_dim=128, num_classes=5)

        x = torch.randn(4, 128, requires_grad=True)
        out = model(x)

        probs = out["probabilities"]
        self.assertEqual(probs.shape, (4, 5))
        # Probabilities sum to 1.0
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5))

        expected = out["expected_score"]
        self.assertEqual(expected.shape, (4, 1))
        self.assertTrue((expected >= 1.0).all() and (expected <= 5.0).all())

        # Test loss and backward gradient flow
        targets = torch.tensor([5, 4, 1, 2], dtype=torch.long)
        loss = model.compute_wasserstein_loss(probs, targets)
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(model.base_threshold.grad)
        self.assertIsNotNone(model.threshold_deltas.grad)


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestMonotonicProportionalOddsHead(unittest.TestCase):
    """Torch head with softplus cutpoints: probabilities can never go negative."""

    def setUp(self):
        import torch
        from gen_zero.model.ordinal_head import ProportionalOddsHead

        torch.manual_seed(0)
        self.torch = torch
        self.cls = ProportionalOddsHead
        self.head = ProportionalOddsHead(hidden_dim=16, num_classes=5)

    def _assert_valid(self, out):
        torch = self.torch
        p = out["probabilities"]
        self.assertTrue(torch.isfinite(p).all())
        self.assertTrue((p >= 0).all())
        self.assertTrue(torch.allclose(p.sum(-1), torch.ones(p.shape[0]), atol=1e-5))

    def _assert_ordered(self, b):
        # Strict while float32 can resolve the 1e-4 gap (|b| < 500).
        # Beyond that a tie is possible, but never an inversion.
        self.assertTrue((b[1:] >= b[:-1]).all())
        if b.abs().max().item() < 500:
            self.assertTrue((b[1:] > b[:-1]).all())

    def test_positive_probabilities(self):
        self._assert_valid(self.head(self.torch.randn(64, 16) * 10))

    def test_initial_thresholds_strictly_increase(self):
        b = self.head.thresholds
        self.assertEqual(b.shape, (4,))
        self.assertTrue((b[1:] > b[:-1]).all())

    def test_random_perturbations_never_negative(self):
        torch = self.torch
        x = torch.randn(32, 16) * 5
        for _ in range(300):
            with torch.no_grad():
                self.head.first_cutpoint.add_(torch.randn(()) * 5)
                self.head.raw_cutpoint_deltas.add_(torch.randn_like(self.head.raw_cutpoint_deltas) * 30)
                self.head.score.weight.add_(torch.randn_like(self.head.score.weight))
            self._assert_ordered(self.head.thresholds)
            self._assert_valid(self.head(x))

    def test_extreme_raw_deltas_keep_order(self):
        torch = self.torch
        for v in (-1e4, 1e4):
            with torch.no_grad():
                self.head.raw_cutpoint_deltas.fill_(v)
            self._assert_ordered(self.head.thresholds)
            self._assert_valid(self.head(torch.randn(8, 16)))

    def test_monotonic_after_sgd_with_large_lr(self):
        torch = self.torch
        opt = torch.optim.SGD(self.head.parameters(), lr=5.0)
        x = torch.randn(32, 16)
        y = torch.randint(1, 6, (32,))
        for _ in range(50):
            opt.zero_grad()
            self.head.nll_loss(x, y).backward()
            opt.step()
            self._assert_ordered(self.head.thresholds)
        self._assert_valid(self.head(x))

    def test_gradients_reach_all_parameters_for_both_losses(self):
        torch = self.torch
        x = torch.randn(8, 16, requires_grad=True)
        y = torch.tensor([1, 2, 3, 4, 5, 1, 3, 5])
        for name in ("nll_loss", "loss"):
            self.head.zero_grad()
            x.grad = None
            loss_fn = self.head.nll_loss if name == "nll_loss" else self.head.loss
            loss_fn(x, y).backward()
            self.assertIsNotNone(x.grad)
            for pname, p in self.head.named_parameters():
                self.assertIsNotNone(p.grad, f"{name}: no grad for {pname}")
                self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(self.head.raw_cutpoint_deltas.grad.abs().sum().item(), 0.0)

    def test_nll_matches_manual_log_likelihood(self):
        torch = self.torch
        x = torch.randn(5, 16)
        y = torch.tensor([1, 2, 3, 4, 5])
        p = self.head(x)["probabilities"]
        manual = -torch.log(p[torch.arange(5), y - 1]).mean()
        self.assertTrue(torch.allclose(self.head.nll_loss(x, y), manual, atol=1e-6))

    def test_training_learns_ordinal_signal(self):
        torch = self.torch
        head = self.cls(hidden_dim=1, num_classes=5)
        x = torch.linspace(-2, 2, 200).unsqueeze(1)
        y = (torch.bucketize(x.squeeze(1), torch.tensor([-1.0, -0.3, 0.3, 1.0])) + 1)
        opt = torch.optim.Adam(head.parameters(), lr=0.05)
        for _ in range(300):
            opt.zero_grad()
            head.nll_loss(x, y).backward()
            opt.step()
        self.assertGreater((head.predict(x) == y).float().mean().item(), 0.8)
        self.assertLess((head.expected_score(x) - y.float()).abs().mean().item(), 0.6)

    def test_predict_and_expected_score(self):
        torch = self.torch
        x = torch.randn(10, 16)
        pred = self.head.predict(x)
        self.assertEqual(pred.shape, (10,))
        self.assertTrue(((pred >= 1) & (pred <= 5)).all())
        es = self.head.expected_score(x)
        self.assertEqual(es.shape, (10,))
        self.assertTrue(((es >= 1) & (es <= 5)).all())
        manual = 1 + self.head(x)["cumulative_probabilities"].sum(-1)
        self.assertTrue(torch.allclose(es, manual, atol=1e-5))

    def test_two_class_head(self):
        head = self.cls(hidden_dim=4, num_classes=2)
        out = head(self.torch.randn(3, 4))
        self._assert_valid(out)
        self.assertEqual(head.thresholds.shape, (1,))


if __name__ == "__main__":
    unittest.main()
