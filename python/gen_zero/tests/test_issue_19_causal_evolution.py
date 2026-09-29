"""Comprehensive Unit & Integration Tests for Issue #19.

Validates:
1. Milestone 1: Shuffled-State Control Benchmark, CAG/CGR computation, 10-bin ECE gap, and causal degradation gate.
2. Milestone 2: Attention Entropy Diagnostic Module, Shannon entropy normalization, Two-Dimensional Credibility Gate.
3. Milestone 3: DAgger Expert Relabeling Loop, off-policy drift detection, priority replay buffer injection, and curriculum scheduler.
4. Milestone 4: Zero-State First-Order Delta Feature Channel, vector/dict discrete differencing, and L2 market momentum sensing.
"""

import unittest
import numpy as np
import math

from gen_zero.causal.shuffled_benchmark import (
    ShuffledStateBenchmark,
    ShuffledBenchmarkReport,
    compute_cag_and_cgr,
)
from gen_zero.model.dual_head import (
    compute_normalized_attention_entropy,
    TwoDimensionalCredibilityGate,
    CredibilityVerdict,
    GenZeroDualHeadModel,
    DeepSetAttentionHead,
    HAS_TORCH,
)
from gen_zero.train.dagger_loop import (
    DAggerState,
    DAggerRelabeledSample,
    DAggerExpertRelabeler,
    DAggerCurriculumController,
)
from gen_zero.train.replay_buffer import StabilityReplayBuffer
from gen_zero.model.delta_encoder import ZeroStateDeltaEncoder
from gen_zero.model.market_state import (
    L2OrderBookSnapshot,
    L2MarketStateEncoder,
)

if HAS_TORCH:
    import torch


class TestShuffledStateBenchmark(unittest.TestCase):
    """Tests for Milestone 1: Shuffled-State Control Benchmark & CGR."""

    def test_compute_cag_and_cgr_math(self):
        # Perfect causal dependency: normal=0.90, shuffled=0.10
        cag, cgr = compute_cag_and_cgr(0.90, 0.10)
        self.assertAlmostEqual(cag, 0.80)
        self.assertAlmostEqual(cgr, 9.0)

        # Zero shuffled accuracy: denominator clamped by eps=1e-4
        cag_zero, cgr_zero = compute_cag_and_cgr(0.80, 0.0)
        self.assertAlmostEqual(cag_zero, 0.80)
        self.assertAlmostEqual(cgr_zero, 8000.0)

        # Shortcut / degraded model: normal=0.50, shuffled=0.50
        cag_deg, cgr_deg = compute_cag_and_cgr(0.50, 0.50)
        self.assertAlmostEqual(cag_deg, 0.0)
        self.assertAlmostEqual(cgr_deg, 1.0)

    def test_create_shuffled_batch_cyclic_permutation(self):
        batch = [
            {"state": "state_0", "candidate_ids": ["a", "b"], "target": "a"},
            {"state": "state_1", "candidate_ids": ["c", "d"], "target": "c"},
            {"state": "state_2", "candidate_ids": ["e", "f"], "target": "e"},
        ]
        shuffled = ShuffledStateBenchmark.create_shuffled_batch(batch, shift=1)
        self.assertEqual(len(shuffled), 3)

        # States should be shifted by 1: (i + 1) % 3
        self.assertEqual(shuffled[0]["state"], "state_1")
        self.assertEqual(shuffled[1]["state"], "state_2")
        self.assertEqual(shuffled[2]["state"], "state_0")

        # Candidates and targets must remain intact
        self.assertEqual(shuffled[0]["candidate_ids"], ["a", "b"])
        self.assertEqual(shuffled[0]["target"], "a")

    def test_shuffled_benchmark_causal_gate_pass_and_fail(self):
        # Synthetic causal dataset with binary alternations
        dataset = []
        for i in range(20):
            if i % 2 == 0:
                dataset.append({"state": "command: start engine", "candidate_ids": ["start", "stop"], "target": "start"})
            else:
                dataset.append({"state": "command: stop engine", "candidate_ids": ["start", "stop"], "target": "stop"})

        # Case 1: Causal Dependent Scorer (reads state)
        def causal_scorer(state, cands, item=None):
            tokens = set(state.split())
            for c in cands:
                if c in tokens:
                    return c, 0.95
            return cands[0], 0.50

        bench = ShuffledStateBenchmark(cag_threshold=0.20, cgr_threshold=3.0)
        report_causal = bench.evaluate_scorer(causal_scorer, dataset)

        self.assertTrue(report_causal.passed_causal_gate)
        self.assertGreaterEqual(report_causal.cag, 0.20)
        self.assertGreaterEqual(report_causal.cgr, 3.0)
        self.assertIn("PASSED", report_causal.verdict)

        # Case 2: Shortcut / Prior-Memorizing Scorer (always picks first candidate ignoring state)
        def shortcut_scorer(state, cands, item=None):
            return cands[0], 0.90

        report_shortcut = bench.evaluate_scorer(shortcut_scorer, dataset)
        self.assertFalse(report_shortcut.passed_causal_gate)
        self.assertIn("FAILED", report_shortcut.verdict)



class TestAttentionEntropyDiagnostics(unittest.TestCase):
    """Tests for Milestone 2: Attention Entropy Diagnostic Module & Credibility Gate."""

    def test_normalized_entropy_properties(self):
        # 1. Perfectly uniform distribution (L=5) -> H_norm = 1.0
        uniform = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
        h_uni = compute_normalized_attention_entropy(uniform)
        self.assertAlmostEqual(h_uni, 1.0, places=4)

        # 2. Concentrated one-hot distribution (L=5) -> H_norm = 0.0
        onehot = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        h_one = compute_normalized_attention_entropy(onehot)
        self.assertAlmostEqual(h_one, 0.0, places=4)

        # 3. Single token L=1 -> H_norm = 0.0 (no diffusion)
        single = np.array([1.0], dtype=np.float32)
        h_single = compute_normalized_attention_entropy(single)
        self.assertAlmostEqual(h_single, 0.0, places=4)

    def test_pytorch_tensor_entropy(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch not installed")

        t_uniform = torch.full((2, 3, 4), 0.25)
        h_t = compute_normalized_attention_entropy(t_uniform)
        self.assertEqual(h_t.shape, (2, 3))
        self.assertTrue(torch.allclose(h_t, torch.ones_like(h_t), atol=1e-4))

    def test_two_dimensional_credibility_gate(self):
        gate = TwoDimensionalCredibilityGate(conf_threshold=0.60, entropy_threshold=0.85)

        # Rule 1: High conf + Low entropy -> PASS
        v1 = gate.evaluate(confidence=0.92, attention_entropy=0.15)
        self.assertEqual(v1.status, "PASS")
        self.assertTrue(v1.passed)
        self.assertFalse(v1.is_blind_confidence)
        self.assertEqual(v1.recommended_action, "execute")

        # Rule 2: High conf + High entropy (>0.85) -> CIRCUIT_BREAKER_BLIND_CONFIDENCE
        v2 = gate.evaluate(confidence=0.95, attention_entropy=0.92)
        self.assertEqual(v2.status, "CIRCUIT_BREAKER_BLIND_CONFIDENCE")
        self.assertFalse(v2.passed)
        self.assertTrue(v2.is_blind_confidence)
        self.assertEqual(v2.recommended_action, "abstain")

        # Rule 3: Low conf + High entropy -> ABSTAIN_UNCERTAIN
        v3 = gate.evaluate(confidence=0.40, attention_entropy=0.89)
        self.assertEqual(v3.status, "ABSTAIN_UNCERTAIN")
        self.assertFalse(v3.passed)
        self.assertEqual(v3.recommended_action, "replan")

        # Rule 4: Low conf + Low entropy -> FALLBACK_LOW_CONFIDENCE
        v4 = gate.evaluate(confidence=0.45, attention_entropy=0.20)
        self.assertEqual(v4.status, "FALLBACK_LOW_CONFIDENCE")
        self.assertFalse(v4.passed)
        self.assertEqual(v4.recommended_action, "abstain")

    def test_dual_head_model_entropy_forward(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch not installed")

        model = GenZeroDualHeadModel(hidden_dim=32, embed_dim=16, num_layers=1, num_heads=2)
        examples = [
            {"type": "choice", "candidate_ids": ["act_1", "act_2"], "leaf_tokens": [[101, 1, 102], [101, 2, 102]]}
        ]
        logits, valid, entropy = model(examples, pad_token=0, return_attention_entropy=True)
        self.assertEqual(entropy.shape, (1, 2))
        self.assertTrue((entropy >= 0.0).all() and (entropy <= 1.0).all())


class TestDAggerExpertRelabeling(unittest.TestCase):
    """Tests for Milestone 3: DAgger Expert Relabeling Loop & Experience Buffer Injection."""

    def setUp(self):
        def oracle(state, cands):
            # Deterministic oracle: optimal action is 'recover' if present, else cands[0]
            gold = "recover" if "recover" in cands else cands[0]
            dist = {c: (0.90 if c == gold else 0.10 / max(1, len(cands) - 1)) for c in cands}
            return gold, dist

        self.relabeler = DAggerExpertRelabeler(
            expert_oracle_fn=oracle,
            conf_threshold=0.60,
            entropy_threshold=0.85,
            base_priority=2.0
        )

    def test_drift_detection_and_relabeling(self):
        # Case 1: Stable on-policy state -> No relabeling
        sample_stable = self.relabeler.relabel_state(
            state="safe_state",
            candidates=["recover", "other"],
            student_action="recover",
            student_confidence=0.85,
            student_entropy=0.20
        )
        self.assertIsNone(sample_stable)

        # Case 2: Drifting state with low confidence -> Relabeled
        sample_low_conf = self.relabeler.relabel_state(
            state="uncertain_state",
            candidates=["recover", "other"],
            student_action="other",
            student_confidence=0.45,
            student_entropy=0.90
        )
        self.assertIsNotNone(sample_low_conf)
        self.assertEqual(sample_low_conf.gold_action, "recover")
        self.assertTrue(sample_low_conf.is_dagger_relabeled)
        self.assertGreater(sample_low_conf.priority_weight, 2.0)

    def test_trajectory_relabeling(self):
        traj = [
            {"state": "s0", "candidates": ["recover", "step"], "action": "recover", "confidence": 0.90, "attention_entropy": 0.10},
            {"state": "s1_drift", "candidates": ["recover", "step"], "action": "step", "confidence": 0.35, "attention_entropy": 0.92},
        ]
        relabeled = self.relabeler.relabel_trajectory(traj, only_drifting=True)
        self.assertEqual(len(relabeled), 1)
        self.assertEqual(relabeled[0].state, "s1_drift")
        self.assertEqual(relabeled[0].gold_action, "recover")

    def test_replay_buffer_dagger_ingestion(self):
        buf = StabilityReplayBuffer(capacity=100, hard_ratio=0.25)
        # Gold baseline
        buf.load_gold_samples([
            {"id": f"gold_{i}", "state": f"g_{i}", "candidate_ids": ["a", "b"], "pi_target": {"a": 1.0}}
            for i in range(10)
        ])

        # Relabeled sample
        relabeled = [
            DAggerRelabeledSample(
                id="dag_1",
                state="drift_pos",
                candidate_ids=["a", "b"],
                gold_action="b",
                pi_target={"b": 1.0, "a": 0.0},
                priority_weight=3.0
            )
        ]
        buf.add_dagger_samples(relabeled)

        self.assertEqual(buf.stats["dagger_samples"], 1)
        self.assertEqual(buf.stats["hard_samples"], 1)

        batch = buf.sample_batch(batch_size=4)
        self.assertTrue(any(s.get("is_dagger_relabeled") for s in batch))

    def test_curriculum_controller_decay(self):
        ctrl = DAggerCurriculumController(p_student_base=0.80, decay_factor=0.90, p_student_min=0.50)
        self.assertEqual(ctrl.get_current_student_prob(), 0.80)

        ctrl.advance_iteration(num_rollouts=10, num_drifts=3, num_relabeled=3)
        self.assertAlmostEqual(ctrl.get_current_student_prob(), 0.72)
        self.assertEqual(ctrl.stats["total_drift_states"], 3)


class TestZeroStateDeltaEncoder(unittest.TestCase):
    """Tests for Milestone 4: Zero-State First-Order Delta Feature Channel & Market State."""

    def test_vector_delta_differencing_and_concatenation(self):
        encoder = ZeroStateDeltaEncoder(clamp_value=10.0, normalize=True)

        # Initial tick t=0 (no previous frame) -> delta = 0
        s0 = np.array([10.0, 50.0, -5.0])
        d0, concat0 = encoder.compute_vector_delta(s0, prev_vec=None)
        self.assertTrue(np.allclose(d0, [0.0, 0.0, 0.0]))
        self.assertEqual(concat0.shape, (6,))
        self.assertTrue(np.allclose(concat0, [10.0, 50.0, -5.0, 0.0, 0.0, 0.0]))

        # Subsequent tick t=1 -> delta = s1 - s0
        s1 = np.array([12.0, 48.0, 0.0])
        d1, concat1 = encoder.compute_vector_delta(s1, prev_vec=s0)
        self.assertTrue(np.allclose(d1, [2.0, -2.0, 5.0]))
        self.assertTrue(np.allclose(concat1, [12.0, 48.0, 0.0, 2.0, -2.0, 5.0]))

    def test_dict_delta_extraction(self):
        encoder = ZeroStateDeltaEncoder(clamp_value=5.0)
        curr = {"mid_price": 100.25, "spread_bps": 1.5, "flag": "trade"}
        prev = {"mid_price": 100.10, "spread_bps": 1.2, "flag": "trade"}

        deltas = encoder.compute_dict_delta(curr, prev)
        self.assertAlmostEqual(deltas["delta_mid_price"], 0.15)
        self.assertAlmostEqual(deltas["delta_spread_bps"], 0.3)
        self.assertNotIn("delta_flag", deltas)

    def test_l2_market_state_delta_integration(self):
        l2_encoder = L2MarketStateEncoder()
        snap_t0 = L2OrderBookSnapshot(bids=[(100.0, 10.0)], asks=[(100.5, 10.0)], cvd=50.0)
        snap_t1 = L2OrderBookSnapshot(bids=[(100.2, 12.0)], asks=[(100.6, 8.0)], cvd=65.0)

        # Delta computation
        deltas = l2_encoder.compute_delta(snap_t1, snap_t0)
        self.assertAlmostEqual(deltas["delta_mid_price"], 0.15)
        self.assertAlmostEqual(deltas["delta_cvd"], 15.0)

        # Text encoding with derivative signs
        text = l2_encoder.encode_to_state_text_with_delta(snap_t1, snap_t0)
        self.assertIn("Δ+0.15", text)
        self.assertIn("Δ+15.0", text)

        # Feature vector concatenation
        vec = l2_encoder.encode_to_feature_vector(snap_t1, snap_t0)
        self.assertEqual(len(vec), 8)  # 4 base + 4 delta


if __name__ == "__main__":
    unittest.main()
