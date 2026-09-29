"""Unit and Integration Tests for Canvas Slot Masking & H1 Entropy-Gated Sampling (Issue #32 & RFC-032).

Covers:
1. Canvas Slot Masking Protocol (Template assembly, slot typing, zero-decoding fixed position alignment).
2. H1 Entropy Evaluation and Fast-Path Single Pass (< 6.5ms).
3. H1 Multi-Read Adaptive Perturbation Sampling (mu ± sigma error bars & CP-SAT conservative bound).
4. Hybrid Attention Masking (Causal context prefix + Full bidirectional decision slots).
5. NanoCore BidirectionalSlotAttention Layer Forward Pass.
6. Multi-Task Coupled Joint Accuracy Improvement (>= 6% delta over causal masking).
7. Top-Level GenZero decide_canvas Integration.
"""

import unittest
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Sequence

try:
    import torch
except ImportError:
    torch = None

from gen_zero.client import GenZero
from gen_zero.protocol.canvas_protocol import (
    CanvasSlotType,
    CanvasSlotSpec,
    SlotResult,
    CanvasDecisionResult,
    CanvasTemplate,
)
from gen_zero.runtime.h1_entropy_gate import (
    H1EntropyGate,
    EntropyEvaluation,
    MultiReadStatistics,
)
from gen_zero.nanocore.bidirectional_slot_attention import (
    create_hybrid_slot_mask,
    BidirectionalSlotAttention,
    BidirectionalNanoCore,
    HAS_TORCH,
)


class TestIssue32CanvasAndH1Entropy(unittest.TestCase):

    def setUp(self):
        self.client = GenZero()
        self.gate = H1EntropyGate(tau_entropy=0.10, multi_read_k=4)

    def test_canvas_template_specification_and_formatting(self):
        """Milestone 1: Standardized Canvas Template formatting, slot typing, and token position alignment."""
        affordances = ["submit_button", "cancel_link", "terms_checkbox"]
        template = CanvasTemplate.standard_4tuple(affordances=affordances)

        # 1. Verify slots definition
        self.assertEqual(len(template.slots), 4)
        slot_names = template.get_slot_names()
        self.assertEqual(slot_names, ["action", "target", "done", "risk"])

        action_slot = template.get_slot("action")
        self.assertEqual(action_slot.slot_type, CanvasSlotType.CHOICE)
        self.assertEqual(action_slot.placeholder, "@{action_choice}")

        done_slot = template.get_slot("done")
        self.assertEqual(done_slot.slot_type, CanvasSlotType.NOUL)
        self.assertEqual(done_slot.placeholder, "@{done_noul}")

        risk_slot = template.get_slot("risk")
        self.assertEqual(risk_slot.slot_type, CanvasSlotType.SCORE)
        self.assertEqual(risk_slot.placeholder, "@{risk_score}")

        # 2. Verify formatted prompt
        prompt = template.format_prompt("Page: Checkout | Cart: 2 items | Total: $45.00")
        self.assertIn("<|decision_canvas|>", prompt)
        self.assertIn("action: @{action_choice}", prompt)
        self.assertIn("target: @{target_choice}", prompt)
        self.assertIn("done: @{done_noul}", prompt)
        self.assertIn("risk: @{risk_score}", prompt)
        self.assertIn("<|end_canvas|>", prompt)

        # 3. Verify parse_canvas_text
        parsed = CanvasTemplate.parse_canvas_text(prompt)
        self.assertEqual(len(parsed), 4)
        parsed_names = [p.name for p in parsed]
        self.assertEqual(parsed_names, ["action", "target", "done", "risk"])

        # 4. Zero-decoding fixed-token alignment
        dummy_tokens = ["Page:", "Checkout", "|", "<|decision_canvas|>", "action:", "@{action_choice}", "target:", "@{target_choice}"]
        positions = template.find_slot_token_positions(dummy_tokens)
        self.assertEqual(positions["action"], 5)
        self.assertEqual(positions["target"], 7)

    def test_h1_entropy_computation_and_fast_path(self):
        """Milestone 2: First-order Logits entropy calculation and Fast-Path early exit (<6.5ms)."""
        # Case A: Sharp, unambiguous distribution (Top prob = 0.98)
        sharp_probs = [0.98, 0.01, 0.01]
        h1_sharp = H1EntropyGate.compute_h1_entropy(sharp_probs)
        # - (0.98 * ln(0.98) + 0.01 * ln(0.01) + 0.01 * ln(0.01)) ≈ 0.11
        # Let's test a very sharp distribution: 0.99, 0.005, 0.005
        very_sharp = [0.99, 0.005, 0.005]
        h1_very_sharp = H1EntropyGate.compute_h1_entropy(very_sharp)
        self.assertLessEqual(h1_very_sharp, 0.10)

        ev = self.gate.evaluate_distribution({"opt_a": 0.99, "opt_b": 0.005, "opt_c": 0.005})
        self.assertFalse(ev.is_ambiguous)
        self.assertLessEqual(ev.entropy, 0.10)

        # Execute Fast-Path scheduler
        call_count = [0]
        def forward_mock(perturbation: float) -> Dict[str, Any]:
            call_count[0] += 1
            return {"slot1": {"opt_a": 0.99, "opt_b": 0.01}}

        def extract_mock(out: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
            return out

        t0 = time.perf_counter()
        out, is_multi, max_ent, stats = self.gate.schedule_execution(forward_mock, extract_mock)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Fast-Path must execute strictly 1 pass in < 6.5ms
        self.assertEqual(call_count[0], 1)
        self.assertFalse(is_multi)
        self.assertLess(elapsed_ms, 6.5)
        # A single read never measures variance: std_dev must be an honest
        # math.nan, not a fabricated 0.0 "zero spread" claim.
        self.assertTrue(math.isnan(stats["slot1"].std_dev))
        self.assertFalse(stats["slot1"].variance_measured)
        self.assertEqual(stats["slot1"].reads_count, 1)

    def test_h1_entropy_multi_read_path_with_error_bars(self):
        """Milestone 2: H1 > 0.10 triggers Multi-Read (K=4) with empirical mu ± sigma error bars,
        when the gate has genuinely opted into perturbation_enabled=True."""
        # Ambiguous distribution: 0.45 vs 0.40 vs 0.15
        ambiguous_probs = {"opt_a": 0.45, "opt_b": 0.40, "opt_c": 0.15}
        ev = self.gate.evaluate_distribution(ambiguous_probs)
        self.assertTrue(ev.is_ambiguous)
        self.assertGreater(ev.entropy, 0.10)

        perturbing_gate = H1EntropyGate(tau_entropy=0.10, multi_read_k=4, perturbation_enabled=True)

        call_count = [0]
        def forward_mock(pert: float) -> Dict[str, Any]:
            call_count[0] += 1
            # Simulate slight variance across perturbations
            base_a = 0.45 + pert * 0.2
            base_b = 0.40 - pert * 0.1
            base_c = max(0.01, 1.0 - (base_a + base_b))
            return {"action": {"opt_a": base_a, "opt_b": base_b, "opt_c": base_c}}

        def extract_mock(out: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
            return out

        out, is_multi, max_ent, stats = perturbing_gate.schedule_execution(forward_mock, extract_mock)

        # Multi-Read must fire exactly K=4 passes
        self.assertEqual(call_count[0], 4)
        self.assertTrue(is_multi)
        self.assertIn("action", stats)

        act_stat = stats["action"]
        self.assertEqual(act_stat.reads_count, 4)
        self.assertGreater(act_stat.std_dev, 0.0)
        self.assertIn("±", act_stat.error_bar)
        self.assertGreater(act_stat.two_sigma_margin, 0.0)

    def test_h1_entropy_rejects_overflow_int_instead_of_crashing(self):
        """Reviewer fix: float(10**1000) raises OverflowError, distinct from the
        TypeError/ValueError already handled. It must be treated as illegal
        input (entropy=+inf, is_valid=False, is_ambiguous=True), never let the
        OverflowError propagate and crash the caller."""
        huge_int = 10 ** 1000

        h1 = H1EntropyGate.compute_h1_entropy([huge_int, 0.5, 0.3])
        self.assertTrue(math.isinf(h1))

        ev = self.gate.evaluate_distribution([huge_int, 0.5, 0.3])
        self.assertEqual(ev.entropy, math.inf)
        self.assertFalse(ev.is_valid)
        self.assertTrue(ev.is_ambiguous)

        ev_dict = self.gate.evaluate_distribution({"opt_a": huge_int, "opt_b": 0.1})
        self.assertFalse(ev_dict.is_valid)
        self.assertTrue(ev_dict.is_ambiguous)

    def test_constructor_rejects_invalid_tau_entropy(self):
        """Reviewer fix: tau_entropy must be validated at construction time.
        A NaN, negative, or overflow-int threshold would silently disable (or
        always trip) the ambiguity gate instead of failing loudly at setup."""
        for bad_tau in (math.nan, math.inf, -math.inf, -1.0, -0.001, 10 ** 1000):
            with self.assertRaises(ValueError):
                H1EntropyGate(tau_entropy=bad_tau)

        # Boundary: exactly 0.0 is a legal (maximally strict) threshold.
        gate = H1EntropyGate(tau_entropy=0.0)
        self.assertEqual(gate.tau_entropy, 0.0)

        for bad_scale in (math.nan, math.inf, -math.inf, -1.0, -0.001, 10 ** 1000):
            with self.assertRaises(ValueError):
                H1EntropyGate(perturbation_scale=bad_scale)

        gate_zero_scale = H1EntropyGate(perturbation_scale=0.0)
        self.assertEqual(gate_zero_scale.perturbation_scale, 0.0)

        with self.assertRaises(ValueError):
            self.gate.compute_multi_read_statistics([float("nan"), 0.5])
        with self.assertRaises(ValueError):
            self.gate.compute_multi_read_statistics([0.5, float("inf")])

    def test_multi_read_taints_slot_whose_first_read_was_already_illegal(self):
        """Reviewer fix: if the stage-1 (unperturbed) read for a slot is itself
        illegal/non-finite, that slot must stay tainted through stage 2 even
        if every perturbed re-read comes back legal -- the fabricated 0.0
        placeholder for the illegal first read must never seed a stat that
        claims variance_measured=True."""
        perturbing_gate = H1EntropyGate(tau_entropy=0.10, multi_read_k=4, perturbation_enabled=True)

        call_count = [0]
        def forward_mock(pert: float) -> Dict[str, Any]:
            call_count[0] += 1
            if call_count[0] == 1:
                # Stage-1 read is corrupted from the start.
                return {"action": {"opt_a": float("nan"), "opt_b": 0.4, "opt_c": 0.2}}
            return {"action": {"opt_a": 0.45, "opt_b": 0.40, "opt_c": 0.15}}

        def extract_mock(out: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
            return out

        out, is_multi, max_ent, stats = perturbing_gate.schedule_execution(forward_mock, extract_mock)

        self.assertTrue(is_multi)
        act_stat = stats["action"]
        self.assertFalse(act_stat.variance_measured)
        self.assertTrue(math.isnan(act_stat.std_dev))

    def test_multi_read_tainted_perturbation_never_reports_measured_variance(self):
        """Reviewer fix: if a perturbation pass in the multi-read stage returns
        an illegal/non-finite distribution, that slot must be reported as
        variance_measured=False, never wrap a NaN sample in a stat that claims
        real variance was measured."""
        ambiguous_probs = {"opt_a": 0.45, "opt_b": 0.40, "opt_c": 0.15}
        ev = self.gate.evaluate_distribution(ambiguous_probs)
        self.assertTrue(ev.is_ambiguous)

        perturbing_gate = H1EntropyGate(tau_entropy=0.10, multi_read_k=4, perturbation_enabled=True)

        call_count = [0]
        def forward_mock(pert: float) -> Dict[str, Any]:
            call_count[0] += 1
            if call_count[0] == 3:
                # Simulate a corrupted perturbation read: NaN probability.
                return {"action": {"opt_a": float("nan"), "opt_b": 0.4, "opt_c": 0.2}}
            return {"action": {"opt_a": 0.45, "opt_b": 0.40, "opt_c": 0.15}}

        def extract_mock(out: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
            return out

        out, is_multi, max_ent, stats = perturbing_gate.schedule_execution(forward_mock, extract_mock)

        self.assertTrue(is_multi)
        act_stat = stats["action"]
        self.assertFalse(act_stat.variance_measured)
        self.assertTrue(math.isnan(act_stat.std_dev))
        for s in act_stat.samples:
            self.assertFalse(math.isnan(s))

    def test_multi_read_taints_slot_missing_in_later_passes(self):
        """Reviewer fix: if a slot present in the initial read is missing in a
        subsequent perturbation pass, that slot must be marked tainted rather than
        silently reporting variance_measured=True on incomplete data."""
        ambiguous_probs = {"opt_a": 0.45, "opt_b": 0.40, "opt_c": 0.15}
        ev = self.gate.evaluate_distribution(ambiguous_probs)
        self.assertTrue(ev.is_ambiguous)

        perturbing_gate = H1EntropyGate(tau_entropy=0.10, multi_read_k=3, perturbation_enabled=True)
        call_count = [0]
        def forward_mock(pert: float) -> Dict[str, Any]:
            call_count[0] += 1
            if call_count[0] == 1:
                return {"action": {"opt_a": 0.45, "opt_b": 0.40, "opt_c": 0.15}}
            return {}  # slot missing in later pass

        def extract_mock(out: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
            return out

        out, is_multi, max_ent, stats = perturbing_gate.schedule_execution(forward_mock, extract_mock)
        self.assertTrue(is_multi)
        act_stat = stats["action"]
        self.assertFalse(act_stat.variance_measured)
        self.assertTrue(math.isnan(act_stat.std_dev))

    def test_hybrid_attention_mask_generation(self):
        """Milestone 3: Causal context prefix + Full bidirectional decision slots."""
        context_len = 6
        num_slots = 4
        mask = create_hybrid_slot_mask(context_len=context_len, num_slots=num_slots, is_causal_context=True)

        if HAS_TORCH and hasattr(mask, "shape"):
            self.assertEqual(mask.shape, (10, 10))

            # 1. Context rows (0..5): lower triangular in context columns, 0 in slot columns
            for i in range(context_len):
                for j in range(context_len):
                    if j <= i:
                        self.assertEqual(mask[i, j].item(), 1.0)
                    else:
                        self.assertEqual(mask[i, j].item(), 0.0)
                for j in range(context_len, 10):
                    self.assertEqual(mask[i, j].item(), 0.0)

            # 2. Slot rows (6..9): 1 across ALL context columns and 1 across ALL slot columns!
            for i in range(context_len, 10):
                for j in range(10):
                    self.assertEqual(mask[i, j].item(), 1.0)
        else:
            # Check NumPy / list array
            self.assertEqual(len(mask), 10)
            self.assertEqual(len(mask[0]), 10)
            self.assertEqual(mask[7][8], 1.0)  # Slot 1 attends to Slot 2
            self.assertEqual(mask[8][7], 1.0)  # Slot 2 attends to Slot 1 (Bidirectional!)
            self.assertEqual(mask[2][7], 0.0)  # Context cannot attend to future slot

    def test_nanocore_bidirectional_slot_attention_forward(self):
        """Milestone 3: PyTorch BidirectionalSlotAttention layer forward pass."""
        if not HAS_TORCH:
            self.skipTest("PyTorch not installed, skipping nn.Module test")

        attn = BidirectionalSlotAttention(embed_dim=128, num_heads=4)
        ctx = torch.randn(2, 8, 128)
        slots = torch.randn(2, 4, 128)

        # Hybrid forward
        ctx_out, slots_out = attn(ctx, slots, mask_mode="hybrid")
        self.assertEqual(ctx_out.shape, (2, 8, 128))
        self.assertEqual(slots_out.shape, (2, 4, 128))

        # Causal forward
        ctx_out_c, slots_out_c = attn(ctx, slots, mask_mode="causal")
        # Representations between hybrid and causal must differ due to bidirectional attention
        diff = torch.norm(slots_out - slots_out_c).item()
        self.assertGreater(diff, 1e-4)

    def test_joint_accuracy_improvement_resonance(self):
        """Milestone 3: Multi-task joint accuracy improvement >= 6% under bidirectional slot resonance."""
        if not HAS_TORCH:
            self.skipTest("PyTorch not installed, benchmark fails closed without it")
        results = BidirectionalNanoCore.evaluate_joint_accuracy_gain(num_samples=200, coupling_strength=0.75)
        self.assertIn("causal_joint_acc", results)
        self.assertIn("hybrid_joint_acc", results)
        self.assertIn("joint_acc_delta", results)

        # Joint accuracy delta must be at least 6.0% (0.06)
        self.assertGreaterEqual(results["joint_acc_delta"], 0.06)
        self.assertGreater(results["hybrid_joint_acc"], results["causal_joint_acc"])

    def test_client_decide_canvas_integration(self):
        """End-to-End: GenZero.decide_canvas multi-slot execution with adaptive sampling."""
        res = self.client.decide_canvas(
            state="User instruction: Click the Confirm Order button to finalize purchase.",
            affordances=["confirm_order_btn", "cancel_order_btn", "view_cart_link"],
            action_types=["click", "hover", "navigate"],
            enable_adaptive_sampling=True
        )

        self.assertIsInstance(res, CanvasDecisionResult)
        self.assertIn("action", res.slots)
        self.assertIn("target", res.slots)
        self.assertIn("done", res.slots)
        self.assertIn("risk", res.slots)

        act_slot = res.get_slot("action")
        self.assertIsNotNone(act_slot.chosen_value)
        self.assertGreater(act_slot.confidence, 0.0)

        tgt_slot = res.get_slot("target")
        self.assertIsNotNone(tgt_slot.chosen_value)

        # Verify to_dict structure
        res_dict = res.to_dict()
        self.assertIn("slots", res_dict)
        self.assertIn("timing_ms", res_dict)
        self.assertIn("is_multi_read", res_dict)


if __name__ == "__main__":
    unittest.main()
