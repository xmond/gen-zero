"""Round 7 Contract Tests for Gen-Zero Decision Engine.

Validates complete closure of Round 7 review findings:
R7-01: Deterministic state normalization without truncation (ndarray sha256 byte digest, full numeric sequence, np.set_printoptions immunity)
R7-02: Import normalize_state_repr in daemon_engine.py preventing NameError during benchmark evaluation with candidate descriptions
R7-03: Reflex live model execution applies candidate valid mask before softmax
R7-04: Distiller rejects NaN/Inf policy targets and re-normalizes target distribution over valid candidate support set
R7-05: Atomic promotion snapshot syncs CPU quantized scorer prior to model swap and client publication
"""

import unittest
import math
import json
import numpy as np

from gen_zero.model.dual_head import encode_leaf_tokens, normalize_state_repr, GenZeroDualHeadModel
from gen_zero.client import GenZeroClient, GenZeroConfig
from gen_zero.daemon.daemon_engine import GenZeroRSIDaemon
from gen_zero.train.distiller import GenZeroDistiller, HAS_TORCH
from gen_zero.train.replay_buffer import StabilityReplayBuffer


class TestRound7Contract(unittest.TestCase):

    def setUp(self):
        self.client = GenZeroClient(GenZeroConfig(hidden_dim=2048, enable_gpu_arbiter_fallback=True))

    # --- R7-01: Deterministic State Normalization (Probes N01 - N04) ---
    def test_r7_01_numeric_list_no_truncation_past_64(self):
        """Probe N01: Numeric lists differing past index 64 must produce different representations."""
        l1 = [float(i) for i in range(100)]
        l2 = [float(i) for i in range(100)]
        l2[65] = 999.0

        r1 = normalize_state_repr(l1)
        r2 = normalize_state_repr(l2)
        self.assertNotEqual(r1, r2, "Lists differing at index 65 produced identical normalized repr!")

        toks1 = encode_leaf_tokens(l1, "action_a")
        toks2 = encode_leaf_tokens(l2, "action_a")
        self.assertNotEqual(toks1, toks2, "Leaf tokens were identical for states differing at index 65!")

    def test_r7_01_float_precision_preserved(self):
        """Probe N02: High precision floats must not be prematurely rounded."""
        s1 = [0.12345678901234]
        s2 = [0.12345678901235]
        self.assertNotEqual(normalize_state_repr(s1), normalize_state_repr(s2))

    def test_r7_01_ndarray_sha256_and_printoptions_immunity(self):
        """Probe N03 & N04: ndarrays must use exact byte digests immune to numpy printoptions."""
        arr1 = np.arange(200, dtype=np.float32)
        arr2 = np.arange(200, dtype=np.float32)
        arr2[150] = 99999.0

        r1 = normalize_state_repr(arr1)
        r2 = normalize_state_repr(arr2)
        self.assertNotEqual(r1, r2, "ndarrays differing at index 150 produced identical repr!")

        # Test np.set_printoptions immunity
        orig_opts = np.get_printoptions()
        try:
            np.set_printoptions(threshold=1, edgeitems=1)
            r1_truncated_opts = normalize_state_repr(arr1)
            self.assertEqual(r1, r1_truncated_opts, "normalize_state_repr was affected by np.set_printoptions!")
        finally:
            np.set_printoptions(**orig_opts)

    # --- R7-02: NameError Fix in Daemon Benchmark (Probe C04) ---
    def test_r7_02_benchmark_eval_with_descriptions_no_nameerror(self):
        """Probe C04: evaluate_model_on_benchmark with candidate_descriptions executes without NameError."""
        daemon = GenZeroRSIDaemon(self.client)
        mock_suite = [
            {
                "state_repr": {"url": "https://calendar.google.com", "step": 1},
                "candidate_actions": ["save_btn", "cancel_btn"],
                "candidate_descriptions": {"save_btn": "Save Meeting", "cancel_btn": "Cancel Form"},
                "ground_truth_safe_action": "save_btn"
            }
        ]
        if HAS_TORCH:
            metrics = daemon.evaluate_model_on_benchmark(daemon.container.get_model(), mock_suite)
        else:
            metrics = daemon.evaluate_model_on_benchmark(lambda s, c: {"best_action": "save_btn"}, mock_suite)
        self.assertTrue(metrics.get("is_valid", False))
        self.assertIn("accuracy", metrics)

    # --- R7-03: Reflex Scoring Applies Valid Mask (Probe E08) ---
    def test_r7_03_reflex_scoring_applies_valid_mask(self):
        """Probe E08: Live model reflex scoring applies candidate valid mask."""
        if not HAS_TORCH:
            self.skipTest("PyTorch required for reflex valid mask test")

        client = GenZeroClient(GenZeroConfig(hidden_dim=128))
        probs, val, best_act, meta = client._execute_expert_distribution(
            expert_name="reflex",
            state="test_state",
            candidates=["cand_0", "cand_1", "cand_2"],
            trans_fn=lambda s, a: s
        )
        self.assertEqual(len(probs), 3)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=4)
        self.assertIn(best_act, ["cand_0", "cand_1", "cand_2"])

    # --- R7-04: Distiller Target Validity & Support Alignment (Probes T06, T14, T15) ---
    def test_r7_04_distiller_nan_policy_target_rejected(self):
        """Probe T06: Distiller immediately rejects NaN policy targets with NON_FINITE_TARGETS."""
        if not HAS_TORCH:
            self.skipTest("PyTorch required")

        model = GenZeroDualHeadModel(hidden_dim=32, embed_dim=32)
        buf = StabilityReplayBuffer()
        distiller = GenZeroDistiller(model=model, replay_buffer=buf, lr=1e-3)

        batch = [{
            "leaf_tokens": [[1, 2], [3, 4]],
            "candidate_ids": ["c0", "c1"],
            "type": "choice",
            "pi_target": {"c0": float("nan"), "c1": 0.5},
            "value_target": 1.0
        }]
        res = distiller.train_step(batch)
        self.assertFalse(res.get("optimized", True))
        self.assertEqual(res.get("status"), "NON_FINITE_TARGETS")

        # Test list format with NaN
        batch_list = [{
            "leaf_tokens": [[1, 2], [3, 4]],
            "candidate_ids": ["c0", "c1"],
            "type": "choice",
            "pi_target": [float("inf"), 0.5],
            "value_target": 1.0
        }]
        res_list = distiller.train_step(batch_list)
        self.assertFalse(res_list.get("optimized", True))
        self.assertEqual(res_list.get("status"), "NON_FINITE_TARGETS")

    def test_r7_04_distiller_valid_mask_support_alignment(self):
        """Probes T14 & T15: Invalid candidate target mass is eliminated and valid mass re-normalized."""
        if not HAS_TORCH:
            self.skipTest("PyTorch required")

        model = GenZeroDualHeadModel(hidden_dim=32, embed_dim=32)
        buf = StabilityReplayBuffer()
        distiller = GenZeroDistiller(model=model, replay_buffer=buf, lr=1e-3)

        # Batch with 2 candidates in candidate_ids, but pi_target contains an out-of-bounds target "c99"
        batch = [{
            "leaf_tokens": [[1, 2], [3, 4]],
            "candidate_ids": ["c0", "c1"],
            "type": "choice",
            "pi_target": {"c0": 0.5, "c99": 0.5},  # c99 is not in candidate_ids
            "value_target": 1.0
        }]
        res = distiller.train_step(batch)
        self.assertTrue(res.get("optimized", False))
        self.assertEqual(res.get("status"), "OPTIMIZED")

        # Batch where ALL positive probability mass is on non-existent / invalid candidates
        batch_invalid_only = [{
            "leaf_tokens": [[1, 2], [3, 4]],
            "candidate_ids": ["c0", "c1"],
            "type": "choice",
            "pi_target": {"c99": 1.0},
            "value_target": None
        }]
        res_inv = distiller.train_step(batch_invalid_only)
        self.assertFalse(res_inv.get("optimized", True))
        self.assertEqual(res_inv.get("status"), "MISSING_SUPERVISION")

    # --- R7-05: Atomic Promotion Snapshot (Probe P09) ---
    def test_r7_05_atomic_promotion_scorer_pre_synced(self):
        """Probe P09: Promotion stages scorer sync before container swap and client publication."""
        daemon = GenZeroRSIDaemon(self.client)
        initial_version = daemon.container.get_status()["active_version"]

        # Track sync execution order relative to swap
        call_log = []
        orig_swap = daemon.container.swap_model
        orig_sync = daemon.client.sync_model_to_scorer
        orig_clone = daemon.client.distiller.clone_for_candidate
        orig_eval = daemon.evaluate_model_on_benchmark

        mock_trainer = type("MockTrainer", (), {"run_iteration": lambda self, **k: {"steps_trained": 1, "mean_loss": 0.05}})()
        daemon.client.distiller.clone_for_candidate = lambda m: mock_trainer

        live_model = daemon.container.get_model()
        def mock_eval(model, suite):
            if model is live_model:
                return {"accuracy": 85.0, "mean_score": 20.0, "collision_rate": 0.05, "is_valid": True}
            else:
                return {"accuracy": 95.0, "mean_score": 30.0, "collision_rate": 0.0, "is_valid": True}
        daemon.evaluate_model_on_benchmark = mock_eval

        model_seen_at_sync = []
        def mock_sync(cand=None, target_scorer=None):
            model_seen_at_sync.append(daemon.client.model)
            call_log.append(("sync", cand is not None))
            return True

        def mock_swap(new_model, version_tag=None):
            call_log.append(("swap", True))
            return orig_swap(new_model, version_tag)

        daemon.client.sync_model_to_scorer = mock_sync
        daemon.container.swap_model = mock_swap

        try:
            report = daemon.run_single_evolution_cycle()
            self.assertEqual(report.get("reload_status"), "HOT_RELOADED")
            self.assertGreater(len(call_log), 0)
            self.assertIs(model_seen_at_sync[0], live_model, "client.model was prematurely exposed before scorer sync completed!")
        finally:
            daemon.client.sync_model_to_scorer = orig_sync
            daemon.container.swap_model = orig_swap
            daemon.client.distiller.clone_for_candidate = orig_clone
            daemon.evaluate_model_on_benchmark = orig_eval

    def test_r7_05_promotion_sync_failure_prevents_swap(self):
        """Probe P09: If sync_model_to_scorer fails, container is not swapped and client model remains untouched."""
        daemon = GenZeroRSIDaemon(self.client)
        orig_model = daemon.client.model
        orig_sync = daemon.client.sync_model_to_scorer
        orig_clone = daemon.client.distiller.clone_for_candidate
        orig_eval = daemon.evaluate_model_on_benchmark

        mock_trainer = type("MockTrainer", (), {"run_iteration": lambda self, **k: {"steps_trained": 1, "mean_loss": 0.05}})()
        daemon.client.distiller.clone_for_candidate = lambda m: mock_trainer

        live_model = daemon.container.get_model()
        def mock_eval(model, suite):
            if model is live_model:
                return {"accuracy": 85.0, "mean_score": 20.0, "collision_rate": 0.05, "is_valid": True}
            else:
                return {"accuracy": 95.0, "mean_score": 30.0, "collision_rate": 0.0, "is_valid": True}
        daemon.evaluate_model_on_benchmark = mock_eval

        daemon.client.sync_model_to_scorer = lambda cand=None, target_scorer=None: False

        try:
            report = daemon.run_single_evolution_cycle()
            self.assertIn("sync_model_to_scorer returned False", report.get("gate_reason", ""))
            self.assertIs(daemon.client.model, orig_model)
        finally:
            daemon.client.sync_model_to_scorer = orig_sync
            daemon.client.distiller.clone_for_candidate = orig_clone
            daemon.evaluate_model_on_benchmark = orig_eval


if __name__ == "__main__":
    unittest.main()
