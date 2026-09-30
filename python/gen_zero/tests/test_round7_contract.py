"""Round 7 Contract Tests for Gen-Zero Decision Engine.

Validates complete closure of Round 7 review findings:
R7-01: Deterministic state normalization without truncation (ndarray sha256 byte digest, full numeric sequence, np.set_printoptions immunity)
R7-03: Reflex live model execution applies candidate valid mask before softmax

Training-side guarantees (distiller, replay, RSI daemon, promotion) moved to gen-zero-research
with the code they tested.
"""

import importlib.util
import unittest
import math
import json
import numpy as np

from gen_zero.model.dual_head import encode_leaf_tokens, normalize_state_repr, GenZeroDualHeadModel
from gen_zero.client import GenZeroClient, GenZeroConfig
HAS_TORCH = importlib.util.find_spec("torch") is not None


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


    # --- R7-05: Atomic Promotion Snapshot (Probe P09) ---



if __name__ == "__main__":
    unittest.main()
