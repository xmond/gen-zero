import unittest
import numpy as np
import time
import threading

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from gen_zero.model.dual_head import normalize_state_repr, _canonicalize_state_obj
from gen_zero.daemon.atomic_container import AtomicModelContainer, ServingSnapshot
from gen_zero.client import GenZeroClient, GenZeroConfig


class TestRound8Contract(unittest.TestCase):
    """Rigorous contract verification suite for Round 8 / Round 9 production audits."""

    def test_r8_n01_nested_ndarrays_and_tensors_in_dicts_and_lists(self):
        """Probes R8_N01-N04: Deep canonicalization of nested structures containing ndarrays and PyTorch tensors."""
        arr1 = np.ones((5, 5), dtype=np.float32)
        arr2 = np.zeros((5, 5), dtype=np.float32)
        
        # Two states with different nested arrays must NEVER collide
        state_a = {
            "metadata": {"step": 1, "agent_id": "browser_01"},
            "features": [arr1, "candidate_context"]
        }
        state_b = {
            "metadata": {"step": 1, "agent_id": "browser_01"},
            "features": [arr2, "candidate_context"]
        }
        
        repr_a = normalize_state_repr(state_a)
        repr_b = normalize_state_repr(state_b)
        
        self.assertNotEqual(repr_a, repr_b, "Nested ndarrays must generate distinct deterministic representations")
        self.assertIn("arr_float32_(5, 5)_", repr_a)
        self.assertIn("arr_float32_(5, 5)_", repr_b)

        if HAS_TORCH:
            t1 = torch.ones((3, 4), dtype=torch.float32)
            t2 = torch.zeros((3, 4), dtype=torch.float32)
            state_t1 = {"obs": t1, "info": "ok"}
            state_t2 = {"obs": t2, "info": "ok"}
            repr_t1 = normalize_state_repr(state_t1)
            repr_t2 = normalize_state_repr(state_t2)
            self.assertNotEqual(repr_t1, repr_t2, "PyTorch tensors in dicts must produce distinct SHA256 hashes")
            self.assertIn("arr_float32_(3, 4)_", repr_t1)

    def test_r8_n02_printoptions_immunity_no_ellipsis_truncation(self):
        """Probes R8_N03: Large arrays must not be affected by numpy printoptions ellipsis."""
        old_opts = np.get_printoptions()
        try:
            # Force severe numpy printing truncation
            np.set_printoptions(threshold=2, edgeitems=1)
            
            # Create two large arrays differing only in the truncated middle
            a1 = np.zeros((100, 100), dtype=np.float32)
            a2 = np.zeros((100, 100), dtype=np.float32)
            a2[50, 50] = 999.0
            
            # String representation would contain '...' and be identical
            self.assertEqual(str(a1), str(a2))
            
            # But normalize_state_repr must hash exact bytes and NEVER collide!
            repr_1 = normalize_state_repr({"grid": a1})
            repr_2 = normalize_state_repr({"grid": a2})
            self.assertNotEqual(repr_1, repr_2, "normalize_state_repr must be immune to numpy printoptions ellipsis")
        finally:
            np.set_printoptions(**old_opts)

    def test_r8_e01_all_invalid_candidates_returns_abstain(self):
        """Probe R8_E01: Reflex live scoring returns ABSTAIN with 0.0 confidence when all candidates are masked invalid."""
        if not HAS_TORCH:
            self.skipTest("PyTorch required for reflex live forward probe")

        client = GenZeroClient(GenZeroConfig())

        # Mock model forward returning all valid = False
        class MockAllInvalidModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scalar = torch.nn.Linear(1, 1)

            def forward(self, batch, pad_token=0, return_value=True):
                logits = torch.tensor([[10.0, 5.0, 2.0]], dtype=torch.float32)
                valid = torch.tensor([[False, False, False]], dtype=torch.bool)
                vals = torch.tensor([[0.5]], dtype=torch.float32)
                return logits, valid, vals

        client.model = MockAllInvalidModel()

        # In reflex mode or direct _execute_expert_distribution
        probs, val, best, meta = client._execute_expert_distribution(
            expert_name="reflex",
            state="Click submit",
            candidates=["btn_1", "btn_2", "btn_3"],
            trans_fn=lambda s, a: s
        )
        self.assertEqual(best, "ABSTAIN", "Must return ABSTAIN when all candidates are masked invalid")
        self.assertEqual(val, 0.0)
        self.assertTrue(all(p == 0.0 for p in probs.values()))
        self.assertEqual(meta.get("status"), "NO_VALID_CANDIDATES")


    def test_r8_p01_serving_snapshot_atomic_coupling(self):
        """Probes R8_P01, R8_P02: Atomic ServingSnapshot couples model and scorer without intermediate mismatch."""
        class MockModel:
            def __init__(self, tag):
                self.tag = tag

        class MockScorer:
            def __init__(self, tag):
                self.tag = tag

        m1 = MockModel("m1")
        s1 = MockScorer("s1")
        container = AtomicModelContainer(initial_model=m1, initial_scorer=s1)

        m2 = MockModel("m2")
        s2 = MockScorer("s2")

        # Concurrent read verification: readers should NEVER observe (m1, s2) or (m2, s1)
        mismatches = []
        stop_threads = threading.Event()

        def reader_loop():
            while not stop_threads.is_set():
                snap = container.get_snapshot()
                if (snap.model.tag == "m1" and snap.scorer.tag != "s1") or \
                   (snap.model.tag == "m2" and snap.scorer.tag != "s2"):
                    mismatches.append((snap.model.tag, snap.scorer.tag))

        threads = [threading.Thread(target=reader_loop) for _ in range(5)]
        for t in threads:
            t.start()

        time.sleep(0.01)
        container.swap_snapshot(m2, s2, version_tag="v2")
        time.sleep(0.02)
        container.rollback()
        time.sleep(0.01)
        
        stop_threads.set()
        for t in threads:
            t.join()

        self.assertEqual(len(mismatches), 0, f"Observed decoupled model/scorer state: {mismatches}")
        # Final active snapshot after rollback must be v1 (m1, s1)
        active = container.get_snapshot()
        self.assertEqual(active.model.tag, "m1")
        self.assertEqual(active.scorer.tag, "s1")



if __name__ == "__main__":
    unittest.main()
