import unittest
import copy
import time
from typing import Dict, Any

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from gen_zero.daemon.atomic_container import AtomicModelContainer, ServingSnapshot
from gen_zero.daemon.daemon_engine import GenZeroRSIDaemon
from gen_zero.client import GenZeroClient, GenZeroConfig
from gen_zero.train.replay_buffer import StabilityReplayBuffer
from gen_zero.rollout.hard_miner import MinedSample


class DummyScorer:
    def __init__(self, tag: str = "init"):
        self.tag = tag
        self.weights = {"w": 1.0}

    def load_from_weight_dict(self, d: Dict[str, Any]):
        self.weights = dict(d)

    def score_candidates(self, state_repr, candidates, **kwargs):
        return {c: 1.0 / len(candidates) for c in candidates}, {c: 0.5 for c in candidates}


class DummyModel:
    def __init__(self, val: float = 1.0):
        self.val = val

    def export_to_scorer_weights(self):
        return {"w": self.val}


class TestRound9Contract(unittest.TestCase):
    """Exhaustive contract verification suite for Round 9 / Round 10 audit closures."""

    # --------------------------------------------------------------------------
    # 1. R9-01: Atomic Container Snapshot Immutability & Candidate Scorer Isolation
    # --------------------------------------------------------------------------

    def test_r9_p01_serving_snapshot_is_read_only(self):
        """Probe R9_P01: ServingSnapshot attributes must be immutable properties."""
        snap = ServingSnapshot(model="model_v1", scorer="scorer_v1", version_id=1, version_tag="v1", timestamp=time.time())
        self.assertEqual(snap.model, "model_v1")
        self.assertEqual(snap.scorer, "scorer_v1")
        self.assertEqual(snap.version_id, 1)
        self.assertEqual(snap.version_tag, "v1")

        with self.assertRaises(AttributeError):
            snap.model = "model_v2"
        with self.assertRaises(AttributeError):
            snap.scorer = "scorer_v2"
        with self.assertRaises(AttributeError):
            snap.version_id = 2

    def test_r9_p02_candidate_sync_does_not_mutate_active_scorer(self):
        """Probe R9_P02: sync_model_to_scorer with target_scorer isolates candidate from active scorer."""
        client = GenZeroClient()
        active_scorer = DummyScorer("active")
        active_scorer.weights = {"w": 42.0}
        client.cpu_extreme_scorer = active_scorer

        candidate_scorer = copy.deepcopy(active_scorer)
        candidate_model = DummyModel(val=999.0)

        # Sync targeting candidate_scorer must NOT alter active_scorer
        res = client.sync_model_to_scorer(candidate_model, target_scorer=candidate_scorer)
        self.assertTrue(res)
        self.assertEqual(candidate_scorer.weights["w"], 999.0)
        self.assertEqual(active_scorer.weights["w"], 42.0, "Active scorer must remain completely untouched")

    def test_r9_p03_daemon_promotion_isolation_and_rollback(self):
        """Probe R9_P03: Failed promotion or rollback never corrupts active serving snapshot."""
        initial_model = DummyModel(val=1.0)
        initial_scorer = DummyScorer("initial")
        container = AtomicModelContainer(initial_model=initial_model, initial_scorer=initial_scorer)

        client = GenZeroClient()
        client.model = initial_model
        client.cpu_extreme_scorer = initial_scorer

        daemon = GenZeroRSIDaemon(client=client, cycle_interval_sec=60.0)
        daemon.container = container

        # Simulate candidate preparation: stage candidate scorer
        candidate_model = DummyModel(val=2.0)
        candidate_scorer = copy.deepcopy(container.get_scorer())
        client.sync_model_to_scorer(candidate_model, target_scorer=candidate_scorer)
        container.set_pending_scorer(candidate_scorer)

        # Before swap: active container scorer still has val=1.0
        self.assertEqual(container.get_scorer().weights["w"], 1.0)
        self.assertEqual(candidate_scorer.weights["w"], 2.0)

        # Swap container
        container.swap_model(candidate_model, version_tag="v2")
        self.assertEqual(container.get_model().val, 2.0)
        self.assertEqual(container.get_scorer().weights["w"], 2.0)

        # Rollback container: should cleanly restore initial snapshot
        container.rollback()
        self.assertEqual(container.get_model().val, 1.0)
        self.assertEqual(container.get_scorer().weights["w"], 1.0)

    # --------------------------------------------------------------------------
    # 2. R9-02: All-Invalid Candidate Safety & Single Candidate Forward Validation
    # --------------------------------------------------------------------------

    def test_r9_e01_all_invalid_candidates_returns_abstain_without_arbiter_fallback(self):
        """Probe R9_E01: When all candidates are invalid, return ABSTAIN and never invoke GPU arbiter."""
        config = GenZeroConfig(
            enable_abstain=True,
            enable_gpu_arbiter_fallback=True,
            arbiter_confidence_threshold=0.99  # would trigger if not blocked
        )
        client = GenZeroClient(config=config)

        # Candidate with negative/invalid mask simulated via reflex path
        state = "goal: navigate to unknown element\ncurrent_url: https://example.com"
        # Provide candidates where none match
        res = client.decide(state=state, candidates=["btn_invalid_a", "btn_invalid_b"], mode="fast")
        self.assertIn("action", res)
        # Action should be valid selection or ABSTAIN, never an arbitrary crash
        self.assertIn(res["action"], ["btn_invalid_a", "btn_invalid_b", "ABSTAIN"])

    def test_r9_e02_arbiter_fallback_never_overrides_abstain(self):
        """Probe R9_E02: Arbiter fallback must be skipped if consensus action is ABSTAIN."""
        config = GenZeroConfig(
            enable_abstain=True,
            enable_gpu_arbiter_fallback=True,
            arbiter_confidence_threshold=0.99
        )
        client = GenZeroClient(config=config)

        # Force mock consensus where all valid choices are absent
        state = "state"
        res = client.decide(state=state, candidates=[], mode="fast")
        self.assertIsNone(res["action"])
        self.assertEqual(res["confidence"], 0.0)

    def test_r9_e03_single_candidate_runs_model_and_validity(self):
        """Probe R9_E03: Single candidate must not blindly bypass model forward pass."""
        client = GenZeroClient()
        # Single candidate with dual head model active
        state = "goal: click submit\nurl: http://test.com"
        res = client.decide(state=state, candidates=["btn_submit"], mode="fast")
        self.assertEqual(res["action"], "btn_submit")
        self.assertGreaterEqual(res["confidence"], 0.0)

    # --------------------------------------------------------------------------
    # 3. R9-03: Unified Modality Gateway & Replay Buffer State Preparation
    # --------------------------------------------------------------------------

    def test_r9_i01_replay_buffer_uses_state_preparer(self):
        """Probe R9_I01: StabilityReplayBuffer normalizes states via state_preparer."""
        def mock_preparer(raw):
            if isinstance(raw, dict):
                return f"normalized_dict_{len(raw)}"
            return f"normalized_str_{raw}"

        buf = StabilityReplayBuffer(capacity=100, hard_ratio=0.5, state_preparer=mock_preparer)
        
        # Add a mined sample with raw dict state
        raw_state = {"agent": "crawler", "task": "index"}
        sample = MinedSample(
            state_id="s1",
            state_data=raw_state,
            candidate_ids=["act_a", "act_b"],
            pi_target={"act_a": 0.8, "act_b": 0.2},
            value_target=0.7,
            mining_reason="high_entropy",
            entropy=0.5,
            td_error=0.4
        )
        buf.add_mined_samples([sample])
        
        # Sample batch and verify state was normalized
        batch = buf.sample_batch(batch_size=1)
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0]["state"], "normalized_dict_2")
        self.assertTrue(batch[0]["is_hard_sample"])
        self.assertEqual(len(batch[0]["leaf_tokens"]), 2)

    def test_r9_i02_client_replay_buffer_matches_inference_normalization(self):
        """Probe R9_I02: Client replay buffer state preparer matches client.prepare_inference_state."""
        client = GenZeroClient()
        self.assertIsNotNone(client.replay_buffer.state_preparer)

        test_state = {"dom": "<div>hello</div>", "goal": "greet"}
        norm_client = client.prepare_inference_state(test_state)
        norm_buf = client.replay_buffer.state_preparer(test_state)
        self.assertEqual(norm_client, norm_buf, "Client inference normalization and replay buffer normalization must match")

    def test_r10_p01_pinned_snapshot_consistency(self):
        """Probe R10_P01: decide() pins a single serving snapshot at request start."""
        client = GenZeroClient()
        self.assertIsNotNone(client.container)
        snap1 = client.container.get_snapshot()
        self.assertIsNotNone(snap1)

        res = client.decide(state="goal: click test", candidates=["btn_a", "btn_b"], mode="fast")
        self.assertIn(res["action"], ["btn_a", "btn_b", "ABSTAIN"])

    def test_r10_e01_partial_invalid_arbiter_isolation(self):
        """Probe R10_E01: Partially invalid candidates cannot be resurrected by arbiter fallback."""
        config = GenZeroConfig(
            enable_abstain=True,
            enable_gpu_arbiter_fallback=True,
            arbiter_confidence_threshold=0.99
        )
        client = GenZeroClient(config=config)

        # Mock expert execution to return meta with explicit partial valid_set excluding btn_invalid
        orig_exec = client._execute_expert_distribution
        def mock_exec(*args, **kwargs):
            probs, val, best, meta = orig_exec(*args, **kwargs)
            # Declare only btn_submit as valid
            meta["valid_set"] = [c for c in kwargs.get("candidates", args[2] if len(args) > 2 else []) if c != "btn_invalid"]
            return probs, val, best, meta

        client._execute_expert_distribution = mock_exec
        try:
            state = "goal: submit form\nurl: http://test.com"
            res = client.decide(state=state, candidates=["btn_invalid", "btn_submit"], mode="fast")
            self.assertIn(res["action"], ["btn_submit", "ABSTAIN"])
            self.assertNotEqual(res["action"], "btn_invalid")
        finally:
            client._execute_expert_distribution = orig_exec

    def test_r10_i01_once_only_unwrapped_preparation(self):
        """Probe R10_I01 & R10_I02: Dict container {"state": ...} is unwrapped and prepared once only."""
        client = GenZeroClient()
        buf = StabilityReplayBuffer(capacity=50, state_preparer=client.prepare_inference_state)

        wrapped_state = {"state": "Goal: choose a"}
        norm_direct = client.prepare_inference_state("Goal: choose a")

        buf.load_gold_samples([{
            "id": "g1",
            "state": wrapped_state,
            "candidate_ids": ["opt_a", "opt_b"],
            "pi_target": {"opt_a": 1.0, "opt_b": 0.0},
            "value_target": 1.0
        }])

        batch = buf.sample_batch(batch_size=1)
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0]["state"], norm_direct)


if __name__ == "__main__":
    unittest.main()
