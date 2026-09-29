"""Unit tests for Gen-Zero Cloud-Edge GPU Arbiter Bridge."""

import json
import unittest
from unittest.mock import MagicMock, patch

from gen_zero.gateway.arbiter_bridge import (
    CloudGPUArbiterBridge,
    ArbiterVerdict,
)
from gen_zero.rollout.hard_miner import HardSampleMiner
from gen_zero.vision.engine import PyTorchVisualDecisionEngine
from gen_zero.client import GenZero, GenZeroConfig
from gen_zero.vision.engine import PyTorchVisualDecisionEngine


class TestCloudGPUArbiterBridge(unittest.TestCase):
    def setUp(self):
        self.miner = HardSampleMiner(entropy_threshold=0.5, td_error_threshold=0.5)
        self.bridge = CloudGPUArbiterBridge(
            confidence_threshold=0.40,
            entropy_threshold=0.75,
            miner=self.miner
        )

    def test_fallback_trigger_logic(self):
        # 1. High confidence -> No fallback
        self.assertFalse(self.bridge.should_trigger_fallback(confidence=0.85, entropy=0.2))

        # 2. Low confidence (< 0.40) -> Must trigger fallback
        self.assertTrue(self.bridge.should_trigger_fallback(confidence=0.25, entropy=0.5))

        # 3. High entropy (> 0.75) -> Must trigger fallback
        self.assertTrue(self.bridge.should_trigger_fallback(confidence=0.50, entropy=0.82))

    def test_async_arbitration_and_hard_mining(self):
        # Test non-blocking async dispatch
        candidates = ["LEFT", "RIGHT", "ABSTAIN"]
        state = {"scenario": "unseen_ood_trap_state", "danger": 0.9}
        
        verdict = self.bridge.arbitrate(
            state=state,
            candidates=candidates,
            local_action="LEFT",
            local_confidence=0.20,
            local_probs={"LEFT": 0.20, "RIGHT": 0.20, "ABSTAIN": 0.60},
            mode="async"
        )
        self.assertTrue(verdict.is_fallback)
        self.assertEqual(verdict.action, "ABSTAIN")
        self.assertIn("async_dispatched_to_gpu", verdict.metadata)

        # Ensure transition was captured into mined history
        self.assertGreater(len(self.bridge.mined_history), 0)

    def test_sync_arbitration_fails_closed_without_any_backend(self):
        # setUp's bridge has no local_gpu_engine and no remote_endpoint: sync mode
        # must fail closed, never fabricate a confident verdict.
        candidates = ["BUY", "HOLD", "SELL"]
        verdict = self.bridge.arbitrate(
            state={"anomaly": "flash_crash"},
            candidates=candidates,
            local_action="BUY",
            local_confidence=0.30,
            local_probs={"BUY": 0.3, "HOLD": 0.35, "SELL": 0.35},
            mode="sync"
        )
        self.assertTrue(verdict.is_fallback)
        self.assertFalse(verdict.backend_reachable)
        self.assertEqual(verdict.confidence, 0.0)
        self.assertIsNone(verdict.action)
        self.assertEqual(verdict.arbitration_source, "fail_closed_unreachable")
        self.assertEqual(verdict.metadata["status"], "failed")

    def test_client_end_to_end_arbiter_fallback(self):
        try:
            import torch
            torch.manual_seed(42)
        except ImportError:
            pass

        # Configure client with strict arbiter threshold
        cfg = GenZeroConfig(
            enable_gpu_arbiter_fallback=True,
            arbiter_confidence_threshold=0.50
        )
        client = GenZero(cfg)

        # Unconfident 4-way choice where max prob is 0.25 (flat uniform)
        unconfident_state = {"type": "obscure_maze_room_never_seen"}
        res = client.decide(
            state=unconfident_state,
            candidates=["NORTH", "SOUTH", "EAST", "WEST", "ABSTAIN"],
            mode="reflex"  # reflex splits evenly -> confidence 0.20 < 0.50
        )
        
        # Verify fallback was triggered
        self.assertIn("arbiter_fallback", res)
        self.assertIsNotNone(res["arbiter_fallback"])
        self.assertTrue(res["arbiter_fallback"]["is_fallback"])
        self.assertEqual(res["action"], "ABSTAIN")

    def test_async_dispatch_does_not_inflate_count_when_vision_engine_unloaded(self):
        # Reviewer-2 repro: a freshly constructed PyTorchVisualDecisionEngine never had
        # load_model_if_needed() called, so is_loaded is False and score_candidates_direct
        # returns a deterministic mock projection. That must never be counted as a real
        # arbitration.
        engine = PyTorchVisualDecisionEngine(device="cpu", feature_dim=64)
        self.assertFalse(engine.is_loaded)
        bridge = CloudGPUArbiterBridge(local_gpu_engine=engine)
        bridge._async_dispatch_and_mine("image", ["alpha", "beta"], "alpha")
        self.assertEqual(
            bridge._arbitration_count, 0,
            f"Arbitration count must be 0 when vision engine is not loaded, "
            f"got {bridge._arbitration_count}"
        )
        self.assertEqual(bridge._unreachable_count, 1)


class TestArbiterRejectsMockEngine(unittest.TestCase):
    """Guards against the mock-vision-engine-as-GPU-arbiter bypass.

    An unloaded ``PyTorchVisualDecisionEngine`` (no real weights) must never
    be invoked for scoring inside the arbiter bridge -- its char-hash mock
    output must not masquerade as a GPU arbitration verdict, and it must not
    permanently shadow the remote HTTP arbiter channel.
    """

    def _candidates_args(self):
        return dict(
            state="state_payload",
            candidates=["A", "B"],
            local_action="A",
            local_confidence=0.1,
            local_probs={"A": 0.5, "B": 0.5},
        )

    def test_unloaded_real_engine_class_is_never_scored(self):
        engine = PyTorchVisualDecisionEngine(model_name_or_path="not-actually-loaded")
        self.assertFalse(engine.is_real)
        bridge = CloudGPUArbiterBridge(local_gpu_engine=engine, remote_endpoint=None)

        with patch.object(engine, "score_candidates_direct") as mock_score, \
             patch.object(engine, "prefill_visual_context") as mock_prefill:
            verdict = bridge.arbitrate(**self._candidates_args(), mode="sync")

        mock_score.assert_not_called()
        mock_prefill.assert_not_called()
        self.assertFalse(verdict.backend_reachable)
        self.assertEqual(verdict.confidence, 0.0)
        self.assertIsNone(verdict.action)
        self.assertEqual(verdict.arbitration_source, "fail_closed_unreachable")

    def test_magicmock_engine_is_not_trusted_by_truthiness(self):
        # MagicMock().is_real is itself a truthy MagicMock; the guard must
        # require strict identity with True, not mere truthiness.
        engine = MagicMock()
        bridge = CloudGPUArbiterBridge(local_gpu_engine=engine, remote_endpoint=None)

        verdict = bridge.arbitrate(**self._candidates_args(), mode="sync")

        engine.prefill_visual_context.assert_not_called()
        engine.score_candidates_direct.assert_not_called()
        self.assertFalse(verdict.backend_reachable)
        self.assertEqual(verdict.confidence, 0.0)
        self.assertEqual(verdict.arbitration_source, "fail_closed_unreachable")

    def test_real_engine_is_trusted_and_scored(self):
        engine = MagicMock()
        engine.is_real = True
        engine.prefill_visual_context.return_value = {"past_key_values": "kv"}
        engine.score_candidates_direct.return_value = {
            "probs": {"A": 0.1, "B": 0.9},
            "best_action": "B",
        }
        bridge = CloudGPUArbiterBridge(local_gpu_engine=engine, remote_endpoint=None)

        verdict = bridge.arbitrate(**self._candidates_args(), mode="sync")

        engine.score_candidates_direct.assert_called_once()
        self.assertTrue(verdict.backend_reachable)
        self.assertEqual(verdict.action, "B")
        self.assertEqual(verdict.arbitration_source, "local_gpu_engine")

    def test_mock_local_engine_falls_back_to_remote_http(self):
        engine = PyTorchVisualDecisionEngine(model_name_or_path="not-actually-loaded")
        remote_payload = {"best_action": "B", "confidence": 0.77, "probs": {"A": 0.23, "B": 0.77}}
        bridge = CloudGPUArbiterBridge(
            local_gpu_engine=engine,
            remote_endpoint="http://example-arbiter.invalid/arbitrate",
        )

        fake_resp = MagicMock()
        fake_resp.read.return_value = json.dumps(remote_payload).encode("utf-8")
        fake_resp.__enter__.return_value = fake_resp
        fake_resp.__exit__.return_value = False

        with patch.object(engine, "score_candidates_direct") as mock_score, \
             patch(
                 "gen_zero.gateway.arbiter_bridge.urllib.request.urlopen",
                 return_value=fake_resp,
             ) as mock_urlopen:
            verdict = bridge.arbitrate(**self._candidates_args(), mode="sync")

        mock_score.assert_not_called()
        mock_urlopen.assert_called_once()
        self.assertTrue(verdict.backend_reachable)
        self.assertEqual(verdict.action, "B")
        self.assertEqual(verdict.arbitration_source, "remote_http_arbiter")

    def test_fail_closed_when_remote_returns_malformed_verdict(self):
        bridge = CloudGPUArbiterBridge(remote_endpoint="http://example-arbiter.invalid/arbitrate")
        for body in (b"{}", b"[]", b"null", b"not json", b'{"best_action": "Z"}',
                     b'{"best_action": "B", "confidence": "high"}'):
            fake_resp = MagicMock()
            fake_resp.read.return_value = body
            fake_resp.__enter__.return_value = fake_resp
            fake_resp.__exit__.return_value = False
            with self.subTest(body=body), patch(
                "gen_zero.gateway.arbiter_bridge.urllib.request.urlopen",
                return_value=fake_resp,
            ):
                verdict = bridge.arbitrate(**self._candidates_args(), mode="sync")
                self.assertFalse(verdict.backend_reachable)
                self.assertIsNone(verdict.action)
                self.assertEqual(verdict.confidence, 0.0)
                self.assertEqual(verdict.arbitration_source, "fail_closed_malformed_remote")

    def test_fail_closed_when_remote_endpoint_unreachable(self):
        engine = PyTorchVisualDecisionEngine(model_name_or_path="not-actually-loaded")
        # Port 1 is a reserved/unassigned port: connection refused is instant,
        # keeping this test fast and network-free in spirit.
        bridge = CloudGPUArbiterBridge(
            local_gpu_engine=engine,
            remote_endpoint="http://127.0.0.1:1/arbitrate",
            timeout_s=0.5,
        )

        with patch.object(engine, "score_candidates_direct") as mock_score:
            verdict = bridge.arbitrate(**self._candidates_args(), mode="sync")

        mock_score.assert_not_called()
        self.assertFalse(verdict.backend_reachable)
        self.assertEqual(verdict.confidence, 0.0)
        self.assertEqual(verdict.probs, {"A": 0.0, "B": 0.0})
        self.assertEqual(verdict.arbitration_source, "fail_closed_unreachable")

    def test_async_dispatch_skipped_when_nothing_real_to_dispatch_to(self):
        engine = PyTorchVisualDecisionEngine(model_name_or_path="not-actually-loaded")
        bridge = CloudGPUArbiterBridge(local_gpu_engine=engine, remote_endpoint=None)

        with patch.object(engine, "score_candidates_direct") as mock_score:
            verdict = bridge.arbitrate(
                state="state_payload",
                candidates=["A", "B", "ABSTAIN"],
                local_action="A",
                local_confidence=0.1,
                local_probs={"A": 0.5, "B": 0.5},
                mode="async",
            )

        mock_score.assert_not_called()
        self.assertFalse(verdict.metadata["async_dispatched_to_gpu"])
        self.assertFalse(verdict.backend_reachable)

    def test_client_end_to_end_sync_arbiter_fails_closed_when_unreachable(self):
        # Reproduces the audited scenario end-to-end: vision_engine is wired
        # in as local_gpu_engine but never loaded, and the remote endpoint is
        # unreachable. The local decision must survive untouched: a run with
        # the arbiter disabled and a run where it fails closed must agree.
        state = {"type": "obscure_maze_room_never_seen"}
        candidates = ["NORTH", "SOUTH", "EAST", "WEST"]

        baseline_cfg = GenZeroConfig(enable_gpu_arbiter_fallback=False)
        baseline = GenZero(baseline_cfg).decide(state=state, candidates=list(candidates), mode="reflex")

        cfg = GenZeroConfig(
            enable_gpu_arbiter_fallback=True,
            arbiter_confidence_threshold=0.99,
            arbiter_endpoint="http://127.0.0.1:1/arbitrate",
            arbiter_timeout_s=0.5,
        )
        client = GenZero(cfg)
        self.assertFalse(client.vision_engine.is_real)

        with patch.object(client.vision_engine, "score_candidates_direct") as mock_score:
            res = client.decide(
                state=state,
                candidates=list(candidates),
                mode="reflex",
                task_hint="sync_arbiter",
            )

        mock_score.assert_not_called()
        report = res["arbiter_fallback"]
        self.assertFalse(report["backend_reachable"])
        self.assertEqual(report["arbitration_source"], "fail_closed_unreachable")
        # Fail-closed arbiter must not perturb the local decision at all.
        self.assertEqual(res["action"], baseline["action"])
        self.assertAlmostEqual(res["confidence"], baseline["confidence"], places=4)


if __name__ == "__main__":
    unittest.main()
