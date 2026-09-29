"""Unit tests for Gen-Zero Direction 3: 24/7 Autonomous Continuous RSI Daemon."""

import unittest
import time
import threading
import numpy as np

from gen_zero.daemon.atomic_container import AtomicModelContainer
from gen_zero.daemon.curriculum_self_play import CurriculumSelfPlayGenerator
from gen_zero.daemon.daemon_engine import GenZeroRSIDaemon
from gen_zero.client import GenZero


class DummyModel:
    def __init__(self, val: int = 1):
        self.val = val


class TestRSIDaemon(unittest.TestCase):

    def test_atomic_container_swap_and_rollback(self):
        """Verify thread-safe atomic pointer swapping and checkpoint rollback."""
        m1 = DummyModel(1)
        m2 = DummyModel(2)
        container = AtomicModelContainer(initial_model=m1, max_history=3)

        self.assertEqual(container.get_model().val, 1)
        self.assertEqual(container.get_status()["active_version"], 1)

        # 1. Swap to m2
        swap_res = container.swap_model(m2, version_tag="v2_candidate")
        self.assertEqual(swap_res["status"], "SWAPPED")
        self.assertEqual(container.get_model().val, 2)
        self.assertEqual(container.get_status()["active_version"], 2)

        # 2. Rollback
        rb_res = container.rollback()
        self.assertEqual(rb_res["status"], "ROLLED_BACK")
        self.assertEqual(container.get_model().val, 1)
        self.assertEqual(container.get_status()["active_version"], 1)

    def test_atomic_container_concurrent_reads_during_swap(self):
        """Verify zero read errors or blocking during rapid model swapping."""
        m_base = DummyModel(0)
        container = AtomicModelContainer(initial_model=m_base)

        read_errors = []
        stop_reads = threading.Event()

        def reader_worker():
            while not stop_reads.is_set():
                try:
                    m = container.get_model()
                    _ = m.val
                except Exception as e:
                    read_errors.append(e)

        threads = [threading.Thread(target=reader_worker, daemon=True) for _ in range(4)]
        for t in threads:
            t.start()

        # Perform 20 rapid swaps
        for i in range(1, 21):
            container.swap_model(DummyModel(i))

        stop_reads.set()
        for t in threads:
            t.join(timeout=1.0)

        self.assertEqual(len(read_errors), 0)
        self.assertEqual(container.get_status()["active_version"], 21)

    def test_curriculum_self_play_generator(self):
        """Verify automated generation of frontier boundary scenarios."""
        generator = CurriculumSelfPlayGenerator(difficulty_level=0.5)

        scenario = generator.generate_boundary_scenario()
        self.assertIn("scenario_id", scenario)
        self.assertIn("archetype", scenario)
        self.assertEqual(len(scenario["state_repr"]), 1024)
        self.assertIn("SAFE_ABSTAIN", scenario["candidate_actions"])

        # Test difficulty adaptation
        init_diff = generator.difficulty_level
        generator.adapt_difficulty(recent_success_rate=0.95)
        self.assertGreater(generator.difficulty_level, init_diff)

        generator.adapt_difficulty(recent_success_rate=0.20)
        self.assertLess(generator.difficulty_level, 0.6)

    def test_daemon_single_cycle_synchronous(self):
        """Verify complete end-to-end self-improvement cycle: Self-Play -> Mining -> Gate -> Reload."""
        client = GenZero()
        daemon = client.rsi_daemon

        report = daemon.run_single_evolution_cycle()
        self.assertIn("cycle_id", report)
        self.assertIn("outcome", report)
        self.assertIn("reload_status", report)
        self.assertIn(report["reload_status"], ["HOT_RELOADED", "REJECTED_ROLLED_BACK", "SKIPPED"])

        telemetry = daemon.get_telemetry()
        self.assertEqual(telemetry["total_cycles"], 1)
        self.assertGreaterEqual(telemetry["hot_reload_success_rate"], 0.0)

    def test_client_daemon_background_lifecycle(self):
        """Verify background daemon threading startup, continuous stepping, and safe shutdown."""
        client = GenZero()

        # Start with short 0.05s interval for testing
        client.start_rsi_daemon(interval_sec=0.05)
        time.sleep(0.2)

        telemetry = client.get_daemon_telemetry()
        self.assertTrue(telemetry["is_running"])
        self.assertGreater(telemetry["total_cycles"], 0)

        # Stop daemon
        client.stop_rsi_daemon()
        stopped_telemetry = client.get_daemon_telemetry()
        self.assertFalse(stopped_telemetry["is_running"])


if __name__ == "__main__":
    unittest.main()
