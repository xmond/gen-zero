"""Unit tests for the AtomicModelContainer serving snapshot swap and rollback."""

import unittest
import time
import threading
import numpy as np

from gen_zero.daemon.atomic_container import AtomicModelContainer
from gen_zero.client import GenZero


class DummyModel:
    def __init__(self, val: int = 1):
        self.val = val


class TestAtomicContainer(unittest.TestCase):

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





if __name__ == "__main__":
    unittest.main()
