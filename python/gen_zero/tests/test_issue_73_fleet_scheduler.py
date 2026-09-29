"""Unit tests for Gen-Zero Issue #73: NanoCore Fleet Scheduler with zstd Hot-Swapping."""

import unittest
import os
import shutil
import tempfile
import threading
import time
import numpy as np

from gen_zero.runtime.base_nano_core import (
    BaseNanoCore,
    CheckpointBoundViolationError,
    CheckpointBudgetExceededError,
    CheckpointFormatError,
)
from gen_zero.runtime.specialist_nano_core import DomainSpecialistNanoCore
from gen_zero.runtime.nano_core_browser import NanoCoreBrowser
from gen_zero.nanocore.fleet_scheduler import (
    NanoCoreFleetScheduler,
    FleetSchedulerConfig,
    CoreDescriptor,
    FleetStatus,
    FleetQuotaExceededError,
    InvalidLeaseError,
    LeaseConflictError,
)
from gen_zero.nanocore.world_model_orchestrator import WorldModelNanoCoreOrchestrator


_REAL_LOAD = BaseNanoCore.load_checkpoint


class _LoaderSpy:
    """Wraps the real checkpoint loader and records every call: the budget
    it was handed and whether it loaded or refused."""

    def __init__(self, delay=0.0):
        self.delay = delay
        self.calls = []
        self._lock = threading.Lock()
        self.in_flight = {}
        self.dup_loads = []

    def __call__(self, path, verify_checksum=True, max_bytes=None):
        with self._lock:
            self.in_flight[path] = self.in_flight.get(path, 0) + 1
            if self.in_flight[path] > 1:
                self.dup_loads.append(path)
        record = {"path": path, "max_bytes": max_bytes}
        try:
            if self.delay:
                time.sleep(self.delay)
            inst = _REAL_LOAD(path, verify_checksum=verify_checksum, max_bytes=max_bytes)
            record["result"] = "ok"
            return inst
        except BaseException as e:
            record["result"] = type(e).__name__
            raise
        finally:
            with self._lock:
                self.in_flight[path] -= 1
                self.calls.append(record)

    def results(self):
        with self._lock:
            return [c["result"] for c in self.calls]


class TestNanoCoreFleetScheduler(unittest.TestCase):
    """Test suite for Issue #73 specialist fleet scheduler."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="gen_zero_fleet_test_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_fleet_registration_and_resident_lifecycle(self):
        """Validates registering micro-cores, persisting to zstd, and cold loading."""
        config = FleetSchedulerConfig(
            max_resident_cores=3,
            storage_dir=self.temp_dir,
            compression_level=3,
        )
        scheduler = NanoCoreFleetScheduler(config)

        # Create two specialist cores
        core_ops = DomainSpecialistNanoCore(domain="ops", state_dim=128, candidate_dim=128, embed_dim=64, seed=101)
        core_db = DomainSpecialistNanoCore(domain="database", state_dim=128, candidate_dim=128, embed_dim=64, seed=102)

        desc_ops = scheduler.register_instance("core_ops", core_ops, persist=True)
        desc_db = scheduler.register_instance("core_db", core_db, persist=True)

        self.assertEqual(scheduler.registered_count, 2)
        self.assertEqual(scheduler.resident_count, 2)
        self.assertTrue(os.path.isfile(desc_ops.checkpoint_path))
        self.assertTrue(os.path.isfile(desc_db.checkpoint_path))

        # Check that file is compressed and smaller than uncompressed raw bytes
        file_size = os.path.getsize(desc_ops.checkpoint_path)
        self.assertGreater(file_size, 0)
        self.assertLess(file_size, 500 * 1024)  # Very compact < 500KB

        # Hot hit test
        t0 = time.perf_counter()
        retrieved_ops = scheduler.acquire_core("core_ops")
        hot_lat = (time.perf_counter() - t0) * 1000.0
        self.assertIs(retrieved_ops, core_ops)
        self.assertLess(hot_lat, 1.0)  # Sub-millisecond hit

        status = scheduler.get_fleet_status()
        self.assertEqual(status.hits, 1)
        self.assertEqual(status.misses, 0)

    def test_02_lru_eviction_and_memory_bounding(self):
        """Validates that resident capacity is strictly bounded by max_resident_cores under LRU."""
        config = FleetSchedulerConfig(
            max_resident_cores=2,  # Strict capacity of 2 cores
            storage_dir=self.temp_dir,
        )
        scheduler = NanoCoreFleetScheduler(config)

        # Register 4 cores
        cores = []
        for i in range(4):
            c = DomainSpecialistNanoCore(domain=f"domain_{i}", state_dim=64, candidate_dim=64, embed_dim=32, seed=i)
            scheduler.register_instance(f"core_{i}", c, persist=True)
            cores.append(c)

        # Because max_resident_cores=2, only the last 2 registered should reside in RAM
        self.assertEqual(scheduler.registered_count, 4)
        self.assertLessEqual(scheduler.resident_count, 2)

        # Access core_0 (currently cold -> should trigger miss and evict)
        c0 = scheduler.acquire_core("core_0")
        self.assertIsNotNone(c0)
        self.assertTrue(scheduler.is_resident("core_0"))
        self.assertLessEqual(scheduler.resident_count, 2)

        # Access core_1 -> resident now has {core_0, core_1}
        c1 = scheduler.acquire_core("core_1")
        self.assertIsNotNone(c1)
        self.assertTrue(scheduler.is_resident("core_0"))
        self.assertTrue(scheduler.is_resident("core_1"))
        self.assertEqual(scheduler.resident_count, 2)

        # Access core_2 -> core_0 was accessed least recently, so core_0 should be evicted!
        c2 = scheduler.acquire_core("core_2")
        self.assertIsNotNone(c2)
        self.assertTrue(scheduler.is_resident("core_1"))
        self.assertTrue(scheduler.is_resident("core_2"))
        self.assertFalse(scheduler.is_resident("core_0"))

        status = scheduler.get_fleet_status()
        self.assertGreaterEqual(status.evictions, 2)

    def test_03_pinned_cores_immunity(self):
        """Validates that pinned cores are protected from LRU eviction."""
        config = FleetSchedulerConfig(
            max_resident_cores=2,
            storage_dir=self.temp_dir,
        )
        scheduler = NanoCoreFleetScheduler(config)

        core_pinned = DomainSpecialistNanoCore(domain="security", state_dim=64, candidate_dim=64, embed_dim=32, seed=777)
        core_a = DomainSpecialistNanoCore(domain="domain_a", state_dim=64, candidate_dim=64, embed_dim=32, seed=1)
        core_b = DomainSpecialistNanoCore(domain="domain_b", state_dim=64, candidate_dim=64, embed_dim=32, seed=2)
        core_c = DomainSpecialistNanoCore(domain="domain_c", state_dim=64, candidate_dim=64, embed_dim=32, seed=3)

        scheduler.register_instance("pinned_core", core_pinned, pin=True, persist=True)
        scheduler.register_instance("core_a", core_a, pin=False, persist=True)
        scheduler.register_instance("core_b", core_b, pin=False, persist=True)
        scheduler.register_instance("core_c", core_c, pin=False, persist=True)

        # Ensure pinned core is loaded
        scheduler.acquire_core("pinned_core")
        self.assertTrue(scheduler.is_resident("pinned_core"))

        # Access other unpinned cores in sequence
        scheduler.acquire_core("core_a")
        scheduler.acquire_core("core_b")
        scheduler.acquire_core("core_c")

        # Pinned core must still reside in RAM!
        self.assertTrue(scheduler.is_resident("pinned_core"))

    def test_04_thread_safe_concurrent_access(self):
        """Validates thread safety and data integrity under multi-threaded alternating access."""
        config = FleetSchedulerConfig(
            max_resident_cores=3,
            storage_dir=self.temp_dir,
        )
        scheduler = NanoCoreFleetScheduler(config)

        # Register 6 cores
        for i in range(6):
            c = DomainSpecialistNanoCore(domain=f"worker_{i}", state_dim=64, candidate_dim=64, embed_dim=32, seed=i)
            scheduler.register_instance(f"worker_{i}", c, persist=True)

        num_threads = 8
        iterations_per_thread = 25
        errors = []

        def worker_thread(thread_id: int):
            try:
                for it in range(iterations_per_thread):
                    core_name = f"worker_{(thread_id + it) % 6}"
                    with scheduler.lease_core(core_name) as core:
                        res = core.score_candidates(
                            state_repr=[0.1] * 64,
                            candidates=["ACTION_A", "ACTION_B", "ACTION_C"]
                        )
                        if "best_action" not in res:
                            errors.append(f"Thread {thread_id} got malformed response")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker_thread, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Concurrent execution had errors: {errors}")
        status = scheduler.get_fleet_status()
        self.assertEqual(status.total_requests, num_threads * iterations_per_thread)
        self.assertLessEqual(status.resident_cores_count, 3)

    def test_05_bit_exact_weights_and_zero_accuracy_loss(self):
        """Validates that cold swapping via zstd produces bit-exact weight restoration (RMSE <= 10^-7)."""
        config = FleetSchedulerConfig(
            max_resident_cores=1,  # Force immediate eviction on switch
            storage_dir=self.temp_dir,
        )
        scheduler = NanoCoreFleetScheduler(config)

        # Core with calibrated deterministic weights
        core_orig = DomainSpecialistNanoCore(domain="audit", state_dim=128, candidate_dim=128, embed_dim=64, seed=999)
        original_weights = {k: v.copy() for k, v in core_orig.weights.items()}

        scheduler.register_instance("core_audit", core_orig, persist=True)

        # Perform initial scoring
        state = np.random.RandomState(42).randn(128).astype(np.float32)
        candidates = ["QUARANTINE_NODE", "RESTART_POD", "DRAIN_TRAFFIC"]
        initial_res = scheduler.score_candidates("core_audit", state, candidates)

        # Force eviction by acquiring a different core
        core_other = DomainSpecialistNanoCore(domain="other", state_dim=128, candidate_dim=128, embed_dim=64, seed=888)
        scheduler.register_instance("core_other", core_other, persist=True)
        scheduler.acquire_core("core_other")

        self.assertFalse(scheduler.is_resident("core_audit"))

        # Cold swap core_audit back into memory
        t0 = time.perf_counter()
        reloaded_core = scheduler.acquire_core("core_audit")
        cold_lat = (time.perf_counter() - t0) * 1000.0

        # Sub-16ms cold reload requirement
        self.assertLess(cold_lat, 16.0)

        # Weight bit-exactness check (RMSE <= 10^-7)
        for k, orig_v in original_weights.items():
            reloaded_v = reloaded_core.weights[k]
            rmse = float(np.sqrt(np.mean((orig_v - reloaded_v) ** 2)))
            self.assertLessEqual(rmse, 1e-7, f"Weight {k} RMSE {rmse} exceeded 10^-7")

        # Decision parity check (identical logits and best action)
        after_res = scheduler.score_candidates("core_audit", state, candidates)
        self.assertEqual(initial_res["best_action"], after_res["best_action"])
        self.assertAlmostEqual(initial_res["confidence"], after_res["confidence"], places=5)
        for c in candidates:
            self.assertAlmostEqual(initial_res["probs"][c], after_res["probs"][c], places=5)

    def test_06_orchestrator_integration(self):
        """Validates that WorldModelNanoCoreOrchestrator can utilize NanoCoreFleetScheduler."""
        config = FleetSchedulerConfig(
            max_resident_cores=2,
            storage_dir=self.temp_dir,
        )
        scheduler = NanoCoreFleetScheduler(config)

        # Register browser core and specialist core
        browser_core = NanoCoreBrowser(state_dim=128, candidate_dim=128, embed_dim=64)
        scheduler.register_instance("browser_core", browser_core, persist=True)

        ops_core = DomainSpecialistNanoCore(domain="ops", state_dim=128, candidate_dim=128, embed_dim=64, seed=123)
        scheduler.register_instance("ops_core", ops_core, persist=True)

        orchestrator = WorldModelNanoCoreOrchestrator(
            latent_dim=128,
            action_dim=32,
            fleet_scheduler=scheduler,
        )
        self.assertIs(orchestrator.fleet_scheduler, scheduler)

        # Verify orchestrator runs imagine step smoothly
        state = np.zeros(128, dtype=np.float32)
        res = orchestrator.imagine_and_orchestrate(
            state=state,
            candidate_actions=["RECOVER", "ROLLBACK", "ALERT"],
            safety_evaluator=lambda z, act: 0.9,
        )
        self.assertIn(res.selected_action, ["RECOVER", "ROLLBACK", "ALERT"])
        self.assertGreater(res.confidence, 0.0)

    def test_07_client_convenience_api(self):
        """Validates GenZeroClient convenience factory create_fleet_scheduler."""
        from gen_zero.client import GenZeroClient
        client = GenZeroClient()
        fleet = client.create_fleet_scheduler(
            max_resident_cores=4,
            storage_dir=self.temp_dir,
        )
        self.assertIsInstance(fleet, NanoCoreFleetScheduler)
        self.assertEqual(fleet.config.max_resident_cores, 4)

    def test_08_quota_exceeded_under_active_leases_and_pins(self):
        """Validates that QuotaExceededError is raised when capacity cannot be freed."""
        from gen_zero.nanocore.fleet_scheduler import FleetQuotaExceededError, QuotaExceededError

        config = FleetSchedulerConfig(
            max_resident_cores=2,
            storage_dir=self.temp_dir,
        )
        scheduler = NanoCoreFleetScheduler(config)

        # Register 3 cores
        core1 = DomainSpecialistNanoCore(domain="core1", state_dim=64, candidate_dim=64, embed_dim=32, seed=1)
        core2 = DomainSpecialistNanoCore(domain="core2", state_dim=64, candidate_dim=64, embed_dim=32, seed=2)
        core3 = DomainSpecialistNanoCore(domain="core3", state_dim=64, candidate_dim=64, embed_dim=32, seed=3)

        scheduler.register_instance("core1", core1, pin=True, persist=True)
        scheduler.register_instance("core2", core2, pin=False, persist=True)
        save_res = core2.save_checkpoint(os.path.join(self.temp_dir, "core3.zst"), compress=True)
        scheduler.register_checkpoint("core3", save_res["path"])

        # core1 is pinned, lease core2 so both resident cores are protected
        with scheduler.lease_core("core2"):
            self.assertEqual(scheduler.resident_count, 2)
            # Acquiring cold core3 should raise FleetQuotaExceededError since neither core1 nor core2 can be evicted
            with self.assertRaises(FleetQuotaExceededError):
                scheduler.acquire_core("core3", timeout=0.02)

        # After lease on core2 expires, acquiring core3 should succeed by evicting core2!
        c3 = scheduler.acquire_core("core3")
        self.assertIsNotNone(c3)
        self.assertTrue(scheduler.is_resident("core3"))
        self.assertTrue(scheduler.is_resident("core1"))  # core1 remained pinned
        self.assertFalse(scheduler.is_resident("core2"))  # core2 was evicted

    def test_09_f07_atomic_lease_acquisition(self):
        """Validates F07: acquire_lease atomically increments active_leases within lock."""
        config = FleetSchedulerConfig(max_resident_cores=2, storage_dir=self.temp_dir)
        scheduler = NanoCoreFleetScheduler(config)
        core = DomainSpecialistNanoCore(domain="test_f07", state_dim=64, candidate_dim=64, embed_dim=32, seed=7)
        scheduler.register_instance("core_f07", core, persist=True)

        lease = scheduler.acquire_lease("core_f07")
        self.assertIs(lease.core, core)
        desc = scheduler.get_core_descriptor("core_f07")
        self.assertEqual(desc.active_leases, 1)
        self.assertEqual((lease.core_id, lease.generation), ("core_f07", desc.generation))

        # Release the lease; a second release of the same lease is refused.
        scheduler.release_lease(lease)
        self.assertEqual(desc.active_leases, 1, "a snapshot must not track later ledger changes")
        self.assertEqual(scheduler.get_core_descriptor("core_f07").active_leases, 0)
        with self.assertRaises(InvalidLeaseError):
            scheduler.release_lease(lease)

    def test_10_f06_declared_footprint_over_budget_never_loads(self):
        """F06: a checkpoint whose header declares more than the whole budget is
        refused at admission; the loader is never called."""
        from unittest.mock import patch

        config = FleetSchedulerConfig(max_resident_cores=5, max_resident_bytes=1000, storage_dir=self.temp_dir)
        scheduler = NanoCoreFleetScheduler(config)
        path = self._declared_ckpt("core_f06", 6, 2000)
        desc = scheduler.register_checkpoint("core_f06", path)
        self.assertEqual(desc.memory_footprint_bytes, 2000)
        self.assertGreater(desc.load_peak_bytes, 2000)

        spy = _LoaderSpy()
        with patch.object(BaseNanoCore, "load_checkpoint", side_effect=spy):
            with self.assertRaises(FleetQuotaExceededError):
                scheduler.acquire_core("core_f06")
        self.assertEqual(spy.calls, [])
        self.assertFalse(scheduler.is_resident("core_f06"))

    def test_11_f08_evict_core_active_lease_protection(self):
        """Validates F08: evict_core(force=False) must never evict cores with active_leases > 0."""
        config = FleetSchedulerConfig(max_resident_cores=2, storage_dir=self.temp_dir)
        scheduler = NanoCoreFleetScheduler(config)
        core = DomainSpecialistNanoCore(domain="test_f08", state_dim=64, candidate_dim=64, embed_dim=32, seed=8)
        scheduler.register_instance("core_f08", core, persist=True)

        with scheduler.lease_core("core_f08"):
            # Attempt to evict active lease without force
            evicted = scheduler.evict_core("core_f08", force=False)
            self.assertFalse(evicted)
            self.assertTrue(scheduler.is_resident("core_f08"))

        # After lease is released, eviction succeeds
        evicted_after = scheduler.evict_core("core_f08", force=False)
        self.assertTrue(evicted_after)
        self.assertFalse(scheduler.is_resident("core_f08"))

    def test_12_r13_02_f06_reservation_ledger_prevents_overshoot(self):
        """R13-02 F06: budget 100MiB, A resident + leased at 70MiB, B declares
        40MiB. 70 + 40 > 100, so B is never admitted and never loaded."""
        from unittest.mock import patch

        MiB = 1024 * 1024
        config = FleetSchedulerConfig(max_resident_cores=5, max_resident_bytes=100 * MiB, storage_dir=self.temp_dir)
        scheduler = NanoCoreFleetScheduler(config)
        scheduler.register_instance("core_a", self._sized_core("A", 1, 70 * MiB), persist=True)
        scheduler.acquire_lease("core_a")
        self.assertEqual(scheduler.get_core_descriptor("core_a").active_leases, 1)
        scheduler.register_checkpoint("core_b", self._declared_ckpt("core_b", 2, 40 * MiB))

        spy = _LoaderSpy()
        with patch.object(BaseNanoCore, "load_checkpoint", side_effect=spy):
            with self.assertRaises(FleetQuotaExceededError):
                scheduler.acquire_core("core_b", timeout=0.05)

        self.assertEqual(spy.calls, [], "B was handed to a loader without a reservation covering it")
        self.assertFalse(scheduler.is_resident("core_b"))
        status = scheduler.get_fleet_status()
        self.assertEqual(status.resident_memory_bytes, 70 * MiB)
        self.assertEqual(status.reserved_memory_bytes, 0)
        self.assertEqual(status.loading_core_ids, [])

    def test_13_r13_02_f03_concurrent_waiters_reuse_same_resident_core(self):
        """Validates R13-02 F03: when two threads concurrently wait to cold-load
        the SAME core_id and the capacity-wait condition variable wakes both,
        the loser must recognize the target is already resident (loaded by the
        winner) and reuse it -- not re-block waiting for "one more slot" of
        physical capacity it will never get (since the winner already
        legitimately occupies that slot with an active lease).

        Reproduces the review's 8/8 repro: T2 must not spuriously fail/timeout.
        """
        MiB = 1024 * 1024
        config = FleetSchedulerConfig(
            max_resident_cores=1,
            max_resident_bytes=15 * MiB,
            storage_dir=self.temp_dir,
            acquire_timeout_seconds=5.0,
        )
        scheduler = NanoCoreFleetScheduler(config)

        # Core A occupies the only slot, held under an active lease so it is
        # initially unevictable -- this forces both waiter threads to block.
        core_a = DomainSpecialistNanoCore(domain="A", state_dim=64, candidate_dim=64, embed_dim=32, seed=1)
        core_a.memory_footprint_bytes = lambda: 10 * MiB
        scheduler.register_instance("core_a", core_a, persist=True)
        lease_a = scheduler.acquire_lease("core_a")

        core_b_ckpt = DomainSpecialistNanoCore(domain="B", state_dim=64, candidate_dim=64, embed_dim=32, seed=2)
        path_b = os.path.join(self.temp_dir, "core_b_race.zst")
        core_b_ckpt.save_checkpoint(path_b, compress=True)
        scheduler.register_checkpoint("core_b", path_b)

        fake_b_instance = DomainSpecialistNanoCore(domain="B", state_dim=64, candidate_dim=64, embed_dim=32, seed=2)

        load_count = [0]
        count_lock = threading.Lock()
        load_started = threading.Event()
        proceed_load = threading.Event()

        def fake_load(path, verify_checksum=True, max_bytes=None):
            with count_lock:
                load_count[0] += 1
            load_started.set()
            proceed_load.wait(timeout=5.0)
            return fake_b_instance

        results = {}
        errors = []
        barrier = threading.Barrier(2)

        def waiter(name):
            try:
                barrier.wait(timeout=5.0)
                results[name] = scheduler.acquire_lease("core_b", timeout=3.0).core
            except Exception as e:  # noqa: BLE001 - capture for assertion below
                errors.append((name, repr(e)))

        from unittest.mock import patch
        with patch.object(BaseNanoCore, "load_checkpoint", side_effect=fake_load):
            t1 = threading.Thread(target=waiter, args=("t1",))
            t2 = threading.Thread(target=waiter, args=("t2",))
            t1.start()
            t2.start()

            # Let both threads enter the capacity-wait loop while core_a is
            # still leased (unevictable) -- neither should have started loading.
            time.sleep(0.2)
            self.assertFalse(load_started.is_set(), "a load started before capacity was freed")

            # Free capacity: releasing core_a's lease wakes both waiters.
            scheduler.release_lease(lease_a)

            # Exactly one thread should win the race and start the (blocked) load.
            self.assertTrue(load_started.wait(timeout=5.0), "no thread started loading core_b")
            proceed_load.set()

            t1.join(timeout=5.0)
            t2.join(timeout=5.0)

        self.assertEqual(errors, [], f"waiter(s) raised unexpectedly: {errors}")
        self.assertEqual(load_count[0], 1, "core_b was physically loaded more than once")
        self.assertIn("t1", results)
        self.assertIn("t2", results)
        self.assertIs(results["t1"], results["t2"], "waiters received two different core_b instances")
        self.assertIs(results["t1"], fake_b_instance)

        desc_b = scheduler.get_core_descriptor("core_b")
        self.assertEqual(desc_b.active_leases, 2)

    def test_14_force_evict_never_bypasses_active_lease(self):
        """Validates that evict_core(force=True) must raise rather than silently
        deleting a core that still has active leases."""
        from gen_zero.nanocore.fleet_scheduler import FleetActiveLeaseError

        config = FleetSchedulerConfig(max_resident_cores=2, storage_dir=self.temp_dir)
        scheduler = NanoCoreFleetScheduler(config)
        core = DomainSpecialistNanoCore(domain="test_force", state_dim=64, candidate_dim=64, embed_dim=32, seed=20)
        scheduler.register_instance("core_force", core, persist=True)

        with scheduler.lease_core("core_force"):
            with self.assertRaises(FleetActiveLeaseError):
                scheduler.evict_core("core_force", force=True)
            self.assertTrue(scheduler.is_resident("core_force"), "force=True deleted a leased core")

        # Once released, force=True proceeds normally.
        self.assertTrue(scheduler.evict_core("core_force", force=True))
        self.assertFalse(scheduler.is_resident("core_force"))

    def test_15_clear_retains_active_lease_cores(self):
        """Validates that clear() never drops a core with active leases, even
        though it wipes every other resident core."""
        config = FleetSchedulerConfig(max_resident_cores=5, storage_dir=self.temp_dir)
        scheduler = NanoCoreFleetScheduler(config)

        core_leased = DomainSpecialistNanoCore(domain="leased", state_dim=64, candidate_dim=64, embed_dim=32, seed=21)
        core_idle = DomainSpecialistNanoCore(domain="idle", state_dim=64, candidate_dim=64, embed_dim=32, seed=22)
        scheduler.register_instance("core_leased", core_leased, persist=True)
        scheduler.register_instance("core_idle", core_idle, persist=True)

        with scheduler.lease_core("core_leased"):
            scheduler.clear()
            self.assertTrue(scheduler.is_resident("core_leased"), "clear() dropped a core with an active lease")
            self.assertFalse(scheduler.is_resident("core_idle"))

        # Once released, a subsequent clear() removes it too.
        scheduler.clear()
        self.assertFalse(scheduler.is_resident("core_leased"))

    # ------------------------------------------------------------------
    # X-F01 / X-F02 / X-F03 (R13-02 capacity lifecycle, second review)
    # ------------------------------------------------------------------

    def _ckpt(self, name, seed):
        core = DomainSpecialistNanoCore(domain=name, state_dim=64, candidate_dim=64, embed_dim=32, seed=seed)
        path = os.path.join(self.temp_dir, f"{name}.zst")
        core.save_checkpoint(path, compress=True)
        return path

    def _sized_core(self, name, seed, size_bytes):
        core = DomainSpecialistNanoCore(domain=name, state_dim=64, candidate_dim=64, embed_dim=32, seed=seed)
        core.memory_footprint_bytes = lambda: size_bytes
        return core

    def _declared_ckpt(self, name, seed, declared_bytes, path=None):
        """Real checkpoint whose header declares `declared_bytes` resident.
        The real weights are small (~29KB), so declared_bytes must be larger."""
        path = path or os.path.join(self.temp_dir, f"{name}.zst")
        self._sized_core(name, seed, declared_bytes).save_checkpoint(path, compress=True)
        return path

    def _ballast_ckpt(self, name, seed, ballast_bytes, path=None):
        """Real checkpoint that physically allocates ~ballast_bytes on load."""
        core = DomainSpecialistNanoCore(domain=name, state_dim=64, candidate_dim=64, embed_dim=32, seed=seed)
        core.weights["ballast"] = np.ones(ballast_bytes // 4, dtype=np.float32)
        path = path or os.path.join(self.temp_dir, f"{name}.zst")
        core.save_checkpoint(path, compress=True)
        return path

    @staticmethod
    def _wait_for(predicate, timeout=5.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if predicate():
                return True
            time.sleep(0.005)
        return predicate()

    def _assert_ledger_within_budget(self, scheduler):
        status = scheduler.get_fleet_status()
        self.assertLessEqual(
            status.resident_memory_bytes + status.reserved_memory_bytes, status.max_resident_bytes
        )
        self.assertLessEqual(
            status.resident_cores_count + len(status.loading_core_ids), status.max_resident_cores
        )
        return status

    def test_16_xf01_grown_image_is_refused_before_physical_allocation(self):
        """X-F01, the review's probe with a real payload: budget 100MiB, A leased
        at 70MiB, B registered at 20MiB but its image on disk was replaced by
        one that physically allocates 40MiB. The real loader must refuse B
        BEFORE allocating (traced peak stays tiny), B waits with nothing
        reserved, and once A is released B loads with a traced peak that
        stays inside the reservation it was given."""
        import tracemalloc
        from unittest.mock import patch

        MiB = 1024 * 1024
        config = FleetSchedulerConfig(max_resident_cores=5, max_resident_bytes=100 * MiB, storage_dir=self.temp_dir)
        scheduler = NanoCoreFleetScheduler(config)
        scheduler.register_instance("core_a", self._sized_core("A", 1, 70 * MiB), persist=True)
        lease_a = scheduler.acquire_lease("core_a")
        path_b = self._declared_ckpt("core_b", 2, 20 * MiB)
        scheduler.register_checkpoint("core_b", path_b)
        self._ballast_ckpt("core_b_40m", 2, 40 * MiB, path=path_b)

        spy = _LoaderSpy()
        violations = []
        stop = threading.Event()

        def sampler():
            while not stop.is_set():
                st = scheduler.get_fleet_status()
                if st.resident_memory_bytes + st.reserved_memory_bytes > st.max_resident_bytes:
                    violations.append(st.to_dict())
                time.sleep(0.001)

        result = {}

        def loader():
            try:
                result["lease"] = scheduler.acquire_lease("core_b", timeout=10.0)
            except Exception as e:  # noqa: BLE001
                result["error"] = repr(e)

        tracemalloc.start()
        self.addCleanup(tracemalloc.stop)
        with patch.object(BaseNanoCore, "load_checkpoint", side_effect=spy):
            samp = threading.Thread(target=sampler, daemon=True)
            samp.start()
            self.addCleanup(stop.set)
            tracemalloc.reset_peak()
            t1 = threading.Thread(target=loader, daemon=True)
            t1.start()

            self.assertTrue(self._wait_for(lambda: len(spy.calls) == 1 and not scheduler.is_loading("core_b")))
            _, refused_peak = tracemalloc.get_traced_memory()
            self.assertEqual(spy.results(), ["CheckpointBudgetExceededError"])
            self.assertLess(refused_peak, 4 * MiB, f"refused load still allocated {refused_peak} bytes")
            desc_b = scheduler.get_core_descriptor("core_b")
            self.assertGreaterEqual(desc_b.load_peak_bytes, 80 * MiB)
            time.sleep(0.1)
            self.assertTrue(t1.is_alive(), "loader should be waiting for capacity")
            self.assertFalse(scheduler.is_resident("core_b"))
            status = self._assert_ledger_within_budget(scheduler)
            self.assertEqual(status.resident_memory_bytes, 70 * MiB)
            self.assertEqual(status.reserved_memory_bytes, 0)

            tracemalloc.reset_peak()
            base_mem, _ = tracemalloc.get_traced_memory()
            scheduler.release_lease(lease_a)
            t1.join(timeout=10.0)
            _, load_peak = tracemalloc.get_traced_memory()
            stop.set()
            samp.join(timeout=5.0)

        self.assertNotIn("error", result, result.get("error"))
        self.assertEqual(violations, [])
        self.assertEqual(spy.results(), ["CheckpointBudgetExceededError", "ok"])
        self.assertEqual(spy.calls[1]["max_bytes"], desc_b.load_peak_bytes)
        self.assertGreater(load_peak - base_mem, 40 * MiB, "the ballast was not physically allocated")
        self.assertLessEqual(
            load_peak - base_mem, desc_b.load_peak_bytes,
            "physical allocation during the load exceeded the reservation it was given",
        )
        self.assertFalse(scheduler.is_resident("core_a"), "idle A should have been evicted to fit B")
        status = self._assert_ledger_within_budget(scheduler)
        self.assertEqual(status.resident_memory_bytes, result["lease"].core.memory_footprint_bytes())
        self.assertGreater(status.resident_memory_bytes, 40 * MiB)
        scheduler.release_lease(result["lease"])

    def test_17_xf01_grown_image_re_reserves_by_evicting_idle_core(self):
        """X-F01: when the image grew but an idle core can be evicted, the
        refused load re-reserves with the declared bound and loads once."""
        from unittest.mock import patch

        MiB = 1024 * 1024
        config = FleetSchedulerConfig(max_resident_cores=5, max_resident_bytes=100 * MiB, storage_dir=self.temp_dir)
        scheduler = NanoCoreFleetScheduler(config)
        scheduler.register_instance("core_a", self._sized_core("A", 1, 70 * MiB), persist=True)  # idle
        path_b = self._declared_ckpt("core_b", 2, 20 * MiB)
        scheduler.register_checkpoint("core_b", path_b)
        self._declared_ckpt("core_b", 2, 40 * MiB, path=path_b)
        spy = _LoaderSpy()

        with patch.object(BaseNanoCore, "load_checkpoint", side_effect=spy):
            core_b = scheduler.acquire_core("core_b", timeout=0.5)
        self.assertEqual(spy.results(), ["CheckpointBudgetExceededError", "ok"])
        self.assertLess(spy.calls[0]["max_bytes"], 40 * MiB)
        self.assertGreater(spy.calls[1]["max_bytes"], 40 * MiB)
        self.assertFalse(scheduler.is_resident("core_a"))
        status = self._assert_ledger_within_budget(scheduler)
        # The reservation shrank to the measured footprint at commit.
        self.assertEqual(status.resident_memory_bytes, core_b.memory_footprint_bytes())
        self.assertEqual(status.reserved_memory_bytes, 0)

    def test_18_xf02_waiter_joins_in_flight_load_instead_of_loading_twice(self):
        """X-F02: while T1 is physically LOADING B, T2 asking for B must park on
        B's ticket, not start a second load, then share T1's instance."""
        from unittest.mock import patch

        config = FleetSchedulerConfig(max_resident_cores=5, storage_dir=self.temp_dir)
        scheduler = NanoCoreFleetScheduler(config)
        scheduler.register_checkpoint("core_b", self._ckpt("core_b", 2))
        instance_b = self._sized_core("B", 2, 1024)
        loads = []
        started = threading.Event()
        proceed = threading.Event()

        def fake_load(path, verify_checksum=True, max_bytes=None):
            loads.append(path)
            started.set()
            self.assertTrue(proceed.wait(timeout=5.0))
            return instance_b

        results, errors = {}, []

        def worker(name):
            try:
                results[name] = scheduler.acquire_lease("core_b", timeout=5.0)
            except Exception as e:  # noqa: BLE001
                errors.append((name, repr(e)))

        with patch.object(BaseNanoCore, "load_checkpoint", side_effect=fake_load):
            t1 = threading.Thread(target=worker, args=("t1",), daemon=True)
            t1.start()
            self.assertTrue(started.wait(timeout=5.0))
            self.assertTrue(scheduler.is_loading("core_b"))
            # The lock is free during the load: other callers are not blocked.
            self.assertEqual(scheduler.get_fleet_status().loading_core_ids, ["core_b"])

            t2 = threading.Thread(target=worker, args=("t2",), daemon=True)
            t2.start()
            time.sleep(0.2)
            self.assertEqual(len(loads), 1, "T2 started a second load of a LOADING core")
            self.assertTrue(t2.is_alive(), "T2 should wait on the in-flight load")

            proceed.set()
            t1.join(timeout=5.0)
            t2.join(timeout=5.0)

        self.assertEqual(errors, [])
        self.assertEqual(len(loads), 1)
        self.assertIs(results["t1"].core, instance_b)
        self.assertIs(results["t2"].core, instance_b)
        self.assertNotEqual(results["t1"].token, results["t2"].token)
        self.assertEqual(scheduler.get_core_descriptor("core_b").active_leases, 2)
        status = scheduler.get_fleet_status()
        self.assertEqual((status.misses, status.hits), (1, 1))

    def test_19_xf02_refused_load_then_concurrent_request_loads_once(self):
        """X-F02 with X-F01: T1's load of B is refused for a grown image and T1
        waits for capacity; T2 then requests B. When capacity frees, B is
        loaded exactly once more and both share one instance."""
        from unittest.mock import patch

        MiB = 1024 * 1024
        config = FleetSchedulerConfig(max_resident_cores=5, max_resident_bytes=100 * MiB, storage_dir=self.temp_dir)
        scheduler = NanoCoreFleetScheduler(config)
        scheduler.register_instance("core_a", self._sized_core("A", 1, 70 * MiB), persist=True)
        lease_a = scheduler.acquire_lease("core_a")
        path_b = self._declared_ckpt("core_b", 2, 20 * MiB)
        scheduler.register_checkpoint("core_b", path_b)
        self._declared_ckpt("core_b", 2, 40 * MiB, path=path_b)
        spy = _LoaderSpy(delay=0.05)
        results, errors = {}, []

        def worker(name):
            try:
                results[name] = scheduler.acquire_lease("core_b", timeout=5.0)
            except Exception as e:  # noqa: BLE001
                errors.append((name, repr(e)))

        with patch.object(BaseNanoCore, "load_checkpoint", side_effect=spy):
            t1 = threading.Thread(target=worker, args=("t1",), daemon=True)
            t1.start()
            self.assertTrue(self._wait_for(lambda: len(spy.calls) == 1 and not scheduler.is_loading("core_b")))
            t2 = threading.Thread(target=worker, args=("t2",), daemon=True)
            t2.start()
            time.sleep(0.2)
            self.assertTrue(t1.is_alive() and t2.is_alive())
            self.assertEqual(len(spy.calls), 1)
            self._assert_ledger_within_budget(scheduler)

            scheduler.release_lease(lease_a)
            t1.join(timeout=5.0)
            t2.join(timeout=5.0)

        self.assertEqual(errors, [])
        self.assertEqual(spy.results(), ["CheckpointBudgetExceededError", "ok"])
        self.assertEqual(spy.dup_loads, [])
        self.assertIs(results["t1"].core, results["t2"].core)
        self.assertEqual(scheduler.get_core_descriptor("core_b").active_leases, 2)
        status = self._assert_ledger_within_budget(scheduler)
        self.assertEqual(status.resident_core_ids, ["core_b"])

    def test_20_xf02_defensive_reuse_when_target_already_resident_at_commit(self):
        """X-F02 defensive branch (white-box: unreachable via the public API,
        because only the ticket holder installs a core). If the target is
        already resident when the loader commits, the loader discards its own
        payload and returns the resident instance."""
        from unittest.mock import patch

        config = FleetSchedulerConfig(max_resident_cores=5, storage_dir=self.temp_dir)
        scheduler = NanoCoreFleetScheduler(config)
        scheduler.register_checkpoint("core_b", self._ckpt("core_b", 2))
        resident_b = self._sized_core("B", 2, 1024)
        duplicate_b = self._sized_core("B", 2, 1024)

        def fake_load(path, verify_checksum=True, max_bytes=None):
            with scheduler._lock:
                scheduler._resident_cores["core_b"] = resident_b
            return duplicate_b

        with patch.object(BaseNanoCore, "load_checkpoint", side_effect=fake_load):
            lease = scheduler.acquire_lease("core_b")
        self.assertIs(lease.core, resident_b)
        self.assertFalse(scheduler.is_loading("core_b"))
        self.assertEqual(scheduler.get_core_descriptor("core_b").active_leases, 1)

    def test_21_xf02_xf01_concurrent_stress_keeps_ledger_and_single_load(self):
        """Stress: 8 threads lease 6 cores; three images grow from 5MiB to 10MiB
        after registration. Invariants: ledger <= budget at every sample,
        never two concurrent loads of one core_id, every load call gets a
        budget covering what it loads, each grown image is refused at most
        once, every lease released cleanly."""
        import random
        from unittest.mock import patch

        MiB = 1024 * 1024
        config = FleetSchedulerConfig(
            max_resident_cores=3, max_resident_bytes=40 * MiB, storage_dir=self.temp_dir,
            acquire_timeout_seconds=10.0,
        )
        scheduler = NanoCoreFleetScheduler(config)
        grown = set()
        for i in range(6):
            cid = f"core_{i}"
            path = self._declared_ckpt(cid, i, 5 * MiB)
            scheduler.register_checkpoint(cid, path)
            if i % 2 == 0:
                self._declared_ckpt(cid, i, 10 * MiB, path=path)
                grown.add(os.path.abspath(path))

        spy = _LoaderSpy(delay=0.002)
        violations, errors = [], []
        stop = threading.Event()

        def sampler():
            while not stop.is_set():
                st = scheduler.get_fleet_status()
                if (st.resident_memory_bytes + st.reserved_memory_bytes > st.max_resident_bytes
                        or st.resident_cores_count + len(st.loading_core_ids) > st.max_resident_cores):
                    violations.append(st.to_dict())

        def worker(seed):
            local = random.Random(seed)
            try:
                for _ in range(30):
                    with scheduler.lease_core(f"core_{local.randrange(6)}", timeout=10.0) as core:
                        self.assertIsNotNone(core)
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

        with patch.object(BaseNanoCore, "load_checkpoint", side_effect=spy):
            samp = threading.Thread(target=sampler, daemon=True)
            samp.start()
            self.addCleanup(stop.set)
            threads = [threading.Thread(target=worker, args=(s,), daemon=True) for s in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=60.0)
            stop.set()
            samp.join(timeout=5.0)

        self.assertEqual(errors, [])
        self.assertEqual(spy.dup_loads, [])
        self.assertEqual(violations, [])
        refused = [c["path"] for c in spy.calls if c["result"] == "CheckpointBudgetExceededError"]
        self.assertEqual(len(refused), len(set(refused)), "a grown image was refused twice")
        self.assertTrue(set(refused) <= grown)
        self.assertTrue(refused, "no grown image was ever loaded; the stress did not exercise X-F01")
        self.assertTrue(all(c["result"] in ("ok", "CheckpointBudgetExceededError") for c in spy.calls))
        status = self._assert_ledger_within_budget(scheduler)
        self.assertEqual(status.active_lease_count, 0)
        self.assertEqual(status.loading_core_ids, [])

    def test_22_xf03_reregistration_refused_while_leased(self):
        """X-F03: re-registering a leased core_id must raise LeaseConflictError
        and leave the descriptor, lease ledger and residency untouched, so a
        following evict_core cannot drop the leased core."""
        from gen_zero.nanocore.fleet_scheduler import FleetActiveLeaseError

        config = FleetSchedulerConfig(max_resident_cores=5, storage_dir=self.temp_dir)
        scheduler = NanoCoreFleetScheduler(config)
        core_a = self._sized_core("A", 1, 1024)
        scheduler.register_instance("core_a", core_a, persist=True)
        desc_before = scheduler.get_core_descriptor("core_a")
        gen_before = desc_before.generation
        lease = scheduler.acquire_lease("core_a")
        path_new = self._ckpt("core_a_v2", 9)

        with self.assertRaises(LeaseConflictError):
            scheduler.register_checkpoint("core_a", path_new)
        with self.assertRaises(LeaseConflictError):
            scheduler.register_instance("core_a", self._sized_core("A2", 9, 1024), persist=False)

        desc = scheduler.get_core_descriptor("core_a")
        self.assertEqual(desc.checkpoint_path, desc_before.checkpoint_path)
        self.assertEqual(desc.generation, gen_before)
        self.assertEqual(desc.active_leases, 1)
        self.assertFalse(scheduler.evict_core("core_a"))
        with self.assertRaises(FleetActiveLeaseError):
            scheduler.evict_core("core_a", force=True)
        self.assertTrue(scheduler.is_resident("core_a"))
        self.assertIs(scheduler.acquire_core("core_a"), core_a)

        # After release, re-registration succeeds, bumps the generation and
        # drops the stale resident instance of the old descriptor.
        scheduler.release_lease(lease)
        desc_new = scheduler.register_checkpoint("core_a", path_new)
        self.assertGreater(desc_new.generation, gen_before)
        self.assertEqual(desc_new.active_leases, 0)
        self.assertFalse(scheduler.is_resident("core_a"))
        with self.assertRaises(InvalidLeaseError):
            scheduler.release_lease(lease)
        lease_new = scheduler.acquire_lease("core_a")
        self.assertEqual(lease_new.generation, desc_new.generation)
        self.assertIsNot(lease_new.core, core_a)
        scheduler.release_lease(lease_new)

    def test_23_xf03_reregistration_refused_while_loading(self):
        """X-F03: a re-registration during an in-flight load would orphan the
        load's ticket; it must be refused and the load must still commit."""
        from unittest.mock import patch

        config = FleetSchedulerConfig(max_resident_cores=5, storage_dir=self.temp_dir)
        scheduler = NanoCoreFleetScheduler(config)
        path_b = self._ckpt("core_b", 2)
        scheduler.register_checkpoint("core_b", path_b)
        instance_b = self._sized_core("B", 2, 1024)
        started, proceed = threading.Event(), threading.Event()

        def fake_load(path, verify_checksum=True, max_bytes=None):
            started.set()
            proceed.wait(timeout=5.0)
            return instance_b

        result = {}
        with patch.object(BaseNanoCore, "load_checkpoint", side_effect=fake_load):
            t = threading.Thread(target=lambda: result.setdefault("core", scheduler.acquire_core("core_b")))
            t.start()
            self.assertTrue(started.wait(timeout=5.0))
            with self.assertRaises(LeaseConflictError):
                scheduler.register_checkpoint("core_b", path_b)
            proceed.set()
            t.join(timeout=5.0)
        self.assertIs(result["core"], instance_b)
        self.assertTrue(scheduler.is_resident("core_b"))

    def test_24_xf01_image_grown_past_whole_budget_fails_fast(self):
        """An image that grew past the entire budget can never fit: the loader
        refuses it before allocating, the scheduler fails at once with the
        reservation released and nothing resident."""
        from unittest.mock import patch

        MiB = 1024 * 1024
        config = FleetSchedulerConfig(max_resident_cores=5, max_resident_bytes=1 * MiB, storage_dir=self.temp_dir,
                                      acquire_timeout_seconds=30.0)
        scheduler = NanoCoreFleetScheduler(config)
        path_b = self._declared_ckpt("core_b", 2, 100 * 1024)
        scheduler.register_checkpoint("core_b", path_b)
        self._declared_ckpt("core_b", 2, 2 * MiB, path=path_b)
        spy = _LoaderSpy()
        t0 = time.monotonic()
        with patch.object(BaseNanoCore, "load_checkpoint", side_effect=spy):
            with self.assertRaises(FleetQuotaExceededError):
                scheduler.acquire_core("core_b")
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertEqual(spy.results(), ["CheckpointBudgetExceededError"])
        status = scheduler.get_fleet_status()
        self.assertEqual((status.reserved_memory_bytes, status.loading_core_ids), (0, []))
        self.assertFalse(scheduler.is_resident("core_b"))

    def test_25_xf02_waiter_timeout_on_in_flight_load_is_explicit(self):
        """A waiter whose deadline expires on another thread's in-flight load
        gets FleetAcquireTimeoutError; the loader still commits."""
        from gen_zero.nanocore.fleet_scheduler import FleetAcquireTimeoutError
        from unittest.mock import patch

        config = FleetSchedulerConfig(max_resident_cores=5, storage_dir=self.temp_dir)
        scheduler = NanoCoreFleetScheduler(config)
        scheduler.register_checkpoint("core_b", self._ckpt("core_b", 2))
        instance_b = self._sized_core("B", 2, 1024)
        started, proceed = threading.Event(), threading.Event()

        def fake_load(path, verify_checksum=True, max_bytes=None):
            started.set()
            proceed.wait(timeout=5.0)
            return instance_b

        result = {}
        with patch.object(BaseNanoCore, "load_checkpoint", side_effect=fake_load):
            t1 = threading.Thread(target=lambda: result.setdefault("core", scheduler.acquire_core("core_b")), daemon=True)
            t1.start()
            self.assertTrue(started.wait(timeout=5.0))
            with self.assertRaises(FleetAcquireTimeoutError):
                scheduler.acquire_lease("core_b", timeout=0.1)
            proceed.set()
            t1.join(timeout=5.0)
        self.assertIs(result["core"], instance_b)
        self.assertTrue(scheduler.is_resident("core_b"))
        self.assertEqual(scheduler.get_fleet_status().active_lease_count, 0)

    def test_26_xf03_spdk_reregistration_refused_while_leased(self):
        """X-F03 on the third overwrite path: register_spdk_core must refuse a
        leased core_id before touching the driver's device image."""
        from unittest.mock import patch
        from gen_zero.nanocore.spdk_nvme_fleet import SpdkStreamingFleetDriver, TransportBackend

        driver = SpdkStreamingFleetDriver(backend=TransportBackend.POSIX_SHM)
        try:
            config = FleetSchedulerConfig(max_resident_cores=2, enable_spdk=True, spdk_driver=driver)
            scheduler = NanoCoreFleetScheduler(config=config, spdk_driver=driver)
            core = DomainSpecialistNanoCore(domain="s", state_dim=32, candidate_dim=32, embed_dim=16, seed=1)
            desc = scheduler.register_spdk_core("core_s", core)
            lease = scheduler.acquire_lease("core_s")
            with patch.object(driver, "register_core", wraps=driver.register_core) as spy:
                with self.assertRaises(LeaseConflictError):
                    scheduler.register_spdk_core("core_s", core)
                spy.assert_not_called()
            current = scheduler.get_core_descriptor("core_s")
            self.assertEqual(current.generation, desc.generation)
            self.assertEqual(current.active_leases, 1)
            self.assertTrue(scheduler.is_resident("core_s"))
            scheduler.release_lease(lease)
            self.assertGreater(scheduler.register_spdk_core("core_s", core).generation, desc.generation)
        finally:
            driver.close()


    # ------------------------------------------------------------------
    # T3-F01 / T3-F02 / X-F01 fail-closed edges (third review)
    # ------------------------------------------------------------------

    def test_27_t3f01_cross_fleet_release_is_rejected(self):
        """T3-F01: two fleets both lease their first core "A" and hand out
        leases with identical (core_id, generation, token). Releasing fleet1's
        lease on fleet2 must raise and must not touch fleet2's lease."""
        import dataclasses
        from gen_zero.nanocore.fleet_scheduler import FleetActiveLeaseError

        fleets, leases = [], []
        for i in range(2):
            fleet = NanoCoreFleetScheduler(FleetSchedulerConfig(max_resident_cores=2, storage_dir=self.temp_dir))
            fleet.register_instance("A", self._sized_core(f"A{i}", i, 1024), persist=True,
                                    checkpoint_path=os.path.join(self.temp_dir, f"fleet{i}_A.zst"))
            fleets.append(fleet)
            leases.append(fleet.acquire_lease("A"))
        fleet1, fleet2 = fleets
        lease1, lease2 = leases
        self.assertNotEqual(fleet1.fleet_id, fleet2.fleet_id)
        self.assertEqual(
            (lease1.core_id, lease1.generation, lease1.token),
            (lease2.core_id, lease2.generation, lease2.token),
            "precondition: the review's colliding (core_id, generation, token)",
        )
        self.assertEqual((lease1.fleet_id, lease2.fleet_id), (fleet1.fleet_id, fleet2.fleet_id))

        with self.assertRaises(InvalidLeaseError):
            fleet2.release_lease(lease1)
        # A copy re-labelled with fleet2's id is still not the lease fleet2 issued.
        with self.assertRaises(InvalidLeaseError):
            fleet2.release_lease(dataclasses.replace(lease1, fleet_id=fleet2.fleet_id))

        self.assertEqual(fleet2.get_core_descriptor("A").active_leases, 1)
        with self.assertRaises(FleetActiveLeaseError):
            fleet2.evict_core("A", force=True)
        self.assertTrue(fleet2.is_resident("A"))

        fleet1.release_lease(lease1)
        fleet2.release_lease(lease2)
        self.assertEqual(fleet1.get_core_descriptor("A").active_leases, 0)
        self.assertEqual(fleet2.get_core_descriptor("A").active_leases, 0)

    def test_28_t3f02_tampered_descriptor_cannot_drive_eviction(self):
        """T3-F02: the review's probe set descriptor.active_leases = 0 and then
        force-evicted a leased core. Descriptors are now frozen snapshots; even
        a forced write into one changes nothing the scheduler decides on."""
        import dataclasses
        from gen_zero.nanocore.fleet_scheduler import FleetActiveLeaseError

        scheduler = NanoCoreFleetScheduler(FleetSchedulerConfig(max_resident_cores=2, storage_dir=self.temp_dir))
        registered = scheduler.register_instance("A", self._sized_core("A", 1, 1024), persist=True,
                                                 metadata={"owner": {"team": "t"}})
        lease = scheduler.acquire_lease("A")
        descriptor = scheduler.get_core_descriptor("A")
        self.assertEqual(descriptor.active_leases, 1)

        with self.assertRaises(dataclasses.FrozenInstanceError):
            descriptor.active_leases = 0
        for snap in (descriptor, registered):
            object.__setattr__(snap, "active_leases", 0)
            object.__setattr__(snap, "is_pinned", False)
            object.__setattr__(snap, "generation", -1)
            snap.metadata["owner"]["team"] = "attacker"

        with self.assertRaises(FleetActiveLeaseError):
            scheduler.evict_core("A", force=True)
        self.assertFalse(scheduler.evict_core("A"))
        scheduler.clear()
        self.assertTrue(scheduler.is_resident("A"), "tampered snapshot let a leased core be dropped")

        fresh = scheduler.get_core_descriptor("A")
        self.assertEqual(fresh.active_leases, 1)
        self.assertNotEqual(fresh.generation, -1)
        self.assertEqual(fresh.metadata, {"owner": {"team": "t"}})
        self.assertIsNot(fresh, descriptor)

        scheduler.release_lease(lease)
        self.assertTrue(scheduler.evict_core("A", force=True))

    def test_29_xf01_lying_header_fails_closed_without_retry(self):
        """A checkpoint whose header under-declares its resident size is corrupt.
        The loader detects it once the core is built and refuses it; the
        scheduler must not admit it and must not retry."""
        from unittest.mock import patch

        scheduler = NanoCoreFleetScheduler(FleetSchedulerConfig(max_resident_cores=5, storage_dir=self.temp_dir))
        path = self._declared_ckpt("liar", 3, 16)  # real footprint ~29KB
        scheduler.register_checkpoint("liar", path)
        spy = _LoaderSpy()
        with patch.object(BaseNanoCore, "load_checkpoint", side_effect=spy):
            with self.assertRaises(CheckpointBoundViolationError):
                scheduler.acquire_core("liar")
        self.assertEqual(spy.results(), ["CheckpointBoundViolationError"])
        status = scheduler.get_fleet_status()
        self.assertEqual((status.reserved_memory_bytes, status.loading_core_ids, status.resident_core_ids), (0, [], []))

    def test_30_xf01_scheduler_refuses_loader_that_ignores_its_budget(self):
        """Defence in depth: a loader that ignores max_bytes and returns a core
        larger than the reservation is refused at commit, never admitted."""
        from unittest.mock import patch

        MiB = 1024 * 1024
        scheduler = NanoCoreFleetScheduler(FleetSchedulerConfig(max_resident_cores=5, storage_dir=self.temp_dir))
        scheduler.register_checkpoint("core_b", self._ckpt("core_b", 2))
        seen = []

        def rogue_load(path, verify_checksum=True, max_bytes=None):
            seen.append(max_bytes)
            return self._sized_core("B", 2, 40 * MiB)

        with patch.object(BaseNanoCore, "load_checkpoint", side_effect=rogue_load):
            with self.assertRaises(CheckpointBoundViolationError):
                scheduler.acquire_core("core_b")
        self.assertEqual(len(seen), 1)
        self.assertIsNotNone(seen[0], "the scheduler called a loader without a budget")
        status = scheduler.get_fleet_status()
        self.assertEqual((status.reserved_memory_bytes, status.resident_core_ids), (0, []))

    def test_31_xf01_unbounded_images_are_refused_explicitly(self):
        """No trusted bound, no load: headerless files, unpersisted cores and
        raw SPDK dicts are refused with an explicit error, not a guess."""
        import pickle

        scheduler = NanoCoreFleetScheduler(FleetSchedulerConfig(max_resident_cores=5, storage_dir=self.temp_dir))
        legacy = os.path.join(self.temp_dir, "legacy.pkl")
        with open(legacy, "wb") as f:
            f.write(pickle.dumps({"weights": {}, "metadata": {}}))
        with self.assertRaises(CheckpointFormatError):
            scheduler.register_checkpoint("legacy", legacy)
        with self.assertRaises(CheckpointFormatError):
            BaseNanoCore.load_checkpoint(legacy, max_bytes=10 * 1024 * 1024)

        scheduler.register_instance("volatile", self._sized_core("V", 4, 1024), persist=False)
        self.assertTrue(scheduler.evict_core("volatile"))
        with self.assertRaises(CheckpointFormatError):
            scheduler.acquire_core("volatile")
        self.assertEqual(scheduler.get_fleet_status().reserved_memory_bytes, 0)

        with self.assertRaises(TypeError):
            scheduler.register_spdk_core("raw", {"weights": {}})

    def test_32_xf01_real_loader_allocates_nothing_when_refused(self):
        """The enforcement point itself, no mocks: a 32MiB image loaded under a
        1MiB budget raises before the body is read (traced peak stays tiny);
        under its declared peak it loads and stays within that peak."""
        import tracemalloc

        MiB = 1024 * 1024
        path = self._ballast_ckpt("big", 5, 32 * MiB)
        tracemalloc.start()
        self.addCleanup(tracemalloc.stop)
        tracemalloc.reset_peak()
        with self.assertRaises(CheckpointBudgetExceededError) as ctx:
            BaseNanoCore.load_checkpoint(path, max_bytes=1 * MiB)
        _, peak = tracemalloc.get_traced_memory()
        self.assertLess(peak, 256 * 1024, f"refused load allocated {peak} bytes")
        bound = ctx.exception.bound

        tracemalloc.reset_peak()
        base, _ = tracemalloc.get_traced_memory()
        core = BaseNanoCore.load_checkpoint(path, max_bytes=bound.load_peak_bytes)
        _, peak = tracemalloc.get_traced_memory()
        self.assertGreater(peak - base, 32 * MiB)
        self.assertLessEqual(peak - base, bound.load_peak_bytes)
        self.assertLessEqual(core.memory_footprint_bytes(), bound.resident_bytes)

    def test_33_xf01_spdk_stream_refuses_before_deserializing(self):
        """SPDK path: the manifest's declared peak is checked against the
        reservation before any chunk is reassembled."""
        from unittest.mock import patch
        from gen_zero.nanocore.spdk_nvme_fleet import (
            SpdkNanoCoreSerializer, SpdkStreamingFleetDriver, TransportBackend,
        )

        driver = SpdkStreamingFleetDriver(backend=TransportBackend.POSIX_SHM)
        try:
            core = DomainSpecialistNanoCore(domain="s", state_dim=32, candidate_dim=32, embed_dim=16, seed=1)
            driver.register_core("core_s", core)
            bound = driver.load_bound("core_s")
            self.assertEqual(bound.resident_bytes, core.memory_footprint_bytes())
            with patch.object(SpdkNanoCoreSerializer, "deserialize_chunks_to_core") as deser:
                with self.assertRaises(CheckpointBudgetExceededError):
                    driver.stream_in_core("core_s", max_bytes=bound.load_peak_bytes - 1)
                deser.assert_not_called()
            loaded = driver.stream_in_core("core_s", max_bytes=bound.load_peak_bytes)
            self.assertEqual(loaded.memory_footprint_bytes(), core.memory_footprint_bytes())

            driver.register_core("raw", {"weights": {"w": np.ones(4, dtype=np.float32)}})
            with self.assertRaises(CheckpointFormatError):
                driver.stream_in_core("raw", max_bytes=10 * 1024 * 1024)
        finally:
            driver.close()


if __name__ == "__main__":
    unittest.main()
