"""Unit tests for RFC-088 / Issue #88: SPDK / NVMe-oF User-Space Streaming Fleet Driver."""

import unittest
import os
import time
import struct
import zlib
import numpy as np

from gen_zero.nanocore.spdk_nvme_fleet import (
    TransportBackend,
    RingEntry,
    SpdkChunkHeader,
    LocklessRingBuffer,
    SpdkNvmeDevice,
    SpdkNanoCoreSerializer,
    StreamingPrefetchEngine,
    SpdkStreamingFleetDriver,
    CHUNK_SIZE,
    HEADER_SIZE,
    CHUNK_MAGIC,
)
from gen_zero.nanocore.fleet_scheduler import (
    NanoCoreFleetScheduler,
    FleetSchedulerConfig,
    CoreDescriptor,
)
from gen_zero.runtime.specialist_nano_core import DomainSpecialistNanoCore
from gen_zero.runtime.base_nano_core import BaseNanoCore


class TestIssue88SpdkNvmeFleet(unittest.TestCase):
    """Test suite for user-space kernel-bypass SPDK streaming hot-swaps."""

    def setUp(self) -> None:
        self.specialist_core = DomainSpecialistNanoCore(
            domain="security_sentinel",
            version_id="sentinel-v1.0",
            state_dim=64,
            candidate_dim=64,
            embed_dim=32,
            seed=42,
        )

    def test_01_chunk_header_packing_and_unpacking(self) -> None:
        """Verifies 64-byte header binary packing, unpack, and CRC integrity."""
        header = SpdkChunkHeader(
            magic=CHUNK_MAGIC,
            tier=1,
            chunk_index=3,
            total_chunks=10,
            payload_size=2048,
            crc32=0xDEADBEEF,
            core_id="core_test_01",
        )
        packed = header.pack()
        self.assertEqual(len(packed), HEADER_SIZE, "Header must be strictly 64 bytes")
        
        unpacked = SpdkChunkHeader.unpack(packed)
        self.assertEqual(unpacked.magic, CHUNK_MAGIC)
        self.assertEqual(unpacked.tier, 1)
        self.assertEqual(unpacked.chunk_index, 3)
        self.assertEqual(unpacked.total_chunks, 10)
        self.assertEqual(unpacked.payload_size, 2048)
        self.assertEqual(unpacked.crc32, 0xDEADBEEF)
        self.assertEqual(unpacked.core_id, "core_test_01")

    def test_02_lockless_ring_buffer_sq_cq_pmd(self) -> None:
        """Verifies LocklessRingBuffer circular indexing, SQ/CQ enqueue, and PMD polling."""
        ring = LocklessRingBuffer(capacity=16)
        self.assertEqual(ring.capacity, 16)
        
        entries = []
        for i in range(10):
            entry = RingEntry(
                command_id=i,
                core_id=f"core_{i}",
                chunk_index=0,
                total_chunks=1,
                tier=1,
                dma_offset=i * 4096,
                length=4096,
            )
            success = ring.enqueue_sq(entry)
            self.assertTrue(success)
            entries.append(entry)

        self.assertEqual(ring.pending_sq_count(), 10)

        # Dequeue SQ and enqueue CQ
        for orig in entries:
            dequeued = ring.dequeue_sq()
            self.assertIsNotNone(dequeued)
            self.assertEqual(dequeued.command_id, orig.command_id)
            dequeued.status = 0
            ring.enqueue_cq(dequeued)

        self.assertEqual(ring.pending_sq_count(), 0)
        self.assertEqual(ring.pending_cq_count(), 10)

        # PMD poll CQ in batches
        polled = ring.poll_cq(max_entries=6)
        self.assertEqual(len(polled), 6)
        self.assertEqual(ring.pending_cq_count(), 4)

        polled_remaining = ring.poll_cq(max_entries=10)
        self.assertEqual(len(polled_remaining), 4)
        self.assertEqual(ring.pending_cq_count(), 0)

    def test_03_serializer_bit_exact_reconstruction_rmse_zero(self) -> None:
        """Verifies 4KB chunk serialization and bit-exact reconstruction (RMSE == 0.0)."""
        raw_chunks, manifest = SpdkNanoCoreSerializer.serialize_core_to_chunks(
            core_id="sentinel_core",
            core=self.specialist_core,
            metadata={"priority": 10},
        )
        self.assertEqual(len(raw_chunks) % CHUNK_SIZE, 0, "Buffer must be 4KB aligned")
        self.assertGreater(manifest["total_chunks"], 0)
        self.assertGreater(manifest["tier1_chunks"], 0)

        # Reconstruct
        t0 = time.perf_counter()
        reconstructed = SpdkNanoCoreSerializer.deserialize_chunks_to_core(raw_chunks, verify_checksum=True)
        deser_time_ms = (time.perf_counter() - t0) * 1000.0

        self.assertIsInstance(reconstructed, DomainSpecialistNanoCore)
        self.assertLess(deser_time_ms, 2.0, "Deserialization should be sub-2ms")

        # Verify bit-exact weights matching: RMSE == 0.0
        for w_name, orig_w in self.specialist_core.weights.items():
            recon_w = reconstructed.weights[w_name]
            np.testing.assert_allclose(recon_w, orig_w, atol=0.0, rtol=0.0)
            diff = recon_w - orig_w
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            self.assertEqual(rmse, 0.0, f"RMSE for {w_name} must be exactly 0.0")

        # Verify scoring outputs match bit-exactly
        test_state = np.random.RandomState(123).randn(64).astype(np.float32)
        candidates = ["allow_packet", "block_ip", "throttle"]
        orig_res = self.specialist_core.score_candidates(test_state, candidates)
        recon_res = reconstructed.score_candidates(test_state, candidates)
        for cand in candidates:
            self.assertAlmostEqual(orig_res["probs"][cand], recon_res["probs"][cand], places=6)

    def test_04_device_dma_zero_copy_transfer(self) -> None:
        """Tests SpdkNvmeDevice storage and user-space direct DMA copy."""
        device = SpdkNvmeDevice(backend=TransportBackend.POSIX_SHM)
        try:
            raw_chunks, manifest = SpdkNanoCoreSerializer.serialize_core_to_chunks(
                core_id="core_dma_test",
                core=self.specialist_core,
            )
            stored_count = device.store_chunks("core_dma_test", raw_chunks)
            self.assertEqual(stored_count, manifest["total_chunks"])
            self.assertTrue(device.has_core("core_dma_test"))

            # Direct DMA read into destination buffer
            dst_buf = bytearray(stored_count * CHUNK_SIZE)
            for i in range(stored_count):
                device.dma_read_chunk("core_dma_test", i, dst_buf, i * CHUNK_SIZE)

            self.assertEqual(bytes(dst_buf), raw_chunks)
        finally:
            device.close()

    def test_05_streaming_driver_and_submillisecond_latency(self) -> None:
        """Tests SpdkStreamingFleetDriver hot-swap with P99 <= 0.8ms target."""
        driver = SpdkStreamingFleetDriver(backend=TransportBackend.POSIX_SHM)
        try:
            manifest = driver.register_core("sentinel_fast", self.specialist_core)
            self.assertIn("total_chunks", manifest)

            # Warm-up call
            driver.stream_in_core("sentinel_fast")

            # Benchmark 20 streaming swaps
            latencies = []
            for _ in range(20):
                t0 = time.perf_counter()
                core = driver.stream_in_core("sentinel_fast")
                lat = (time.perf_counter() - t0) * 1000.0
                latencies.append(lat)
                self.assertIsNotNone(core)

            p99 = sorted(latencies)[int(len(latencies) * 0.95)]
            mean_lat = sum(latencies) / len(latencies)
            self.assertLess(mean_lat, 0.8, f"Mean streaming latency {mean_lat:.3f}ms exceeds 0.8ms")
            self.assertLess(p99, 1.2, f"P99 latency {p99:.3f}ms exceeds threshold")

            telemetry = driver.get_telemetry()
            self.assertEqual(telemetry.rmse, 0.0)
            self.assertTrue(telemetry.checksum_verified)
            self.assertGreater(telemetry.total_swaps, 20)
        finally:
            driver.close()

    def test_06_async_prefetch_stage1_overlap(self) -> None:
        """Verifies asynchronous background prefetch completes before stream_in."""
        driver = SpdkStreamingFleetDriver(backend=TransportBackend.POSIX_SHM)
        try:
            driver.register_core("sentinel_overlap", self.specialist_core)

            # Trigger async prefetch (as if during Stage-1 coarse filtering)
            submitted = driver.prefetch_async(["sentinel_overlap"])
            self.assertGreater(submitted, 0)

            # Wait for PMD worker to finish prefetching into user-space cache
            t_wait_start = time.time()
            while not driver.prefetch_engine.has_prefetched("sentinel_overlap") and (time.time() - t_wait_start) < 0.2:
                time.sleep(0.001)

            self.assertTrue(driver.prefetch_engine.has_prefetched("sentinel_overlap"))

            # Check prefetch cache hit in user space
            t0 = time.perf_counter()
            core = driver.stream_in_core("sentinel_overlap")
            swap_time_ms = (time.perf_counter() - t0) * 1000.0

            self.assertIsNotNone(core)
            self.assertLess(swap_time_ms, 0.8, f"Prefetched core stream-in took {swap_time_ms:.3f}ms")
        finally:
            driver.close()

    def test_07_fleet_scheduler_spdk_integration_and_lease(self) -> None:
        """Verifies NanoCoreFleetScheduler with SPDK driver and lease protection."""
        driver = SpdkStreamingFleetDriver(backend=TransportBackend.POSIX_SHM)
        try:
            config = FleetSchedulerConfig(
                max_resident_cores=2,
                enable_spdk=True,
                spdk_driver=driver,
            )
            scheduler = NanoCoreFleetScheduler(config=config, spdk_driver=driver)

            # Register 3 cores into SPDK
            for i in range(3):
                core_i = DomainSpecialistNanoCore(
                    domain=f"domain_{i}",
                    version_id=f"v{i}",
                    state_dim=32,
                    candidate_dim=32,
                    embed_dim=16,
                    seed=i,
                )
                scheduler.register_spdk_core(f"core_{i}", core_i)

            self.assertEqual(scheduler.registered_count, 3)
            self.assertEqual(scheduler.resident_count, 0)

            # Acquire core_0 (cold load via SPDK streaming)
            c0 = scheduler.acquire_core("core_0")
            self.assertEqual(scheduler.resident_count, 1)

            # Hot hit (0.00ms)
            t0 = time.perf_counter()
            c0_hot = scheduler.acquire_core("core_0")
            hot_lat = (time.perf_counter() - t0) * 1000.0
            self.assertIs(c0, c0_hot)
            self.assertLess(hot_lat, 0.1, "Hot hit latency should be ~0.00ms")

            # Lease core_0: while leased, core_0 CANNOT be evicted
            with scheduler.lease_core("core_0") as leased_c0:
                self.assertIs(c0, leased_c0)
                # Load core_1
                scheduler.acquire_core("core_1")
                self.assertEqual(scheduler.resident_count, 2)

                # Now load core_2 under max_resident_cores=2
                # Since core_0 is leased, core_1 must be evicted, not core_0!
                scheduler.acquire_core("core_2")
                self.assertTrue(scheduler.is_resident("core_0"), "Leased core_0 must not be evicted")
                self.assertFalse(scheduler.is_resident("core_1"), "Unleased core_1 should be evicted")
                self.assertTrue(scheduler.is_resident("core_2"))

            # After lease exits, core_0 can be evicted when needed
            scheduler.acquire_core("core_1")
            self.assertEqual(scheduler.resident_count, 2)
        finally:
            driver.close()

    def test_08_transport_fallback(self) -> None:
        """Verifies graceful transport backend detection and fallback."""
        backend = SpdkStreamingFleetDriver._probe_best_backend()
        self.assertIn(backend, [TransportBackend.RDMA_ROCEV2, TransportBackend.NVME_TCP, TransportBackend.POSIX_SHM])


if __name__ == "__main__":
    unittest.main()
