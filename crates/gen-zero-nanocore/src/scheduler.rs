//! gen-zero-nanocore Fleet Scheduler with bounded RAM budget and LRU eviction.

#![allow(deprecated)]

use crate::core_type::{DomainId, NanoCoreInstance};
use crate::error::NanoCoreError;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

pub const DEFAULT_RAM_BUDGET_BYTES: usize = 450 * 1024 * 1024; // 450 MB

struct HotCoreEntry {
    core: Arc<NanoCoreInstance>,
    last_accessed: AtomicU64,
}

struct ColdCoreEntry {
    compressed_data: Vec<u8>,
}

/// NanoCore Fleet Scheduler enforcing a budget for cached core representations.
/// Outstanding `Arc` handles and temporary codec allocations are not included.
/// All map mutations acquire hot_cores before cold_cores; never reverse this order.
pub struct NanoCoreFleetScheduler {
    ram_budget_bytes: usize,
    current_ram_bytes: AtomicUsize,
    clock: AtomicU64,
    hot_cores: RwLock<HashMap<DomainId, HotCoreEntry>>,
    cold_cores: RwLock<HashMap<DomainId, ColdCoreEntry>>,
}

impl NanoCoreFleetScheduler {
    pub fn new(ram_budget_bytes: usize) -> Self {
        Self {
            ram_budget_bytes,
            current_ram_bytes: AtomicUsize::new(0),
            clock: AtomicU64::new(1),
            hot_cores: RwLock::new(HashMap::new()),
            cold_cores: RwLock::new(HashMap::new()),
        }
    }

    /// Register a new core into the fleet. If memory budget allows, loads hot; otherwise cold.
    pub fn register_core(&self, core: NanoCoreInstance) -> Result<(), NanoCoreError> {
        core.validate()?;
        let domain_id = core.domain_id;
        let size = core.size_in_bytes();

        if size > self.ram_budget_bytes {
            return Err(NanoCoreError::RamBudgetExceeded {
                current_bytes: self.current_ram_bytes(),
                limit_bytes: self.ram_budget_bytes,
            });
        }

        let mut hot_map = self.hot_cores.write();
        let mut cold_map = self.cold_cores.write();

        // If core already exists, release its previous representation before replacing it.
        let old_hot_entry = hot_map.remove(&domain_id);
        if let Some(old_entry) = old_hot_entry.as_ref() {
            let old_size = old_entry.core.size_in_bytes();
            let _ =
                self.current_ram_bytes
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |curr| {
                        Some(curr.saturating_sub(old_size))
                    });
        }
        let old_cold_entry = cold_map.remove(&domain_id);
        if let Some(old_entry) = old_cold_entry.as_ref() {
            let old_size = old_entry.compressed_data.len();
            let _ =
                self.current_ram_bytes
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |curr| {
                        Some(curr.saturating_sub(old_size))
                    });
        }

        // Evict LRU cores until fits
        while self
            .current_ram_bytes
            .load(Ordering::SeqCst)
            .saturating_add(size)
            > self.ram_budget_bytes
            && !hot_map.is_empty()
        {
            if let Err(error) = self.evict_lru_internal(&mut hot_map, &mut cold_map) {
                self.restore_core_entry(
                    domain_id,
                    old_hot_entry,
                    old_cold_entry,
                    &mut hot_map,
                    &mut cold_map,
                );
                return Err(error);
            }
        }

        if self
            .current_ram_bytes
            .load(Ordering::SeqCst)
            .saturating_add(size)
            > self.ram_budget_bytes
        {
            self.restore_core_entry(
                domain_id,
                old_hot_entry,
                old_cold_entry,
                &mut hot_map,
                &mut cold_map,
            );
            return Err(NanoCoreError::RamBudgetExceeded {
                current_bytes: self.current_ram_bytes(),
                limit_bytes: self.ram_budget_bytes,
            });
        }

        let tick = self.next_tick();
        hot_map.insert(
            domain_id,
            HotCoreEntry {
                core: Arc::new(core),
                last_accessed: AtomicU64::new(tick),
            },
        );
        self.current_ram_bytes.fetch_add(size, Ordering::SeqCst);

        Ok(())
    }

    /// Retrieve core for inference.
    /// Hot hit: direct Arc clone. Cold miss: decompress and promote under the map locks.
    pub fn get_core(&self, domain_id: DomainId) -> Result<Arc<NanoCoreInstance>, NanoCoreError> {
        // Fast path: check hot cores with read lock
        {
            let hot_map = self.hot_cores.read();
            if let Some(entry) = hot_map.get(&domain_id) {
                let tick = self.next_tick();
                entry.last_accessed.fetch_max(tick, Ordering::Relaxed);
                return Ok(Arc::clone(&entry.core));
            }
        }

        // Serialize the cold lookup and promotion with registration/eviction.
        // A detached snapshot can resurrect an obsolete core after replacement;
        // separate hot/cold lookups can also miss a concurrent promotion entirely.
        let mut hot_map = self.hot_cores.write();
        let mut cold_map = self.cold_cores.write();

        // Double check hot_map in case another thread promoted it
        if let Some(entry) = hot_map.get(&domain_id) {
            let tick = self.next_tick();
            entry.last_accessed.fetch_max(tick, Ordering::Relaxed);
            return Ok(Arc::clone(&entry.core));
        }

        let cold_entry = cold_map
            .get(&domain_id)
            .ok_or(NanoCoreError::CoreNotFound(domain_id.0))?;
        // Keep the entry and accounting intact if decompression fails.
        let core = NanoCoreInstance::decompress_zstd(&cold_entry.compressed_data)?;
        let size = core.size_in_bytes();

        // Remove the compressed representation from the budget before promoting it.
        let cold_entry = cold_map
            .remove(&domain_id)
            .ok_or(NanoCoreError::CoreNotFound(domain_id.0))?;
        let compressed_size = cold_entry.compressed_data.len();
        let _ = self
            .current_ram_bytes
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |curr| {
                Some(curr.saturating_sub(compressed_size))
            });

        // Evict until fits
        while self
            .current_ram_bytes
            .load(Ordering::SeqCst)
            .saturating_add(size)
            > self.ram_budget_bytes
            && !hot_map.is_empty()
        {
            if let Err(error) = self.evict_lru_internal(&mut hot_map, &mut cold_map) {
                cold_map.insert(domain_id, cold_entry);
                self.current_ram_bytes
                    .fetch_add(compressed_size, Ordering::SeqCst);
                return Err(error);
            }
        }

        if self
            .current_ram_bytes
            .load(Ordering::SeqCst)
            .saturating_add(size)
            > self.ram_budget_bytes
        {
            // Keep the cold core available when its uncompressed representation cannot fit.
            let compressed_data = cold_entry.compressed_data;
            cold_map.insert(domain_id, ColdCoreEntry { compressed_data });
            self.current_ram_bytes
                .fetch_add(compressed_size, Ordering::SeqCst);
            return Err(NanoCoreError::RamBudgetExceeded {
                current_bytes: self.current_ram_bytes(),
                limit_bytes: self.ram_budget_bytes,
            });
        }

        let tick = self.next_tick();
        let arc_core = Arc::new(core);
        hot_map.insert(
            domain_id,
            HotCoreEntry {
                core: Arc::clone(&arc_core),
                last_accessed: AtomicU64::new(tick),
            },
        );
        self.current_ram_bytes.fetch_add(size, Ordering::SeqCst);

        Ok(arc_core)
    }

    fn next_tick(&self) -> u64 {
        // Saturate rather than wrap and make newly accessed entries look oldest.
        // At exhaustion, tied timestamps still permit eviction and forward progress.
        self.clock
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |tick| {
                Some(tick.saturating_add(1))
            })
            .expect("clock update always succeeds")
    }

    fn restore_core_entry(
        &self,
        domain_id: DomainId,
        hot_entry: Option<HotCoreEntry>,
        cold_entry: Option<ColdCoreEntry>,
        hot_map: &mut HashMap<DomainId, HotCoreEntry>,
        cold_map: &mut HashMap<DomainId, ColdCoreEntry>,
    ) {
        if let Some(entry) = hot_entry {
            let size = entry.core.size_in_bytes();
            hot_map.insert(domain_id, entry);
            self.current_ram_bytes.fetch_add(size, Ordering::SeqCst);
        } else if let Some(entry) = cold_entry {
            let size = entry.compressed_data.len();
            cold_map.insert(domain_id, entry);
            self.current_ram_bytes.fetch_add(size, Ordering::SeqCst);
        }
    }

    /// Internal LRU eviction: finds oldest core, compresses it into cold_cores, and frees RAM.
    fn evict_lru_internal(
        &self,
        hot_map: &mut HashMap<DomainId, HotCoreEntry>,
        cold_map: &mut HashMap<DomainId, ColdCoreEntry>,
    ) -> Result<(), NanoCoreError> {
        if hot_map.is_empty() {
            return Ok(());
        }

        let mut oldest_domain = None;
        let mut min_tick = u64::MAX;

        for (id, entry) in hot_map.iter() {
            let t = entry.last_accessed.load(Ordering::Relaxed);
            if oldest_domain.is_none() || t < min_tick {
                min_tick = t;
                oldest_domain = Some(*id);
            }
        }

        if let Some(target_id) = oldest_domain {
            if let Some(entry) = hot_map.get(&target_id) {
                let size = entry.core.size_in_bytes();
                // Compress first before removing to avoid data loss on compression failure
                let compressed = entry.core.compress_zstd()?;
                let compressed_size = compressed.len();
                let current = self.current_ram_bytes.load(Ordering::SeqCst);
                let next = current.saturating_sub(size).saturating_add(compressed_size);
                // Eviction must free space, including when a failed admission restores
                // an entry that was temporarily removed from the accounting.
                if compressed_size >= size || next > self.ram_budget_bytes {
                    return Err(NanoCoreError::RamBudgetExceeded {
                        current_bytes: current,
                        limit_bytes: self.ram_budget_bytes,
                    });
                }
                hot_map.remove(&target_id);
                cold_map.insert(
                    target_id,
                    ColdCoreEntry {
                        compressed_data: compressed,
                    },
                );

                let _ = self.current_ram_bytes.fetch_update(
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                    |curr| Some(curr.saturating_sub(size).saturating_add(compressed_size)),
                );
            }
        }

        Ok(())
    }

    #[inline]
    pub fn current_ram_bytes(&self) -> usize {
        self.current_ram_bytes.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn ram_budget_bytes(&self) -> usize {
        self.ram_budget_bytes
    }

    #[inline]
    pub fn hot_core_count(&self) -> usize {
        self.hot_cores.read().len()
    }

    #[inline]
    pub fn cold_core_count(&self) -> usize {
        self.cold_cores.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_type::fixtures::synthetic_core;
    use crate::core_type::*;
    use gen_zero_core::CompressedLatent;

    #[test]
    fn saturated_clock_still_evicts() {
        let core = synthetic_core(
            DOMAIN_VISION,
            "Core",
            CompressedLatent::zeros(),
            16,
            0.9,
            &["a", "b"],
        );
        let mut other = core.clone();
        other.domain_id = DOMAIN_BROWSER;
        let scheduler = NanoCoreFleetScheduler::new(
            core.size_in_bytes() + core.compress_zstd().unwrap().len() + 64,
        );
        scheduler.clock.store(u64::MAX, Ordering::Relaxed);
        scheduler.register_core(core).unwrap();
        scheduler.register_core(other).unwrap();
        assert_eq!(scheduler.clock.load(Ordering::Relaxed), u64::MAX);
        assert_eq!(scheduler.hot_core_count(), 1);
        assert_eq!(scheduler.cold_core_count(), 1);
    }

    #[test]
    fn concurrent_promotions_and_replacements_preserve_registry() {
        use std::sync::Barrier;
        use std::thread;

        let core = synthetic_core(
            DOMAIN_VISION,
            "Core",
            CompressedLatent::zeros(),
            16,
            0.9,
            &["a", "b"],
        );
        let mut other = core.clone();
        other.domain_id = DOMAIN_BROWSER;
        let scheduler = NanoCoreFleetScheduler::new(core.size_in_bytes() * 2 - 1);
        scheduler.register_core(core.clone()).unwrap();
        scheduler.register_core(other).unwrap();
        let barrier = Barrier::new(4);
        thread::scope(|scope| {
            for worker in 0..4 {
                let scheduler = &scheduler;
                let barrier = &barrier;
                let core = &core;
                scope.spawn(move || {
                    barrier.wait();
                    for generation in 0..100 {
                        if worker == 0 {
                            let mut replacement = core.clone();
                            replacement.base_confidence = generation as f32 / 100.0;
                            scheduler.register_core(replacement).unwrap();
                        } else {
                            for domain in [DOMAIN_VISION, DOMAIN_BROWSER] {
                                assert_eq!(scheduler.get_core(domain).unwrap().domain_id, domain);
                            }
                        }
                    }
                });
            }
        });
        // An old detached decompression must never overwrite the last registration.
        assert_eq!(
            scheduler.get_core(DOMAIN_VISION).unwrap().base_confidence,
            0.99
        );
        let hot = scheduler.hot_cores.read();
        let cold = scheduler.cold_cores.read();
        assert_eq!(hot.len() + cold.len(), 2);
        assert!(hot.keys().all(|domain| !cold.contains_key(domain)));
        let actual: usize = hot
            .values()
            .map(|entry| entry.core.size_in_bytes())
            .sum::<usize>()
            + cold
                .values()
                .map(|entry| entry.compressed_data.len())
                .sum::<usize>();
        assert_eq!(scheduler.current_ram_bytes(), actual);
        assert!(actual <= scheduler.ram_budget_bytes());
    }

    #[test]
    fn test_fleet_scheduler_lru_eviction() {
        let proto = CompressedLatent::zeros();
        let core1 = synthetic_core(
            DOMAIN_VISION,
            "VisionCore",
            proto.clone(),
            16,
            0.9,
            &["a", "b"],
        );
        let core2 = synthetic_core(
            DOMAIN_BROWSER,
            "BrowserCore",
            proto.clone(),
            16,
            0.9,
            &["a", "b"],
        );
        let core3 = synthetic_core(DOMAIN_SQL, "SqlCore", proto, 16, 0.9, &["a", "b"]);
        let budget =
            core2.size_in_bytes() + core3.size_in_bytes() + core1.compress_zstd().unwrap().len();
        let scheduler = NanoCoreFleetScheduler::new(budget);

        scheduler.register_core(core1).unwrap();
        scheduler.register_core(core2).unwrap();
        assert_eq!(scheduler.hot_core_count(), 2);
        assert_eq!(scheduler.cold_core_count(), 0);

        // Registering core3 must trigger eviction of core1 into cold storage!
        scheduler.register_core(core3).unwrap();
        assert!(scheduler.cold_core_count() >= 1);
        assert!(scheduler.current_ram_bytes() <= budget);

        // Reloading the cold core1 must succeed and promote it back to hot
        let retrieved = scheduler.get_core(DOMAIN_VISION).unwrap();
        assert_eq!(retrieved.domain_id, DOMAIN_VISION);
    }

    #[test]
    fn test_compressed_cold_core_counts_toward_ram_budget() {
        let proto = CompressedLatent::zeros();
        let core1 = synthetic_core(
            DOMAIN_VISION,
            "CoreOne",
            proto.clone(),
            16,
            0.9,
            &["a", "b"],
        );
        let core2 = synthetic_core(DOMAIN_BROWSER, "CoreTwo", proto, 16, 0.9, &["a", "b"]);
        let core1_size = core1.size_in_bytes();
        let core2_size = core2.size_in_bytes();
        let core1_compressed_size = core1.compress_zstd().unwrap().len();
        let core2_compressed_size = core2.compress_zstd().unwrap().len();

        // Leave room for one hot core and one compressed cold core, but not two hot cores.
        let budget = (core1_size + core2_size - 1)
            .max(core1_size + core2_compressed_size)
            .max(core2_size + core1_compressed_size);
        assert!(budget < core1_size + core2_size);

        let scheduler = NanoCoreFleetScheduler::new(budget);
        scheduler.register_core(core1).unwrap();
        scheduler.register_core(core2).unwrap();

        assert_eq!(scheduler.hot_core_count(), 1);
        assert_eq!(scheduler.cold_core_count(), 1);
        assert_eq!(
            scheduler.current_ram_bytes(),
            core2_size + core1_compressed_size
        );

        scheduler.get_core(DOMAIN_VISION).unwrap();
        assert_eq!(scheduler.hot_core_count(), 1);
        assert_eq!(scheduler.cold_core_count(), 1);
        assert_eq!(
            scheduler.current_ram_bytes(),
            core1_size + core2_compressed_size
        );
        assert!(scheduler.current_ram_bytes() <= budget);
    }

    #[test]
    fn test_budget_rejection_keeps_cold_memory_accounted() {
        let proto = CompressedLatent::zeros();
        let core1 = synthetic_core(
            DOMAIN_VISION,
            "CoreOne",
            proto.clone(),
            16,
            0.9,
            &["a", "b"],
        );
        let core2 = synthetic_core(
            DOMAIN_BROWSER,
            "CoreTwo",
            proto.clone(),
            16,
            0.9,
            &["a", "b"],
        );
        let core3 = synthetic_core(DOMAIN_SQL, "CoreThree", proto, 16, 0.9, &["a", "b"]);
        let core1_compressed_size = core1.compress_zstd().unwrap().len();
        let core2_compressed_size = core2.compress_zstd().unwrap().len();
        let budget = core2.size_in_bytes() + core1_compressed_size;

        let scheduler = NanoCoreFleetScheduler::new(budget);
        scheduler.register_core(core1).unwrap();
        scheduler.register_core(core2).unwrap();

        let error = scheduler.register_core(core3).unwrap_err();
        assert!(matches!(error, NanoCoreError::RamBudgetExceeded { .. }));
        assert_eq!(scheduler.hot_core_count(), 0);
        assert_eq!(scheduler.cold_core_count(), 2);
        assert_eq!(
            scheduler.current_ram_bytes(),
            core1_compressed_size + core2_compressed_size
        );
        assert!(scheduler.current_ram_bytes() <= budget);
    }
}
