//! gen-zero-core thread-safe action symbol interner.

use crate::error::InternerError;
use crate::types::ActionId;
use indexmap::IndexSet;
use parking_lot::RwLock;
use std::sync::Arc;

/// Thread-safe canonical action symbol registry.
/// Maps string action names to dense u32 ActionIds with zero string duplication.
#[deprecated(
    note = "ActionInterner is no longer used by Gen-Zero; prefer canonical ActionId values at the API boundary"
)]
#[derive(Debug, Default)]
pub struct ActionInterner {
    symbols: RwLock<IndexSet<Arc<str>>>,
}

#[allow(deprecated)]
impl ActionInterner {
    /// Create a new empty interner.
    pub fn new() -> Self {
        Self {
            symbols: RwLock::new(IndexSet::new()),
        }
    }

    /// Retrieve or register an action string, returning its canonical ActionId.
    pub fn get_or_intern(&self, action: &str) -> Result<ActionId, InternerError> {
        // Fast path: acquire read lock first
        {
            let read_guard = self.symbols.read();
            if let Some(index) = read_guard.get_index_of(action) {
                if index > u32::MAX as usize {
                    return Err(InternerError::AddressSpaceExhausted);
                }
                return Ok(ActionId(index as u32));
            }
        }

        // Slow path: acquire write lock and insert
        let mut write_guard = self.symbols.write();
        if let Some(index) = write_guard.get_index_of(action) {
            if index > u32::MAX as usize {
                return Err(InternerError::AddressSpaceExhausted);
            }
            return Ok(ActionId(index as u32));
        }

        // Defend against address space overflow before mutating set
        if write_guard.len() >= u32::MAX as usize {
            return Err(InternerError::AddressSpaceExhausted);
        }

        let arc_str: Arc<str> = Arc::from(action);
        let index = write_guard.insert_full(arc_str).0;
        Ok(ActionId(index as u32))
    }

    /// Resolve an ActionId back to its canonical string representation.
    pub fn resolve(&self, id: ActionId) -> Option<Arc<str>> {
        let read_guard = self.symbols.read();
        read_guard.get_index(id.0 as usize).cloned()
    }

    /// Take a point-in-time snapshot of all registered symbols.
    pub fn snapshot_registry(&self) -> Vec<(ActionId, Arc<str>)> {
        let read_guard = self.symbols.read();
        read_guard
            .iter()
            .enumerate()
            .map(|(idx, s)| (ActionId(idx as u32), Arc::clone(s)))
            .collect()
    }

    /// Return count of registered symbols.
    pub fn len(&self) -> usize {
        self.symbols.read().len()
    }

    /// Check if interner is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_interning_keeps_ids_canonical() {
        let interner = ActionInterner::new();
        let barrier = std::sync::Barrier::new(8);
        let ids = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|worker| {
                    let interner = &interner;
                    let barrier = &barrier;
                    scope.spawn(move || {
                        barrier.wait();
                        let shared = interner.get_or_intern("shared").unwrap();
                        let name = format!("worker-{worker}");
                        let own = interner.get_or_intern(&name).unwrap();
                        assert_eq!(interner.resolve(own).unwrap().as_ref(), name);
                        (shared, own)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(ids.iter().all(|(shared, _)| *shared == ids[0].0));
        let distinct: std::collections::HashSet<_> = ids.iter().map(|(_, own)| own).collect();
        assert_eq!(distinct.len(), 8);
        assert_eq!(interner.len(), 9);
        for (id, name) in interner.snapshot_registry() {
            assert_eq!(interner.resolve(id).unwrap(), name);
        }
    }

    #[test]
    fn test_interner_roundtrip() {
        let interner = ActionInterner::new();
        let id1 = interner.get_or_intern("trade.buy").unwrap();
        let id2 = interner.get_or_intern("trade.sell").unwrap();
        let id1_repeat = interner.get_or_intern("trade.buy").unwrap();

        assert_eq!(id1, id1_repeat);
        assert_ne!(id1, id2);

        assert_eq!(interner.resolve(id1).unwrap().as_ref(), "trade.buy");
        assert_eq!(interner.resolve(id2).unwrap().as_ref(), "trade.sell");
        assert_eq!(interner.resolve(ActionId(999)), None);
    }
}
