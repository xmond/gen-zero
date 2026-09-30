//! gen-zero-storage
//!
//! Columnar Causal Replay Buffer, Fenwick Tree Prioritized Causal Sampling (Causal PER),
//! Chunked zstd Streaming, and Golden Snapshot Rollback.
//!
//! The production service uses `GoldenSnapshotManager` to capture mounted cognitive
//! assets and restore them through its validated, versioned publication path.
//! Replay primitives remain available to training consumers.

pub mod error;
pub mod fenwick;
pub mod reflex_store;
pub mod replay;
pub mod snapshot;

pub use error::StorageError;
pub use fenwick::FenwickTree;
pub use reflex_store::{
    ReflexFeedbackBatchItem, ReflexFeedbackStatus, ReflexTraceRecord, SqliteFeedbackStore,
};
pub use replay::{CausalSampleBatch, ColumnarCausalReplayBuffer, TrajectoryStep};
pub use snapshot::{GoldenSnapshot, GoldenSnapshotManager};

#[cfg(test)]
mod tests {
    use super::*;
    use gen_zero_core::{ActionId, FullLatent};

    #[test]
    fn test_storage_end_to_end() {
        let mut buffer = ColumnarCausalReplayBuffer::new(20, 0.6, 0.5);
        for i in 0..10 {
            buffer
                .push(TrajectoryStep {
                    step_id: i as u64,
                    state: FullLatent::default(),
                    action: ActionId(i as u32),
                    reward: 2.0,
                    next_state: FullLatent::default(),
                    done: false,
                    td_error: 0.5,
                    causal_shock: 0.1,
                })
                .unwrap();
        }
        assert_eq!(buffer.len(), 10);

        let batch = buffer.sample(4, 0.5).unwrap();
        assert_eq!(batch.indices.len(), 4);

        let mut snapshot_mgr = GoldenSnapshotManager::new(3, [0x11u8; 32], 5);
        let raw_blob = vec![123u8; 1024];
        let snap = snapshot_mgr.capture_snapshot(1, 10, &raw_blob).unwrap();
        assert_eq!(snap.snapshot_id, 1);

        let restored = snapshot_mgr.rollback_snapshot(1).unwrap();
        assert_eq!(restored, raw_blob);
    }
}
