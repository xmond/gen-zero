//! `SqliteFeedbackStore`: durable, low-latency storage for reflex inference
//! traces and their late-arriving human/production feedback.
//!
//! A reflex plugin (`gen_zero_model::reflex::ReflexPlugin`) predicts in
//! sub-millisecond time; the label it predicted is not automatically known to
//! be correct. This store records every inference trace (the input state,
//! compressed; the prediction; the reflex telemetry's `max_gamma`) so that
//! feedback which arrives later — seconds, hours, or days after the
//! inference — can be joined back to it by `trace_id`, and so the training
//! pipeline can pull exactly the traces that have both a prediction and a
//! feedback label, exactly once each.
//!
//! WAL mode (`PRAGMA journal_mode = WAL`) lets the hot insert path (an
//! inference trace) run concurrently with the batch reader (the training
//! pipeline's `fetch_unconsumed`) without blocking each other on the SQLite
//! file lock. `PRAGMA synchronous = NORMAL` trades the last few milliseconds
//! of durability under a hard power loss (WAL still fsyncs at checkpoints)
//! for materially lower per-insert latency — the standard trade for a WAL
//! database, and the right one for a proposal store this is not the final
//! system of record for.

use crate::error::StorageError;
use std::io::{Read, Write};
use std::path::Path;

/// Every column of `reflex_feedback_traces`, in schema order. Enforced on
/// open so that a database file created with a stale or hand-edited schema
/// fails closed instead of silently accepting writes into missing or
/// misordered columns (`CREATE TABLE IF NOT EXISTS` alone does not detect
/// this: it happily no-ops against an existing table with any shape).
const SCHEMA_COLUMNS: [&str; 14] = [
    "trace_id",
    "created_at",
    "task",
    "head",
    "plugin_version",
    "z0_blob",
    "head_input_blob",
    "pred_label",
    "pred_confidence",
    "max_gamma",
    "feedback_label",
    "feedback_type",
    "joined_at",
    "trained",
];

const ZSTD_LEVEL: i32 = 3;

/// One inference trace to record: the reflex state, the prediction, and the
/// telemetry needed to audit or retrain on it later.
#[derive(Debug, Clone, PartialEq)]
pub struct ReflexTraceRecord {
    pub trace_id: String,
    /// Unix epoch milliseconds.
    pub created_at_unix_ms: i64,
    pub task: String,
    /// Name of the [`gen_zero_model::reflex::ReflexHead`] that produced
    /// `pred_label`. Required so a multi-head plugin's online adapter knows
    /// which head's `weight`/`bias` a retrained trace belongs to.
    pub head: String,
    pub plugin_version: String,
    /// The reflex operator's initial state `z_0`, float32. Stored as
    /// zstd-compressed float16, which is lossy (~3 decimal digits) but ample
    /// for a training feature and roughly a quarter the size of float32.
    pub z0: Vec<f32>,
    /// The exact feature vector `head`'s linear readout was applied to
    /// (`ReflexPlugin::restored_features`'s output at prediction time), same
    /// compression as `z0`. `z0` alone cannot reconstruct this: it is the
    /// *pre*-recurrence state, while a head reads the *post*-recurrence,
    /// denormalized state. An online adapter that retrained on `z0` would be
    /// training on a different feature distribution than the one the head is
    /// actually evaluated against at serving time.
    pub head_input: Vec<f32>,
    pub pred_label: String,
    pub pred_confidence: f64,
    /// `None` when the reflex recurrence took at most one step (no
    /// consecutive residual pair to ratio).
    pub max_gamma: Option<f64>,
}

/// One row pulled by `fetch_unconsumed`: a trace with feedback attached and
/// not yet marked as consumed by training.
#[derive(Debug, Clone, PartialEq)]
pub struct ReflexFeedbackBatchItem {
    pub trace_id: String,
    pub task: String,
    pub head: String,
    pub plugin_version: String,
    pub z0: Vec<f32>,
    pub head_input: Vec<f32>,
    pub pred_label: String,
    pub pred_confidence: f64,
    pub max_gamma: Option<f64>,
    pub feedback_label: String,
    pub feedback_type: String,
}

/// Row counts from `reflex_feedback_traces`, optionally scoped to one task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReflexFeedbackStatus {
    pub total: u64,
    /// Rows with no `feedback_label` yet.
    pub unlabeled: u64,
    /// Rows with a `feedback_label` that have not been marked trained.
    pub unconsumed: u64,
    /// Rows marked trained.
    pub trained: u64,
}

/// SQLite-backed store for reflex inference traces and joined feedback.
pub struct SqliteFeedbackStore {
    conn: rusqlite::Connection,
    input_dim: usize,
}

impl SqliteFeedbackStore {
    /// Open (creating if needed) the feedback database at `path`.
    ///
    /// `input_dim` is the expected length of every `z0` vector this store's
    /// caller will insert or read back; it is fixed for the store's
    /// lifetime, matching one reflex operator's state width.
    ///
    /// Fails closed if WAL mode cannot be enabled (e.g. `path` names an
    /// in-memory or otherwise WAL-incapable database — the caller asked for
    /// concurrent-safe durability, so a silent fallback to a weaker journal
    /// mode would be a correctness regression the caller cannot see) or if a
    /// pre-existing table at `reflex_feedback_traces` does not match the
    /// exact expected column layout.
    pub fn open(path: &Path, input_dim: usize) -> Result<Self, StorageError> {
        if input_dim == 0 {
            return Err(StorageError::InvalidSampleParameters(
                "input_dim must be positive",
            ));
        }
        let conn = rusqlite::Connection::open(path)?;
        Self::configure(&conn)?;
        Self::ensure_schema(&conn)?;
        Ok(Self { conn, input_dim })
    }

    fn configure(conn: &rusqlite::Connection) -> Result<(), StorageError> {
        let mode: String =
            conn.pragma_update_and_check(None, "journal_mode", "WAL", |row| row.get(0))?;
        if !mode.eq_ignore_ascii_case("wal") {
            return Err(StorageError::PragmaFailed {
                pragma: "journal_mode",
                expected: "wal".into(),
                actual: mode,
            });
        }
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        let synchronous: i64 = conn.query_row("PRAGMA synchronous", [], |row| row.get(0))?;
        if synchronous != 1 {
            return Err(StorageError::PragmaFailed {
                pragma: "synchronous",
                expected: "1 (NORMAL)".into(),
                actual: synchronous.to_string(),
            });
        }
        Ok(())
    }

    fn ensure_schema(conn: &rusqlite::Connection) -> Result<(), StorageError> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS reflex_feedback_traces (
                trace_id        TEXT PRIMARY KEY,
                created_at      INTEGER NOT NULL,
                task            TEXT NOT NULL,
                head            TEXT NOT NULL,
                plugin_version  TEXT NOT NULL,
                z0_blob         BLOB NOT NULL,
                head_input_blob BLOB NOT NULL,
                pred_label      TEXT NOT NULL,
                pred_confidence REAL NOT NULL,
                max_gamma       REAL,
                feedback_label  TEXT,
                feedback_type   TEXT,
                joined_at       INTEGER,
                trained         INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_reflex_feedback_unconsumed
                ON reflex_feedback_traces (joined_at)
                WHERE feedback_label IS NOT NULL AND trained = 0;",
        )?;

        let mut stmt = conn.prepare("PRAGMA table_info(reflex_feedback_traces)")?;
        let actual: Vec<String> = stmt
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<Result<_, _>>()?;
        let expected: Vec<String> = SCHEMA_COLUMNS.iter().map(|s| s.to_string()).collect();
        if actual != expected {
            return Err(StorageError::SchemaMismatch { expected, actual });
        }
        Ok(())
    }

    /// Insert one inference trace. Fails closed on a duplicate `trace_id`
    /// (surfaced as the underlying SQLite `UNIQUE`/`PRIMARY KEY` constraint
    /// error, not silently ignored or overwritten) and on a `z0` whose
    /// length does not match this store's `input_dim`.
    pub fn insert_trace(&self, record: &ReflexTraceRecord) -> Result<(), StorageError> {
        if record.z0.len() != self.input_dim {
            return Err(StorageError::InputDimMismatch {
                expected: self.input_dim,
                actual: record.z0.len(),
            });
        }
        if record.head_input.len() != self.input_dim {
            return Err(StorageError::InputDimMismatch {
                expected: self.input_dim,
                actual: record.head_input.len(),
            });
        }
        let z0_blob = compress_f16(&record.z0)?;
        let head_input_blob = compress_f16(&record.head_input)?;
        self.conn.execute(
            "INSERT INTO reflex_feedback_traces
                (trace_id, created_at, task, head, plugin_version, z0_blob,
                 head_input_blob, pred_label, pred_confidence, max_gamma, trained)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 0)",
            rusqlite::params![
                record.trace_id,
                record.created_at_unix_ms,
                record.task,
                record.head,
                record.plugin_version,
                z0_blob,
                head_input_blob,
                record.pred_label,
                record.pred_confidence,
                record.max_gamma,
            ],
        )?;
        Ok(())
    }

    /// Attach late-arriving feedback to a previously inserted trace.
    ///
    /// Feedback is write-once per trace: a trace that already has a
    /// `feedback_label` refuses a second write rather than silently
    /// overwriting it (a second, different label arriving for the same
    /// trace is almost always a bug upstream, and overwriting would destroy
    /// the evidence of it). An unknown `trace_id` is also an error: nothing
    /// falls back to inserting a new row.
    pub fn record_feedback(
        &self,
        trace_id: &str,
        feedback_label: &str,
        feedback_type: &str,
        joined_at_unix_ms: i64,
    ) -> Result<(), StorageError> {
        let tx = self.conn.unchecked_transaction()?;
        let existing: Option<Option<String>> = tx
            .query_row(
                "SELECT feedback_label FROM reflex_feedback_traces WHERE trace_id = ?1",
                [trace_id],
                |row| row.get(0),
            )
            .ok();
        match existing {
            None => return Err(StorageError::TraceNotFound(trace_id.to_string())),
            Some(Some(_)) => {
                return Err(StorageError::FeedbackAlreadyRecorded(trace_id.to_string()))
            }
            Some(None) => {}
        }
        tx.execute(
            "UPDATE reflex_feedback_traces
             SET feedback_label = ?1, feedback_type = ?2, joined_at = ?3
             WHERE trace_id = ?4",
            rusqlite::params![feedback_label, feedback_type, joined_at_unix_ms, trace_id],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Fetch up to `limit` traces that have feedback but have not yet been
    /// marked as consumed by training, oldest feedback first.
    ///
    /// This does not mark the returned rows as trained: call
    /// [`Self::mark_trained`] only after the training step that consumed
    /// them has itself succeeded, so a crash between fetch and a completed
    /// training step re-delivers the batch instead of silently losing it.
    pub fn fetch_unconsumed(
        &self,
        limit: usize,
    ) -> Result<Vec<ReflexFeedbackBatchItem>, StorageError> {
        let mut stmt = self.conn.prepare(
            "SELECT trace_id, task, head, plugin_version, z0_blob, head_input_blob,
                    pred_label, pred_confidence, max_gamma, feedback_label, feedback_type
             FROM reflex_feedback_traces
             WHERE feedback_label IS NOT NULL AND trained = 0
             ORDER BY joined_at ASC
             LIMIT ?1",
        )?;
        let input_dim = self.input_dim;
        let rows = stmt.query_map([limit as i64], move |row| {
            let z0_blob: Vec<u8> = row.get(4)?;
            let head_input_blob: Vec<u8> = row.get(5)?;
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                z0_blob,
                head_input_blob,
                row.get::<_, String>(6)?,
                row.get::<_, f64>(7)?,
                row.get::<_, Option<f64>>(8)?,
                row.get::<_, String>(9)?,
                row.get::<_, String>(10)?,
            ))
        })?;

        let mut out = Vec::new();
        for row in rows {
            let (
                trace_id,
                task,
                head,
                plugin_version,
                z0_blob,
                head_input_blob,
                pred_label,
                pred_confidence,
                max_gamma,
                feedback_label,
                feedback_type,
            ) = row?;
            let z0 = decompress_f16(&z0_blob, input_dim)
                .map_err(|e| StorageError::CorruptBlob(format!("{trace_id}: {e}")))?;
            let head_input = decompress_f16(&head_input_blob, input_dim)
                .map_err(|e| StorageError::CorruptBlob(format!("{trace_id}: {e}")))?;
            out.push(ReflexFeedbackBatchItem {
                trace_id,
                task,
                head,
                plugin_version,
                z0,
                head_input,
                pred_label,
                pred_confidence,
                max_gamma,
                feedback_label,
                feedback_type,
            });
        }
        Ok(out)
    }

    /// Row counts from `reflex_feedback_traces`, optionally restricted to one
    /// `task`. `unlabeled + unconsumed + trained == total` always holds.
    pub fn feedback_status(
        &self,
        task: Option<&str>,
    ) -> Result<ReflexFeedbackStatus, StorageError> {
        const COUNTS_SQL: &str = "SELECT COUNT(*),
                    SUM(CASE WHEN feedback_label IS NULL THEN 1 ELSE 0 END),
                    SUM(CASE WHEN feedback_label IS NOT NULL AND trained = 0 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN trained = 1 THEN 1 ELSE 0 END)
             FROM reflex_feedback_traces";
        let row =
            |total: i64, unlabeled: Option<i64>, unconsumed: Option<i64>, trained: Option<i64>| {
                ReflexFeedbackStatus {
                    total: total as u64,
                    unlabeled: unlabeled.unwrap_or(0) as u64,
                    unconsumed: unconsumed.unwrap_or(0) as u64,
                    trained: trained.unwrap_or(0) as u64,
                }
            };
        match task {
            Some(t) => self
                .conn
                .query_row(&format!("{COUNTS_SQL} WHERE task = ?1"), [t], |r| {
                    Ok(row(r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
                }),
            None => self.conn.query_row(COUNTS_SQL, [], |r| {
                Ok(row(r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
            }),
        }
        .map_err(StorageError::from)
    }

    /// Delete traces created before `cutoff_unix_ms`.
    ///
    /// By default this only deletes rows that are safe to lose without
    /// destroying evidence of untrained feedback: already-trained rows, and
    /// rows with no feedback label at all. A labeled row that has not yet
    /// been consumed by training is kept unless `include_unconsumed_labeled`
    /// is explicitly set, since deleting it would silently discard feedback
    /// the training loop has not had a chance to learn from.
    ///
    /// Returns the number of rows deleted.
    pub fn prune_before(
        &self,
        cutoff_unix_ms: i64,
        include_unconsumed_labeled: bool,
    ) -> Result<u64, StorageError> {
        let n = if include_unconsumed_labeled {
            self.conn.execute(
                "DELETE FROM reflex_feedback_traces WHERE created_at < ?1",
                [cutoff_unix_ms],
            )?
        } else {
            self.conn.execute(
                "DELETE FROM reflex_feedback_traces
                 WHERE created_at < ?1 AND (trained = 1 OR feedback_label IS NULL)",
                [cutoff_unix_ms],
            )?
        };
        Ok(n as u64)
    }

    /// Mark the given traces as consumed by training. Unknown `trace_id`s
    /// are silently skipped by the `UPDATE ... WHERE trace_id = ?` per id (a
    /// trace mentioned twice or already deleted is not itself an error: the
    /// caller's contract is "these are trained now", which is idempotent).
    pub fn mark_trained(&self, trace_ids: &[String]) -> Result<(), StorageError> {
        if trace_ids.is_empty() {
            return Ok(());
        }
        let tx = self.conn.unchecked_transaction()?;
        {
            let mut stmt =
                tx.prepare("UPDATE reflex_feedback_traces SET trained = 1 WHERE trace_id = ?1")?;
            for id in trace_ids {
                stmt.execute([id])?;
            }
        }
        tx.commit()?;
        Ok(())
    }
}

fn compress_f16(values: &[f32]) -> Result<Vec<u8>, StorageError> {
    let mut raw = Vec::with_capacity(values.len() * 2);
    for &v in values {
        raw.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
    }
    let mut encoder = zstd::stream::Encoder::new(Vec::new(), ZSTD_LEVEL)
        .map_err(|e| StorageError::Compression(e.to_string()))?;
    encoder
        .write_all(&raw)
        .map_err(|e| StorageError::Compression(e.to_string()))?;
    encoder
        .finish()
        .map_err(|e| StorageError::Compression(e.to_string()))
}

fn decompress_f16(compressed: &[u8], expected_len: usize) -> Result<Vec<f32>, StorageError> {
    let decoder = zstd::stream::Decoder::new(compressed)
        .map_err(|e| StorageError::Compression(e.to_string()))?;
    let expected_bytes = expected_len
        .checked_mul(2)
        .ok_or_else(|| StorageError::CorruptBlob("decoded length overflows usize".into()))?;
    // Bound the decompressed size to guard against a decompression bomb: a
    // tiny compressed blob is never allowed to inflate past the caller's
    // declared vector length, regardless of what the stream claims.
    let mut bounded = Read::take(decoder, expected_bytes as u64 + 1);
    let mut raw = Vec::new();
    bounded.read_to_end(&mut raw)?;
    if raw.len() != expected_bytes {
        return Err(StorageError::CorruptBlob(format!(
            "expected {expected_bytes} decompressed bytes ({expected_len} f16 values), got {}",
            raw.len()
        )));
    }
    Ok(raw
        .chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn open_temp_store(input_dim: usize) -> (TempDir, SqliteFeedbackStore) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("reflex_feedback.sqlite3");
        let store = SqliteFeedbackStore::open(&path, input_dim).unwrap();
        (dir, store)
    }

    fn sample_record(trace_id: &str, z0: Vec<f32>) -> ReflexTraceRecord {
        let head_input = z0.iter().map(|&v| v * 2.0 + 1.0).collect();
        ReflexTraceRecord {
            trace_id: trace_id.to_string(),
            created_at_unix_ms: 1_700_000_000_000,
            task: "multinli".to_string(),
            head: "entail".to_string(),
            plugin_version: "v1".to_string(),
            head_input,
            z0,
            pred_label: "entailment".to_string(),
            pred_confidence: 0.91,
            max_gamma: Some(0.42),
        }
    }

    #[test]
    fn open_enables_wal_and_normal_synchronous() {
        let (_dir, store) = open_temp_store(4);
        let mode: String = store
            .conn
            .query_row("PRAGMA journal_mode", [], |r| r.get(0))
            .unwrap();
        assert_eq!(mode, "wal");
        let sync: i64 = store
            .conn
            .query_row("PRAGMA synchronous", [], |r| r.get(0))
            .unwrap();
        assert_eq!(sync, 1);
    }

    #[test]
    fn reopening_existing_database_reuses_schema_without_drift() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("reflex_feedback.sqlite3");
        {
            let store = SqliteFeedbackStore::open(&path, 4).unwrap();
            store
                .insert_trace(&sample_record("t1", vec![1.0, 2.0, 3.0, 4.0]))
                .unwrap();
        }
        let store = SqliteFeedbackStore::open(&path, 4).unwrap();
        assert_eq!(store.fetch_unconsumed(10).unwrap().len(), 0);
    }

    #[test]
    fn full_lifecycle_insert_feedback_fetch_mark_trained() {
        let (_dir, store) = open_temp_store(4);
        let z0 = vec![0.5, -0.25, 1.5, -1.5];
        store
            .insert_trace(&sample_record("t1", z0.clone()))
            .unwrap();

        assert_eq!(store.fetch_unconsumed(10).unwrap().len(), 0);

        store
            .record_feedback("t1", "entailment", "human_review", 1_700_000_100_000)
            .unwrap();

        let batch = store.fetch_unconsumed(10).unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].trace_id, "t1");
        assert_eq!(batch[0].head, "entail");
        assert_eq!(batch[0].feedback_label, "entailment");
        assert_eq!(batch[0].feedback_type, "human_review");
        assert_eq!(batch[0].max_gamma, Some(0.42));
        for (a, b) in batch[0].z0.iter().zip(z0.iter()) {
            // float16 round trip: within its ~3 decimal digit precision.
            assert!((a - b).abs() < 1e-2, "{a} vs {b}");
        }
        let expected_head_input: Vec<f32> = z0.iter().map(|&v| v * 2.0 + 1.0).collect();
        for (a, b) in batch[0].head_input.iter().zip(expected_head_input.iter()) {
            assert!((a - b).abs() < 1e-2, "{a} vs {b}");
        }

        store.mark_trained(&["t1".to_string()]).unwrap();
        assert_eq!(store.fetch_unconsumed(10).unwrap().len(), 0);
    }

    #[test]
    fn feedback_status_counts_unlabeled_unconsumed_and_trained() {
        let (_dir, store) = open_temp_store(2);
        store
            .insert_trace(&sample_record("unlabeled", vec![0.0, 0.0]))
            .unwrap();
        store
            .insert_trace(&sample_record("pending", vec![0.0, 0.0]))
            .unwrap();
        store
            .insert_trace(&sample_record("done", vec![0.0, 0.0]))
            .unwrap();
        store.record_feedback("pending", "y", "auto", 1).unwrap();
        store.record_feedback("done", "y", "auto", 2).unwrap();
        store.mark_trained(&["done".to_string()]).unwrap();

        let status = store.feedback_status(None).unwrap();
        assert_eq!(status.total, 3);
        assert_eq!(status.unlabeled, 1);
        assert_eq!(status.unconsumed, 1);
        assert_eq!(status.trained, 1);
    }

    #[test]
    fn feedback_status_filters_by_task() {
        let (_dir, store) = open_temp_store(2);
        let mut a = sample_record("a", vec![0.0, 0.0]);
        a.task = "task_a".into();
        let mut b = sample_record("b", vec![0.0, 0.0]);
        b.task = "task_b".into();
        store.insert_trace(&a).unwrap();
        store.insert_trace(&b).unwrap();

        let status = store.feedback_status(Some("task_a")).unwrap();
        assert_eq!(status.total, 1);
        assert_eq!(status.unlabeled, 1);
    }

    #[test]
    fn prune_before_default_keeps_unconsumed_labeled_rows() {
        let (_dir, store) = open_temp_store(2);
        let mut old_unlabeled = sample_record("old_unlabeled", vec![0.0, 0.0]);
        old_unlabeled.created_at_unix_ms = 100;
        let mut old_trained = sample_record("old_trained", vec![0.0, 0.0]);
        old_trained.created_at_unix_ms = 100;
        let mut old_pending = sample_record("old_pending", vec![0.0, 0.0]);
        old_pending.created_at_unix_ms = 100;
        store.insert_trace(&old_unlabeled).unwrap();
        store.insert_trace(&old_trained).unwrap();
        store.insert_trace(&old_pending).unwrap();
        store
            .record_feedback("old_trained", "y", "auto", 100)
            .unwrap();
        store
            .record_feedback("old_pending", "y", "auto", 100)
            .unwrap();
        store.mark_trained(&["old_trained".to_string()]).unwrap();

        let deleted = store.prune_before(1_000, false).unwrap();
        assert_eq!(deleted, 2); // old_unlabeled + old_trained, not old_pending
        assert_eq!(store.feedback_status(None).unwrap().total, 1);
        assert_eq!(store.fetch_unconsumed(10).unwrap().len(), 1);
    }

    #[test]
    fn prune_before_force_also_deletes_unconsumed_labeled_rows() {
        let (_dir, store) = open_temp_store(2);
        let mut old_pending = sample_record("old_pending", vec![0.0, 0.0]);
        old_pending.created_at_unix_ms = 100;
        store.insert_trace(&old_pending).unwrap();
        store
            .record_feedback("old_pending", "y", "auto", 100)
            .unwrap();

        let deleted = store.prune_before(1_000, true).unwrap();
        assert_eq!(deleted, 1);
        assert_eq!(store.feedback_status(None).unwrap().total, 0);
    }

    #[test]
    fn prune_before_ignores_rows_at_or_after_cutoff() {
        let (_dir, store) = open_temp_store(2);
        let mut recent = sample_record("recent", vec![0.0, 0.0]);
        recent.created_at_unix_ms = 5_000;
        store.insert_trace(&recent).unwrap();

        let deleted = store.prune_before(1_000, true).unwrap();
        assert_eq!(deleted, 0);
        assert_eq!(store.feedback_status(None).unwrap().total, 1);
    }

    #[test]
    fn insert_rejects_wrong_input_dim() {
        let (_dir, store) = open_temp_store(4);
        let err = store
            .insert_trace(&sample_record("t1", vec![1.0, 2.0]))
            .unwrap_err();
        assert!(matches!(err, StorageError::InputDimMismatch { .. }));
    }

    #[test]
    fn insert_rejects_duplicate_trace_id() {
        let (_dir, store) = open_temp_store(2);
        store
            .insert_trace(&sample_record("dup", vec![1.0, 2.0]))
            .unwrap();
        let err = store
            .insert_trace(&sample_record("dup", vec![3.0, 4.0]))
            .unwrap_err();
        assert!(matches!(err, StorageError::Sqlite(_)));
    }

    #[test]
    fn record_feedback_rejects_unknown_trace() {
        let (_dir, store) = open_temp_store(2);
        let err = store
            .record_feedback("missing", "x", "human_review", 0)
            .unwrap_err();
        assert!(matches!(err, StorageError::TraceNotFound(_)));
    }

    #[test]
    fn record_feedback_is_write_once() {
        let (_dir, store) = open_temp_store(2);
        store
            .insert_trace(&sample_record("t1", vec![1.0, 2.0]))
            .unwrap();
        store
            .record_feedback("t1", "a", "human_review", 10)
            .unwrap();
        let err = store
            .record_feedback("t1", "b", "human_review", 20)
            .unwrap_err();
        assert!(matches!(err, StorageError::FeedbackAlreadyRecorded(_)));
        // The original feedback must survive the rejected overwrite attempt.
        let batch = store.fetch_unconsumed(10).unwrap();
        assert_eq!(batch[0].feedback_label, "a");
    }

    #[test]
    fn fetch_unconsumed_excludes_trained_and_unlabeled_rows() {
        let (_dir, store) = open_temp_store(2);
        store
            .insert_trace(&sample_record("no_feedback", vec![0.0, 0.0]))
            .unwrap();
        store
            .insert_trace(&sample_record("trained", vec![0.0, 0.0]))
            .unwrap();
        store
            .insert_trace(&sample_record("pending", vec![0.0, 0.0]))
            .unwrap();

        store.record_feedback("trained", "x", "auto", 1).unwrap();
        store.record_feedback("pending", "y", "auto", 2).unwrap();
        store.mark_trained(&["trained".to_string()]).unwrap();

        let batch = store.fetch_unconsumed(10).unwrap();
        let ids: Vec<&str> = batch.iter().map(|b| b.trace_id.as_str()).collect();
        assert_eq!(ids, vec!["pending"]);
    }

    #[test]
    fn fetch_unconsumed_respects_limit() {
        let (_dir, store) = open_temp_store(1);
        for i in 0..5 {
            let id = format!("t{i}");
            store
                .insert_trace(&sample_record(&id, vec![i as f32]))
                .unwrap();
            store.record_feedback(&id, "y", "auto", i as i64).unwrap();
        }
        assert_eq!(store.fetch_unconsumed(2).unwrap().len(), 2);
        assert_eq!(store.fetch_unconsumed(100).unwrap().len(), 5);
    }

    #[test]
    fn mark_trained_on_unknown_id_is_a_harmless_no_op() {
        let (_dir, store) = open_temp_store(2);
        store
            .insert_trace(&sample_record("t1", vec![0.0, 0.0]))
            .unwrap();
        store.record_feedback("t1", "y", "auto", 1).unwrap();
        store.mark_trained(&["does-not-exist".to_string()]).unwrap();
        assert_eq!(store.fetch_unconsumed(10).unwrap().len(), 1);
    }

    #[test]
    fn max_gamma_null_round_trips_as_none() {
        let (_dir, store) = open_temp_store(2);
        let mut record = sample_record("t1", vec![0.0, 0.0]);
        record.max_gamma = None;
        store.insert_trace(&record).unwrap();
        store.record_feedback("t1", "y", "auto", 1).unwrap();
        let batch = store.fetch_unconsumed(10).unwrap();
        assert_eq!(batch[0].max_gamma, None);
    }

    #[test]
    fn f16_compression_round_trips_within_precision() {
        let values = vec![1.0f32, -2.5, 0.001, 12345.6, -0.0, f32::MIN_POSITIVE];
        let blob = compress_f16(&values).unwrap();
        let restored = decompress_f16(&blob, values.len()).unwrap();
        for (a, b) in values.iter().zip(restored.iter()) {
            let rel_err = ((a - b).abs()) / a.abs().max(1e-6);
            assert!(rel_err < 2e-3, "{a} vs {b} (rel_err={rel_err})");
        }
    }

    #[test]
    fn decompress_rejects_length_mismatch() {
        let blob = compress_f16(&[1.0, 2.0, 3.0]).unwrap();
        let err = decompress_f16(&blob, 5).unwrap_err();
        assert!(matches!(err, StorageError::CorruptBlob(_)));
    }

    #[test]
    fn decompress_rejects_bomb_inflating_past_declared_length() {
        // A blob honestly compressing far more data than the caller declares
        // must be rejected, not silently truncated into a "valid" answer.
        let huge = vec![0.0f32; 10_000];
        let blob = compress_f16(&huge).unwrap();
        let err = decompress_f16(&blob, 4).unwrap_err();
        assert!(matches!(err, StorageError::CorruptBlob(_)));
    }
}
