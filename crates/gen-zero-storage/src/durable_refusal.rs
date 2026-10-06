//! `DurableRefusalStore`: crash-durable storage for answerability-gateway
//! refusals awaiting arbitration, and for the patches built from them.
//!
//! This is Phase 1 of the self-evolving training pipeline: when the
//! answerability gateway refuses to answer, the refusal (context, question,
//! candidate, and the scores that drove the refusal) is recorded here so a
//! human or model arbiter can review it later, out of the hot inference
//! path. Once arbitrated, a batch of traces can be consumed into a training
//! patch; if that patch misbehaves in canary and is rolled back, every trace
//! it consumed is revoked so it can never be silently resurrected into a
//! later patch's training set.
//!
//! WAL mode and `synchronous = NORMAL` are configured the same way, and for
//! the same reasons, as `reflex_store::SqliteFeedbackStore` (see that
//! module's doc comment).
//!
//! The connection is wrapped in a [`Mutex`] rather than held bare: unlike
//! `SqliteFeedbackStore` (which is only ever owned by one caller),
//! `DurableRefusalStore` is meant to be shared behind an `Arc` across
//! threads — a future gateway hook will call it from `&self` methods on a
//! different thread than the arbitration loop that drains it.
//! `rusqlite::Connection` is `Send` but not `Sync`, so a bare field would
//! make the store `!Sync` and therefore unusable behind `Arc` from multiple
//! threads; the mutex makes it `Sync` by constraining every access to hold
//! the lock for the duration of one SQLite call or transaction.

use crate::error::StorageError;
use rusqlite::OptionalExtension;
use std::path::Path;
use std::str::FromStr;
use std::sync::{Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

const TRACE_COLUMNS: [&str; 21] = [
    "trace_id",
    "created_at_ms",
    "context",
    "question",
    "candidate",
    "best_span_score",
    "null_score",
    "score_diff",
    "verifier_output",
    "arbitration_status",
    "is_answerable",
    "gold_answer",
    "evidence_span",
    "contradiction_fact",
    "arbitrator_model",
    "arbitration_raw",
    "arbitrated_at_ms",
    "retry_count",
    "last_error",
    "is_consumed",
    "applied_patch_id",
];

const PATCH_COLUMNS: [&str; 11] = [
    "patch_id",
    "created_at_ms",
    "base_model_hash",
    "patch_file_path",
    "sha256",
    "samples_count",
    "validation_status",
    "status",
    "rollback_reason",
    "rolled_back_at_ms",
    "deployed_at_ms",
];

const CANARY_COLUMNS: [&str; 7] = [
    "sample_id",
    "patch_id",
    "recorded_at_ms",
    "is_fast_pass",
    "is_refused",
    "latency_us",
    "is_hard_stop",
];

const SELECT_TRACE_COLUMNS: &str = "trace_id, created_at_ms, context, question, candidate, \
    best_span_score, null_score, score_diff, verifier_output, arbitration_status, \
    is_answerable, gold_answer, evidence_span, contradiction_fact, arbitrator_model, \
    arbitration_raw, arbitrated_at_ms, retry_count, last_error, is_consumed, applied_patch_id";

const SELECT_PATCH_COLUMNS: &str = "patch_id, created_at_ms, base_model_hash, patch_file_path, \
    sha256, samples_count, validation_status, status, rollback_reason, rolled_back_at_ms, \
    deployed_at_ms";

/// `durable_refusal_traces.arbitration_status`. Only three states are ever
/// written: a row starts `Pending`, moves to `Arbitrated` on a successful
/// [`DurableRefusalStore::complete_arbitration`], or to `Revoked` if the
/// patch that consumed it is later rolled back. A failed arbitration
/// ([`DurableRefusalStore::fail_arbitration`]) returns the row to `Pending`
/// rather than introducing a terminal failure state — the retry loop is the
/// caller's responsibility, not this storage layer's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArbitrationStatus {
    Pending,
    Arbitrated,
    Revoked,
}

impl ArbitrationStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "PENDING",
            Self::Arbitrated => "ARBITRATED",
            Self::Revoked => "REVOKED",
        }
    }
}

impl FromStr for ArbitrationStatus {
    type Err = StorageError;

    fn from_str(s: &str) -> Result<Self, StorageError> {
        match s {
            "PENDING" => Ok(Self::Pending),
            "ARBITRATED" => Ok(Self::Arbitrated),
            "REVOKED" => Ok(Self::Revoked),
            other => Err(StorageError::InvalidArbitrationStatusString(
                other.to_string(),
            )),
        }
    }
}

/// One refusal to record: the gateway's inputs and the scores that drove
/// the refusal decision. Every other column of `durable_refusal_traces`
/// (arbitration fields, retry bookkeeping, consumption) starts out NULL or
/// at its default and is only ever written by a later method call.
#[derive(Debug, Clone, PartialEq)]
pub struct RefusalTraceInput {
    pub trace_id: String,
    pub created_at_ms: i64,
    pub context: String,
    pub question: String,
    pub candidate: String,
    pub best_span_score: f32,
    pub null_score: f32,
    pub score_diff: f32,
    pub verifier_output: Option<String>,
}

/// The outcome of a human/model arbiter reviewing one refusal trace, as
/// written by [`DurableRefusalStore::complete_arbitration`].
#[derive(Debug, Clone, PartialEq)]
pub struct ArbitrationResult {
    pub is_answerable: bool,
    pub gold_answer: Option<String>,
    pub evidence_span: Option<String>,
    pub contradiction_fact: Option<String>,
    pub arbitrator_model: String,
    pub arbitration_raw: String,
    pub arbitrated_at_ms: i64,
}

/// One full row of `durable_refusal_traces`.
#[derive(Debug, Clone, PartialEq)]
pub struct DurableRefusalTrace {
    pub trace_id: String,
    pub created_at_ms: i64,
    pub context: String,
    pub question: String,
    pub candidate: String,
    pub best_span_score: f32,
    pub null_score: f32,
    pub score_diff: f32,
    pub verifier_output: Option<String>,
    pub arbitration_status: ArbitrationStatus,
    pub is_answerable: Option<bool>,
    pub gold_answer: Option<String>,
    pub evidence_span: Option<String>,
    pub contradiction_fact: Option<String>,
    pub arbitrator_model: Option<String>,
    pub arbitration_raw: Option<String>,
    pub arbitrated_at_ms: Option<i64>,
    pub retry_count: i64,
    pub last_error: Option<String>,
    pub is_consumed: bool,
    pub applied_patch_id: Option<String>,
}

/// Input to [`DurableRefusalStore::record_patch`]. `rollback_reason` and
/// `rolled_back_at_ms` are deliberately absent: those columns start NULL and
/// are only ever written by [`DurableRefusalStore::rollback_patch`].
#[derive(Debug, Clone, PartialEq)]
pub struct DurablePatchRecord {
    pub patch_id: String,
    pub created_at_ms: i64,
    pub base_model_hash: String,
    pub patch_file_path: String,
    pub sha256: String,
    pub samples_count: i64,
    pub validation_status: String,
    pub status: String,
    pub deployed_at_ms: Option<i64>,
}

/// One full row of `durable_patches`, including the rollback columns that
/// [`DurablePatchRecord`] cannot carry as input.
#[derive(Debug, Clone, PartialEq)]
pub struct DurablePatchRow {
    pub patch_id: String,
    pub created_at_ms: i64,
    pub base_model_hash: String,
    pub patch_file_path: String,
    pub sha256: String,
    pub samples_count: i64,
    pub validation_status: String,
    pub status: String,
    pub rollback_reason: Option<String>,
    pub rolled_back_at_ms: Option<i64>,
    pub deployed_at_ms: Option<i64>,
}

/// One canary observation to append to `canary_metrics`: how one serving
/// decision went while `patch_id` was the deployed patch.
#[derive(Debug, Clone, PartialEq)]
pub struct CanaryMetricInput {
    pub patch_id: String,
    pub recorded_at_ms: i64,
    pub is_fast_pass: bool,
    pub is_refused: bool,
    pub latency_us: i64,
    pub is_hard_stop: bool,
}

/// One full row of `canary_metrics`.
#[derive(Debug, Clone, PartialEq)]
pub struct CanaryMetricRow {
    pub sample_id: i64,
    pub patch_id: String,
    pub recorded_at_ms: i64,
    pub is_fast_pass: bool,
    pub is_refused: bool,
    pub latency_us: i64,
    pub is_hard_stop: bool,
}

/// Aggregate of the latest `window` canary rows of one patch, as computed by
/// [`DurableRefusalStore::compute_canary_stats`].
#[derive(Debug, Clone, PartialEq)]
pub struct CanaryStats {
    pub patch_id: String,
    /// The requested window size; `total_samples <= window`.
    pub window: usize,
    pub total_samples: usize,
    pub refused_count: usize,
    pub fast_pass_count: usize,
    pub hard_stop_count: usize,
    pub refusal_rate: f64,
    pub fast_pass_rate: f64,
    pub hard_stop_rate: f64,
    pub avg_latency_us: f64,
}

/// SQLite-backed store for answerability-gateway refusals, their
/// arbitration, and the patches trained from them.
pub struct DurableRefusalStore {
    conn: Mutex<rusqlite::Connection>,
}

impl DurableRefusalStore {
    /// Open (creating if needed) the durable refusal database at `path`.
    ///
    /// Fails closed if WAL mode cannot be enabled, or if any of the three
    /// tables already exists with a column layout that does not match
    /// exactly (see the per-table `*_COLUMNS` constants).
    pub fn new(path: &Path) -> Result<Self, StorageError> {
        let conn = rusqlite::Connection::open(path)?;
        Self::configure(&conn)?;
        Self::ensure_schema(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
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
            "CREATE TABLE IF NOT EXISTS durable_refusal_traces (
                trace_id            TEXT PRIMARY KEY,
                created_at_ms        INTEGER NOT NULL,
                context              TEXT NOT NULL,
                question             TEXT NOT NULL,
                candidate            TEXT NOT NULL,
                best_span_score      REAL NOT NULL,
                null_score           REAL NOT NULL,
                score_diff           REAL NOT NULL,
                verifier_output      TEXT,
                arbitration_status   TEXT NOT NULL,
                is_answerable        INTEGER,
                gold_answer          TEXT,
                evidence_span        TEXT,
                contradiction_fact   TEXT,
                arbitrator_model     TEXT,
                arbitration_raw      TEXT,
                arbitrated_at_ms     INTEGER,
                retry_count          INTEGER NOT NULL DEFAULT 0,
                last_error           TEXT,
                is_consumed          INTEGER NOT NULL DEFAULT 0,
                applied_patch_id     TEXT
            );
            CREATE TABLE IF NOT EXISTS durable_patches (
                patch_id            TEXT PRIMARY KEY,
                created_at_ms       INTEGER NOT NULL,
                base_model_hash     TEXT NOT NULL,
                patch_file_path     TEXT NOT NULL,
                sha256              TEXT NOT NULL,
                samples_count       INTEGER NOT NULL,
                validation_status   TEXT NOT NULL,
                status              TEXT NOT NULL,
                rollback_reason     TEXT,
                rolled_back_at_ms   INTEGER,
                deployed_at_ms      INTEGER
            );
            CREATE TABLE IF NOT EXISTS canary_metrics (
                sample_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                patch_id       TEXT NOT NULL,
                recorded_at_ms INTEGER NOT NULL,
                is_fast_pass   INTEGER NOT NULL,
                is_refused     INTEGER NOT NULL,
                latency_us     INTEGER NOT NULL,
                is_hard_stop   INTEGER NOT NULL
            );",
        )?;
        check_columns(conn, "durable_refusal_traces", &TRACE_COLUMNS)?;
        check_columns(conn, "durable_patches", &PATCH_COLUMNS)?;
        check_columns(conn, "canary_metrics", &CANARY_COLUMNS)?;
        Ok(())
    }

    fn lock(&self) -> Result<MutexGuard<'_, rusqlite::Connection>, StorageError> {
        self.conn.lock().map_err(|_| StorageError::LockPoisoned)
    }

    /// Record a new refusal. Starts `arbitration_status = 'PENDING'`,
    /// `retry_count = 0`, `is_consumed = 0`, and every arbitration/consumption
    /// field NULL.
    ///
    /// A duplicate `trace_id` surfaces as the underlying SQLite
    /// `PRIMARY KEY` constraint error (`StorageError::Sqlite`) and is not
    /// caught or remapped: silently accepting a second insert for the same
    /// trace would risk overwriting a refusal already in arbitration.
    pub fn record_refusal(&self, record: &RefusalTraceInput) -> Result<(), StorageError> {
        let conn = self.lock()?;
        conn.execute(
            "INSERT INTO durable_refusal_traces
                (trace_id, created_at_ms, context, question, candidate, best_span_score,
                 null_score, score_diff, verifier_output, arbitration_status, retry_count,
                 is_consumed)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 'PENDING', 0, 0)",
            rusqlite::params![
                record.trace_id,
                record.created_at_ms,
                record.context,
                record.question,
                record.candidate,
                record.best_span_score,
                record.null_score,
                record.score_diff,
                record.verifier_output,
            ],
        )?;
        Ok(())
    }

    /// Fetch up to `limit` traces awaiting arbitration, oldest first.
    pub fn fetch_pending_arbitration(
        &self,
        limit: usize,
    ) -> Result<Vec<DurableRefusalTrace>, StorageError> {
        let conn = self.lock()?;
        let sql = format!(
            "SELECT {SELECT_TRACE_COLUMNS} FROM durable_refusal_traces
             WHERE arbitration_status = 'PENDING'
             ORDER BY created_at_ms ASC
             LIMIT ?1"
        );
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query([limit as i64])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_trace(row)?);
        }
        Ok(out)
    }

    /// Like [`Self::fetch_pending_arbitration`], but excludes rows whose
    /// `retry_count` has already reached `max_retries`, so the arbitration
    /// loop never starves on permanently-failing rows while leaving them
    /// queryable for all other purposes (no terminal failure state is
    /// introduced).
    pub fn fetch_pending_arbitration_under_retry(
        &self,
        limit: usize,
        max_retries: u32,
    ) -> Result<Vec<DurableRefusalTrace>, StorageError> {
        let conn = self.lock()?;
        let sql = format!(
            "SELECT {SELECT_TRACE_COLUMNS} FROM durable_refusal_traces
             WHERE arbitration_status = 'PENDING' AND retry_count < ?2
             ORDER BY created_at_ms ASC
             LIMIT ?1"
        );
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query(rusqlite::params![limit as i64, max_retries as i64])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_trace(row)?);
        }
        Ok(out)
    }

    /// Fetch up to `limit` arbitrated-but-not-yet-consumed traces, ordered
    /// by when they were arbitrated.
    pub fn fetch_unconsumed_arbitrated(
        &self,
        limit: usize,
    ) -> Result<Vec<DurableRefusalTrace>, StorageError> {
        let conn = self.lock()?;
        let sql = format!(
            "SELECT {SELECT_TRACE_COLUMNS} FROM durable_refusal_traces
             WHERE arbitration_status = 'ARBITRATED' AND is_consumed = 0
             ORDER BY arbitrated_at_ms ASC
             LIMIT ?1"
        );
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query([limit as i64])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_trace(row)?);
        }
        Ok(out)
    }

    /// Fetch one trace by id, regardless of its arbitration/consumption
    /// state. `None` if no such trace exists.
    pub fn get_trace(&self, trace_id: &str) -> Result<Option<DurableRefusalTrace>, StorageError> {
        let conn = self.lock()?;
        let sql = format!(
            "SELECT {SELECT_TRACE_COLUMNS} FROM durable_refusal_traces WHERE trace_id = ?1"
        );
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query([trace_id])?;
        match rows.next()? {
            Some(row) => Ok(Some(row_to_trace(row)?)),
            None => Ok(None),
        }
    }

    /// Complete arbitration of `trace_id` with `result`.
    ///
    /// The row must currently be `PENDING`; a row that is `ARBITRATED`
    /// (already completed) or `REVOKED` (its patch was rolled back) fails
    /// with [`StorageError::InvalidArbitrationState`] instead of being
    /// resurrected or overwritten. Unknown `trace_id` fails with
    /// [`StorageError::DurableTraceNotFound`].
    ///
    /// On success: `arbitration_status` becomes `'ARBITRATED'`, every
    /// `ArbitrationResult` field is written, `last_error` is cleared to
    /// NULL, and `retry_count` is left untouched.
    pub fn complete_arbitration(
        &self,
        trace_id: &str,
        result: &ArbitrationResult,
    ) -> Result<(), StorageError> {
        let conn = self.lock()?;
        let tx = conn.unchecked_transaction()?;
        let status = current_trace_status(&tx, trace_id)?;
        if status != ArbitrationStatus::Pending.as_str() {
            return Err(StorageError::InvalidArbitrationState {
                trace_id: trace_id.to_string(),
                expected: "PENDING",
                actual: status,
            });
        }
        tx.execute(
            "UPDATE durable_refusal_traces
             SET arbitration_status = 'ARBITRATED',
                 is_answerable = ?1,
                 gold_answer = ?2,
                 evidence_span = ?3,
                 contradiction_fact = ?4,
                 arbitrator_model = ?5,
                 arbitration_raw = ?6,
                 arbitrated_at_ms = ?7,
                 last_error = NULL
             WHERE trace_id = ?8",
            rusqlite::params![
                result.is_answerable,
                result.gold_answer,
                result.evidence_span,
                result.contradiction_fact,
                result.arbitrator_model,
                result.arbitration_raw,
                result.arbitrated_at_ms,
                trace_id,
            ],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Record a failed arbitration attempt on `trace_id`.
    ///
    /// Same `PENDING`-only guard as [`Self::complete_arbitration`]: a
    /// `REVOKED` row must not be flipped back to `PENDING` by a stray
    /// failure callback either, which would re-deliver it to the
    /// arbitration loop in contradiction of the rollback that revoked it.
    ///
    /// On success: `retry_count` is incremented, `last_error` is recorded,
    /// and `arbitration_status` stays/returns to `'PENDING'` so
    /// [`Self::fetch_pending_arbitration`] re-delivers it. There is no
    /// retry cap at this layer.
    pub fn fail_arbitration(&self, trace_id: &str, error: &str) -> Result<(), StorageError> {
        let conn = self.lock()?;
        let tx = conn.unchecked_transaction()?;
        let status = current_trace_status(&tx, trace_id)?;
        if status != ArbitrationStatus::Pending.as_str() {
            return Err(StorageError::InvalidArbitrationState {
                trace_id: trace_id.to_string(),
                expected: "PENDING",
                actual: status,
            });
        }
        tx.execute(
            "UPDATE durable_refusal_traces
             SET retry_count = retry_count + 1,
                 last_error = ?1,
                 arbitration_status = 'PENDING'
             WHERE trace_id = ?2",
            rusqlite::params![error, trace_id],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Mark every id in `trace_ids` as consumed by `patch_id`.
    ///
    /// Strict, all-or-nothing: for every id, the row must exist, be
    /// `ARBITRATED`, and not already consumed, or the whole call fails
    /// without applying any of it (checked before any write, inside one
    /// transaction) — this links training provenance to a patch, so
    /// silently no-op'ing a bad id would corrupt the provenance chain.
    pub fn mark_consumed(&self, trace_ids: &[&str], patch_id: &str) -> Result<(), StorageError> {
        let conn = self.lock()?;
        let tx = conn.unchecked_transaction()?;
        consume_traces_in_tx(&tx, trace_ids, patch_id)?;
        tx.commit()?;
        Ok(())
    }

    /// Record a new patch. `rollback_reason`/`rolled_back_at_ms` start NULL.
    ///
    /// A duplicate `patch_id` surfaces as the underlying SQLite constraint
    /// error (`StorageError::Sqlite`), not remapped.
    pub fn record_patch(&self, patch: &DurablePatchRecord) -> Result<(), StorageError> {
        let conn = self.lock()?;
        insert_patch(&conn, patch)
    }

    /// Insert `patch` and mark every id in `trace_ids` as consumed by it,
    /// as one transaction: both commit together or neither does.
    ///
    /// This is what a patch builder must use instead of calling
    /// [`Self::record_patch`] then [`Self::mark_consumed`]. As two
    /// transactions, a failed `mark_consumed` (e.g. one trace consumed by a
    /// concurrent builder) would leave a patch row whose traces are still
    /// unconsumed, and the next batch would train on them a second time.
    ///
    /// Fails with [`StorageError::InvalidPatchCommit`] if `trace_ids` is
    /// empty or its length differs from `patch.samples_count`, and with the
    /// same per-trace errors as [`Self::mark_consumed`].
    pub fn commit_patch(
        &self,
        patch: &DurablePatchRecord,
        trace_ids: &[&str],
    ) -> Result<(), StorageError> {
        if trace_ids.is_empty() {
            return Err(StorageError::InvalidPatchCommit(format!(
                "patch {} consumes no traces",
                patch.patch_id
            )));
        }
        if trace_ids.len() as i64 != patch.samples_count {
            return Err(StorageError::InvalidPatchCommit(format!(
                "patch {} declares samples_count {} but consumes {} traces",
                patch.patch_id,
                patch.samples_count,
                trace_ids.len()
            )));
        }
        let conn = self.lock()?;
        let tx = conn.unchecked_transaction()?;
        insert_patch(&tx, patch)?;
        consume_traces_in_tx(&tx, trace_ids, &patch.patch_id)?;
        tx.commit()?;
        Ok(())
    }

    /// Fetch one patch row by id, including its rollback columns. `None` if
    /// no such patch exists.
    pub fn get_patch(&self, patch_id: &str) -> Result<Option<DurablePatchRow>, StorageError> {
        let conn = self.lock()?;
        let sql = format!("SELECT {SELECT_PATCH_COLUMNS} FROM durable_patches WHERE patch_id = ?1");
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query([patch_id])?;
        match rows.next()? {
            Some(row) => Ok(Some(row_to_patch(row)?)),
            None => Ok(None),
        }
    }

    /// Fetch up to `limit` patches that are `ACTIVE` and have been deployed,
    /// most recently deployed first. A hot-reload manager restores its
    /// active patch from the first row and its snapshot from the second.
    pub fn fetch_deployed_active_patches(
        &self,
        limit: usize,
    ) -> Result<Vec<DurablePatchRow>, StorageError> {
        let conn = self.lock()?;
        let sql = format!(
            "SELECT {SELECT_PATCH_COLUMNS} FROM durable_patches
             WHERE status = 'ACTIVE' AND deployed_at_ms IS NOT NULL
             ORDER BY deployed_at_ms DESC, rowid DESC
             LIMIT ?1"
        );
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query([limit as i64])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_patch(row)?);
        }
        Ok(out)
    }

    /// Record that `patch_id` was loaded into a serving process at
    /// `deployed_at_ms`. Unknown patch fails with
    /// [`StorageError::DurablePatchNotFound`]; a patch that is not `ACTIVE`
    /// (for example already rolled back) fails with
    /// [`StorageError::PatchNotActive`], so a rolled-back patch can never be
    /// marked deployed again.
    pub fn mark_patch_deployed(
        &self,
        patch_id: &str,
        deployed_at_ms: i64,
    ) -> Result<(), StorageError> {
        let conn = self.lock()?;
        let tx = conn.unchecked_transaction()?;
        require_active_patch(&tx, patch_id)?;
        tx.execute(
            "UPDATE durable_patches SET deployed_at_ms = ?1 WHERE patch_id = ?2",
            rusqlite::params![deployed_at_ms, patch_id],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Append one canary observation for `metric.patch_id` and return its
    /// `sample_id`.
    ///
    /// The patch must exist and be `ACTIVE`: metrics for an unknown patch
    /// fail with [`StorageError::DurablePatchNotFound`], and metrics for a
    /// rolled-back patch fail with [`StorageError::PatchNotActive`]. The
    /// second case is how a serving process learns that another process
    /// rolled its patch back. A negative `latency_us` fails with
    /// [`StorageError::InvalidCanaryMetric`].
    pub fn record_canary_metric(&self, metric: &CanaryMetricInput) -> Result<i64, StorageError> {
        if metric.latency_us < 0 {
            return Err(StorageError::InvalidCanaryMetric(format!(
                "latency_us must be >= 0, got {}",
                metric.latency_us
            )));
        }
        let conn = self.lock()?;
        let tx = conn.unchecked_transaction()?;
        require_active_patch(&tx, &metric.patch_id)?;
        tx.execute(
            "INSERT INTO canary_metrics
                (patch_id, recorded_at_ms, is_fast_pass, is_refused, latency_us, is_hard_stop)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![
                metric.patch_id,
                metric.recorded_at_ms,
                metric.is_fast_pass,
                metric.is_refused,
                metric.latency_us,
                metric.is_hard_stop,
            ],
        )?;
        let sample_id = tx.last_insert_rowid();
        tx.commit()?;
        Ok(sample_id)
    }

    /// Fetch the `limit` most recent canary rows of `patch_id`, newest
    /// first. `limit == 0` fails with [`StorageError::InvalidCanaryMetric`].
    pub fn fetch_canary_metrics(
        &self,
        patch_id: &str,
        limit: usize,
    ) -> Result<Vec<CanaryMetricRow>, StorageError> {
        check_window(limit)?;
        let conn = self.lock()?;
        let mut stmt = conn.prepare(
            "SELECT sample_id, patch_id, recorded_at_ms, is_fast_pass, is_refused,
                    latency_us, is_hard_stop
             FROM canary_metrics WHERE patch_id = ?1
             ORDER BY sample_id DESC
             LIMIT ?2",
        )?;
        let mut rows = stmt.query(rusqlite::params![patch_id, limit as i64])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(CanaryMetricRow {
                sample_id: row.get(0)?,
                patch_id: row.get(1)?,
                recorded_at_ms: row.get(2)?,
                is_fast_pass: row.get::<_, i64>(3)? != 0,
                is_refused: row.get::<_, i64>(4)? != 0,
                latency_us: row.get(5)?,
                is_hard_stop: row.get::<_, i64>(6)? != 0,
            });
        }
        Ok(out)
    }

    /// Aggregate the `limit` most recent canary rows of `patch_id` in one
    /// SQL query. The window is the latest rows, not all history, so a
    /// patch that degrades late is still caught. An empty window yields
    /// `total_samples == 0` and every rate `0.0`; callers must gate on
    /// `total_samples` before trusting a rate.
    pub fn compute_canary_stats(
        &self,
        patch_id: &str,
        limit: usize,
    ) -> Result<CanaryStats, StorageError> {
        check_window(limit)?;
        let conn = self.lock()?;
        let (total, refused, fast_pass, hard_stop, latency_sum): (i64, i64, i64, i64, i64) = conn
            .query_row(
            "SELECT COUNT(*),
                        COALESCE(SUM(is_refused), 0),
                        COALESCE(SUM(is_fast_pass), 0),
                        COALESCE(SUM(is_hard_stop), 0),
                        COALESCE(SUM(latency_us), 0)
                 FROM (SELECT is_refused, is_fast_pass, is_hard_stop, latency_us
                       FROM canary_metrics WHERE patch_id = ?1
                       ORDER BY sample_id DESC LIMIT ?2)",
            rusqlite::params![patch_id, limit as i64],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )?;
        let rate = |n: i64| {
            if total == 0 {
                0.0
            } else {
                n as f64 / total as f64
            }
        };
        Ok(CanaryStats {
            patch_id: patch_id.to_string(),
            window: limit,
            total_samples: total as usize,
            refused_count: refused as usize,
            fast_pass_count: fast_pass as usize,
            hard_stop_count: hard_stop as usize,
            refusal_rate: rate(refused),
            fast_pass_rate: rate(fast_pass),
            hard_stop_rate: rate(hard_stop),
            avg_latency_us: rate(latency_sum),
        })
    }

    /// Roll back `patch_id` with `reason`, and revoke every refusal trace
    /// it consumed, as one atomic transaction (both writes commit together
    /// or neither does).
    ///
    /// Unknown `patch_id` fails with [`StorageError::DurablePatchNotFound`].
    /// A patch already `'ROLLED_BACK'` fails with
    /// [`StorageError::PatchAlreadyRolledBack`] rather than silently
    /// no-op'ing a double rollback.
    pub fn rollback_patch(&self, patch_id: &str, reason: &str) -> Result<(), StorageError> {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| StorageError::Serialization(format!("system clock before epoch: {e}")))?
            .as_millis() as i64;
        let conn = self.lock()?;
        let tx = conn.unchecked_transaction()?;
        let status: Option<String> = tx
            .query_row(
                "SELECT status FROM durable_patches WHERE patch_id = ?1",
                [patch_id],
                |row| row.get(0),
            )
            .optional()?;
        let status = match status {
            None => return Err(StorageError::DurablePatchNotFound(patch_id.to_string())),
            Some(s) => s,
        };
        if status == "ROLLED_BACK" {
            return Err(StorageError::PatchAlreadyRolledBack(patch_id.to_string()));
        }
        tx.execute(
            "UPDATE durable_patches
             SET status = 'ROLLED_BACK', rollback_reason = ?1, rolled_back_at_ms = ?2
             WHERE patch_id = ?3",
            rusqlite::params![reason, now_ms, patch_id],
        )?;
        tx.execute(
            "UPDATE durable_refusal_traces SET arbitration_status = 'REVOKED'
             WHERE applied_patch_id = ?1",
            [patch_id],
        )?;
        tx.commit()?;
        Ok(())
    }
}

/// Shared body of [`DurableRefusalStore::mark_consumed`] and
/// [`DurableRefusalStore::commit_patch`]: check every id first (exists,
/// `ARBITRATED`, not consumed), then write. The caller owns the transaction.
fn consume_traces_in_tx(
    tx: &rusqlite::Transaction<'_>,
    trace_ids: &[&str],
    patch_id: &str,
) -> Result<(), StorageError> {
    for id in trace_ids {
        let row: Option<(String, i64)> = tx
            .query_row(
                "SELECT arbitration_status, is_consumed FROM durable_refusal_traces
                 WHERE trace_id = ?1",
                [*id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        match row {
            None => return Err(StorageError::DurableTraceNotFound(id.to_string())),
            Some((status, is_consumed)) => {
                if status != ArbitrationStatus::Arbitrated.as_str() {
                    return Err(StorageError::InvalidArbitrationState {
                        trace_id: id.to_string(),
                        expected: "ARBITRATED",
                        actual: status,
                    });
                }
                if is_consumed != 0 {
                    return Err(StorageError::TraceAlreadyConsumed(id.to_string()));
                }
            }
        }
    }
    for id in trace_ids {
        tx.execute(
            "UPDATE durable_refusal_traces SET is_consumed = 1, applied_patch_id = ?1
             WHERE trace_id = ?2",
            rusqlite::params![patch_id, *id],
        )?;
    }
    Ok(())
}

/// Shared body of [`DurableRefusalStore::record_patch`] and
/// [`DurableRefusalStore::commit_patch`]. A `Transaction` derefs to a
/// `Connection`, so this runs inside or outside one.
fn insert_patch(
    conn: &rusqlite::Connection,
    patch: &DurablePatchRecord,
) -> Result<(), StorageError> {
    conn.execute(
        "INSERT INTO durable_patches
            (patch_id, created_at_ms, base_model_hash, patch_file_path, sha256,
             samples_count, validation_status, status, deployed_at_ms)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        rusqlite::params![
            patch.patch_id,
            patch.created_at_ms,
            patch.base_model_hash,
            patch.patch_file_path,
            patch.sha256,
            patch.samples_count,
            patch.validation_status,
            patch.status,
            patch.deployed_at_ms,
        ],
    )?;
    Ok(())
}

/// Fail unless `patch_id` exists and is `ACTIVE`. The caller owns the
/// transaction, so the check and the write that depends on it are atomic.
fn require_active_patch(
    tx: &rusqlite::Transaction<'_>,
    patch_id: &str,
) -> Result<(), StorageError> {
    let status: Option<String> = tx
        .query_row(
            "SELECT status FROM durable_patches WHERE patch_id = ?1",
            [patch_id],
            |row| row.get(0),
        )
        .optional()?;
    match status {
        None => Err(StorageError::DurablePatchNotFound(patch_id.to_string())),
        Some(s) if s == "ACTIVE" => Ok(()),
        Some(s) => Err(StorageError::PatchNotActive {
            patch_id: patch_id.to_string(),
            status: s,
        }),
    }
}

fn check_window(limit: usize) -> Result<(), StorageError> {
    if limit == 0 {
        return Err(StorageError::InvalidCanaryMetric(
            "window limit must be > 0".to_string(),
        ));
    }
    Ok(())
}

/// Build one [`DurablePatchRow`] from a row selected with
/// [`SELECT_PATCH_COLUMNS`] column order.
fn row_to_patch(row: &rusqlite::Row<'_>) -> Result<DurablePatchRow, StorageError> {
    Ok(DurablePatchRow {
        patch_id: row.get(0)?,
        created_at_ms: row.get(1)?,
        base_model_hash: row.get(2)?,
        patch_file_path: row.get(3)?,
        sha256: row.get(4)?,
        samples_count: row.get(5)?,
        validation_status: row.get(6)?,
        status: row.get(7)?,
        rollback_reason: row.get(8)?,
        rolled_back_at_ms: row.get(9)?,
        deployed_at_ms: row.get(10)?,
    })
}

/// Look up the current `arbitration_status` of `trace_id` inside an
/// in-progress transaction. Shared by [`DurableRefusalStore::complete_arbitration`]
/// and [`DurableRefusalStore::fail_arbitration`], which both apply the same
/// `PENDING`-only guard.
fn current_trace_status(
    tx: &rusqlite::Transaction<'_>,
    trace_id: &str,
) -> Result<String, StorageError> {
    tx.query_row(
        "SELECT arbitration_status FROM durable_refusal_traces WHERE trace_id = ?1",
        [trace_id],
        |row| row.get(0),
    )
    .optional()?
    .ok_or_else(|| StorageError::DurableTraceNotFound(trace_id.to_string()))
}

/// Build one [`DurableRefusalTrace`] from a row selected with
/// [`SELECT_TRACE_COLUMNS`] column order.
fn row_to_trace(row: &rusqlite::Row<'_>) -> Result<DurableRefusalTrace, StorageError> {
    let status_str: String = row.get(9)?;
    let arbitration_status = ArbitrationStatus::from_str(&status_str)?;
    let is_answerable: Option<i64> = row.get(10)?;
    let is_consumed: i64 = row.get(19)?;
    Ok(DurableRefusalTrace {
        trace_id: row.get(0)?,
        created_at_ms: row.get(1)?,
        context: row.get(2)?,
        question: row.get(3)?,
        candidate: row.get(4)?,
        best_span_score: row.get(5)?,
        null_score: row.get(6)?,
        score_diff: row.get(7)?,
        verifier_output: row.get(8)?,
        arbitration_status,
        is_answerable: is_answerable.map(|v| v != 0),
        gold_answer: row.get(11)?,
        evidence_span: row.get(12)?,
        contradiction_fact: row.get(13)?,
        arbitrator_model: row.get(14)?,
        arbitration_raw: row.get(15)?,
        arbitrated_at_ms: row.get(16)?,
        retry_count: row.get(17)?,
        last_error: row.get(18)?,
        is_consumed: is_consumed != 0,
        applied_patch_id: row.get(20)?,
    })
}

/// Enforce that `table`'s actual columns, in order, match `expected`
/// exactly. `CREATE TABLE IF NOT EXISTS` alone would happily no-op against
/// a pre-existing table of any shape; this makes a stale or hand-edited
/// schema fail closed instead of silently accepting writes into missing or
/// misordered columns.
fn check_columns(
    conn: &rusqlite::Connection,
    table: &'static str,
    expected_cols: &[&str],
) -> Result<(), StorageError> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let actual: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<Result<_, _>>()?;
    let expected: Vec<String> = expected_cols.iter().map(|s| s.to_string()).collect();
    if actual != expected {
        return Err(StorageError::DurableSchemaMismatch {
            table,
            expected,
            actual,
        });
    }
    Ok(())
}
