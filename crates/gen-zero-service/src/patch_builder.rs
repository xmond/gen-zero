//! Phase 3 of the closed-loop self-evolving pipeline: compile arbitrated
//! refusal traces into a content-addressed training-sample bundle (a
//! "patch" file) and record its provenance in
//! [`gen_zero_storage::DurableRefusalStore`].
//!
//! Scope, stated plainly: this module does not train anything and does not
//! change any model weights. It turns a batch of `ARBITRATED`, unconsumed
//! traces into a JSON file of [`TrainingSamplePair`]s, verifies the file on
//! disk, then records the patch row and marks the traces consumed in one
//! SQLite transaction. A fine-tuning job is the intended reader of the file;
//! no such job exists in this repository yet.
//!
//! Order of operations in [`PatchBuilder::run_once`] is the safety property:
//! nothing is written to SQLite until the file has been written, re-read and
//! verified. If any step fails, the file is deleted and no trace is marked
//! consumed, so the next run picks the same traces up again.
//!
//! The daemon mirrors [`crate::arbitrator::spawn_arbitrator_daemon`]: the
//! same `watch::Receiver<bool>` shutdown pattern, and per-tick errors are
//! logged, never allowed to kill the loop.

use gen_zero_storage::{DurablePatchRecord, DurableRefusalStore, DurableRefusalTrace};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::watch;

pub const DEFAULT_BATCH_SIZE: usize = 50;
pub const DEFAULT_BASE_MODEL_HASH: &str = "sha256:baseline-0.5b-v1";
pub const DEFAULT_POLL_INTERVAL_SECS: u64 = 60;

/// Wiring for the patch builder: where the durable store lives, where patch
/// files go, and batch/poll sizing.
#[derive(Clone, Debug)]
pub struct PatchBuildConfig {
    /// Durable refusal database. Used by [`PatchBuilder::open`]; a caller
    /// that already holds a store uses [`PatchBuilder::new`] instead.
    pub db_path: PathBuf,
    pub output_dir: PathBuf,
    pub batch_size: usize,
    pub base_model_hash: String,
    /// Daemon tick. Each tick drains every full batch available.
    pub poll_interval_secs: u64,
}

impl PatchBuildConfig {
    pub fn new(db_path: impl Into<PathBuf>, output_dir: impl Into<PathBuf>) -> Self {
        Self {
            db_path: db_path.into(),
            output_dir: output_dir.into(),
            batch_size: DEFAULT_BATCH_SIZE,
            base_model_hash: DEFAULT_BASE_MODEL_HASH.to_string(),
            poll_interval_secs: DEFAULT_POLL_INTERVAL_SECS,
        }
    }
}

/// One supervised training example derived from one arbitrated trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingSamplePair {
    pub trace_id: String,
    pub context: String,
    pub question: String,
    pub is_answerable: bool,
    pub gold_answer: Option<String>,
    pub evidence_span: Option<String>,
    pub contradiction_fact: Option<String>,
}

/// The patch file, as written to `output_dir/<patch_id>.json`.
///
/// `sha256` is the hex SHA-256 of the canonical body: the `serde_json`
/// bytes of `{base_model_hash, created_at_ms, samples_count, samples}` in
/// that field order (see [`body_sha256`]). It cannot cover the whole file,
/// because the file contains `sha256`, `patch_id` (derived from it) and
/// `patch_file_path` (derived from `patch_id`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledPatch {
    pub patch_id: String,
    pub base_model_hash: String,
    pub created_at_ms: i64,
    pub samples_count: usize,
    pub sha256: String,
    pub patch_file_path: String,
    pub samples: Vec<TrainingSamplePair>,
}

#[derive(Debug, thiserror::Error)]
pub enum PatchBuilderError {
    #[error("storage error: {0}")]
    Storage(#[from] gen_zero_storage::StorageError),
    #[error("I/O error on {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("patch JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("arbitrated trace {trace_id} cannot become a training sample: {reason}")]
    CorruptTrace { trace_id: String, reason: String },
    #[error("patch validation failed for {path}: {reason}")]
    Validation { path: PathBuf, reason: String },
    #[error("patch path is not valid UTF-8: {0}")]
    NonUtf8Path(PathBuf),
    #[error("system clock is before the UNIX epoch: {0}")]
    Clock(String),
    #[error("patch builder config invalid: {0}")]
    Config(String),
}

impl PatchBuilderError {
    fn io(path: &Path, source: std::io::Error) -> Self {
        Self::Io {
            path: path.to_path_buf(),
            source,
        }
    }
}

/// Field order here is the hash contract. Do not reorder.
#[derive(Serialize)]
struct PatchBody<'a> {
    base_model_hash: &'a str,
    created_at_ms: i64,
    samples_count: usize,
    samples: &'a [TrainingSamplePair],
}

/// Hex SHA-256 of the canonical patch body. See [`CompiledPatch`].
pub fn body_sha256(
    base_model_hash: &str,
    created_at_ms: i64,
    samples: &[TrainingSamplePair],
) -> Result<String, PatchBuilderError> {
    let body = PatchBody {
        base_model_hash,
        created_at_ms,
        samples_count: samples.len(),
        samples,
    };
    let bytes = serde_json::to_vec(&body)?;
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

fn patch_id_for(sha256: &str) -> String {
    format!("patch_{}", &sha256[..16])
}

/// Re-read the patch file at `path` and check it against itself: the body
/// hash matches the header `sha256`, `patch_id` is derived from that hash,
/// `patch_file_path` names this file, and `samples_count` equals the number
/// of samples and is non-zero. Returns the parsed patch on success.
pub fn validate_patch(path: &Path) -> Result<CompiledPatch, PatchBuilderError> {
    let fail = |reason: String| PatchBuilderError::Validation {
        path: path.to_path_buf(),
        reason,
    };
    let bytes = std::fs::read(path).map_err(|e| PatchBuilderError::io(path, e))?;
    let patch: CompiledPatch =
        serde_json::from_slice(&bytes).map_err(|e| fail(format!("not a patch file: {e}")))?;
    if patch.samples.is_empty() {
        return Err(fail("patch has no samples".to_string()));
    }
    if patch.samples_count != patch.samples.len() {
        return Err(fail(format!(
            "header samples_count {} but file holds {} samples",
            patch.samples_count,
            patch.samples.len()
        )));
    }
    let actual = body_sha256(&patch.base_model_hash, patch.created_at_ms, &patch.samples)?;
    if actual != patch.sha256 {
        return Err(fail(format!(
            "sha256 mismatch: header {}, recomputed {actual}",
            patch.sha256
        )));
    }
    let expected_id = patch_id_for(&actual);
    if patch.patch_id != expected_id {
        return Err(fail(format!(
            "patch_id {} does not match sha256 (expected {expected_id})",
            patch.patch_id
        )));
    }
    let path_str = path
        .to_str()
        .ok_or_else(|| PatchBuilderError::NonUtf8Path(path.to_path_buf()))?;
    if patch.patch_file_path != path_str {
        return Err(fail(format!(
            "header patch_file_path {} does not name this file",
            patch.patch_file_path
        )));
    }
    Ok(patch)
}

/// Convert one arbitrated trace into a training sample, failing closed on
/// any row the arbitrator could not have written (see
/// `arbitrator::judge_one`): a NULL verdict, or a verdict missing the field
/// that justifies it.
fn to_sample(trace: &DurableRefusalTrace) -> Result<TrainingSamplePair, PatchBuilderError> {
    let corrupt = |reason: &str| PatchBuilderError::CorruptTrace {
        trace_id: trace.trace_id.clone(),
        reason: reason.to_string(),
    };
    let is_answerable = trace
        .is_answerable
        .ok_or_else(|| corrupt("ARBITRATED row has NULL is_answerable"))?;
    let non_empty = |v: &Option<String>| v.as_deref().is_some_and(|s| !s.trim().is_empty());
    if is_answerable {
        if !non_empty(&trace.gold_answer) {
            return Err(corrupt("answerable verdict without gold_answer"));
        }
        if !non_empty(&trace.evidence_span) {
            return Err(corrupt("answerable verdict without evidence_span"));
        }
    } else if !non_empty(&trace.contradiction_fact) {
        return Err(corrupt("unanswerable verdict without contradiction_fact"));
    }
    Ok(TrainingSamplePair {
        trace_id: trace.trace_id.clone(),
        context: trace.context.clone(),
        question: trace.question.clone(),
        is_answerable,
        gold_answer: trace.gold_answer.clone(),
        evidence_span: trace.evidence_span.clone(),
        contradiction_fact: trace.contradiction_fact.clone(),
    })
}

fn now_ms() -> Result<i64, PatchBuilderError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .map_err(|e| PatchBuilderError::Clock(e.to_string()))
}

/// Write `bytes` to `path` via a sibling temp file, fsync, then rename, so a
/// crash never leaves a half-written patch under its final name. Refuses to
/// replace an existing file.
fn write_new_file(path: &Path, bytes: &[u8]) -> Result<(), PatchBuilderError> {
    if path.exists() {
        return Err(PatchBuilderError::io(
            path,
            std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "patch file already exists",
            ),
        ));
    }
    let tmp = path.with_extension("json.tmp");
    let result = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        std::fs::rename(&tmp, path)
    })();
    if let Err(e) = result {
        if let Err(cleanup) = std::fs::remove_file(&tmp) {
            if cleanup.kind() != std::io::ErrorKind::NotFound {
                tracing::error!(path = %tmp.display(), error = %cleanup, "failed to remove temp patch file");
            }
        }
        return Err(PatchBuilderError::io(path, e));
    }
    Ok(())
}

/// Delete a patch file whose batch did not commit, so `output_dir` never
/// holds a file with no `durable_patches` row behind it.
fn discard_patch_file(path: &Path, cause: &PatchBuilderError) {
    tracing::error!(path = %path.display(), error = %cause, "patch build failed after file write; deleting file, no trace consumed");
    if let Err(e) = std::fs::remove_file(path) {
        tracing::error!(path = %path.display(), error = %e, "failed to delete uncommitted patch file");
    }
}

pub struct PatchBuilder {
    config: PatchBuildConfig,
    store: Arc<DurableRefusalStore>,
}

impl PatchBuilder {
    pub fn new(config: PatchBuildConfig, store: Arc<DurableRefusalStore>) -> Self {
        Self { config, store }
    }

    /// Open the store at `config.db_path` and build on it.
    pub fn open(config: PatchBuildConfig) -> Result<Self, PatchBuilderError> {
        let store = DurableRefusalStore::new(&config.db_path)?;
        Ok(Self::new(config, Arc::new(store)))
    }

    pub fn config(&self) -> &PatchBuildConfig {
        &self.config
    }

    /// Compile at most one batch. `Ok(None)` means there was nothing to
    /// consume. On `Err`, no trace has been marked consumed and no patch
    /// file is left behind.
    pub fn run_once(&self) -> Result<Option<CompiledPatch>, PatchBuilderError> {
        if self.config.batch_size == 0 {
            return Err(PatchBuilderError::Config(
                "batch_size must be > 0".to_string(),
            ));
        }
        let traces = self
            .store
            .fetch_unconsumed_arbitrated(self.config.batch_size)?;
        if traces.is_empty() {
            return Ok(None);
        }
        let samples = traces
            .iter()
            .map(to_sample)
            .collect::<Result<Vec<_>, _>>()?;

        let created_at_ms = now_ms()?;
        let sha256 = body_sha256(&self.config.base_model_hash, created_at_ms, &samples)?;
        let patch_id = patch_id_for(&sha256);

        std::fs::create_dir_all(&self.config.output_dir)
            .map_err(|e| PatchBuilderError::io(&self.config.output_dir, e))?;
        // The path is recorded in SQLite and read by other processes, so it
        // must not depend on this process's working directory.
        let output_dir = std::path::absolute(&self.config.output_dir)
            .map_err(|e| PatchBuilderError::io(&self.config.output_dir, e))?;
        let file_path = output_dir.join(format!("{patch_id}.json"));
        let file_path_str = file_path
            .to_str()
            .ok_or_else(|| PatchBuilderError::NonUtf8Path(file_path.clone()))?
            .to_string();

        let compiled = CompiledPatch {
            patch_id: patch_id.clone(),
            base_model_hash: self.config.base_model_hash.clone(),
            created_at_ms,
            samples_count: samples.len(),
            sha256: sha256.clone(),
            patch_file_path: file_path_str.clone(),
            samples,
        };
        write_new_file(&file_path, &serde_json::to_vec_pretty(&compiled)?)?;

        if let Err(e) = self.validate_and_commit(&compiled, &file_path) {
            discard_patch_file(&file_path, &e);
            return Err(e);
        }
        tracing::info!(
            patch_id = %compiled.patch_id,
            samples = compiled.samples_count,
            path = %compiled.patch_file_path,
            "patch compiled and committed"
        );
        Ok(Some(compiled))
    }

    fn validate_and_commit(
        &self,
        compiled: &CompiledPatch,
        file_path: &Path,
    ) -> Result<(), PatchBuilderError> {
        let on_disk = validate_patch(file_path)?;
        if &on_disk != compiled {
            return Err(PatchBuilderError::Validation {
                path: file_path.to_path_buf(),
                reason: "file content differs from the patch compiled in memory".to_string(),
            });
        }
        let record = DurablePatchRecord {
            patch_id: compiled.patch_id.clone(),
            created_at_ms: compiled.created_at_ms,
            base_model_hash: compiled.base_model_hash.clone(),
            patch_file_path: compiled.patch_file_path.clone(),
            sha256: compiled.sha256.clone(),
            samples_count: compiled.samples_count as i64,
            validation_status: "PASSED".to_string(),
            // ACTIVE = current, not rolled back. Nothing loads this file
            // into a model yet, so no deployment time is claimed.
            status: "ACTIVE".to_string(),
            deployed_at_ms: None,
        };
        let trace_ids: Vec<&str> = compiled
            .samples
            .iter()
            .map(|s| s.trace_id.as_str())
            .collect();
        self.store.commit_patch(&record, &trace_ids)?;
        Ok(())
    }
}

/// Spawn the background patch builder. Each tick drains every batch that is
/// available (calling [`PatchBuilder::run_once`] until it returns
/// `Ok(None)` or an error). The blocking SQLite and file work runs on the
/// blocking pool. Per-tick errors are logged; the loop keeps running until
/// `shutdown` reports `true` or its sender is dropped.
pub fn spawn_patch_builder_daemon(
    builder: PatchBuilder,
    mut shutdown: watch::Receiver<bool>,
) -> tokio::task::JoinHandle<()> {
    let builder = Arc::new(builder);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(
            builder.config.poll_interval_secs.max(1),
        ));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            // Same pattern as `arbitrator::spawn_arbitrator_daemon`.
            if *shutdown.borrow() {
                tracing::info!("patch builder daemon shutting down");
                break;
            }
            tokio::select! {
                biased;
                changed = shutdown.changed() => {
                    if changed.is_err() {
                        tracing::info!("patch builder daemon shutting down (sender dropped)");
                        break;
                    }
                }
                _ = interval.tick() => {
                    let worker = Arc::clone(&builder);
                    let stop = shutdown.clone();
                    match tokio::task::spawn_blocking(move || drain(&worker, &stop)).await {
                        Ok(Ok(0)) => tracing::debug!("patch builder: nothing to consume"),
                        Ok(Ok(n)) => tracing::info!(patches = n, "patch builder tick complete"),
                        Ok(Err(error)) => tracing::warn!(%error, "patch build failed"),
                        Err(error) => tracing::error!(%error, "patch builder worker panicked"),
                    }
                }
            }
        }
    })
}

/// Build batches until none is left, stopping early between batches once
/// shutdown is requested so a large backlog does not delay exit.
fn drain(
    builder: &PatchBuilder,
    shutdown: &watch::Receiver<bool>,
) -> Result<usize, PatchBuilderError> {
    let mut built = 0;
    while !*shutdown.borrow() && builder.run_once()?.is_some() {
        built += 1;
    }
    Ok(built)
}
