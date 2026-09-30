//! Native Rust closed-loop subsystem: runtime feedback synchronization and
//! autonomous tuning-patch polling against the external tuning server
//! (`ai-server`, `http://100.102.231.124:8099` or
//! `https://tuning-zero.dx-app.site`).
//!
//! The edge runtime on Windows (`aws-win`) must not depend on Python for this
//! loop: both directions run as plain Tokio async tasks using `reqwest`.
//!
//! - [`spawn_feedback_syncer`] periodically drains [`FeedbackBuffer`] and
//!   POSTs batches to `<tuning_endpoint>/api/v1/feedback`. A failed POST
//!   requeues the batch instead of dropping it, so a transient outage on the
//!   tuning server never silently loses feedback.
//! - [`spawn_patch_poller`] periodically GETs
//!   `<tuning_endpoint>/api/v1/patch/latest`, and downloads + verifies a new
//!   patch by SHA-256 before writing it to `models_dir` and publishing the
//!   new version through an `ArcSwap`. A checksum mismatch is a hard refusal
//!   (logged, not applied), never a silent partial write.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;

/// One runtime feedback observation queued for the tuning server.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FeedbackRecord {
    pub trace_id: String,
    pub task: String,
    pub input_features: Vec<f32>,
    pub label: i64,
    pub timestamp: f64,
}

/// Closed-loop wiring: where the tuning server lives, how to authenticate to
/// it, and how often to talk to it. `tuning_endpoint: None` means the closed
/// loop is disabled; callers must not spawn the background tasks in that case.
#[derive(Clone, Debug)]
pub struct ClosedLoopConfig {
    pub tuning_endpoint: Option<String>,
    pub tuning_token: Option<String>,
    pub task: String,
    pub sync_interval_secs: u64,
    pub poll_interval_secs: u64,
    pub models_dir: PathBuf,
}

impl Default for ClosedLoopConfig {
    fn default() -> Self {
        Self {
            tuning_endpoint: None,
            tuning_token: None,
            task: "synthetic".to_string(),
            sync_interval_secs: 30,
            poll_interval_secs: 60,
            models_dir: PathBuf::from("./models"),
        }
    }
}

/// Bounded in-memory feedback queue. Oldest records are dropped once the
/// buffer is full: losing the oldest, least-actionable feedback under
/// sustained overload is preferable to unbounded memory growth.
pub const MAX_FEEDBACK_BUFFER_LEN: usize = 5_000;

pub struct FeedbackBuffer {
    inner: Mutex<VecDeque<FeedbackRecord>>,
    capacity: usize,
}

impl Default for FeedbackBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl FeedbackBuffer {
    pub fn new() -> Self {
        Self::with_capacity(MAX_FEEDBACK_BUFFER_LEN)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(VecDeque::with_capacity(capacity.min(1024))),
            capacity,
        }
    }

    /// Queue one record, evicting the oldest entry first if the buffer is full.
    pub fn record(&self, record: FeedbackRecord) {
        let mut buf = self.inner.lock();
        if buf.len() >= self.capacity {
            buf.pop_front();
        }
        buf.push_back(record);
    }

    /// Remove and return up to `max` records, oldest first.
    pub fn drain(&self, max: usize) -> Vec<FeedbackRecord> {
        let mut buf = self.inner.lock();
        let n = max.min(buf.len());
        buf.drain(..n).collect()
    }

    /// Return previously drained records to the front of the queue (oldest
    /// first) after a failed delivery, so the next sync attempt retries them
    /// before newer records. If this overflows capacity, the newest records
    /// (the ones that arrived after the failed batch was drained) are
    /// dropped first, since the requeued batch is the one already in flight.
    pub fn requeue(&self, records: Vec<FeedbackRecord>) {
        let mut buf = self.inner.lock();
        for record in records.into_iter().rev() {
            buf.push_front(record);
        }
        while buf.len() > self.capacity {
            buf.pop_back();
        }
    }

    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Tuning-server-side latest-patch metadata (`GET /api/v1/patch/latest`).
/// Every field beyond the checksum is informational; `target_sha256` (falling
/// back to `sha256`) is the only value the poller acts on.
#[derive(Debug, Deserialize)]
struct PatchMetadata {
    #[serde(default)]
    sha256: Option<String>,
    #[serde(default)]
    target_sha256: Option<String>,
}

impl PatchMetadata {
    fn effective_sha256(&self) -> Option<&str> {
        self.target_sha256
            .as_deref()
            .or(self.sha256.as_deref())
            .filter(|s| !s.is_empty())
    }
}

#[derive(Debug, thiserror::Error)]
enum ClosedLoopError {
    #[error("network transport error: {0}")]
    Transport(String),
    #[error("tuning server returned HTTP {0}")]
    Status(u16),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    #[error("patch checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Periodically drain [`FeedbackBuffer`] and POST batches of up to 500
/// records to `<tuning_endpoint>/api/v1/feedback`. Runs until `shutdown`
/// reports `true` (or its sender is dropped).
pub fn spawn_feedback_syncer(
    config: ClosedLoopConfig,
    buffer: Arc<FeedbackBuffer>,
    mut shutdown: watch::Receiver<bool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let Some(endpoint) = config.tuning_endpoint.clone() else {
            tracing::warn!("feedback syncer started without a tuning endpoint configured; exiting");
            return;
        };
        let client = reqwest::Client::new();
        let url = format!("{}/api/v1/feedback", endpoint.trim_end_matches('/'));
        let mut interval =
            tokio::time::interval(Duration::from_secs(config.sync_interval_secs.max(1)));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            // Checked before each select so a shutdown signal sent before
            // this task subscribed (or between ticks) is never missed: a
            // `watch::Receiver` only wakes on the *next* change, not on the
            // value already current when it subscribed.
            if *shutdown.borrow() {
                tracing::info!("feedback syncer shutting down");
                break;
            }
            tokio::select! {
                biased;
                // `watch::Ref` (wait_for's output) is `!Send`, which would
                // make this spawned future `!Send` if held across the
                // `.await` below. `changed()` returns a plain `Result` with
                // no guard, so this select arm stays Send-safe; the fresh
                // value is read via the plain `borrow()` above on the next
                // loop iteration instead of from this arm's output.
                changed = shutdown.changed() => {
                    if changed.is_err() {
                        tracing::info!("feedback syncer shutting down (sender dropped)");
                        break;
                    }
                }
                _ = interval.tick() => {
                    let records = buffer.drain(500);
                    if records.is_empty() {
                        continue;
                    }
                    let count = records.len();
                    let mut req = client.post(&url).json(&records);
                    if let Some(token) = &config.tuning_token {
                        req = req.bearer_auth(token);
                    }
                    match req.send().await {
                        Ok(resp) if resp.status().is_success() => {
                            tracing::debug!(count, "synced feedback records to tuning server");
                        }
                        Ok(resp) => {
                            let status = resp.status();
                            let body = resp.text().await.unwrap_or_default();
                            tracing::warn!(
                                %status, count, body = %body.chars().take(512).collect::<String>(),
                                "feedback sync rejected by tuning server; requeuing"
                            );
                            buffer.requeue(records);
                        }
                        Err(error) => {
                            tracing::warn!(%error, count, "feedback sync network error; requeuing");
                            buffer.requeue(records);
                        }
                    }
                }
            }
        }
    })
}

/// Periodically GET `<tuning_endpoint>/api/v1/patch/latest`; when the target
/// checksum differs from `current_version`, download, verify by SHA-256, and
/// write `<models_dir>/<target_sha256>.npz`. Runs until `shutdown` reports
/// `true` (or its sender is dropped).
pub fn spawn_patch_poller(
    config: ClosedLoopConfig,
    current_version: Arc<arc_swap::ArcSwap<String>>,
    mut shutdown: watch::Receiver<bool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let Some(endpoint) = config.tuning_endpoint.clone() else {
            tracing::warn!("patch poller started without a tuning endpoint configured; exiting");
            return;
        };
        let endpoint = endpoint.trim_end_matches('/').to_string();
        let client = reqwest::Client::new();
        let mut interval =
            tokio::time::interval(Duration::from_secs(config.poll_interval_secs.max(1)));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            // See the matching comment in `spawn_feedback_syncer`: checked
            // up front so a shutdown already current at subscribe time is
            // never missed, and `changed()` (not `wait_for`) is used in the
            // select below to keep this spawned future `Send`.
            if *shutdown.borrow() {
                tracing::info!("patch poller shutting down");
                break;
            }
            tokio::select! {
                biased;
                changed = shutdown.changed() => {
                    if changed.is_err() {
                        tracing::info!("patch poller shutting down (sender dropped)");
                        break;
                    }
                }
                _ = interval.tick() => {
                    if let Err(error) = poll_and_apply_patch(&client, &endpoint, &config, &current_version).await {
                        tracing::warn!(%error, "tuning patch poll failed");
                    }
                }
            }
        }
    })
}

async fn poll_and_apply_patch(
    client: &reqwest::Client,
    endpoint: &str,
    config: &ClosedLoopConfig,
    current_version: &Arc<arc_swap::ArcSwap<String>>,
) -> Result<(), ClosedLoopError> {
    let meta_url = format!("{endpoint}/api/v1/patch/latest");
    let mut req = client.get(&meta_url).query(&[("task", &config.task)]);
    if let Some(token) = &config.tuning_token {
        req = req.bearer_auth(token);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| ClosedLoopError::Transport(e.to_string()))?;
    if !resp.status().is_success() {
        return Err(ClosedLoopError::Status(resp.status().as_u16()));
    }
    let meta: PatchMetadata = resp
        .json()
        .await
        .map_err(|e| ClosedLoopError::InvalidResponse(e.to_string()))?;
    let Some(target_sha256) = meta.effective_sha256().map(str::to_owned) else {
        return Err(ClosedLoopError::InvalidResponse(
            "patch metadata is missing both target_sha256 and sha256".into(),
        ));
    };

    if current_version.load().as_str() == target_sha256 {
        tracing::debug!(version = %target_sha256, "tuning patch already current");
        return Ok(());
    }

    let download_url = format!("{endpoint}/api/v1/patch/latest/download");
    let mut req = client.get(&download_url).query(&[("task", &config.task)]);
    if let Some(token) = &config.tuning_token {
        req = req.bearer_auth(token);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| ClosedLoopError::Transport(e.to_string()))?;
    if !resp.status().is_success() {
        return Err(ClosedLoopError::Status(resp.status().as_u16()));
    }
    let header_sha256 = resp
        .headers()
        .get("X-SHA256")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let downloaded = resp
        .bytes()
        .await
        .map_err(|e| ClosedLoopError::Transport(e.to_string()))?;

    let mut hasher = Sha256::new();
    hasher.update(&downloaded);
    let actual_sha256 = hex_encode(&hasher.finalize());

    if !actual_sha256.eq_ignore_ascii_case(&target_sha256) {
        return Err(ClosedLoopError::ChecksumMismatch {
            expected: target_sha256,
            actual: actual_sha256,
        });
    }
    if let Some(header_sha256) = &header_sha256 {
        if !header_sha256.eq_ignore_ascii_case(&actual_sha256) {
            return Err(ClosedLoopError::ChecksumMismatch {
                expected: header_sha256.clone(),
                actual: actual_sha256,
            });
        }
    }

    tokio::fs::create_dir_all(&config.models_dir).await?;
    let out_path = config.models_dir.join(format!("{target_sha256}.npz"));
    tokio::fs::write(&out_path, &downloaded).await?;
    current_version.store(Arc::new(target_sha256.clone()));
    tracing::info!(
        version = %target_sha256,
        size = downloaded.len(),
        "successfully applied new tuning patch"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(id: &str) -> FeedbackRecord {
        FeedbackRecord {
            trace_id: id.to_string(),
            task: "synthetic".to_string(),
            input_features: vec![0.1, -0.2],
            label: 0,
            timestamp: 1727700000.0,
        }
    }

    #[test]
    fn record_and_drain_preserve_fifo_order() {
        let buffer = FeedbackBuffer::new();
        buffer.record(record("a"));
        buffer.record(record("b"));
        buffer.record(record("c"));
        assert_eq!(buffer.len(), 3);
        let drained = buffer.drain(2);
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].trace_id, "a");
        assert_eq!(drained[1].trace_id, "b");
        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn buffer_evicts_oldest_once_full() {
        let buffer = FeedbackBuffer::with_capacity(2);
        buffer.record(record("a"));
        buffer.record(record("b"));
        buffer.record(record("c"));
        assert_eq!(buffer.len(), 2);
        let drained = buffer.drain(2);
        assert_eq!(drained[0].trace_id, "b");
        assert_eq!(drained[1].trace_id, "c");
    }

    #[test]
    fn requeue_puts_records_back_in_original_order_ahead_of_newer_ones() {
        let buffer = FeedbackBuffer::new();
        let batch = vec![record("a"), record("b")];
        buffer.record(record("c"));
        buffer.requeue(batch);
        let drained = buffer.drain(3);
        assert_eq!(
            drained
                .iter()
                .map(|r| r.trace_id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );
    }

    #[test]
    fn requeue_drops_newest_records_when_over_capacity() {
        let buffer = FeedbackBuffer::with_capacity(2);
        buffer.record(record("c"));
        buffer.requeue(vec![record("a"), record("b")]);
        assert_eq!(buffer.len(), 2);
        let drained = buffer.drain(2);
        assert_eq!(
            drained
                .iter()
                .map(|r| r.trace_id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
    }

    #[test]
    fn patch_metadata_prefers_target_sha256_over_sha256() {
        let meta: PatchMetadata = serde_json::from_value(serde_json::json!({
            "sha256": "base",
            "target_sha256": "target",
        }))
        .unwrap();
        assert_eq!(meta.effective_sha256(), Some("target"));

        let meta: PatchMetadata = serde_json::from_value(serde_json::json!({
            "sha256": "only-sha256",
        }))
        .unwrap();
        assert_eq!(meta.effective_sha256(), Some("only-sha256"));

        let meta: PatchMetadata = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(meta.effective_sha256(), None);
    }
}
