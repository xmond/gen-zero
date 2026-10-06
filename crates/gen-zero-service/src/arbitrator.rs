//! Phase 2 of the closed-loop self-evolving pipeline: an async LLM
//! arbitrator that drains `PENDING` rows from
//! [`gen_zero_storage::DurableRefusalStore`], asks an LLM (OpenAI
//! chat-completions wire format) to judge each one, and writes the verdict
//! back through [`gen_zero_storage::DurableRefusalStore::complete_arbitration`]
//! or [`gen_zero_storage::DurableRefusalStore::fail_arbitration`].
//!
//! This module mirrors [`crate::closed_loop`] structurally: a plain config
//! struct, a `spawn_*_daemon` function with the same `watch::Receiver<bool>`
//! shutdown pattern, and per-tick errors that are logged, never panicked or
//! allowed to kill the loop.
//!
//! The core safety property is fail-closed anti-hallucination validation: an
//! `is_answerable: true` judgement is only ever written back if its
//! `evidence_span` is a verbatim, non-empty substring of the trace's
//! `context`. Any judgement that fails validation is treated as a per-trace
//! failure (via `fail_arbitration`), never written as a result, and the
//! batch continues to the next trace.

use gen_zero_storage::{ArbitrationResult, DurableRefusalStore, DurableRefusalTrace};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::watch;

/// Wiring for the LLM arbitrator: which endpoint/model to call, how to
/// authenticate, batch/poll sizing, and the retry cap.
#[derive(Clone, Debug)]
pub struct LlmArbitratorConfig {
    pub api_endpoint: String,
    /// Resolved by the caller: config value, else env
    /// `GENZERO_ARBITRATOR_API_KEY`, else env `OPENAI_API_KEY`. `None` means
    /// the arbitrator must not run (see [`run_once`] and
    /// [`spawn_arbitrator_daemon`]).
    pub api_key: Option<String>,
    pub model: String,
    pub batch_size: usize,
    pub poll_interval_secs: u64,
    pub max_retries: u32,
    /// Per-request HTTP timeout. A hung upstream API must not block a batch
    /// (or a test) forever.
    pub request_timeout_secs: u64,
}

impl Default for LlmArbitratorConfig {
    fn default() -> Self {
        Self {
            api_endpoint: "https://api.openai.com/v1/chat/completions".to_string(),
            api_key: None,
            model: "gpt-4o".to_string(),
            batch_size: 10,
            poll_interval_secs: 30,
            max_retries: 5,
            request_timeout_secs: 30,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ArbitratorError {
    #[error(
        "no API key configured (set --api-key, GENZERO_ARBITRATOR_API_KEY, or OPENAI_API_KEY)"
    )]
    MissingApiKey,
    #[error("storage error: {0}")]
    Storage(#[from] gen_zero_storage::StorageError),
    #[error("failed to build HTTP client: {0}")]
    HttpClient(String),
}

/// Counts from one [`run_once`] batch. A plain struct of counts a CLI can
/// print as JSON.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ArbitrationBatchReport {
    pub fetched: usize,
    pub arbitrated: usize,
    pub hallucinations_rejected: usize,
    pub transport_failures: usize,
    pub parse_failures: usize,
}

/// The JSON object the arbitrator LLM is asked to return: exactly these
/// five keys, nothing else required.
#[derive(Debug, Clone, Deserialize)]
struct LlmJudgement {
    is_answerable: bool,
    #[serde(default)]
    gold_answer: String,
    #[serde(default)]
    evidence_span: String,
    #[serde(default)]
    contradiction_fact: String,
    // Requested from the model (per the system prompt) and parsed so a
    // missing/extra key never fails deserialization, but not surfaced
    // separately: the full raw judgement (rationale included) is already
    // preserved verbatim in `ArbitrationResult::arbitration_raw`.
    #[allow(dead_code)]
    #[serde(default)]
    rationale: String,
}

#[derive(Debug, Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: String,
}

#[derive(Debug, Serialize)]
struct ResponseFormat<'a> {
    #[serde(rename = "type")]
    kind: &'a str,
}

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    response_format: ResponseFormat<'a>,
    temperature: f32,
}

#[derive(Debug, Deserialize)]
struct ChatChoiceMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

const SYSTEM_PROMPT: &str = "You are an arbitration judge for a question-answering \
    system. You will be given a context passage, a question, and a candidate \
    answer that an earlier stage refused to commit to. Judge whether the \
    question is answerable strictly from the given context.\n\n\
    Respond with ONLY a single JSON object, no markdown code fences, no prose \
    before or after it. The JSON object must have exactly these keys:\n\
    - \"is_answerable\": boolean\n\
    - \"gold_answer\": string (the correct answer if answerable, else empty)\n\
    - \"evidence_span\": string (if answerable, this MUST be an exact, \
    verbatim substring copied character-for-character from the provided \
    context that supports the answer; if not answerable, leave empty)\n\
    - \"contradiction_fact\": string (if not answerable, a short statement of \
    the fact in the context that contradicts or fails to support the \
    candidate; if answerable, leave empty)\n\
    - \"rationale\": string (a brief explanation of your judgement)\n\n\
    It is critical that evidence_span be copied verbatim from the context: it \
    will be mechanically checked against the context text, and a fabricated \
    or paraphrased span will cause your judgement to be rejected.";

fn user_prompt(trace: &DurableRefusalTrace) -> String {
    format!(
        "Context:\n{}\n\nQuestion:\n{}\n\nCandidate answer (refused by an earlier \
         stage):\n{}\n\nJudge whether the question is answerable strictly from the \
         context above, and respond with the JSON object described in the system \
         prompt.",
        trace.context, trace.question, trace.candidate
    )
}

/// Strip a leading/trailing markdown code fence if present, then take the
/// substring from the first `{` to the last `}`. Defends against stray
/// prose even when `response_format: json_object` is set on providers that
/// don't fully honor it.
fn extract_json_object(raw: &str) -> Option<&str> {
    let mut s = raw.trim();
    if let Some(rest) = s.strip_prefix("```json") {
        s = rest;
    } else if let Some(rest) = s.strip_prefix("```") {
        s = rest;
    }
    if let Some(rest) = s.strip_suffix("```") {
        s = rest;
    }
    s = s.trim();
    let start = s.find('{')?;
    let end = s.rfind('}')?;
    if end < start {
        return None;
    }
    Some(&s[start..=end])
}

/// Outcome of arbitrating one trace, used internally to decide which store
/// method to call and how to tally the report.
enum TraceOutcome {
    Completed(ArbitrationResult),
    Hallucination(String),
    Transport(String),
    Parse(String),
}

async fn judge_one(
    client: &reqwest::Client,
    config: &LlmArbitratorConfig,
    api_key: &str,
    trace: &DurableRefusalTrace,
) -> TraceOutcome {
    let request = ChatRequest {
        model: &config.model,
        messages: vec![
            ChatMessage {
                role: "system",
                content: SYSTEM_PROMPT.to_string(),
            },
            ChatMessage {
                role: "user",
                content: user_prompt(trace),
            },
        ],
        response_format: ResponseFormat {
            kind: "json_object",
        },
        temperature: 0.0,
    };

    let resp = match client
        .post(&config.api_endpoint)
        .bearer_auth(api_key)
        .json(&request)
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(error) => return TraceOutcome::Transport(format!("request error: {error}")),
    };

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return TraceOutcome::Transport(format!(
            "HTTP {status}: {}",
            body.chars().take(512).collect::<String>()
        ));
    }

    let body_text = match resp.text().await {
        Ok(t) => t,
        Err(error) => {
            return TraceOutcome::Transport(format!("failed reading response body: {error}"))
        }
    };

    let chat_response: ChatResponse = match serde_json::from_str(&body_text) {
        Ok(v) => v,
        Err(error) => {
            return TraceOutcome::Parse(format!(
                "response is not a valid chat-completions envelope: {error}"
            ))
        }
    };

    let Some(choice) = chat_response.choices.into_iter().next() else {
        return TraceOutcome::Parse("response had no choices".to_string());
    };
    let content = choice.message.content;

    let Some(json_slice) = extract_json_object(&content) else {
        return TraceOutcome::Parse(format!(
            "could not locate a JSON object in model content: {}",
            content.chars().take(256).collect::<String>()
        ));
    };

    let judgement: LlmJudgement = match serde_json::from_str(json_slice) {
        Ok(j) => j,
        Err(error) => {
            return TraceOutcome::Parse(format!("failed to parse judgement JSON: {error}"))
        }
    };

    let now_ms = now_ms();

    if judgement.is_answerable {
        let trimmed_span = judgement.evidence_span.trim();
        // The empty guard must come first and short-circuit: an empty
        // trimmed_span would otherwise make `context.contains("")` return
        // true, trivially "finding" nothing as something.
        let found = !trimmed_span.is_empty() && trace.context.contains(trimmed_span);
        let gold_answer = judgement.gold_answer.trim();
        if !found {
            return TraceOutcome::Hallucination(format!(
                "hallucinated evidence_span not found in context (fail-closed): {:?}",
                trimmed_span.chars().take(200).collect::<String>()
            ));
        }
        if gold_answer.is_empty() {
            return TraceOutcome::Hallucination(
                "hallucinated judgement: is_answerable=true but gold_answer is empty".to_string(),
            );
        }
        TraceOutcome::Completed(ArbitrationResult {
            is_answerable: true,
            gold_answer: Some(gold_answer.to_string()),
            evidence_span: Some(trimmed_span.to_string()),
            contradiction_fact: None,
            arbitrator_model: config.model.clone(),
            arbitration_raw: content,
            arbitrated_at_ms: now_ms,
        })
    } else {
        let contradiction_fact = judgement.contradiction_fact.trim();
        if contradiction_fact.is_empty() {
            return TraceOutcome::Hallucination(
                "hallucinated judgement: is_answerable=false but contradiction_fact is empty"
                    .to_string(),
            );
        }
        TraceOutcome::Completed(ArbitrationResult {
            is_answerable: false,
            gold_answer: None,
            evidence_span: None,
            contradiction_fact: Some(contradiction_fact.to_string()),
            arbitrator_model: config.model.clone(),
            arbitration_raw: content,
            arbitrated_at_ms: now_ms,
        })
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Drain up to `config.batch_size` PENDING-and-under-retry-cap traces from
/// `store`, ask the configured LLM to judge each, and write the verdict
/// back with fail-closed validation. Returns `Err` only for "can't even
/// start" conditions (missing API key, or a `StorageError` from the fetch
/// itself); every per-trace failure (transport, parse, hallucination) is
/// recorded via `fail_arbitration` and tallied in the returned report, and
/// the loop continues to the next trace.
pub async fn run_once(
    config: &LlmArbitratorConfig,
    store: &DurableRefusalStore,
) -> Result<ArbitrationBatchReport, ArbitratorError> {
    let Some(api_key) = config.api_key.as_deref() else {
        return Err(ArbitratorError::MissingApiKey);
    };

    let traces =
        store.fetch_pending_arbitration_under_retry(config.batch_size, config.max_retries)?;

    let mut report = ArbitrationBatchReport {
        fetched: traces.len(),
        ..Default::default()
    };
    if traces.is_empty() {
        return Ok(report);
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(config.request_timeout_secs.max(1)))
        .build()
        .map_err(|e| ArbitratorError::HttpClient(e.to_string()))?;

    for trace in &traces {
        let outcome = judge_one(&client, config, api_key, trace).await;
        match outcome {
            TraceOutcome::Completed(result) => {
                match store.complete_arbitration(&trace.trace_id, &result) {
                    Ok(()) => report.arbitrated += 1,
                    Err(gen_zero_storage::StorageError::InvalidArbitrationState {
                        trace_id,
                        expected,
                        actual,
                    }) => {
                        tracing::warn!(
                            trace_id,
                            expected,
                            actual,
                            "trace changed state mid-batch (e.g. revoked); skipping"
                        );
                    }
                    Err(gen_zero_storage::StorageError::DurableTraceNotFound(trace_id)) => {
                        tracing::warn!(trace_id, "trace disappeared mid-batch; skipping");
                    }
                    Err(other) => return Err(ArbitratorError::Storage(other)),
                }
            }
            TraceOutcome::Hallucination(reason) => {
                report.hallucinations_rejected += 1;
                record_failure(store, &trace.trace_id, &reason)?;
            }
            TraceOutcome::Transport(reason) => {
                report.transport_failures += 1;
                record_failure(store, &trace.trace_id, &reason)?;
            }
            TraceOutcome::Parse(reason) => {
                report.parse_failures += 1;
                record_failure(store, &trace.trace_id, &reason)?;
            }
        }
    }

    Ok(report)
}

/// Shared tail of the three failure branches in [`run_once`]: call
/// `fail_arbitration`, and treat a state-change-mid-batch the same
/// non-fatal way `complete_arbitration` does above.
fn record_failure(
    store: &DurableRefusalStore,
    trace_id: &str,
    reason: &str,
) -> Result<(), ArbitratorError> {
    match store.fail_arbitration(trace_id, reason) {
        Ok(()) => Ok(()),
        Err(gen_zero_storage::StorageError::InvalidArbitrationState {
            trace_id,
            expected,
            actual,
        }) => {
            tracing::warn!(
                trace_id,
                expected,
                actual,
                "trace changed state mid-batch (e.g. revoked); skipping fail_arbitration"
            );
            Ok(())
        }
        Err(gen_zero_storage::StorageError::DurableTraceNotFound(trace_id)) => {
            tracing::warn!(
                trace_id,
                "trace disappeared mid-batch; skipping fail_arbitration"
            );
            Ok(())
        }
        Err(other) => Err(ArbitratorError::Storage(other)),
    }
}

/// Spawn the background arbitration poller. Mirrors
/// [`crate::closed_loop::spawn_feedback_syncer`]: if no API key is
/// configured, warns and returns without spinning; otherwise loops on
/// `config.poll_interval_secs`, calling [`run_once`] each tick and logging
/// the report or error. Never panics or exits the loop on a per-tick error.
/// Runs until `shutdown` reports `true` (or its sender is dropped).
pub fn spawn_arbitrator_daemon(
    config: LlmArbitratorConfig,
    store: Arc<DurableRefusalStore>,
    mut shutdown: watch::Receiver<bool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        if config.api_key.is_none() {
            tracing::warn!("arbitrator daemon started without an API key configured; exiting");
            return;
        }
        let mut interval =
            tokio::time::interval(Duration::from_secs(config.poll_interval_secs.max(1)));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            // See the matching comment in `closed_loop::spawn_feedback_syncer`:
            // checked up front so a shutdown already current at subscribe time
            // is never missed, and `changed()` (not `wait_for`) is used in the
            // select below to keep this spawned future `Send`.
            if *shutdown.borrow() {
                tracing::info!("arbitrator daemon shutting down");
                break;
            }
            tokio::select! {
                biased;
                changed = shutdown.changed() => {
                    if changed.is_err() {
                        tracing::info!("arbitrator daemon shutting down (sender dropped)");
                        break;
                    }
                }
                _ = interval.tick() => {
                    match run_once(&config, &store).await {
                        Ok(report) => {
                            tracing::info!(
                                fetched = report.fetched,
                                arbitrated = report.arbitrated,
                                hallucinations_rejected = report.hallucinations_rejected,
                                transport_failures = report.transport_failures,
                                parse_failures = report.parse_failures,
                                "arbitration batch complete"
                            );
                        }
                        Err(error) => {
                            tracing::warn!(%error, "arbitration batch failed");
                        }
                    }
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_json_object_strips_fences_and_prose() {
        assert_eq!(
            extract_json_object("```json\n{\"a\": 1}\n```"),
            Some("{\"a\": 1}")
        );
        assert_eq!(extract_json_object("{\"a\": 1}"), Some("{\"a\": 1}"));
        assert_eq!(
            extract_json_object("sure, here you go: {\"a\": 1} thanks"),
            Some("{\"a\": 1}")
        );
        assert_eq!(extract_json_object("no json here"), None);
    }

    #[test]
    fn empty_evidence_span_is_never_treated_as_found() {
        let context = "some context";
        let trimmed_span = "";
        let found = !trimmed_span.is_empty() && context.contains(trimmed_span);
        assert!(!found);
    }
}
