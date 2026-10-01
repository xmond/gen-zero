//! Semantic bridge client: Rust `zero` tool -> Python semantic scorer.
//!
//! This is the remote variant of [`crate::semantic::SemanticBackend`]. The
//! in-process variant (native Qwen on candle) returns the same response types
//! and passes the same validators ([`validate_ask`], [`validate_risk`],
//! [`validate_route`]).
//!
//! The Python service (`python/gen_zero/service/app.py`, started with
//! `python3 -m gen_zero.cli semantic`, default port 8995) exposes `/v1/semantic_ask`,
//! `/v1/semantic_route` and `/v1/semantic_risk`. All three run the local Zero
//! backbone. `/v1/semantic_health` identifies the service. This client adds the
//! production plumbing around those calls: a short connect timeout, bounded
//! retries with backoff, and a consecutive-failure circuit breaker, so a dead
//! Python process costs one fast failure instead of a stalled request.
//!
//! Plain HTTP only (no TLS): the bridge is meant for a co-located process.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use thiserror::Error;

/// Default semantic scorer endpoint. Port 8995 is reserved for the scorer:
/// 8999 is the MCP SSE port ([`crate::server::DEFAULT_MCP_SSE_PORT`]), and a
/// bridge pointed there reaches an MCP server (or an older service) that
/// answers 404.
pub const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:8995";

/// Path of the identity probe on the Python side.
pub const HEALTH_PATH: &str = "/v1/semantic_health";
/// Routes a real scorer must advertise on [`HEALTH_PATH`].
pub const REQUIRED_ROUTES: [&str; 3] = [
    "/v1/semantic_ask",
    "/v1/semantic_route",
    "/v1/semantic_risk",
];

#[derive(Debug, Error, Clone, PartialEq)]
pub enum BridgeError {
    #[error("semantic bridge circuit open for another {retry_in_ms} ms")]
    CircuitOpen { retry_in_ms: u64 },
    #[error("semantic bridge transport error: {0}")]
    Transport(String),
    #[error("semantic bridge returned HTTP {status}: {body}")]
    Status { status: u16, body: String },
    /// The endpoint has no such route: something other than the gen-zero
    /// semantic scorer holds the port (an MCP server, an older service).
    #[error(
        "semantic bridge endpoint {url} answered HTTP 404: the service on that port is not \
         the gen-zero semantic scorer (start it with `python3 -m gen_zero.cli semantic`, default port 8995)"
    )]
    WrongService { url: String },
    #[error("semantic bridge response rejected: {0}")]
    InvalidResponse(String),
    /// The in-process Qwen scorer refused the input or failed to compute.
    #[error("native Qwen scorer failed: {0}")]
    Native(String),
    /// Every in-process scorer slot stayed busy for the whole queue timeout.
    #[error("native Qwen scorer overloaded: no free slot after {waited_ms} ms")]
    Overloaded { waited_ms: u64 },
}

impl BridgeError {
    fn is_retryable(&self) -> bool {
        match self {
            Self::Transport(_) => true,
            Self::Status { status, .. } => *status >= 500,
            _ => false,
        }
    }

    /// A 400/422 means our request was bad, not that the service is down.
    fn counts_against_breaker(&self) -> bool {
        !matches!(
            self,
            Self::Status {
                status: 400 | 422,
                ..
            } | Self::CircuitOpen { .. }
        )
    }
}

#[derive(Clone, Debug)]
pub struct BridgeConfig {
    pub endpoint: String,
    pub api_key: Option<String>,
    pub connect_timeout: Duration,
    pub request_timeout: Duration,
    pub max_retries: u32,
    pub breaker_threshold: u32,
    pub breaker_cooldown: Duration,
}

impl BridgeConfig {
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into().trim_end_matches('/').to_string(),
            api_key: None,
            connect_timeout: Duration::from_millis(300),
            request_timeout: Duration::from_secs(15),
            max_retries: 1,
            breaker_threshold: 3,
            breaker_cooldown: Duration::from_secs(30),
        }
    }

    /// Read the bridge settings from the environment.
    ///
    /// `GENZERO_PYTHON_ENDPOINT` (default `http://127.0.0.1:8995`); the values
    /// `off`, `none`, `disabled` or an empty string turn the bridge off and
    /// return `None`. The bearer token is `GENZERO_PYTHON_API_KEY`, else
    /// `GENZERO_API_KEY` (the Python service checks the same variable).
    pub fn from_env() -> Option<Self> {
        let endpoint =
            std::env::var("GENZERO_PYTHON_ENDPOINT").unwrap_or_else(|_| DEFAULT_ENDPOINT.into());
        let trimmed = endpoint.trim();
        if trimmed.is_empty()
            || matches!(
                trimmed.to_ascii_lowercase().as_str(),
                "off" | "none" | "disabled"
            )
        {
            return None;
        }
        let ms = |name: &str, default: u64| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .map(Duration::from_millis)
                .unwrap_or(Duration::from_millis(default))
        };
        let count = |name: &str, default: u32| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.trim().parse::<u32>().ok())
                .unwrap_or(default)
        };
        let api_key = std::env::var("GENZERO_PYTHON_API_KEY")
            .or_else(|_| std::env::var("GENZERO_API_KEY"))
            .ok()
            .map(|k| k.trim().to_string())
            .filter(|k| !k.is_empty());
        Some(Self {
            endpoint: trimmed.trim_end_matches('/').to_string(),
            api_key,
            connect_timeout: ms("GENZERO_BRIDGE_CONNECT_TIMEOUT_MS", 300),
            request_timeout: ms("GENZERO_BRIDGE_TIMEOUT_MS", 15_000),
            max_retries: count("GENZERO_BRIDGE_RETRIES", 1),
            breaker_threshold: count("GENZERO_BRIDGE_BREAKER_THRESHOLD", 3).max(1),
            breaker_cooldown: ms("GENZERO_BRIDGE_BREAKER_COOLDOWN_MS", 30_000),
        })
    }
}

/// One scored candidate. Same shape from the Python scorer and the native one.
pub use gen_zero_model::semantic_qwen::CandidateScore;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SemanticAskResponse {
    pub chosen: String,
    pub chosen_index: usize,
    pub candidates: Vec<CandidateScore>,
    pub entropy: f64,
    pub scorer: Value,
    pub embedding_dim: usize,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
    pub timing_ms: f64,
}

/// Safety-risk judgement of one text by the Python few-shot classifier.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SemanticRiskResponse {
    pub p_dangerous: f64,
    pub log_odds: f64,
    pub windows: usize,
    pub thresholds: RiskThresholds,
    pub classifier: Value,
    pub forward_ms: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct RiskThresholds {
    pub escalate: f64,
    pub hard_stop: f64,
}

/// Result of the identity probe on [`HEALTH_PATH`].
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum BridgeHealth {
    /// The gen-zero semantic scorer answered and lists every required route.
    Ready { backbone_loaded: bool },
    /// Something answered, but it is not the semantic scorer.
    WrongService { detail: String },
    /// Nothing answered (connection refused, timeout).
    Unreachable { detail: String },
}

impl BridgeHealth {
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready { .. })
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct HealthReport {
    pub endpoint: String,
    #[serde(flatten)]
    pub health: BridgeHealth,
    pub probed_at_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SemanticRouteResponse {
    pub ranked: Vec<CandidateScore>,
    pub selected: Vec<String>,
    pub entropy: f64,
    pub scorer: Value,
    pub timing_ms: f64,
}

/// Inputs for one `semantic_ask` call.
#[derive(Clone, Debug, Default)]
pub struct AskInput<'a> {
    pub context: &'a str,
    pub candidates: &'a [String],
    pub state: Option<&'a Value>,
    pub history: &'a [String],
    pub return_embedding: bool,
}

pub struct SemanticBridgeClient {
    config: BridgeConfig,
    http: reqwest::Client,
    consecutive_failures: AtomicU32,
    open_until_ms: AtomicU64,
    last_health: RwLock<Option<HealthReport>>,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn check_probabilities(scores: &[CandidateScore], expected: &[String]) -> Result<(), BridgeError> {
    if scores.len() != expected.len() {
        return Err(BridgeError::InvalidResponse(format!(
            "{} scores for {} candidates",
            scores.len(),
            expected.len()
        )));
    }
    let mut sum = 0.0;
    for s in scores {
        if !s.probability.is_finite() || !(0.0..=1.0).contains(&s.probability) {
            return Err(BridgeError::InvalidResponse(format!(
                "probability {} for '{}' is outside [0, 1]",
                s.probability, s.name
            )));
        }
        if !expected.contains(&s.name) {
            return Err(BridgeError::InvalidResponse(format!(
                "unknown candidate '{}'",
                s.name
            )));
        }
        sum += s.probability;
    }
    if (sum - 1.0).abs() > 1e-3 {
        return Err(BridgeError::InvalidResponse(format!(
            "probabilities sum to {sum}"
        )));
    }
    Ok(())
}

/// Contract of an ask answer: a distribution over exactly the candidates,
/// a consistent choice, and a normalized entropy.
pub fn validate_ask(r: &SemanticAskResponse, candidates: &[String]) -> Result<(), BridgeError> {
    check_probabilities(&r.candidates, candidates)?;
    if r.chosen_index >= r.candidates.len() || r.candidates[r.chosen_index].name != r.chosen {
        return Err(BridgeError::InvalidResponse(
            "chosen does not match chosen_index".into(),
        ));
    }
    if !(0.0..=1.0).contains(&r.entropy) {
        return Err(BridgeError::InvalidResponse(format!(
            "entropy {} outside [0, 1]",
            r.entropy
        )));
    }
    Ok(())
}

/// Contract of a risk answer: probability and thresholds in [0, 1], ordered.
pub fn validate_risk(r: &SemanticRiskResponse) -> Result<(), BridgeError> {
    let t = r.thresholds;
    let unit = |x: f64| x.is_finite() && (0.0..=1.0).contains(&x);
    if !unit(r.p_dangerous) || !unit(t.escalate) || !unit(t.hard_stop) {
        return Err(BridgeError::InvalidResponse(format!(
            "risk {} or thresholds {:?} outside [0, 1]",
            r.p_dangerous, t
        )));
    }
    if t.escalate > t.hard_stop {
        return Err(BridgeError::InvalidResponse(format!(
            "escalate threshold {} above hard-stop threshold {}",
            t.escalate, t.hard_stop
        )));
    }
    Ok(())
}

/// Contract of a route answer: a distribution over exactly the tools, sorted.
pub fn validate_route(r: &SemanticRouteResponse, tool_names: &[String]) -> Result<(), BridgeError> {
    check_probabilities(&r.ranked, tool_names)?;
    if r.ranked
        .windows(2)
        .any(|w| w[0].probability < w[1].probability)
    {
        return Err(BridgeError::InvalidResponse("ranking is not sorted".into()));
    }
    Ok(())
}

impl SemanticBridgeClient {
    pub fn new(config: BridgeConfig) -> Result<Self, BridgeError> {
        let http = reqwest::Client::builder()
            .connect_timeout(config.connect_timeout)
            .timeout(config.request_timeout)
            .build()
            .map_err(|e| BridgeError::Transport(e.to_string()))?;
        Ok(Self {
            config,
            http,
            consecutive_failures: AtomicU32::new(0),
            open_until_ms: AtomicU64::new(0),
            last_health: RwLock::new(None),
        })
    }

    /// Ask the endpoint who it is (`GET /v1/semantic_health`, no auth, no
    /// model load) and cache the answer for [`Self::last_health`].
    ///
    /// A 404 or a body without the required routes is `WrongService`: the
    /// failure mode where an older service or the MCP server holds the port
    /// and every semantic call would silently fall back.
    pub async fn probe(&self) -> HealthReport {
        let url = format!("{}{}", self.config.endpoint, HEALTH_PATH);
        let health = match self.http.get(&url).send().await {
            Err(e) => BridgeHealth::Unreachable {
                detail: e.to_string(),
            },
            Ok(resp) if resp.status() == reqwest::StatusCode::NOT_FOUND => {
                BridgeHealth::WrongService {
                    detail: BridgeError::WrongService { url }.to_string(),
                }
            }
            Ok(resp) if !resp.status().is_success() => BridgeHealth::WrongService {
                detail: format!("{url} answered HTTP {}", resp.status().as_u16()),
            },
            Ok(resp) => match resp.json::<Value>().await {
                Err(e) => BridgeHealth::WrongService {
                    detail: format!("{url} did not answer JSON: {e}"),
                },
                Ok(body) => {
                    let routes: Vec<&str> = body["endpoints"]
                        .as_array()
                        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
                        .unwrap_or_default();
                    let missing: Vec<&str> = REQUIRED_ROUTES
                        .iter()
                        .copied()
                        .filter(|r| !routes.contains(r))
                        .collect();
                    if body["service"] != "gen-zero-semantic" || !missing.is_empty() {
                        BridgeHealth::WrongService {
                            detail: format!(
                                "{url} is not a current gen-zero semantic scorer \
                                 (service={}, missing routes {:?})",
                                body["service"], missing
                            ),
                        }
                    } else {
                        BridgeHealth::Ready {
                            backbone_loaded: body["backbone_loaded"].as_bool().unwrap_or(false),
                        }
                    }
                }
            },
        };
        let report = HealthReport {
            endpoint: self.config.endpoint.clone(),
            health,
            probed_at_ms: now_ms(),
        };
        if let Ok(mut slot) = self.last_health.write() {
            *slot = Some(report.clone());
        }
        report
    }

    /// Last probe result, if any.
    pub fn last_health(&self) -> Option<HealthReport> {
        self.last_health.read().ok().and_then(|g| g.clone())
    }

    pub fn from_env() -> Option<Self> {
        let config = BridgeConfig::from_env()?;
        match Self::new(config) {
            Ok(client) => Some(client),
            Err(e) => {
                tracing::error!("semantic bridge disabled: {e}");
                None
            }
        }
    }

    pub fn endpoint(&self) -> &str {
        &self.config.endpoint
    }

    /// True when the endpoint is a loopback address on `port`, i.e. the Rust
    /// server listening on `port` would call itself instead of Python.
    pub fn targets_local_port(&self, port: u16) -> bool {
        let Ok(url) = reqwest::Url::parse(&self.config.endpoint) else {
            return false;
        };
        let local = matches!(
            url.host_str(),
            Some("127.0.0.1") | Some("localhost") | Some("0.0.0.0") | Some("[::1]") | Some("::1")
        );
        local && url.port_or_known_default() == Some(port)
    }

    pub fn circuit_open(&self) -> bool {
        now_ms() < self.open_until_ms.load(Ordering::Acquire)
    }

    fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::Release);
        self.open_until_ms.store(0, Ordering::Release);
    }

    fn record_failure(&self) {
        let failures = self.consecutive_failures.fetch_add(1, Ordering::AcqRel) + 1;
        if failures >= self.config.breaker_threshold {
            let until = now_ms() + self.config.breaker_cooldown.as_millis() as u64;
            self.open_until_ms.store(until, Ordering::Release);
            tracing::warn!(
                endpoint = %self.config.endpoint,
                failures,
                "semantic bridge circuit opened"
            );
        }
    }

    async fn post_once<T: DeserializeOwned>(
        &self,
        path: &str,
        body: &Value,
    ) -> Result<T, BridgeError> {
        let mut req = self
            .http
            .post(format!("{}{}", self.config.endpoint, path))
            .json(body);
        if let Some(key) = &self.config.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| BridgeError::Transport(e.to_string()))?;
        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Err(BridgeError::WrongService {
                url: format!("{}{}", self.config.endpoint, path),
            });
        }
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(BridgeError::Status {
                status: status.as_u16(),
                body: text.chars().take(512).collect(),
            });
        }
        resp.json::<T>()
            .await
            .map_err(|e| BridgeError::InvalidResponse(e.to_string()))
    }

    async fn post<T: DeserializeOwned>(
        &self,
        path: &str,
        body: &Value,
        validate: impl Fn(&T) -> Result<(), BridgeError>,
    ) -> Result<T, BridgeError> {
        let open_until = self.open_until_ms.load(Ordering::Acquire);
        let now = now_ms();
        if now < open_until {
            return Err(BridgeError::CircuitOpen {
                retry_in_ms: open_until - now,
            });
        }
        let mut attempt = 0;
        loop {
            let result = match self.post_once::<T>(path, body).await {
                Ok(v) => validate(&v).map(|_| v),
                Err(e) => Err(e),
            };
            match result {
                Ok(v) => {
                    self.record_success();
                    return Ok(v);
                }
                Err(e) if e.is_retryable() && attempt < self.config.max_retries => {
                    attempt += 1;
                    tokio::time::sleep(Duration::from_millis(50 << attempt.min(6))).await;
                }
                Err(e) => {
                    if e.counts_against_breaker() {
                        self.record_failure();
                    }
                    return Err(e);
                }
            }
        }
    }

    pub async fn semantic_ask(
        &self,
        input: &AskInput<'_>,
    ) -> Result<SemanticAskResponse, BridgeError> {
        let mut body = json!({
            "context": input.context,
            "candidates": input.candidates,
            "return_embedding": input.return_embedding,
        });
        if let Some(state) = input.state {
            body["state"] = state.clone();
        }
        if !input.history.is_empty() {
            body["history"] = json!(input.history);
        }
        self.post("/v1/semantic_ask", &body, |r: &SemanticAskResponse| {
            validate_ask(r, input.candidates)
        })
        .await
    }

    /// Safety risk of `text` (any language) from the Python classifier.
    pub async fn semantic_risk(&self, text: &str) -> Result<SemanticRiskResponse, BridgeError> {
        self.post("/v1/semantic_risk", &json!({ "text": text }), validate_risk)
            .await
    }

    pub async fn semantic_route(
        &self,
        intent: &str,
        tools: &[Value],
        tool_names: &[String],
        top_k: usize,
        state: Option<&Value>,
    ) -> Result<SemanticRouteResponse, BridgeError> {
        let mut body = json!({ "intent": intent, "tools": tools, "top_k": top_k.max(1) });
        if let Some(state) = state {
            body["state"] = state.clone();
        }
        self.post("/v1/semantic_route", &body, |r: &SemanticRouteResponse| {
            validate_route(r, tool_names)
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{extract::State, http::StatusCode, routing::post, Json, Router};
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    /// Transport stub: an in-process HTTP server that returns canned bodies.
    /// It tests the client (retry, breaker, validation), not semantics.
    async fn transport_stub(status: StatusCode, body: Value) -> (String, Arc<AtomicUsize>) {
        let hits = Arc::new(AtomicUsize::new(0));
        let app =
            Router::new()
                .route(
                    "/v1/semantic_ask",
                    post(
                        |State((hits, status, body)): State<(
                            Arc<AtomicUsize>,
                            StatusCode,
                            Value,
                        )>| async move {
                            hits.fetch_add(1, Ordering::SeqCst);
                            (status, Json(body))
                        },
                    ),
                )
                .with_state((hits.clone(), status, body));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), hits)
    }

    fn cands() -> Vec<String> {
        vec!["delete".into(), "backup".into()]
    }

    fn well_formed() -> Value {
        json!({
            "chosen": "backup", "chosen_index": 1, "entropy": 0.5,
            "candidates": [
                {"name": "delete", "log_likelihood": -2.0, "baseline_log_likelihood": -1.0, "pmi": -1.0, "probability": 0.25},
                {"name": "backup", "log_likelihood": -1.0, "baseline_log_likelihood": -1.5, "pmi": 0.5, "probability": 0.75}
            ],
            "scorer": {"id": "stub"}, "embedding_dim": 896, "timing_ms": 1.0
        })
    }

    #[tokio::test]
    async fn parses_and_validates_a_well_formed_response() {
        let (endpoint, hits) = transport_stub(StatusCode::OK, well_formed()).await;
        let client = SemanticBridgeClient::new(BridgeConfig::new(endpoint)).unwrap();
        let c = cands();
        let r = client
            .semantic_ask(&AskInput {
                context: "x",
                candidates: &c,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(r.chosen, "backup");
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn rejects_probabilities_that_do_not_sum_to_one() {
        let mut bad = well_formed();
        bad["candidates"][1]["probability"] = json!(0.9);
        let (endpoint, _) = transport_stub(StatusCode::OK, bad).await;
        let client = SemanticBridgeClient::new(BridgeConfig::new(endpoint)).unwrap();
        let c = cands();
        let err = client
            .semantic_ask(&AskInput {
                context: "x",
                candidates: &c,
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert!(matches!(err, BridgeError::InvalidResponse(_)), "{err}");
    }

    #[tokio::test]
    async fn retries_server_errors_then_opens_the_circuit() {
        let (endpoint, hits) =
            transport_stub(StatusCode::SERVICE_UNAVAILABLE, json!({"error": "loading"})).await;
        let mut cfg = BridgeConfig::new(endpoint);
        cfg.max_retries = 2;
        cfg.breaker_threshold = 2;
        let client = SemanticBridgeClient::new(cfg).unwrap();
        let c = cands();
        let input = AskInput {
            context: "x",
            candidates: &c,
            ..Default::default()
        };

        for _ in 0..2 {
            let err = client.semantic_ask(&input).await.unwrap_err();
            assert!(
                matches!(err, BridgeError::Status { status: 503, .. }),
                "{err}"
            );
        }
        // 2 calls x (1 try + 2 retries)
        assert_eq!(hits.load(Ordering::SeqCst), 6);
        assert!(client.circuit_open());
        let err = client.semantic_ask(&input).await.unwrap_err();
        assert!(matches!(err, BridgeError::CircuitOpen { .. }), "{err}");
        assert_eq!(
            hits.load(Ordering::SeqCst),
            6,
            "open circuit must not hit the network"
        );
    }

    #[tokio::test]
    async fn bad_requests_do_not_trip_the_breaker() {
        let (endpoint, hits) =
            transport_stub(StatusCode::BAD_REQUEST, json!({"error": "bad"})).await;
        let mut cfg = BridgeConfig::new(endpoint);
        cfg.breaker_threshold = 1;
        let client = SemanticBridgeClient::new(cfg).unwrap();
        let c = cands();
        let input = AskInput {
            context: "x",
            candidates: &c,
            ..Default::default()
        };
        for _ in 0..3 {
            assert!(client.semantic_ask(&input).await.is_err());
        }
        assert_eq!(hits.load(Ordering::SeqCst), 3);
        assert!(!client.circuit_open());
    }

    #[tokio::test]
    async fn unreachable_endpoint_fails_fast_as_transport_error() {
        let mut cfg = BridgeConfig::new("http://127.0.0.1:1");
        cfg.max_retries = 0;
        let client = SemanticBridgeClient::new(cfg).unwrap();
        let c = cands();
        let started = std::time::Instant::now();
        let err = client
            .semantic_ask(&AskInput {
                context: "x",
                candidates: &c,
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert!(matches!(err, BridgeError::Transport(_)), "{err}");
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn detects_self_loop_on_the_serving_port() {
        let client = SemanticBridgeClient::new(BridgeConfig::new("http://127.0.0.1:8999")).unwrap();
        assert!(client.targets_local_port(8999));
        assert!(!client.targets_local_port(8080));
        // 192.0.2.0/24 is the RFC 5737 TEST-NET-1 block reserved for documentation/tests.
        let remote = SemanticBridgeClient::new(BridgeConfig::new("http://192.0.2.1:8999")).unwrap();
        assert!(!remote.targets_local_port(8999));
    }

    #[test]
    fn default_endpoint_is_the_dedicated_scorer_port_not_a_server_port() {
        assert_eq!(DEFAULT_ENDPOINT, "http://127.0.0.1:8995");
        let client = SemanticBridgeClient::new(BridgeConfig::new(DEFAULT_ENDPOINT)).unwrap();
        assert!(client.targets_local_port(8995));
        for server_port in [
            crate::server::DEFAULT_MCP_SSE_PORT,
            crate::server::DEFAULT_SERVE_PORT,
        ] {
            assert!(
                !client.targets_local_port(server_port),
                "default bridge would call the Rust server on {server_port}"
            );
        }
    }

    /// A service that knows nothing of the semantic routes (like the older
    /// gen_zero app that held 8999 on the dev box).
    async fn foreign_service() -> String {
        let app = Router::new().route(
            "/health",
            axum::routing::get(|| async { Json(json!({"status": "healthy"})) }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn a_404_is_reported_as_wrong_service_not_as_a_generic_error() {
        let endpoint = foreign_service().await;
        let mut cfg = BridgeConfig::new(endpoint.clone());
        cfg.max_retries = 0;
        let client = SemanticBridgeClient::new(cfg).unwrap();
        let c = cands();
        let err = client
            .semantic_ask(&AskInput {
                context: "x",
                candidates: &c,
                ..Default::default()
            })
            .await
            .unwrap_err();
        assert!(matches!(err, BridgeError::WrongService { .. }), "{err}");
        assert!(err.to_string().contains("not the gen-zero semantic scorer"));

        let report = client.probe().await;
        assert!(
            matches!(report.health, BridgeHealth::WrongService { .. }),
            "{report:?}"
        );
        assert_eq!(client.last_health().unwrap().endpoint, endpoint);
    }

    #[tokio::test]
    async fn probe_accepts_only_a_scorer_that_lists_every_route() {
        async fn serve(body: Value) -> String {
            let app = Router::new().route(
                HEALTH_PATH,
                axum::routing::get(move || {
                    let body = body.clone();
                    async move { Json(body) }
                }),
            );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            format!("http://{addr}")
        }
        let good = serve(json!({"service": "gen-zero-semantic", "endpoints": REQUIRED_ROUTES, "backbone_loaded": true})).await;
        let old =
            serve(json!({"service": "gen-zero-semantic", "endpoints": ["/v1/semantic_ask"]})).await;
        let client = SemanticBridgeClient::new(BridgeConfig::new(good)).unwrap();
        assert_eq!(
            client.probe().await.health,
            BridgeHealth::Ready {
                backbone_loaded: true
            }
        );
        let client = SemanticBridgeClient::new(BridgeConfig::new(old)).unwrap();
        assert!(!client.probe().await.health.is_ready());
        let mut cfg = BridgeConfig::new("http://127.0.0.1:1");
        cfg.max_retries = 0;
        let client = SemanticBridgeClient::new(cfg).unwrap();
        assert!(matches!(
            client.probe().await.health,
            BridgeHealth::Unreachable { .. }
        ));
    }
}
