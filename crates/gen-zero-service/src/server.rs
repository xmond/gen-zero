//! Dual-Transport MCP Server (Stdio & SSE / HTTP REST Gateway).
//!
//! Adheres strictly to Doc 07:
//! - simd-json stdio parsing loop with single outer preallocation and SIMD_JSON_PADDING
//! - Axum 0.7 HTTP/SSE routing with /sse, /message, /v1/decisions, /v1/decisions/stream,
//!   /v1/causal_fold
//! - MCP compliant error framing (top-level JSON-RPC error vs result { isError: true, _meta })
//! - HTTP routes never answer 200 for a refused or failed call: a typed
//!   refusal carries its own status (400 / 404 / 409 / 422 / 503), other
//!   `isError` outcomes map by their error code, engine errors are 400 / 500.

use crate::bridge::BridgeHealth;
use crate::closed_loop::{
    spawn_feedback_syncer, spawn_patch_poller, ClosedLoopConfig, FeedbackBuffer, FeedbackRecord,
};
use crate::error::ServiceError;
use crate::mount::{MountKey, MountRegistry};
use crate::semantic::SemanticBackend;
use crate::zero::{PolymorphicZeroEngine, ZeroEngineConfig, ZeroToolOutcome};
use arc_swap::ArcSwap;
use axum::{
    extract::{Extension, Json, Path, Query, Request},
    http::{header, HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Router,
};
use futures_util::stream::{self, Stream};
use gen_zero_provenance::mmr::ProofSibling;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt};
use tokio::sync::{mpsc, watch};

/// Per-connection MCP SSE sessions. The sender is kept in a process-local map
/// so a POST to `/message?session_id=...` can enqueue a JSON-RPC response for
/// the matching GET `/sse` stream. A standard mutex is intentional here: each
/// critical section only performs a map lookup/insert, and a synchronous drop
/// guard can remove a disconnected stream without an async cleanup task.
#[derive(Clone)]
struct SseSessionManager {
    sessions: Arc<StdMutex<HashMap<String, mpsc::Sender<Value>>>>,
    shutdown: watch::Sender<bool>,
}

impl Default for SseSessionManager {
    fn default() -> Self {
        Self {
            sessions: Arc::default(),
            shutdown: watch::channel(false).0,
        }
    }
}

impl SseSessionManager {
    fn shutdown(&self) {
        self.shutdown.send_replace(true);
        self.sessions
            .lock()
            .expect("SSE session lock poisoned")
            .clear();
    }

    fn create(&self) -> (String, mpsc::Receiver<Value>) {
        loop {
            let bytes: [u8; 16] = rand::random();
            let session_id = bytes
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            let (sender, receiver) = mpsc::channel(64);
            let mut sessions = self.sessions.lock().expect("SSE session lock poisoned");
            if sessions.contains_key(&session_id) {
                continue;
            }
            if !*self.shutdown.borrow() {
                sessions.insert(session_id.clone(), sender);
            }
            return (session_id, receiver);
        }
    }

    fn sender(&self, session_id: &str) -> Option<mpsc::Sender<Value>> {
        self.sessions
            .lock()
            .expect("SSE session lock poisoned")
            .get(session_id)
            .cloned()
    }

    fn remove(&self, session_id: &str) {
        self.sessions
            .lock()
            .expect("SSE session lock poisoned")
            .remove(session_id);
    }

    fn contains(&self, session_id: &str) -> bool {
        self.sessions
            .lock()
            .expect("SSE session lock poisoned")
            .contains_key(session_id)
    }
}

struct SseSessionGuard {
    manager: SseSessionManager,
    session_id: String,
}

impl Drop for SseSessionGuard {
    fn drop(&mut self) {
        self.manager.remove(&self.session_id);
    }
}

#[derive(Debug, Default, Deserialize)]
struct MessageQuery {
    session_id: Option<String>,
}

/// Constant-time token verification using BLAKE3 to prevent timing side-channel attacks.
#[inline]
pub fn constant_time_token_match(provided: &str, expected: &str) -> bool {
    let hash_provided = blake3::hash(provided.as_bytes());
    let hash_expected = blake3::hash(expected.as_bytes());
    let mut diff = 0u8;
    for (a, b) in hash_provided
        .as_bytes()
        .iter()
        .zip(hash_expected.as_bytes().iter())
    {
        diff |= a ^ b;
    }
    diff == 0
}

/// Default port of `gen-zero mcp` (MCP over SSE).
pub const DEFAULT_MCP_SSE_PORT: u16 = 8999;
/// Default port of `gen-zero serve --mode sse`.
pub const DEFAULT_SERVE_PORT: u16 = 8080;
/// How long a bridge probe result is reused by readiness probes.
const HEALTH_PROBE_TTL_MS: u64 = 10_000;

// Labels are drawn only from fixed enums/methods and numeric status codes;
// request data and paths never create unbounded metric series.
const LATENCY_BUCKETS: [f64; 8] = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0];

#[derive(Default)]
pub(crate) struct ServiceMetrics(parking_lot::Mutex<MetricValues>);

#[derive(Default)]
struct MetricValues {
    requests: std::collections::BTreeMap<(String, u16), u64>,
    http_requests: std::collections::BTreeMap<(&'static str, u16), u64>,
    gates: [u64; 4],
    latency_buckets: [u64; 8],
    latency_count: u64,
    latency_sum: f64,
}

impl ServiceMetrics {
    pub(crate) fn record_execution(
        &self,
        arguments: &Value,
        result: &Result<ZeroToolOutcome, ServiceError>,
        elapsed: f64,
    ) {
        let verb = result
            .as_ref()
            .ok()
            .map(|o| o.verb)
            .or_else(|| crate::zero::ZeroVerb::infer_from_input(arguments).ok());
        let verb = verb
            .and_then(|v| serde_json::to_value(v).ok())
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_else(|| "unknown".into());
        let status = match result {
            Ok(outcome) => outcome_status(outcome),
            Err(error) => engine_error_status(error).0,
        };
        let mut metrics = self.0.lock();
        *metrics.requests.entry((verb, status.as_u16())).or_default() += 1;
        if let Ok(outcome) = result {
            // Only count an actual gate verdict, never infer Proceed from 200.
            let tier = outcome.meta.get("tier").and_then(Value::as_str);
            let index = match tier {
                Some("Proceed") => Some(0),
                Some("Confirm") => Some(1),
                Some("Escalate") => Some(2),
                Some("HardStop") => Some(3),
                _ => None,
            };
            if let Some(index) = index {
                metrics.gates[index] += 1;
            }
        }
        metrics.latency_count += 1;
        metrics.latency_sum += elapsed;
        for (i, bound) in LATENCY_BUCKETS.iter().enumerate() {
            if elapsed <= *bound {
                metrics.latency_buckets[i] += 1;
            }
        }
    }

    fn render(&self) -> String {
        use std::fmt::Write;
        let metrics = self.0.lock();
        let mut text = String::from(
            "# HELP genzero_requests_total Completed engine calls by cognitive verb and result status.\n\
             # TYPE genzero_requests_total counter\n",
        );
        for ((verb, status), count) in &metrics.requests {
            writeln!(
                text,
                "genzero_requests_total{{verb=\"{verb}\",status=\"{status}\"}} {count}"
            )
            .unwrap();
        }
        text.push_str("# HELP genzero_http_requests_total HTTP responses by method and status (through response headers).\n# TYPE genzero_http_requests_total counter\n");
        for ((method, status), count) in &metrics.http_requests {
            writeln!(
                text,
                "genzero_http_requests_total{{method=\"{method}\",status=\"{status}\"}} {count}"
            )
            .unwrap();
        }
        text.push_str("# HELP genzero_gate_tier_total Completed gate verdicts by tier.\n# TYPE genzero_gate_tier_total counter\n");
        for (tier, count) in ["Proceed", "Tier1", "Tier2Escalate", "Tier3HardStop"]
            .iter()
            .zip(metrics.gates)
        {
            writeln!(text, "genzero_gate_tier_total{{tier=\"{tier}\"}} {count}").unwrap();
        }
        text.push_str("# HELP genzero_request_duration_seconds Engine call latency in seconds.\n# TYPE genzero_request_duration_seconds histogram\n");
        for (bound, count) in LATENCY_BUCKETS.iter().zip(metrics.latency_buckets) {
            writeln!(
                text,
                "genzero_request_duration_seconds_bucket{{le=\"{bound}\"}} {count}"
            )
            .unwrap();
        }
        writeln!(
            text,
            "genzero_request_duration_seconds_bucket{{le=\"+Inf\"}} {}",
            metrics.latency_count
        )
        .unwrap();
        writeln!(
            text,
            "genzero_request_duration_seconds_sum {}",
            metrics.latency_sum
        )
        .unwrap();
        writeln!(
            text,
            "genzero_request_duration_seconds_count {}",
            metrics.latency_count
        )
        .unwrap();
        text
    }
}

async fn metrics_handler(
    Extension(engine): Extension<Arc<PolymorphicZeroEngine>>,
) -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        engine.metrics.render(),
    )
}

async fn http_metrics_middleware(
    Extension(engine): Extension<Arc<PolymorphicZeroEngine>>,
    req: Request,
    next: Next,
) -> Response {
    let method = match req.method().as_str() {
        "GET" => "GET",
        "POST" => "POST",
        "PUT" => "PUT",
        "PATCH" => "PATCH",
        "DELETE" => "DELETE",
        "HEAD" => "HEAD",
        "OPTIONS" => "OPTIONS",
        "CONNECT" => "CONNECT",
        "TRACE" => "TRACE",
        _ => "OTHER",
    };
    let response = next.run(req).await;
    *engine
        .metrics
        .0
        .lock()
        .http_requests
        .entry((method, response.status().as_u16()))
        .or_default() += 1;
    response
}

async fn audit_ledger_handler(
    Extension(engine): Extension<Arc<PolymorphicZeroEngine>>,
) -> Json<Value> {
    let (leaf_count, root) = engine.audit_ledger();
    Json(json!({"leaf_count": leaf_count, "root": crate::mount::digest_hex(&root)}))
}

async fn audit_proof_handler(
    RestPath(index): RestPath<u64>,
    Extension(engine): Extension<Arc<PolymorphicZeroEngine>>,
) -> Result<Json<Value>, Response> {
    let proof = engine
        .audit_proof(index)
        .map_err(|_| rest_error(StatusCode::NOT_FOUND, "Audit proof not found"))?;
    let siblings: Vec<Value> = proof
        .siblings
        .iter()
        .map(|s| match s {
            ProofSibling::Left(h) => json!({"left": crate::mount::digest_hex(h)}),
            ProofSibling::Right(h) => json!({"right": crate::mount::digest_hex(h)}),
        })
        .collect();
    Ok(Json(
        json!({"leaf_index": proof.leaf_index, "leaf_count": proof.leaf_count,
        "leaf_hash": crate::mount::digest_hex(&proof.leaf_hash), "root": crate::mount::digest_hex(&proof.mmr_root), "siblings": siblings}),
    ))
}

/// `GENZERO_BRIDGE_REQUIRED=1` makes a server refuse to start when the
/// semantic scorer is not ready, instead of serving labeled fallbacks.
/// Read once, when the server is built.
fn bridge_required_from_env() -> bool {
    std::env::var("GENZERO_BRIDGE_REQUIRED")
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

#[derive(Clone, Default)]
pub struct AuthConfig {
    pub expected_token: Option<Arc<str>>,
}

pub struct McpServer {
    pub engine: Arc<PolymorphicZeroEngine>,
    pub auth_token: Option<String>,
    /// Refuse to start unless the semantic scorer is ready.
    pub bridge_required: bool,
    /// Native Rust closed-loop wiring (feedback sync + tuning patch poll).
    /// `None` disables both background tasks; `POST /v1/feedback` still
    /// buffers records in that case, they are just never shipped anywhere.
    pub closed_loop: Option<ClosedLoopConfig>,
}

impl Default for McpServer {
    fn default() -> Self {
        Self::from_engine(Arc::new(PolymorphicZeroEngine::new()))
    }
}

impl McpServer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Server over an engine built from `config`; construction errors (a
    /// native Qwen model that fails to load, an audit snapshot that cannot be
    /// opened) are returned instead of panicking.
    pub fn try_from_config(config: ZeroEngineConfig) -> Result<Self, ServiceError> {
        Ok(Self::from_engine(Arc::new(
            PolymorphicZeroEngine::try_from_config(config)?,
        )))
    }

    fn from_engine(engine: Arc<PolymorphicZeroEngine>) -> Self {
        Self {
            engine,
            auth_token: None,
            bridge_required: bridge_required_from_env(),
            closed_loop: None,
        }
    }

    pub fn with_bridge_required(mut self, required: bool) -> Self {
        self.bridge_required = required;
        self
    }

    pub fn with_auth_token(mut self, token: Option<String>) -> Self {
        self.auth_token = token;
        self
    }

    pub fn with_closed_loop_config(mut self, config: Option<ClosedLoopConfig>) -> Self {
        self.closed_loop = config;
        self
    }

    /// Process a single JSON-RPC 2.0 frame from an aligned buffer.
    pub async fn handle_jsonrpc_frame(&self, aligned_buf: &mut [u8]) -> String {
        // Strip trailing null padding bytes
        let trimmed_len = aligned_buf
            .iter()
            .rposition(|&b| b != 0 && b != b'\n' && b != b'\r' && b != b' ')
            .map(|pos| pos + 1)
            .unwrap_or(aligned_buf.len());

        // Retain uncorrupted byte copy in case in-situ simd_json parsing mutates buffer on failure
        let fallback_bytes = aligned_buf[..trimmed_len].to_vec();

        // Try ultra-fast simd_json first, fallback to serde_json
        let req = match simd_json::from_slice::<Value>(aligned_buf) {
            Ok(val) => val,
            Err(_) => match serde_json::from_slice::<Value>(&fallback_bytes) {
                Ok(val) => val,
                Err(e) => {
                    let err_resp = json!({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32700,
                            "message": format!("Parse error: {}", e)
                        },
                        "id": Value::Null
                    });
                    return err_resp.to_string();
                }
            },
        };

        let Some(req_object) = req.as_object() else {
            return jsonrpc_error(Value::Null, -32600, "Invalid Request").to_string();
        };
        if req_object.get("jsonrpc") != Some(&Value::String("2.0".to_owned())) {
            return jsonrpc_error(
                Value::Null,
                -32600,
                "Invalid Request: jsonrpc must be \"2.0\"",
            )
            .to_string();
        }
        if let Some(request_id) = req_object.get("id") {
            let valid_id = request_id.is_null() || request_id.is_string() || request_id.is_number();
            if !valid_id {
                return jsonrpc_error(
                    Value::Null,
                    -32600,
                    "Invalid Request: id must be string, number or null",
                )
                .to_string();
            }
        }
        let id = req.get("id").cloned().unwrap_or(Value::Null);
        let Some(method) = req.get("method").and_then(Value::as_str) else {
            return jsonrpc_error(id, -32600, "Invalid Request: method must be a string")
                .to_string();
        };
        if method == "tools/call" {
            let Some(params) = req.get("params").and_then(Value::as_object) else {
                return jsonrpc_error(
                    id,
                    -32602,
                    "Invalid params: tools/call params must be an object",
                )
                .to_string();
            };
            if params.get("name").and_then(Value::as_str).is_none() {
                return jsonrpc_error(
                    id,
                    -32602,
                    "Invalid params: tools/call name must be a string",
                )
                .to_string();
            }
            if let Some(arguments) = params.get("arguments") {
                if !arguments.is_object() {
                    return jsonrpc_error(
                        id,
                        -32602,
                        "Invalid params: tools/call arguments must be an object",
                    )
                    .to_string();
                }
            }
        }
        // JSON-RPC notifications, including `notifications/initialized`, do
        // not receive a response. Keep this behavior consistent for stdio;
        // the HTTP handler separately returns 204/202 acknowledgements.
        let notification = !req_object.contains_key("id");

        let resp = match method {
            "initialize" => {
                json!({
                    "jsonrpc": "2.0",
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "gen-zero",
                            "version": env!("CARGO_PKG_VERSION")
                        }
                    },
                    "id": id
                })
            }
            "ping" => json!({
                "jsonrpc": "2.0",
                "result": {},
                "id": id
            }),
            "tools/list" => {
                let mut zero_tool = json!({
                                "name": "zero",
                                "description": "Universal Gen-Zero Polymorphic Decision, Planning & Cognitive Primitive (0-Token Pure-Prefill). Supports 20 cognitive verbs: ask (alias decide), route, imagine, stream, grep, compact, entail, causal_fold, pipeline, simulate, what_if, audit, graph_deposit, graph_recall, graph_rag, graph_ppr, graph_prune, graph_evolve, graph_coarse_grain, graph_zoom.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "action": {
                                            "type": "string",
                                            "enum": ["ask", "route", "imagine", "stream", "grep", "compact", "entail", "causal_fold", "pipeline", "simulate", "what_if", "audit", "graph_deposit", "graph_recall", "graph_rag", "graph_ppr", "graph_prune", "graph_evolve", "graph_coarse_grain", "graph_zoom"],
                                            "description": "Optional explicit verb. If omitted, intent is deduced automatically."
                                        },
                                        "candidates": {
                                            "type": "array",
                                            "items": { "type": "string" },
                                            "description": "Candidate actions for 'ask' and 'what_if' verbs."
                                        },
                                        "questions": {
                                            "description": "Decision questions or prompts for 'ask' verb."
                                        },
                                        "tools": {
                                            "type": "array",
                                            "items": { "type": "string" },
                                            "description": "Candidate tools to prune for 'route' verb."
                                        },
                                        "task_goal": {
                                            "type": "string",
                                            "description": "Task goal for 'route' verb."
                                        },
                                        "candidate_actions": {
                                            "type": "array",
                                            "items": { "type": "string" },
                                            "description": "Candidate action names for 'imagine' verb."
                                        },
                                        "horizon": {
                                            "type": "integer",
                                            "description": "Lookahead depth for 'imagine'; rollout length (1..=256, refused outside) for 'simulate', 'what_if' and 'audit'."
                                        },
                                        "enforce_cpsat": {
                                            "type": "boolean",
                                            "description": "For ask/decide and imagine, keep caller safety masks enabled and optionally run the 0-1 ILP CP-SAT formal check. Setting false skips only the extra formal check; hard safety masks remain enforced."
                                        },
                                        "cognitive": {
                                            "type": "object",
                                            "description": "Numeric manifold request for 'stream' (state, window_start_ns, events, pins) or 'ask' (state, goal, window_start_ns, controls per candidate). Runs the mounted tangent SSM and geometry gate; refusals are typed Spec 25 codes."
                                        },
                                        "entailment": {
                                            "type": "object",
                                            "description": "Numeric request for 'entail': {passage, question}, each [H | R | S] coordinates of the topology preset the mount seals: compact_64d (H^32 x R^16 x S^15, 64 values), balanced_128d (H^64 x R^32 x S^31, 128), boolq_128d (H^80 x R^24 x S^23, 128) or extended_256d (H^160 x R^48 x S^47, 256). Any other width is refused with FiberMismatch. Runs the asymmetric Busemann containment test on the geometry the mount seals (assets `entailment` block). Optional scheme 2: add passage_events and question_events (equal-length [{time_ns, input: tangent coordinates of the same width}] at the passage and question points) plus window_start_ns; the question-minus-transported-passage difference is scanned by the sealed `entailment.dynamics` SSM and moves the question point before the test. No text encoder; margin is not a calibrated probability."
                                        },
                                        "graph": {
                                            "type": "object",
                                            "description": "Request for the graph verbs on the engine's live LodGraph (the graph the PolicyGate reads for revocations). Name nodes by entity_id or by action (entity id = the action's id). graph_deposit: {nodes?: [{entity_id | action, label, band?: 0..3 (absent: the band the coordinate implies, from its hyperbolic depth: the origin is band 3, the boundary band 0), status: hypothesized | validated | falsified, coord?: {hyperbolic: [4], spherical: [4], euclidean: [8]}, hdc?: [4 u64], confidence, payload?: knowledge text (<= 64 KiB, BLAKE3 digest stored), source_uri?, timestamp_ns? (default: deposit wall-clock time), aliases?: [<= 16 other names, synonyms or translations, each <= 256 bytes], embedding?: [16..8192 floats from the caller's embedding model, one dimension per graph]}] (give coord and hdc together, or neither: then the payload is projected by the graph's lexical n-gram SimHash projector, or with no payload the embedding by its deterministic dense random-sign/SimHash projection, and band is required; each alias is one more anchor for text queries, and nodes sharing an alias are linked by semantic edges both ways, counted in alias_link_edges), edges?: [{source: {entity_id | action}, target: {...}, type: validates | falsifies | causal_transition | semantic | coarse_grain | depends_on, weight}]}, one transaction, flushed into the CSR snapshot, rolled back whole on any error. graph_recall: {coord, hdc, top_k, crag_margin}. graph_rag: {query_text and/or query_vector | (coord, hdc), top_k: 1..32, crag_margin? (default 0), alpha?, max_iters?}: per track, HDC Hamming prefilter to 4*top_k candidates and product-geodesic rerank to top_k anchors, a node counted by its closest anchor (its own coordinate or an alias for text and coord; its embedding for query_vector, whose dimension must equal the graph's), then each track normalizes anchor distances by its maximum among the recalled top_k anchors (a zero maximum leaves distances at zero) and PPR is seeded with 1/(1+normalized_distance) along every edge in its direction; this candidate-relative normalization is reported as distance_normalization=per_track_max and is not semantic calibration (a single nonzero candidate has normalized distance 1); returns every anchor plus up to top_k diffusion-reached nodes, each with ppr_score, dimensionless anchor_distance (null when reached by diffusion), matched (primary | alias | embedding), matched_alias, aliases, confidence, payload, source_uri, timestamp_ns and payload_digest, and searchable_nodes: the live nodes the query could be compared with. Internal reflection evidence is excluded from general recall, anchors and diffusion hits. The text projector is lexical (shared n-grams), not a semantic embedding: a translation or paraphrase is found only through an alias, an edge, or a query_vector from the caller's embedding model; the engine holds no such model. The dense projection is deterministic fixed random-sign/SimHash plus a 16-row real-valued sketch; it performs no learned manifold alignment or isometry, and its relation to embedding angle is weakly statistical with no cosine, distance or ranking guarantee. graph_ppr: {seeds: [{entity_id | action, weight}], top_k, alpha?, max_iters?, tolerance?}. graph_prune: {entity_id | action, dry_run?, beta?, gamma?, tolerance?, theta_lo?, theta_hi?, max_steps?}: records evidence against the node (confidence pinned to 0, entity revoked), then evolves every confidence to the fixed point of c = (1 - beta) prior + beta max(0, P+ c - gamma P- c), clamped to [0, 1], P+ the row-normalized DependsOn/CausalTransition/CoarseGrain weights into each node (a node without them is supported by its own prior), P- the row-normalized falsifies weights (gamma 0 leaves them out). The dependency graph is split into strongly connected components (Tarjan) solved sources first: a node outside every cycle takes one evaluation, a cycle is iterated alone; a cycle whose Lipschitz bound beta * max row sum of (P+ + gamma P-) inside it is >= 1 is refused with FixedPointNotContractive (beta < 1 / (1 + gamma) always contracts); nodes whose confidence falls below theta_lo are falsified and their entities revoked, which hard-stops those actions at the gate. The effect on a dependent shrinks with its prior, its other dependencies and its distance from the root: it is not a whole-subtree cascade. dry_run reports and rolls back. graph_evolve: {retract?: [{entity_id | action}], dry_run?, beta?, gamma?, tolerance?, theta_lo?, theta_hi?, max_steps?}: withdraws such evidence, then evolves; nodes above theta_hi become validated and lose a revocation the evolution made, nodes between the thresholds keep their status. graph_coarse_grain: {members: [{entity_id | action}], entity_id | action, coord, hdc, dry_run?}: inserts a summary node on the band its coord implies, which must be strictly coarser than every member, links each member to it with a coarse_grain edge (weight 1), makes it the members' parent and flushes; refused (nothing changed) for a falsified, revoked or already-parented member. graph_zoom: {entity_id | action, direction: in | out | to_coord, dry_run?}: moves the node one band, or to the band its coord implies; refused when a member would reach its summary's band or a summary its member's. A coarse_grain edge must go from a finer band to a coarser one. Defaults beta 0.85, gamma 1, tolerance 1e-6, theta_lo 0.2, theta_hi 0.8, max_steps 10000 (uncalibrated presets, echoed). A run that does not converge inside its step bound is refused with FixedPointDiverged and changes nothing. The fixed_point report echoes gamma and the block counts scc_count, trivial_scc_count, cyclic_scc_count, max_scc_size, plus falsification_edges, contraction (largest cycle Lipschitz bound) and node_updates. A deposited node's confidence is its prior. Distances and coordinate domains use the graph geometry {curvature, radius, alpha_h, alpha_e, alpha_s} set at startup by GENZERO_GRAPH_GEOMETRY (unit when unset), echoed as graph.geometry: hyperbolic must satisfy curvature * |x|^2 < 1, spherical is a direction scaled to the radius. Durable when GENZERO_GRAPH_PERSIST_DIR is configured (MicroVM data disk: /var/lib/gen3/lodgraph): commits sync changed blocks and atomically replace a SHA-256-checked manifest; startup refuses corrupt snapshots. Without that setting the graph is process-local."
                                        },
                                        "causal_fold": {
                                            "type": "object",
                                            "description": "Request for 'causal_fold': {edges: [relation ids], genders: [node genders, edges.len() + 1 of them, each Male/Female/Unknown], axioms?: [{r1, r2, gender, result: [relation ids]}], strategy?: \"chart\" (S3, default: a CYK-style chart over every bracketing), \"tiered\" (Band 0 -> Band 1) or \"left\" (S1, strict left fold)}. The axiom table T(r1, r2, gender) is supplied entirely by the caller; missing `axioms` means an empty table, so any chain of 2 or more edges refuses. A conflict key (two or more results for one (r1, r2, gender)) makes left refuse; chart carries every candidate forward and concludes only if exactly one relation survives at the root, refusing when two or more do. strategy \"soft\": {edges | soft_sets: [[{relation, p}]], genders, weights, axioms?, entropy_threshold}: soft kernels are the relative frequencies of weights, leaves are deltas on edges or the given distributions, the chain is left-folded by (a ⊗ b)(r) = Σ a(r1) b(r2) T(r1, r2, g → r) with unclosed mass kept, and the fold refuses when the entropy (nats) exceeds entropy_threshold, when the unclosed mass is at least the top relation's, or when the top two tie. entropy_threshold has no default and is not calibrated. Caps: 64 edges, 4096 axioms, 64 distinct result relation ids."
                                        },
                                        "pipeline": {
                                            "type": "object",
                                            "description": "Request for 'pipeline': {op, state: 1024 numbers, ...} on the latent world model. op=simulate {actions: [ids], horizon?}; what_if {candidates: [ids], horizon? (default 5)}; audit_action {action, horizon? (5), continuation_actions?, warn_risk? (0.3)}; decide {candidates: [ids, at most 16], mode: auto|mcts|mpc_cem|astar|manifold_gflownet|cfr_nash|reflex, entropy: [0,1], return_trajectory?, horizon? (5)}. PolicyGate hard stops are pruned first; no legal action, a divergent state or a model without safety estimates is refused. safe_prob of the default model is an uncalibrated margin to its termination norm."
                                        },
                                        "lines": {
                                            "type": "array",
                                            "items": { "type": "string" },
                                            "description": "Text lines searched literally by 'grep'."
                                        },
                                        "mount_version": {
                                            "type": "integer",
                                            "description": "Refuse (EpochMismatch) unless this is the mounted generation."
                                        },
                                        "stream_id": {
                                            "type": "string",
                                            "description": "Stream identifier for 'stream' verb."
                                        },
                                        "paths": {
                                            "type": "array",
                                            "items": { "type": "string" },
                                            "description": "Refused by 'grep' (no file backend): send `lines` instead."
                                        },
                                        "expr": {
                                            "type": "string",
                                            "description": "Refused by 'grep' (no boolean evaluator yet)."
                                        },
                                        "query": {
                                            "type": "string",
                                            "description": "Literal, case-sensitive substring searched by 'grep'."
                                        },
                                        "text": {
                                            "type": "string",
                                            "description": "Text to compress for 'compact' verb."
                                        },
                                        "head_lines": {
                                            "type": "integer",
                                            "description": "Leading lines to retain verbatim for 'compact' verb."
                                        },
                                        "tail_lines": {
                                            "type": "integer",
                                            "description": "Trailing lines to retain verbatim for 'compact' verb."
                                        },
                                        "messages": {
                                            "type": "array",
                                            "items": { "type": "object" },
                                            "description": "Messages to losslessly compact for 'compact' verb."
                                        }
                                    }
                                }
                });
                add_world_model_properties(&mut zero_tool);
                let causal_fold_tool = json!({
                                "name": "causal_fold",
                                "description": "Causal relation trace fold on the caller-supplied discrete relation semiring (Spec 24 §8.6.2): fold a chain of relation ids through a learned composition table under S1 (left), tiered dispatch or S3 (chart, default). The axiom table T(r1, r2, gender) is supplied entirely by the caller via `axioms`; a missing table means every composition is the empty set, so any chain of 2 or more edges refuses. A conflict key (two or more results for the same (r1, r2, gender)) makes left refuse; chart carries every candidate forward and concludes only if exactly one relation survives at the root (meta.causal_fold.conflict_keys counts conflict keys in the table), refusing when two or more survive. Strategy soft folds relation distributions built from weights and refuses above the caller's entropy_threshold (no default, uncalibrated), when unclosed mass dominates, or on a tie. Caps: 64 edges, 4096 axioms, 64 distinct result relation ids; above them the request is refused with InvalidParams.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "sets": {
                                            "type": "array", "minItems": 1, "maxItems": 64,
                                            "items": {
                                                "type": "array", "minItems": 1, "maxItems": 64, "uniqueItems": true,
                                                "items": { "type": "integer", "minimum": 0, "maximum": 65535 }
                                            },
                                            "description": "Alternative set chain for chart strategy; at most 64 IDs per set and 256 total. Requires gender and excludes nonempty edges/genders."
                                        },
                                        "gender": {
                                            "type": "string", "enum": ["Male", "Female", "Unknown"],
                                            "description": "Required uniform composition gender when sets is supplied."
                                        },
                                        "edges": {
                                            "type": "array",
                                            "items": { "type": "integer", "minimum": 0, "maximum": 65535 },
                                            "minItems": 1,
                                            "maxItems": 64,
                                            "description": "Chain of relation ids, one per edge."
                                        },
                                        "genders": {
                                            "type": "array",
                                            "items": { "type": "string", "enum": ["Male", "Female", "Unknown"] },
                                            "description": "Node genders along the chain; must have edges.len() + 1 entries."
                                        },
                                        "axioms": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "r1": { "type": "integer", "minimum": 0, "maximum": 65535 },
                                                    "r2": { "type": "integer", "minimum": 0, "maximum": 65535 },
                                                    "gender": { "type": "string", "enum": ["Male", "Female", "Unknown"] },
                                                    "result": { "type": "array", "items": { "type": "integer", "minimum": 0, "maximum": 65535 }, "minItems": 1, "uniqueItems": true }
                                                },
                                                "required": ["r1", "r2", "gender", "result"],
                                                "additionalProperties": false
                                            },
                                            "maxItems": 4096,
                                            "description": "The learned composition table T(r1, r2, gender) = result, supplied entirely by the caller. Missing entirely means an empty table, so any chain of 2 or more edges refuses."
                                        },
                                        "weights": {
                                            "type": "array", "minItems": 1, "maxItems": 4096,
                                            "description": "Required for weighted strategies; counts use plain relative frequencies (pseudo_count=0). Axioms remain the support table. Singleton output margin is the string infinity; confidence is an uncalibrated root-score share.",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "r1": { "type": "integer", "minimum": 0, "maximum": 65535 },
                                                    "r2": { "type": "integer", "minimum": 0, "maximum": 65535 },
                                                    "gender": { "type": "string", "enum": ["Male", "Female", "Unknown"] },
                                                    "relation": { "type": "integer", "minimum": 0, "maximum": 65535 },
                                                    "count": { "type": "integer", "minimum": 1 }
                                                },
                                                "required": ["r1", "r2", "gender", "relation", "count"],
                                                "additionalProperties": false
                                            }
                                        },
                                        "margin_threshold": { "type": "number", "minimum": 0, "default": 0 },
                                        "soft_sets": {
                                            "type": "array", "minItems": 1, "maxItems": 64,
                                            "items": {
                                                "type": "array", "minItems": 1, "maxItems": 64,
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "relation": { "type": "integer", "minimum": 0, "maximum": 65535 },
                                                        "p": { "type": "number", "exclusiveMinimum": 0, "maximum": 1 }
                                                    },
                                                    "required": ["relation", "p"],
                                                    "additionalProperties": false
                                                }
                                            },
                                            "description": "Strategy soft only: leaf distributions (each sums to 1 within 1e-4), used instead of edges; genders has soft_sets.len() + 1 entries."
                                        },
                                        "entropy_threshold": {
                                            "type": "number", "minimum": 0,
                                            "description": "Strategy soft only, and required by it: refuse when the folded distribution's entropy in nats (unclosed mass included) exceeds this. No default; not calibrated."
                                        },
                                        "semiring": {"type": "string", "enum": ["logprob", "tropical"], "description": "Band 1 semiring for tiered dispatch; defaults to logprob."},
                                        "strategy": {
                                            "type": "string",
                                            "enum": ["chart", "tiered", "left", "weighted_tropical", "weighted_logprob", "soft"],
                                            "description": "Fold strategy: chart (S3, default without weights), tiered (Band 0 -> Band 1), left (S1), weighted_tropical (best derivation), weighted_logprob (unnormalized sum over derivations), or soft (left fold of probability distributions with an entropy gate; requires weights and entropy_threshold). Tiered defaults when weights are supplied: chart Band 0 first, then weighted Band 1 on refusal, with explicit dispatch metadata. Tiered and weighted strategies use edges and genders; weighted strategies require weights."
                                        }
                                    },
                                    "anyOf": [{"required": ["edges", "genders"]}, {"required": ["sets", "gender"]}, {"required": ["soft_sets", "genders"]}]
                                }
                });
                let pipeline_tool = json!({
                    "name": "pipeline",
                    "description": "Latent world-model pipeline (Rust ProductionPipeline): simulate a fixed plan, what_if over candidate first moves with trap detection, audit_action (Approved / WarnHazard / RejectLethal with risk_score), or decide in mode auto (K-MoE router), mcts, mpc_cem, astar, manifold_gflownet, cfr_nash or reflex (gated one-step). PolicyGate hard stops are pruned before any engine runs. Same body as the `pipeline` block of `zero`.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "op": { "type": "string", "enum": ["simulate", "what_if", "audit_action", "decide"] },
                            "state": { "type": "array", "items": { "type": "number" }, "minItems": 1024, "maxItems": 1024 },
                            "actions": { "type": "array", "items": { "type": "integer", "minimum": 0 } },
                            "candidates": { "type": "array", "items": { "type": "integer", "minimum": 0 } },
                            "action": { "type": "integer", "minimum": 0 },
                            "continuation_actions": { "type": "array", "items": { "type": "integer", "minimum": 0 } },
                            "horizon": { "type": "integer", "minimum": 1, "maximum": 128 },
                            "warn_risk": { "type": "number", "exclusiveMinimum": 0, "maximum": 1 },
                            "auto_reflect": { "type": "boolean", "description": "Opt in to persistent graph quarantine from simulate, what_if or audit_action failure observations; default false." },
                            "mode": { "type": "string", "enum": ["auto", "mcts", "mpc_cem", "astar", "manifold_gflownet", "cfr_nash", "reflex"] },
                            "entropy": { "type": "number", "minimum": 0, "maximum": 1 },
                            "return_trajectory": { "type": "boolean" }
                        },
                        "required": ["op", "state"]
                    }
                });
                json!({
                    "jsonrpc": "2.0",
                    "result": {"tools": [zero_tool, causal_fold_tool, pipeline_tool]},
                    "id": id
                })
            }
            "tools/call" => {
                let params = req.get("params").cloned().unwrap_or(Value::Null);
                let tool_name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
                let arguments = params
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(|| json!({}));

                // Both tools reach the same `zero` engine; `causal_fold` is
                // just `zero` with the verb and block already wrapped, so a
                // flat MCP call does not need to know the wire shape `zero`
                // itself expects.
                let engine_args = match tool_name {
                    "zero" => Some(arguments),
                    "causal_fold" => {
                        Some(json!({"action": "causal_fold", "causal_fold": arguments}))
                    }
                    "pipeline" => Some(json!({"action": "pipeline", "pipeline": arguments})),
                    _ => None,
                };

                match engine_args {
                    Some(args) => match self.engine.execute(&args).await {
                        Ok(outcome) => {
                            json!({
                                "jsonrpc": "2.0",
                                "result": {
                                    "content": outcome.content,
                                    "isError": outcome.is_error,
                                    "_meta": outcome.meta
                                },
                                "id": id
                            })
                        }
                        Err(e) => {
                            json!({
                                "jsonrpc": "2.0",
                                "result": {
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": format!("Error: {}", e)
                                        }
                                    ],
                                    "isError": true,
                                    "_meta": {
                                        "error_code": -32004,
                                        "details": e.to_string()
                                    }
                                },
                                "id": id
                            })
                        }
                    },
                    None => json!({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32601,
                            "message": format!("Method or tool '{}' not found", tool_name)
                        },
                        "id": id
                    }),
                }
            }
            _ => {
                json!({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": format!("Method '{}' not found", method)
                    },
                    "id": id
                })
            }
        };

        if notification {
            String::new()
        } else {
            resp.to_string()
        }
    }

    /// Check the semantic backend once at startup and say loudly what was
    /// found. The native backend is ready once loaded. For the Python bridge,
    /// a port held by the wrong service (HTTP 404) is an error, not a quiet
    /// fallback. With `GENZERO_BRIDGE_REQUIRED=1` anything but a ready scorer
    /// stops the server.
    pub async fn check_semantic(&self, serving_port: Option<u16>) -> Result<(), ServiceError> {
        let bridge = match self.engine.semantic() {
            Some(SemanticBackend::Native(native)) => {
                let info = native.info();
                tracing::info!(
                    model = %info.source.display(),
                    sha256 = %info.weights_sha256,
                    "semantic backend: native Qwen in process (no Python runtime)"
                );
                return Ok(());
            }
            Some(SemanticBackend::Remote(client)) => client,
            None => {
                tracing::warn!(
                    "no semantic backend (GENZERO_QWEN_MODEL_PATH unset, GENZERO_PYTHON_ENDPOINT=off); \
                     ask/route/imagine return labeled fallbacks and every request with text is escalated"
                );
                return if self.bridge_required {
                    Err(ServiceError::Core(
                        "GENZERO_BRIDGE_REQUIRED=1 but no semantic backend is configured".into(),
                    ))
                } else {
                    Ok(())
                };
            }
        };
        if let Some(port) = serving_port.filter(|&p| bridge.targets_local_port(p)) {
            tracing::error!(
                "GENZERO_PYTHON_ENDPOINT={} points at this server's own port {}; \
                 run the Python scorer on its own port (`python3 -m gen_zero.cli semantic`, default 8995)",
                bridge.endpoint(),
                port
            );
        }
        let report = bridge.probe().await;
        match &report.health {
            BridgeHealth::Ready { backbone_loaded } => tracing::info!(
                endpoint = %report.endpoint,
                backbone_loaded,
                "semantic bridge ready"
            ),
            BridgeHealth::WrongService { detail } => tracing::error!(
                endpoint = %report.endpoint,
                "semantic bridge BLOCKED: wrong service on the endpoint: {detail}. \
                 ask/route/imagine will return labeled fallbacks."
            ),
            BridgeHealth::Unreachable { detail } => tracing::error!(
                endpoint = %report.endpoint,
                "semantic bridge unreachable: {detail}. Start `python3 -m gen_zero.cli semantic`; \
                 until then ask/route/imagine return labeled fallbacks."
            ),
        }
        if self.bridge_required && !report.health.is_ready() {
            return Err(ServiceError::Core(format!(
                "GENZERO_BRIDGE_REQUIRED=1 and the semantic bridge at {} is not ready",
                report.endpoint
            )));
        }
        Ok(())
    }

    /// Run zero-allocation Stdio loop using simd-json.
    pub async fn run_stdio(&self) -> Result<(), ServiceError> {
        self.check_semantic(None).await?;
        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut reader = tokio::io::BufReader::new(stdin);
        let mut line_buf = String::new();

        // Single outer preallocated buffer with 64-byte alignment and SIMDJSON_PADDING
        let mut aligned_buf: Vec<u8> = Vec::with_capacity(128 * 1024);

        while read_stdio_frame(&mut reader, &mut line_buf).await? > 0 {
            aligned_buf.clear();
            aligned_buf.extend_from_slice(line_buf.as_bytes());
            // Ensure simd-json SIMD padding bytes at the end
            aligned_buf.resize(aligned_buf.len() + simd_json::SIMDJSON_PADDING, 0);

            let mut resp = self.handle_jsonrpc_frame(&mut aligned_buf).await;
            if resp.is_empty() {
                line_buf.clear();
                continue;
            }
            resp.push('\n');
            stdout.write_all(resp.as_bytes()).await?;
            stdout.flush().await?;
            line_buf.clear();
        }

        Ok(())
    }

    /// Build Axum Router for SSE & REST gateway.
    pub fn build_router(engine: Arc<PolymorphicZeroEngine>, auth_token: Option<String>) -> Router {
        Self::router_with_sessions(
            engine,
            auth_token,
            SseSessionManager::default(),
            Arc::new(FeedbackBuffer::new()),
        )
    }

    fn router_with_sessions(
        engine: Arc<PolymorphicZeroEngine>,
        auth_token: Option<String>,
        sessions: SseSessionManager,
        feedback_buffer: Arc<FeedbackBuffer>,
    ) -> Router {
        let auth_config = AuthConfig {
            expected_token: auth_token.map(|t| Arc::from(t.trim())),
        };
        Router::new()
            .route("/health", get(health_handler))
            .route("/healthz", get(readiness_handler))
            .route("/ready", get(readiness_handler))
            .route("/metrics", get(metrics_handler))
            .route("/sse", get(sse_handler))
            .route("/message", post(message_handler))
            // Keep the plural spelling accepted by older MCP clients. The
            // endpoint advertised by this server remains `/message`.
            .route("/messages", post(message_handler))
            .route("/v1/decisions", post(legacy_decisions_handler))
            .route("/v1/plan", post(legacy_decisions_handler))
            .route("/audit/ledger", get(audit_ledger_handler))
            .route("/audit/ledger/:index", get(audit_proof_handler))
            .route("/v1/causal_fold", post(causal_fold_handler))
            .route("/v1/pipeline/:op", post(pipeline_handler))
            .route("/v1/simulate", post(simulate_handler))
            .route("/v1/what_if", post(what_if_handler))
            .route("/v1/audit_action", post(audit_action_handler))
            .route("/v1/mounts", post(publish_mount_handler))
            .route("/v1/feedback", post(feedback_handler))
            .route(
                "/v1/decisions/stream",
                get(sse_handler).post(legacy_decisions_handler),
            )
            .fallback(not_found_handler)
            .method_not_allowed_fallback(method_not_allowed_handler)
            .layer(middleware::from_fn(auth_middleware))
            .layer(middleware::from_fn(http_metrics_middleware))
            .layer(Extension(engine))
            .layer(Extension(sessions))
            .layer(Extension(auth_config))
            .layer(Extension(feedback_buffer))
    }

    /// Run Tokio + Axum SSE/REST server on specified SocketAddr. When
    /// `closed_loop` names a tuning endpoint, this also spawns the feedback
    /// syncer and patch poller background tasks for the lifetime of the
    /// server, and aborts them once the listener stops draining.
    pub async fn run_sse(&self, addr: SocketAddr) -> Result<(), ServiceError> {
        self.check_semantic(Some(addr.port())).await?;
        let sessions = SseSessionManager::default();
        let feedback_buffer = Arc::new(FeedbackBuffer::new());
        let closed_loop_tasks = self.closed_loop.clone().map(|config| {
            let shutdown_rx = sessions.shutdown.subscribe();
            let syncer =
                spawn_feedback_syncer(config.clone(), feedback_buffer.clone(), shutdown_rx.clone());
            let current_version = Arc::new(ArcSwap::from_pointee(String::new()));
            let poller = spawn_patch_poller(config, current_version, shutdown_rx);
            (syncer, poller)
        });
        let app = Self::router_with_sessions(
            self.engine.clone(),
            self.auth_token.clone(),
            sessions.clone(),
            feedback_buffer,
        );
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        let result = serve_with_shutdown(
            listener,
            app,
            sessions,
            shutdown_signal(),
            Duration::from_secs(5),
        )
        .await;
        if let Some((syncer, poller)) = closed_loop_tasks {
            syncer.abort();
            poller.abort();
        }
        result?;
        Ok(())
    }
}

const MAX_STDIO_FRAME_BYTES: usize = 2 * 1024 * 1024;

/// Read at most one frame plus a sentinel byte, even without a newline.
/// An oversized frame terminates stdio rather than parsing a truncated request.
async fn read_stdio_frame<R: tokio::io::AsyncBufRead + Unpin>(
    reader: &mut R,
    frame: &mut String,
) -> std::io::Result<usize> {
    frame.clear();
    let size = reader
        .take((MAX_STDIO_FRAME_BYTES + 1) as u64)
        .read_line(frame)
        .await?;
    if size > MAX_STDIO_FRAME_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "stdio frame exceeds 2 MiB limit",
        ));
    }
    Ok(size)
}

async fn serve_with_shutdown(
    listener: tokio::net::TcpListener,
    app: Router,
    sessions: SseSessionManager,
    shutdown: impl std::future::Future<Output = ()>,
    drain_timeout: Duration,
) -> std::io::Result<()> {
    // Axum owns detached connection tasks: dropping its serve future alone
    // does not drop an in-flight handler. Cancel those handlers at the deadline.
    let (force_stop, force_stopped) = watch::channel(false);
    let app = app.layer(middleware::from_fn(move |request: Request, next: Next| {
        let mut stopped = force_stopped.clone();
        async move {
            tokio::select! {
                biased;
                _ = stopped.wait_for(|stop| *stop) => {
                    rest_error(StatusCode::SERVICE_UNAVAILABLE, "Server shutting down")
                }
                response = next.run(request) => response,
            }
        }
    }));
    let mut cancelled = sessions.shutdown.subscribe();
    let server = axum::serve(listener, app).with_graceful_shutdown(async move {
        let _ = cancelled.wait_for(|stopped| *stopped).await;
    });
    let server = std::future::IntoFuture::into_future(server);
    tokio::pin!(server);
    tokio::select! {
        result = &mut server => return result,
        () = shutdown => sessions.shutdown(),
    }
    match tokio::time::timeout(drain_timeout, server).await {
        Ok(result) => result,
        Err(_) => {
            force_stop.send_replace(true);
            tracing::warn!("HTTP drain deadline reached; cancelling remaining handlers");
            Ok(())
        }
    }
}

fn rest_error(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({"error": message.into(), "code": status.as_u16()})),
    )
        .into_response()
}

async fn not_found_handler() -> Response {
    rest_error(StatusCode::NOT_FOUND, "Route not found")
}

async fn method_not_allowed_handler() -> Response {
    rest_error(StatusCode::METHOD_NOT_ALLOWED, "Method not allowed")
}

/// Normalize REST JSON extractor rejections, including invalid content types.
struct RestJson(Value);

#[axum::async_trait]
impl<S: Send + Sync> axum::extract::FromRequest<S> for RestJson {
    type Rejection = Response;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        Json::<Value>::from_request(request, state)
            .await
            .map(|Json(value)| Self(value))
            .map_err(|error| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": error.body_text(), "code": 400})),
                )
                    .into_response()
            })
    }
}

struct RestPath<T>(T);

#[axum::async_trait]
impl<S, T> axum::extract::FromRequestParts<S> for RestPath<T>
where
    S: Send + Sync,
    T: serde::de::DeserializeOwned + Send,
{
    type Rejection = Response;

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        state: &S,
    ) -> Result<Self, Self::Rejection> {
        Path::<T>::from_request_parts(parts, state)
            .await
            .map(|Path(value)| Self(value))
            .map_err(|error| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": error.body_text(), "code": 400})),
                )
                    .into_response()
            })
    }
}

fn extract_query_token(query_str: &str) -> Option<&str> {
    for pair in query_str.split('&') {
        if let Some((key, val)) = pair.split_once('=') {
            if key == "token" || key == "api_key" {
                return Some(val);
            }
        }
    }
    None
}

fn extract_query_session_id(query_str: &str) -> Option<&str> {
    for pair in query_str.split('&') {
        if let Some((key, val)) = pair.split_once('=') {
            if key == "session_id" {
                return Some(val);
            }
        }
    }
    None
}

async fn auth_middleware(
    Extension(auth): Extension<AuthConfig>,
    Extension(sessions): Extension<SseSessionManager>,
    req: Request,
    next: Next,
) -> Result<Response, Response> {
    let path = req.uri().path();
    if matches!(path, "/health" | "/healthz" | "/ready") {
        return Ok(next.run(req).await);
    }

    // Active SSE sessions were already authenticated during initial GET /sse.
    // Subsequent POST /message or /messages requests associated with an active
    // session are permitted.
    if matches!(path, "/message" | "/messages") {
        if let Some(session_id) = req.uri().query().and_then(extract_query_session_id) {
            if sessions.contains(session_id) {
                return Ok(next.run(req).await);
            }
        }
    }

    if let Some(expected) = &auth.expected_token {
        let from_bearer = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|h| h.to_str().ok())
            .and_then(|s| {
                if let Some(token) = s.strip_prefix("Bearer ") {
                    Some(token.trim())
                } else if let Some(token) = s.strip_prefix("bearer ") {
                    Some(token.trim())
                } else {
                    None
                }
            });

        let from_x_api = req
            .headers()
            .get("x-api-key")
            .and_then(|h| h.to_str().ok())
            .map(|s| s.trim());

        let from_query = req.uri().query().and_then(extract_query_token);

        let provided = from_bearer.or(from_x_api).or(from_query);

        match provided {
            Some(token) if constant_time_token_match(token, expected) => {
                // Authorized
            }
            _ => {
                let err_body = json!({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32001,
                        "message": "Unauthorized: invalid or missing API token"
                    }
                });
                let mut resp = (StatusCode::UNAUTHORIZED, Json(err_body)).into_response();
                resp.headers_mut().insert(
                    header::WWW_AUTHENTICATE,
                    HeaderValue::from_static("Bearer realm=\"gen-zero\""),
                );
                return Err(resp);
            }
        }
    }

    Ok(next.run(req).await)
}

/// Liveness never waits on external dependencies or asset loading.
async fn health_handler() -> Json<Value> {
    Json(json!({"status": "ok", "service": "gen-zero"}))
}

/// Readiness requires a reachable scorer and valid assets in the default mount.
async fn readiness_handler(Extension(engine): Extension<Arc<PolymorphicZeroEngine>>) -> Response {
    let bridge = match engine.semantic() {
        None => json!({"status": "disabled"}),
        Some(SemanticBackend::Native(native)) => native.health(),
        Some(SemanticBackend::Remote(bridge)) => {
            let fresh = bridge
                .last_health()
                .filter(|r| now_ms().saturating_sub(r.probed_at_ms) < HEALTH_PROBE_TTL_MS);
            let mut report = match fresh {
                Some(report) => serde_json::to_value(report).unwrap_or(Value::Null),
                None => match tokio::time::timeout(Duration::from_secs(3), bridge.probe()).await {
                    Ok(report) => serde_json::to_value(report).unwrap_or(Value::Null),
                    Err(_) => {
                        json!({"status": "unreachable", "detail": "readiness probe timed out"})
                    }
                },
            };
            report["backend"] = json!("python_http");
            report
        }
    };
    // Decode off the async executor: loading/validating a large asset must not
    // prevent the independent liveness route from responding.
    let assets = tokio::task::spawn_blocking(move || {
        let key = MountKey::new(crate::zero::DEFAULT_TENANT, crate::zero::DEFAULT_WORKSPACE);
        match engine.mounts().load(&key) {
            Ok(snapshot) => match crate::cognitive::CognitiveAssets::from_snapshot(&snapshot) {
                Ok(_) => json!({"status": "loaded", "version": snapshot.version().0}),
                Err(e) => json!({"status": "unavailable", "detail": e.detail}),
            },
            Err(e) => json!({"status": "unavailable", "detail": e.to_string()}),
        }
    })
    .await
    .unwrap_or_else(|_| json!({"status": "unavailable"}));
    let ready = bridge["status"] == "ready" && assets["status"] == "loaded";
    (
        if ready {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        },
        Json(json!({
            "status": if ready { "ok" } else { "degraded" },
            "service": "gen-zero",
            "semantic_bridge": bridge,
            "cognitive_assets": assets,
        })),
    )
        .into_response()
}

/// Stop accepting connections, then let Axum drain in-flight requests.
async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            tracing::error!(%error, "could not install Ctrl-C handler");
        }
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(error) => tracing::error!(%error, "could not install SIGTERM handler"),
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
    tracing::info!("shutdown requested; draining in-flight requests");
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// Handlers for Axum routes
async fn sse_handler(
    Extension(sessions): Extension<SseSessionManager>,
    auth: Option<Extension<AuthConfig>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (session_id, receiver) = sessions.create();
    let endpoint = match auth.and_then(|a| a.0.expected_token) {
        Some(token) => format!("/message?session_id={session_id}&token={token}"),
        None => format!("/message?session_id={session_id}"),
    };
    let cancelled = sessions.shutdown.subscribe();
    let guard = SseSessionGuard {
        manager: sessions,
        session_id,
    };
    // The endpoint event is emitted first. Subsequent JSON-RPC responses are
    // sent by `/message?session_id=...` through this session's bounded queue.
    // Keeping the guard in the unfold state also removes a session when the
    // client disconnects and drops the stream before the channel closes.
    let stream = stream::unfold(
        (true, endpoint, receiver, guard, cancelled),
        |(first, endpoint, mut receiver, guard, mut cancelled)| async move {
            if *cancelled.borrow() {
                return None;
            }
            if first {
                let event = Event::default().event("endpoint").data(endpoint.clone());
                return Some((Ok(event), (false, endpoint, receiver, guard, cancelled)));
            }
            let payload = tokio::select! {
                biased;
                _ = cancelled.wait_for(|stopped| *stopped) => { receiver.close(); return None; }
                payload = receiver.recv() => payload?,
            };
            let event = Event::default().event("message").data(payload.to_string());
            Some((Ok(event), (false, endpoint, receiver, guard, cancelled)))
        },
    );
    Sse::new(stream).keep_alive(KeepAlive::default().interval(Duration::from_secs(15)))
}

/// Status of a finished `zero` call. A typed refusal keeps its own status;
/// other errors map by the MCP error code the engine already set.
fn outcome_status(outcome: &ZeroToolOutcome) -> StatusCode {
    if let Some(r) = &outcome.rejection {
        return StatusCode::from_u16(r.http_status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    }
    if !outcome.is_error {
        return StatusCode::OK;
    }
    match outcome.meta.get("error_code").and_then(Value::as_i64) {
        Some(-32001) => StatusCode::FORBIDDEN,
        Some(-32002) => StatusCode::PRECONDITION_REQUIRED,
        Some(-32003) => StatusCode::SERVICE_UNAVAILABLE,
        _ => StatusCode::UNPROCESSABLE_ENTITY,
    }
}

/// Typed error object for an `isError` outcome.
fn outcome_error(outcome: &ZeroToolOutcome) -> Value {
    let message = outcome
        .content
        .iter()
        .map(|b| b.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    match &outcome.rejection {
        Some(r) => json!({"code": r.code, "stage": r.stage, "message": message}),
        None => {
            json!({"code": outcome.meta.get("error_code").cloned().unwrap_or(Value::Null), "message": message})
        }
    }
}

/// An engine error (not an outcome) is a failed request, never an
/// ask-shaped fake outcome.
fn engine_error_status(e: &ServiceError) -> (StatusCode, &'static str) {
    match e {
        ServiceError::Overloaded => (StatusCode::SERVICE_UNAVAILABLE, "Overloaded"),
        ServiceError::InvalidVerb(_)
        | ServiceError::JsonRpc(_)
        | ServiceError::MethodNotFound(_) => (StatusCode::BAD_REQUEST, "InvalidParams"),
        ServiceError::SafetyRejected(_) => (StatusCode::FORBIDDEN, "SafetyRejected"),
        ServiceError::ConfirmationRequired(_) => {
            (StatusCode::PRECONDITION_REQUIRED, "ConfirmationRequired")
        }
        ServiceError::Core(_) | ServiceError::Io(_) => {
            (StatusCode::INTERNAL_SERVER_ERROR, "InternalError")
        }
    }
}

fn engine_error_parts(e: &ServiceError) -> (StatusCode, Value) {
    let (status, code) = engine_error_status(e);
    (
        status,
        json!({"isError": true, "error": {"code": code, "message": e.to_string()}}),
    )
}

fn engine_error_response(e: &ServiceError) -> Response {
    let (status, body) = engine_error_parts(e);
    tracing::warn!(%status, "zero call failed: {e}");
    (status, Json(body)).into_response()
}

fn jsonrpc_error(id: Value, code: i64, message: impl Into<String>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "error": {"code": code, "message": message.into()},
        "id": id,
    })
}

/// Reuse the stdio dispatcher for HTTP JSON-RPC. Keeping one dispatcher is
/// important: initialize, tools/list and tools/call have identical wire
/// shapes over both transports.
async fn dispatch_jsonrpc_payload(
    engine: Arc<PolymorphicZeroEngine>,
    payload: Value,
) -> Option<Value> {
    let id = payload.get("id").cloned().unwrap_or(Value::Null);
    let mut frame = match serde_json::to_vec(&payload) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Some(jsonrpc_error(
                id,
                -32600,
                format!("Invalid Request: {error}"),
            ));
        }
    };
    frame.resize(frame.len() + simd_json::SIMDJSON_PADDING, 0);
    let server = McpServer {
        engine,
        auth_token: None,
        bridge_required: false,
        closed_loop: None,
    };
    let response = server.handle_jsonrpc_frame(&mut frame).await;
    if response.is_empty() {
        None
    } else {
        Some(serde_json::from_str(&response).unwrap_or_else(|error| {
            jsonrpc_error(
                id,
                -32603,
                format!("Internal JSON-RPC serialization error: {error}"),
            )
        }))
    }
}

fn is_jsonrpc_request(payload: &Value) -> bool {
    !payload.is_object()
        || ["method", "jsonrpc", "id", "params"]
            .iter()
            .any(|field| payload.get(field).is_some())
}

/// HTTP status for a JSON-RPC response. MCP tool failures remain represented
/// in `result.isError`, but the HTTP envelope must still communicate failure
/// to callers that only inspect the status line.
fn jsonrpc_response_status(response: &Value) -> StatusCode {
    if let Some(error) = response.get("error") {
        return match error.get("code").and_then(Value::as_i64) {
            Some(-32001) => StatusCode::FORBIDDEN,
            Some(-32002) => StatusCode::PRECONDITION_REQUIRED,
            Some(-32003) => StatusCode::SERVICE_UNAVAILABLE,
            Some(-32004) => StatusCode::INTERNAL_SERVER_ERROR,
            Some(-32700 | -32600 | -32601 | -32602) => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
    }
    let Some(result) = response.get("result") else {
        return StatusCode::INTERNAL_SERVER_ERROR;
    };
    if result.get("isError") != Some(&Value::Bool(true)) {
        return StatusCode::OK;
    }
    if let Some(status) = result
        .pointer("/_meta/reject/http_status")
        .and_then(Value::as_u64)
        .and_then(|status| u16::try_from(status).ok())
        .and_then(|status| StatusCode::from_u16(status).ok())
    {
        return status;
    }
    match result.pointer("/_meta/error_code").and_then(Value::as_i64) {
        Some(-32001) => StatusCode::FORBIDDEN,
        Some(-32002) => StatusCode::PRECONDITION_REQUIRED,
        Some(-32003) => StatusCode::SERVICE_UNAVAILABLE,
        Some(-32004) => StatusCode::INTERNAL_SERVER_ERROR,
        _ => StatusCode::UNPROCESSABLE_ENTITY,
    }
}

async fn message_handler(
    Extension(engine): Extension<Arc<PolymorphicZeroEngine>>,
    Extension(sessions): Extension<SseSessionManager>,
    query: Result<Query<MessageQuery>, axum::extract::rejection::QueryRejection>,
    payload: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Query(query) = match query {
        Ok(query) => query,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(jsonrpc_error(
                    Value::Null,
                    -32600,
                    format!("Invalid query: {error}"),
                )),
            )
                .into_response();
        }
    };
    // Resolve the session before parsing or executing anything. An unknown
    // session must not cause an engine call (or an audit append) as a side
    // effect. The sender is cloned so a bounded permit can be reserved before
    // the potentially expensive engine dispatch.
    let session_sender = match query.session_id.as_deref() {
        None => None,
        Some(session_id) => match sessions.sender(session_id) {
            Some(sender) => Some(sender),
            None => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(jsonrpc_error(
                        Value::Null,
                        -32004,
                        "SSE session not found or expired",
                    )),
                )
                    .into_response()
            }
        },
    };

    let payload = match payload {
        Ok(Json(payload)) => payload,
        Err(error) => {
            let response = jsonrpc_error(Value::Null, -32700, format!("Parse error: {error}"));
            if let Some(sender) = session_sender {
                let permit = match tokio::time::timeout(
                    Duration::from_millis(100),
                    sender.clone().reserve_owned(),
                )
                .await
                {
                    Ok(Ok(permit)) => permit,
                    _ => {
                        return (
                            StatusCode::SERVICE_UNAVAILABLE,
                            Json(jsonrpc_error(
                                Value::Null,
                                -32004,
                                "SSE session is no longer accepting responses",
                            )),
                        )
                            .into_response()
                    }
                };
                permit.send(response);
                return StatusCode::ACCEPTED.into_response();
            }
            return (StatusCode::BAD_REQUEST, Json(response)).into_response();
        }
    };

    if is_jsonrpc_request(&payload) {
        let has_id = payload.get("id").is_some();
        let permit = if let Some(sender) = &session_sender {
            match tokio::time::timeout(Duration::from_millis(100), sender.clone().reserve_owned())
                .await
            {
                Ok(Ok(permit)) => Some(permit),
                _ => {
                    return (
                        StatusCode::SERVICE_UNAVAILABLE,
                        Json(jsonrpc_error(
                            payload.get("id").cloned().unwrap_or(Value::Null),
                            -32004,
                            "SSE session is no longer accepting responses",
                        )),
                    )
                        .into_response()
                }
            }
        } else {
            None
        };
        let response = dispatch_jsonrpc_payload(Arc::clone(&engine), payload).await;
        if session_sender.is_some() {
            // JSON-RPC notifications have no response body. A valid session
            // still receives the normal 202 acknowledgement.
            if let (Some(permit), Some(response)) = (permit, response) {
                permit.send(response);
            }
            return StatusCode::ACCEPTED.into_response();
        }
        let Some(response) = response else {
            return StatusCode::NO_CONTENT.into_response();
        };
        if !has_id && response.get("error").is_none() {
            return StatusCode::NO_CONTENT.into_response();
        }
        let status = jsonrpc_response_status(&response);
        return (status, Json(response)).into_response();
    }

    // Preserve the original direct Zero HTTP body for non-MCP callers. If a
    // session is supplied, enqueue that same body as an SSE message instead
    // of returning it synchronously.
    let permit = if let Some(sender) = &session_sender {
        match tokio::time::timeout(
            Duration::from_millis(100),
            sender.clone().reserve_owned(),
        )
        .await
        {
            Ok(Ok(permit)) => Some(permit),
            _ => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(json!({
                        "isError": true,
                        "error": {"code": "SessionUnavailable", "message": "SSE session is no longer accepting responses"}
                    })),
                )
                    .into_response()
            }
        }
    } else {
        None
    };
    let (status, body) = match engine.execute(&payload).await {
        Ok(outcome) => {
            let status = outcome_status(&outcome);
            let error = (status != StatusCode::OK).then(|| outcome_error(&outcome));
            let mut body = json!({"result": outcome});
            if let Some(error) = error {
                body["error"] = error;
            }
            (status, body)
        }
        Err(error) => engine_error_parts(&error),
    };
    if let Some(permit) = permit {
        permit.send(body);
        return StatusCode::ACCEPTED.into_response();
    }
    (status, Json(body)).into_response()
}

/// `POST /v1/causal_fold`: flat body `{edges, genders, axioms?, strategy?}`,
/// wrapped as `{"action": "causal_fold", "causal_fold": <body>}` and run
/// through the same `causal_fold` verb as `zero` and MCP. Same body shape as
/// [`message_handler`]: `{"result": outcome}`, plus `error` when the status
/// is not 200.
async fn causal_fold_handler(
    Extension(engine): Extension<Arc<PolymorphicZeroEngine>>,
    RestJson(payload): RestJson,
) -> Response {
    let request = json!({"action": "causal_fold", "causal_fold": payload});
    let outcome = match engine.execute(&request).await {
        Ok(o) => o,
        Err(e) => return engine_error_response(&e),
    };
    let status = outcome_status(&outcome);
    let mut body = json!({"result": outcome});
    if status != StatusCode::OK {
        body["error"] = outcome_error(&outcome);
    }
    (status, Json(body)).into_response()
}

/// `POST /v1/pipeline/{op}`: flat body, `op` taken from the path, run through the
/// same `pipeline` verb as `zero` and MCP. A body that also names an `op` must agree
/// with the path. Same body shape as [`message_handler`].
async fn pipeline_handler(
    Extension(engine): Extension<Arc<PolymorphicZeroEngine>>,
    RestPath(op): RestPath<String>,
    RestJson(mut payload): RestJson,
) -> Response {
    let Some(body) = payload.as_object_mut() else {
        return engine_error_response(&ServiceError::JsonRpc(
            "pipeline body must be a JSON object".into(),
        ));
    };
    if let Some(named) = body.get("op").and_then(Value::as_str) {
        if named != op {
            return engine_error_response(&ServiceError::JsonRpc(format!(
                "body op {named:?} disagrees with path op {op:?}"
            )));
        }
    }
    body.insert("op".into(), json!(op));
    let request = json!({"action": "pipeline", "pipeline": payload});
    let outcome = match engine.execute(&request).await {
        Ok(o) => o,
        Err(e) => return engine_error_response(&e),
    };
    let status = outcome_status(&outcome);
    let mut body = json!({"result": outcome});
    if status != StatusCode::OK {
        body["error"] = outcome_error(&outcome);
    }
    (status, Json(body)).into_response()
}

/// Schema properties of the world-model verbs and the `decide` modes. Added
/// after the fact: one `json!` holding every property overflows the macro
/// recursion limit.
fn add_world_model_properties(zero_tool: &mut Value) {
    let extra = json!({
                                        "state": {
                                            "type": ["string", "array", "object"],
                                            "description": "Context or state representation for 'ask' or 'imagine' (text or object). For 'simulate', 'what_if' and 'audit': a latent state, an array of exactly 1024 finite numbers (nothing is padded or truncated)."
                                        },
                                        "dynamics": {
                                            "type": "string",
                                            "enum": ["residual", "symplectic", "contact"],
                                            "description": "World model for 'simulate', 'what_if', 'audit' and the latent 'ask' modes. residual (default): untrained residual latent prior. symplectic: untrained Hamiltonian well stepped by Stormer-Verlet; each step reports its energy ledger, and 'simulate' also returns the full (q, p) phase trajectory. contact: the same well on the contact manifold with conformal damping 'damping' (gamma >= 0); gamma = 0 is exactly the symplectic flow, gamma > 0 contracts each (q_i, p_i) pair by exp(-2 gamma dt) per step, and the ledger reports the decay and the volume factor. Any other value is refused."
                                        },
                                        "damping": {
                                            "type": "number",
                                            "minimum": 0,
                                            "description": "Conformal damping rate gamma for dynamics 'contact' (default 0.5). Read only with dynamics 'contact'; with any other dynamics the field is refused. Negative, non-finite and non-numeric values are refused, never clamped."
                                        },
                                        "mode": {
                                            "type": "string",
                                            "enum": ["auto", "reflex", "mcts", "mpc_cem", "astar"],
                                            "description": "Engine for 'ask' (decide). auto and reflex run the semantic ask. mcts runs the semantic PUCT lookahead, or the planner crate's MCTS when 'latent' is given. mpc_cem and astar need 'latent'. Latent modes run on an untrained latent prior."
                                        },
                                        "nanocore_domain": {
                                            "type": "integer", "minimum": 0, "maximum": 4294967295_u64,
                                            "description": "One operator-loaded domain for text ask. Mutually exclusive with nanocore_domains."
                                        },
                                        "nanocore_domains": {
                                            "type": "array", "minItems": 1, "uniqueItems": true,
                                            "items": {"type": "integer", "minimum": 0, "maximum": 4294967295_u64},
                                            "description": "Operator-loaded domains to fuse for text ask; every requested core must be available. Requires nanocore_state."
                                        },
                                        "nanocore_state": {
                                            "type": "array", "minItems": 128, "maxItems": 128,
                                            "items": {"type": "number"},
                                            "description": "Compressed state for the selected operator Nanocore domains."
                                        },
                                        "engine": {
                                            "type": "string", "enum": ["generic", "nanocore"],
                                            "description": "Decision backend for ask/decide. nanocore requires nanocore_core and decision_state; missing assets are refused."
                                        },
                                        "head": {
                                            "type": "string", "enum": ["linear", "etf"],
                                            "description": "Decision head for ask/decide. ETF scores cosine(state, candidate_rep)/temperature; it requires candidate_reps plus etf_rep (generic) or decision_state (nanocore)."
                                        },
                                        "nanocore_core": {
                                            "type": "object", "description": "Serialized NanoCoreInstance parameters supplied by caller; no trained core is bundled."
                                        },
                                        "decision_state": {
                                            "type": "array", "items": {"type": "number"}, "minItems": 128, "maxItems": 128,
                                            "description": "Explicit 128-dimensional numeric state for nanocore; text is not encoded."
                                        },
                                        "etf_rep": {
                                            "type": "array", "items": {"type": "number"},
                                            "description": "Explicit numeric decision state for generic ETF."
                                        },
                                        "candidate_reps": {
                                            "type": "object",
                                            "additionalProperties": {"type": "array", "items": {"type": "number"}},
                                            "description": "head=etf only: candidate name -> manifold representation, same width as the state. Every feasible candidate is required; unknown names, zero-norm or non-finite vectors are refused."
                                        },
                                        "etf_temperature": {
                                            "type": "number", "exclusiveMinimum": 0,
                                            "description": "head=etf only: softmax temperature over cosine scores. Omitted means an uncalibrated default of 0.25, reported as temperature_source=default_uncalibrated."
                                        },
                                        "etf_metric": {
                                            "type": "object",
                                            "required": ["kind"],
                                            "properties": {
                                                "kind": {"type": "string", "enum": ["isotropic", "diagonal_mahalanobis", "whitened"]},
                                                "precision": {"type": "array", "items": {"type": "number", "exclusiveMinimum": 0}},
                                                "matrix": {"type": "array", "items": {"type": "number"}, "maxItems": 1048576},
                                                "out_dim": {"type": "integer", "minimum": 1}
                                            },
                                            "additionalProperties": false,
                                            "description": "head=etf only: cosine geometry. Omitted means isotropic. diagonal_mahalanobis needs `precision` (one finite positive weight per state dimension). whitened needs `out_dim` and a row-major `matrix` of out_dim x state-width floats, no all-zero rows, at most 1048576 entries. Parameters are caller-supplied and uncalibrated; the result reports etf.metric and its dimensions."
                                        },
                                        "latent": {
                                            "type": "array",
                                            "items": { "type": "number" },
                                            "description": "Latent state (1024 numbers) for 'ask' in mode mcts, mpc_cem or astar."
                                        },
                                        "return_trajectory": {
                                            "type": "boolean",
                                            "description": "For 'ask' in a latent mode: also roll the chosen action forward on the latent prior. On a text state the response says the trajectory is unsupported."
                                        },
                                        "forbidden_actions": {
                                            "type": "array",
                                            "items": { "type": "string", "minLength": 1 },
                                            "uniqueItems": true,
                                            "description": "Safety mask for ask/decide and imagine. Every listed action is hard-blocked. Unknown keys, non-string entries, empty names and duplicate names are refused."
                                        },
                                        "constraints": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "type": {
                                                        "type": "string",
                                                        "enum": ["forbid", "mutually_exclusive"],
                                                        "description": "`mutual_exclusive` is accepted as a legacy alias on input."
                                                    },
                                                    "actions": {
                                                        "type": "array",
                                                        "items": { "type": "string", "minLength": 1 },
                                                        "minItems": 1,
                                                        "uniqueItems": true
                                                    }
                                                },
                                                "required": ["type", "actions"],
                                                "additionalProperties": false,
                                                "description": "Supported constraints are `forbid` (one or more actions) and `mutually_exclusive` (at least two actions). Probability bounds and other constraint types are unsupported and refused."
                                            },
                                            "description": "Hard safety constraints accepted only by ask/decide and imagine, including latent decision mode. The first feasible action in caller order wins a mutually exclusive group; masks apply across the complete lookahead."
                                        },
                                        "actions": {
                                            "type": "array",
                                            "items": { "type": "string" },
                                            "description": "Fixed action plan for 'simulate'."
                                        },
                                        "target_action": {
                                            "type": "string",
                                            "description": "Planned action to review for 'audit'."
                                        },
                                        "continuation_actions": {
                                            "type": "array",
                                            "items": { "type": "string" },
                                            "description": "Optional action set for the greedy continuation after the audited action."
                                        }
    });
    match (
        zero_tool
            .pointer_mut("/inputSchema/properties")
            .and_then(Value::as_object_mut),
        extra,
    ) {
        (Some(properties), Value::Object(extra)) => properties.extend(extra),
        _ => tracing::error!(
            "zero tool schema has no properties object; world-model verbs are not advertised"
        ),
    }
}

/// `POST /v1/simulate`: flat body `{state, actions, horizon?}`.
async fn simulate_handler(
    Extension(engine): Extension<Arc<PolymorphicZeroEngine>>,
    RestJson(payload): RestJson,
) -> Response {
    worldmodel_response(&engine, "simulate", payload).await
}

/// `POST /v1/what_if`: flat body `{state, candidates, horizon?}`.
async fn what_if_handler(
    Extension(engine): Extension<Arc<PolymorphicZeroEngine>>,
    RestJson(payload): RestJson,
) -> Response {
    worldmodel_response(&engine, "what_if", payload).await
}

/// `POST /v1/audit_action`: flat body `{state, action, horizon?,
/// continuation_actions?}`. `action` is the audited action; on the `zero` tool
/// that name is `target_action`, because `action` there selects the verb.
async fn audit_action_handler(
    Extension(engine): Extension<Arc<PolymorphicZeroEngine>>,
    RestJson(payload): RestJson,
) -> Response {
    worldmodel_response(&engine, "audit", payload).await
}

/// Wrap a flat world-model body as a `zero` request for `verb`, run it, and
/// answer in the [`message_handler`] shape. A body that already carries a verb
/// selector is refused, not overridden.
async fn worldmodel_response(
    engine: &PolymorphicZeroEngine,
    verb: &str,
    payload: Value,
) -> Response {
    let refuse = |message: &str| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "isError": true,
                "error": {"code": "InvalidParams", "message": message},
            })),
        )
            .into_response()
    };
    let Value::Object(mut body) = payload else {
        return refuse("body must be a JSON object");
    };
    if body.contains_key("verb") || (verb != "audit" && body.contains_key("action")) {
        return refuse("body must not carry `verb` or `action`: the route selects the verb");
    }
    if verb == "audit" {
        if let Some(audited) = body.remove("action") {
            if body.contains_key("target_action") {
                return refuse("give the audited action as `action` or `target_action`, not both");
            }
            body.insert("target_action".to_string(), audited);
        }
    }
    body.insert("verb".to_string(), json!(verb));
    let outcome = match engine.execute(&Value::Object(body)).await {
        Ok(o) => o,
        Err(e) => return engine_error_response(&e),
    };
    let status = outcome_status(&outcome);
    let mut body = json!({"result": outcome});
    if status != StatusCode::OK {
        body["error"] = outcome_error(&outcome);
    }
    (status, Json(body)).into_response()
}

/// Add the compact decision contract expected by HTTP clients while keeping
/// the richer engine metadata intact. The Rust engine uses `chosen_action`
/// plus candidate records; callers of the HTTP adapter also get the Python
/// style `best_action`/`probs` names and an explicit backend/status signal.
fn add_decision_adapter_fields(body: &mut Value, status: StatusCode) {
    let Some(object) = body.as_object_mut() else {
        return;
    };

    if object
        .get("best_action")
        .map(Value::is_null)
        .unwrap_or(true)
    {
        object.insert(
            "best_action".to_owned(),
            object.get("chosen_action").cloned().unwrap_or(Value::Null),
        );
    }

    if !object.get("probs").map(Value::is_object).unwrap_or(false) {
        let mut probs = serde_json::Map::new();
        if let Some(existing) = object.get("probabilities").and_then(Value::as_object) {
            probs.extend(existing.clone());
        } else if let Some(candidates) = object.get("candidates").and_then(Value::as_array) {
            for candidate in candidates {
                if let (Some(name), Some(probability)) = (
                    candidate.get("name").and_then(Value::as_str),
                    candidate.get("probability"),
                ) {
                    probs.insert(name.to_owned(), probability.clone());
                }
            }
        }
        object.insert("probs".to_owned(), Value::Object(probs));
    }

    if !object
        .get("confidence")
        .map(Value::is_number)
        .unwrap_or(false)
    {
        let confidence = object
            .get("best_action")
            .and_then(Value::as_str)
            .and_then(|best| object.get("probs")?.get(best))
            .and_then(Value::as_f64)
            .filter(|confidence| confidence.is_finite())
            .unwrap_or(0.0);
        object.insert("confidence".to_owned(), json!(confidence));
    }

    // This endpoint is the decision backend. Its reachable flag describes
    // this HTTP response, not the optional Python semantic scorer.
    object.insert("backend_reachable".to_owned(), json!(true));
    let status_name = if !status.is_success() || object.get("isError") == Some(&Value::Bool(true)) {
        "error"
    } else if object.get("degraded") == Some(&Value::Bool(true)) {
        "degraded"
    } else {
        "ok"
    };
    object.insert("status".to_owned(), json!(status_name));
}

async fn legacy_decisions_handler(
    Extension(engine): Extension<Arc<PolymorphicZeroEngine>>,
    RestJson(payload): RestJson,
) -> Response {
    let outcome = match engine.execute(&payload).await {
        Ok(o) => o,
        Err(e) => {
            let (status, mut body) = engine_error_parts(&e);
            add_decision_adapter_fields(&mut body, status);
            return (status, Json(body)).into_response();
        }
    };
    let status = outcome_status(&outcome);
    let error = (status != StatusCode::OK).then(|| outcome_error(&outcome));
    // Legacy REST shape is the meta object plus the outcome status.
    let mut body = outcome.meta;
    body["isError"] = json!(outcome.is_error);
    body["content"] = json!(outcome.content);
    if let Some(error) = error {
        body["error"] = error;
    }
    add_decision_adapter_fields(&mut body, status);
    (status, Json(body)).into_response()
}

/// `POST /v1/mounts`: publish a new generation of cognitive assets by CAS on
/// `base_version`. Only a server started with an API token accepts it: an
/// open server must not let anyone replace its geometry.
async fn publish_mount_handler(
    Extension(engine): Extension<Arc<PolymorphicZeroEngine>>,
    Extension(auth): Extension<AuthConfig>,
    RestJson(payload): RestJson,
) -> Response {
    if auth.expected_token.is_none() {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"isError": true, "error": {
                "code": "PublishDisabled",
                "message": "mount publishing requires the server to run with GENZERO_API_KEY",
            }})),
        )
            .into_response();
    }
    match engine.publish_assets(&payload) {
        Ok(body) => (StatusCode::OK, Json(body)).into_response(),
        Err(r) => {
            tracing::warn!(code = %r.code, "mount publish refused: {}", r.detail);
            let status =
                StatusCode::from_u16(r.http_status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            (
                status,
                Json(json!({"isError": true, "error": {
                    "code": r.code, "stage": r.stage, "message": r.detail,
                }})),
            )
                .into_response()
        }
    }
}

/// `POST /v1/feedback`: ingest a JSON array of [`FeedbackRecord`] into the
/// in-process [`FeedbackBuffer`], for later delivery to the tuning server by
/// [`spawn_feedback_syncer`]. Always available (even without a configured
/// tuning endpoint), so MCP/REST clients never need to know whether the
/// closed loop is wired up.
async fn feedback_handler(
    Extension(buffer): Extension<Arc<FeedbackBuffer>>,
    RestJson(payload): RestJson,
) -> Response {
    let records: Vec<FeedbackRecord> = match serde_json::from_value(payload) {
        Ok(records) => records,
        Err(error) => {
            return rest_error(
                StatusCode::BAD_REQUEST,
                format!("body must be a JSON array of FeedbackRecord: {error}"),
            )
        }
    };
    let ingested = records.len();
    for record in records {
        buffer.record(record);
    }
    (
        StatusCode::OK,
        Json(json!({"ingested": ingested, "buffered": buffer.len()})),
    )
        .into_response()
}

#[cfg(test)]
mod gate_status_tests {
    use super::*;
    use crate::zero::{PolymorphicZeroEngine, RiskCheck, ZeroVerb};
    use gen_zero_gate::PolicyTier;

    /// A Tier0 request with a Tier1/Tier2 plan is 428, never 200.
    #[test]
    fn confirm_tier_plan_maps_to_428_not_200() {
        for plan in [PolicyTier::Tier1Confirm, PolicyTier::Tier2Escalate] {
            let out = PolymorphicZeroEngine::gated_outcome(
                ZeroVerb::Imagine,
                plan,
                &RiskCheck::NotApplicable,
                json!({}),
                String::new(),
            );
            assert_eq!(
                outcome_status(&out),
                StatusCode::PRECONDITION_REQUIRED,
                "{plan:?}"
            );
            assert_eq!(outcome_error(&out)["code"], -32002);
        }
        let stop = PolymorphicZeroEngine::gated_outcome(
            ZeroVerb::Imagine,
            PolicyTier::Tier3HardStop,
            &RiskCheck::NotApplicable,
            json!({}),
            String::new(),
        );
        assert_eq!(outcome_status(&stop), StatusCode::FORBIDDEN);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_overload_returns_service_unavailable() {
        assert_eq!(
            engine_error_response(&ServiceError::Overloaded).status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[tokio::test]
    async fn probes_separate_liveness_dependencies_and_assets() {
        use crate::bridge::{BridgeConfig, SemanticBridgeClient, HEALTH_PATH, REQUIRED_ROUTES};
        use axum::body::Body;
        use tower::ServiceExt;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let dependency = Router::new().route(HEALTH_PATH, get(|| async {
            Json(json!({"service": "gen-zero-semantic", "endpoints": REQUIRED_ROUTES, "backbone_loaded": true}))
        }));
        let task = tokio::spawn(async move { axum::serve(listener, dependency).await.unwrap() });
        let bridge =
            SemanticBridgeClient::new(BridgeConfig::new(format!("http://{addr}"))).unwrap();
        let engine = Arc::new(
            PolymorphicZeroEngine::new()
                .with_semantic(Some(Arc::new(SemanticBackend::Remote(bridge)))),
        );
        let app = McpServer::build_router(engine.clone(), Some("secret".into()));
        let request = |path| Request::builder().uri(path).body(Body::empty()).unwrap();
        assert_eq!(
            app.clone()
                .oneshot(request("/health"))
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        assert!(
            match engine.semantic().unwrap() {
                SemanticBackend::Remote(client) => client.last_health(),
                SemanticBackend::Native(_) => unreachable!("remote backend under test"),
            }
            .is_none(),
            "liveness must not probe dependencies"
        );
        assert_eq!(
            app.clone()
                .oneshot(request("/ready"))
                .await
                .unwrap()
                .status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        let assets: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/cognitive_assets_linear2d.json"
        ))
        .unwrap();
        engine
            .publish_assets(
                &json!({"base_version": 1, "assets": assets, "reason": "readiness test"}),
            )
            .unwrap();
        for path in ["/ready", "/healthz"] {
            assert_eq!(
                app.clone().oneshot(request(path)).await.unwrap().status(),
                StatusCode::OK
            );
        }
        task.abort();
        let disabled = McpServer::build_router(
            Arc::new(PolymorphicZeroEngine::new().with_semantic(None)),
            None,
        );
        assert_eq!(
            disabled
                .clone()
                .oneshot(request("/health"))
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        assert_eq!(
            disabled.oneshot(request("/ready")).await.unwrap().status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[tokio::test]
    async fn metrics_capture_calls_failures_gates_and_latency() {
        use axum::body::Body;
        use tower::ServiceExt;
        let engine = Arc::new(PolymorphicZeroEngine::new().with_semantic(None));
        engine
            .execute(&json!({"action": "grep", "lines": ["needle"], "query": "needle"}))
            .await
            .unwrap();
        assert!(
            engine
                .execute(&json!({"action": "grep"}))
                .await
                .unwrap()
                .is_error
        );
        for tier in [
            gen_zero_gate::PolicyTier::Tier0Proceed,
            gen_zero_gate::PolicyTier::Tier1Confirm,
            gen_zero_gate::PolicyTier::Tier2Escalate,
            gen_zero_gate::PolicyTier::Tier3HardStop,
        ] {
            let out = PolymorphicZeroEngine::gated_outcome(
                crate::zero::ZeroVerb::Imagine,
                tier,
                &crate::zero::RiskCheck::NotApplicable,
                json!({}),
                String::new(),
            );
            engine.metrics.record_execution(&json!({}), &Ok(out), 0.02);
        }
        let app = McpServer::build_router(engine, Some("secret".into()));
        let denied = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/message")
                    .method("POST")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::UNAUTHORIZED);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .header(header::AUTHORIZATION, "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/plain; version=0.0.4; charset=utf-8"
        );
        let bytes = axum::body::to_bytes(response.into_body(), 65536)
            .await
            .unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(
            text.contains("genzero_requests_total{verb=\"grep\",status=\"200\"} 1"),
            "{text}"
        );
        assert!(
            text.contains("genzero_requests_total{verb=\"grep\",status=\"400\"} 1"),
            "{text}"
        );
        assert!(text.contains("genzero_http_requests_total{method=\"POST\",status=\"401\"} 1"));
        for tier in ["Tier1", "Tier2Escalate", "Tier3HardStop"] {
            assert!(
                text.contains(&format!("genzero_gate_tier_total{{tier=\"{tier}\"}} 1")),
                "{text}"
            );
        }
        assert!(text.contains("genzero_request_duration_seconds_count 6"));
        assert!(text.contains("genzero_request_duration_seconds_bucket{le=\"+Inf\"} 6"));
    }

    #[tokio::test]
    async fn metrics_record_real_engine_gate_verdicts() {
        // An ask needs a real candidate and an assessed request risk to proceed.
        let (engine, _risk) = crate::zero::test_support::engine_with_benign_risk().await;
        let proceed = engine
            .execute(&json!({"action": "ask", "candidates": ["noop"]}))
            .await
            .unwrap();
        assert_eq!(proceed.meta["tier"], "Proceed", "{proceed:?}");
        let escalate = engine
            .execute(&json!({"action": "ask", "candidates": ["review", "skip"]}))
            .await
            .unwrap();
        assert_eq!(escalate.meta["tier"], "Escalate");
        let text = engine.metrics.render();
        assert!(
            text.contains("genzero_gate_tier_total{tier=\"Proceed\"} 1"),
            "{text}"
        );
        assert!(
            text.contains("genzero_gate_tier_total{tier=\"Tier2Escalate\"} 1"),
            "{text}"
        );
        assert!(
            text.contains("genzero_requests_total{verb=\"ask\",status=\"200\"} 1"),
            "{text}"
        );
        assert!(
            text.contains("genzero_requests_total{verb=\"ask\",status=\"428\"} 1"),
            "{text}"
        );
        assert!(text.contains("genzero_request_duration_seconds_count 2"));
    }

    #[tokio::test]
    async fn graceful_shutdown_drains_in_flight_request() {
        let entered = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let app = Router::new().route(
            "/slow",
            get({
                let entered = entered.clone();
                let release = release.clone();
                move || {
                    let entered = entered.clone();
                    let release = release.clone();
                    async move {
                        entered.notify_one();
                        release.notified().await;
                        "completed"
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (stop, shutdown) = tokio::sync::oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            serve_with_shutdown(
                listener,
                app,
                SseSessionManager::default(),
                async {
                    let _ = shutdown.await;
                },
                Duration::from_secs(5),
            )
            .await
            .unwrap();
        });
        let request = tokio::spawn(async move {
            reqwest::get(format!("http://{addr}/slow"))
                .await
                .unwrap()
                .text()
                .await
                .unwrap()
        });
        tokio::time::timeout(Duration::from_secs(5), entered.notified())
            .await
            .unwrap();
        stop.send(()).unwrap();
        tokio::task::yield_now().await;
        assert!(!server.is_finished());
        release.notify_one();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(5), request)
                .await
                .unwrap()
                .unwrap(),
            "completed"
        );
        tokio::time::timeout(Duration::from_secs(5), server)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn test_mcp_server_jsonrpc_frame_handling() {
        let (engine, _risk) = crate::zero::test_support::engine_with_benign_risk().await;
        let server = McpServer {
            engine: Arc::new(engine),
            ..McpServer::new()
        };

        let mut init_buf =
            b"{\"jsonrpc\": \"2.0\", \"method\": \"initialize\", \"id\": 1}".to_vec();
        init_buf.resize(init_buf.len() + simd_json::SIMDJSON_PADDING, 0);

        let resp = server.handle_jsonrpc_frame(&mut init_buf).await;
        let v: Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(v["result"]["serverInfo"]["name"], "gen-zero");

        // Tool call test
        let mut tool_buf = b"{\"jsonrpc\": \"2.0\", \"method\": \"tools/call\", \"params\": {\"name\": \"zero\", \"arguments\": {\"action\": \"ask\", \"candidates\": [\"noop\"]}}, \"id\": 2}".to_vec();
        tool_buf.resize(tool_buf.len() + simd_json::SIMDJSON_PADDING, 0);

        let tool_resp = server.handle_jsonrpc_frame(&mut tool_buf).await;
        let tv: Value = serde_json::from_str(&tool_resp).unwrap();
        assert_eq!(tv["result"]["isError"], false, "{tv}");
    }

    #[tokio::test]
    async fn readiness_reports_a_wrong_service_on_the_bridge_port_as_degraded() {
        use crate::bridge::{BridgeConfig, SemanticBridgeClient};
        use axum::body::Body;
        use axum::http::Request;
        use tower::util::ServiceExt;

        // Stand-in for the older service that held the bridge port: it has
        // /health but none of the semantic routes.
        let foreign = Router::new().route(
            "/health",
            get(|| async { Json(json!({"status": "healthy"})) }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, foreign).await.unwrap() });

        let bridge =
            SemanticBridgeClient::new(BridgeConfig::new(format!("http://{addr}"))).unwrap();
        let engine = Arc::new(
            PolymorphicZeroEngine::new()
                .with_semantic(Some(Arc::new(SemanticBackend::Remote(bridge)))),
        );
        let server = McpServer {
            engine: engine.clone(),
            auth_token: None,
            bridge_required: false,
            closed_loop: None,
        };
        server
            .check_semantic(Some(DEFAULT_MCP_SSE_PORT))
            .await
            .unwrap();

        let app = McpServer::build_router(engine, None);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/ready")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), 1 << 16)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["status"], "degraded", "{body}");
        assert_eq!(body["semantic_bridge"]["status"], "wrong_service", "{body}");
        assert!(body["semantic_bridge"]["detail"]
            .as_str()
            .unwrap()
            .contains("not the gen-zero semantic scorer"));
    }

    #[tokio::test]
    async fn bridge_required_refuses_to_start_without_a_ready_scorer() {
        use crate::bridge::{BridgeConfig, SemanticBridgeClient};
        let mut cfg = BridgeConfig::new("http://127.0.0.1:1");
        cfg.max_retries = 0;
        let bridge = SemanticBridgeClient::new(cfg).unwrap();
        let engine = Arc::new(
            PolymorphicZeroEngine::new()
                .with_semantic(Some(Arc::new(SemanticBackend::Remote(bridge)))),
        );
        let lenient = McpServer {
            engine,
            auth_token: None,
            bridge_required: false,
            closed_loop: None,
        };
        assert!(
            lenient.check_semantic(None).await.is_ok(),
            "default: warn, keep serving"
        );
        let strict = lenient.with_bridge_required(true);
        assert!(strict.check_semantic(None).await.is_err());
        let disabled = McpServer {
            engine: Arc::new(PolymorphicZeroEngine::new().with_semantic(None)),
            auth_token: None,
            bridge_required: true,
            closed_loop: None,
        };
        assert!(disabled.check_semantic(None).await.is_err());
    }

    #[test]
    fn test_constant_time_token_match() {
        assert!(constant_time_token_match(
            "gz_live_abc123",
            "gz_live_abc123"
        ));
        assert!(!constant_time_token_match(
            "gz_live_abc123",
            "gz_live_abc124"
        ));
        assert!(!constant_time_token_match("gz_live_abc", "gz_live_abc123"));
        assert!(!constant_time_token_match("", "gz_live_abc123"));
    }

    #[tokio::test]
    async fn test_auth_middleware_flow() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::util::ServiceExt;

        let engine = Arc::new(PolymorphicZeroEngine::new());
        let secret = "gz_live_supersecret456";
        let app = McpServer::build_router(engine, Some(secret.to_string()));

        // 1. Health endpoint should always pass without auth
        let health_req = Request::builder()
            .uri("/health")
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(health_req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // 2. Unauthenticated request to /sse should fail with 401
        let unauth_req = Request::builder()
            .uri("/sse")
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(unauth_req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        assert!(resp.headers().contains_key(header::WWW_AUTHENTICATE));

        // 3. Authorization Bearer header should succeed
        let bearer_req = Request::builder()
            .uri("/sse")
            .method("GET")
            .header(header::AUTHORIZATION, format!("Bearer {}", secret))
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(bearer_req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // 4. X-API-Key header should succeed
        let x_api_req = Request::builder()
            .uri("/sse")
            .method("GET")
            .header("x-api-key", secret)
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(x_api_req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // 5. Query param ?token= should succeed
        let query_token_req = Request::builder()
            .uri(format!("/sse?token={}", secret))
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(query_token_req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // 6. Query param ?api_key= should succeed
        let query_api_key_req = Request::builder()
            .uri(format!("/sse?api_key={}", secret))
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(query_api_key_req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // 7. Wrong token should fail with 401
        let wrong_token_req = Request::builder()
            .uri("/sse?token=wrong_token")
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(wrong_token_req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        // 8. Connecting with valid token propagates token into endpoint URI
        let sse_req = Request::builder()
            .uri(format!("/sse?token={}", secret))
            .method("GET")
            .body(Body::empty())
            .unwrap();
        let sse_resp = app.clone().oneshot(sse_req).await.unwrap();
        assert_eq!(sse_resp.status(), StatusCode::OK);
        let mut sse_body = sse_resp.into_body().into_data_stream();
        let chunk = futures_util::StreamExt::next(&mut sse_body)
            .await
            .unwrap()
            .unwrap();
        let event_text = std::str::from_utf8(&chunk).unwrap();
        let endpoint = event_text
            .lines()
            .find_map(|l| l.strip_prefix("data: "))
            .expect("SSE must emit endpoint event");
        assert!(
            endpoint.contains(&format!("&token={}", secret)),
            "Endpoint must include token for standard MCP clients: {endpoint}"
        );

        // 9. POST to /message using active session without token header/param passes auth
        let session_id = endpoint
            .split("session_id=")
            .nth(1)
            .and_then(|s| s.split('&').next())
            .expect("session_id must be present");
        let post_req = Request::builder()
            .uri(format!("/message?session_id={}", session_id))
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).to_string(),
            ))
            .unwrap();
        let post_resp = app.clone().oneshot(post_req).await.unwrap();
        assert_eq!(post_resp.status(), StatusCode::ACCEPTED);

        // 10. POST to /message with non-existent session and no token fails auth
        let fake_post_req = Request::builder()
            .uri("/message?session_id=deadbeefcafebabe")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).to_string(),
            ))
            .unwrap();
        let fake_resp = app.clone().oneshot(fake_post_req).await.unwrap();
        assert_eq!(fake_resp.status(), StatusCode::UNAUTHORIZED);
    }
}

#[cfg(test)]
mod transport_regression_tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use futures_util::StreamExt;
    use tower::ServiceExt;

    #[tokio::test]
    async fn stdio_bounds_unterminated_frames_and_accepts_exact_limit() {
        let mut frame = String::new();
        let input = vec![b'x'; MAX_STDIO_FRAME_BYTES + 100];
        let mut reader = &input[..];
        let error = read_stdio_frame(&mut reader, &mut frame).await.unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(frame.len(), MAX_STDIO_FRAME_BYTES + 1);
        let mut reader = &input[..MAX_STDIO_FRAME_BYTES];
        assert_eq!(
            read_stdio_frame(&mut reader, &mut frame).await.unwrap(),
            MAX_STDIO_FRAME_BYTES
        );
        let mut reader = &b"one\ntwo\n"[..];
        read_stdio_frame(&mut reader, &mut frame).await.unwrap();
        assert_eq!(frame, "one\n");
        read_stdio_frame(&mut reader, &mut frame).await.unwrap();
        assert_eq!(frame, "two\n");
    }

    #[tokio::test]
    async fn shutdown_closes_sse_even_with_sender_clones_and_late_clients() {
        let sessions = SseSessionManager::default();
        let response = sse_handler(Extension(sessions.clone()), None)
            .await
            .into_response();
        let mut stream = response.into_body().into_data_stream();
        assert!(stream.next().await.unwrap().is_ok());
        let sender = sessions
            .sessions
            .lock()
            .unwrap()
            .values()
            .next()
            .unwrap()
            .clone();
        sessions.shutdown();
        assert!(tokio::time::timeout(Duration::from_secs(1), stream.next())
            .await
            .unwrap()
            .is_none());
        assert!(sender.is_closed());
        let late = sse_handler(Extension(sessions.clone()), None)
            .await
            .into_response();
        assert!(to_bytes(late.into_body(), 1024).await.unwrap().is_empty());
        assert!(sessions.sessions.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn shutdown_broadcast_reaches_every_sse_receiver() {
        let sessions = SseSessionManager::default();
        let mut streams = Vec::new();
        for _ in 0..3 {
            let mut stream = sse_handler(Extension(sessions.clone()), None)
                .await
                .into_response()
                .into_body()
                .into_data_stream();
            assert!(stream.next().await.unwrap().is_ok());
            streams.push(stream);
        }
        sessions.shutdown();
        for mut stream in streams {
            assert!(tokio::time::timeout(Duration::from_secs(1), stream.next())
                .await
                .unwrap()
                .is_none());
        }
        assert!(sessions.sessions.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn http_shutdown_finishes_with_an_active_sse_client() {
        let sessions = SseSessionManager::default();
        let app = Router::new()
            .route("/sse", get(sse_handler))
            .layer(Extension(sessions.clone()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (stop, stopped) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(serve_with_shutdown(
            listener,
            app,
            sessions.clone(),
            async {
                let _ = stopped.await;
            },
            Duration::from_secs(5),
        ));
        let mut response = reqwest::get(format!("http://{addr}/sse")).await.unwrap();
        assert!(response.chunk().await.unwrap().is_some());
        stop.send(()).unwrap();
        assert!(
            tokio::time::timeout(Duration::from_secs(1), response.chunk())
                .await
                .unwrap()
                .unwrap()
                .is_none()
        );
        tokio::time::timeout(Duration::from_secs(1), server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(sessions.sessions.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn shutdown_deadline_bounds_a_stuck_http_request() {
        let entered = Arc::new(tokio::sync::Notify::new());
        let notify = entered.clone();
        let dropped = Arc::new(tokio::sync::Notify::new());
        struct DropSignal(Arc<tokio::sync::Notify>);
        impl Drop for DropSignal {
            fn drop(&mut self) {
                self.0.notify_one();
            }
        }
        let signal = dropped.clone();
        let app = Router::new().route(
            "/",
            get(move || {
                let notify = notify.clone();
                let signal = signal.clone();
                async move {
                    let _guard = DropSignal(signal);
                    notify.notify_one();
                    std::future::pending::<&'static str>().await
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (stop, stopped) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(serve_with_shutdown(
            listener,
            app,
            SseSessionManager::default(),
            async {
                let _ = stopped.await;
            },
            Duration::from_millis(30),
        ));
        let client = tokio::spawn(async move { reqwest::get(format!("http://{addr}/")).await });
        tokio::time::timeout(Duration::from_secs(2), entered.notified())
            .await
            .unwrap();
        stop.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(1), server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        tokio::time::timeout(Duration::from_secs(1), dropped.notified())
            .await
            .expect("deadline must drop the handler, not just the server future");
        let response = tokio::time::timeout(Duration::from_secs(1), client)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn malformed_message_query_returns_jsonrpc_error() {
        let app = McpServer::build_router(
            Arc::new(PolymorphicZeroEngine::new().with_semantic(None)),
            None,
        );
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/message?session_id=a&session_id=b")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"jsonrpc":"2.0","method":"ping","id":1}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 8192).await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], -32600);
    }

    #[tokio::test]
    async fn rest_routing_and_missing_proof_errors_are_json() {
        let app = McpServer::build_router(
            Arc::new(PolymorphicZeroEngine::new().with_semantic(None)),
            None,
        );
        for (path, status) in [
            ("/v1/missing", StatusCode::NOT_FOUND),
            ("/v1/plan", StatusCode::METHOD_NOT_ALLOWED),
            ("/audit/ledger/0", StatusCode::NOT_FOUND),
        ] {
            let response = app
                .clone()
                .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), status);
            assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
            let body: Value =
                serde_json::from_slice(&to_bytes(response.into_body(), 8192).await.unwrap())
                    .unwrap();
            assert_eq!(body["code"], status.as_u16());
            assert!(body["error"].is_string());
        }
    }

    #[tokio::test]
    async fn rest_extractors_return_json_and_plan_matches_decisions() {
        let (engine, _risk) = crate::zero::test_support::engine_with_benign_risk().await;
        let app = McpServer::build_router(Arc::new(engine), None);
        for path in [
            "/v1/simulate",
            "/v1/what_if",
            "/v1/audit_action",
            "/v1/decisions",
            "/v1/plan",
            "/v1/causal_fold",
            "/v1/pipeline/test",
            "/v1/mounts",
        ] {
            for content_type in ["application/json", "text/plain"] {
                let response = app
                    .clone()
                    .oneshot(
                        Request::builder()
                            .method("POST")
                            .uri(path)
                            .header(header::CONTENT_TYPE, content_type)
                            .body(Body::from("{"))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{path}");
                assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
                let body: Value =
                    serde_json::from_slice(&to_bytes(response.into_body(), 8192).await.unwrap())
                        .unwrap();
                assert_eq!(body["code"], 400);
                assert!(body["error"].is_string());
            }
        }
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/audit/ledger/invalid")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
        let mut results = Vec::new();
        for path in ["/v1/plan", "/v1/decisions"] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(path)
                        .header(header::CONTENT_TYPE, "application/json")
                        .body(Body::from(r#"{"action":"ask","candidates":["noop"]}"#))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let mut body: Value =
                serde_json::from_slice(&to_bytes(response.into_body(), 8192).await.unwrap())
                    .unwrap();
            assert_eq!(body["decision_audit"]["leaf_index"], results.len());
            body.as_object_mut().unwrap().remove("decision_audit");
            results.push(body);
        }
        assert_eq!(results[0], results[1]);
        assert_eq!(results[0]["status"], "ok");
    }
}
