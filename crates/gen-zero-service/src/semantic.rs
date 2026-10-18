//! The semantic backend behind `ask`, `route`, `imagine` and the request risk check.
//!
//! Exactly one backend is chosen at startup and never switched at run time:
//!
//! * [`SemanticBackend::Native`]: Qwen2.5-0.5B in this process on candle
//!   ([`gen_zero_model::QwenSemanticScorer`]), selected by
//!   `GENZERO_QWEN_MODEL_PATH` (or `--qwen-model`). A load failure is a
//!   startup error; the service never falls back to the Python bridge.
//! * [`SemanticBackend::Remote`]: the Python scorer over HTTP
//!   ([`SemanticBridgeClient`], `GENZERO_PYTHON_ENDPOINT`), for hosts that
//!   run the Python service and have no local weights.
//!
//! Both return the same response types and pass the same validation
//! ([`crate::bridge::validate_ask`] and friends), so the gate sees one contract.

use crate::bridge::{
    validate_ask, validate_risk, validate_route, AskInput, BridgeError, CandidateScore,
    RiskThresholds, SemanticAskResponse, SemanticBridgeClient, SemanticRiskResponse,
    SemanticRouteResponse,
};
use crate::error::ServiceError;
use gen_zero_model::semantic_qwen::{
    causal_utility_prior, normalized_entropy, select_ask_frame, state_text, tool_continuation,
    RiskScoreReport, CAUSAL_PRIOR_VERSION, RISK_THRESHOLDS_VERSION, ROUTE_FRAME,
};
use gen_zero_model::{
    QwenModelInfo, QwenSemanticScorer, RISK_ESCALATE_THRESHOLD, RISK_HARD_STOP_THRESHOLD,
};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;

/// `_meta.engine` of a semantic outcome scored in process.
pub const ENGINE_NATIVE_QWEN: &str = "native_qwen";
/// `_meta.engine` of a semantic outcome scored by the Python service.
pub const ENGINE_SEMANTIC_BRIDGE: &str = "semantic_bridge";

/// Turns on the contextual policy prior ([`causal_utility_prior`]) for
/// `ask` and everything built on it (`imagine`): `contextual_v2` (or `on`).
/// Unset, empty or `off`: the response is the model's scores, untouched.
pub const CAUSAL_PRIOR_ENV: &str = "GENZERO_CAUSAL_PRIOR";

fn parse_causal_prior_env(raw: &str) -> Result<bool, BridgeError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "off" | "0" | "false" => Ok(false),
        "on" | "1" | "true" => Ok(true),
        v if v == CAUSAL_PRIOR_VERSION => Ok(true),
        _ => Err(BridgeError::Native(format!(
            "{CAUSAL_PRIOR_ENV}={raw:?}: expected `off` or `{CAUSAL_PRIOR_VERSION}`"
        ))),
    }
}

/// Whether [`CAUSAL_PRIOR_ENV`] enables the prior. An unknown value is an
/// error, not a silent default.
pub(crate) fn causal_prior_enabled() -> Result<bool, BridgeError> {
    match std::env::var(CAUSAL_PRIOR_ENV) {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(e) => Err(BridgeError::Native(format!("{CAUSAL_PRIOR_ENV}: {e}"))),
        Ok(raw) => parse_causal_prior_env(&raw),
    }
}

/// Record the raw model distribution and, only when `enabled`, apply the
/// policy prior on top of it. Either way `scorer.policy_prior` says what ran.
fn apply_policy_prior(
    context: &str,
    response: &mut SemanticAskResponse,
    enabled: bool,
) -> Result<(), BridgeError> {
    for candidate in &mut response.candidates {
        candidate
            .raw_probability
            .get_or_insert(candidate.probability);
        candidate.policy_prior_adjustment = 0.0;
    }
    if !response.scorer.is_object() {
        response.scorer = json!({ "backend_scorer": response.scorer.take() });
    }
    response.scorer["policy_prior"] = json!({
        "enabled": enabled,
        "source": null,
        "env": CAUSAL_PRIOR_ENV,
    });
    if enabled {
        apply_causal_prior(context, response)?;
    }
    Ok(())
}

/// Re-rank by `log raw_probability + causal_utility_prior`. `pmi` and
/// `raw_probability` keep the model's values; the adjustment goes to
/// `policy_prior_adjustment` and only `probability` changes.
fn apply_causal_prior(
    context: &str,
    response: &mut SemanticAskResponse,
) -> Result<(), BridgeError> {
    let adjustments: Vec<f64> = response
        .candidates
        .iter()
        .map(|candidate| causal_utility_prior(context, &candidate.name))
        .collect();
    let raw_chosen = response.chosen.clone();
    response.scorer["policy_prior"] = json!({
        "enabled": true,
        "source": "causal_utility_prior",
        "version": CAUSAL_PRIOR_VERSION,
        "env": CAUSAL_PRIOR_ENV,
        "adjusted_candidates": adjustments.iter().filter(|v| **v != 0.0).count(),
        "raw_chosen": raw_chosen,
    });
    if adjustments.iter().all(|value| *value == 0.0) {
        return Ok(());
    }
    let logits: Vec<f64> = response
        .candidates
        .iter()
        .zip(&adjustments)
        .map(|(candidate, adjustment)| {
            let raw = candidate.raw_probability.unwrap_or(candidate.probability);
            raw.max(f64::MIN_POSITIVE).ln() + adjustment
        })
        .collect();
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(BridgeError::InvalidResponse(
            "non-finite adjusted candidate score".into(),
        ));
    }
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logits.iter().map(|value| (value - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(BridgeError::InvalidResponse(
            "invalid adjusted candidate weights".into(),
        ));
    }
    for (index, candidate) in response.candidates.iter_mut().enumerate() {
        candidate.policy_prior_adjustment = adjustments[index];
        candidate.probability = weights[index] / total;
    }
    // First argmax, as `validate_ask` requires: `max_by` keeps the last of a tie.
    if response.candidates.is_empty() {
        return Err(BridgeError::InvalidResponse("no candidates".into()));
    }
    let mut best = 0;
    for (index, candidate) in response.candidates.iter().enumerate() {
        if candidate.probability > response.candidates[best].probability {
            best = index;
        }
    }
    response.chosen_index = best;
    response.chosen = response.candidates[response.chosen_index].name.clone();
    response.entropy = normalized_entropy(
        &response
            .candidates
            .iter()
            .map(|c| c.probability)
            .collect::<Vec<_>>(),
    );
    Ok(())
}

/// `classifier` metadata of a native risk response: the score's provenance
/// ([`RiskScoreReport`]), never a claim the thresholds were fit on this backbone.
fn risk_classifier_meta(
    id: &str,
    per_order_log_odds: &[f64],
    report: Option<&RiskScoreReport>,
) -> Value {
    let calibrated = report.is_some_and(|r| r.calibration.is_some());
    json!({
        "id": id,
        "method": "in_context_pmi_log_odds",
        "per_order_log_odds": per_order_log_odds,
        "backbone_id": report.map(|r| r.backbone_id.as_str()),
        "calibration_method": report.map(|r| r.method.as_str()),
        "calibration_split": report.and_then(|r| r.calibration_split.as_deref()),
        "heldout_split": report.map(|r| r.heldout_split.as_str()),
        "heldout": report.and_then(|r| r.heldout),
        "calibration": report.and_then(|r| r.calibration.as_ref()),
        "thresholds_version": RISK_THRESHOLDS_VERSION,
        "thresholds_calibrated_on": if calibrated {
            "python fp32 Qwen2.5-0.5B (risk_data/report.json); this backbone's log-odds are \
             mapped onto that scale by the LDA fit on calibration.jsonl"
        } else {
            "python fp32 Qwen2.5-0.5B (risk_data/report.json); this backbone is uncalibrated"
        },
    })
}

#[cfg(test)]
mod causal_prior_tests {
    use super::*;

    fn score(context: &str, names: &[&str]) -> SemanticAskResponse {
        let mut response = raw_response(names);
        apply_policy_prior(context, &mut response, true).unwrap();
        response
    }

    /// Model scores that favor the first name (pmi 3, probability 0.95).
    fn raw_response(names: &[&str]) -> SemanticAskResponse {
        SemanticAskResponse {
            chosen: names[0].into(),
            chosen_index: 0,
            candidates: names
                .iter()
                .enumerate()
                .map(|(index, name)| CandidateScore {
                    name: (*name).into(),
                    log_likelihood: 0.0,
                    baseline_log_likelihood: 0.0,
                    pmi: if index == 0 { 3.0 } else { 0.0 },
                    probability: if index == 0 { 0.95 } else { 0.05 },
                    raw_probability: None,
                    policy_prior_adjustment: 0.0,
                })
                .collect(),
            entropy: 0.3,
            scorer: json!({}),
            embedding_dim: 0,
            embedding: None,
            timing_ms: 0.0,
        }
    }

    #[test]
    fn spoiled_milk_favors_disposal() {
        let context = "The milk in the fridge is one week past expiration date and smells sour";
        let response = score(context, &["drink it", "throw it away"]);
        assert_eq!(response.chosen, "throw it away");
        assert!(response.candidates[0].probability < 0.01);
        assert!(causal_utility_prior(context, "drink it") < 0.0);
    }

    #[test]
    fn exposed_high_voltage_wire_favors_safe_remediation() {
        let context = "An uninsulated high-voltage wire is hanging from the wall";
        for safe in ["call an electrician", "shut off main breaker"] {
            let response = score(context, &["touch it with bare hands", safe]);
            assert_eq!(response.chosen, safe);
            assert!(response.candidates[0].probability < 0.01);
        }
    }

    #[test]
    fn destructive_command_is_penalized() {
        let response = score(
            "Clean up temporary files",
            &["rm -rf /", "review the files"],
        );
        assert_eq!(response.chosen, "review the files");
        assert!(response.candidates[0].probability < 0.01);
    }

    #[test]
    fn hazardous_material_contact_is_penalized() {
        let response = score(
            "A radioactive object is on the floor",
            &["touch it", "keep away"],
        );
        assert_eq!(response.chosen, "keep away");
        assert!(response.candidates[0].probability < 0.01);
    }

    /// WebGPT round 1 (M01) counterexamples: negation, word boundaries,
    /// parsed paths and Chinese.
    #[test]
    fn negated_hazards_are_not_penalized() {
        let p = causal_utility_prior;
        let milk = "The milk is expired";
        assert!(p(milk, "do not drink") >= 0.0);
        assert!(p(milk, "do not drink") > p(milk, "drink it"));
        assert!(p(milk, "discard it") >= 0.0);
        let wire = "The wire is high voltage";
        assert!(p(wire, "do not touch it") >= 0.0);
        assert!(p(wire, "do not touch it") > p(wire, "touch it"));
        assert!(p(wire, "don't touch it, call an electrician") > 0.0);
        assert!(p(milk, "never drink it") > 0.0);
        assert_eq!(p(milk, "do not discard it. Drink it"), -16.0);
        assert_eq!(p(milk, "Do not hesitate to drink it"), -16.0);
        assert_eq!(p(milk, "Do not drink it: drink it"), -16.0);
    }

    #[test]
    fn substrings_and_safe_paths_get_no_penalty() {
        let p = causal_utility_prior;
        // `create` contains `eat`; a certificate is not food.
        assert_eq!(
            p(
                "The certificate has expired",
                "create a replacement certificate"
            ),
            0.0
        );
        assert_eq!(p("Clean up temporary files", "rm -rf /tmp/cache"), 0.0);
        assert_eq!(p("Clean up temporary files", "rm -rf /tmp/_/"), 0.0);
        assert_eq!(p("Clean up temporary files", "rm -rf ./build/*"), 0.0);
        for root in [
            "rm -rf /",
            "rm -rf /*",
            "sudo rm -r -f /",
            "rm --recursive --force /",
            "never hesitate. rm -rf /",
            "echo never && rm -rf --no-preserve-root /",
        ] {
            assert_eq!(p("Clean up temporary files", root), -16.0, "{root}");
        }
        assert_eq!(p("Clean up temporary files", "never run rm -rf /"), 0.0);
        assert_eq!(p("The cheese is not expired", "eat it"), 0.0);
    }

    #[test]
    fn chinese_terms_follow_the_same_rules() {
        let p = causal_utility_prior;
        let milk = "牛奶已经过期";
        assert!(p(milk, "喝掉它") < 0.0);
        assert!(p(milk, "特别想喝") < 0.0);
        assert!(p(milk, "别喝，倒掉") > 0.0);
        assert!(p(milk, "不要喝，扔掉它") >= 0.0);
        assert!(p(milk, "不要喝，扔掉它") > p(milk, "喝掉它"));
        assert_eq!(p(milk, "喝掉它"), p("The milk is expired", "drink it"));
        assert_eq!(
            p(milk, "不要喝，扔掉它"),
            p("The milk is expired", "do not drink it, throw it away")
        );
        assert!(p("电线裸露带电", "徒手去摸") < 0.0);
        assert!(p("电线裸露带电", "切断电源，叫电工") > 0.0);
    }

    #[test]
    fn prior_never_changes_the_raw_model_scores() {
        let response = score("The milk is expired", &["drink it", "throw it away"]);
        assert_eq!(response.chosen, "throw it away");
        assert_eq!(response.candidates[0].pmi, 3.0);
        assert_eq!(response.candidates[1].pmi, 0.0);
        assert_eq!(response.candidates[0].raw_probability, Some(0.95));
        assert_eq!(response.candidates[1].raw_probability, Some(0.05));
        assert_eq!(response.candidates[0].policy_prior_adjustment, -16.0);
        assert_eq!(response.candidates[1].policy_prior_adjustment, 8.0);
        assert_eq!(response.scorer["policy_prior"]["enabled"], json!(true));
        assert_eq!(
            response.scorer["policy_prior"]["version"],
            json!(CAUSAL_PRIOR_VERSION)
        );
        assert_eq!(
            response.scorer["policy_prior"]["raw_chosen"],
            json!("drink it")
        );
    }

    #[test]
    fn prior_is_off_unless_enabled() {
        for off in ["", "0", "off", "false", " FALSE "] {
            assert!(!parse_causal_prior_env(off).unwrap(), "{off}");
        }
        for on in ["1", "on", "true", "TRUE", CAUSAL_PRIOR_VERSION] {
            assert!(parse_causal_prior_env(on).unwrap(), "{on}");
        }
        assert!(parse_causal_prior_env("invalid").is_err());
        let mut raw = raw_response(&["drink it", "throw it away"]);
        let before = raw.candidates.clone();
        apply_policy_prior("The milk is expired", &mut raw, false).unwrap();
        assert_eq!(raw.chosen, "drink it");
        for (after, before) in raw.candidates.iter().zip(&before) {
            assert_eq!(after.pmi, before.pmi);
            assert_eq!(after.probability, before.probability);
            assert_eq!(after.raw_probability, Some(before.probability));
            assert_eq!(after.policy_prior_adjustment, 0.0);
        }
        assert_eq!(raw.scorer["policy_prior"]["enabled"], json!(false));
        assert_eq!(raw.scorer["policy_prior"]["source"], Value::Null);
    }

    #[test]
    fn risk_metadata_names_its_disjoint_splits() {
        let meta = risk_classifier_meta("id", &[0.1], None);
        assert_eq!(meta["thresholds_version"], json!(RISK_THRESHOLDS_VERSION));
        assert_eq!(meta["calibration_split"], Value::Null);
        assert!(meta["thresholds_calibrated_on"]
            .as_str()
            .unwrap()
            .contains("uncalibrated"));
    }

    /// Two hazards tie after the prior. The pick stays on the first argmax,
    /// as `validate_ask` requires; `max_by` would move it to the last.
    #[test]
    fn a_tie_after_the_prior_keeps_the_first_argmax() {
        let names = ["drink it", "eat it"];
        let mut response = raw_response(&names);
        for candidate in &mut response.candidates {
            candidate.probability = 0.5;
        }
        response.entropy = 1.0;
        apply_policy_prior("expired milk", &mut response, true).unwrap();
        assert_eq!(response.candidates[0].policy_prior_adjustment, -16.0);
        assert_eq!(response.candidates[1].policy_prior_adjustment, -16.0);
        assert_eq!(response.chosen_index, 0);
        assert_eq!(response.chosen, "drink it");
        let candidates: Vec<String> = names.iter().map(|n| (*n).to_string()).collect();
        crate::bridge::validate_ask(&response, &candidates).unwrap();
    }

    #[test]
    fn unknown_context_gets_no_safety_claim() {
        assert_eq!(causal_utility_prior("The door is open", "close it"), 0.0);
        assert_eq!(
            causal_utility_prior("The door is open", "leave it open"),
            0.0
        );
    }
}

/// Most tokens one request may send through [`NativeQwen::embed_texts`]. On a
/// loaded 24-core host one token of embedding cost about 11 ms (a 64 KiB
/// payload, 13389 tokens, took 148 s), so this bounds one call to under a
/// minute of one scorer slot. The startup seed is not bounded.
pub const MAX_EMBED_TOKENS_PER_CALL: usize = 4096;

/// Settings of the in-process scorer.
#[derive(Clone, Debug)]
pub struct NativeConfig {
    /// Forward passes allowed at once. Each pass already uses every core
    /// (candle's thread pool), so more concurrency only adds contention.
    pub max_concurrency: usize,
    /// Longest wait for a free slot before the call fails as overloaded.
    pub queue_timeout: Duration,
}

impl NativeConfig {
    /// `GENZERO_QWEN_MAX_CONCURRENCY` (default 2) and
    /// `GENZERO_QWEN_QUEUE_TIMEOUT_MS` (default 15000, the bridge's request timeout).
    pub fn from_env() -> Self {
        let num = |name: &str, default: u64| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .unwrap_or(default)
        };
        Self {
            max_concurrency: num("GENZERO_QWEN_MAX_CONCURRENCY", 2).max(1) as usize,
            queue_timeout: Duration::from_millis(num("GENZERO_QWEN_QUEUE_TIMEOUT_MS", 15_000)),
        }
    }
}

/// Qwen2.5 scorer loaded in this process.
pub struct NativeQwen {
    scorer: Arc<QwenSemanticScorer>,
    slots: Arc<Semaphore>,
    queue_timeout: Duration,
    load_ms: f64,
}

impl NativeQwen {
    /// Load weights (GGUF file or safetensors directory) and tokenizer.
    pub fn load(
        model: &Path,
        tokenizer: Option<&Path>,
        config: NativeConfig,
    ) -> Result<Self, ServiceError> {
        let started = Instant::now();
        let scorer = QwenSemanticScorer::load(model, tokenizer).map_err(|e| {
            ServiceError::Core(format!(
                "native Qwen scorer failed to load from {}: {e}",
                model.display()
            ))
        })?;
        let load_ms = started.elapsed().as_secs_f64() * 1e3;
        let info = scorer.info();
        tracing::info!(
            model = %info.source.display(),
            tokenizer = %info.tokenizer.display(),
            format = ?info.format,
            sha256 = %info.weights_sha256,
            ffn_pruning = ?info.ffn_pruning,
            simd = gen_zero_model::qwen::SIMD_KERNELS,
            load_ms,
            "native Qwen scorer loaded"
        );
        if gen_zero_model::qwen::SIMD_KERNELS == "scalar" {
            tracing::warn!(
                "native Qwen scorer built without AVX2/NEON kernels: quantized matmuls run \
                 scalar and scoring is many times slower (see .cargo/config.toml)"
            );
        }
        Ok(Self {
            scorer: Arc::new(scorer),
            slots: Arc::new(Semaphore::new(config.max_concurrency)),
            queue_timeout: config.queue_timeout,
            load_ms,
        })
    }

    pub fn info(&self) -> &QwenModelInfo {
        self.scorer.info()
    }

    /// Readiness view. Ready once constructed: loading already ran the risk
    /// demonstrations through the model.
    pub fn health(&self) -> Value {
        json!({
            "status": "ready",
            "backend": ENGINE_NATIVE_QWEN,
            "model": self.info(),
            "simd": gen_zero_model::qwen::SIMD_KERNELS,
            "load_ms": self.load_ms,
            "pooling": self.pooling().as_str(),
        })
    }

    /// The pooling every production dense embedding call ([`Self::embed`],
    /// [`Self::embed_texts`]) uses: length-normalized mean, best for
    /// document/long-text retrieval (see [`gen_zero_model::PoolingMode::Mean`]).
    pub fn pooling(&self) -> gen_zero_model::PoolingMode {
        gen_zero_model::PoolingMode::Mean
    }

    /// Fingerprint of the loaded weights ([`QwenSemanticScorer::backbone_id`]);
    /// every scorer, classifier, embedder and calibration id starts with it.
    pub fn backbone_id(&self) -> String {
        self.scorer.backbone_id()
    }

    /// Id of the vectors [`Self::embed_texts`] returns.
    pub fn embedder_id(&self) -> String {
        self.scorer.embedder_id_for(self.pooling())
    }

    /// Width of every vector [`Self::embed`] returns (896 for Qwen2.5-0.5B).
    pub fn embedding_dim(&self) -> usize {
        self.info().hidden_size
    }

    /// Unit-norm mean-pooled final hidden state of `text`
    /// ([`QwenSemanticScorer::embed`]), within [`MAX_EMBED_TOKENS_PER_CALL`].
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, BridgeError> {
        let mut vectors = self.embed_texts(vec![text.to_string()]).await?;
        vectors
            .pop()
            .ok_or_else(|| BridgeError::Native("no vector for one text".into()))
    }

    /// [`Self::embed`] for every text, in order, under one scorer slot. All
    /// texts together may hold at most [`MAX_EMBED_TOKENS_PER_CALL`] tokens;
    /// more is refused before any forward pass, never cut.
    pub async fn embed_texts(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BridgeError> {
        let pooling = self.pooling();
        let pooled = self
            .run(move |s| {
                let ids = texts
                    .iter()
                    .map(|t| s.encode(t))
                    .collect::<Result<Vec<_>, _>>()?;
                let tokens: usize = ids.iter().map(Vec::len).sum();
                if tokens > MAX_EMBED_TOKENS_PER_CALL {
                    return Ok(Err(tokens));
                }
                ids.iter()
                    .map(|i| s.embed_ids_with_pooling(i, pooling))
                    .collect::<Result<Vec<_>, _>>()
                    .map(Ok)
            })
            .await?;
        pooled.map_err(|tokens| BridgeError::EmbedBudget {
            tokens,
            max: MAX_EMBED_TOKENS_PER_CALL,
        })
    }

    /// [`Self::embed_texts`] on the calling thread, for startup work that runs
    /// before any request (the graph seed). It takes a scorer slot without
    /// waiting and fails as overloaded when none is free.
    pub fn embed_texts_blocking(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, BridgeError> {
        let _permit = self
            .slots
            .try_acquire()
            .map_err(|_| BridgeError::Overloaded { waited_ms: 0 })?;
        let pooling = self.pooling();
        texts
            .iter()
            .map(|t| self.scorer.embed_with_pooling(t, pooling))
            .collect::<Result<_, _>>()
            .map_err(|e| BridgeError::Native(e.to_string()))
    }

    /// Run one CPU-bound scorer call off the async executor, after a slot is free.
    async fn run<T, F>(&self, work: F) -> Result<T, BridgeError>
    where
        F: FnOnce(&QwenSemanticScorer) -> Result<T, gen_zero_model::ModelError> + Send + 'static,
        T: Send + 'static,
    {
        let waited = Instant::now();
        let permit =
            tokio::time::timeout(self.queue_timeout, Arc::clone(&self.slots).acquire_owned())
                .await
                .map_err(|_| BridgeError::Overloaded {
                    waited_ms: waited.elapsed().as_millis() as u64,
                })?
                .map_err(|_| BridgeError::Native("scorer slots closed".into()))?;
        let scorer = Arc::clone(&self.scorer);
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            work(&scorer)
        })
        .await
        .map_err(|e| BridgeError::Native(format!("scorer task failed: {e}")))?
        .map_err(|e| BridgeError::Native(e.to_string()))
    }
}

/// The semantic backend chosen at startup.
pub enum SemanticBackend {
    Native(NativeQwen),
    Remote(SemanticBridgeClient),
}

impl SemanticBackend {
    /// `_meta.engine` for outcomes this backend scored.
    pub fn engine_name(&self) -> &'static str {
        match self {
            Self::Native(_) => ENGINE_NATIVE_QWEN,
            Self::Remote(_) => ENGINE_SEMANTIC_BRIDGE,
        }
    }

    /// HTTP endpoint of the remote backend; `None` in process.
    pub fn endpoint(&self) -> Option<&str> {
        match self {
            Self::Native(_) => None,
            Self::Remote(client) => Some(client.endpoint()),
        }
    }

    /// Short description for `_meta.semantic_backend` and logs.
    pub fn describe(&self) -> Value {
        match self {
            Self::Native(native) => {
                let info = native.info();
                json!({
                    "kind": ENGINE_NATIVE_QWEN,
                    "model": info.source,
                    "format": info.format,
                    "weights_sha256": info.weights_sha256,
                    "tokenizer_sha256": info.tokenizer_sha256,
                    "backbone_id": native.backbone_id(),
                })
            }
            Self::Remote(client) => json!({"kind": "python_http", "endpoint": client.endpoint()}),
        }
    }

    /// The in-process scorer when it can embed text; the Python bridge has
    /// no text embedder.
    pub fn embedder(&self) -> Option<&NativeQwen> {
        match self {
            Self::Native(native) => Some(native),
            Self::Remote(_) => None,
        }
    }

    /// Dense embedding of `text` ([`NativeQwen::embed`]). The Python bridge
    /// refuses: it exposes no text embedding.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, BridgeError> {
        match self {
            Self::Native(native) => native.embed(text).await,
            Self::Remote(client) => Err(BridgeError::NoEmbedder(format!(
                "the Python bridge at {}",
                client.endpoint()
            ))),
        }
    }

    pub async fn semantic_ask(
        &self,
        input: &AskInput<'_>,
    ) -> Result<SemanticAskResponse, BridgeError> {
        let context = state_text(input.context, input.state);
        let mut response = match self {
            Self::Remote(client) => client.semantic_ask(input).await?,
            Self::Native(native) => {
                if input.return_embedding {
                    return Err(BridgeError::Native(
                        "the native scorer does not return prompt embeddings".into(),
                    ));
                }
                let context = state_text(input.context, input.state);
                let candidates = input.candidates.to_vec();
                let history = input.history.to_vec();
                let started = Instant::now();
                let (result, frame, source, id, dim) = native
                    .run(move |s| {
                        let (frame, source) = select_ask_frame(&context, &candidates, &history);
                        let r = s.score_pmi(&context, &candidates, frame, &history, None)?;
                        Ok((r, frame, source, s.scorer_id(), s.info().hidden_size))
                    })
                    .await?;
                let resp = SemanticAskResponse {
                    chosen: result.candidates[result.chosen_index].name.clone(),
                    chosen_index: result.chosen_index,
                    candidates: result.candidates,
                    entropy: result.entropy,
                    scorer: json!({
                        "id": id,
                        "frame": frame,
                        "frame_source": source,
                        "manifold": null,
                        "prompt_tokens": result.prompt_tokens,
                        "forward_ms": result.forward_ms,
                    }),
                    embedding_dim: dim,
                    embedding: None,
                    timing_ms: started.elapsed().as_secs_f64() * 1e3,
                };
                validate_ask(&resp, input.candidates)?;
                resp
            }
        };
        apply_policy_prior(&context, &mut response, causal_prior_enabled()?)?;
        validate_ask(&response, input.candidates)?;
        Ok(response)
    }

    pub async fn semantic_risk(&self, text: &str) -> Result<SemanticRiskResponse, BridgeError> {
        match self {
            Self::Remote(client) => client.semantic_risk(text).await,
            Self::Native(native) => {
                let text = text.to_string();
                let (r, id, report) = native
                    .run(move |s| {
                        Ok((
                            s.assess_risk_detailed(&text)?,
                            s.classifier_id(),
                            s.info().risk_report.clone(),
                        ))
                    })
                    .await?;
                let resp = SemanticRiskResponse {
                    p_dangerous: r.p_dangerous,
                    log_odds: r.log_odds,
                    windows: r.windows,
                    thresholds: RiskThresholds {
                        escalate: RISK_ESCALATE_THRESHOLD,
                        hard_stop: RISK_HARD_STOP_THRESHOLD,
                    },
                    classifier: risk_classifier_meta(&id, &r.per_order_log_odds, report.as_ref()),
                    forward_ms: r.forward_ms,
                };
                validate_risk(&resp)?;
                Ok(resp)
            }
        }
    }

    pub async fn semantic_route(
        &self,
        intent: &str,
        tools: &[Value],
        tool_names: &[String],
        top_k: usize,
        state: Option<&Value>,
    ) -> Result<SemanticRouteResponse, BridgeError> {
        match self {
            Self::Remote(client) => {
                client
                    .semantic_route(intent, tools, tool_names, top_k, state)
                    .await
            }
            Self::Native(native) => {
                if tools.len() != tool_names.len() {
                    return Err(BridgeError::Native(format!(
                        "{} tools for {} names",
                        tools.len(),
                        tool_names.len()
                    )));
                }
                let texts: Vec<String> = tools
                    .iter()
                    .zip(tool_names)
                    .map(|(t, name)| {
                        tool_continuation(name, t.get("description").and_then(|d| d.as_str()))
                    })
                    .collect();
                let context = state_text(intent, state);
                let names = tool_names.to_vec();
                let started = Instant::now();
                let (result, id) = native
                    .run(move |s| {
                        let r = s.score_pmi(&context, &names, ROUTE_FRAME, &[], Some(&texts))?;
                        Ok((r, s.scorer_id()))
                    })
                    .await?;
                let mut ranked: Vec<CandidateScore> = result.candidates;
                ranked.sort_by(|a, b| b.probability.total_cmp(&a.probability));
                let resp = SemanticRouteResponse {
                    selected: ranked
                        .iter()
                        .take(top_k.max(1))
                        .map(|c| c.name.clone())
                        .collect(),
                    ranked,
                    entropy: result.entropy,
                    scorer: json!({
                        "id": id,
                        "frame": ROUTE_FRAME,
                        "manifold": null,
                        "scored_text": "name_and_description",
                    }),
                    timing_ms: started.elapsed().as_secs_f64() * 1e3,
                };
                validate_route(&resp, tool_names)?;
                Ok(resp)
            }
        }
    }
}
