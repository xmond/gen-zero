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
    ROUTE_FRAME,
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

fn apply_causal_prior(
    context: &str,
    response: &mut SemanticAskResponse,
) -> Result<(), BridgeError> {
    let adjustments: Vec<f64> = response
        .candidates
        .iter()
        .map(|candidate| causal_utility_prior(context, &candidate.name))
        .collect();
    if adjustments.iter().all(|value| *value == 0.0) {
        return Ok(());
    }
    let logits: Vec<f64> = response
        .candidates
        .iter()
        .zip(&adjustments)
        .map(|(candidate, adjustment)| {
            candidate.probability.max(f64::MIN_POSITIVE).ln() + adjustment
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
        candidate.pmi += adjustments[index];
        candidate.probability = weights[index] / total;
    }
    response.chosen_index = response
        .candidates
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.probability.total_cmp(&b.probability))
        .map(|(index, _)| index)
        .ok_or_else(|| BridgeError::InvalidResponse("no candidates".into()))?;
    response.chosen = response.candidates[response.chosen_index].name.clone();
    response.entropy = normalized_entropy(
        &response
            .candidates
            .iter()
            .map(|c| c.probability)
            .collect::<Vec<_>>(),
    );
    response.scorer["causal_utility_prior"] = json!("contextual_v1");
    Ok(())
}

#[cfg(test)]
mod causal_prior_tests {
    use super::*;

    fn score(context: &str, names: &[&str]) -> SemanticAskResponse {
        let mut response = SemanticAskResponse {
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
                })
                .collect(),
            entropy: 0.3,
            scorer: json!({}),
            embedding_dim: 0,
            embedding: None,
            timing_ms: 0.0,
        };
        apply_causal_prior(context, &mut response).unwrap();
        response
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

    #[test]
    fn unknown_context_gets_no_safety_claim() {
        assert_eq!(causal_utility_prior("The door is open", "close it"), 0.0);
        assert_eq!(
            causal_utility_prior("The door is open", "leave it open"),
            0.0
        );
    }
}

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
        })
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
                })
            }
            Self::Remote(client) => json!({"kind": "python_http", "endpoint": client.endpoint()}),
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
        apply_causal_prior(&context, &mut response)?;
        validate_ask(&response, input.candidates)?;
        Ok(response)
    }

    pub async fn semantic_risk(&self, text: &str) -> Result<SemanticRiskResponse, BridgeError> {
        match self {
            Self::Remote(client) => client.semantic_risk(text).await,
            Self::Native(native) => {
                let text = text.to_string();
                let (r, id) = native
                    .run(move |s| Ok((s.assess_risk_detailed(&text)?, s.classifier_id())))
                    .await?;
                let resp = SemanticRiskResponse {
                    p_dangerous: r.p_dangerous,
                    log_odds: r.log_odds,
                    windows: r.windows,
                    thresholds: RiskThresholds {
                        escalate: RISK_ESCALATE_THRESHOLD,
                        hard_stop: RISK_HARD_STOP_THRESHOLD,
                    },
                    classifier: json!({
                        "id": id,
                        "method": "in_context_pmi_log_odds",
                        "per_order_log_odds": r.per_order_log_odds,
                        "thresholds_calibrated_on": "python fp32 backbone (risk_data/report.json)",
                    }),
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
