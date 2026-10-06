//! In-process semantic scorer and risk classifier over [`QwenModel`].
//!
//! This is the Rust port of the Python scorer that used to sit behind the
//! HTTP bridge (`python/gen_zero/service/semantic_scorer.py` and
//! `semantic_risk.py`). Prompt strings, frames, demonstrations, demonstration
//! orders, windows and thresholds are the same, so the Python calibration in
//! `python/gen_zero/service/risk_data/report.json` applies and can be checked
//! row by row (`tests::risk_parity_with_python_report`).
//!
//! Candidate rule (Holtzman et al. 2021, domain-conditional PMI):
//! `score(c) = log P(c | context + frame) - log P(c | frame)`, softmax over
//! candidates. The second term removes the prior of the candidate string.
//!
//! Risk rule: few-shot in-context classification. The calibrated log-odds of
//! the next token `" dangerous"` over `" safe"` after the request, minus the
//! same log-odds after an empty request, averaged over three demonstration
//! orders; the riskiest 192-token window decides.
//!
//! No keyword table, hash, or fallback model exists here. Every failure is a
//! typed [`ModelError`]; the caller decides how to fail closed.

use crate::error::ModelError;
use crate::qwen::{candle_err, sha256_file, KvCache, QwenModel, WeightFormat, MAX_SEQ_LEN};
use candle_core::{IndexOp, Tensor};
use serde::Serialize;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;
use tokenizers::Tokenizer;

pub const ASK_FRAME: &str = "The first action to take is:";
pub const NEXT_FRAME: &str = "The next action to take is:";
pub const ROUTE_FRAME: &str = "The tool to use is:";
/// An action frame after a question asks the model the wrong thing.
pub const QA_FRAME: &str = "The correct answer is:";
const QA_CONTEXT_MARKERS: [&str; 2] = ["Question:", "Answer with"];
pub const MAX_CANDIDATES: usize = 64;
const BASELINE_CACHE_SIZE: usize = 512;

/// Thresholds calibrated on `risk_data/calibration.jsonl` with the Python
/// fp32 backbone (see `risk_data/README.md`). Same values the Python service
/// returned; the parity test measures how far the native backends move scores.
pub const RISK_ESCALATE_THRESHOLD: f64 = 0.4494;
pub const RISK_HARD_STOP_THRESHOLD: f64 = 0.7620;

const RISK_HEADER: &str = "Each request is rated for safety risk. dangerous = destructive, \
irreversible, privilege escalation, or bypassing security. safe = ordinary work.\n\n";
const RISK_LABELS: [&str; 2] = ["safe", "dangerous"];
const RISK_WINDOW_TOKENS: usize = 192;
const RISK_WINDOW_STRIDE: usize = 128;
/// The demonstrations, shared with the Python classifier (one source of truth).
const RISK_SHOTS_JSONL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../python/gen_zero/service/risk_data/shots.jsonl"
));
/// Demonstration orders of the Python classifier: order 0 is the file order,
/// orders 1 and 2 are `random.Random(0).shuffle` applied twice in sequence to
/// the 16 shots. Generated with:
/// `python3 -c "import random; r=random.Random(0); o=[list(range(16))]; \
///  [o.append((lambda l: (r.shuffle(l), l)[1])(list(range(16)))) for _ in (1,2)]; print(o)"`
const RISK_ORDERS: [[usize; 16]; 3] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [10, 14, 5, 1, 9, 2, 3, 11, 13, 7, 8, 4, 0, 6, 15, 12],
    [3, 7, 13, 11, 6, 5, 0, 10, 14, 8, 4, 15, 1, 12, 2, 9],
];

/// How token hidden states become one dense embedding
/// ([`QwenSemanticScorer::embed_with_pooling`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolingMode {
    /// Length-normalized mean of every token's final hidden state, then
    /// scaled to unit L2 norm. Every token of the text contributes equally,
    /// so a long and a short paraphrase of the same meaning still land close:
    /// the production choice for document and payload retrieval
    /// (`graph_deposit` / `graph_rag` dense track).
    Mean,
    /// The final hidden state of the last token only, unit L2 normalized. An
    /// autoregressive decoder's last-token state already summarizes the
    /// prefix that produced it, so this pools a chain-of-thought prefix
    /// without averaging its intermediate reasoning tokens into the vector.
    LastToken,
}

impl PoolingMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::LastToken => "last_token",
        }
    }
}

/// Continuation text for one candidate. Same rule for every language.
pub fn candidate_text(name: &str) -> String {
    format!(" {}", name.replace('_', " ").trim())
}

/// What the model scores for a tool: its name and, if given, its description.
pub fn tool_continuation(name: &str, description: Option<&str>) -> String {
    let desc = description
        .map(|d| d.split_whitespace().collect::<Vec<_>>().join(" "))
        .unwrap_or_default();
    if desc.is_empty() {
        candidate_text(name)
    } else {
        format!("{}: {desc}", candidate_text(name))
    }
}

/// `(frame, source)` for an ask: QA context first, then history, then the default.
pub fn select_ask_frame(
    context: &str,
    candidates: &[String],
    history: &[String],
) -> (&'static str, &'static str) {
    let mut names: Vec<String> = candidates.iter().map(|c| c.trim().to_lowercase()).collect();
    names.sort();
    names.dedup();
    let qa_set = names == ["no", "yes"] || names == ["false", "true"];
    if QA_CONTEXT_MARKERS.iter().any(|m| context.contains(m)) || qa_set {
        (QA_FRAME, "qa")
    } else if !history.is_empty() {
        (NEXT_FRAME, "history")
    } else {
        (ASK_FRAME, "default")
    }
}

/// Context, action history and frame joined by newlines.
pub fn compose_prompt(context: &str, frame: &str, history: &[String]) -> String {
    let mut parts: Vec<String> = Vec::new();
    if !context.trim().is_empty() {
        parts.push(context.trim().to_string());
    }
    if !history.is_empty() {
        let taken: Vec<&str> = history.iter().map(|h| h.trim()).collect();
        parts.push(format!("Actions already taken: {}", taken.join(", ")));
    }
    parts.push(frame.to_string());
    parts.join("\n")
}

/// Context text with the request `state` appended, serialized as Python's
/// `json.dumps(state, ensure_ascii=False, sort_keys=True)` (keys sorted,
/// `", "` and `": "` separators), so the prompt matches the Python scorer.
pub fn state_text(context: &str, state: Option<&serde_json::Value>) -> String {
    let Some(state) = state else {
        return context.to_string();
    };
    let extra = match state {
        serde_json::Value::String(s) => s.clone(),
        other => python_json(other),
    };
    if context.is_empty() {
        extra
    } else {
        format!("{context}\n{extra}")
    }
}

fn python_json(value: &serde_json::Value) -> String {
    use serde_json::Value;
    match value {
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(python_json).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let body: Vec<String> = keys
                .into_iter()
                .map(|k| format!("{}: {}", Value::String(k.clone()), python_json(&map[k])))
                .collect();
            format!("{{{}}}", body.join(", "))
        }
        scalar => scalar.to_string(),
    }
}

/// Normalized Shannon entropy in [0, 1] (1 = uniform).
pub fn normalized_entropy(probs: &[f64]) -> f64 {
    if probs.len() <= 1 {
        return 0.0;
    }
    let h: f64 = probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();
    (h / (probs.len() as f64).ln()).clamp(0.0, 1.0)
}

/// Numerically stable softmax. Non-finite input is an error, not a NaN output.
pub fn softmax(values: &[f64]) -> Result<Vec<f64>, ModelError> {
    if values.is_empty() {
        return Err(ModelError::EmptyOptions);
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(ModelError::NumericalInstability(
            "non-finite log-likelihood".into(),
        ));
    }
    let top = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = values.iter().map(|v| (v - top).exp()).collect();
    let total: f64 = exps.iter().sum();
    Ok(exps.into_iter().map(|e| e / total).collect())
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Overlapping token windows; nothing is truncated.
fn risk_windows(ids: &[u32]) -> Vec<&[u32]> {
    if ids.len() <= RISK_WINDOW_TOKENS {
        return vec![ids];
    }
    let mut out = Vec::new();
    let mut start = 0;
    loop {
        let end = (start + RISK_WINDOW_TOKENS).min(ids.len());
        out.push(&ids[start..end]);
        if start + RISK_WINDOW_TOKENS >= ids.len() {
            break;
        }
        start += RISK_WINDOW_STRIDE;
    }
    out
}

#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize)]
pub struct CandidateScore {
    pub name: String,
    pub log_likelihood: f64,
    pub baseline_log_likelihood: f64,
    pub pmi: f64,
    pub probability: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ScoreResult {
    pub candidates: Vec<CandidateScore>,
    pub chosen_index: usize,
    pub entropy: f64,
    pub prompt_tokens: usize,
    pub forward_ms: f64,
}

/// A conservative contextual preference prior in log-odds units.
/// Positive values reward a remediation; negative values flag a known hazard.
/// Unknown situations receive no adjustment and remain subject to the usual gate.
pub fn causal_utility_prior(context: &str, action: &str) -> f64 {
    let c = context.to_ascii_lowercase();
    let a = action.to_ascii_lowercase().replace('_', " ");
    let has = |terms: &[&str]| terms.iter().any(|term| c.contains(term));
    let does = |terms: &[&str]| terms.iter().any(|term| a.contains(term));
    let spoiled = has(&[
        "expired",
        "past expiration",
        "spoiled",
        "smells sour",
        "rotten",
        "poisonous",
        "poisoned",
    ]);
    if spoiled {
        if does(&["drink", "eat", "consume", "taste", "swallow"]) {
            return -16.0;
        }
        if does(&[
            "throw away",
            "throw it away",
            "discard",
            "dispose",
            "do not drink",
            "do not eat",
        ]) {
            return 8.0;
        }
    }
    let electrical = has(&[
        "high-voltage",
        "high voltage",
        "live wire",
        "exposed wire",
        "uninsulated wire",
    ]);
    if electrical {
        if does(&["touch", "grab", "hold", "bare hand"]) {
            return -16.0;
        }
        if does(&[
            "electrician",
            "shut off",
            "turn off",
            "main breaker",
            "keep away",
        ]) {
            return 8.0;
        }
    }
    if has(&[
        "radioactive",
        "corrosive chemical",
        "toxic chemical",
        "hazardous chemical",
    ]) {
        if does(&[
            "touch",
            "handle with bare hands",
            "pick up",
            "inhale",
            "taste",
        ]) {
            return -16.0;
        }
        if does(&[
            "keep away",
            "call emergency",
            "contact hazardous materials",
            "wear protective equipment",
        ]) {
            return 8.0;
        }
    }
    if does(&[
        "rm -rf /",
        "rm -rf /*",
        "recursive root deletion",
        "delete root directory",
        "erase the entire disk",
    ]) {
        return -16.0;
    }
    0.0
}

#[derive(Clone, Debug, Serialize)]
pub struct RiskAssessment {
    pub p_dangerous: f64,
    pub log_odds: f64,
    pub per_order_log_odds: Vec<f64>,
    pub windows: usize,
    pub forward_ms: f64,
}

/// Identity of the loaded weights, reported in scorer ids and readiness.
#[derive(Clone, Debug, Serialize)]
pub struct QwenModelInfo {
    pub source: PathBuf,
    pub tokenizer: PathBuf,
    pub format: WeightFormat,
    /// SHA-256 of the GGUF file or of `model.safetensors`.
    pub weights_sha256: String,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
}

/// Demonstration prefixes of the risk classifier (one per order) and the
/// log-odds each gives an empty request (the PMI baseline).
struct RiskPrefixes {
    caches: RiskCaches,
    baselines: Vec<f64>,
}

enum RiskCaches {
    /// All orders in one batch-`n` cache: their token counts match (they do
    /// for the shipped demonstrations), so one forward scores every order.
    Batched(KvCache),
    /// One batch-1 cache per order, run one after another.
    Separate(Vec<KvCache>),
}

impl RiskCaches {
    fn orders(&self) -> usize {
        match self {
            Self::Batched(cache) => cache.batch(),
            Self::Separate(caches) => caches.len(),
        }
    }
}

type BaselineKey = (String, Vec<String>);

/// Small LRU for premise-only baselines: they depend on the frame and the
/// candidate texts only, and repeat across requests.
#[derive(Default)]
struct BaselineCache {
    map: HashMap<BaselineKey, Vec<f64>>,
    order: VecDeque<BaselineKey>,
}

impl BaselineCache {
    fn get(&mut self, key: &BaselineKey) -> Option<Vec<f64>> {
        let hit = self.map.get(key).cloned()?;
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            let k = self.order.remove(pos).expect("position is in range");
            self.order.push_back(k);
        }
        Some(hit)
    }

    fn put(&mut self, key: BaselineKey, scores: Vec<f64>) {
        if self.map.insert(key.clone(), scores).is_none() {
            self.order.push_back(key);
        }
        while self.order.len() > BASELINE_CACHE_SIZE {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            }
        }
    }
}

/// Qwen2.5 semantic scorer and risk classifier, fully in process.
pub struct QwenSemanticScorer {
    model: QwenModel,
    tokenizer: Tokenizer,
    info: QwenModelInfo,
    label_ids: [u32; 2],
    request_ids: Vec<u32>,
    frame_ids: Vec<u32>,
    risk: Option<RiskPrefixes>,
    baselines: Mutex<BaselineCache>,
}

impl QwenSemanticScorer {
    /// Load weights and tokenizer. `model_path` is a `.gguf` file or a Hugging
    /// Face directory. `tokenizer` defaults to `tokenizer.json` in the model
    /// directory (or next to the GGUF file); a missing tokenizer is an error.
    ///
    /// The risk demonstrations are encoded here, once, so a broken model fails
    /// at startup instead of on the first request.
    pub fn load(model_path: &Path, tokenizer: Option<&Path>) -> Result<Self, ModelError> {
        let tokenizer_path = match tokenizer {
            Some(p) => p.to_path_buf(),
            None if model_path.is_dir() => model_path.join("tokenizer.json"),
            None => model_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("tokenizer.json"),
        };
        let tok = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            ModelError::QwenLoad(format!(
                "tokenizer {}: {e} (pass the Qwen2.5 tokenizer.json explicitly)",
                tokenizer_path.display()
            ))
        })?;
        let model = QwenModel::load(model_path)?;
        if !model_path.is_dir() {
            check_gguf_vocab(model_path, &tok)?;
        }
        let weights_file = if model_path.is_dir() {
            model_path.join("model.safetensors")
        } else {
            model_path.to_path_buf()
        };
        let cfg = model.config().clone();
        let info = QwenModelInfo {
            source: model_path.to_path_buf(),
            tokenizer: tokenizer_path,
            format: model.format().clone(),
            weights_sha256: sha256_file(&weights_file)?,
            hidden_size: cfg.hidden_size,
            num_layers: cfg.num_layers,
            vocab_size: cfg.vocab_size,
        };
        let mut scorer = Self {
            model,
            label_ids: [0, 0],
            request_ids: Vec::new(),
            frame_ids: Vec::new(),
            tokenizer: tok,
            info,
            risk: None,
            baselines: Mutex::new(BaselineCache::default()),
        };
        scorer.label_ids = [
            scorer.single_token(&format!(" {}", RISK_LABELS[0]))?,
            scorer.single_token(&format!(" {}", RISK_LABELS[1]))?,
        ];
        scorer.request_ids = scorer.encode("Request: ")?;
        scorer.frame_ids = scorer.encode("\nRisk:")?;
        scorer.risk = Some(scorer.build_risk_prefixes()?);
        Ok(scorer)
    }

    pub fn info(&self) -> &QwenModelInfo {
        &self.info
    }

    pub fn model(&self) -> &QwenModel {
        &self.model
    }

    /// Stable id of the candidate scorer, e.g. `qwen2:gguf-q8_0/tied-lm-head/pmi`.
    pub fn scorer_id(&self) -> String {
        format!("{}/tied-lm-head/pmi", self.backbone_id())
    }

    pub fn classifier_id(&self) -> String {
        format!(
            "{}/icl-{}shot-x{}/pmi",
            self.backbone_id(),
            RISK_ORDERS[0].len(),
            RISK_ORDERS.len()
        )
    }

    fn backbone_id(&self) -> String {
        match &self.info.format {
            WeightFormat::Gguf { dtype } => format!("qwen2-native:gguf-{dtype}-dequant-f32"),
            WeightFormat::SafetensorsF32 => "qwen2-native:safetensors-f32".into(),
        }
    }

    /// Token ids without special tokens (the Python runtime's `token_ids`).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, ModelError> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| ModelError::QwenInput(format!("tokenize: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Id of the text embedding [`Self::embed`] returns.
    pub fn embedder_id(&self) -> String {
        self.embedder_id_for(PoolingMode::Mean)
    }

    /// [`Self::embedder_id`] for a given [`PoolingMode`], e.g.
    /// `qwen2-native:.../final-norm-mean-pool-l2` or
    /// `.../final-norm-last_token-pool-l2`.
    pub fn embedder_id_for(&self, pooling: PoolingMode) -> String {
        format!(
            "{}/final-norm-{}-pool-l2",
            self.backbone_id(),
            pooling.as_str()
        )
    }

    /// [`Self::embed_with_pooling`] with [`PoolingMode::Mean`], the pooling
    /// every production dense-track caller uses today.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, ModelError> {
        self.embed_with_pooling(text, PoolingMode::Mean)
    }

    /// Dense embedding of `text` under `pooling` (`hidden_size` wide, 896 for
    /// Qwen2.5-0.5B), scaled to unit L2 norm. Refused: text that tokenizes to
    /// nothing, and a pooled vector that is not finite or has norm 0.
    pub fn embed_with_pooling(
        &self,
        text: &str,
        pooling: PoolingMode,
    ) -> Result<Vec<f32>, ModelError> {
        self.embed_ids_with_pooling(&self.encode(text)?, pooling)
    }

    /// [`Self::embed`] of text already tokenized with [`Self::encode`].
    pub fn embed_ids(&self, ids: &[u32]) -> Result<Vec<f32>, ModelError> {
        self.embed_ids_with_pooling(ids, PoolingMode::Mean)
    }

    /// [`Self::embed_with_pooling`] of text already tokenized with [`Self::encode`].
    ///
    /// [`PoolingMode::Mean`]: a text longer than [`MAX_SEQ_LEN`] tokens runs as
    /// consecutive windows of at most that many tokens, each with no context
    /// from the one before, and every token of every window counts once in the
    /// mean: nothing is cut.
    ///
    /// [`PoolingMode::LastToken`]: only the final hidden state of the last
    /// token is pooled; a text longer than [`MAX_SEQ_LEN`] tokens is truncated
    /// to its last [`MAX_SEQ_LEN`] tokens first, so the pooled state's context
    /// is the most recent window, not the whole text.
    pub fn embed_ids_with_pooling(
        &self,
        ids: &[u32],
        pooling: PoolingMode,
    ) -> Result<Vec<f32>, ModelError> {
        if ids.is_empty() {
            return Err(ModelError::QwenInput(
                "text to embed tokenized to nothing".into(),
            ));
        }
        let err = candle_err("text embedding");
        match pooling {
            PoolingMode::Mean => {
                let mut sum = Tensor::zeros(
                    self.info.hidden_size,
                    candle_core::DType::F32,
                    self.model.device(),
                )
                .map_err(&err)?;
                for window in ids.chunks(MAX_SEQ_LEN) {
                    let (hidden, _) = self.model.forward(&[window.to_vec()], None)?;
                    let window_sum = hidden.i(0).and_then(|h| h.sum(0)).map_err(&err)?;
                    sum = (sum + window_sum).map_err(&err)?;
                }
                let pooled = sum.to_vec1::<f32>().map_err(&err)?;
                mean_l2_normalize(pooled, ids.len())
            }
            PoolingMode::LastToken => {
                let window: Vec<u32> = ids
                    .len()
                    .checked_sub(MAX_SEQ_LEN)
                    .map(|start| ids[start..].to_vec())
                    .unwrap_or_else(|| ids.to_vec());
                let (hidden, _) = self.model.forward(std::slice::from_ref(&window), None)?;
                let last = hidden.i((0, window.len() - 1)).map_err(&err)?;
                let pooled = last.to_vec1::<f32>().map_err(&err)?;
                mean_l2_normalize(pooled, 1)
            }
        }
    }

    fn single_token(&self, text: &str) -> Result<u32, ModelError> {
        match self.encode(text)?.as_slice() {
            [id] => Ok(*id),
            ids => Err(ModelError::QwenLoad(format!(
                "label {text:?} must be one token, got {ids:?}"
            ))),
        }
    }

    /// Sum of token log-probabilities of each continuation after `prompt`,
    /// and the prompt length in tokens.
    ///
    /// One prompt pass fills the KV cache; all continuations then run as one
    /// right-padded batch on top of it.
    pub fn continuation_log_likelihoods(
        &self,
        prompt: &str,
        texts: &[String],
    ) -> Result<(Vec<f64>, usize), ModelError> {
        let prompt_ids = self.encode(prompt)?;
        if prompt_ids.is_empty() {
            return Err(ModelError::QwenInput("prompt tokenized to nothing".into()));
        }
        let rows = texts
            .iter()
            .map(|t| self.encode(t))
            .collect::<Result<Vec<_>, _>>()?;
        if rows.iter().any(Vec::is_empty) {
            return Err(ModelError::QwenInput(
                "a candidate tokenized to nothing".into(),
            ));
        }
        if rows
            .iter()
            .any(|r| prompt_ids.len() + r.len() > MAX_SEQ_LEN)
        {
            return Err(ModelError::QwenInput(format!(
                "prompt + candidate exceeds {MAX_SEQ_LEN} tokens; refusing to truncate"
            )));
        }
        let err = candle_err("continuation scoring");
        let (hidden, cache) = self
            .model
            .forward(std::slice::from_ref(&prompt_ids), None)?;
        let last = hidden.i((0, prompt_ids.len() - 1)).map_err(&err)?;
        let firsts: Vec<u32> = rows.iter().map(|r| r[0]).collect();
        let first_lp = self
            .model
            .log_probs_of(&last.unsqueeze(0).map_err(&err)?, &[firsts])?
            .remove(0);
        let mut scores = first_lp;
        let longest = rows.iter().map(Vec::len).max().unwrap_or(1);
        if longest > 1 {
            // Every row but its last token is fed; position j predicts token j + 1.
            let width = longest - 1;
            let batch: Vec<Vec<u32>> = rows
                .iter()
                .map(|r| {
                    let mut fed = r[..r.len() - 1].to_vec();
                    fed.resize(width, 0);
                    fed
                })
                .collect();
            let (hidden, _) = self.model.forward(&batch, Some(&cache))?;
            let (b, t, h) = hidden.dims3().map_err(&err)?;
            let flat = hidden.reshape((b * t, h)).map_err(&err)?;
            let mut index = Vec::new();
            let mut targets = Vec::new();
            let mut owner = Vec::new();
            for (i, row) in rows.iter().enumerate() {
                for (j, &next) in row.iter().enumerate().skip(1) {
                    index.push((i * t + j - 1) as u32);
                    targets.push(vec![next]);
                    owner.push(i);
                }
            }
            let index = Tensor::from_vec(index, owner.len(), self.model.device()).map_err(&err)?;
            let picked = flat.index_select(&index, 0).map_err(&err)?;
            for (lp, i) in self
                .model
                .log_probs_of(&picked, &targets)?
                .into_iter()
                .zip(owner)
            {
                scores[i] += lp[0];
            }
        }
        if scores.iter().any(|s| !s.is_finite()) {
            return Err(ModelError::NumericalInstability(
                "non-finite candidate log-likelihood".into(),
            ));
        }
        Ok((scores, prompt_ids.len()))
    }

    /// Probability distribution over `candidates` after `prompt`: the softmax
    /// of each candidate's conditional log-likelihood (no calibration).
    pub fn score_candidates(
        &self,
        prompt: &str,
        candidates: &[String],
    ) -> Result<Vec<f64>, ModelError> {
        validate_candidates(candidates)?;
        let texts: Vec<String> = candidates.iter().map(|c| candidate_text(c)).collect();
        let (ll, _) = self.continuation_log_likelihoods(prompt, &texts)?;
        softmax(&ll)
    }

    /// Calibrated (PMI) candidate ranking, the rule of the ask/route/imagine verbs.
    ///
    /// `texts` overrides the continuation of each name (route scores
    /// `"<name>: <description>"`). The premise is the prompt without the
    /// context, so the PMI measures what the context adds.
    pub fn score_pmi(
        &self,
        context: &str,
        names: &[String],
        frame: &str,
        history: &[String],
        texts: Option<&[String]>,
    ) -> Result<ScoreResult, ModelError> {
        validate_candidates(names)?;
        if context.trim().is_empty() {
            return Err(ModelError::QwenInput(
                "context must be a non-empty string".into(),
            ));
        }
        let texts: Vec<String> = match texts {
            Some(t) if t.len() == names.len() => t.to_vec(),
            Some(t) => {
                return Err(ModelError::CandidateMismatch {
                    expected: names.len(),
                    actual: t.len(),
                })
            }
            None => names.iter().map(|n| candidate_text(n)).collect(),
        };
        let started = Instant::now();
        let prompt = compose_prompt(context, frame, history);
        let (conditional, prompt_tokens) = self.continuation_log_likelihoods(&prompt, &texts)?;
        let premise = compose_prompt("", frame, history);
        let key = (premise.clone(), texts.clone());
        let cached = self
            .baselines
            .lock()
            .map_err(|_| ModelError::Inference("baseline cache poisoned".into()))?
            .get(&key);
        let baseline = match cached {
            Some(b) => b,
            None => {
                let (b, _) = self.continuation_log_likelihoods(&premise, &texts)?;
                self.baselines
                    .lock()
                    .map_err(|_| ModelError::Inference("baseline cache poisoned".into()))?
                    .put(key, b.clone());
                b
            }
        };
        let pmi: Vec<f64> = conditional
            .iter()
            .zip(&baseline)
            .map(|(c, b)| c - b)
            .collect();
        let probs = softmax(&pmi)?;
        let candidates: Vec<CandidateScore> = names
            .iter()
            .enumerate()
            .map(|(i, n)| CandidateScore {
                name: n.clone(),
                log_likelihood: conditional[i],
                baseline_log_likelihood: baseline[i],
                pmi: pmi[i],
                probability: probs[i],
            })
            .collect();
        // First maximum wins ties, as Python's max over indices does.
        let chosen_index = probs
            .iter()
            .enumerate()
            .fold(0, |best, (i, &p)| if p > probs[best] { i } else { best });
        Ok(ScoreResult {
            chosen_index,
            entropy: normalized_entropy(&probs),
            candidates,
            prompt_tokens,
            forward_ms: started.elapsed().as_secs_f64() * 1e3,
        })
    }

    fn build_risk_prefixes(&self) -> Result<RiskPrefixes, ModelError> {
        let shots: Vec<(String, usize)> = RISK_SHOTS_JSONL
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let v: serde_json::Value = serde_json::from_str(l)
                    .map_err(|e| ModelError::QwenLoad(format!("risk shot: {e}")))?;
                let text = v["text"].as_str().unwrap_or_default().to_string();
                let label = v["label"].as_u64().unwrap_or(u64::MAX) as usize;
                if text.is_empty() || label > 1 {
                    return Err(ModelError::QwenLoad(format!("malformed risk shot {l}")));
                }
                Ok((text, label))
            })
            .collect::<Result<_, _>>()?;
        if shots.len() != RISK_ORDERS[0].len() {
            return Err(ModelError::QwenLoad(format!(
                "{} risk demonstrations, the pinned orders cover {}",
                shots.len(),
                RISK_ORDERS[0].len()
            )));
        }
        let demos = RISK_ORDERS
            .iter()
            .map(|order| {
                let mut demo = RISK_HEADER.to_string();
                for &i in order {
                    let (text, label) = &shots[i];
                    demo.push_str(&format!(
                        "Request: {text}\nRisk: {}\n\n",
                        RISK_LABELS[*label]
                    ));
                }
                self.encode(&demo)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let caches = if demos.iter().all(|d| d.len() == demos[0].len()) {
            RiskCaches::Batched(self.model.forward(&demos, None)?.1)
        } else {
            RiskCaches::Separate(
                demos
                    .iter()
                    .map(|d| {
                        self.model
                            .forward(std::slice::from_ref(d), None)
                            .map(|(_, c)| c)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        };
        let mut prefixes = RiskPrefixes {
            caches,
            baselines: Vec::new(),
        };
        prefixes.baselines = self.raw_log_odds(&prefixes, &[])?;
        Ok(prefixes)
    }

    /// Raw log-odds `log P(" dangerous") - log P(" safe")` after each
    /// demonstration order followed by `Request: <text_ids>\nRisk:`.
    fn raw_log_odds(
        &self,
        prefixes: &RiskPrefixes,
        text_ids: &[u32],
    ) -> Result<Vec<f64>, ModelError> {
        let tail: Vec<u32> = self
            .request_ids
            .iter()
            .chain(text_ids)
            .chain(&self.frame_ids)
            .copied()
            .collect();
        let err = candle_err("risk log-odds");
        let last_rows = match &prefixes.caches {
            RiskCaches::Batched(cache) => {
                let rows = vec![tail.clone(); cache.batch()];
                let (hidden, _) = self.model.forward(&rows, Some(cache))?;
                hidden.i((.., tail.len() - 1)).map_err(&err)?
            }
            RiskCaches::Separate(caches) => {
                let mut rows = Vec::with_capacity(caches.len());
                for cache in caches {
                    let (hidden, _) = self
                        .model
                        .forward(std::slice::from_ref(&tail), Some(cache))?;
                    rows.push(hidden.i((.., tail.len() - 1)).map_err(&err)?);
                }
                Tensor::cat(&rows, 0).map_err(&err)?
            }
        };
        let targets = vec![self.label_ids.to_vec(); prefixes.caches.orders()];
        self.model
            .log_probs_of(&last_rows, &targets)?
            .into_iter()
            .map(|lp| {
                let value = lp[1] - lp[0];
                if value.is_finite() {
                    Ok(value)
                } else {
                    Err(ModelError::NumericalInstability(
                        "non-finite risk log-odds".into(),
                    ))
                }
            })
            .collect()
    }

    /// Full risk assessment of `text` (any language).
    pub fn assess_risk_detailed(&self, text: &str) -> Result<RiskAssessment, ModelError> {
        let text = text.trim();
        if text.is_empty() {
            return Err(ModelError::QwenInput(
                "text must be a non-empty string".into(),
            ));
        }
        let ids = self.encode(text)?;
        if ids.is_empty() {
            return Err(ModelError::QwenInput("text tokenized to nothing".into()));
        }
        let started = Instant::now();
        let windows = risk_windows(&ids);
        let prefixes = self
            .risk
            .as_ref()
            .ok_or_else(|| ModelError::Inference("risk demonstrations not loaded".into()))?;
        let mut best: Option<Vec<f64>> = None;
        for window in &windows {
            let per_order: Vec<f64> = self
                .raw_log_odds(prefixes, window)?
                .iter()
                .zip(&prefixes.baselines)
                .map(|(raw, base)| raw - base)
                .collect();
            let sum: f64 = per_order.iter().sum();
            if best.as_ref().is_none_or(|b| sum > b.iter().sum::<f64>()) {
                best = Some(per_order);
            }
        }
        let best = best.expect("at least one window");
        let mean = best.iter().sum::<f64>() / best.len() as f64;
        Ok(RiskAssessment {
            p_dangerous: sigmoid(mean),
            log_odds: mean,
            per_order_log_odds: best,
            windows: windows.len(),
            forward_ms: started.elapsed().as_secs_f64() * 1e3,
        })
    }

    /// Probability that carrying out `text` is destructive, irreversible,
    /// privilege-escalating or security-bypassing.
    pub fn assess_risk(&self, text: &str) -> Result<f32, ModelError> {
        Ok(self.assess_risk_detailed(text)?.p_dangerous as f32)
    }
}

fn validate_candidates(names: &[String]) -> Result<(), ModelError> {
    if names.len() < 2 {
        return Err(ModelError::QwenInput("need at least two candidates".into()));
    }
    if names.len() > MAX_CANDIDATES {
        return Err(ModelError::QwenInput(format!(
            "at most {MAX_CANDIDATES} candidates"
        )));
    }
    if let Some(i) = names.iter().position(|n| n.trim().is_empty()) {
        return Err(ModelError::EmptyOptionLength { option_index: i });
    }
    let mut seen = std::collections::HashSet::new();
    if !names.iter().all(|n| seen.insert(n)) {
        return Err(ModelError::QwenInput("candidates must be distinct".into()));
    }
    Ok(())
}

/// Divide a token sum by `tokens` and scale it to unit L2 norm. A value that
/// is not finite or a norm of 0 is refused, never returned as a vector.
fn mean_l2_normalize(sum: Vec<f32>, tokens: usize) -> Result<Vec<f32>, ModelError> {
    if tokens == 0 {
        return Err(ModelError::QwenInput("no token to pool".into()));
    }
    let mean: Vec<f64> = sum.iter().map(|&v| f64::from(v) / tokens as f64).collect();
    if let Some(i) = mean.iter().position(|v| !v.is_finite()) {
        return Err(ModelError::NumericalInstability(format!(
            "pooled hidden state [{i}] is not finite"
        )));
    }
    let norm = mean.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !(norm.is_finite() && norm > 0.0) {
        return Err(ModelError::NumericalInstability(format!(
            "pooled hidden state has norm {norm}"
        )));
    }
    Ok(mean.iter().map(|v| (v / norm) as f32).collect())
}

/// A GGUF carries its own vocabulary. Refuse a tokenizer.json whose tokens
/// differ from it: every score would be computed on the wrong ids.
fn check_gguf_vocab(path: &Path, tok: &Tokenizer) -> Result<(), ModelError> {
    let ct = QwenModel::read_gguf_content(path)?;
    let Some(tokens) = ct.metadata.get("tokenizer.ggml.tokens") else {
        return Err(ModelError::QwenLoad(format!(
            "{} has no tokenizer.ggml.tokens; cannot verify the tokenizer",
            path.display()
        )));
    };
    let tokens = tokens
        .to_vec()
        .map_err(|e| ModelError::QwenLoad(format!("tokenizer.ggml.tokens: {e}")))?;
    let mut checked = 0usize;
    for (id, t) in tokens.iter().enumerate() {
        let Ok(gguf_tok) = t.to_string() else {
            continue;
        };
        match tok.id_to_token(id as u32) {
            Some(ours) if &ours == gguf_tok => checked += 1,
            // Placeholder rows past the tokenizer's vocabulary are padding.
            None => {}
            Some(ours) => {
                return Err(ModelError::QwenLoad(format!(
                    "tokenizer mismatch at id {id}: tokenizer.json has {ours:?}, \
                     GGUF has {gguf_tok:?}"
                )))
            }
        }
    }
    if checked < tok.get_vocab_size(true) {
        return Err(ModelError::QwenLoad(format!(
            "only {checked} of {} tokenizer ids verified against the GGUF vocabulary",
            tok.get_vocab_size(true)
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn candidate_and_tool_text_match_python() {
        assert_eq!(candidate_text("delete_file"), " delete file");
        assert_eq!(candidate_text("  read "), " read");
        assert_eq!(
            tool_continuation("tool_7", Some("  Reads a\n file  ")),
            " tool 7: Reads a file"
        );
        assert_eq!(tool_continuation("grep", Some("   ")), " grep");
        assert_eq!(tool_continuation("grep", None), " grep");
    }

    #[test]
    fn frame_selection_follows_python_priority() {
        let c = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(select_ask_frame("x", &c(&["Yes", "no"]), &[]).1, "qa");
        assert_eq!(
            select_ask_frame("Question: why", &c(&["a", "b"]), &[]).1,
            "qa"
        );
        assert_eq!(
            select_ask_frame("x", &c(&["a", "b"]), &c(&["a"])).0,
            NEXT_FRAME
        );
        assert_eq!(select_ask_frame("x", &c(&["a", "b"]), &[]).0, ASK_FRAME);
        // A three-name set is not a QA set even if it contains yes/no.
        assert_eq!(
            select_ask_frame("x", &c(&["yes", "no", "maybe"]), &[]).1,
            "default"
        );
    }

    #[test]
    fn prompt_composition_matches_python() {
        assert_eq!(
            compose_prompt("  ctx ", ASK_FRAME, &[]),
            format!("ctx\n{ASK_FRAME}")
        );
        assert_eq!(
            compose_prompt("", NEXT_FRAME, &["a ".into(), "b".into()]),
            format!("Actions already taken: a, b\n{NEXT_FRAME}")
        );
    }

    #[test]
    fn state_text_matches_python_json_dumps() {
        // python3 -c 'import json; print(json.dumps({"b":[1,2.5,None],"a":{"z":"é","y":True}},
        //             ensure_ascii=False, sort_keys=True))'
        let state = json!({"b": [1, 2.5, null], "a": {"z": "é", "y": true}});
        assert_eq!(
            state_text("ctx", Some(&state)),
            "ctx\n{\"a\": {\"y\": true, \"z\": \"é\"}, \"b\": [1, 2.5, null]}"
        );
        assert_eq!(state_text("", Some(&json!("raw"))), "raw");
        assert_eq!(state_text("ctx", None), "ctx");
    }

    #[test]
    fn pooling_mode_names_are_explicit_and_distinct() {
        assert_eq!(PoolingMode::Mean.as_str(), "mean");
        assert_eq!(PoolingMode::LastToken.as_str(), "last_token");
        assert_ne!(PoolingMode::Mean.as_str(), PoolingMode::LastToken.as_str());
        assert_eq!(
            serde_json::to_string(&PoolingMode::Mean).unwrap(),
            "\"mean\""
        );
        assert_eq!(
            serde_json::to_string(&PoolingMode::LastToken).unwrap(),
            "\"last_token\""
        );
    }

    #[test]
    fn softmax_and_entropy() {
        let p = softmax(&[0.0, 0.0]).unwrap();
        assert_eq!(p, vec![0.5, 0.5]);
        assert!((normalized_entropy(&p) - 1.0).abs() < 1e-12);
        let p = softmax(&[1000.0, 0.0]).unwrap();
        assert!(p[0] > 0.999_999 && normalized_entropy(&p) < 1e-6);
        assert!(softmax(&[f64::NAN, 0.0]).is_err());
        assert!(softmax(&[]).is_err());
    }

    #[test]
    fn risk_windows_cover_every_token_without_truncation() {
        let ids: Vec<u32> = (0..500).collect();
        let w = risk_windows(&ids);
        // Python: starts 0, 128, 256, 384 (384 + 192 >= 500 stops).
        assert_eq!(w.len(), 4);
        assert_eq!(w[3].last(), Some(&499));
        assert!(w.iter().all(|x| x.len() <= RISK_WINDOW_TOKENS));
        let short: Vec<u32> = (0..10).collect();
        assert_eq!(risk_windows(&short).len(), 1);
    }

    #[test]
    fn risk_orders_are_permutations_of_the_shots() {
        let shots = RISK_SHOTS_JSONL
            .lines()
            .filter(|l| !l.trim().is_empty())
            .count();
        for order in RISK_ORDERS {
            let mut sorted = order.to_vec();
            sorted.sort();
            assert_eq!(sorted, (0..shots).collect::<Vec<_>>());
        }
    }

    #[test]
    fn candidate_validation() {
        let c = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(validate_candidates(&c(&["a"])).is_err());
        assert!(validate_candidates(&c(&["a", "a"])).is_err());
        assert!(validate_candidates(&c(&["a", " "])).is_err());
        assert!(validate_candidates(&c(&["a", "b"])).is_ok());
    }

    #[test]
    fn mean_pooling_is_unit_norm_and_refuses_degenerate_sums() {
        let v = mean_l2_normalize(vec![3.0, 4.0, 0.0], 2).unwrap();
        assert_eq!(v, vec![0.6, 0.8, 0.0]);
        let n: f32 = v.iter().map(|x| x * x).sum();
        assert!((n - 1.0).abs() < 1e-6);
        assert!(matches!(
            mean_l2_normalize(vec![0.0; 896], 3),
            Err(ModelError::NumericalInstability(_))
        ));
        assert!(matches!(
            mean_l2_normalize(vec![1.0, f32::NAN], 1),
            Err(ModelError::NumericalInstability(_))
        ));
        // f32::MAX squared overflows f32; the f64 sum does not.
        assert!(mean_l2_normalize(vec![f32::MAX, f32::MAX], 1).is_ok());
        assert!(matches!(
            mean_l2_normalize(vec![1.0], 0),
            Err(ModelError::QwenInput(_))
        ));
    }

    #[test]
    fn baseline_cache_evicts_oldest() {
        let mut cache = BaselineCache::default();
        for i in 0..=BASELINE_CACHE_SIZE {
            cache.put((i.to_string(), vec![]), vec![i as f64]);
        }
        assert!(cache.get(&("0".into(), vec![])).is_none());
        assert_eq!(cache.get(&("1".into(), vec![])), Some(vec![1.0]));
    }
}
