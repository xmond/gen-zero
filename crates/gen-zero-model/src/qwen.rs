//! Native Qwen2 causal language model on candle, CPU only. Architecture sizes
//! come from the checkpoint metadata, including Qwen2.5-1.5B's 28 layers,
//! 1536 hidden units, and 8960 intermediate units.
//!
//! One forward implementation serves two weight formats:
//! * a GGUF file (llama.cpp tensor names, any quantization candle reads:
//!   Q8_0, Q4_K_M, ...);
//! * a Hugging Face directory (`config.json` + `model.safetensors`).
//!
//! Both compute in f32: GGUF matrices are dequantized once at load. Measured
//! on a 24-core AVX2 host (2026-10-01) with candle 0.9.2's quantized Q8_0 CPU
//! matmul instead:
//! * speed: a 600-token forward took 40-83 s against 5 s in f32, and 10
//!   tokens took 0.7-1.5 s against 0.4-1.1 s;
//! * accuracy: that kernel also quantizes the activations to 8 bits, which is
//!   discontinuous. A 1e-5 change upstream moved a final hidden state by up to
//!   1.4, and the risk scores of the 91 calibration and held-out rows moved by
//!   up to 0.034 from the Python fp32 reference (bare `rm -rf /` fell below
//!   the escalate threshold). With dequantized weights the same rows move by
//!   at most 0.015 and every dangerous row stays gated.
//!
//! The GGUF file therefore saves disk and download size, not the f32 weight
//! memory used at inference. Its weight quantization error is kept: the
//! dequantized values are exactly the quantized weights.
//!
//! Why not `candle_transformers::models::quantized_qwen2` or `qwen2`: both
//! keep the KV cache inside the model (`&mut self`) and return logits of the
//! last position only, and the quantized one builds a `t x t` mask that is
//! wrong once a cache is present. Candidate scoring needs the log-probability
//! of every token of every candidate after a shared prompt, so this forward
//! is stateless (`&self`), takes the past KV cache as an argument, masks
//! `[T, P + T]`, and returns the hidden state of every position. Layer math
//! is the same as those two modules (pre-norm RMSNorm, q/k/v bias, GQA,
//! non-interleaved RoPE, SwiGLU MLP, tied output head).

use crate::error::ModelError;
use crate::ffn_prune::{FfnPruneReport, StreamingFfnPruner};
use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, Module, Tensor, D};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Longest sequence (past + new tokens) any call may build. Longer input is
/// refused, never truncated. Same bound as the Python runtime this replaces.
pub const MAX_SEQ_LEN: usize = 1024;

/// Which SIMD dot-product kernels candle's quantized matmuls were compiled
/// with. They are chosen by `cfg(target_feature)`, not at run time.
pub const SIMD_KERNELS: &str = if cfg!(target_feature = "avx2") {
    "avx2"
} else if cfg!(target_feature = "neon") {
    "neon"
} else {
    "scalar"
};

/// Positions per chunk when projecting hidden states onto the vocabulary, so
/// a batch of candidates never materializes a `[positions, vocab]` matrix
/// larger than `LOGIT_CHUNK * vocab` floats (about 19 MB for Qwen2.5).
const LOGIT_CHUNK: usize = 32;

pub(crate) fn candle_err(context: &str) -> impl Fn(candle_core::Error) -> ModelError + '_ {
    move |e| ModelError::Inference(format!("{context}: {e}"))
}

/// Architecture numbers of a Qwen2 checkpoint.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct QwenConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,
}

impl QwenConfig {
    fn validate(&self) -> Result<(), ModelError> {
        let ok = self.hidden_size > 0
            && self.num_layers > 0
            && self.num_heads > 0
            && self.num_kv_heads > 0
            && self.num_heads.is_multiple_of(self.num_kv_heads)
            && self.head_dim * self.num_heads == self.hidden_size
            && self.head_dim.is_multiple_of(2)
            && self.vocab_size > 0
            && self.rms_norm_eps > 0.0
            && self.rope_theta > 0.0;
        if ok {
            Ok(())
        } else {
            Err(ModelError::QwenLoad(format!(
                "inconsistent Qwen2 config {self:?}"
            )))
        }
    }
}

/// Which weight format was loaded, for provenance in scorer ids and logs.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightFormat {
    /// GGUF; `dtype` is the ggml type of the first attention matrix (e.g. `q8_0`).
    /// Matrices are dequantized to f32 at load.
    Gguf { dtype: String },
    /// Hugging Face safetensors, computed in f32.
    SafetensorsF32,
}

#[cfg_attr(test, derive(Clone))]
struct Layer {
    attn_norm: Tensor,
    q: QMatMul,
    k: QMatMul,
    v: QMatMul,
    q_bias: Tensor,
    k_bias: Tensor,
    v_bias: Tensor,
    o: QMatMul,
    ffn_norm: Tensor,
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}

/// Keys and values of every layer for a processed prefix, `[B, kv_heads, len, head_dim]`.
#[derive(Clone)]
pub struct KvCache {
    layers: Vec<(Tensor, Tensor)>,
    len: usize,
    batch: usize,
}

impl KvCache {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of sequences in the cache.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Repeat a batch-1 cache `n` times so `n` continuations share one prompt.
    fn expand(&self, n: usize) -> Result<Self, ModelError> {
        if self.batch == n {
            return Ok(self.clone());
        }
        if self.batch != 1 {
            return Err(ModelError::Inference(format!(
                "cannot expand a batch-{} cache to {n}",
                self.batch
            )));
        }
        let err = candle_err("expand kv cache");
        let layers = self
            .layers
            .iter()
            .map(|(k, v)| {
                let (_, h, l, d) = k.dims4().map_err(&err)?;
                Ok((
                    k.broadcast_as((n, h, l, d))
                        .and_then(|t| t.contiguous())
                        .map_err(&err)?,
                    v.broadcast_as((n, h, l, d))
                        .and_then(|t| t.contiguous())
                        .map_err(&err)?,
                ))
            })
            .collect::<Result<Vec<_>, ModelError>>()?;
        Ok(Self {
            layers,
            len: self.len,
            batch: n,
        })
    }
}

/// Attention projections that accept a merged LoRA delta.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoraTarget {
    QProj,
    VProj,
}

/// Inputs every layer of one forward pass shares.
struct Prepared {
    b: usize,
    t: usize,
    p_len: usize,
    past: Option<KvCache>,
    x: Tensor,
    mask: Tensor,
    cos: Tensor,
    sin: Tensor,
}

/// What a [`QwenModel::forward_with_hook`] hook sees after layer `layer`.
pub struct LayerObservation<'a> {
    pub layer: usize,
    pub num_layers: usize,
    /// Residual stream after this layer, before the final norm, `[B, T, H]`.
    pub residual: &'a Tensor,
    /// Per row, `||h_l - h_{l-1}|| / (||h_l|| + eps)` at the last position.
    pub delta: &'a [f64],
}

/// A hook's verdict after one layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HookAction {
    Continue,
    /// Stop: skip every remaining layer. Ignored after the last layer.
    Exit,
}

/// Result of [`QwenModel::forward_with_hook`].
pub struct HookedForward {
    /// Final-norm hidden state of every position, `[B, T, H]`, taken after
    /// the last executed layer.
    pub hidden: Tensor,
    pub layers_executed: usize,
    /// The layer after which the hook stopped the pass, if it did.
    pub exited_at: Option<usize>,
    /// SwiGLU FFN work of the executed layers: channels summed once per
    /// executed layer, MACs (`3 * H * channels` per token row) over all
    /// `B * T` rows. The dense counts use each layer's width at load, so a
    /// pruned model reports `executed < dense` and a dense one `executed == dense`.
    pub ffn: FfnWork,
}

/// FFN channels and multiply-accumulates of one forward pass, pruned against dense.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct FfnWork {
    pub ffn_channels_executed: u64,
    pub ffn_dense_channels: u64,
    pub ffn_macs_executed: u64,
    pub ffn_dense_macs: u64,
}

impl FfnWork {
    fn add_layer(&mut self, hidden: usize, kept: usize, dense: usize, rows: usize) {
        let (h, k, d, r) = (hidden as u64, kept as u64, dense as u64, rows as u64);
        self.ffn_channels_executed += k;
        self.ffn_dense_channels += d;
        self.ffn_macs_executed += 3 * h * k * r;
        self.ffn_dense_macs += 3 * h * d * r;
    }
}

/// Per-channel sum of `|silu(gate(x)) * up(x)|` over the token rows of one layer.
#[derive(Clone, Debug)]
pub(crate) struct FfnActivationSum {
    pub(crate) sums: Vec<f64>,
    pub(crate) rows: usize,
}

const KINETIC_EPS: f64 = 1e-6;

/// Relative step of the last position between two residuals `[B, T, H]`:
/// `||cur - prev|| / (||cur|| + eps)` per row.
fn relative_kinetic(prev: &Tensor, cur: &Tensor) -> Result<Vec<f64>, ModelError> {
    let err = candle_err("relative kinetic");
    let (_, t, _) = cur.dims3().map_err(&err)?;
    let last = |x: &Tensor| x.narrow(1, t - 1, 1).and_then(|y| y.squeeze(1));
    let (p, c) = (last(prev).map_err(&err)?, last(cur).map_err(&err)?);
    let step = (&c - &p)
        .and_then(|d| d.sqr())
        .and_then(|d| d.sum(1))
        .and_then(|d| d.sqrt())
        .and_then(|d| d.to_vec1::<f32>())
        .map_err(&err)?;
    let norm = c
        .sqr()
        .and_then(|d| d.sum(1))
        .and_then(|d| d.sqrt())
        .and_then(|d| d.to_vec1::<f32>())
        .map_err(&err)?;
    step.iter()
        .zip(&norm)
        .map(|(&s, &n)| {
            let v = s as f64 / (n as f64 + KINETIC_EPS);
            if v.is_finite() {
                Ok(v)
            } else {
                Err(ModelError::Inference("non-finite layer step".into()))
            }
        })
        .collect()
}

/// Qwen2 decoder with tied (or separate) output head. Immutable after load,
/// so one instance serves concurrent requests.
#[cfg_attr(test, derive(Clone))]
pub struct QwenModel {
    cfg: QwenConfig,
    format: WeightFormat,
    embed: Tensor,
    layers: Vec<Layer>,
    /// SwiGLU width of every layer at load, before any channel pruning.
    ffn_dense: Vec<usize>,
    norm: Tensor,
    lm_head: QMatMul,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

/// The f32 matrix behind a projection; FFN slicing needs the dense values.
pub(crate) fn dense_matrix(m: &QMatMul) -> Result<&Tensor, ModelError> {
    match m {
        QMatMul::Tensor(t) => Ok(t),
        QMatMul::TensorF16(_) | QMatMul::QTensor(_) => Err(ModelError::Inference(
            "projection is not an f32 matrix; FFN pruning needs dequantized weights".into(),
        )),
    }
}

/// FFN projections of one freshly loaded layer. Records the dense width,
/// then slices the matrices at once when `pruner` covers the layer, so the
/// dense originals are dropped before the next layer loads.
fn load_ffn(
    pruner: &mut Option<StreamingFfnPruner>,
    ffn_dense: &mut Vec<usize>,
    layer: usize,
    (gate, up, down): (Tensor, Tensor, Tensor),
) -> Result<(QMatMul, QMatMul, QMatMul), ModelError> {
    ffn_dense.push(gate.dims2().map_err(candle_err("ffn width"))?.0);
    let sliced = match pruner {
        Some(p) => p.prune_layer(layer, &gate, &up, &down)?,
        None => None,
    };
    let (gate, up, down) = sliced.unwrap_or((gate, up, down));
    Ok((
        QMatMul::Tensor(gate),
        QMatMul::Tensor(up),
        QMatMul::Tensor(down),
    ))
}

fn rope_tables(cfg: &QwenConfig, device: &Device) -> Result<(Tensor, Tensor), ModelError> {
    let err = candle_err("rope tables");
    // f32 inverse frequencies and angles, as the reference implementations do.
    let inv: Vec<f32> = (0..cfg.head_dim)
        .step_by(2)
        .map(|i| 1f32 / (cfg.rope_theta as f32).powf(i as f32 / cfg.head_dim as f32))
        .collect();
    let inv = Tensor::from_vec(inv, (1, cfg.head_dim / 2), device).map_err(&err)?;
    let pos = Tensor::arange(0u32, MAX_SEQ_LEN as u32, device)
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.reshape((MAX_SEQ_LEN, 1)))
        .map_err(&err)?;
    let angles = pos.matmul(&inv).map_err(&err)?;
    Ok((angles.cos().map_err(&err)?, angles.sin().map_err(&err)?))
}

impl QwenModel {
    /// Resolve a weights file in a model directory, refusing ambiguous GGUF sets.
    pub(crate) fn weights_path(path: &Path) -> Result<std::path::PathBuf, ModelError> {
        if !path.is_dir() {
            return Ok(path.to_path_buf());
        }
        let safetensors = path.join("model.safetensors");
        if safetensors.is_file() {
            return Ok(safetensors);
        }
        let mut ggufs = Vec::new();
        for entry in std::fs::read_dir(path)
            .map_err(|e| ModelError::QwenLoad(format!("read directory {}: {e}", path.display())))?
        {
            let entry = entry.map_err(|e| {
                ModelError::QwenLoad(format!("read directory {}: {e}", path.display()))
            })?;
            let file = entry.path();
            if file.is_file()
                && file
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
            {
                ggufs.push(file);
            }
        }
        if ggufs.len() != 1 {
            return Err(ModelError::QwenLoad(format!(
                "{} has no model.safetensors and {} .gguf files; expected exactly one",
                path.display(),
                ggufs.len()
            )));
        }
        Ok(ggufs.remove(0))
    }

    /// Load a `.gguf` file, or a directory holding `model.safetensors`
    /// or exactly one `.gguf` file.
    pub fn load(path: &Path) -> Result<Self, ModelError> {
        Self::load_streaming(path, None).map(|(model, _)| model)
    }

    /// [`Self::load`] with the upper FFN layers ([`crate::default_upper_ffn_layers`])
    /// pruned by weight norm to `keep_ratio` while they load: each layer is
    /// sliced right after its weights are read, before the next layer loads,
    /// so the dense upper FFNs are never all resident. The weights and the
    /// report are bit-exact with `load` followed by
    /// [`Self::prune_upper_ffn_by_weight_norm`].
    pub fn load_with_weight_norm_pruning(
        path: &Path,
        keep_ratio: f64,
    ) -> Result<(Self, FfnPruneReport), ModelError> {
        let (model, report) = Self::load_streaming(path, Some(keep_ratio))?;
        let report = report.ok_or_else(|| {
            ModelError::QwenLoad("streaming FFN pruning produced no report".into())
        })?;
        Ok((model, report))
    }

    fn load_streaming(
        path: &Path,
        keep_ratio: Option<f64>,
    ) -> Result<(Self, Option<FfnPruneReport>), ModelError> {
        let weights = Self::weights_path(path)?;
        if weights
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            Self::gguf_streaming(&weights, keep_ratio)
        } else if path.is_dir() {
            Self::safetensors_streaming(path, keep_ratio)
        } else {
            Err(ModelError::QwenLoad(format!(
                "{} is neither a .gguf file nor a directory with model.safetensors",
                path.display()
            )))
        }
    }

    pub fn config(&self) -> &QwenConfig {
        &self.cfg
    }

    pub fn format(&self) -> &WeightFormat {
        &self.format
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Read the GGUF metadata and tensor index only (no weights).
    pub fn read_gguf_content(path: &Path) -> Result<gguf_file::Content, ModelError> {
        let mut file = File::open(path)
            .map_err(|e| ModelError::QwenLoad(format!("open {}: {e}", path.display())))?;
        gguf_file::Content::read(&mut file)
            .map_err(|e| ModelError::QwenLoad(format!("read GGUF {}: {e}", path.display())))
    }

    pub fn from_gguf(path: &Path) -> Result<Self, ModelError> {
        Self::gguf_streaming(path, None).map(|(model, _)| model)
    }

    fn gguf_streaming(
        path: &Path,
        keep_ratio: Option<f64>,
    ) -> Result<(Self, Option<FfnPruneReport>), ModelError> {
        let device = Device::Cpu;
        let file = File::open(path)
            .map_err(|e| ModelError::QwenLoad(format!("open {}: {e}", path.display())))?;
        let mut reader = BufReader::new(file);
        let ct = gguf_file::Content::read(&mut reader)
            .map_err(|e| ModelError::QwenLoad(format!("read GGUF {}: {e}", path.display())))?;
        let arch = ct
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_default();
        if arch != "qwen2" {
            return Err(ModelError::QwenLoad(format!(
                "{} is a GGUF of architecture {arch:?}, expected \"qwen2\"",
                path.display()
            )));
        }
        let md_u32 = |key: &str| -> Result<usize, ModelError> {
            ct.metadata
                .get(key)
                .ok_or_else(|| ModelError::QwenLoad(format!("GGUF metadata {key} missing")))?
                .to_u32()
                .map(|v| v as usize)
                .map_err(|e| ModelError::QwenLoad(format!("GGUF metadata {key}: {e}")))
        };
        let md_f32 = |key: &str| -> Result<f64, ModelError> {
            ct.metadata
                .get(key)
                .ok_or_else(|| ModelError::QwenLoad(format!("GGUF metadata {key} missing")))?
                .to_f32()
                .map(|v| v as f64)
                .map_err(|e| ModelError::QwenLoad(format!("GGUF metadata {key}: {e}")))
        };
        let hidden_size = md_u32("qwen2.embedding_length")?;
        let num_heads = md_u32("qwen2.attention.head_count")?;
        let embd_info = ct
            .tensor_infos
            .get("token_embd.weight")
            .ok_or_else(|| ModelError::QwenLoad("GGUF tensor token_embd.weight missing".into()))?;
        let cfg = QwenConfig {
            hidden_size,
            num_layers: md_u32("qwen2.block_count")?,
            num_heads,
            num_kv_heads: md_u32("qwen2.attention.head_count_kv")?,
            head_dim: hidden_size / num_heads.max(1),
            rms_norm_eps: md_f32("qwen2.attention.layer_norm_rms_epsilon")?,
            // A missing rope base is an error: the Qwen2.5 value (1e6) differs
            // from the llama default, and guessing would shift every score.
            rope_theta: md_f32("qwen2.rope.freq_base")?,
            vocab_size: embd_info.shape.dims()[0],
        };
        cfg.validate()?;
        let mut pruner = keep_ratio
            .map(|r| StreamingFfnPruner::new(cfg.num_layers, cfg.hidden_size, r))
            .transpose()?;
        let dtype = ct
            .tensor_infos
            .get("blk.0.attn_q.weight")
            .map(|i| format!("{:?}", i.ggml_dtype).to_ascii_lowercase())
            .unwrap_or_else(|| "unknown".into());

        let mut qt = |name: &str| {
            ct.tensor(&mut reader, name, &device)
                .map_err(|e| ModelError::QwenLoad(format!("GGUF tensor {name}: {e}")))
        };
        let dense = |q: candle_core::quantized::QTensor, name: &str| {
            q.dequantize(&device)
                .map_err(|e| ModelError::QwenLoad(format!("dequantize {name}: {e}")))
        };
        let qmm =
            |q: candle_core::quantized::QTensor, name: &str| dense(q, name).map(QMatMul::Tensor);

        let embed_q = qt("token_embd.weight")?;
        let embed = dense(embed_q, "token_embd.weight")?;
        let norm = dense(qt("output_norm.weight")?, "output_norm.weight")?;
        let lm_head = if ct.tensor_infos.contains_key("output.weight") {
            qmm(qt("output.weight")?, "output.weight")?
        } else {
            // Tied head: the (dequantized) embedding matrix is the output projection.
            QMatMul::Tensor(embed.clone())
        };
        let mut layers = Vec::with_capacity(cfg.num_layers);
        let mut ffn_dense = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let p = format!("blk.{i}");
            let mut take = |s: &str| qt(&format!("{p}.{s}"));
            let attn_norm = take("attn_norm.weight")?;
            let q = take("attn_q.weight")?;
            let k = take("attn_k.weight")?;
            let v = take("attn_v.weight")?;
            let q_bias = take("attn_q.bias")?;
            let k_bias = take("attn_k.bias")?;
            let v_bias = take("attn_v.bias")?;
            let o = take("attn_output.weight")?;
            let ffn_norm = take("ffn_norm.weight")?;
            let gate = take("ffn_gate.weight")?;
            let up = take("ffn_up.weight")?;
            let down = take("ffn_down.weight")?;
            let (gate, up, down) = load_ffn(
                &mut pruner,
                &mut ffn_dense,
                i,
                (dense(gate, &p)?, dense(up, &p)?, dense(down, &p)?),
            )?;
            layers.push(Layer {
                attn_norm: dense(attn_norm, &p)?,
                q: qmm(q, &p)?,
                k: qmm(k, &p)?,
                v: qmm(v, &p)?,
                q_bias: dense(q_bias, &p)?,
                k_bias: dense(k_bias, &p)?,
                v_bias: dense(v_bias, &p)?,
                o: qmm(o, &p)?,
                ffn_norm: dense(ffn_norm, &p)?,
                gate,
                up,
                down,
            });
        }
        let (cos, sin) = rope_tables(&cfg, &device)?;
        let model = Self {
            cfg,
            format: WeightFormat::Gguf { dtype },
            embed,
            ffn_dense,
            layers,
            norm,
            lm_head,
            cos,
            sin,
            device,
        };
        Ok((model, pruner.map(StreamingFfnPruner::finish).transpose()?))
    }

    pub fn from_safetensors_dir(dir: &Path) -> Result<Self, ModelError> {
        Self::safetensors_streaming(dir, None).map(|(model, _)| model)
    }

    fn safetensors_streaming(
        dir: &Path,
        keep_ratio: Option<f64>,
    ) -> Result<(Self, Option<FfnPruneReport>), ModelError> {
        let device = Device::Cpu;
        let config_path = dir.join("config.json");
        let weights = dir.join("model.safetensors");
        let raw = std::fs::read_to_string(&config_path)
            .map_err(|e| ModelError::QwenLoad(format!("read {}: {e}", config_path.display())))?;
        let json: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| ModelError::QwenLoad(format!("parse {}: {e}", config_path.display())))?;
        if json["model_type"] != "qwen2" {
            return Err(ModelError::QwenLoad(format!(
                "{} has model_type {}, expected \"qwen2\"",
                config_path.display(),
                json["model_type"]
            )));
        }
        let num = |key: &str| -> Result<f64, ModelError> {
            json[key].as_f64().ok_or_else(|| {
                ModelError::QwenLoad(format!("{} lacks numeric {key}", config_path.display()))
            })
        };
        let hidden_size = num("hidden_size")? as usize;
        let num_heads = num("num_attention_heads")? as usize;
        let cfg = QwenConfig {
            hidden_size,
            num_layers: num("num_hidden_layers")? as usize,
            num_heads,
            num_kv_heads: num("num_key_value_heads")? as usize,
            head_dim: hidden_size / num_heads.max(1),
            rms_norm_eps: num("rms_norm_eps")?,
            rope_theta: num("rope_theta")?,
            vocab_size: num("vocab_size")? as usize,
        };
        cfg.validate()?;
        let mut pruner = keep_ratio
            .map(|r| StreamingFfnPruner::new(cfg.num_layers, cfg.hidden_size, r))
            .transpose()?;
        // SAFETY: the file is memory-mapped read-only for the duration of the load;
        // every tensor is copied (converted to f32) before the map is dropped.
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&weights),
                DType::F32,
                &device,
            )
        }
        .map_err(|e| ModelError::QwenLoad(format!("map {}: {e}", weights.display())))?;
        let get = |shape: &[usize], name: &str| {
            vb.get(shape, name)
                .map_err(|e| ModelError::QwenLoad(format!("safetensors {name}: {e}")))
        };
        let (h, kv, im) = (
            cfg.hidden_size,
            cfg.num_kv_heads * cfg.head_dim,
            num("intermediate_size")? as usize,
        );
        let embed = get(&[cfg.vocab_size, h], "model.embed_tokens.weight")?;
        let norm = get(&[h], "model.norm.weight")?;
        let tied = json["tie_word_embeddings"].as_bool().unwrap_or(false);
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            QMatMul::Tensor(get(&[cfg.vocab_size, h], "lm_head.weight")?)
        } else if tied {
            QMatMul::Tensor(embed.clone())
        } else {
            return Err(ModelError::QwenLoad(
                "no lm_head.weight and tie_word_embeddings is false".into(),
            ));
        };
        let mut layers = Vec::with_capacity(cfg.num_layers);
        let mut ffn_dense = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let p = format!("model.layers.{i}");
            let raw = |shape: &[usize], s: &str| get(shape, &format!("{p}.{s}"));
            let mat = |shape: &[usize], s: &str| raw(shape, s).map(QMatMul::Tensor);
            let vec = |n: usize, s: &str| get(&[n], &format!("{p}.{s}"));
            let (gate, up, down) = load_ffn(
                &mut pruner,
                &mut ffn_dense,
                i,
                (
                    raw(&[im, h], "mlp.gate_proj.weight")?,
                    raw(&[im, h], "mlp.up_proj.weight")?,
                    raw(&[h, im], "mlp.down_proj.weight")?,
                ),
            )?;
            layers.push(Layer {
                attn_norm: vec(h, "input_layernorm.weight")?,
                q: mat(&[h, h], "self_attn.q_proj.weight")?,
                k: mat(&[kv, h], "self_attn.k_proj.weight")?,
                v: mat(&[kv, h], "self_attn.v_proj.weight")?,
                q_bias: vec(h, "self_attn.q_proj.bias")?,
                k_bias: vec(kv, "self_attn.k_proj.bias")?,
                v_bias: vec(kv, "self_attn.v_proj.bias")?,
                o: mat(&[h, h], "self_attn.o_proj.weight")?,
                ffn_norm: vec(h, "post_attention_layernorm.weight")?,
                gate,
                up,
                down,
            });
        }
        let (cos, sin) = rope_tables(&cfg, &device)?;
        let model = Self {
            cfg,
            format: WeightFormat::SafetensorsF32,
            embed,
            ffn_dense,
            layers,
            norm,
            lm_head,
            cos,
            sin,
            device,
        };
        Ok((model, pruner.map(StreamingFfnPruner::finish).transpose()?))
    }

    /// Fold a LoRA update into one attention projection, `W += delta`, where
    /// `delta = (alpha / rank) * B @ A` has the weight's `[out, in]` shape.
    /// Exact for inference (LoRA without dropout is linear). Only full
    /// precision safetensors weights can be patched; quantized GGUF weights
    /// are refused rather than dequantized behind the caller's back.
    pub fn merge_linear_delta(
        &mut self,
        layer: usize,
        target: LoraTarget,
        delta: &Tensor,
    ) -> Result<(), ModelError> {
        let n = self.layers.len();
        let l = self.layers.get_mut(layer).ok_or_else(|| {
            ModelError::QwenLoad(format!("LoRA delta for layer {layer}, model has {n}"))
        })?;
        let slot = match target {
            LoraTarget::QProj => &mut l.q,
            LoraTarget::VProj => &mut l.v,
        };
        let QMatMul::Tensor(w) = slot else {
            return Err(ModelError::QwenLoad(format!(
                "layer {layer} {target:?} is quantized; LoRA merge needs f32 safetensors weights"
            )));
        };
        if w.dims() != delta.dims() {
            return Err(ModelError::QwenLoad(format!(
                "layer {layer} {target:?}: weight {:?} vs LoRA delta {:?}",
                w.dims(),
                delta.dims()
            )));
        }
        let merged = w
            .add(
                &delta
                    .to_dtype(w.dtype())
                    .map_err(candle_err("lora delta dtype"))?,
            )
            .map_err(candle_err("lora merge"))?;
        *slot = QMatMul::Tensor(merged);
        Ok(())
    }

    /// Causal mask for `t` new tokens after `past` cached ones: 0 where a
    /// query may attend, -inf where the key lies in its future.
    fn mask(&self, t: usize, past: usize) -> Result<Tensor, ModelError> {
        let data: Vec<f32> = (0..t)
            .flat_map(|i| {
                (0..past + t).map(move |j| if j > past + i { f32::NEG_INFINITY } else { 0.0 })
            })
            .collect();
        Tensor::from_vec(data, (t, past + t), &self.device).map_err(candle_err("mask"))
    }

    /// Validate `rows`, embed them, and build the mask and RoPE slices shared
    /// by every layer.
    fn prepare(&self, rows: &[Vec<u32>], past: Option<&KvCache>) -> Result<Prepared, ModelError> {
        let b = rows.len();
        let t = rows.first().map_or(0, Vec::len);
        if b == 0 || t == 0 || rows.iter().any(|r| r.len() != t) {
            return Err(ModelError::QwenInput(
                "forward needs a non-empty rectangular token batch".into(),
            ));
        }
        let past = past.map(|p| p.expand(b)).transpose()?;
        let p_len = past.as_ref().map_or(0, KvCache::len);
        if p_len + t > MAX_SEQ_LEN {
            return Err(ModelError::QwenInput(format!(
                "{} tokens exceed the {MAX_SEQ_LEN}-token limit; refusing to truncate",
                p_len + t
            )));
        }
        if let Some(bad) = rows
            .iter()
            .flatten()
            .find(|&&id| id as usize >= self.cfg.vocab_size)
        {
            return Err(ModelError::QwenInput(format!(
                "token id {bad} outside vocabulary of {}",
                self.cfg.vocab_size
            )));
        }
        let err = candle_err("qwen forward");
        let ids: Vec<u32> = rows.iter().flatten().copied().collect();
        let ids = Tensor::from_vec(ids, (b, t), &self.device).map_err(&err)?;
        let x = self
            .embed
            .index_select(&ids.flatten_all().map_err(&err)?, 0)
            .and_then(|e| e.reshape((b, t, self.cfg.hidden_size)))
            .map_err(&err)?;
        Ok(Prepared {
            b,
            t,
            p_len,
            past,
            x,
            mask: self.mask(t, p_len)?,
            cos: self.cos.narrow(0, p_len, t).map_err(&err)?,
            sin: self.sin.narrow(0, p_len, t).map_err(&err)?,
        })
    }

    /// One decoder layer on the residual stream `x`. Returns the new residual
    /// and this layer's keys and values (past included).
    fn layer_step(
        &self,
        li: usize,
        x: Tensor,
        p: &Prepared,
        ffn_capture: Option<&mut FfnActivationSum>,
    ) -> Result<(Tensor, (Tensor, Tensor)), ModelError> {
        let err = candle_err("qwen forward");
        let cfg = &self.cfg;
        let layer = &self.layers[li];
        let (b, t) = (p.b, p.t);
        let scale = 1.0 / (cfg.head_dim as f64).sqrt();
        let eps = cfg.rms_norm_eps as f32;
        let h = candle_nn::ops::rms_norm(&x, &layer.attn_norm, eps).map_err(&err)?;
        let split = |w: &QMatMul, bias: &Tensor, heads: usize| {
            w.forward(&h)
                .and_then(|y| y.broadcast_add(bias))
                .and_then(|y| y.reshape((b, t, heads, cfg.head_dim)))
                .and_then(|y| y.transpose(1, 2))
                .and_then(|y| y.contiguous())
        };
        let q = split(&layer.q, &layer.q_bias, cfg.num_heads).map_err(&err)?;
        let k = split(&layer.k, &layer.k_bias, cfg.num_kv_heads).map_err(&err)?;
        let v = split(&layer.v, &layer.v_bias, cfg.num_kv_heads).map_err(&err)?;
        let q = candle_nn::rotary_emb::rope(&q, &p.cos, &p.sin).map_err(&err)?;
        let k = candle_nn::rotary_emb::rope(&k, &p.cos, &p.sin).map_err(&err)?;
        let (k, v) = match &p.past {
            Some(past) => {
                let (pk, pv) = &past.layers[li];
                (
                    Tensor::cat(&[pk, &k], 2).map_err(&err)?,
                    Tensor::cat(&[pv, &v], 2).map_err(&err)?,
                )
            }
            None => (k, v),
        };
        let kv = (k.clone(), v.clone());
        let rep = cfg.num_heads / cfg.num_kv_heads;
        let k = candle_transformers::utils::repeat_kv(k, rep).map_err(&err)?;
        let v = candle_transformers::utils::repeat_kv(v, rep)
            .and_then(|v| v.contiguous())
            .map_err(&err)?;
        let att = q
            .matmul(&k.t().map_err(&err)?)
            .and_then(|a| a * scale)
            .and_then(|a| a.broadcast_add(&p.mask))
            .and_then(|a| candle_nn::ops::softmax_last_dim(&a))
            .map_err(&err)?;
        let y = att
            .matmul(&v)
            .and_then(|y| y.transpose(1, 2))
            .and_then(|y| y.reshape((b, t, cfg.hidden_size)))
            .and_then(|y| layer.o.forward(&y))
            .map_err(&err)?;
        let x = (x + y).map_err(&err)?;
        let h = candle_nn::ops::rms_norm(&x, &layer.ffn_norm, eps).map_err(&err)?;
        let gate = layer
            .gate
            .forward(&h)
            .and_then(|g| candle_nn::ops::silu(&g))
            .map_err(&err)?;
        let up = layer.up.forward(&h).map_err(&err)?;
        let activated = (gate * up).map_err(&err)?;
        if let Some(acc) = ffn_capture {
            let sums = activated
                .abs()
                .and_then(|a| a.sum(0))
                .and_then(|a| a.sum(0))
                .and_then(|a| a.to_vec1::<f32>())
                .map_err(&err)?;
            if acc.sums.len() != sums.len() {
                return Err(ModelError::Inference(format!(
                    "layer {li} FFN width {} vs accumulator {}",
                    sums.len(),
                    acc.sums.len()
                )));
            }
            for (s, v) in acc.sums.iter_mut().zip(&sums) {
                *s += *v as f64;
            }
            acc.rows += b * t;
        }
        let y = layer.down.forward(&activated).map_err(&err)?;
        Ok(((x + y).map_err(&err)?, kv))
    }

    fn final_norm(&self, x: &Tensor) -> Result<Tensor, ModelError> {
        candle_nn::ops::rms_norm(x, &self.norm, self.cfg.rms_norm_eps as f32)
            .map_err(candle_err("qwen forward"))
    }

    /// Run `rows` (a rectangular `[B, T]` batch of token ids) after `past`.
    ///
    /// Returns the final-norm hidden state of every position, `[B, T, H]`,
    /// and the cache extended by these tokens. A batch-1 `past` is shared by
    /// all `B` rows. Right padding needs no extra mask: under the causal mask
    /// a real position never attends to the padding after it.
    pub fn forward(
        &self,
        rows: &[Vec<u32>],
        past: Option<&KvCache>,
    ) -> Result<(Tensor, KvCache), ModelError> {
        let p = self.prepare(rows, past)?;
        let mut x = p.x.clone();
        let mut cache = Vec::with_capacity(self.layers.len());
        for li in 0..self.layers.len() {
            let (next, kv) = self.layer_step(li, x, &p, None)?;
            x = next;
            cache.push(kv);
        }
        Ok((
            self.final_norm(&x)?,
            KvCache {
                layers: cache,
                len: p.p_len + p.t,
                batch: p.b,
            },
        ))
    }

    /// Like [`forward`](Self::forward) without a past cache, but `hook` sees
    /// every layer's output and may stop the pass. On
    /// [`HookAction::Exit`] the remaining layers are not computed at all.
    ///
    /// No [`KvCache`] is returned: after an exit the deeper layers have no
    /// keys or values, and a partial cache must never be reused.
    pub fn forward_with_hook(
        &self,
        rows: &[Vec<u32>],
        hook: &mut dyn FnMut(&LayerObservation<'_>) -> Result<HookAction, ModelError>,
    ) -> Result<HookedForward, ModelError> {
        let p = self.prepare(rows, None)?;
        let mut x = p.x.clone();
        let total = self.layers.len();
        let mut ffn = FfnWork::default();
        for li in 0..total {
            let (next, _) = self.layer_step(li, x.clone(), &p, None)?;
            ffn.add_layer(
                self.cfg.hidden_size,
                self.ffn_width(li)?,
                self.ffn_dense[li],
                p.b * p.t,
            );
            let delta = relative_kinetic(&x, &next)?;
            x = next;
            let obs = LayerObservation {
                layer: li,
                num_layers: total,
                residual: &x,
                delta: &delta,
            };
            if hook(&obs)? == HookAction::Exit && li + 1 < total {
                return Ok(HookedForward {
                    hidden: self.final_norm(&x)?,
                    layers_executed: li + 1,
                    exited_at: Some(li),
                    ffn,
                });
            }
        }
        Ok(HookedForward {
            hidden: self.final_norm(&x)?,
            layers_executed: total,
            exited_at: None,
            ffn,
        })
    }

    /// Full forward passes over `anchors` (each one `[T]` row, no past) that
    /// sum `|silu(gate(x)) * up(x)|` per channel at each layer in `layers`.
    pub(crate) fn ffn_activation_sums(
        &self,
        anchors: &[Vec<u32>],
        layers: &std::collections::BTreeSet<usize>,
    ) -> Result<std::collections::BTreeMap<usize, FfnActivationSum>, ModelError> {
        let mut acc = layers
            .iter()
            .map(|&li| {
                Ok((
                    li,
                    FfnActivationSum {
                        sums: vec![0.0; self.ffn_width(li)?],
                        rows: 0,
                    },
                ))
            })
            .collect::<Result<std::collections::BTreeMap<_, _>, ModelError>>()?;
        for ids in anchors {
            let p = self.prepare(std::slice::from_ref(ids), None)?;
            let mut x = p.x.clone();
            for li in 0..self.layers.len() {
                x = self.layer_step(li, x, &p, acc.get_mut(&li))?.0;
            }
        }
        Ok(acc)
    }

    /// Current SwiGLU width (kept channels) of `layer`.
    pub fn ffn_width(&self, layer: usize) -> Result<usize, ModelError> {
        let l = self.layers.get(layer).ok_or_else(|| {
            ModelError::QwenInput(format!(
                "FFN layer {layer} outside {} layers",
                self.layers.len()
            ))
        })?;
        dense_matrix(&l.gate)?
            .dims2()
            .map(|(i, _)| i)
            .map_err(candle_err("ffn width"))
    }

    /// SwiGLU width of `layer` at load, before any pruning.
    pub fn ffn_dense_width(&self, layer: usize) -> Result<usize, ModelError> {
        self.ffn_dense.get(layer).copied().ok_or_else(|| {
            ModelError::QwenInput(format!(
                "FFN layer {layer} outside {} layers",
                self.layers.len()
            ))
        })
    }

    /// `(gate [I, H], up [I, H], down [H, I])` of `layer`.
    pub(crate) fn ffn_matrices(
        &self,
        layer: usize,
    ) -> Result<(&Tensor, &Tensor, &Tensor), ModelError> {
        self.ffn_width(layer)?;
        let l = &self.layers[layer];
        Ok((
            dense_matrix(&l.gate)?,
            dense_matrix(&l.up)?,
            dense_matrix(&l.down)?,
        ))
    }

    /// Swap in sliced FFN matrices for `layer`. Shapes must agree with each
    /// other and with the hidden size; the caller slices.
    pub(crate) fn replace_ffn(
        &mut self,
        layer: usize,
        gate: Tensor,
        up: Tensor,
        down: Tensor,
    ) -> Result<(), ModelError> {
        let h = self.cfg.hidden_size;
        let err = candle_err("replace ffn");
        let (gk, gh) = gate.dims2().map_err(&err)?;
        let (uk, uh) = up.dims2().map_err(&err)?;
        let (dh, dk) = down.dims2().map_err(&err)?;
        if gh != h || uh != h || dh != h || gk != uk || gk != dk || gk == 0 {
            return Err(ModelError::Inference(format!(
                "layer {layer}: sliced FFN shapes gate [{gk}, {gh}] up [{uk}, {uh}] \
                 down [{dh}, {dk}] do not fit hidden size {h}"
            )));
        }
        let l = self
            .layers
            .get_mut(layer)
            .ok_or_else(|| ModelError::QwenInput(format!("FFN layer {layer} out of range")))?;
        l.gate = QMatMul::Tensor(gate);
        l.up = QMatMul::Tensor(up);
        l.down = QMatMul::Tensor(down);
        Ok(())
    }

    /// Final RMSNorm of the last position of a residual `[1, T, H]`, as `[H]` f32.
    pub(crate) fn final_norm_last(&self, residual: &Tensor) -> Result<Vec<f32>, ModelError> {
        let err = candle_err("final norm last");
        let (_, t, _) = residual.dims3().map_err(&err)?;
        let last = residual.narrow(1, t - 1, 1).map_err(&err)?;
        self.final_norm(&last)?
            .flatten_all()
            .and_then(|v| v.to_vec1::<f32>())
            .map_err(&err)
    }

    /// Rows of the output head for `ids`, `[ids.len(), H]` as f32 rows.
    pub(crate) fn lm_head_rows(&self, ids: &[u32]) -> Result<Vec<Vec<f32>>, ModelError> {
        let err = candle_err("lm head rows");
        if let Some(bad) = ids.iter().find(|&&id| id as usize >= self.cfg.vocab_size) {
            return Err(ModelError::QwenInput(format!(
                "label token id {bad} outside vocabulary of {}",
                self.cfg.vocab_size
            )));
        }
        let idx = Tensor::from_vec(ids.to_vec(), ids.len(), &self.device).map_err(&err)?;
        dense_matrix(&self.lm_head)?
            .index_select(&idx, 0)
            .and_then(|r| r.to_vec2::<f32>())
            .map_err(&err)
    }

    /// For each row of `hidden` (`[N, H]`), the log-probabilities of the token
    /// ids in `targets[row]`. The vocabulary projection and log-softmax run in
    /// chunks of [`LOGIT_CHUNK`] rows; only the requested entries are kept.
    pub fn log_probs_of(
        &self,
        hidden: &Tensor,
        targets: &[Vec<u32>],
    ) -> Result<Vec<Vec<f64>>, ModelError> {
        let err = candle_err("log-probs");
        let (n, _) = hidden.dims2().map_err(&err)?;
        if n != targets.len() {
            return Err(ModelError::Inference(format!(
                "{n} hidden rows for {} target lists",
                targets.len()
            )));
        }
        let mut out = Vec::with_capacity(n);
        for start in (0..n).step_by(LOGIT_CHUNK) {
            let len = LOGIT_CHUNK.min(n - start);
            let lp = hidden
                .narrow(0, start, len)
                .and_then(|h| self.lm_head.forward(&h))
                .and_then(|l| candle_nn::ops::log_softmax(&l, D::Minus1))
                .and_then(|l| l.to_vec2::<f32>())
                .map_err(&err)?;
            for (row, want) in lp.iter().zip(&targets[start..start + len]) {
                out.push(
                    want.iter()
                        .map(|&id| {
                            row.get(id as usize).map(|&v| v as f64).ok_or_else(|| {
                                ModelError::Inference(format!("token id {id} outside logits"))
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
        }
        Ok(out)
    }
}

/// SHA-256 of a weight file (hex), streamed so a 1 GB file is not held in memory.
pub fn sha256_file(path: &Path) -> Result<String, ModelError> {
    use sha2::{Digest, Sha256};
    let mut file = File::open(path)
        .map_err(|e| ModelError::QwenLoad(format!("open {}: {e}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| ModelError::QwenLoad(format!("read {}: {e}", path.display())))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn directory_weights_resolve_only_unambiguous_gguf() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();
        assert!(matches!(
            QwenModel::weights_path(path),
            Err(ModelError::QwenLoad(_))
        ));
        let first = path.join("qwen.Q4_K_M.GGUF");
        std::fs::write(&first, b"invalid gguf").unwrap();
        assert_eq!(QwenModel::weights_path(path).unwrap(), first);
        let err = QwenModel::load(path).err().unwrap();
        assert!(matches!(err, ModelError::QwenLoad(_)), "{err}");
        assert!(format!("{err}").contains("GGUF"), "{err}");
        std::fs::write(path.join("other.gguf"), b"invalid gguf").unwrap();
        assert!(matches!(
            QwenModel::weights_path(path),
            Err(ModelError::QwenLoad(_))
        ));
        let safetensors = path.join("model.safetensors");
        std::fs::write(&safetensors, b"invalid safetensors").unwrap();
        assert_eq!(QwenModel::weights_path(path).unwrap(), safetensors);
    }

    #[test]
    fn load_refuses_unknown_paths() {
        let err = QwenModel::load(Path::new("/nonexistent/model.bin"))
            .err()
            .unwrap();
        assert!(matches!(err, ModelError::QwenLoad(_)), "{err}");
    }

    #[test]
    fn config_validation_rejects_bad_head_split() {
        let cfg = QwenConfig {
            hidden_size: 896,
            num_layers: 24,
            num_heads: 14,
            num_kv_heads: 3,
            head_dim: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 1e6,
            vocab_size: 151_936,
        };
        assert!(cfg.validate().is_err());
    }

    /// A random-weight Qwen2 small enough for unit tests. The layer math is
    /// the production path; only the weights are synthetic.
    pub(crate) fn tiny_model(num_layers: usize) -> QwenModel {
        let device = Device::Cpu;
        let cfg = QwenConfig {
            hidden_size: 16,
            num_layers,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 1e4,
            vocab_size: 32,
        };
        cfg.validate().unwrap();
        let (h, kv, im) = (16, 8, 32);
        let r = |shape: &[usize]| Tensor::randn(0f32, 0.3, shape, &device).unwrap();
        let m = |shape: &[usize]| QMatMul::Tensor(r(shape));
        let ones = |n: usize| Tensor::ones(n, DType::F32, &device).unwrap();
        let layers: Vec<Layer> = (0..num_layers)
            .map(|_| Layer {
                attn_norm: ones(h),
                q: m(&[h, h]),
                k: m(&[kv, h]),
                v: m(&[kv, h]),
                q_bias: r(&[h]),
                k_bias: r(&[kv]),
                v_bias: r(&[kv]),
                o: m(&[h, h]),
                ffn_norm: ones(h),
                gate: m(&[im, h]),
                up: m(&[im, h]),
                down: m(&[h, im]),
            })
            .collect();
        let embed = r(&[cfg.vocab_size, h]);
        let (cos, sin) = rope_tables(&cfg, &device).unwrap();
        QwenModel {
            lm_head: QMatMul::Tensor(embed.clone()),
            cfg,
            format: WeightFormat::SafetensorsF32,
            embed,
            ffn_dense: vec![im; num_layers],
            layers,
            norm: ones(h),
            cos,
            sin,
            device,
        }
    }

    /// Write `model` (dense f32, tied head) to disk twice: a Hugging Face
    /// directory (`config.json` + `model.safetensors`) and an f32 GGUF file,
    /// so tests can drive the real loaders. Returns `(hf_dir, gguf_file)`.
    pub(crate) fn write_tiny_checkpoints(
        model: &QwenModel,
        root: &Path,
    ) -> (std::path::PathBuf, std::path::PathBuf) {
        use candle_core::quantized::{gguf_file::Value, GgmlDType, QTensor};
        let cfg = &model.cfg;
        let im = model.ffn_dense[0];
        let dense = |m: &QMatMul| dense_matrix(m).unwrap().clone();
        let mut hf = vec![
            ("model.embed_tokens.weight".to_string(), model.embed.clone()),
            ("model.norm.weight".to_string(), model.norm.clone()),
        ];
        let mut gg = vec![
            ("token_embd.weight".to_string(), model.embed.clone()),
            ("output_norm.weight".to_string(), model.norm.clone()),
        ];
        for (i, l) in model.layers.iter().enumerate() {
            let pairs = [
                (
                    "input_layernorm.weight",
                    "attn_norm.weight",
                    l.attn_norm.clone(),
                ),
                ("self_attn.q_proj.weight", "attn_q.weight", dense(&l.q)),
                ("self_attn.k_proj.weight", "attn_k.weight", dense(&l.k)),
                ("self_attn.v_proj.weight", "attn_v.weight", dense(&l.v)),
                ("self_attn.q_proj.bias", "attn_q.bias", l.q_bias.clone()),
                ("self_attn.k_proj.bias", "attn_k.bias", l.k_bias.clone()),
                ("self_attn.v_proj.bias", "attn_v.bias", l.v_bias.clone()),
                ("self_attn.o_proj.weight", "attn_output.weight", dense(&l.o)),
                (
                    "post_attention_layernorm.weight",
                    "ffn_norm.weight",
                    l.ffn_norm.clone(),
                ),
                ("mlp.gate_proj.weight", "ffn_gate.weight", dense(&l.gate)),
                ("mlp.up_proj.weight", "ffn_up.weight", dense(&l.up)),
                ("mlp.down_proj.weight", "ffn_down.weight", dense(&l.down)),
            ];
            for (hf_name, gg_name, t) in pairs {
                hf.push((format!("model.layers.{i}.{hf_name}"), t.clone()));
                gg.push((format!("blk.{i}.{gg_name}"), t));
            }
        }
        let hf_dir = root.join("hf");
        std::fs::create_dir_all(&hf_dir).unwrap();
        let config = serde_json::json!({
            "model_type": "qwen2",
            "hidden_size": cfg.hidden_size,
            "num_hidden_layers": cfg.num_layers,
            "num_attention_heads": cfg.num_heads,
            "num_key_value_heads": cfg.num_kv_heads,
            "rms_norm_eps": cfg.rms_norm_eps,
            "rope_theta": cfg.rope_theta,
            "vocab_size": cfg.vocab_size,
            "intermediate_size": im,
            "tie_word_embeddings": true,
        });
        std::fs::write(hf_dir.join("config.json"), config.to_string()).unwrap();
        let map: std::collections::HashMap<String, Tensor> = hf.into_iter().collect();
        candle_core::safetensors::save(&map, hf_dir.join("model.safetensors")).unwrap();

        let gguf = root.join("tiny.gguf");
        let u = |v: usize| Value::U32(v as u32);
        let metadata = [
            ("general.architecture", Value::String("qwen2".into())),
            ("qwen2.embedding_length", u(cfg.hidden_size)),
            ("qwen2.block_count", u(cfg.num_layers)),
            ("qwen2.attention.head_count", u(cfg.num_heads)),
            ("qwen2.attention.head_count_kv", u(cfg.num_kv_heads)),
            (
                "qwen2.attention.layer_norm_rms_epsilon",
                Value::F32(cfg.rms_norm_eps as f32),
            ),
            ("qwen2.rope.freq_base", Value::F32(cfg.rope_theta as f32)),
        ];
        let qts: Vec<(String, QTensor)> = gg
            .into_iter()
            .map(|(n, t)| (n, QTensor::quantize(&t, GgmlDType::F32).unwrap()))
            .collect();
        let md: Vec<(&str, &Value)> = metadata.iter().map(|(k, v)| (*k, v)).collect();
        let ts: Vec<(&str, &QTensor)> = qts.iter().map(|(n, q)| (n.as_str(), q)).collect();
        let mut file = File::create(&gguf).unwrap();
        gguf_file::write(&mut file, &md, &ts).unwrap();
        (hf_dir, gguf)
    }

    #[test]
    fn forward_with_hook_continue_is_bit_exact_with_forward() {
        let model = tiny_model(4);
        let rows = vec![vec![1, 5, 7, 9, 2], vec![3, 3, 8, 30, 4]];
        let (reference, cache) = model.forward(&rows, None).unwrap();
        assert_eq!(cache.len(), 5);
        let mut seen = Vec::new();
        let out = model
            .forward_with_hook(&rows, &mut |obs| {
                assert_eq!(obs.num_layers, 4);
                assert_eq!(obs.delta.len(), 2);
                assert!(obs.delta.iter().all(|d| d.is_finite() && *d > 0.0));
                seen.push(obs.layer);
                Ok(HookAction::Continue)
            })
            .unwrap();
        assert_eq!(seen, vec![0, 1, 2, 3]);
        assert_eq!(out.layers_executed, 4);
        assert_eq!(out.exited_at, None);
        assert_eq!(
            out.hidden.to_vec3::<f32>().unwrap(),
            reference.to_vec3::<f32>().unwrap()
        );
    }

    #[test]
    fn semantic_scorer_hook_continues_and_exits() {
        let scorer = crate::semantic_qwen::QwenSemanticScorer::from_model_for_test(tiny_model(4));
        let ids = [1, 5, 7, 9];
        let mut seen = Vec::new();
        let full = scorer
            .forward_with_hook(&ids, &mut |obs| {
                seen.push(obs.layer);
                Ok(HookAction::Continue)
            })
            .unwrap();
        assert_eq!(seen, vec![0, 1, 2, 3]);
        assert_eq!(full.layers_executed, 4);
        assert_eq!(full.exited_at, None);
        let early = scorer
            .forward_with_hook(&ids, &mut |obs| {
                Ok(if obs.layer == 1 {
                    HookAction::Exit
                } else {
                    HookAction::Continue
                })
            })
            .unwrap();
        assert_eq!(early.layers_executed, 2);
        assert_eq!(early.exited_at, Some(1));
    }

    #[test]
    fn forward_with_hook_exit_skips_remaining_layers() {
        let model = tiny_model(4);
        let rows = vec![vec![1, 5, 7, 9, 2]];
        let mut calls = 0usize;
        let mut residual = None;
        let out = model
            .forward_with_hook(&rows, &mut |obs| {
                calls += 1;
                if obs.layer == 1 {
                    residual = Some(obs.residual.clone());
                    return Ok(HookAction::Exit);
                }
                Ok(HookAction::Continue)
            })
            .unwrap();
        // The hook runs after each computed layer, so two calls means layers 2
        // and 3 were never computed.
        assert_eq!(calls, 2);
        assert_eq!(out.layers_executed, 2);
        assert!(out.layers_executed < model.config().num_layers);
        assert_eq!(out.exited_at, Some(1));
        let expected = model.final_norm(&residual.unwrap()).unwrap();
        assert_eq!(
            out.hidden.to_vec3::<f32>().unwrap(),
            expected.to_vec3::<f32>().unwrap()
        );
        let (full, _) = model.forward(&rows, None).unwrap();
        assert_ne!(
            out.hidden.to_vec3::<f32>().unwrap(),
            full.to_vec3::<f32>().unwrap()
        );
    }

    #[test]
    fn forward_with_hook_exit_after_last_layer_is_a_full_pass() {
        let model = tiny_model(3);
        let rows = vec![vec![4, 6, 8]];
        let out = model
            .forward_with_hook(&rows, &mut |_| Ok(HookAction::Exit))
            .unwrap();
        assert_eq!(out.layers_executed, 1);
        let out = model
            .forward_with_hook(&rows, &mut |obs| {
                Ok(if obs.layer == 2 {
                    HookAction::Exit
                } else {
                    HookAction::Continue
                })
            })
            .unwrap();
        assert_eq!((out.layers_executed, out.exited_at), (3, None));
    }

    #[test]
    fn forward_with_hook_delta_matches_definition_and_propagates_errors() {
        let model = tiny_model(2);
        let rows = vec![vec![1, 2, 3]];
        let mut prev = model.prepare(&rows, None).unwrap().x;
        let mut deltas = Vec::new();
        model
            .forward_with_hook(&rows, &mut |obs| {
                let last = |x: &Tensor| x.narrow(1, 2, 1).unwrap().flatten_all().unwrap();
                let (c, p) = (
                    last(obs.residual).to_vec1::<f32>().unwrap(),
                    last(&prev).to_vec1::<f32>().unwrap(),
                );
                let step: f64 = c
                    .iter()
                    .zip(&p)
                    .map(|(a, b)| ((a - b) as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let norm: f64 = c.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
                deltas.push((obs.delta[0], step / (norm + KINETIC_EPS)));
                prev = obs.residual.clone();
                Ok(HookAction::Continue)
            })
            .unwrap();
        assert_eq!(deltas.len(), 2);
        for (got, want) in deltas {
            assert!(
                (got - want).abs() <= 1e-5 * want.max(1.0),
                "{got} vs {want}"
            );
        }
        let err = model
            .forward_with_hook(&rows, &mut |_| Err(ModelError::Inference("stop".into())))
            .err()
            .unwrap();
        assert!(matches!(err, ModelError::Inference(_)));
        assert!(model
            .forward_with_hook(&[], &mut |_| Ok(HookAction::Continue))
            .is_err());
    }

    /// Real-weight check of the hook path. Run explicitly:
    /// `GENZERO_QWEN_HOOK_WEIGHTS=<.gguf or HF dir> cargo test -p gen-zero-model
    /// --lib forward_with_hook_on_real_weights -- --ignored --nocapture`.
    /// A missing variable fails the test.
    #[test]
    #[ignore = "needs Qwen2 weights at $GENZERO_QWEN_HOOK_WEIGHTS"]
    fn forward_with_hook_on_real_weights() {
        let path = std::env::var("GENZERO_QWEN_HOOK_WEIGHTS")
            .expect("set GENZERO_QWEN_HOOK_WEIGHTS to a Qwen2 .gguf file or HF directory");
        let model = QwenModel::load(Path::new(&path)).unwrap();
        let layers = model.config().num_layers;
        let rows = vec![vec![785, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315]];
        let (reference, _) = model.forward(&rows, None).unwrap();
        let mut deltas = Vec::new();
        let full = model
            .forward_with_hook(&rows, &mut |obs| {
                deltas.push(obs.delta[0]);
                Ok(HookAction::Continue)
            })
            .unwrap();
        assert_eq!(full.layers_executed, layers);
        assert_eq!(
            full.hidden.to_vec3::<f32>().unwrap(),
            reference.to_vec3::<f32>().unwrap()
        );
        let cut = layers / 2;
        let mut calls = 0usize;
        let early = model
            .forward_with_hook(&rows, &mut |obs| {
                calls += 1;
                Ok(if obs.layer == cut {
                    HookAction::Exit
                } else {
                    HookAction::Continue
                })
            })
            .unwrap();
        assert_eq!((calls, early.layers_executed), (cut + 1, cut + 1));
        assert_eq!(early.exited_at, Some(cut));
        println!(
            "real weights {path}: layers={layers} early_exit_layers={} deltas={deltas:.4?}",
            early.layers_executed
        );
    }
}
