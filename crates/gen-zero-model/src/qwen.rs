//! Native Qwen2 (Qwen2.5-0.5B) causal language model on candle, CPU only.
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
//! The GGUF file therefore saves disk and download size, not RAM (about
//! 2.5 GB resident either way). Its weight quantization error is kept: the
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
use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, Module, Tensor, D};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

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

/// Qwen2 decoder with tied (or separate) output head. Immutable after load,
/// so one instance serves concurrent requests.
pub struct QwenModel {
    cfg: QwenConfig,
    format: WeightFormat,
    #[allow(dead_code)]
    source: PathBuf,
    embed: Tensor,
    layers: Vec<Layer>,
    norm: Tensor,
    lm_head: QMatMul,
    cos: Tensor,
    sin: Tensor,
    device: Device,
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
    /// Load a `.gguf` file, or a directory holding `config.json` and
    /// `model.safetensors`.
    pub fn load(path: &Path) -> Result<Self, ModelError> {
        if path.is_dir() {
            Self::from_safetensors_dir(path)
        } else if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            Self::from_gguf(path)
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

    pub fn source(&self) -> &Path {
        &self.source
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
                gate: qmm(gate, &p)?,
                up: qmm(up, &p)?,
                down: qmm(down, &p)?,
            });
        }
        let (cos, sin) = rope_tables(&cfg, &device)?;
        Ok(Self {
            cfg,
            format: WeightFormat::Gguf { dtype },
            source: path.to_path_buf(),
            embed,
            layers,
            norm,
            lm_head,
            cos,
            sin,
            device,
        })
    }

    pub fn from_safetensors_dir(dir: &Path) -> Result<Self, ModelError> {
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
        for i in 0..cfg.num_layers {
            let p = format!("model.layers.{i}");
            let mat =
                |shape: &[usize], s: &str| get(shape, &format!("{p}.{s}")).map(QMatMul::Tensor);
            let vec = |n: usize, s: &str| get(&[n], &format!("{p}.{s}"));
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
                gate: mat(&[im, h], "mlp.gate_proj.weight")?,
                up: mat(&[im, h], "mlp.up_proj.weight")?,
                down: mat(&[h, im], "mlp.down_proj.weight")?,
            });
        }
        let (cos, sin) = rope_tables(&cfg, &device)?;
        Ok(Self {
            cfg,
            format: WeightFormat::SafetensorsF32,
            source: dir.to_path_buf(),
            embed,
            layers,
            norm,
            lm_head,
            cos,
            sin,
            device,
        })
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
        let cfg = &self.cfg;
        let ids: Vec<u32> = rows.iter().flatten().copied().collect();
        let ids = Tensor::from_vec(ids, (b, t), &self.device).map_err(&err)?;
        let mut x = self
            .embed
            .index_select(&ids.flatten_all().map_err(&err)?, 0)
            .and_then(|e| e.reshape((b, t, cfg.hidden_size)))
            .map_err(&err)?;
        let mask = self.mask(t, p_len)?;
        let cos = self.cos.narrow(0, p_len, t).map_err(&err)?;
        let sin = self.sin.narrow(0, p_len, t).map_err(&err)?;
        let scale = 1.0 / (cfg.head_dim as f64).sqrt();
        let eps = cfg.rms_norm_eps as f32;
        let mut cache = Vec::with_capacity(self.layers.len());
        for (li, layer) in self.layers.iter().enumerate() {
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
            let q = candle_nn::rotary_emb::rope(&q, &cos, &sin).map_err(&err)?;
            let k = candle_nn::rotary_emb::rope(&k, &cos, &sin).map_err(&err)?;
            let (k, v) = match &past {
                Some(p) => {
                    let (pk, pv) = &p.layers[li];
                    (
                        Tensor::cat(&[pk, &k], 2).map_err(&err)?,
                        Tensor::cat(&[pv, &v], 2).map_err(&err)?,
                    )
                }
                None => (k, v),
            };
            cache.push((k.clone(), v.clone()));
            let rep = cfg.num_heads / cfg.num_kv_heads;
            let k = candle_transformers::utils::repeat_kv(k, rep).map_err(&err)?;
            let v = candle_transformers::utils::repeat_kv(v, rep)
                .and_then(|v| v.contiguous())
                .map_err(&err)?;
            let att = q
                .matmul(&k.t().map_err(&err)?)
                .and_then(|a| a * scale)
                .and_then(|a| a.broadcast_add(&mask))
                .and_then(|a| candle_nn::ops::softmax_last_dim(&a))
                .map_err(&err)?;
            let y = att
                .matmul(&v)
                .and_then(|y| y.transpose(1, 2))
                .and_then(|y| y.reshape((b, t, cfg.hidden_size)))
                .and_then(|y| layer.o.forward(&y))
                .map_err(&err)?;
            x = (x + y).map_err(&err)?;
            let h = candle_nn::ops::rms_norm(&x, &layer.ffn_norm, eps).map_err(&err)?;
            let gate = layer
                .gate
                .forward(&h)
                .and_then(|g| candle_nn::ops::silu(&g))
                .map_err(&err)?;
            let up = layer.up.forward(&h).map_err(&err)?;
            let y = (gate * up)
                .and_then(|m| layer.down.forward(&m))
                .map_err(&err)?;
            x = (x + y).map_err(&err)?;
        }
        let hidden = candle_nn::ops::rms_norm(&x, &self.norm, eps).map_err(&err)?;
        Ok((
            hidden,
            KvCache {
                layers: cache,
                len: p_len + t,
                batch: b,
            },
        ))
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
mod tests {
    use super::*;

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
}
