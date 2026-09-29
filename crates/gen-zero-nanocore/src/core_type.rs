//! gen-zero-nanocore domain specifications and micro-kernel instances.

use crate::error::NanoCoreError;
use gen_zero_core::{CompressedLatent, SimplexEtfFrame};
use serde::{Deserialize, Serialize};

/// Strongly-typed domain identifier for micro-cores
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DomainId(pub u32);

pub const DOMAIN_GENERAL: DomainId = DomainId(0);
pub const DOMAIN_VISION: DomainId = DomainId(1);
pub const DOMAIN_BROWSER: DomainId = DomainId(2);
pub const DOMAIN_SQL: DomainId = DomainId(3);
pub const DOMAIN_CODE: DomainId = DomainId(4);
pub const DOMAIN_TRADING: DomainId = DomainId(5);
pub const DOMAIN_SAFETY: DomainId = DomainId(6);

/// Upper bound on `out_dim`, shared by every load path.
pub const MAX_OUT_DIM: usize = 4096;

/// Specialized micro-kernel instance representing a domain expert.
///
/// Output channels are bound to `action_vocab`, not to request positions:
/// vertex `i` of the simplex ETF always scores `action_vocab[i]`. A request
/// selects its candidates by name, so reordering or pruning candidates never
/// changes the score of the actions that remain.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NanoCoreInstance {
    pub domain_id: DomainId,
    pub name: String,
    pub prototype: CompressedLatent,
    /// Linear projection matrix: shape (out_dim, 128)
    pub projection_weights: Vec<f32>,
    /// State value evaluation weights: 128 floats
    pub value_weights: CompressedLatent,
    pub out_dim: usize,
    /// Inherent core capability / calibration scale
    pub base_confidence: f32,
    /// Stable action vocabulary: ETF vertex `i` scores `action_vocab[i]`.
    /// Required on every load path; a core without it cannot be deserialized.
    pub action_vocab: Vec<String>,
}

impl NanoCoreInstance {
    /// Build a core from trained parameters. Fails closed on any invalid field.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        domain_id: DomainId,
        name: &str,
        prototype: CompressedLatent,
        projection_weights: Vec<f32>,
        value_weights: CompressedLatent,
        out_dim: usize,
        base_confidence: f32,
        action_vocab: Vec<String>,
    ) -> Result<Self, NanoCoreError> {
        let core = Self {
            domain_id,
            name: name.to_string(),
            prototype,
            projection_weights,
            value_weights,
            out_dim,
            base_confidence,
            action_vocab,
        };
        core.validate()?;
        Ok(core)
    }

    /// Check every invariant `forward` relies on. Every load path calls this.
    pub fn validate(&self) -> Result<(), NanoCoreError> {
        let invalid = |detail: String| Err(NanoCoreError::InvalidCore(detail));
        if self.out_dim == 0 || self.out_dim > MAX_OUT_DIM {
            return invalid(format!(
                "out_dim must be between 1 and {MAX_OUT_DIM} (got {})",
                self.out_dim
            ));
        }
        if self.projection_weights.len() != self.out_dim * 128 {
            return invalid(format!(
                "projection_weights must contain out_dim * 128 values (expected {}, got {})",
                self.out_dim * 128,
                self.projection_weights.len()
            ));
        }
        if self
            .projection_weights
            .iter()
            .chain(self.value_weights.as_slice())
            .chain(self.prototype.as_slice())
            .any(|value| !value.is_finite())
        {
            return invalid("weights and prototype must contain only finite numbers".into());
        }
        if !self.base_confidence.is_finite() || !(0.0..=1.0).contains(&self.base_confidence) {
            return invalid(format!(
                "base_confidence must be finite and in [0, 1] (got {})",
                self.base_confidence
            ));
        }
        if self.action_vocab.is_empty() {
            return invalid("action_vocab must name at least one action".into());
        }
        let mut seen = std::collections::BTreeSet::new();
        for action in &self.action_vocab {
            if action.is_empty() {
                return invalid("action_vocab contains an empty action name".into());
            }
            if !seen.insert(action.as_str()) {
                return invalid(format!("action_vocab repeats action {action:?}"));
            }
        }
        // SimplexEtfFrame needs dimension >= K - 1 for K vertices.
        if self.out_dim + 1 < self.action_vocab.len() {
            return invalid(format!(
                "out_dim {} cannot host an ETF over {} vocabulary actions (needs >= {})",
                self.out_dim,
                self.action_vocab.len(),
                self.action_vocab.len() - 1
            ));
        }
        Ok(())
    }

    /// Vocabulary slot of `action`, if the core was trained on it.
    #[inline]
    pub fn vocab_index(&self, action: &str) -> Option<usize> {
        self.action_vocab.iter().position(|a| a == action)
    }

    /// Compute gating alignment score with input state: cosine similarity in [0.0, 1.0].
    #[inline]
    pub fn gating_score(&self, state: &CompressedLatent) -> f32 {
        let sim = self.prototype.cosine_similarity(state);
        // Rectified gating: only positive alignment triggers activation
        sim.max(0.0)
    }

    /// Forward inference for a given input state and named candidates.
    ///
    /// The ETF is built over the whole vocabulary, so the logit of an action
    /// depends only on the state and that action's vertex, never on which
    /// other candidates were requested or in what order. An unknown or
    /// repeated candidate is an error, never a guessed channel.
    ///
    /// Returns:
    /// - logits: one raw score per candidate, in request order
    /// - confidence: calibrated confidence score in [0.0, 1.0]
    /// - value: state value estimate
    pub fn forward(
        &self,
        state: &CompressedLatent,
        candidates: &[&str],
    ) -> Result<(Vec<f32>, f32, f32), NanoCoreError> {
        if candidates.is_empty() {
            return Err(NanoCoreError::Model(
                gen_zero_model::ModelError::EmptyOptions,
            ));
        }
        self.validate()?;
        let mut slots = Vec::with_capacity(candidates.len());
        for &candidate in candidates {
            let slot = self
                .vocab_index(candidate)
                .ok_or_else(|| NanoCoreError::UnknownAction(candidate.to_string()))?;
            if slots.contains(&slot) {
                return Err(NanoCoreError::DuplicateAction(candidate.to_string()));
            }
            slots.push(slot);
        }

        let mut projected = vec![0.0_f32; self.out_dim];

        // Linear layer: projected = W * state
        for (i, proj) in projected.iter_mut().enumerate() {
            let row_offset = i * 128;
            *proj = gen_zero_core::dot_product_f32(
                &self.projection_weights[row_offset..row_offset + 128],
                state.as_slice(),
            );
        }

        // Architecture Invariant: RMSNorm hypersphere geometric normalization (v / RMS(v))
        let sum_sq: f32 = projected.iter().map(|&x| x * x).sum();
        let rms = (sum_sq / (self.out_dim as f32) + 1e-8).sqrt();
        let inv_rms = 1.0 / rms;
        for x in projected.iter_mut() {
            *x *= inv_rms;
        }

        // Project onto the vocabulary-wide Simplex ETF, then read the requested vertices.
        let etf = SimplexEtfFrame::new(self.action_vocab.len(), self.out_dim)?;
        let mut vocab_logits = vec![0.0_f32; self.action_vocab.len()];
        etf.project_logits(&projected, &mut vocab_logits);
        let logits = slots.iter().map(|&slot| vocab_logits[slot]).collect();

        // Compute alignment-dependent confidence (strictly 0.0 if gate == 0.0)
        let gate = self.gating_score(state);
        let confidence = (self.base_confidence * gate).clamp(0.0, 1.0);

        // Independent state value evaluation (immune to ETF zero-sum property)
        let value =
            gen_zero_core::dot_product_f32(self.value_weights.as_slice(), state.as_slice()).tanh();

        Ok((logits, confidence, value))
    }

    /// Estimated RAM footprint in bytes.
    pub fn size_in_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.name.len()
            + self.projection_weights.len() * std::mem::size_of::<f32>()
            + self
                .action_vocab
                .iter()
                .map(|a| a.len() + std::mem::size_of::<String>())
                .sum::<usize>()
    }

    /// Serialize into compact binary and zstd compress.
    pub fn compress_zstd(&self) -> Result<Vec<u8>, NanoCoreError> {
        let mut binary = Vec::with_capacity(self.size_in_bytes());
        binary.extend_from_slice(&self.domain_id.0.to_le_bytes());
        binary.extend_from_slice(&(self.out_dim as u32).to_le_bytes());
        binary.extend_from_slice(&self.base_confidence.to_le_bytes());

        let name_bytes = self.name.as_bytes();
        binary.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        binary.extend_from_slice(name_bytes);

        for &val in self.prototype.as_slice() {
            binary.extend_from_slice(&val.to_le_bytes());
        }

        for &val in &self.projection_weights {
            binary.extend_from_slice(&val.to_le_bytes());
        }

        for &val in self.value_weights.as_slice() {
            binary.extend_from_slice(&val.to_le_bytes());
        }

        binary.extend_from_slice(&(self.action_vocab.len() as u32).to_le_bytes());
        for action in &self.action_vocab {
            binary.extend_from_slice(&(action.len() as u32).to_le_bytes());
            binary.extend_from_slice(action.as_bytes());
        }

        zstd::encode_all(&binary[..], 3).map_err(|e| NanoCoreError::CompressionError(e.to_string()))
    }

    /// Decompress from zstd compressed byte buffer into NanoCoreInstance.
    pub fn decompress_zstd(compressed: &[u8]) -> Result<Self, NanoCoreError> {
        let binary = zstd::decode_all(compressed)
            .map_err(|e| NanoCoreError::CompressionError(e.to_string()))?;

        if binary.len() < 16 {
            return Err(NanoCoreError::CompressionError("Buffer too small".into()));
        }

        let mut offset = 0;
        let domain_id = DomainId(u32::from_le_bytes(
            binary[offset..offset + 4].try_into().unwrap(),
        ));
        offset += 4;
        let out_dim = u32::from_le_bytes(binary[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let base_confidence = f32::from_le_bytes(binary[offset..offset + 4].try_into().unwrap());
        offset += 4;

        let name_len = u32::from_le_bytes(binary[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        if binary.len() < offset + name_len {
            return Err(NanoCoreError::CompressionError(
                "Invalid name length".into(),
            ));
        }
        let name = String::from_utf8_lossy(&binary[offset..offset + name_len]).to_string();
        offset += name_len;

        // Prototype: 128 floats (512 bytes)
        if binary.len() < offset + 128 * 4 {
            return Err(NanoCoreError::CompressionError(
                "Invalid prototype length".into(),
            ));
        }
        let mut proto_vals = [0.0_f32; 128];
        for val in &mut proto_vals {
            *val = f32::from_le_bytes(binary[offset..offset + 4].try_into().unwrap());
            offset += 4;
        }
        let prototype = CompressedLatent { values: proto_vals };

        // Weights: out_dim * 128 floats
        let num_weights = out_dim * 128;
        if binary.len() < offset + num_weights * 4 {
            return Err(NanoCoreError::CompressionError(
                "Invalid weights length".into(),
            ));
        }
        let mut projection_weights = Vec::with_capacity(num_weights);
        for _ in 0..num_weights {
            projection_weights.push(f32::from_le_bytes(
                binary[offset..offset + 4].try_into().unwrap(),
            ));
            offset += 4;
        }

        // Value weights: 128 floats (512 bytes)
        if binary.len() < offset + 128 * 4 {
            return Err(NanoCoreError::CompressionError(
                "Invalid value weights length".into(),
            ));
        }
        let mut val_arr = [0.0_f32; 128];
        for val in &mut val_arr {
            *val = f32::from_le_bytes(binary[offset..offset + 4].try_into().unwrap());
            offset += 4;
        }
        let value_weights = CompressedLatent { values: val_arr };

        let read_u32 = |offset: &mut usize, what: &str| -> Result<usize, NanoCoreError> {
            let bytes = binary
                .get(*offset..*offset + 4)
                .ok_or_else(|| NanoCoreError::CompressionError(format!("Missing {what}")))?;
            *offset += 4;
            Ok(u32::from_le_bytes(bytes.try_into().unwrap()) as usize)
        };
        let vocab_len = read_u32(&mut offset, "action vocabulary length")?;
        let mut action_vocab = Vec::with_capacity(vocab_len.min(4097));
        for _ in 0..vocab_len {
            let len = read_u32(&mut offset, "action name length")?;
            let bytes = binary.get(offset..offset + len).ok_or_else(|| {
                NanoCoreError::CompressionError("Invalid action name length".into())
            })?;
            let action = std::str::from_utf8(bytes).map_err(|e| {
                NanoCoreError::CompressionError(format!("Action name is not UTF-8: {e}"))
            })?;
            action_vocab.push(action.to_string());
            offset += len;
        }
        if offset != binary.len() {
            return Err(NanoCoreError::CompressionError(
                "Trailing bytes after action vocabulary".into(),
            ));
        }

        let core = Self {
            domain_id,
            name,
            prototype,
            projection_weights,
            value_weights,
            out_dim,
            base_confidence,
            action_vocab,
        };
        core.validate()?;
        Ok(core)
    }
}

/// Synthetic sin/cos weights for tests only. These are not trained parameters
/// and carry no domain knowledge, so the module is compiled only for this
/// crate's tests or when a dependent enables the `test-fixtures` feature
/// from its `[dev-dependencies]`.
#[cfg(any(test, feature = "test-fixtures"))]
pub mod fixtures {
    use super::*;

    /// Deterministic untrained core over `action_vocab`.
    pub fn synthetic_core(
        domain_id: DomainId,
        name: &str,
        prototype: CompressedLatent,
        out_dim: usize,
        base_confidence: f32,
        action_vocab: &[&str],
    ) -> NanoCoreInstance {
        let mut weights = vec![0.0_f32; out_dim * 128];
        for i in 0..out_dim {
            for j in 0..128 {
                let angle = (i * 128 + j) as f32 * 0.1337;
                weights[i * 128 + j] = (angle.sin() / (128.0_f32).sqrt()).clamp(-1.0, 1.0);
            }
        }
        let mut val_arr = [0.0_f32; 128];
        for (j, val) in val_arr.iter_mut().enumerate() {
            *val = ((j as f32) * 0.25).cos() / (128.0_f32).sqrt();
        }
        NanoCoreInstance::from_parts(
            domain_id,
            name,
            prototype,
            weights,
            CompressedLatent { values: val_arr },
            out_dim,
            base_confidence,
            action_vocab.iter().map(|a| a.to_string()).collect(),
        )
        .expect("synthetic fixture parameters are valid")
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::synthetic_core;
    use super::*;

    fn state(seed: f32) -> CompressedLatent {
        let mut s = CompressedLatent::zeros();
        for (j, v) in s.values.iter_mut().enumerate() {
            *v = ((j as f32) * 0.37 + seed).sin();
        }
        s
    }

    #[test]
    fn test_nanocore_compression_roundtrip() {
        let proto = CompressedLatent::zeros();
        let core = synthetic_core(
            DOMAIN_TRADING,
            "TradingCore",
            proto,
            32,
            0.95,
            &["buy", "sell"],
        );

        let compressed = core.compress_zstd().expect("Compression should succeed");
        assert!(!compressed.is_empty());

        let restored =
            NanoCoreInstance::decompress_zstd(&compressed).expect("Decompression should succeed");
        assert_eq!(core.domain_id, restored.domain_id);
        assert_eq!(core.name, restored.name);
        assert_eq!(core.out_dim, restored.out_dim);
        assert_eq!(core.projection_weights, restored.projection_weights);
        assert_eq!(core.action_vocab, restored.action_vocab);
    }

    #[test]
    fn forward_scores_are_bound_to_action_identity_not_position() {
        let vocab = ["a", "b", "c", "d", "e"];
        let core = synthetic_core(DOMAIN_GENERAL, "g", state(0.0), 16, 0.9, &vocab);
        let s = state(1.3);
        let (full, conf, value) = core.forward(&s, &vocab).unwrap();
        let by_name = |names: &[&str], logits: &[f32], name: &str| {
            logits[names.iter().position(|n| *n == name).unwrap()]
        };

        // Permutation: every action keeps its bit-exact logit.
        let permuted = ["d", "a", "e", "c", "b"];
        let (perm, conf_p, value_p) = core.forward(&s, &permuted).unwrap();
        for name in vocab {
            assert_eq!(
                by_name(&vocab, &full, name).to_bits(),
                by_name(&permuted, &perm, name).to_bits(),
                "{name}"
            );
        }
        assert_eq!(
            (conf.to_bits(), value.to_bits()),
            (conf_p.to_bits(), value_p.to_bits())
        );

        // Pruning: [a, b, c] vs [c, a] keeps the survivors' channels.
        let pruned = ["c", "a"];
        let (pr, _, _) = core.forward(&s, &pruned).unwrap();
        for name in pruned {
            assert_eq!(
                by_name(&vocab, &full, name).to_bits(),
                by_name(&pruned, &pr, name).to_bits(),
                "{name}"
            );
        }
        // The logits are not trivially equal, otherwise the test proves nothing.
        assert!(full.iter().any(|&l| (l - full[0]).abs() > 1e-4), "{full:?}");
    }

    #[test]
    fn forward_rejects_unknown_duplicate_and_empty_candidates() {
        let core = synthetic_core(DOMAIN_GENERAL, "g", state(0.0), 8, 0.9, &["a", "b"]);
        let s = state(0.5);
        assert_eq!(
            core.forward(&s, &["a", "zzz"]).unwrap_err(),
            NanoCoreError::UnknownAction("zzz".into())
        );
        assert_eq!(
            core.forward(&s, &["b", "b"]).unwrap_err(),
            NanoCoreError::DuplicateAction("b".into())
        );
        assert!(core.forward(&s, &[]).is_err());
    }

    #[test]
    fn validate_rejects_bad_vocabulary() {
        let base = synthetic_core(DOMAIN_GENERAL, "g", state(0.0), 2, 0.9, &["a", "b", "c"]);
        for vocab in [
            vec![],
            vec!["a", "a"],
            vec!["a", ""],
            vec!["a", "b", "c", "d"],
        ] {
            let mut core = base.clone();
            core.action_vocab = vocab.iter().map(|a| a.to_string()).collect();
            assert!(
                matches!(core.validate(), Err(NanoCoreError::InvalidCore(_))),
                "{vocab:?}"
            );
        }
    }

    #[test]
    fn serde_requires_action_vocab() {
        let core = synthetic_core(DOMAIN_GENERAL, "g", state(0.0), 4, 0.9, &["a", "b"]);
        let mut json = serde_json::to_value(&core).unwrap();
        json.as_object_mut().unwrap().remove("action_vocab");
        let err = serde_json::from_value::<NanoCoreInstance>(json).unwrap_err();
        assert!(err.to_string().contains("action_vocab"), "{err}");
    }
}
