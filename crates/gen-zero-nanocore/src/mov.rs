//! gen-zero-nanocore Mixture of Vectors (MoV) fusion engine and Fallback Watchdog.

use crate::core_type::{DomainId, NanoCoreInstance};
use crate::error::NanoCoreError;
use gen_zero_core::{ActionId, CompressedLatent, LocalActionFrame, NormalizedEntropy};

/// Configuration thresholds for the Fallback Watchdog.
#[derive(Debug, Clone, Copy)]
pub struct WatchdogConfig {
    pub min_composite_confidence: f32,
    pub min_composite_value: f32,
    pub gating_temperature: f32,
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        Self {
            min_composite_confidence: 0.30,
            min_composite_value: -0.85,
            gating_temperature: 1.0,
        }
    }
}

/// Output of MoV fusion across active micro-cores.
#[derive(Debug, Clone)]
pub struct FusedDecision {
    pub selected_action: ActionId,
    pub composite_entropy: NormalizedEntropy,
    pub composite_confidence: f32,
    pub composite_value: f32,
    pub probabilities: Vec<f32>,
    pub weights: Vec<(DomainId, f32)>,
}

/// Mixture of Vectors (MoV) Fusion Engine.
#[derive(Debug, Clone)]
pub struct MoVFusionEngine {
    config: WatchdogConfig,
}

impl MoVFusionEngine {
    pub fn new(config: WatchdogConfig) -> Self {
        Self { config }
    }

    /// Fuse predictions from a slice of active micro-cores using RMSNorm and Bayesian confidence weighting.
    ///
    /// Each core scores the frame's candidates by name through its own action
    /// vocabulary. Every reduction and the argmax run in ascending `ActionId`
    /// order, so permuting the frame returns bit-identical per-action results.
    pub fn fuse(
        &self,
        cores: &[&NanoCoreInstance],
        state: &CompressedLatent,
        frame: &LocalActionFrame<'_>,
    ) -> Result<FusedDecision, NanoCoreError> {
        let n_candidates = frame.len();
        if n_candidates == 0 {
            return Err(NanoCoreError::Model(
                gen_zero_model::ModelError::EmptyOptions,
            ));
        }

        if cores.is_empty() {
            return Err(NanoCoreError::CoreNotFound(0));
        }

        let names = frame.candidate_names();
        let ids = frame.actions();
        // Canonical order: ascending ActionId. Positions in the request carry no meaning.
        let mut order: Vec<usize> = (0..n_candidates).collect();
        order.sort_by_key(|&i| ids[i]);
        if let Some(pair) = order.windows(2).find(|w| ids[w[0]] == ids[w[1]]) {
            return Err(NanoCoreError::DuplicateAction(names[pair[1]].to_string()));
        }

        let num_cores = cores.len();
        let mut gating_weights = Vec::with_capacity(num_cores);
        let mut core_outputs = Vec::with_capacity(num_cores);

        // Step 1: Forward each active core and compute unnormalized weights
        for core in cores {
            let (logits, conf, val) = core.forward(state, names)?;
            let gate = core
                .gating_score(state)
                .powf(self.config.gating_temperature);
            let unnorm_weight = gate * conf;
            gating_weights.push(unnorm_weight);
            core_outputs.push((logits, conf, val));
        }

        // Step 2: Normalize fusion weights
        let sum_weight: f32 = gating_weights.iter().sum();
        let mut normalized_weights = Vec::with_capacity(num_cores);
        if sum_weight <= 1e-8 {
            // All gates are ~0. Composite confidence is then ~0 too, so the
            // confidence watchdog below refuses; this never returns silently.
            let uniform = 1.0 / (num_cores as f32);
            for _ in 0..num_cores {
                normalized_weights.push(uniform);
            }
        } else {
            let inv_sum = 1.0 / sum_weight;
            for &w in &gating_weights {
                normalized_weights.push(w * inv_sum);
            }
        }

        // Step 3: Weighted Log-Probability aggregation
        // Convert logits to local softmax probs for each core, then fuse:
        // log P_comp(a) = \sum_k w_k log P_k(a)
        let mut fused_log_probs = vec![0.0_f32; n_candidates];
        let mut composite_confidence = 0.0_f32;
        let mut composite_value = 0.0_f32;

        for (k, (logits, conf, val)) in core_outputs.iter().enumerate() {
            let w = normalized_weights[k];

            composite_confidence += w * conf;
            composite_value += w * val;

            // Compute local softmax
            let max_l = order
                .iter()
                .map(|&i| logits[i])
                .fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0_f32;
            let mut probs_k = vec![0.0_f32; n_candidates];
            for &i in &order {
                let e = (logits[i] - max_l).exp();
                probs_k[i] = e;
                exp_sum += e;
            }
            let inv_exp_sum = 1.0 / exp_sum.max(1e-12);
            for &i in &order {
                let p = (probs_k[i] * inv_exp_sum).max(1e-15);
                fused_log_probs[i] += w * p.ln();
            }
        }

        // Exponentiate fused log probs and renormalize
        let max_fused = order
            .iter()
            .map(|&i| fused_log_probs[i])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut fused_probs = vec![0.0_f32; n_candidates];
        let mut sum_fused = 0.0_f32;
        for &i in &order {
            let e = (fused_log_probs[i] - max_fused).exp();
            fused_probs[i] = e;
            sum_fused += e;
        }
        let inv_fused_sum = 1.0 / sum_fused.max(1e-12);
        for p in fused_probs.iter_mut() {
            *p *= inv_fused_sum;
        }

        // Step 4: Fallback Watchdog Check
        if composite_confidence < self.config.min_composite_confidence {
            return Err(NanoCoreError::ConfidenceWatchdogTripped {
                confidence: composite_confidence,
                threshold: self.config.min_composite_confidence,
            });
        }
        if composite_value < self.config.min_composite_value {
            return Err(NanoCoreError::ValueWatchdogTripped {
                value: composite_value,
                threshold: self.config.min_composite_value,
            });
        }

        // Step 5: Argmax selection; ties go to the smallest ActionId.
        let mut best_idx = order[0];
        for &i in &order[1..] {
            if fused_probs[i] > fused_probs[best_idx] {
                best_idx = i;
            }
        }

        let selected_action = ids[best_idx];
        let canonical_probs: Vec<f32> = order.iter().map(|&i| fused_probs[i]).collect();
        let entropy = NormalizedEntropy::from_probabilities(&canonical_probs);

        let weights = cores
            .iter()
            .zip(normalized_weights.iter())
            .map(|(c, &w)| (c.domain_id, w))
            .collect();

        Ok(FusedDecision {
            selected_action,
            composite_entropy: entropy,
            composite_confidence,
            composite_value,
            probabilities: fused_probs,
            weights,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_type::fixtures::synthetic_core;
    use crate::core_type::*;

    const VOCAB: [&str; 4] = ["buy", "hold", "sell", "wait"];

    fn aligned() -> (NanoCoreInstance, NanoCoreInstance, CompressedLatent) {
        let mut proto = CompressedLatent::zeros();
        proto.values[0] = 1.0;
        let core1 = synthetic_core(
            DOMAIN_GENERAL,
            "GeneralCore",
            proto.clone(),
            32,
            0.9,
            &VOCAB,
        );
        // Different vocabulary order: channel binding must follow names, not slots.
        let core2 = synthetic_core(
            DOMAIN_TRADING,
            "TradingCore",
            proto,
            32,
            0.85,
            &["wait", "sell", "buy", "hold"],
        );
        let mut state = CompressedLatent::zeros();
        state.values[0] = 1.0;
        for (j, v) in state.values.iter_mut().enumerate().skip(1) {
            *v = ((j as f32) * 0.21).sin() * 0.05;
        }
        (core1, core2, state)
    }

    #[test]
    fn test_mov_fusion_and_watchdog() {
        let (core1, core2, state_aligned) = aligned();
        let engine = MoVFusionEngine::new(WatchdogConfig::default());

        let actions = ["buy", "hold", "sell"];
        let ids = [ActionId(1), ActionId(2), ActionId(3)];
        let frame = LocalActionFrame::new(&actions, &ids).unwrap();

        let decision = engine
            .fuse(&[&core1, &core2], &state_aligned, &frame)
            .unwrap();
        assert_eq!(decision.probabilities.len(), 3);
        assert!(
            decision.composite_confidence >= WatchdogConfig::default().min_composite_confidence
        );
        assert_eq!(decision.weights.len(), 2);

        // Orthogonal state (no domain expert matches): gate == 0 -> confidence watchdog must trip!
        let mut state_orthogonal = CompressedLatent::zeros();
        state_orthogonal.values[1] = 1.0;
        let err = engine
            .fuse(&[&core1, &core2], &state_orthogonal, &frame)
            .unwrap_err();
        assert!(matches!(
            err,
            NanoCoreError::ConfidenceWatchdogTripped { .. }
        ));
    }

    #[test]
    fn fuse_is_permutation_equivariant_bit_exact() {
        let (core1, core2, state) = aligned();
        let engine = MoVFusionEngine::new(WatchdogConfig::default());
        let names = ["buy", "hold", "sell", "wait"];
        let ids = [ActionId(10), ActionId(20), ActionId(30), ActionId(40)];
        let frame = LocalActionFrame::new(&names, &ids).unwrap();
        let base = engine.fuse(&[&core1, &core2], &state, &frame).unwrap();

        let p_names = ["sell", "wait", "buy", "hold"];
        let p_ids = [ActionId(30), ActionId(40), ActionId(10), ActionId(20)];
        let p_frame = LocalActionFrame::new(&p_names, &p_ids).unwrap();
        let perm = engine.fuse(&[&core1, &core2], &state, &p_frame).unwrap();

        assert_eq!(base.selected_action, perm.selected_action);
        assert_eq!(
            base.composite_entropy.0.to_bits(),
            perm.composite_entropy.0.to_bits()
        );
        for (i, id) in ids.iter().enumerate() {
            let j = p_ids.iter().position(|p| p == id).unwrap();
            assert_eq!(
                base.probabilities[i].to_bits(),
                perm.probabilities[j].to_bits(),
                "{}",
                names[i]
            );
        }
        // Distinct probabilities, so the check is not vacuous.
        assert!(base
            .probabilities
            .iter()
            .any(|p| (p - base.probabilities[0]).abs() > 1e-5));
    }

    #[test]
    fn fuse_pruning_keeps_survivor_channels() {
        let (core1, _, state) = aligned();
        let full = core1.forward(&state, &["buy", "hold", "sell"]).unwrap().0;
        let pruned = core1.forward(&state, &["buy", "sell"]).unwrap().0;
        assert_eq!(full[0].to_bits(), pruned[0].to_bits());
        assert_eq!(full[2].to_bits(), pruned[1].to_bits());
    }

    #[test]
    fn fuse_rejects_unknown_and_duplicate_candidates() {
        let (core1, core2, state) = aligned();
        let engine = MoVFusionEngine::new(WatchdogConfig::default());
        let frame = LocalActionFrame::new(&["buy", "short"], &[ActionId(1), ActionId(2)]).unwrap();
        assert_eq!(
            engine.fuse(&[&core1, &core2], &state, &frame).unwrap_err(),
            NanoCoreError::UnknownAction("short".into())
        );
        let dup = LocalActionFrame::new(&["buy", "sell"], &[ActionId(1), ActionId(1)]).unwrap();
        assert!(matches!(
            engine.fuse(&[&core1], &state, &dup).unwrap_err(),
            NanoCoreError::DuplicateAction(_)
        ));
    }
}
