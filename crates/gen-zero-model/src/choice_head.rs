//! gen-zero-model Action ETF Choice Head.
//!
//! Non-autoregressive 0-token choice head. Every candidate carries its own
//! manifold representation; its logit is the cosine similarity between that
//! representation and the decision state, divided by a temperature.
//!
//! History: this head used to bind candidates to Helmert simplex ETF vertices
//! in ActionId (name-hash) order. The score then depended on the name, not on
//! the candidate, and the fixed vertex geometry at T=1 kept the normalized
//! entropy above the 0.65 PolicyGate threshold for every K >= 3, so every
//! 3+ candidate decision escalated. That binding is gone; the type name stays
//! because the service and CLI expose `head=etf`.

use crate::error::ModelError;
use gen_zero_core::{ActionId, LocalActionFrame, NormalizedEntropy};

/// Default temperature when the caller supplies none. It is NOT fit to data.
///
/// Derivation: a candidate aligned with the state (cos = 1) against
/// orthogonal alternatives (cos = 0) must pass the 0.65 entropy gate for
/// every frame size the service accepts (K <= 16). At tau = 0.25 the
/// normalized entropy is ~0.16 at K = 3 and ~0.40 at K = 16. A spread of
/// 0.02 in cosine stays near uniform (entropy > 0.99 at K = 3), so real
/// ties still escalate.
pub const DEFAULT_UNCALIBRATED_TEMPERATURE: f32 = 0.25;

/// Result of one choice-head evaluation, indexed by frame slot.
#[derive(Debug, Clone, PartialEq)]
pub struct ChoiceScores {
    pub selected: ActionId,
    pub entropy: NormalizedEntropy,
    pub probabilities: Vec<f32>,
    /// Cosine similarity of each candidate representation to the state.
    pub similarities: Vec<f32>,
}

/// Absolute floor on the metric-space norm for the non-isotropic metrics
/// (fixed by the spec). It refuses collapsed or near-collapsed vectors. It is
/// NOT a rounding test: rounding is relative, so a large input can still lose
/// its direction to cancellation above this floor, and a globally tiny but
/// well-conditioned precision or matrix (e.g. all weights 1e-30) is refused
/// even though a global scale cancels in the cosine.
pub const METRIC_NORM_FLOOR: f32 = 1e-12;

/// Inner-product geometry the head scores cosines in.
///
/// None of these parameters are fit by this crate. A caller that supplies a
/// precision vector or a whitening matrix owns its calibration.
#[derive(Debug, Clone, PartialEq)]
pub enum MetricKind {
    /// Plain Euclidean cosine.
    Isotropic,
    /// `<u, v>_w = sum_i w_i u_i v_i` with every `w_i` finite and positive.
    DiagonalMahalanobis(Vec<f32>),
    /// Project with `W` (row-major, `out_dim` rows by input-width columns),
    /// then take the Euclidean cosine of `W u` and `W v`.
    WhitenedProjection { matrix: Vec<f32>, out_dim: usize },
}

impl MetricKind {
    /// Stable wire name, reported by the service in `etf.metric`.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Isotropic => "isotropic",
            Self::DiagonalMahalanobis(_) => "diagonal_mahalanobis",
            Self::WhitenedProjection { .. } => "whitened",
        }
    }
}

/// Content-scored choice head over per-candidate manifold representations.
#[derive(Debug, Clone)]
pub struct ActionETFChoiceHead {
    dimension: usize,
    temperature: f32,
    metric: MetricKind,
}

impl ActionETFChoiceHead {
    /// Create a head for `dimension`-wide representations and softmax temperature.
    pub fn new(dimension: usize, temperature: f32) -> Result<Self, ModelError> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(ModelError::InvalidTemperature(format!("{temperature}")));
        }
        if dimension == 0 {
            return Err(ModelError::Core(
                gen_zero_core::CoreError::DimensionMismatch {
                    expected: 1,
                    actual: 0,
                },
            ));
        }
        Ok(Self {
            dimension,
            temperature,
            metric: MetricKind::Isotropic,
        })
    }

    /// Head whose cosine uses the diagonal precision `precision` (one weight per dimension).
    pub fn with_diagonal_precision(
        dimension: usize,
        temperature: f32,
        precision: &[f32],
    ) -> Result<Self, ModelError> {
        let base = Self::new(dimension, temperature)?;
        if precision.len() != dimension {
            return Err(ModelError::Core(
                gen_zero_core::CoreError::DimensionMismatch {
                    expected: dimension,
                    actual: precision.len(),
                },
            ));
        }
        if let Some(i) = precision.iter().position(|w| !(w.is_finite() && *w > 0.0)) {
            return Err(ModelError::NumericalInstability(format!(
                "diagonal precision[{i}] = {} is not a finite positive number",
                precision[i]
            )));
        }
        Ok(Self {
            metric: MetricKind::DiagonalMahalanobis(precision.to_vec()),
            ..base
        })
    }

    /// Head that scores in the space `W u`, with `matrix` row-major `out_dim x in_dim`.
    pub fn with_whitening_matrix(
        in_dim: usize,
        out_dim: usize,
        temperature: f32,
        matrix: &[f32],
    ) -> Result<Self, ModelError> {
        let base = Self::new(in_dim, temperature)?;
        if out_dim == 0 {
            return Err(ModelError::Core(
                gen_zero_core::CoreError::DimensionMismatch {
                    expected: 1,
                    actual: 0,
                },
            ));
        }
        let expected = in_dim.checked_mul(out_dim).ok_or_else(|| {
            ModelError::NumericalInstability("whitening matrix size overflows usize".into())
        })?;
        if matrix.len() != expected {
            return Err(ModelError::Core(
                gen_zero_core::CoreError::DimensionMismatch {
                    expected,
                    actual: matrix.len(),
                },
            ));
        }
        if matrix.iter().any(|x| !x.is_finite()) {
            return Err(ModelError::NumericalInstability(
                "whitening matrix contains a non-finite entry".into(),
            ));
        }
        // A zero row maps every input to 0 on that axis: a dead output
        // dimension that a real whitening transform never has.
        if let Some(row) = matrix
            .chunks_exact(in_dim)
            .position(|r| r.iter().all(|&x| x == 0.0))
        {
            return Err(ModelError::NumericalInstability(format!(
                "whitening matrix row {row} is all zeros"
            )));
        }
        Ok(Self {
            metric: MetricKind::WhitenedProjection {
                matrix: matrix.to_vec(),
                out_dim,
            },
            ..base
        })
    }

    /// Build a head for any [`MetricKind`] through the matching validated constructor.
    pub fn with_metric(
        dimension: usize,
        temperature: f32,
        metric: &MetricKind,
    ) -> Result<Self, ModelError> {
        match metric {
            MetricKind::Isotropic => Self::new(dimension, temperature),
            MetricKind::DiagonalMahalanobis(precision) => {
                Self::with_diagonal_precision(dimension, temperature, precision)
            }
            MetricKind::WhitenedProjection { matrix, out_dim } => {
                Self::with_whitening_matrix(dimension, *out_dim, temperature, matrix)
            }
        }
    }

    /// Input width: the length every state and candidate must have.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    #[inline]
    pub fn metric(&self) -> &MetricKind {
        &self.metric
    }

    /// Width of the space the cosine is taken in.
    pub fn metric_dimension(&self) -> usize {
        match &self.metric {
            MetricKind::WhitenedProjection { out_dim, .. } => *out_dim,
            _ => self.dimension,
        }
    }

    #[inline]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Score each frame slot by `cos_M(state, candidate_reps[slot]) / temperature`,
    /// where `cos_M` is the cosine under the head's [`MetricKind`].
    ///
    /// Invariants:
    /// - Permutation equivariance: a candidate's score depends only on its own
    ///   representation. The softmax reduction and argmax tie-break run in
    ///   ascending ActionId order, so outputs are bit-identical under any
    ///   presentation order of (name, representation) pairs.
    /// - Scale invariance: both sides are unit-normalized in metric space
    ///   before the dot product.
    /// - Fail-closed: exact dimension match, finite components, non-zero norms.
    ///   Non-isotropic metrics also refuse a metric-space norm <= 1e-12.
    pub fn evaluate(
        &self,
        state: &[f32],
        candidate_reps: &[&[f32]],
        action_frame: &LocalActionFrame<'_>,
    ) -> Result<ChoiceScores, ModelError> {
        let k = action_frame.len();
        if k == 0 {
            return Err(ModelError::EmptyOptions);
        }
        if candidate_reps.len() != k {
            return Err(ModelError::CandidateMismatch {
                expected: k,
                actual: candidate_reps.len(),
            });
        }
        let unit_state = self.unit(state, "state")?;
        let similarities = candidate_reps
            .iter()
            .map(|rep| {
                let unit = self.unit(rep, "candidate")?;
                let cos = gen_zero_core::dot_product_f32(&unit_state, &unit);
                if !cos.is_finite() {
                    return Err(ModelError::NumericalInstability(
                        "choice head cosine is non-finite".into(),
                    ));
                }
                Ok(cos.clamp(-1.0, 1.0))
            })
            .collect::<Result<Vec<f32>, ModelError>>()?;

        let ids = action_frame.actions();
        let mut canonical_order: Vec<usize> = (0..k).collect();
        canonical_order.sort_by_key(|&slot| ids[slot]);

        // Highest similarity wins; exact ties go to the smallest ActionId.
        let best_slot = canonical_order
            .iter()
            .copied()
            .reduce(|best, slot| {
                if similarities[slot] > similarities[best] {
                    slot
                } else {
                    best
                }
            })
            .ok_or(ModelError::EmptyOptions)?;

        // (s_i - s_max) / tau <= 0: no overflow, and the winner contributes exp(0) = 1.
        let max_sim = similarities[best_slot];
        let inv_temp = 1.0 / self.temperature;
        let exps: Vec<f32> = similarities
            .iter()
            .map(|&s| ((s - max_sim) * inv_temp).exp())
            .collect();
        let sum: f32 = canonical_order.iter().map(|&slot| exps[slot]).sum();
        if !sum.is_finite() || sum < 1.0 {
            return Err(ModelError::NumericalInstability(
                "choice head softmax mass is invalid".into(),
            ));
        }
        let probabilities: Vec<f32> = exps.iter().map(|e| e / sum).collect();
        let entropy = NormalizedEntropy::from_probabilities(&probabilities);

        Ok(ChoiceScores {
            selected: ids[best_slot],
            entropy,
            probabilities,
            similarities,
        })
    }

    /// Map one representation into metric space and unit-normalize it, or refuse it.
    fn unit(&self, rep: &[f32], role: &str) -> Result<Vec<f32>, ModelError> {
        match &self.metric {
            MetricKind::Isotropic => self.unit_euclidean(rep, role),
            MetricKind::DiagonalMahalanobis(precision) => {
                self.check_input(rep, role)?;
                // <u, v>_w equals the Euclidean dot of sqrt(w) * u and sqrt(w) * v.
                let mapped: Vec<f32> = rep
                    .iter()
                    .zip(precision)
                    .map(|(x, w)| x * w.sqrt())
                    .collect();
                self.unit_metric_space(&mapped, role)
            }
            MetricKind::WhitenedProjection { matrix, .. } => {
                self.check_input(rep, role)?;
                let mapped: Vec<f32> = matrix
                    .chunks_exact(self.dimension)
                    .map(|row| gen_zero_core::dot_product_f32(row, rep))
                    .collect();
                self.unit_metric_space(&mapped, role)
            }
        }
    }

    fn check_input(&self, rep: &[f32], role: &str) -> Result<(), ModelError> {
        if rep.len() != self.dimension {
            return Err(ModelError::Core(
                gen_zero_core::CoreError::DimensionMismatch {
                    expected: self.dimension,
                    actual: rep.len(),
                },
            ));
        }
        if rep.iter().any(|x| !x.is_finite()) {
            return Err(ModelError::NumericalInstability(format!(
                "{role} representation contains a non-finite component"
            )));
        }
        Ok(())
    }

    /// Normalize a vector already mapped into metric space. The mapping can
    /// overflow a finite input or collapse it (rank-deficient `W`), so both
    /// are checked here, after the transform.
    fn unit_metric_space(&self, mapped: &[f32], role: &str) -> Result<Vec<f32>, ModelError> {
        let metric = self.metric.name();
        if mapped.iter().any(|x| !x.is_finite()) {
            return Err(ModelError::NumericalInstability(format!(
                "{role} representation overflows in {metric} metric space"
            )));
        }
        let norm_sq = gen_zero_core::dot_product_f32(mapped, mapped);
        if !norm_sq.is_finite() {
            return Err(ModelError::NumericalInstability(format!(
                "{role} {metric} squared norm overflow"
            )));
        }
        let norm = norm_sq.sqrt();
        if norm <= METRIC_NORM_FLOOR {
            return Err(ModelError::NumericalInstability(format!(
                "{role} representation has zero norm in {metric} metric space; cosine is undefined"
            )));
        }
        let inv_norm = 1.0 / norm;
        Ok(mapped.iter().map(|x| x * inv_norm).collect())
    }

    /// Isotropic path. Kept byte-for-byte as before the metric work, including
    /// its `f32::MIN_POSITIVE` zero-norm floor, so isotropic outputs do not move.
    fn unit_euclidean(&self, rep: &[f32], role: &str) -> Result<Vec<f32>, ModelError> {
        if rep.len() != self.dimension {
            return Err(ModelError::Core(
                gen_zero_core::CoreError::DimensionMismatch {
                    expected: self.dimension,
                    actual: rep.len(),
                },
            ));
        }
        // Reject before normalizing: an infinite squared norm would make the
        // inverse norm zero and silently erase the direction of a finite input.
        if rep.iter().any(|x| !x.is_finite()) {
            return Err(ModelError::NumericalInstability(format!(
                "{role} representation contains a non-finite component"
            )));
        }
        let norm_sq = gen_zero_core::dot_product_f32(rep, rep);
        if !norm_sq.is_finite() {
            return Err(ModelError::NumericalInstability(format!(
                "{role} representation squared norm overflow"
            )));
        }
        if norm_sq <= f32::MIN_POSITIVE {
            return Err(ModelError::NumericalInstability(format!(
                "{role} representation has zero norm; cosine is undefined"
            )));
        }
        let inv_norm = 1.0 / norm_sq.sqrt();
        Ok(rep.iter().map(|x| x * inv_norm).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GATE_THRESHOLD: f32 = 0.65;

    fn basis(dim: usize, axis: usize) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[axis] = 1.0;
        v
    }

    fn eval(
        head: &ActionETFChoiceHead,
        state: &[f32],
        names: &[&str],
        ids: &[ActionId],
        reps: &[Vec<f32>],
    ) -> Result<ChoiceScores, ModelError> {
        let frame = LocalActionFrame::new(names, ids).unwrap();
        let refs: Vec<&[f32]> = reps.iter().map(Vec::as_slice).collect();
        head.evaluate(state, &refs, &frame)
    }

    #[test]
    fn rejects_nonfinite_overflowing_zero_and_mismatched_inputs() {
        let head = ActionETFChoiceHead::new(2, 1.0).unwrap();
        let names = ["a", "b"];
        let ids = [ActionId(1), ActionId(2)];
        let reps = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        for state in [
            [3e38, 0.0],
            [-3e38, 0.0],
            [1.4e19, 1.4e19],
            [f32::INFINITY, 0.0],
            [f32::NEG_INFINITY, 0.0],
            [f32::NAN, 0.0],
            [0.0, 0.0],
        ] {
            assert!(
                matches!(
                    eval(&head, &state, &names, &ids, &reps),
                    Err(ModelError::NumericalInstability(_))
                ),
                "{state:?}"
            );
        }
        let bad_candidate = vec![vec![1.0, 0.0], vec![f32::NAN, 1.0]];
        assert!(matches!(
            eval(&head, &[1.0, 0.0], &names, &ids, &bad_candidate),
            Err(ModelError::NumericalInstability(_))
        ));
        let zero_candidate = vec![vec![1.0, 0.0], vec![0.0, 0.0]];
        assert!(matches!(
            eval(&head, &[1.0, 0.0], &names, &ids, &zero_candidate),
            Err(ModelError::NumericalInstability(_))
        ));
        // Longer inputs are refused, not truncated.
        assert!(matches!(
            eval(&head, &[1.0, 0.0, 5.0], &names, &ids, &reps),
            Err(ModelError::Core(_))
        ));
        let wide_candidate = vec![vec![1.0, 0.0], vec![0.0, 1.0, 0.0]];
        assert!(matches!(
            eval(&head, &[1.0, 0.0], &names, &ids, &wide_candidate),
            Err(ModelError::Core(_))
        ));
        assert!(matches!(
            eval(&head, &[1.0, 0.0], &names, &ids, &reps[..1]),
            Err(ModelError::CandidateMismatch { .. })
        ));
        for t in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(ActionETFChoiceHead::new(2, t).is_err(), "{t}");
        }
        assert!(ActionETFChoiceHead::new(0, 1.0).is_err());

        let ordinary = eval(&head, &[1.0, 0.0], &names, &ids, &reps).unwrap();
        let large = eval(&head, &[1e19, 0.0], &names, &ids, &reps).unwrap();
        assert!(large.probabilities[0] > large.probabilities[1]);
        assert_eq!(ordinary.probabilities, large.probabilities);
    }

    /// Regression for the old defect: fixed ETF vertices at T=1 escalated every K >= 3.
    #[test]
    fn distinct_manifold_scores_sharpen_below_gate_for_every_frame_size() {
        let head = ActionETFChoiceHead::new(16, DEFAULT_UNCALIBRATED_TEMPERATURE).unwrap();
        let state = basis(16, 0);
        for k in 2..=16 {
            let names: Vec<String> = (0..k).map(|i| format!("c{i}")).collect();
            let names: Vec<&str> = names.iter().map(String::as_str).collect();
            let ids: Vec<ActionId> = (0..k as u32).map(|i| ActionId(100 - i)).collect();
            let reps: Vec<Vec<f32>> = (0..k).map(|i| basis(16, i)).collect();
            let out = eval(&head, &state, &names, &ids, &reps).unwrap();
            assert_eq!(out.selected, ids[0], "K={k}");
            assert!(
                out.entropy.value() < GATE_THRESHOLD,
                "K={k}: {}",
                out.entropy.value()
            );
        }
    }

    #[test]
    fn near_equal_manifold_scores_stay_above_gate() {
        let head = ActionETFChoiceHead::new(3, DEFAULT_UNCALIBRATED_TEMPERATURE).unwrap();
        let state = [1.0, 0.0, 0.0];
        // Cosines 0.50, 0.49, 0.48 against the state.
        let reps: Vec<Vec<f32>> = [0.50_f32, 0.49, 0.48]
            .iter()
            .map(|&c| vec![c, (1.0 - c * c).sqrt(), 0.0])
            .collect();
        let names = ["x", "y", "z"];
        let ids = [ActionId(7), ActionId(8), ActionId(9)];
        let out = eval(&head, &state, &names, &ids, &reps).unwrap();
        assert!((out.similarities[0] - 0.50).abs() < 1e-5);
        assert!(
            out.entropy.value() > GATE_THRESHOLD,
            "{}",
            out.entropy.value()
        );
    }

    #[test]
    fn score_follows_candidate_content_not_name() {
        let head = ActionETFChoiceHead::new(4, DEFAULT_UNCALIBRATED_TEMPERATURE).unwrap();
        let names = ["approve", "reject", "escalate"];
        let ids = [ActionId(101), ActionId(102), ActionId(103)];
        let reps = vec![basis(4, 0), basis(4, 1), basis(4, 2)];
        // Same names and ids, different states: the choice tracks the content.
        for (axis, expected) in [(0, ActionId(101)), (1, ActionId(102)), (2, ActionId(103))] {
            let out = eval(&head, &basis(4, axis), &names, &ids, &reps).unwrap();
            assert_eq!(out.selected, expected);
        }
        // Same state, swapped representations: the choice moves with the content.
        let swapped = vec![basis(4, 2), basis(4, 1), basis(4, 0)];
        let out = eval(&head, &basis(4, 0), &names, &ids, &swapped).unwrap();
        assert_eq!(out.selected, ActionId(103));
    }

    #[test]
    fn permutation_of_name_and_rep_pairs_is_bit_identical() {
        let head = ActionETFChoiceHead::new(3, 0.3).unwrap();
        let state = [0.9, 0.3, -0.2];
        let names = ["approve", "reject", "escalate"];
        let ids = [ActionId(101), ActionId(102), ActionId(103)];
        let reps = vec![
            vec![0.2, 0.9, 0.1],
            vec![1.0, 0.1, 0.0],
            vec![0.3, 0.3, 0.9],
        ];
        let fwd = eval(&head, &state, &names, &ids, &reps).unwrap();
        let perm = [2, 0, 1];
        let p_names: Vec<&str> = perm.iter().map(|&i| names[i]).collect();
        let p_ids: Vec<ActionId> = perm.iter().map(|&i| ids[i]).collect();
        let p_reps: Vec<Vec<f32>> = perm.iter().map(|&i| reps[i].clone()).collect();
        let rev = eval(&head, &state, &p_names, &p_ids, &p_reps).unwrap();
        assert_eq!(fwd.selected, ActionId(102));
        assert_eq!(rev.selected, fwd.selected);
        assert_eq!(rev.entropy.value().to_bits(), fwd.entropy.value().to_bits());
        for (slot, &orig) in perm.iter().enumerate() {
            assert_eq!(rev.probabilities[slot], fwd.probabilities[orig]);
            assert_eq!(rev.similarities[slot], fwd.similarities[orig]);
        }
    }

    #[test]
    fn exact_tie_breaks_to_smallest_action_id_in_any_order() {
        let head = ActionETFChoiceHead::new(2, 1.0).unwrap();
        let reps = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        let a = eval(
            &head,
            &[1.0, 0.0],
            &["p", "q"],
            &[ActionId(9), ActionId(3)],
            &reps,
        );
        let b = eval(
            &head,
            &[1.0, 0.0],
            &["q", "p"],
            &[ActionId(3), ActionId(9)],
            &reps,
        );
        assert_eq!(a.unwrap().selected, ActionId(3));
        let b = b.unwrap();
        assert_eq!(b.selected, ActionId(3));
        assert!((b.entropy.value() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn caller_temperature_controls_sharpness() {
        let names = ["a", "b", "c"];
        let ids = [ActionId(1), ActionId(2), ActionId(3)];
        let reps = vec![basis(3, 0), basis(3, 1), basis(3, 2)];
        let state = [1.0, 0.0, 0.0];
        let hot = ActionETFChoiceHead::new(3, 1.0).unwrap();
        let cold = ActionETFChoiceHead::new(3, 0.1).unwrap();
        let hot = eval(&hot, &state, &names, &ids, &reps).unwrap();
        let cold = eval(&cold, &state, &names, &ids, &reps).unwrap();
        // T=1 on a bounded cosine cannot clear the gate; that was the old failure mode.
        assert!(hot.entropy.value() > GATE_THRESHOLD);
        assert!(cold.entropy.value() < GATE_THRESHOLD);
        assert_eq!(hot.selected, cold.selected);
    }

    #[test]
    fn foundation_width_state_is_scored() {
        use gen_zero_core::FoundationLatent;
        let dim = FoundationLatent::default().values.len();
        let head = ActionETFChoiceHead::new(dim, DEFAULT_UNCALIBRATED_TEMPERATURE).unwrap();
        let names = ["commit", "rollback", "retry", "escalate"];
        let ids = [ActionId(1), ActionId(2), ActionId(3), ActionId(4)];
        let reps: Vec<Vec<f32>> = (0..4).map(|i| basis(dim, i * 7)).collect();
        let mut latent = Box::new(FoundationLatent::default());
        latent.values[7] = 50.0;
        latent.values[0] = 5.0;
        let out = eval(&head, latent.as_slice(), &names, &ids, &reps).unwrap();
        assert_eq!(out.selected, ActionId(2));
        assert!(out.entropy.value() < GATE_THRESHOLD);
    }

    fn golden_case() -> (Vec<f32>, Vec<Vec<f32>>) {
        (
            vec![0.7, 0.2, -0.1, 0.4],
            vec![
                vec![0.9, 0.3, -0.2, 0.1],
                vec![0.1, 0.8, 0.4, -0.3],
                vec![0.5, 0.5, 0.5, 0.5],
            ],
        )
    }

    /// Bits captured from the pre-metric head (commit 94b81b1). Isotropic
    /// scoring must not move by a single ulp.
    #[test]
    fn isotropic_output_is_bit_identical_to_pre_metric_head() {
        let head = ActionETFChoiceHead::new(4, DEFAULT_UNCALIBRATED_TEMPERATURE).unwrap();
        assert_eq!(head.metric(), &MetricKind::Isotropic);
        assert_eq!(head.metric_dimension(), 4);
        let (state, reps) = golden_case();
        let names = ["approve", "reject", "escalate"];
        let ids = [ActionId(101), ActionId(102), ActionId(103)];
        let out = eval(&head, &state, &names, &ids, &reps).unwrap();
        assert_eq!(out.selected, ActionId(101));
        assert_eq!(out.entropy.value().to_bits(), 0x3f26faeb);
        let p: Vec<u32> = out.probabilities.iter().map(|x| x.to_bits()).collect();
        let s: Vec<u32> = out.similarities.iter().map(|x| x.to_bits()).collect();
        assert_eq!(p, [0x3f2ce536, 0x3cc6cf91, 0x3e99c89b]);
        assert_eq!(s, [0x3f6b720b, 0x3db49dd8, 0x3f37964c]);

        // Unit precision is the Euclidean metric: sqrt(1) * x == x exactly.
        // (Only the zero-norm floor differs: 1e-12 here vs f32::MIN_POSITIVE.)
        let unit = ActionETFChoiceHead::with_diagonal_precision(
            4,
            DEFAULT_UNCALIBRATED_TEMPERATURE,
            &[1.0; 4],
        )
        .unwrap();
        let same = eval(&unit, &state, &names, &ids, &reps).unwrap();
        assert_eq!(same, out);
    }

    /// Synthetic adversarial frame: axes 0-2 carry the signal, axes 3-4 carry
    /// high-variance noise that the state shares with a wrong candidate.
    #[test]
    fn diagonal_precision_suppresses_noise_axes_and_sharpens_below_gate() {
        let state = [1.0, 0.0, 0.0, 5.0, 5.0];
        let names = ["right", "noisy_twin", "other"];
        let ids = [ActionId(11), ActionId(12), ActionId(13)];
        let reps = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 5.0, 5.0],
            vec![0.0, 0.0, 1.0, 5.0, -5.0],
        ];
        let iso = ActionETFChoiceHead::new(5, DEFAULT_UNCALIBRATED_TEMPERATURE).unwrap();
        let iso = eval(&iso, &state, &names, &ids, &reps).unwrap();
        // The isotropic cosine is captured by the shared noise.
        assert_eq!(iso.selected, ActionId(12));

        let precision = [1.0, 1.0, 1.0, 1e-4, 1e-4];
        let head = ActionETFChoiceHead::with_diagonal_precision(
            5,
            DEFAULT_UNCALIBRATED_TEMPERATURE,
            &precision,
        )
        .unwrap();
        assert_eq!(head.metric().name(), "diagonal_mahalanobis");
        let out = eval(&head, &state, &names, &ids, &reps).unwrap();
        assert_eq!(out.selected, ActionId(11));
        assert!(out.probabilities[0] > 0.9, "{:?}", out.probabilities);
        assert!(out.probabilities[0] > iso.probabilities[0]);
        assert!(
            out.entropy.value() < GATE_THRESHOLD,
            "{}",
            out.entropy.value()
        );

        // Closed form: <s, c0>_w / sqrt(<s, s>_w <c0, c0>_w) = 1 / sqrt(1.005).
        assert!((out.similarities[0] - 1.0 / 1.005_f32.sqrt()).abs() < 1e-5);
    }

    /// Scores `reps` against `state` and returns the similarities.
    fn sims(head: &ActionETFChoiceHead, state: &[f32], reps: &[Vec<f32>]) -> Vec<f32> {
        let names = ["a", "b", "c"];
        let ids = [ActionId(1), ActionId(2), ActionId(3)];
        eval(head, state, &names, &ids, reps).unwrap().similarities
    }

    fn apply(matrix: &[f32], v: &[f32]) -> Vec<f32> {
        matrix
            .chunks_exact(v.len())
            .map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum())
            .collect()
    }

    fn assert_close(a: &[f32], b: &[f32], eps: f32) {
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < eps, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn whitened_projection_preserves_angles_under_orthogonal_and_whitening_maps() {
        let z_state = vec![0.9, 0.3, -0.2];
        let z_reps = vec![
            vec![0.2, 0.9, 0.1],
            vec![1.0, 0.1, 0.0],
            vec![0.3, 0.3, 0.9],
        ];
        let iso = ActionETFChoiceHead::new(3, 0.3).unwrap();
        let reference = sims(&iso, &z_state, &z_reps);

        // Orthogonal W (3-4-5 rotation): angles are unchanged.
        let rotation = [0.6, -0.8, 0.0, 0.8, 0.6, 0.0, 0.0, 0.0, 1.0];
        let rotated = ActionETFChoiceHead::with_whitening_matrix(3, 3, 0.3, &rotation).unwrap();
        assert_eq!(rotated.metric().name(), "whitened");
        assert_close(&sims(&rotated, &z_state, &z_reps), &reference, 1e-6);

        // Anisotropic mixing x = A z distorts the Euclidean angles; W = A^-1
        // (exact in binary) restores them.
        let a = [4.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 8.0, 0.25];
        let a_inv = [0.25, 0.0, 0.0, -1.0, 2.0, 0.0, 32.0, -64.0, 4.0];
        let x_state = apply(&a, &z_state);
        let x_reps: Vec<Vec<f32>> = z_reps.iter().map(|z| apply(&a, z)).collect();
        let distorted = sims(&iso, &x_state, &x_reps);
        assert!(
            distorted
                .iter()
                .zip(&reference)
                .any(|(d, r)| (d - r).abs() > 0.1),
            "mixing should distort isotropic cosines: {distorted:?} vs {reference:?}"
        );
        let whitened = ActionETFChoiceHead::with_whitening_matrix(3, 3, 0.3, &a_inv).unwrap();
        assert_close(&sims(&whitened, &x_state, &x_reps), &reference, 1e-5);

        // A global scale on W cancels in the cosine.
        let scaled: Vec<f32> = a_inv.iter().map(|x| x * 7.0).collect();
        let scaled = ActionETFChoiceHead::with_whitening_matrix(3, 3, 0.3, &scaled).unwrap();
        assert_close(&sims(&scaled, &x_state, &x_reps), &reference, 1e-5);

        // Projection to fewer dimensions scores in the reduced space.
        let reduce = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let reduced = ActionETFChoiceHead::with_whitening_matrix(3, 2, 0.3, &reduce).unwrap();
        assert_eq!(reduced.dimension(), 3);
        assert_eq!(reduced.metric_dimension(), 2);
        let two_d = sims(&reduced, &z_state, &z_reps);
        let expected = sims(
            &ActionETFChoiceHead::new(2, 0.3).unwrap(),
            &z_state[..2],
            &z_reps.iter().map(|r| r[..2].to_vec()).collect::<Vec<_>>(),
        );
        assert_close(&two_d, &expected, 1e-6);
    }

    #[test]
    fn metric_heads_keep_permutation_equivariance_and_id_tie_break() {
        let heads = [
            ActionETFChoiceHead::with_diagonal_precision(3, 0.3, &[2.0, 0.5, 1.0]).unwrap(),
            ActionETFChoiceHead::with_whitening_matrix(
                3,
                3,
                0.3,
                &[0.25, 0.0, 0.0, -1.0, 2.0, 0.0, 32.0, -64.0, 4.0],
            )
            .unwrap(),
        ];
        let state = [0.9, 0.3, -0.2];
        let names = ["approve", "reject", "escalate"];
        let ids = [ActionId(101), ActionId(102), ActionId(103)];
        let reps = vec![
            vec![0.2, 0.9, 0.1],
            vec![1.0, 0.1, 0.0],
            vec![0.3, 0.3, 0.9],
        ];
        for head in &heads {
            let fwd = eval(head, &state, &names, &ids, &reps).unwrap();
            let perm = [2, 0, 1];
            let p_names: Vec<&str> = perm.iter().map(|&i| names[i]).collect();
            let p_ids: Vec<ActionId> = perm.iter().map(|&i| ids[i]).collect();
            let p_reps: Vec<Vec<f32>> = perm.iter().map(|&i| reps[i].clone()).collect();
            let rev = eval(head, &state, &p_names, &p_ids, &p_reps).unwrap();
            assert_eq!(rev.selected, fwd.selected);
            assert_eq!(rev.entropy.value().to_bits(), fwd.entropy.value().to_bits());
            for (slot, &orig) in perm.iter().enumerate() {
                assert_eq!(rev.probabilities[slot], fwd.probabilities[orig]);
                assert_eq!(rev.similarities[slot], fwd.similarities[orig]);
            }
            let tie = vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]];
            for (n, i) in [
                (["p", "q"], [ActionId(9), ActionId(3)]),
                (["q", "p"], [ActionId(3), ActionId(9)]),
            ] {
                let out = eval(head, &state, &n, &i, &tie).unwrap();
                assert_eq!(out.selected, ActionId(3));
            }
        }
    }

    #[test]
    fn metric_constructors_fail_closed_on_bad_parameters() {
        for bad in [0.0, -1.0, -0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                matches!(
                    ActionETFChoiceHead::with_diagonal_precision(3, 1.0, &[1.0, bad, 1.0]),
                    Err(ModelError::NumericalInstability(_))
                ),
                "{bad}"
            );
        }
        for len in [0, 2, 4] {
            assert!(matches!(
                ActionETFChoiceHead::with_diagonal_precision(3, 1.0, &vec![1.0; len]),
                Err(ModelError::Core(_))
            ));
        }
        assert!(ActionETFChoiceHead::with_diagonal_precision(0, 1.0, &[]).is_err());
        assert!(ActionETFChoiceHead::with_diagonal_precision(1, 0.0, &[1.0]).is_err());

        let eye = [1.0, 0.0, 0.0, 1.0];
        assert!(ActionETFChoiceHead::with_whitening_matrix(2, 2, 1.0, &eye).is_ok());
        assert!(ActionETFChoiceHead::with_whitening_matrix(0, 2, 1.0, &[]).is_err());
        assert!(ActionETFChoiceHead::with_whitening_matrix(2, 0, 1.0, &[]).is_err());
        assert!(ActionETFChoiceHead::with_whitening_matrix(2, 2, f32::NAN, &eye).is_err());
        assert!(ActionETFChoiceHead::with_whitening_matrix(usize::MAX, 2, 1.0, &eye).is_err());
        for len in [3, 5] {
            assert!(matches!(
                ActionETFChoiceHead::with_whitening_matrix(2, 2, 1.0, &vec![1.0; len]),
                Err(ModelError::Core(_))
            ));
        }
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                ActionETFChoiceHead::with_whitening_matrix(2, 2, 1.0, &[1.0, bad, 0.0, 1.0]),
                Err(ModelError::NumericalInstability(_))
            ));
        }
        assert!(matches!(
            ActionETFChoiceHead::with_whitening_matrix(2, 2, 1.0, &[1.0, 0.0, 0.0, -0.0]),
            Err(ModelError::NumericalInstability(_))
        ));
    }

    #[test]
    fn metric_evaluation_fails_closed_on_bad_inputs() {
        let names = ["a", "b"];
        let ids = [ActionId(1), ActionId(2)];
        let reps = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let diag = ActionETFChoiceHead::with_diagonal_precision(2, 1.0, &[1.0, 4.0]).unwrap();
        let huge = ActionETFChoiceHead::with_diagonal_precision(2, 1.0, &[f32::MAX, 1.0]).unwrap();
        // Rank one: [1, 1] collapses to zero.
        let collapse =
            ActionETFChoiceHead::with_whitening_matrix(2, 2, 1.0, &[1.0, -1.0, 2.0, -2.0]).unwrap();
        let instability = |r: Result<ChoiceScores, ModelError>| {
            matches!(r, Err(ModelError::NumericalInstability(_)))
        };
        for head in [&diag, &collapse] {
            for state in [
                [f32::NAN, 0.0],
                [f32::INFINITY, 0.0],
                [0.0, 0.0],
                [1e-13, 0.0],
            ] {
                assert!(
                    instability(eval(head, &state, &names, &ids, &reps)),
                    "{state:?}"
                );
            }
            let bad = vec![vec![1.0, 0.0], vec![0.0, f32::NEG_INFINITY]];
            assert!(instability(eval(head, &[1.0, 0.0], &names, &ids, &bad)));
            assert!(matches!(
                eval(head, &[1.0, 0.0, 0.0], &names, &ids, &reps),
                Err(ModelError::Core(_))
            ));
        }
        // Finite input, finite weight, overflowing metric-space vector.
        assert!(instability(eval(&huge, &[1e20, 0.0], &names, &ids, &reps)));
        // Nonzero input that a rank-deficient W maps to zero.
        assert!(instability(eval(
            &collapse,
            &[1.0, 1.0],
            &names,
            &ids,
            &reps
        )));
        let collapsing_candidate = vec![vec![1.0, 0.0], vec![3.0, 3.0]];
        assert!(instability(eval(
            &collapse,
            &[1.0, 0.0],
            &names,
            &ids,
            &collapsing_candidate
        )));
        // Just above the floor still scores.
        assert!(eval(&diag, &[1e-11, 0.0], &names, &ids, &reps).is_ok());
    }
}
