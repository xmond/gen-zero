//! Weighted chart fold: energy-scored relation composition over the learned table
//! in [`crate::semiring`].
//!
//! Every score here is an **energy**, `E = -ln p`, so lower is better and path
//! composition is addition. The two semirings differ only in how alternative
//! derivations of the same relation for one span are combined:
//!
//! * [`TropicalSemiring`]: `plus = min`. The span score is the energy of the
//!   single best derivation (Viterbi).
//! * [`LogProbSemiring`]: `plus = -ln(e^-a + e^-b)`. The span score is the
//!   energy of the summed probability over every derivation (marginal).
//!   Bracketings of one chain are not mutually exclusive events, so this sum is
//!   an unnormalized weight: it can exceed 1 and the energy can go below 0.
//!
//! Axiom energies come from caller-supplied counts ([`AxiomWeights::from_counts`]);
//! gen-zero carries no vote-tally machinery of its own, so nothing is learned
//! silently here.
//!
//! The [`RelationSemiring`] table (not [`AxiomWeights`]) is the *support*: a span
//! only considers a relation `r` at key `(a, b, gender)` if `r` is actually a
//! member of the learned table's entry for that key. If the table has an entry but
//! [`AxiomWeights`] has no energy for one of its members, that is a caller data-
//! consistency bug, not a "missing axiom" in the P(R) sense, so it refuses rather
//! than defaulting an energy (Hard rule: no default energy for missing data).

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::semiring::{
    Gender, RelId, RelationKey, RelationSemiring, BUDGET_EXCEEDED_REASON, DEFAULT_CHART_STEP_BUDGET,
};

/// A commutative semiring over energies (`-ln p`). `zero` absorbs under `times`
/// and is the identity of `plus`; `one` is the identity of `times`.
pub trait WeightedSemiring {
    const NAME: &'static str;
    fn zero() -> f64;
    fn one() -> f64;
    /// Combine two alternative derivations of the same relation.
    fn plus(a: f64, b: f64) -> f64;
    /// Chain two derivations (compose along a path).
    fn times(a: f64, b: f64) -> f64;
}

/// `(min, +)` over energies: the best single derivation wins (Viterbi).
#[derive(Debug, Clone, Copy, Default)]
pub struct TropicalSemiring;

impl WeightedSemiring for TropicalSemiring {
    const NAME: &'static str = "tropical";
    fn zero() -> f64 {
        f64::INFINITY
    }
    fn one() -> f64 {
        0.0
    }
    fn plus(a: f64, b: f64) -> f64 {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.min(b)
        }
    }
    fn times(a: f64, b: f64) -> f64 {
        a + b
    }
}

/// `(logsumexp, +)` over energies: derivations add as probabilities (marginal).
#[derive(Debug, Clone, Copy, Default)]
pub struct LogProbSemiring;

impl WeightedSemiring for LogProbSemiring {
    const NAME: &'static str = "logprob";
    fn zero() -> f64 {
        f64::INFINITY
    }
    fn one() -> f64 {
        0.0
    }
    fn plus(a: f64, b: f64) -> f64 {
        if a.is_nan() || b.is_nan() {
            return f64::NAN;
        }
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        if hi == f64::INFINITY {
            return lo;
        }
        // -ln(e^-lo + e^-hi) = lo - ln(1 + e^-(hi - lo)), stable for large gaps.
        lo - (-(hi - lo)).exp().ln_1p()
    }
    fn times(a: f64, b: f64) -> f64 {
        a + b
    }
}

/// Why an [`AxiomWeights`] could not be built from counts.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum AxiomWeightError {
    /// The pseudo-count must be finite and `>= 0`.
    #[error("pseudo-count must be finite and >= 0, got {0}")]
    InvalidPseudoCount(f64),
    /// A count of `0` would make `E = -ln(0 / denom) = +inf`; never silently
    /// dropped, since that would change which relations are reachable at `key`.
    #[error("zero count for relation {relation} at key {key:?}")]
    ZeroCount { key: RelationKey, relation: RelId },
    /// The same relation was listed twice for one key.
    #[error("relation {relation} listed twice at key {key:?}")]
    DuplicateRelation { key: RelationKey, relation: RelId },
    /// The same key was supplied more than once across the input.
    #[error("key {0:?} supplied more than once")]
    DuplicateKey(RelationKey),
    /// A key was supplied with an empty candidate list.
    #[error("key {0:?} has an empty candidate list")]
    EmptyKey(RelationKey),
}

/// The energy of every `(key, relation)` a weighted fold may use.
///
/// `E(r | key) = -ln(count_r / (sum of counts at key + pseudo_count))`. With
/// `pseudo_count = 0` this is the plain relative frequency; a positive
/// pseudo-count reserves probability mass for outcomes the caller did not list.
/// The pseudo-count is the caller's choice; nothing is defaulted here.
#[derive(Debug, Clone, PartialEq)]
pub struct AxiomWeights {
    energies: BTreeMap<RelationKey, BTreeMap<RelId, f64>>,
}

impl AxiomWeights {
    pub fn from_counts<I>(entries: I, pseudo_count: f64) -> Result<Self, AxiomWeightError>
    where
        I: IntoIterator<Item = (RelationKey, Vec<(RelId, u64)>)>,
    {
        if !pseudo_count.is_finite() || pseudo_count < 0.0 {
            return Err(AxiomWeightError::InvalidPseudoCount(pseudo_count));
        }
        let mut energies: BTreeMap<RelationKey, BTreeMap<RelId, f64>> = BTreeMap::new();
        for (key, counts) in entries {
            if energies.contains_key(&key) {
                return Err(AxiomWeightError::DuplicateKey(key));
            }
            if counts.is_empty() {
                return Err(AxiomWeightError::EmptyKey(key));
            }
            let mut seen: BTreeSet<RelId> = BTreeSet::new();
            let mut total: f64 = 0.0;
            for &(relation, count) in &counts {
                if !seen.insert(relation) {
                    return Err(AxiomWeightError::DuplicateRelation { key, relation });
                }
                if count == 0 {
                    return Err(AxiomWeightError::ZeroCount { key, relation });
                }
                total += count as f64;
            }
            let denom = total + pseudo_count;
            let per_rel: BTreeMap<RelId, f64> = counts
                .into_iter()
                .map(|(relation, count)| (relation, -((count as f64) / denom).ln()))
                .collect();
            energies.insert(key, per_rel);
        }
        Ok(Self { energies })
    }

    /// The energy of one `(key, relation)`, or `None` if it was never supplied.
    pub fn energy(&self, key: &RelationKey, relation: RelId) -> Option<f64> {
        self.energies.get(key)?.get(&relation).copied()
    }
}

/// One root candidate of a weighted chart fold.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WeightedCandidate {
    pub relation: RelId,
    pub energy: f64,
}

/// Outcome of [`RelationSemiring::weighted_chart_fold_chain`].
#[derive(Debug, Clone, PartialEq)]
pub enum WeightedFoldOutcome {
    Concluded {
        predicted: RelId,
        steps: usize,
        /// Post-order witness of the lowest-energy single derivation of
        /// `predicted`. Under [`LogProbSemiring`] this is one contributing
        /// derivation, not the whole marginal `energy` is the score of: `energy`
        /// is the semiring score, `proof_path` is the best single derivation.
        proof_path: Vec<RelId>,
        energy: f64,
        /// `E(top2) - E(top1)`; `+inf` when exactly one root candidate closes.
        margin: f64,
        /// `e^-E1 / sum_i e^-Ei` over the root candidates: a softmax of the root
        /// scores, not a calibrated probability of being correct.
        confidence: f64,
        /// Every root candidate, ascending energy (ties broken by relation id).
        candidates: Vec<WeightedCandidate>,
    },
    Refused {
        step_failed: usize,
        reason: String,
        /// Every root candidate that closed, if any (empty when nothing closed,
        /// when the chain shape was invalid, or when the budget was exceeded).
        candidates: Vec<WeightedCandidate>,
    },
}

/// One chart cell: the semiring score (combines every derivation via `S::plus`),
/// and, independent of `S`, the best single derivation's plain-sum energy and its
/// witness `(split, left, right)`. `best`/`witness` always use ordinary `+`/`min`
/// rather than `S::times`/`S::plus`, so a proof path exists even under
/// [`LogProbSemiring`], whose `score` is not attached to any one derivation.
#[derive(Debug, Clone, Copy)]
struct WeightedCell {
    score: f64,
    best: f64,
    witness: Option<(usize, RelId, RelId)>,
}

type WeightedSpan = BTreeMap<RelId, WeightedCell>;

impl RelationSemiring {
    /// Fold a chain over every bracketing at once, as [`Self::chart_fold_chain`]
    /// does, but scoring each candidate with `weights` under semiring `S` instead
    /// of only checking whether the table closes. See [`weighted_chart_fold_core`]
    /// for the exact semantics (support, budget guard, margin/confidence).
    pub fn weighted_chart_fold_chain<S: WeightedSemiring>(
        &self,
        weights: &AxiomWeights,
        edges: &[RelId],
        genders: &[Gender],
        margin_threshold: f64,
    ) -> WeightedFoldOutcome {
        weighted_chart_fold_core::<S>(
            self,
            weights,
            edges,
            genders,
            margin_threshold,
            DEFAULT_CHART_STEP_BUDGET,
        )
    }
}

/// Core of [`RelationSemiring::weighted_chart_fold_chain`], parameterized by an
/// explicit `budget` so tests can exercise the guard without a combinatorial
/// fixture. See the module doc for the support/no-default-energy discipline.
fn weighted_chart_fold_core<S: WeightedSemiring>(
    sem: &RelationSemiring,
    weights: &AxiomWeights,
    edges: &[RelId],
    genders: &[Gender],
    margin_threshold: f64,
    budget: usize,
) -> WeightedFoldOutcome {
    if edges.is_empty() || genders.len() != edges.len() + 1 {
        return WeightedFoldOutcome::Refused {
            step_failed: 0,
            reason: format!(
                "invalid chain: {} edges and {} node genders",
                edges.len(),
                genders.len()
            ),
            candidates: Vec::new(),
        };
    }
    if !margin_threshold.is_finite() || margin_threshold < 0.0 {
        return WeightedFoldOutcome::Refused {
            step_failed: 0,
            reason: format!(
                "invalid margin threshold: {margin_threshold} (must be finite and >= 0)"
            ),
            candidates: Vec::new(),
        };
    }

    let k = edges.len();
    if k > 64 {
        return WeightedFoldOutcome::Refused {
            step_failed: 0,
            reason: "chain length exceeds maximum supported bound (64)".into(),
            candidates: Vec::new(),
        };
    }
    let mut spans: Vec<Vec<WeightedSpan>> = vec![vec![WeightedSpan::new(); k + 1]; k + 1];
    for (i, &rel) in edges.iter().enumerate() {
        spans[i][i + 1].insert(
            rel,
            WeightedCell {
                score: S::one(),
                best: S::one(),
                witness: None,
            },
        );
    }

    let mut ops: usize = 0;
    for len in 2..=k {
        for i in 0..=k - len {
            let j = i + len;
            let mut out = WeightedSpan::new();
            #[allow(clippy::needless_range_loop)]
            for m in i + 1..j {
                for (&a, left) in &spans[i][m] {
                    for (&b, right) in &spans[m][j] {
                        let key = RelationKey::new(a, b, genders[j]);
                        ops += 1;
                        if ops > budget {
                            return WeightedFoldOutcome::Refused {
                                step_failed: j,
                                reason: BUDGET_EXCEEDED_REASON.into(),
                                candidates: Vec::new(),
                            };
                        }
                        let Some(set) = sem.entry(&key) else {
                            continue;
                        };
                        for r in set.iter() {
                            ops += 1;
                            if ops > budget {
                                return WeightedFoldOutcome::Refused {
                                    step_failed: j,
                                    reason: BUDGET_EXCEEDED_REASON.into(),
                                    candidates: Vec::new(),
                                };
                            }
                            let Some(axiom_energy) = weights.energy(&key, r) else {
                                return WeightedFoldOutcome::Refused {
                                    step_failed: j,
                                    reason: format!(
                                        "missing weighted energy for table result {key:?} -> {r:?}"
                                    ),
                                    candidates: Vec::new(),
                                };
                            };
                            let score = S::times(S::times(left.score, right.score), axiom_energy);
                            let best = left.best + right.best + axiom_energy;
                            match out.get_mut(&r) {
                                None => {
                                    out.insert(
                                        r,
                                        WeightedCell {
                                            score,
                                            best,
                                            witness: Some((m, a, b)),
                                        },
                                    );
                                }
                                Some(cell) => {
                                    cell.score = S::plus(cell.score, score);
                                    // Strictly lower only: ties keep the first witness found.
                                    if best < cell.best {
                                        cell.best = best;
                                        cell.witness = Some((m, a, b));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            spans[i][j] = out;
        }
    }

    let mut candidates: Vec<WeightedCandidate> = spans[0][k]
        .iter()
        .map(|(&relation, cell)| WeightedCandidate {
            relation,
            energy: cell.score,
        })
        .collect();
    candidates.sort_by(|a, b| {
        a.energy
            .total_cmp(&b.energy)
            .then_with(|| a.relation.cmp(&b.relation))
    });

    // Fail closed on any non-finite root energy rather than letting it silently
    // win or lose a margin comparison against NaN/inf.
    if let Some(bad) = candidates.iter().find(|c| !c.energy.is_finite()) {
        return WeightedFoldOutcome::Refused {
            step_failed: k,
            reason: format!(
                "root candidate energy is not finite ({} semiring): {:?} energy {}",
                S::NAME,
                bad.relation,
                bad.energy
            ),
            candidates,
        };
    }

    let Some(top) = candidates.first().cloned() else {
        return WeightedFoldOutcome::Refused {
            step_failed: k,
            reason: "no bracketing of the chain closes under the weighted table".into(),
            candidates,
        };
    };

    let margin = candidates
        .get(1)
        .map_or(f64::INFINITY, |second| second.energy - top.energy);
    // An exact tie never concludes, whatever the threshold: picking by relation
    // order would be a guess, not a judgment made by this module.
    if margin <= 0.0 || margin < margin_threshold {
        return WeightedFoldOutcome::Refused {
            step_failed: k,
            reason: format!(
                "bracketings disagree ({} semiring): margin {margin:.6} < threshold {margin_threshold}; candidates {:?}",
                S::NAME,
                candidates.iter().map(|c| (c.relation, c.energy)).collect::<Vec<_>>()
            ),
            candidates,
        };
    }

    let partition: f64 = candidates
        .iter()
        .map(|c| (top.energy - c.energy).exp())
        .sum();
    let mut proof_path = Vec::with_capacity(2 * k);
    collect_weighted_proof(&spans, 0, k, top.relation, &mut proof_path);
    WeightedFoldOutcome::Concluded {
        predicted: top.relation,
        steps: k - 1,
        proof_path,
        energy: top.energy,
        margin,
        confidence: 1.0 / partition,
        candidates,
    }
}

/// Post-order walk of the best single derivation tracked in `WeightedCell::best`/
/// `witness`, mirroring [`crate::semiring::collect_proof`] for the unweighted chart.
fn collect_weighted_proof(
    spans: &[Vec<WeightedSpan>],
    i: usize,
    j: usize,
    rel: RelId,
    out: &mut Vec<RelId>,
) {
    if let Some(WeightedCell {
        witness: Some((m, a, b)),
        ..
    }) = spans[i][j].get(&rel)
    {
        collect_weighted_proof(spans, i, *m, *a, out);
        collect_weighted_proof(spans, *m, j, *b, out);
    }
    out.push(rel);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semiring::ResultSet;

    const R_A: RelId = 100;
    const R_B: RelId = 101;
    const R_C: RelId = 102;
    const R_M: RelId = 110;
    const R_N: RelId = 111;
    const R_X: RelId = 120;
    const R_Y: RelId = 121;
    const R_W: RelId = 122;

    // -----------------------------------------------------------------
    // (a) Operator correctness.
    // -----------------------------------------------------------------

    #[test]
    fn logprob_plus_matches_probability_sum() {
        let a = -(0.25f64).ln();
        let b = -(0.5f64).ln();
        let combined = LogProbSemiring::plus(a, b);
        assert!(
            (combined - (-(0.75f64).ln())).abs() < 1e-12,
            "combined={combined}"
        );
        assert_eq!(LogProbSemiring::plus(a, LogProbSemiring::zero()), a);
    }

    #[test]
    fn tropical_plus_is_min_with_zero_identity() {
        assert_eq!(TropicalSemiring::plus(1.0, 2.0), 1.0);
        assert_eq!(TropicalSemiring::plus(1.0, TropicalSemiring::zero()), 1.0);
    }

    #[test]
    fn nan_propagates_through_plus_in_both_semirings() {
        assert!(TropicalSemiring::plus(f64::NAN, 1.0).is_nan());
        assert!(LogProbSemiring::plus(f64::NAN, 1.0).is_nan());
    }

    // -----------------------------------------------------------------
    // (b) AxiomWeights::from_counts.
    // -----------------------------------------------------------------

    #[test]
    fn from_counts_computes_negative_log_frequency() {
        let key = RelationKey::new(R_A, R_B, Gender::Male);
        let weights = AxiomWeights::from_counts([(key, vec![(R_X, 9), (R_Y, 1)])], 0.0).unwrap();
        let e_x = weights.energy(&key, R_X).unwrap();
        let e_y = weights.energy(&key, R_Y).unwrap();
        assert!((e_x - -(9.0f64 / 10.0).ln()).abs() < 1e-12, "e_x={e_x}");
        assert!((e_y - -(1.0f64 / 10.0).ln()).abs() < 1e-12, "e_y={e_y}");
    }

    #[test]
    fn from_counts_rejects_invalid_pseudo_count() {
        let key = RelationKey::new(R_A, R_B, Gender::Male);
        assert_eq!(
            AxiomWeights::from_counts([(key, vec![(R_X, 1)])], -1.0),
            Err(AxiomWeightError::InvalidPseudoCount(-1.0))
        );
        assert!(matches!(
            AxiomWeights::from_counts([(key, vec![(R_X, 1)])], f64::NAN),
            Err(AxiomWeightError::InvalidPseudoCount(_))
        ));
    }

    #[test]
    fn from_counts_rejects_zero_count() {
        let key = RelationKey::new(R_A, R_B, Gender::Male);
        assert_eq!(
            AxiomWeights::from_counts([(key, vec![(R_X, 0)])], 0.0),
            Err(AxiomWeightError::ZeroCount { key, relation: R_X })
        );
    }

    #[test]
    fn from_counts_rejects_duplicate_relation_and_duplicate_key() {
        let key = RelationKey::new(R_A, R_B, Gender::Male);
        assert_eq!(
            AxiomWeights::from_counts([(key, vec![(R_X, 1), (R_X, 2)])], 0.0),
            Err(AxiomWeightError::DuplicateRelation { key, relation: R_X })
        );
        assert_eq!(
            AxiomWeights::from_counts([(key, vec![(R_X, 1)]), (key, vec![(R_Y, 1)])], 0.0),
            Err(AxiomWeightError::DuplicateKey(key))
        );
    }

    #[test]
    fn from_counts_rejects_an_empty_key() {
        let key = RelationKey::new(R_A, R_B, Gender::Male);
        assert_eq!(
            AxiomWeights::from_counts([(key, vec![])], 0.0),
            Err(AxiomWeightError::EmptyKey(key))
        );
    }

    // -----------------------------------------------------------------
    // (c)/(d) Margin: resolving and refusing a false ambiguity.
    // -----------------------------------------------------------------

    fn conflict_key_fixture() -> (RelationSemiring, [RelId; 2], [Gender; 3]) {
        let mut sem = RelationSemiring::new();
        let edges = [10u16, 11u16];
        let genders = [Gender::Male, Gender::Male, Gender::Male];
        sem.insert_axiom(
            RelationKey::new(10, 11, Gender::Male),
            ResultSet::from_ids([14u16, 15u16]),
        );
        (sem, edges, genders)
    }

    #[test]
    fn unweighted_chart_refuses_the_conflict_key() {
        let (sem, edges, genders) = conflict_key_fixture();
        let outcome = sem.chart_fold_chain(&edges, &genders);
        match outcome {
            crate::semiring::FoldOutcome::Refused { reason, .. } => {
                assert!(reason.contains("bracketings disagree"), "reason={reason}");
            }
            other => panic!("expected the unweighted chart to refuse, got {other:?}"),
        }
    }

    #[test]
    fn margin_resolves_a_false_ambiguity() {
        let (sem, edges, genders) = conflict_key_fixture();
        let key = RelationKey::new(10, 11, Gender::Male);
        let weights = AxiomWeights::from_counts([(key, vec![(14, 9), (15, 1)])], 0.0).unwrap();

        match sem.weighted_chart_fold_chain::<TropicalSemiring>(&weights, &edges, &genders, 0.5) {
            WeightedFoldOutcome::Concluded {
                predicted,
                margin,
                confidence,
                ..
            } => {
                assert_eq!(predicted, 14);
                assert!((margin - 9.0f64.ln()).abs() < 1e-9, "margin={margin}");
                assert!((confidence - 0.9).abs() < 1e-9, "confidence={confidence}");
            }
            other => panic!("expected the weighted fold to conclude 14, got {other:?}"),
        }
    }

    #[test]
    fn margin_above_threshold_refuses_with_all_candidates() {
        let (sem, edges, genders) = conflict_key_fixture();
        let key = RelationKey::new(10, 11, Gender::Male);
        let weights = AxiomWeights::from_counts([(key, vec![(14, 9), (15, 1)])], 0.0).unwrap();

        match sem.weighted_chart_fold_chain::<TropicalSemiring>(&weights, &edges, &genders, 5.0) {
            WeightedFoldOutcome::Refused {
                reason, candidates, ..
            } => {
                assert!(reason.contains("margin"), "reason={reason}");
                let relations: BTreeSet<RelId> = candidates.iter().map(|c| c.relation).collect();
                assert_eq!(relations, BTreeSet::from([14, 15]));
            }
            other => panic!("expected a margin refusal, got {other:?}"),
        }
    }

    #[test]
    fn an_exact_tie_refuses_even_at_zero_threshold() {
        let (sem, edges, genders) = conflict_key_fixture();
        let key = RelationKey::new(10, 11, Gender::Male);
        let weights = AxiomWeights::from_counts([(key, vec![(14, 5), (15, 5)])], 0.0).unwrap();

        match sem.weighted_chart_fold_chain::<TropicalSemiring>(&weights, &edges, &genders, 0.0) {
            WeightedFoldOutcome::Refused { reason, .. } => {
                assert!(reason.contains("margin"), "reason={reason}");
            }
            other => panic!("expected an exact tie to refuse, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // (e) Tropical (Viterbi) and LogProb (marginal) disagree by design.
    // -----------------------------------------------------------------

    #[test]
    fn tropical_and_logprob_can_pick_different_relations() {
        // A(B C) vs (A B)C: `X` closes via BOTH bracketings (two mediocre
        // derivations); `Y`/`W` each close via only one bracketing, alongside `X`,
        // as the table's other conflict-key member.
        let mut sem = RelationSemiring::new();
        sem.insert_axiom(
            RelationKey::new(R_A, R_B, Gender::Female),
            ResultSet::single(R_M),
        );
        sem.insert_axiom(
            RelationKey::new(R_B, R_C, Gender::Male),
            ResultSet::single(R_N),
        );
        sem.insert_axiom(
            RelationKey::new(R_A, R_N, Gender::Male),
            ResultSet::from_ids([R_X, R_Y]),
        );
        sem.insert_axiom(
            RelationKey::new(R_M, R_C, Gender::Male),
            ResultSet::from_ids([R_X, R_W]),
        );

        let weights = AxiomWeights::from_counts(
            [
                (RelationKey::new(R_A, R_B, Gender::Female), vec![(R_M, 1)]),
                (RelationKey::new(R_B, R_C, Gender::Male), vec![(R_N, 1)]),
                (
                    RelationKey::new(R_A, R_N, Gender::Male),
                    vec![(R_X, 2), (R_Y, 3)],
                ),
                (
                    RelationKey::new(R_M, R_C, Gender::Male),
                    vec![(R_X, 9), (R_W, 11)],
                ),
            ],
            0.0,
        )
        .unwrap();

        let edges = [R_A, R_B, R_C];
        let genders = [Gender::Male, Gender::Male, Gender::Female, Gender::Male];

        match sem.weighted_chart_fold_chain::<LogProbSemiring>(&weights, &edges, &genders, 0.0) {
            WeightedFoldOutcome::Concluded { predicted, .. } => assert_eq!(predicted, R_X),
            other => panic!("expected logprob to pick X, got {other:?}"),
        }
        match sem.weighted_chart_fold_chain::<TropicalSemiring>(&weights, &edges, &genders, 0.0) {
            WeightedFoldOutcome::Concluded { predicted, .. } => assert_eq!(predicted, R_Y),
            other => panic!("expected tropical to pick Y, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // (f) A table result with no matching weight refuses, never defaults.
    // -----------------------------------------------------------------

    #[test]
    fn missing_energy_for_a_table_result_refuses() {
        let mut sem = RelationSemiring::new();
        let key = RelationKey::new(R_A, R_B, Gender::Male);
        sem.insert_axiom(key, ResultSet::single(R_X));
        // No AxiomWeights entry at all: the table has a result but no energy.
        let weights = AxiomWeights::from_counts(std::iter::empty(), 0.0).unwrap();

        let edges = [R_A, R_B];
        let genders = [Gender::Male, Gender::Male, Gender::Male];
        match sem.weighted_chart_fold_chain::<TropicalSemiring>(&weights, &edges, &genders, 0.0) {
            WeightedFoldOutcome::Refused {
                reason, candidates, ..
            } => {
                assert!(
                    reason.contains("missing weighted energy"),
                    "reason={reason}"
                );
                assert!(candidates.is_empty());
            }
            other => panic!("expected a missing-energy refusal, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // (g) Dense bomb under the budget guard.
    // -----------------------------------------------------------------

    #[test]
    fn dense_bomb_weighted_fold_refuses_on_budget() {
        let n: RelId = 64;
        let all: Vec<RelId> = (0..n).collect();
        let mut sem = RelationSemiring::new();
        let mut entries: Vec<(RelationKey, Vec<(RelId, u64)>)> = Vec::new();
        for &r1 in &all {
            for &r2 in &all {
                let key = RelationKey::new(r1, r2, Gender::Male);
                sem.insert_axiom(key, ResultSet::from_ids(all.iter().copied()));
                entries.push((key, all.iter().map(|&r| (r, 1u64)).collect()));
            }
        }
        let weights = AxiomWeights::from_counts(entries, 0.0).unwrap();
        let edges: Vec<RelId> = (0..n).collect();
        let genders = vec![Gender::Male; (n as usize) + 1];

        let start = std::time::Instant::now();
        let outcome =
            sem.weighted_chart_fold_chain::<TropicalSemiring>(&weights, &edges, &genders, 0.0);
        let elapsed = start.elapsed();
        eprintln!("dense_bomb_weighted_fold_refuses_on_budget: elapsed={elapsed:?}");

        match outcome {
            WeightedFoldOutcome::Refused {
                reason, candidates, ..
            } => {
                assert_eq!(reason, BUDGET_EXCEEDED_REASON);
                assert!(candidates.is_empty());
            }
            other => panic!("expected a budget refusal, got {other:?}"),
        }
        assert!(elapsed.as_secs_f64() < 2.0, "took too long: {elapsed:?}");
    }

    // -----------------------------------------------------------------
    // (h) An invalid margin threshold refuses before any table work.
    // -----------------------------------------------------------------

    #[test]
    fn invalid_margin_threshold_refuses_at_step_zero() {
        let mut sem = RelationSemiring::new();
        sem.insert_axiom(
            RelationKey::new(R_A, R_A, Gender::Male),
            ResultSet::single(R_X),
        );
        let weights = AxiomWeights::from_counts(std::iter::empty(), 0.0).unwrap();
        let edges = [R_A];
        let genders = [Gender::Male, Gender::Male];

        for bad in [f64::NAN, -1.0, f64::INFINITY] {
            match sem.weighted_chart_fold_chain::<TropicalSemiring>(&weights, &edges, &genders, bad)
            {
                WeightedFoldOutcome::Refused {
                    step_failed,
                    candidates,
                    ..
                } => {
                    assert_eq!(step_failed, 0);
                    assert!(candidates.is_empty());
                }
                other => panic!("expected threshold {bad} to refuse, got {other:?}"),
            }
        }
    }
}
