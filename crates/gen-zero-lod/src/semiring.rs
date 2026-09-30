//! Learned relation semiring (Spec 24 §8.6.2, Definition 8.6.1).
//!
//! `P(R)` under `⊕ = ∪` (idempotent, zero element `∅`) is the additive structure.
//! `⊗_g : P(R) x P(R) -> P(R)` is a *partial* operation lifted pointwise from a learned
//! table `T : R x R x G ⇀ P(R)`: a missing `(r1, r2, g)` entry contributes `∅`, not an
//! error. Associativity of `⊗_g` is not assumed; [`RelationSemiring::associativity_audit`]
//! measures how often it actually holds on the table's own entries, matching the Gen-2
//! "460/0" audit discipline the spec cites rather than asserting it in the type system.
//!
//! The relation vocabulary (`RelId`) is an opaque integer: this module carries no
//! domain-specific (e.g. kinship, English-language) data. Callers supply their own axioms.
//!
//! Next to the discrete table there is a soft one: [`SoftResultSet`] is a
//! probability distribution over relations, and `T_soft(r1, r2, g)` is one per
//! key ([`RelationSemiring::insert_soft_axiom`]). The soft product
//! `(a ⊗_g b)(r) = Σ a(r1) b(r2) T_soft(r1, r2, g → r)` is multilinear in `a`
//! and `b`. Mass whose pair has no kernel is kept as an explicit *unclosed*
//! bucket, never renormalized away, and counts toward the entropy that
//! [`RelationSemiring::fold_chain_soft`] gates on.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// Opaque relation identifier into the caller's own finite relation vocabulary.
pub type RelId = u16;

/// Context the composition table is keyed on (`g` in the spec, e.g. an endpoint
/// attribute). Kept as a small closed enum rather than a string so the table stays a
/// compact array-indexable key; the three variants carry no domain semantics here.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Gender {
    Male = 0,
    Female = 1,
    Unknown = 2,
}

impl Gender {
    pub const ALL: [Gender; 3] = [Gender::Male, Gender::Female, Gender::Unknown];
}

/// `(r1, r2, gender)` lookup key into the learned partial table `T`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RelationKey {
    pub r1: RelId,
    pub r2: RelId,
    pub gender: Gender,
}

impl RelationKey {
    #[inline]
    pub fn new(r1: RelId, r2: RelId, gender: Gender) -> Self {
        Self { r1, r2, gender }
    }
}

/// An element of `P(R)`: strictly distinguishes "no derivation" from "exactly one
/// derivation" from "multiple, conflicting derivations", so conflict keys can be
/// recorded and reported rather than silently resolved inside this module.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResultSet {
    #[default]
    Empty,
    Single(RelId),
    /// Two or more distinct relations reachable for the same key: a conflict key.
    Multi(BTreeSet<RelId>),
}

impl ResultSet {
    #[inline]
    pub fn single(r: RelId) -> Self {
        Self::Single(r)
    }

    /// Normalize an arbitrary collection of relation ids into the canonical `Empty` /
    /// `Single` / `Multi` form (deduplicated, since `P(R)` is a set, not a multiset).
    pub fn from_ids<I: IntoIterator<Item = RelId>>(ids: I) -> Self {
        let set: BTreeSet<RelId> = ids.into_iter().collect();
        Self::from_set(set)
    }

    fn from_set(set: BTreeSet<RelId>) -> Self {
        match set.len() {
            0 => Self::Empty,
            1 => Self::Single(*set.iter().next().unwrap()),
            _ => Self::Multi(set),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// A conflict key: two or more distinct relations derived for one context.
    #[inline]
    pub fn is_conflict(&self) -> bool {
        matches!(self, Self::Multi(_))
    }

    pub fn iter(&self) -> Box<dyn Iterator<Item = RelId> + '_> {
        match self {
            Self::Empty => Box::new(std::iter::empty()),
            Self::Single(r) => Box::new(std::iter::once(*r)),
            Self::Multi(set) => Box::new(set.iter().copied()),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Single(_) => 1,
            Self::Multi(set) => set.len(),
        }
    }

    /// `⊕ = ∪`: idempotent union, zero element `∅`.
    pub fn union(&self, other: &Self) -> Self {
        if self == other {
            return self.clone(); // idempotence, no allocation on the common path
        }
        let set: BTreeSet<RelId> = self.iter().chain(other.iter()).collect();
        Self::from_set(set)
    }
}

/// One associativity check that failed: `(r1 ⊗ r2) ⊗ r3 != r1 ⊗ (r2 ⊗ r3)` under `gender`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssociativityViolation {
    pub r1: RelId,
    pub r2: RelId,
    pub r3: RelId,
    pub gender: Gender,
    pub left: ResultSet,
    pub right: ResultSet,
}

/// Result of a full associativity self-audit: how many triples were checked and which
/// ones (if any) violated associativity. Deliberately a count + list, not a bool, so
/// callers can report evidence ("460/0") instead of a single pass/fail bit.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AssociativityReport {
    pub checked: u64,
    pub violations: Vec<AssociativityViolation>,
}

impl AssociativityReport {
    #[inline]
    pub fn is_associative(&self) -> bool {
        self.violations.is_empty()
    }

    #[inline]
    pub fn violation_count(&self) -> usize {
        self.violations.len()
    }
}

/// Outcome of folding a chain of relations down to a single predicted relation.
/// `Refused` is not an error: a chain that does not close under the learned table is
/// an expected, reportable outcome, not a bug, so the reason is carried as data
/// rather than surfaced through `Result`/`panic!`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FoldOutcome {
    Concluded {
        predicted: RelId,
        steps: usize,
        proof_path: Vec<RelId>,
    },
    /// A soft fold ([`RelationSemiring::fold_chain_soft`]) that passed its
    /// entropy gate. `probability` is the mass of `predicted` in the folded
    /// distribution: a model probability, not a calibrated chance of being right.
    SoftConcluded {
        predicted: RelId,
        probability: f32,
        /// `H(p)` in nats of the folded distribution, unclosed bucket included.
        entropy: f32,
        steps: usize,
        /// Every relation with mass, highest first (ties by relation id).
        distribution: Vec<(RelId, f32)>,
        /// Mass that no kernel closed.
        unclosed_mass: f32,
        /// Entropy after each fold step, starting with the first leaf.
        step_entropies: Vec<f32>,
    },
    Refused {
        step_failed: usize,
        reason: String,
    },
}

/// Cap on the number of table-lookup + candidate-insertion operations a chart fold may
/// perform: the CYK chart is cubic in chain length and, for a dense/conflicted table,
/// each cell can hold up to `|R|` candidates, so an adversarial or merely large table
/// can blow up combinatorially. The budget makes that fail closed (a `Refused`) instead
/// of hanging or exhausting memory.
pub const DEFAULT_CHART_STEP_BUDGET: usize = 100_000;

/// Reason string used by every chart fold (unweighted and weighted) when
/// [`DEFAULT_CHART_STEP_BUDGET`] (or a caller-supplied budget) is exceeded.
pub const BUDGET_EXCEEDED_REASON: &str =
    "causal fold operation budget exceeded (potential combinatorial explosion)";

/// Largest deviation from 1 of the total mass of a caller-supplied distribution.
pub const SOFT_MASS_TOLERANCE: f64 = 1e-4;

/// Two top probabilities closer than this are a tie: the argmax is undefined.
pub const SOFT_TIE_TOLERANCE: f64 = 1e-6;

/// Why a [`SoftResultSet`] was refused.
#[derive(thiserror::Error, Debug, Clone, PartialEq)]
pub enum SoftSetError {
    #[error("a soft result set needs at least one relation")]
    Empty,
    #[error("relation {relation} has probability {p}; each must be finite and in (0, 1]")]
    Probability { relation: RelId, p: f64 },
    #[error("relation {0} is listed twice")]
    Duplicate(RelId),
    #[error("relation {relation} has count 0")]
    ZeroCount { relation: RelId },
    #[error("pseudo-count must be finite and >= 0, got {0}")]
    PseudoCount(f64),
    #[error("probabilities sum to {total}, not 1 within {SOFT_MASS_TOLERANCE}")]
    Mass { total: f64 },
}

/// A probability distribution over relations: the soft counterpart of
/// [`ResultSet`]. `probs` holds only positive masses. `unclosed` is the mass no
/// relation received (a composition with no kernel, or a pseudo-count), so
/// `sum(probs) + unclosed = 1`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SoftResultSet {
    probs: BTreeMap<RelId, f32>,
    unclosed: f32,
}

impl SoftResultSet {
    /// A distribution from `(relation, probability)` pairs. Each probability
    /// must be finite and in `(0, 1]`, no relation may repeat, and the total must
    /// be 1 within [`SOFT_MASS_TOLERANCE`]; the pairs are then rescaled to sum to 1.
    pub fn from_probs<I: IntoIterator<Item = (RelId, f32)>>(
        entries: I,
    ) -> Result<Self, SoftSetError> {
        let mut raw: BTreeMap<RelId, f64> = BTreeMap::new();
        for (relation, p) in entries {
            let p = f64::from(p);
            if !(p.is_finite() && p > 0.0 && p <= 1.0) {
                return Err(SoftSetError::Probability { relation, p });
            }
            if raw.insert(relation, p).is_some() {
                return Err(SoftSetError::Duplicate(relation));
            }
        }
        if raw.is_empty() {
            return Err(SoftSetError::Empty);
        }
        let total: f64 = raw.values().sum();
        if (total - 1.0).abs() > SOFT_MASS_TOLERANCE {
            return Err(SoftSetError::Mass { total });
        }
        Ok(Self::from_masses(
            raw.into_iter().map(|(r, p)| (r, p / total)).collect(),
        ))
    }

    /// Relative frequencies `count / (total + pseudo_count)`. A positive
    /// pseudo-count leaves `pseudo_count / (total + pseudo_count)` unclosed: mass
    /// for outcomes the counts never saw.
    pub fn from_counts(counts: &[(RelId, u64)], pseudo_count: f64) -> Result<Self, SoftSetError> {
        if !(pseudo_count.is_finite() && pseudo_count >= 0.0) {
            return Err(SoftSetError::PseudoCount(pseudo_count));
        }
        if counts.is_empty() {
            return Err(SoftSetError::Empty);
        }
        let mut seen = BTreeSet::new();
        let mut total = 0.0_f64;
        for &(relation, count) in counts {
            if !seen.insert(relation) {
                return Err(SoftSetError::Duplicate(relation));
            }
            if count == 0 {
                return Err(SoftSetError::ZeroCount { relation });
            }
            total += count as f64;
        }
        let denom = total + pseudo_count;
        Ok(Self::from_masses(
            counts.iter().map(|&(r, c)| (r, c as f64 / denom)).collect(),
        ))
    }

    /// All mass on one relation.
    pub fn delta(relation: RelId) -> Self {
        Self {
            probs: BTreeMap::from([(relation, 1.0)]),
            unclosed: 0.0,
        }
    }

    /// Build from nonnegative f64 masses summing to at most 1; the rest is unclosed.
    fn from_masses(masses: BTreeMap<RelId, f64>) -> Self {
        let closed: f64 = masses.values().sum();
        Self {
            probs: masses
                .into_iter()
                .filter(|&(_, p)| p > 0.0)
                .map(|(r, p)| (r, p as f32))
                .collect(),
            unclosed: (1.0 - closed).max(0.0) as f32,
        }
    }

    /// Mass of `relation`, 0 when it has none.
    pub fn probability(&self, relation: RelId) -> f32 {
        self.probs.get(&relation).copied().unwrap_or(0.0)
    }

    pub fn iter(&self) -> impl Iterator<Item = (RelId, f32)> + '_ {
        self.probs.iter().map(|(&r, &p)| (r, p))
    }

    /// Number of relations with mass.
    pub fn len(&self) -> usize {
        self.probs.len()
    }

    /// No relation has mass: everything is unclosed.
    pub fn is_empty(&self) -> bool {
        self.probs.is_empty()
    }

    pub fn unclosed_mass(&self) -> f32 {
        self.unclosed
    }

    /// The relations with mass, as a discrete [`ResultSet`].
    pub fn support(&self) -> ResultSet {
        ResultSet::from_ids(self.probs.keys().copied())
    }

    /// `H(p) = -Σ p ln p` in nats over every relation and the unclosed bucket.
    /// 0 for a delta, `ln n` for a uniform distribution over `n` outcomes.
    pub fn entropy(&self) -> f32 {
        let h: f64 = self
            .probs
            .values()
            .copied()
            .chain(std::iter::once(self.unclosed))
            .map(f64::from)
            .filter(|&p| p > 0.0)
            .map(|p| -p * p.ln())
            .sum();
        h.max(0.0) as f32
    }

    /// Relations by mass, highest first; equal masses by relation id.
    pub fn ranked(&self) -> Vec<(RelId, f32)> {
        let mut ranked: Vec<(RelId, f32)> = self.iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        ranked
    }
}

/// The learned relation semiring: a compact `BTreeMap`-backed lookup table for `T`, plus
/// the set of keys recorded as conflicted (multi-valued), plus the soft table
/// `T_soft` of the soft product.
#[derive(Clone, Debug, Default)]
pub struct RelationSemiring {
    table: BTreeMap<RelationKey, ResultSet>,
    conflict_keys: BTreeSet<RelationKey>,
    soft_table: BTreeMap<RelationKey, SoftResultSet>,
}

impl RelationSemiring {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert one axiom `T(key) = result`. A multi-valued `result` marks `key` as a
    /// conflict key; overwriting a previously-conflicted key with a determined result
    /// clears that flag (axioms can be corrected, not just accumulated).
    pub fn insert_axiom(&mut self, key: RelationKey, result: ResultSet) {
        if result.is_conflict() {
            self.conflict_keys.insert(key);
        } else {
            self.conflict_keys.remove(&key);
        }
        self.table.insert(key, result);
    }

    /// Number of axiom entries currently in the table.
    #[inline]
    pub fn len(&self) -> usize {
        self.table.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Direct table lookup `T(r1, r2, g)`; a missing entry returns `∅` (definition
    /// 8.6.1: "a missing key makes this bracketed contribution ∅").
    pub fn lookup(&self, key: &RelationKey) -> ResultSet {
        self.table.get(key).cloned().unwrap_or_default()
    }

    /// Borrowing counterpart to [`Self::lookup`] for hot inner loops (the chart folds):
    /// a missing entry is `None` here rather than a cloned `ResultSet::Empty`, so a
    /// budget-guarded scan over many candidate keys does not pay an allocation per
    /// lookup just to test for absence or iterate a conflict key's members.
    pub(crate) fn entry(&self, key: &RelationKey) -> Option<&ResultSet> {
        self.table.get(key)
    }

    /// Insert one soft axiom `T_soft(key) = kernel`, replacing an earlier one.
    pub fn insert_soft_axiom(&mut self, key: RelationKey, kernel: SoftResultSet) {
        self.soft_table.insert(key, kernel);
    }

    /// Number of soft axioms.
    pub fn soft_len(&self) -> usize {
        self.soft_table.len()
    }

    /// The soft kernel of `key`: its soft axiom, else a `Single` discrete axiom
    /// as a delta. `Ok(None)` when the key has neither (the pair's mass stays
    /// unclosed). Refused: a conflict key with no soft axiom (its candidates
    /// carry no weights, and this module never invents them), and a soft axiom
    /// whose support differs from the discrete axiom of the same key.
    fn soft_kernel(&self, key: &RelationKey) -> Result<Option<Cow<'_, SoftResultSet>>, String> {
        match (self.soft_table.get(key), self.table.get(key)) {
            (Some(soft), Some(hard)) if soft.support() != *hard && !hard.is_empty() => {
                Err(format!(
                    "soft axiom {key:?} has support {:?} but the discrete axiom says {hard:?}",
                    soft.support()
                ))
            }
            (Some(soft), _) => Ok(Some(Cow::Borrowed(soft))),
            (None, Some(ResultSet::Single(r))) => Ok(Some(Cow::Owned(SoftResultSet::delta(*r)))),
            (None, Some(ResultSet::Multi(candidates))) => Err(format!(
                "conflict key {key:?} has candidates {candidates:?} but no soft axiom weighs them"
            )),
            (None, Some(ResultSet::Empty) | None) => Ok(None),
        }
    }

    /// The soft product `(a ⊗_g b)(r) = Σ_{r1, r2} a(r1) b(r2) T_soft(r1, r2, g → r)`.
    /// Mass on a pair without a kernel, and the unclosed mass of `a` and `b`,
    /// is unclosed in the result. Refused on a kernel refusal (see
    /// [`Self::insert_soft_axiom`]) and when more than `DEFAULT_CHART_STEP_BUDGET`
    /// pair-and-relation operations would be needed.
    pub fn compose_soft(
        &self,
        a: &SoftResultSet,
        b: &SoftResultSet,
        gender: Gender,
    ) -> Result<SoftResultSet, String> {
        let mut budget = DEFAULT_CHART_STEP_BUDGET;
        self.compose_soft_within(a, b, gender, &mut budget)
    }

    fn compose_soft_within(
        &self,
        a: &SoftResultSet,
        b: &SoftResultSet,
        gender: Gender,
        budget: &mut usize,
    ) -> Result<SoftResultSet, String> {
        let mut spend = |n: usize| {
            *budget = budget
                .checked_sub(n)
                .ok_or_else(|| BUDGET_EXCEEDED_REASON.to_string())?;
            Ok::<(), String>(())
        };
        let mut masses: BTreeMap<RelId, f64> = BTreeMap::new();
        for (r1, p1) in a.iter() {
            for (r2, p2) in b.iter() {
                spend(1)?;
                let Some(kernel) = self.soft_kernel(&RelationKey::new(r1, r2, gender))? else {
                    continue;
                };
                spend(kernel.len())?;
                let joint = f64::from(p1) * f64::from(p2);
                for (r, pt) in kernel.iter() {
                    *masses.entry(r).or_insert(0.0) += joint * f64::from(pt);
                }
            }
        }
        Ok(SoftResultSet::from_masses(masses))
    }

    /// Left fold of a chain of soft result sets, `(...((e1 ⊗ e2) ⊗ e3)...)`, with
    /// the same gender convention as [`Self::left_fold_chain`]: step `i` joins at
    /// `genders[i + 1]`, and `genders.len() == chain.len() + 1`. The soft product
    /// is not assumed associative, so the bracketing is fixed to the left one.
    ///
    /// Fails closed. `Refused` when the chain is malformed, `tau_h` is not finite
    /// and nonnegative, a step's kernel lookup is refused or over budget, no mass
    /// closes, or at the root:
    /// - the entropy `H(p)` (unclosed bucket included) exceeds `tau_h`;
    /// - the unclosed mass is at least the top relation's mass ("no derivation"
    ///   is the most likely outcome);
    /// - the top two relations tie within [`SOFT_TIE_TOLERANCE`] (a bimodal
    ///   root has no argmax).
    ///
    /// `tau_h` is the caller's; nothing here calibrates it.
    pub fn fold_chain_soft(
        &self,
        chain: &[SoftResultSet],
        genders: &[Gender],
        tau_h: f32,
    ) -> FoldOutcome {
        let refuse = |step_failed: usize, reason: String| FoldOutcome::Refused {
            step_failed,
            reason,
        };
        if chain.is_empty() || genders.len() != chain.len() + 1 {
            return refuse(
                0,
                format!(
                    "invalid chain: {} sets and {} node genders",
                    chain.len(),
                    genders.len()
                ),
            );
        }
        if chain.len() > 64 {
            return refuse(
                0,
                "chain length exceeds maximum supported bound (64)".into(),
            );
        }
        if !(tau_h.is_finite() && tau_h >= 0.0) {
            return refuse(
                0,
                format!("entropy threshold must be finite and >= 0, got {tau_h}"),
            );
        }
        if let Some(i) = chain.iter().position(SoftResultSet::is_empty) {
            return refuse(0, format!("leaf {i} has no relation with mass"));
        }
        let mut budget = DEFAULT_CHART_STEP_BUDGET;
        let mut acc = chain[0].clone();
        let mut step_entropies = vec![acc.entropy()];
        for i in 1..chain.len() {
            acc = match self.compose_soft_within(&acc, &chain[i], genders[i + 1], &mut budget) {
                Ok(next) => next,
                Err(reason) => return refuse(i + 1, reason),
            };
            if acc.is_empty() {
                return refuse(
                    i + 1,
                    format!(
                        "no soft composition closes at step {} (node gender {:?})",
                        i + 1,
                        genders[i + 1]
                    ),
                );
            }
            step_entropies.push(acc.entropy());
        }
        let steps = chain.len() - 1;
        let entropy = acc.entropy();
        let ranked = acc.ranked();
        let top: Vec<_> = ranked.iter().take(3).collect();
        let unclosed = acc.unclosed_mass();
        if entropy > tau_h {
            return refuse(
                steps + 1,
                format!(
                    "entropy {entropy:.6} nats exceeds tau_H {tau_h}: top {top:?}, \
                     unclosed mass {unclosed:.6}"
                ),
            );
        }
        let (predicted, probability) = ranked[0];
        if unclosed >= probability {
            return refuse(
                steps + 1,
                format!(
                    "unclosed mass {unclosed:.6} is at least the top relation {predicted} \
                     ({probability:.6}): no derivation is the likeliest outcome"
                ),
            );
        }
        if let Some(&(second, p2)) = ranked.get(1) {
            if f64::from(probability - p2) <= SOFT_TIE_TOLERANCE {
                return refuse(
                    steps + 1,
                    format!(
                        "multimodal root: relations {predicted} and {second} tie at \
                         {probability:.6}; no argmax"
                    ),
                );
            }
        }
        FoldOutcome::SoftConcluded {
            predicted,
            probability,
            entropy,
            steps,
            distribution: ranked,
            unclosed_mass: unclosed,
            step_entropies,
        }
    }

    pub fn conflict_keys(&self) -> impl Iterator<Item = &RelationKey> {
        self.conflict_keys.iter()
    }

    pub fn conflict_key_count(&self) -> usize {
        self.conflict_keys.len()
    }

    /// `⊗_g` lifted pointwise over `P(R)`: union of `T(r1, r2, g)` over every pair
    /// `(r1, r2) ∈ a × b`. This is the partial operation of definition 8.6.1; it does
    /// not assume associativity (see [`Self::associativity_audit`]).
    pub fn compose_sets(&self, a: &ResultSet, b: &ResultSet, gender: Gender) -> ResultSet {
        if a.is_empty() || b.is_empty() {
            return ResultSet::Empty;
        }
        let mut acc: BTreeSet<RelId> = BTreeSet::new();
        for r1 in a.iter() {
            for r2 in b.iter() {
                acc.extend(self.lookup(&RelationKey::new(r1, r2, gender)).iter());
            }
        }
        ResultSet::from_set(acc)
    }

    /// Left-associative fold of a chain of derivation sets: `(...((e1⊗e2)⊗e3)...⊗ek)`.
    pub fn fold_left(&self, chain: &[ResultSet], gender: Gender) -> ResultSet {
        let mut iter = chain.iter();
        let Some(first) = iter.next() else {
            return ResultSet::Empty;
        };
        let mut acc = first.clone();
        for next in iter {
            acc = self.compose_sets(&acc, next, gender);
        }
        acc
    }

    /// Right-associative fold of a chain of derivation sets: `(e1⊗(...⊗(ek-1⊗ek)...))`.
    pub fn fold_right(&self, chain: &[ResultSet], gender: Gender) -> ResultSet {
        let mut iter = chain.iter().rev();
        let Some(last) = iter.next() else {
            return ResultSet::Empty;
        };
        let mut acc = last.clone();
        for prev in iter {
            acc = self.compose_sets(prev, &acc, gender);
        }
        acc
    }

    /// Associativity self-audit (Proposition 8.6.1). Checks, for every gender and every
    /// pair of relation ids that appear anywhere in the table as `r1`/`r2` keys, whether
    /// `(r1⊗r2)⊗r3 == r1⊗(r2⊗r3)`. Reports a checked/violated count rather than a bool,
    /// so the caller can cite evidence (e.g. "N/0") instead of trusting a single flag.
    pub fn associativity_audit(&self) -> AssociativityReport {
        let ids: BTreeSet<RelId> = self.table.keys().flat_map(|k| [k.r1, k.r2]).collect();
        let mut checked = 0u64;
        let mut violations = Vec::new();
        for gender in Gender::ALL {
            for &r1 in &ids {
                let a = ResultSet::single(r1);
                for &r2 in &ids {
                    let b = ResultSet::single(r2);
                    let ab = self.compose_sets(&a, &b, gender);
                    for &r3 in &ids {
                        let c = ResultSet::single(r3);
                        let bc = self.compose_sets(&b, &c, gender);
                        let left = self.compose_sets(&ab, &c, gender);
                        let right = self.compose_sets(&a, &bc, gender);
                        checked += 1;
                        if left != right {
                            violations.push(AssociativityViolation {
                                r1,
                                r2,
                                r3,
                                gender,
                                left,
                                right,
                            });
                        }
                    }
                }
            }
        }
        AssociativityReport {
            checked,
            violations,
        }
    }

    /// Perform sequential left-to-right relational composition along an edge chain:
    /// `(...((e1⊗e2)⊗e3)...⊗ek)`. A missing composition refuses; a conflict key
    /// (`ResultSet::Multi`) also refuses rather than picking one of its candidates,
    /// since this module never resolves a conflict on the caller's behalf.
    pub fn left_fold_chain(&self, edges: &[RelId], genders: &[Gender]) -> FoldOutcome {
        if edges.is_empty() || genders.len() != edges.len() + 1 {
            return FoldOutcome::Refused {
                step_failed: 0,
                reason: format!(
                    "invalid chain: {} edges and {} node genders",
                    edges.len(),
                    genders.len()
                ),
            };
        }
        let mut current = edges[0];
        let mut path = vec![current];
        for i in 1..edges.len() {
            let key = RelationKey::new(current, edges[i], genders[i + 1]);
            let next = match self.lookup(&key) {
                ResultSet::Empty => {
                    return FoldOutcome::Refused {
                        step_failed: i + 1,
                        reason: format!(
                            "missing composition {:?} + {:?} at node gender {:?}",
                            current,
                            edges[i],
                            genders[i + 1]
                        ),
                    };
                }
                ResultSet::Single(r) => r,
                ResultSet::Multi(candidates) => {
                    return FoldOutcome::Refused {
                        step_failed: i + 1,
                        reason: format!("conflict key {key:?} has candidates {candidates:?}"),
                    };
                }
            };
            current = next;
            path.push(current);
        }
        FoldOutcome::Concluded {
            predicted: current,
            steps: edges.len() - 1,
            proof_path: path,
        }
    }

    /// Shared CYK core: `spans[i][j]` holds every relation reachable from node `i` to
    /// node `j` by *some* bracketing of `leaves[i..j]` under the learned table (a
    /// chart over every bracketing at once, not just one). The table is only read
    /// here; a `Multi` table entry contributes every one of its candidates to the
    /// span it feeds, so a real disagreement between bracketings propagates to the
    /// root and fails closed there instead of one candidate being picked silently.
    /// The chain is answered only when the full span holds exactly one relation.
    ///
    /// `budget` caps the number of table-lookup and candidate-insertion operations the
    /// inner loop performs (one op per `(a, b)` pair considered, one more per candidate
    /// relation it yields); exceeding it refuses immediately with
    /// [`BUDGET_EXCEEDED_REASON`] rather than letting a dense or heavily-conflicted
    /// table run the cubic chart to combinatorial blowup.
    fn chart_fold_core(
        &self,
        leaves: &[ResultSet],
        gender_at: impl Fn(usize) -> Gender,
        budget: usize,
    ) -> FoldOutcome {
        let k = leaves.len();
        if k > 64 {
            return FoldOutcome::Refused {
                step_failed: 0,
                reason: "chain length exceeds maximum supported bound (64)".into(),
            };
        }
        let mut spans: Vec<Vec<Witness>> = vec![vec![Witness::new(); k + 1]; k + 1];
        for (i, leaf) in leaves.iter().enumerate() {
            if leaf.is_empty() {
                return FoldOutcome::Refused {
                    step_failed: 0,
                    reason: format!("leaf {i} has no derivation (empty result set)"),
                };
            }
            for r in leaf.iter() {
                spans[i][i + 1].insert(r, None);
            }
        }
        let mut ops: usize = 0;
        for len in 2..=k {
            for i in 0..=k - len {
                let j = i + len;
                let mut out = Witness::new();
                // `m` indexes both `spans[i][m]` and `spans[m][j]`, not just one
                // collection, so clippy's iterator/enumerate suggestion does not fit.
                #[allow(clippy::needless_range_loop)]
                for m in i + 1..j {
                    for &a in spans[i][m].keys() {
                        for &b in spans[m][j].keys() {
                            ops += 1;
                            if ops > budget {
                                return FoldOutcome::Refused {
                                    step_failed: j,
                                    reason: BUDGET_EXCEEDED_REASON.into(),
                                };
                            }
                            if let Some(set) = self.entry(&RelationKey::new(a, b, gender_at(j))) {
                                for r in set.iter() {
                                    ops += 1;
                                    if ops > budget {
                                        return FoldOutcome::Refused {
                                            step_failed: j,
                                            reason: BUDGET_EXCEEDED_REASON.into(),
                                        };
                                    }
                                    out.entry(r).or_insert(Some((m, a, b)));
                                }
                            }
                        }
                    }
                }
                spans[i][j] = out;
            }
        }
        let root: Vec<RelId> = spans[0][k].keys().copied().collect();
        match root.as_slice() {
            [predicted] => {
                let mut proof_path = Vec::with_capacity(2 * k);
                collect_proof(&spans, 0, k, *predicted, &mut proof_path);
                FoldOutcome::Concluded {
                    predicted: *predicted,
                    steps: k - 1,
                    proof_path,
                }
            }
            [] => FoldOutcome::Refused {
                step_failed: k,
                reason: "no bracketing of the chain closes under the learned table".into(),
            },
            candidates => FoldOutcome::Refused {
                step_failed: k,
                reason: format!("bracketings disagree: {candidates:?}"),
            },
        }
    }

    /// Fold a chain over every bracketing at once (a CYK chart over the learned
    /// table), rather than committing to one bracketing up front like
    /// [`Self::left_fold_chain`]. The table is
    /// only read; a `Multi` table entry contributes every one of its candidates, so
    /// bracketings that disagree fail closed at the root instead of one being chosen.
    pub fn chart_fold_chain(&self, edges: &[RelId], genders: &[Gender]) -> FoldOutcome {
        if edges.is_empty() || genders.len() != edges.len() + 1 {
            return FoldOutcome::Refused {
                step_failed: 0,
                reason: format!(
                    "invalid chain: {} edges and {} node genders",
                    edges.len(),
                    genders.len()
                ),
            };
        }
        if edges.len() > 64 {
            return FoldOutcome::Refused {
                step_failed: 0,
                reason: "chain length exceeds maximum supported bound (64)".into(),
            };
        }
        let leaves: Vec<ResultSet> = edges.iter().map(|&r| ResultSet::single(r)).collect();
        self.chart_fold_core(&leaves, |j| genders[j], DEFAULT_CHART_STEP_BUDGET)
    }

    /// Chart-fold a chain whose leaves are already derivation sets (e.g. produced by
    /// an earlier [`Self::compose_sets`] stage) rather than single relation ids,
    /// joining every span under one shared `gender`. See [`Self::chart_fold_chain`]
    /// for the bracketing and conflict-propagation discipline.
    pub fn chart_fold_sets(&self, chain: &[ResultSet], gender: Gender) -> FoldOutcome {
        if chain.is_empty() {
            return FoldOutcome::Refused {
                step_failed: 0,
                reason: "invalid chain: 0 elements".into(),
            };
        }
        self.chart_fold_core(chain, |_| gender, DEFAULT_CHART_STEP_BUDGET)
    }
}

/// `spans[i][j][rel]`: the split point and the left/right relations that first
/// derived `rel` for that span, or `None` for a leaf. Named so the type is spelled
/// once, in [`RelationSemiring::chart_fold_core`] and [`collect_proof`] alike.
type Witness = BTreeMap<RelId, Option<(usize, RelId, RelId)>>;

/// Post-order walk of one witness derivation out of [`RelationSemiring::chart_fold_core`]:
/// leaves are chain elements, inner entries are the composed relations, the last
/// entry pushed is the answer for `spans[i][j]`.
fn collect_proof(spans: &[Vec<Witness>], i: usize, j: usize, rel: RelId, out: &mut Vec<RelId>) {
    if let Some(&Some((m, a, b))) = spans[i][j].get(&rel) {
        collect_proof(spans, i, m, a, out);
        collect_proof(spans, m, j, b, out);
    }
    out.push(rel);
}

#[cfg(test)]
mod tests {
    use super::*;

    // Relation ids: pure integers, no domain meaning is baked into this module.
    const R_A: RelId = 0;
    const R_B: RelId = 1;
    const R_C: RelId = 2;
    const R_D: RelId = 3;

    // Extra ids used only by the fold tests below (edges and their composites).
    const R_E1: RelId = 10;
    const R_E2: RelId = 11;
    const R_E3: RelId = 12;
    const R_E4: RelId = 13;
    const R_X: RelId = 14;
    const R_Y: RelId = 15;
    const R_Z: RelId = 16;
    const R_P: RelId = 17;
    const R_Q: RelId = 18;
    const R_R: RelId = 19;

    fn associative_table() -> RelationSemiring {
        // Encodes r_i ⊗ r_j = r_{(i+j) mod 4} for every gender: this is exactly Z/4Z
        // under addition, which is associative by construction, so the audit over it
        // must report zero violations.
        let mut sem = RelationSemiring::new();
        let ids = [R_A, R_B, R_C, R_D];
        for gender in Gender::ALL {
            for (i, &ri) in ids.iter().enumerate() {
                for (j, &rj) in ids.iter().enumerate() {
                    let result = ids[(i + j) % ids.len()];
                    sem.insert_axiom(RelationKey::new(ri, rj, gender), ResultSet::single(result));
                }
            }
        }
        sem
    }

    #[test]
    fn chart_chain_length_bound_refuses_before_chart_allocation() {
        let sem = RelationSemiring::new();
        for outcome in [
            sem.chart_fold_chain(&[1; 65], &[Gender::Male; 66]),
            sem.chart_fold_sets(&vec![ResultSet::single(1); 65], Gender::Male),
        ] {
            assert_eq!(
                outcome,
                FoldOutcome::Refused {
                    step_failed: 0,
                    reason: "chain length exceeds maximum supported bound (64)".into(),
                }
            );
        }
    }

    #[test]
    fn result_set_normalizes_to_canonical_form() {
        assert_eq!(ResultSet::from_ids(std::iter::empty()), ResultSet::Empty);
        assert_eq!(ResultSet::from_ids([R_A, R_A]), ResultSet::Single(R_A));
        let multi = ResultSet::from_ids([R_A, R_B]);
        assert!(multi.is_conflict());
        assert_eq!(multi.len(), 2);
    }

    #[test]
    fn union_is_idempotent_with_empty_zero() {
        let s = ResultSet::single(R_A);
        assert_eq!(s.union(&s), s);
        assert_eq!(s.union(&ResultSet::Empty), s);
        assert_eq!(ResultSet::Empty.union(&ResultSet::Empty), ResultSet::Empty);
    }

    #[test]
    fn missing_key_composes_to_empty() {
        let sem = RelationSemiring::new();
        let out = sem.compose_sets(
            &ResultSet::single(R_A),
            &ResultSet::single(R_B),
            Gender::Male,
        );
        assert_eq!(out, ResultSet::Empty);
    }

    #[test]
    fn insert_axiom_records_and_clears_conflict_keys() {
        let mut sem = RelationSemiring::new();
        let key = RelationKey::new(R_A, R_B, Gender::Unknown);
        sem.insert_axiom(key, ResultSet::from_ids([R_C, R_D]));
        assert_eq!(sem.conflict_key_count(), 1);
        assert!(sem.conflict_keys().any(|k| *k == key));

        // Correcting the axiom to a determined result must clear the conflict flag.
        sem.insert_axiom(key, ResultSet::single(R_C));
        assert_eq!(sem.conflict_key_count(), 0);
    }

    #[test]
    fn associativity_audit_reports_zero_violations_on_an_associative_table() {
        let sem = associative_table();
        let report = sem.associativity_audit();
        assert!(report.checked > 0);
        assert!(
            report.is_associative(),
            "unexpected violations: {:?}",
            report.violations
        );
        assert_eq!(report.violation_count(), 0);

        // Cross-check fold_left / fold_right agree on a concrete chain, since
        // associativity should make bracketing irrelevant.
        let chain = [
            ResultSet::single(R_A),
            ResultSet::single(R_B),
            ResultSet::single(R_C),
            ResultSet::single(R_D),
        ];
        assert_eq!(
            sem.fold_left(&chain, Gender::Male),
            sem.fold_right(&chain, Gender::Male)
        );
    }

    #[test]
    fn associativity_audit_detects_a_genuine_violation() {
        // Deliberately non-associative table: (A⊗A)=B, B⊗A=C, but A⊗(A⊗A) uses A⊗B=D.
        // (A⊗A)⊗A = B⊗A = C ;  A⊗(A⊗A) = A⊗B = D  =>  C != D.
        let mut sem = RelationSemiring::new();
        sem.insert_axiom(
            RelationKey::new(R_A, R_A, Gender::Male),
            ResultSet::single(R_B),
        );
        sem.insert_axiom(
            RelationKey::new(R_B, R_A, Gender::Male),
            ResultSet::single(R_C),
        );
        sem.insert_axiom(
            RelationKey::new(R_A, R_B, Gender::Male),
            ResultSet::single(R_D),
        );

        let report = sem.associativity_audit();
        assert!(!report.is_associative());
        assert!(report
            .violations
            .iter()
            .any(|v| v.r1 == R_A && v.r2 == R_A && v.r3 == R_A && v.gender == Gender::Male));

        // fold_left and fold_right must disagree on this chain since bracketing matters.
        let chain = [
            ResultSet::single(R_A),
            ResultSet::single(R_A),
            ResultSet::single(R_A),
        ];
        assert_ne!(
            sem.fold_left(&chain, Gender::Male),
            sem.fold_right(&chain, Gender::Male)
        );
    }

    // -----------------------------------------------------------------
    // Chain folding: left_fold_chain, chart_fold_chain,
    // chart_fold_sets (S1 / S3, ported from gen3-lodgraph's causal_closure::fold).
    // -----------------------------------------------------------------

    #[test]
    fn empty_chain_is_refused_by_every_fold() {
        let sem = RelationSemiring::new();
        assert!(matches!(
            sem.left_fold_chain(&[], &[]),
            FoldOutcome::Refused { step_failed: 0, .. }
        ));
        assert!(matches!(
            sem.chart_fold_chain(&[], &[]),
            FoldOutcome::Refused { step_failed: 0, .. }
        ));
        assert!(matches!(
            sem.chart_fold_sets(&[], Gender::Male),
            FoldOutcome::Refused { step_failed: 0, .. }
        ));
    }

    #[test]
    fn gender_length_mismatch_is_refused() {
        let sem = RelationSemiring::new();
        let edges = [R_A, R_B];
        let genders = [Gender::Male]; // needs 3 (edges.len() + 1), has 1
        assert!(matches!(
            sem.left_fold_chain(&edges, &genders),
            FoldOutcome::Refused { step_failed: 0, .. }
        ));
        assert!(matches!(
            sem.chart_fold_chain(&edges, &genders),
            FoldOutcome::Refused { step_failed: 0, .. }
        ));
    }

    #[test]
    fn chart_closes_a_2plus2_chain_that_left_fold_refuses() {
        // e1⊗e2 = X (gender genders[2]), e3⊗e4 = Y (gender genders[4]), X⊗Y = Z
        // (gender genders[4]); no other axioms, so left-to-right folding cannot
        // close (e1⊗e2 = X, but X⊗e3 has no entry), while the chart, which tries
        // every bracketing, finds the (e1⊗e2)⊗(e3⊗e4) split.
        let mut sem = RelationSemiring::new();
        let edges = [R_E1, R_E2, R_E3, R_E4];
        let genders = [
            Gender::Male,
            Gender::Male,
            Gender::Female,
            Gender::Male,
            Gender::Unknown,
        ];
        sem.insert_axiom(
            RelationKey::new(R_E1, R_E2, genders[2]),
            ResultSet::single(R_X),
        );
        sem.insert_axiom(
            RelationKey::new(R_E3, R_E4, genders[4]),
            ResultSet::single(R_Y),
        );
        sem.insert_axiom(
            RelationKey::new(R_X, R_Y, genders[4]),
            ResultSet::single(R_Z),
        );

        assert!(matches!(
            sem.left_fold_chain(&edges, &genders),
            FoldOutcome::Refused { .. }
        ));

        match sem.chart_fold_chain(&edges, &genders) {
            FoldOutcome::Concluded {
                predicted,
                steps,
                proof_path,
            } => {
                assert_eq!(predicted, R_Z);
                assert_eq!(steps, 3);
                assert_eq!(proof_path, vec![R_E1, R_E2, R_X, R_E3, R_E4, R_Y, R_Z]);
            }
            other => panic!("expected chart to conclude Z, got {other:?}"),
        }
    }

    #[test]
    fn chart_fold_chain_refuses_when_bracketings_disagree() {
        // Non-associative table: (A⊗A)⊗A = B⊗A = C, A⊗(A⊗A) = A⊗B = D, C != D, so
        // the two bracketings of a 3-edge all-A chain disagree and the chart must
        // fail closed rather than pick one.
        let mut sem = RelationSemiring::new();
        sem.insert_axiom(
            RelationKey::new(R_A, R_A, Gender::Male),
            ResultSet::single(R_B),
        );
        sem.insert_axiom(
            RelationKey::new(R_B, R_A, Gender::Male),
            ResultSet::single(R_C),
        );
        sem.insert_axiom(
            RelationKey::new(R_A, R_B, Gender::Male),
            ResultSet::single(R_D),
        );

        let edges = [R_A, R_A, R_A];
        let genders = [Gender::Male, Gender::Male, Gender::Male, Gender::Male];
        match sem.chart_fold_chain(&edges, &genders) {
            FoldOutcome::Refused {
                step_failed,
                reason,
            } => {
                assert_eq!(step_failed, 3);
                assert!(reason.contains(&format!("{R_C:?}")), "reason={reason}");
                assert!(reason.contains(&format!("{R_D:?}")), "reason={reason}");
            }
            other => panic!("expected chart to refuse on disagreement, got {other:?}"),
        }
    }

    #[test]
    fn a_conflict_key_makes_chart_refuse_and_left_fold_never_picks() {
        // The table entry for (e1, e2, gender) is itself a conflict key (Multi):
        // both `left_fold_chain` and `chart_fold_chain` must refuse rather than
        // silently picking one of the two candidates.
        let mut sem = RelationSemiring::new();
        let edges = [R_E1, R_E2];
        let genders = [Gender::Male, Gender::Male, Gender::Male];
        let key = RelationKey::new(R_E1, R_E2, genders[2]);
        sem.insert_axiom(key, ResultSet::from_ids([R_X, R_Y]));

        assert!(matches!(
            sem.left_fold_chain(&edges, &genders),
            FoldOutcome::Refused { .. }
        ));

        match sem.chart_fold_chain(&edges, &genders) {
            FoldOutcome::Refused { reason, .. } => {
                assert!(reason.contains("bracketings disagree"), "reason={reason}");
                assert!(reason.contains(&format!("{R_X:?}")), "reason={reason}");
                assert!(reason.contains(&format!("{R_Y:?}")), "reason={reason}");
            }
            other => panic!("expected chart to refuse on a surviving conflict key, got {other:?}"),
        }
    }

    #[test]
    fn chart_fold_sets_agrees_with_fold_left_on_a_single_valued_chain() {
        let sem = associative_table();
        let chain = [
            ResultSet::single(R_A),
            ResultSet::single(R_B),
            ResultSet::single(R_C),
            ResultSet::single(R_D),
        ];
        let expected = sem.fold_left(&chain, Gender::Male);

        match sem.chart_fold_sets(&chain, Gender::Male) {
            FoldOutcome::Concluded { predicted, .. } => {
                assert_eq!(ResultSet::single(predicted), expected);
            }
            other => panic!("expected chart_fold_sets to conclude, got {other:?}"),
        }
    }

    #[test]
    fn chart_closes_inner_first_bracketing() {
        // The chart discovers e1⊗((e2⊗e3)⊗e4) despite missing endpoint pairs.
        let mut sem = RelationSemiring::new();
        let edges = [R_E1, R_E2, R_E3, R_E4];
        let genders = [
            Gender::Male,
            Gender::Female,
            Gender::Unknown,
            Gender::Female,
            Gender::Male,
        ];
        sem.insert_axiom(
            RelationKey::new(R_E2, R_E3, genders[3]),
            ResultSet::single(R_P),
        );
        sem.insert_axiom(
            RelationKey::new(R_P, R_E4, genders[4]),
            ResultSet::single(R_Q),
        );
        sem.insert_axiom(
            RelationKey::new(R_E1, R_Q, genders[4]),
            ResultSet::single(R_R),
        );

        match sem.chart_fold_chain(&edges, &genders) {
            FoldOutcome::Concluded {
                predicted,
                steps,
                proof_path,
            } => {
                assert_eq!(predicted, R_R);
                assert_eq!(steps, 3);
                assert_eq!(proof_path, vec![R_E1, R_E2, R_E3, R_P, R_E4, R_Q, R_R]);
            }
            other => panic!("expected chart to conclude R, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Budget guard: DEFAULT_CHART_STEP_BUDGET / BUDGET_EXCEEDED_REASON.
    // -----------------------------------------------------------------

    /// Builds a maximally dense/conflicted table over `ids 0..n`: every
    /// `(r1, r2, Male)` composes to the full `{0..n}` conflict set, so every chart
    /// cell beyond a leaf holds all `n` candidates and the cubic chart's op count
    /// blows up as fast as possible for a chain of this length.
    fn dense_bomb_table(n: RelId) -> RelationSemiring {
        let mut sem = RelationSemiring::new();
        let all: Vec<RelId> = (0..n).collect();
        for &r1 in &all {
            for &r2 in &all {
                sem.insert_axiom(
                    RelationKey::new(r1, r2, Gender::Male),
                    ResultSet::from_ids(all.iter().copied()),
                );
            }
        }
        sem
    }

    #[test]
    fn dense_bomb_chart_fold_chain_refuses_on_budget() {
        let sem = dense_bomb_table(64);
        let edges: Vec<RelId> = (0..64).collect();
        let genders = vec![Gender::Male; 65];

        let start = std::time::Instant::now();
        let outcome = sem.chart_fold_chain(&edges, &genders);
        let elapsed = start.elapsed();
        eprintln!("dense_bomb_chart_fold_chain_refuses_on_budget: elapsed={elapsed:?}");

        match outcome {
            FoldOutcome::Refused { reason, .. } => {
                assert_eq!(reason, BUDGET_EXCEEDED_REASON);
            }
            other => panic!("expected budget refusal, got {other:?}"),
        }
        // Loose bound only: debug builds vary a lot machine to machine. The number
        // is reported above, not asserted tightly.
        assert!(elapsed.as_secs_f64() < 2.0, "took too long: {elapsed:?}");
    }

    #[test]
    fn dense_bomb_chart_fold_sets_refuses_on_budget() {
        let sem = dense_bomb_table(64);
        let all_ids: ResultSet = ResultSet::from_ids((0..64).collect::<Vec<RelId>>());
        let chain: Vec<ResultSet> = vec![all_ids; 64];

        let start = std::time::Instant::now();
        let outcome = sem.chart_fold_sets(&chain, Gender::Male);
        let elapsed = start.elapsed();
        eprintln!("dense_bomb_chart_fold_sets_refuses_on_budget: elapsed={elapsed:?}");

        match outcome {
            FoldOutcome::Refused { reason, .. } => {
                assert_eq!(reason, BUDGET_EXCEEDED_REASON);
            }
            other => panic!("expected budget refusal, got {other:?}"),
        }
        assert!(elapsed.as_secs_f64() < 2.0, "took too long: {elapsed:?}");
    }

    #[test]
    fn budget_is_not_over_tight_for_a_long_associative_chain() {
        // Z/4Z chain of 64 edges: C(65, 3) = 43,680 (i, m, j) triples, each with
        // exactly one (a, b) pair (every span is a singleton under an associative
        // table), so 2 ops per triple (~87,360) comfortably clears the default
        // 100,000 budget.
        let sem = associative_table();
        let ids = [R_A, R_B, R_C, R_D];
        let edges: Vec<RelId> = (0..64).map(|i| ids[i % ids.len()]).collect();
        let genders = vec![Gender::Male; edges.len() + 1];

        let chain: Vec<ResultSet> = edges.iter().map(|&r| ResultSet::single(r)).collect();
        let expected = sem.fold_left(&chain, Gender::Male);

        match sem.chart_fold_chain(&edges, &genders) {
            FoldOutcome::Concluded { predicted, .. } => {
                assert_eq!(ResultSet::single(predicted), expected);
            }
            other => panic!("expected the budget to be sufficient, got {other:?}"),
        }
    }

    #[test]
    fn a_tiny_explicit_budget_trips_inside_chart_fold_core() {
        // Same 2+2 fixture as `chart_closes_a_2plus2_chain_that_left_fold_refuses`,
        // called directly against `chart_fold_core` with a budget of 3: the third
        // (i, j) = (2, 4) span at length 2 is the one whose first (a, b) pair pushes
        // the op count from 3 to 4, so it must refuse there with step_failed == 4.
        let mut sem = RelationSemiring::new();
        let edges = [R_E1, R_E2, R_E3, R_E4];
        let genders = [
            Gender::Male,
            Gender::Male,
            Gender::Female,
            Gender::Male,
            Gender::Unknown,
        ];
        sem.insert_axiom(
            RelationKey::new(R_E1, R_E2, genders[2]),
            ResultSet::single(R_X),
        );
        sem.insert_axiom(
            RelationKey::new(R_E3, R_E4, genders[4]),
            ResultSet::single(R_Y),
        );
        sem.insert_axiom(
            RelationKey::new(R_X, R_Y, genders[4]),
            ResultSet::single(R_Z),
        );

        let leaves: Vec<ResultSet> = edges.iter().map(|&r| ResultSet::single(r)).collect();
        match sem.chart_fold_core(&leaves, |j| genders[j], 3) {
            FoldOutcome::Refused {
                step_failed,
                reason,
            } => {
                assert_eq!(step_failed, 4);
                assert_eq!(reason, BUDGET_EXCEEDED_REASON);
            }
            other => panic!("expected a tiny budget to refuse, got {other:?}"),
        }
    }

    // ---- soft semiring ------------------------------------------------------

    fn soft(entries: &[(RelId, f32)]) -> SoftResultSet {
        SoftResultSet::from_probs(entries.iter().copied()).unwrap()
    }

    fn close(a: f32, b: f64) -> bool {
        (f64::from(a) - b).abs() < 1e-5
    }

    /// `T_soft(A, C) = {X: .9, Y: .1}`, `T_soft(B, C) = {Y: 1}` under every gender.
    fn soft_table() -> RelationSemiring {
        let mut sem = RelationSemiring::new();
        for g in Gender::ALL {
            sem.insert_soft_axiom(
                RelationKey::new(R_A, R_C, g),
                soft(&[(R_X, 0.9), (R_Y, 0.1)]),
            );
            sem.insert_soft_axiom(RelationKey::new(R_B, R_C, g), soft(&[(R_Y, 1.0)]));
        }
        sem
    }

    #[test]
    fn soft_result_set_validates_and_measures_entropy() {
        assert!(close(SoftResultSet::delta(R_A).entropy(), 0.0));
        let uniform = soft(&[(R_A, 0.25), (R_B, 0.25), (R_C, 0.25), (R_D, 0.25)]);
        assert!(close(uniform.entropy(), 4f64.ln()));
        // The unclosed bucket counts as an outcome: 3 seen of 4 counts.
        let counted = SoftResultSet::from_counts(&[(R_A, 2), (R_B, 1)], 1.0).unwrap();
        assert!(close(counted.probability(R_A), 0.5));
        assert!(close(counted.unclosed_mass(), 0.25));
        assert!(close(
            counted.entropy(),
            -(0.5f64 * 0.5f64.ln() + 2.0 * 0.25 * 0.25f64.ln())
        ));

        type E = SoftSetError;
        let bad = |e: &[(RelId, f32)]| SoftResultSet::from_probs(e.iter().copied()).unwrap_err();
        assert_eq!(bad(&[]), E::Empty);
        assert_eq!(bad(&[(R_A, 0.5), (R_A, 0.5)]), E::Duplicate(R_A));
        assert!(matches!(
            bad(&[(R_A, 0.0), (R_B, 1.0)]),
            E::Probability { .. }
        ));
        assert!(matches!(bad(&[(R_A, f32::NAN)]), E::Probability { .. }));
        assert!(matches!(bad(&[(R_A, 1.5)]), E::Probability { .. }));
        assert!(matches!(bad(&[(R_A, 0.5), (R_B, 0.4)]), E::Mass { .. }));
        assert!(matches!(
            SoftResultSet::from_counts(&[(R_A, 0)], 0.0),
            Err(E::ZeroCount { .. })
        ));
        assert!(matches!(
            SoftResultSet::from_counts(&[(R_A, 1)], -1.0),
            Err(E::PseudoCount(_))
        ));
    }

    #[test]
    fn compose_soft_is_the_kernel_sum_and_keeps_unclosed_mass() {
        let sem = soft_table();
        let a = soft(&[(R_A, 0.7), (R_B, 0.3)]);
        let c = SoftResultSet::delta(R_C);
        let out = sem.compose_soft(&a, &c, Gender::Male).unwrap();
        // X = .7 * .9; Y = .7 * .1 + .3 * 1.
        assert!(close(out.probability(R_X), 0.63));
        assert!(close(out.probability(R_Y), 0.37));
        assert!(close(out.unclosed_mass(), 0.0));

        // (D, C) has no kernel: D's mass stays unclosed, not renormalized away.
        let a = soft(&[(R_A, 0.4), (R_D, 0.6)]);
        let out = sem.compose_soft(&a, &c, Gender::Male).unwrap();
        assert!(close(out.probability(R_X), 0.36));
        assert!(close(out.probability(R_Y), 0.04));
        assert!(close(out.unclosed_mass(), 0.6));
    }

    #[test]
    fn compose_soft_is_linear_in_each_argument() {
        let sem = soft_table();
        let c = SoftResultSet::delta(R_C);
        let (a1, a2) = (soft(&[(R_A, 1.0)]), soft(&[(R_B, 1.0)]));
        let lambda = 0.35_f32;
        let mix = soft(&[(R_A, lambda), (R_B, 1.0 - lambda)]);
        let (o1, o2) = (
            sem.compose_soft(&a1, &c, Gender::Female).unwrap(),
            sem.compose_soft(&a2, &c, Gender::Female).unwrap(),
        );
        let om = sem.compose_soft(&mix, &c, Gender::Female).unwrap();
        for r in [R_X, R_Y] {
            let expected =
                f64::from(lambda * o1.probability(r) + (1.0 - lambda) * o2.probability(r));
            assert!(close(om.probability(r), expected), "relation {r}");
        }
    }

    #[test]
    fn fold_chain_soft_concludes_under_the_entropy_gate_and_refuses_above_it() {
        let sem = soft_table();
        let chain = [soft(&[(R_A, 0.7), (R_B, 0.3)]), SoftResultSet::delta(R_C)];
        let genders = [Gender::Male; 3];
        let h = -(0.63f64 * 0.63f64.ln() + 0.37 * 0.37f64.ln());
        match sem.fold_chain_soft(&chain, &genders, 0.7) {
            FoldOutcome::SoftConcluded {
                predicted,
                probability,
                entropy,
                steps,
                distribution,
                unclosed_mass,
                step_entropies,
            } => {
                assert_eq!(predicted, R_X);
                assert!(close(probability, 0.63));
                assert!(close(entropy, h));
                assert_eq!(steps, 1);
                assert_eq!(distribution.len(), 2);
                assert_eq!(distribution[0].0, R_X);
                assert!(close(unclosed_mass, 0.0));
                assert_eq!(step_entropies.len(), 2);
            }
            other => panic!("expected SoftConcluded, got {other:?}"),
        }
        match sem.fold_chain_soft(&chain, &genders, 0.5) {
            FoldOutcome::Refused {
                step_failed,
                reason,
            } => {
                assert_eq!(step_failed, 2);
                assert!(reason.contains("exceeds tau_H"), "{reason}");
            }
            other => panic!("expected an entropy refusal, got {other:?}"),
        }
    }

    #[test]
    fn fold_chain_soft_refuses_a_tie_and_a_dominant_unclosed_mass() {
        let mut sem = soft_table();
        sem.insert_soft_axiom(
            RelationKey::new(R_A, R_D, Gender::Male),
            soft(&[(R_X, 0.5), (R_Y, 0.5)]),
        );
        let genders = [Gender::Male; 3];
        // Bimodal root: entropy ln 2 passes a loose gate, the tie does not.
        let tie = [SoftResultSet::delta(R_A), SoftResultSet::delta(R_D)];
        match sem.fold_chain_soft(&tie, &genders, 1.0) {
            FoldOutcome::Refused { reason, .. } => {
                assert!(reason.contains("multimodal"), "{reason}")
            }
            other => panic!("expected a tie refusal, got {other:?}"),
        }
        // 60% of the mass never closes: "no derivation" beats relation X.
        let open = [soft(&[(R_A, 0.4), (R_D, 0.6)]), SoftResultSet::delta(R_C)];
        match sem.fold_chain_soft(&open, &genders, 2.0) {
            FoldOutcome::Refused { reason, .. } => {
                assert!(reason.contains("unclosed mass"), "{reason}")
            }
            other => panic!("expected an unclosed refusal, got {other:?}"),
        }
        // Nothing closes at all.
        let none = [SoftResultSet::delta(R_D), SoftResultSet::delta(R_D)];
        assert!(matches!(
            sem.fold_chain_soft(&none, &genders, 2.0),
            FoldOutcome::Refused { step_failed: 2, .. }
        ));
    }

    #[test]
    fn fold_chain_soft_refuses_bad_input_and_unweighted_conflict_keys() {
        let sem = soft_table();
        let chain = [SoftResultSet::delta(R_A), SoftResultSet::delta(R_C)];
        for (genders, tau) in [
            (&[Gender::Male; 2][..], 1.0),
            (&[Gender::Male; 3][..], -0.1),
            (&[Gender::Male; 3][..], f32::NAN),
        ] {
            assert!(matches!(
                sem.fold_chain_soft(&chain, genders, tau),
                FoldOutcome::Refused { step_failed: 0, .. }
            ));
        }
        assert!(matches!(
            sem.fold_chain_soft(&[], &[Gender::Male], 1.0),
            FoldOutcome::Refused { step_failed: 0, .. }
        ));

        // A discrete conflict key has no weights: the soft fold refuses it.
        let mut conflicted = RelationSemiring::new();
        let key = RelationKey::new(R_A, R_C, Gender::Male);
        conflicted.insert_axiom(key, ResultSet::from_ids([R_X, R_Y]));
        match conflicted.fold_chain_soft(&chain, &[Gender::Male; 3], 2.0) {
            FoldOutcome::Refused { reason, .. } => {
                assert!(reason.contains("no soft axiom"), "{reason}")
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        // Weighing the same candidates makes it foldable.
        conflicted.insert_soft_axiom(key, soft(&[(R_X, 0.95), (R_Y, 0.05)]));
        assert!(matches!(
            conflicted.fold_chain_soft(&chain, &[Gender::Male; 3], 0.5),
            FoldOutcome::SoftConcluded { predicted: R_X, .. }
        ));
        // A soft axiom that disagrees with the discrete one is refused.
        conflicted.insert_soft_axiom(key, soft(&[(R_Z, 1.0)]));
        match conflicted.fold_chain_soft(&chain, &[Gender::Male; 3], 2.0) {
            FoldOutcome::Refused { reason, .. } => {
                assert!(reason.contains("discrete axiom"), "{reason}")
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn fold_chain_soft_on_deltas_matches_the_discrete_left_fold() {
        let sem = associative_table();
        let edges = [R_A, R_B, R_C, R_D, R_B];
        let genders = [Gender::Unknown; 6];
        let FoldOutcome::Concluded { predicted, .. } = sem.left_fold_chain(&edges, &genders) else {
            panic!("the associative table closes every chain");
        };
        let chain: Vec<_> = edges.iter().map(|&r| SoftResultSet::delta(r)).collect();
        match sem.fold_chain_soft(&chain, &genders, 0.0) {
            FoldOutcome::SoftConcluded {
                predicted: soft_predicted,
                probability,
                entropy,
                ..
            } => {
                assert_eq!(soft_predicted, predicted);
                assert!(close(probability, 1.0));
                assert!(close(entropy, 0.0));
            }
            other => panic!("expected SoftConcluded, got {other:?}"),
        }
    }

    #[test]
    fn fold_chain_soft_fails_closed_on_budget() {
        let wide = SoftResultSet::from_probs((0..64).map(|r| (r as RelId, 1.0 / 64.0))).unwrap();
        let mut sem = RelationSemiring::new();
        for r1 in 0..64 {
            for r2 in 0..64 {
                sem.insert_soft_axiom(RelationKey::new(r1, r2, Gender::Male), wide.clone());
            }
        }
        let chain = vec![wide; 40];
        match sem.fold_chain_soft(&chain, &[Gender::Male; 41], 100.0) {
            FoldOutcome::Refused { reason, .. } => assert_eq!(reason, BUDGET_EXCEEDED_REASON),
            other => panic!("expected a budget refusal, got {other:?}"),
        }
    }
}
