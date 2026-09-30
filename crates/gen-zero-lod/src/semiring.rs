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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FoldOutcome {
    Concluded {
        predicted: RelId,
        steps: usize,
        proof_path: Vec<RelId>,
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

/// The learned relation semiring: a compact `BTreeMap`-backed lookup table for `T`, plus
/// the set of keys recorded as conflicted (multi-valued).
#[derive(Clone, Debug, Default)]
pub struct RelationSemiring {
    table: BTreeMap<RelationKey, ResultSet>,
    conflict_keys: BTreeSet<RelationKey>,
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
}
